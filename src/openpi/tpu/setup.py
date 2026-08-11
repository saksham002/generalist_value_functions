"""TPU setup operations for NFS mount and shared environment verification."""

import logging
import time

from openpi.tpu.config import TPUConfigWithType
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)


def wait_for_workers(
    tpu_name: str,
    config: TPUConfigWithType,
    *,
    attempts: int = 5,
    delay: float = 60.0,
    timeout: float = 30.0,
) -> None:
    """Block until every worker answers ssh, or give up.

    A pod reporting READY has not necessarily finished bringing up all of its workers, and
    ``--worker=all`` is all-or-nothing: gcloud exits non-zero if a single worker fails, so
    the first command sent to a fresh pod fails while the rest are still coming up.

    This is the one place that waits. Once every worker has answered once, later commands
    can be issued normally — so the cost is paid at most once per setup, and each attempt
    is short enough that a genuinely dead pod is rejected in minutes rather than hung on.
    """
    for attempt in range(1, attempts + 1):
        try:
            result = ssh_command(
                tpu_name,
                config.zone,
                "true",
                project=config.project,
                worker="all",
                check=False,
                timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("All workers of %s answered ssh (attempt %d)", tpu_name, attempt)
                return
            detail = f"rc={result.returncode}"
        except Exception as e:
            detail = type(e).__name__
        if attempt < attempts:
            logger.info(
                "Not all workers of %s are up yet (%s, attempt %d/%d); waiting %.0fs",
                tpu_name,
                detail,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Workers of {tpu_name} in {config.zone} did not all answer ssh after "
        f"{attempts} attempts over {(attempts - 1) * delay / 60:.0f} minutes"
    )


# Written on every worker while a launcher is setting a pod up, and removed when it is
# done. A pod being set up has no python running yet, so a second launcher's idle check
# reads it as free and claims it; the two then race on apt and ssh and both fail.
SETUP_MARKER_PATH = "$HOME/.openpi_setup_in_progress"


def try_claim_setup(tpu_name: str, config: TPUConfigWithType) -> bool:
    """Atomically claim a pod for setup. Returns whether this caller won it.

    `mkdir` without -p fails when the directory exists, so the check and the claim are a
    single operation. Checking first and touching afterwards leaves a window of seconds —
    long enough for two launchers sweeping at the same moment to both see the pod as free,
    which is exactly how two jobs ended up on one TPU.
    """
    result = ssh_command(
        tpu_name,
        config.zone,
        f"mkdir {SETUP_MARKER_PATH} 2>/dev/null && echo CLAIMED || echo TAKEN",
        project=config.project,
        worker="0",
        check=False,
        timeout=180,
    )
    if result.returncode != 0:
        logger.info("Could not claim %s (ssh rc=%s); treating as taken", tpu_name, result.returncode)
        return False
    won = "CLAIMED" in result.stdout
    logger.info("Claim on %s: %s", tpu_name, "won" if won else "already held by another launcher")
    return won


def mark_setup_started(tpu_name: str, config: TPUConfigWithType) -> None:
    """Ensure the claim exists on every worker, for visibility during setup."""
    logger.info("Claiming %s for setup (marker on all workers)", tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        f"mkdir -p {SETUP_MARKER_PATH}",
        project=config.project,
        worker="all",
        check=False,
    )


def mark_setup_finished(tpu_name: str, config: TPUConfigWithType) -> None:
    """Release a setup claim. Best effort: a failure here must not mask a setup error."""
    try:
        ssh_command(
            tpu_name,
            config.zone,
            f"rm -rf {SETUP_MARKER_PATH}",
            project=config.project,
            worker="all",
            check=False,
        )
    except Exception as e:
        logger.warning("Could not clear the setup marker on %s: %s", tpu_name, e)


def setup_in_progress(tpu_name: str, config: TPUConfigWithType) -> bool:
    """Whether any worker is currently being set up by some launcher.

    Checked across all workers, and a marker on a single one is enough: setup touches
    every worker, so a partial answer still means someone else got there first.
    """
    try:
        result = ssh_command(
            tpu_name,
            config.zone,
            f"ls -d {SETUP_MARKER_PATH} 2>/dev/null",
            project=config.project,
            worker="all",
            check=False,
            timeout=180,
        )
    except Exception as e:
        # Cannot prove the pod is free, so treat it as taken rather than collide.
        logger.warning("Could not read the setup marker on %s (%s); assuming busy", tpu_name, e)
        return True
    return ".openpi_setup_in_progress" in result.stdout


def setup_tpu(tpu_name: str, config: TPUConfigWithType) -> None:
    """Full TPU setup.

    A pod in a region with a Filestore mounts it and shares one environment. A pod
    without one runs entirely out of each worker's own home directory, so the NFS steps
    are skipped rather than dereferencing a null mount path.
    """
    logger.info("Setting up TPU %s (%s)", tpu_name, "NFS" if config.uses_nfs else "local disk")
    # Everything below assumes every worker is reachable; wait for that once, here.
    wait_for_workers(tpu_name, config)
    mark_setup_started(tpu_name, config)
    try:
        kill_unattended_upgrades(tpu_name, config)
        if config.uses_nfs:
            mount_nfs(tpu_name, config)
        else:
            install_host_packages(tpu_name, config)
        fix_tpu_logs_permissions(tpu_name, config.zone, config.project)
        if config.uses_nfs:
            # Only meaningful for a shared cache; per-worker homes are already owned.
            fix_val_cache_permissions(tpu_name, config.zone, config.project, config.nfs_mount_path)
    finally:
        # Cleared even when setup fails, so a failed attempt does not wedge the pod.
        mark_setup_finished(tpu_name, config)
    logger.info("TPU %s setup complete", tpu_name)


def kill_unattended_upgrades(tpu_name: str, config: TPUConfigWithType) -> None:
    """Stop the apt machinery that holds the dpkg lock on a freshly created VM.

    Stopping is not enough on its own: the timers restart the service between the kill and
    the install, which is what makes a fresh pod fail with apt exit 100. Mask them, then
    clear any lock left behind.
    """
    logger.info("Killing unattended-upgrades on TPU %s (if running)", tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        "sudo systemctl stop unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null || true; "
        "sudo systemctl mask unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null || true; "
        "sudo killall -9 unattended-upgrade apt-get apt 2>/dev/null || true; "
        "sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock; "
        "sudo dpkg --configure -a 2>&1 | tail -3",
        project=config.project,
        worker="all",
    )


def install_host_packages(tpu_name: str, config: TPUConfigWithType) -> None:
    """Install the host packages a local-disk pod needs (no nfs-common)."""
    logger.info("Installing host packages on TPU %s", tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        "sudo apt -y update && sudo apt -y install ffmpeg",
        project=config.project,
        worker="all",
    )


def mount_nfs(tpu_name: str, config: TPUConfigWithType) -> None:
    """Mount NFS on all workers of a TPU.

    Args:
        tpu_name: TPU VM name
        config: TPU configuration
    """
    zone = config.zone
    project = config.project
    nfs_server = config.nfs_server
    mount_path = config.nfs_mount_path

    logger.info("Installing nfs-common on TPU %s", tpu_name)
    ssh_command(
        tpu_name,
        zone,
        "sudo apt -y update && sudo apt -y install nfs-common",
        project=project,
        worker="all",
    )

    logger.info("Creating mount directory %s on TPU %s", mount_path, tpu_name)
    ssh_command(
        tpu_name,
        zone,
        f"sudo mkdir -p -m 777 {mount_path}",
        project=project,
        worker="all",
    )

    logger.info("Mounting NFS %s to %s on TPU %s", nfs_server, mount_path, tpu_name)
    ssh_command(
        tpu_name,
        zone,
        f"mountpoint -q {mount_path} || sudo mount -o rw,intr {nfs_server} {mount_path}",
        project=project,
        worker="all",
    )


def fix_tpu_logs_permissions(tpu_name: str, zone: str, project: str) -> None:
    """Fix /tmp/tpu_logs permissions on all workers.

    TPU logs directory is often created by root or another user, causing permission errors.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        project: GCP project ID
    """
    logger.info("Fixing TPU logs permissions on %s", tpu_name)
    ssh_command(
        tpu_name,
        zone,
        "sudo mkdir -p /tmp/tpu_logs && sudo chmod 777 /tmp/tpu_logs",
        project=project,
        worker="all",
    )


def fix_val_cache_permissions(tpu_name: str, zone: str, project: str, nfs_mount_path: str) -> None:
    """Fix permissions on validation episode cache directories.

    These directories may end up root-owned after sudo rm cleanup, preventing
    the training script from writing new cache files.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        project: GCP project ID
        nfs_mount_path: NFS mount path (e.g. /nfs/aidm_nfs)
    """
    logger.info("Fixing val episode cache permissions on %s", tpu_name)
    ssh_command(
        tpu_name,
        zone,
        f"sudo chmod -R 777 {nfs_mount_path}/saksham3/robocoin/val_episodes_cache* 2>/dev/null || true",
        project=project,
        worker="all",
    )


def verify_setup(tpu_name: str, config: TPUConfigWithType, nfs_user: str = "saksham3") -> bool:
    """Verify that TPU setup is complete.

    Checks that NFS is mounted and the shared TPU environment is available.

    Args:
        tpu_name: TPU VM name
        config: TPU configuration
        nfs_user: NFS username whose venv to verify

    Returns:
        True if setup is verified, False otherwise
    """
    zone = config.zone
    project = config.project
    mount_path = config.nfs_mount_path

    # A local-disk pod has no mount to check, but it is not therefore ready: its readiness
    # is that every worker has its own uv. Returning True here would skip the setup step
    # that installs it, and the first command needing uv fails with "command not found".
    if not config.uses_nfs:
        # check=False: "uv is missing" is the normal answer on a fresh pod and must read as
        # "not set up yet", not as an exception that escapes this function.
        result = ssh_command(
            tpu_name,
            zone,
            "test -x $HOME/uv/bin/uv",
            project=project,
            worker="all",
            check=False,
        )
        if result.returncode == 0:
            logger.info("TPU %s local-disk setup verified: uv present on every worker", tpu_name)
            return True
        logger.info("TPU %s local-disk setup incomplete: uv missing on some worker", tpu_name)
        return False

    try:
        result = ssh_command(
            tpu_name,
            zone,
            f"mountpoint -q {mount_path}",
            project=project,
            worker="0",
            check=False,
        )
        if result.returncode == 0:
            logger.info("TPU %s setup verified: NFS mounted", tpu_name)
            return True
        logger.info("TPU %s setup incomplete: NFS not mounted", tpu_name)
        return False
    except Exception as e:
        logger.warning("Failed to verify TPU %s setup: %s", tpu_name, e)
        return False
