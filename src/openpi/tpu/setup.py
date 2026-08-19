"""Preparing a pod to run a job: workers up, filesystem mounted, permissions sane.

Nothing here claims or releases a pod. Occupancy is the run certificate's job
(:mod:`openpi.tpu.certificate`), which is taken the moment a pod is assigned and therefore
already covers the whole setup window — the separate setup marker this module used to keep
covered a strictly smaller one and could disagree with it.
"""

import logging
import time

from openpi.tpu.config import PodConfig
from openpi.tpu.config import RemoteLayout
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)


def wait_for_workers(
    tpu_name: str,
    config: PodConfig,
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


def setup_tpu(tpu_name: str, config: PodConfig, layout: RemoteLayout) -> None:
    """Full TPU setup.

    A pod in a region with a Filestore mounts it and shares one environment. A pod without
    one runs entirely out of each worker's own home directory, so the NFS steps are skipped
    rather than dereferencing a null mount path.
    """
    logger.info("Setting up TPU %s (%s)", tpu_name, "NFS" if config.uses_nfs else "local disk")
    # Everything below assumes every worker is reachable; wait for that once, here.
    wait_for_workers(tpu_name, config)
    kill_unattended_upgrades(tpu_name, config)
    if config.uses_nfs:
        mount_nfs(tpu_name, config)
    else:
        install_host_packages(tpu_name, config)
    fix_tpu_logs_permissions(tpu_name, config.zone, config.project)
    if config.uses_nfs:
        # Only meaningful for a shared tree; per-worker homes are already owned.
        ensure_nfs_user_tree_writable(tpu_name, config, layout)
    logger.info("TPU %s setup complete", tpu_name)


def kill_unattended_upgrades(tpu_name: str, config: PodConfig) -> None:
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


def install_host_packages(tpu_name: str, config: PodConfig) -> None:
    """Install the host packages a local-disk pod needs (no nfs-common)."""
    logger.info("Installing host packages on TPU %s", tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        "sudo apt -y update && sudo apt -y install ffmpeg",
        project=config.project,
        worker="all",
    )


def mount_nfs(tpu_name: str, config: PodConfig) -> None:
    """Mount NFS on all workers of a TPU."""
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
    ssh_command(tpu_name, zone, f"sudo mkdir -p -m 777 {mount_path}", project=project, worker="all")

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

    The TPU logs directory is often created by root or another user, causing permission
    errors when the job starts.
    """
    logger.info("Fixing TPU logs permissions on %s", tpu_name)
    ssh_command(
        tpu_name,
        zone,
        "sudo mkdir -p /tmp/tpu_logs && sudo chmod 777 /tmp/tpu_logs",
        project=project,
        worker="all",
    )


# Top-level directories under the user's NFS root that jobs create into at runtime. Every
# entry is guaranteed to exist and be world-writable on every worker before the job starts,
# so a job's first os.makedirs() beneath one of them cannot hit a foreign-owned parent.
NFS_USER_SUBDIRS = ("robocoin", "lego", "sim_bimanual_assembly", "sim_xarm_packing", "helper", "gemma")


def ensure_nfs_user_tree_writable(tpu_name: str, config: PodConfig, layout: RemoteLayout) -> None:
    """Guarantee the user's NFS tree is creatable-into by every worker of this pod.

    The uid a login name maps to differs per worker (2001, 2004, 2006, 2010 on one v4 pod)
    and per filer, so a directory made by one worker — or on another region's filer by
    another pod — is routinely owned by a uid the current worker cannot write as.
    ``chmod`` on the directory does not help when the *parent* is the foreign-owned one:
    ``os.makedirs`` fails at the first missing component. The only robust move is to create
    the parents with ``sudo`` and open them, on every worker, before anything runs.

    Runs on all workers (idempotent), and never fails setup: a filer with an unusual layout
    should surface as the job's own error, not as an unrunnable pod.
    """
    root = layout.base
    subdirs = " ".join(f"{root}/{name}" for name in NFS_USER_SUBDIRS)
    logger.info("Ensuring NFS user tree %s is world-writable on %s", root, tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        (
            f"sudo mkdir -p {root} {subdirs} 2>/dev/null; "
            # -R on the top-level dirs only, not the whole tree: recursing a filer holding
            # tens of GB of caches on every launch is slow for no gain — the leaves that
            # matter are the ones a job is about to create.
            f"sudo chmod 777 {root} {subdirs} 2>/dev/null; "
            f"sudo chmod -R 777 {root}/robocoin/val_episodes_cache* 2>/dev/null; "
            "true"
        ),
        project=config.project,
        worker="all",
        check=False,
    )


def verify_setup(tpu_name: str, config: PodConfig) -> bool:
    """Whether this pod has already been prepared.

    A local-disk pod has no mount to check, but it is not therefore ready: its readiness is
    that every worker has its own uv. Returning True there would skip the step that
    installs it, and the first command needing uv fails with "command not found".
    """
    if config.uses_nfs:
        probe = f"mountpoint -q {config.nfs_mount_path}"
        worker = "0"
        ready, missing = "NFS mounted", "NFS not mounted"
    else:
        probe = "test -x $HOME/uv/bin/uv"
        worker = "all"
        ready, missing = "uv present on every worker", "uv missing on some worker"

    try:
        # check=False: "not set up yet" is the normal answer on a fresh pod and must read as
        # False, not as an exception that escapes this function.
        result = ssh_command(tpu_name, config.zone, probe, project=config.project, worker=worker, check=False)
    except Exception as e:
        logger.warning("Failed to verify TPU %s setup: %s", tpu_name, e)
        return False
    if result.returncode == 0:
        logger.info("TPU %s setup verified: %s", tpu_name, ready)
        return True
    logger.info("TPU %s setup incomplete: %s", tpu_name, missing)
    return False
