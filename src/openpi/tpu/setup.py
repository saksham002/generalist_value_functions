"""TPU setup operations for NFS mount and shared environment verification."""

import logging

from openpi.tpu.config import TPUConfigWithType
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)


def setup_tpu(tpu_name: str, config: TPUConfigWithType) -> None:
    """Full TPU setup: NFS mount and permissions fixes.

    Args:
        tpu_name: TPU VM name
        config: TPU configuration
    """
    logger.info("Setting up TPU %s", tpu_name)
    mount_nfs(tpu_name, config)
    fix_tpu_logs_permissions(tpu_name, config.zone, config.project)
    fix_val_cache_permissions(tpu_name, config.zone, config.project, config.nfs_mount_path)
    logger.info("TPU %s setup complete", tpu_name)


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

    try:
        result = ssh_command(
            tpu_name,
            zone,
            (
                f"mountpoint -q {mount_path} && "
                f"source {mount_path}/{nfs_user}/uv/vla/bin/activate && "
                f'export PATH="{mount_path}/{nfs_user}/uv/bin:$PATH" && '
                f'export UV_PROJECT_ENVIRONMENT="{mount_path}/{nfs_user}/uv/vla" && '
                "uv --version"
            ),
            project=project,
            worker="0",
            check=False,
        )
        if result.returncode == 0:
            logger.info("TPU %s setup verified: NFS mounted and shared environment available", tpu_name)
            return True
        logger.info("TPU %s setup incomplete", tpu_name)
        return False
    except Exception as e:
        logger.warning("Failed to verify TPU %s setup: %s", tpu_name, e)
        return False
