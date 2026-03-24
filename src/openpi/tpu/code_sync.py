"""Code synchronization utilities for TPU."""

import logging
from pathlib import Path
import subprocess

from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)


def sync_code(
    local_dir: str | Path,
    tpu_name: str,
    zone: str,
    remote_dir: str,
    project: str,
    *,
    dry_run: bool = False,
) -> None:
    """Sync code to TPU using rsync over gcloud SSH.

    Uses rsync with gcloud as the SSH transport for efficient incremental sync.
    Respects .gitignore for exclusions.

    Args:
        local_dir: Local code directory
        tpu_name: TPU VM name
        zone: GCP zone
        remote_dir: Remote directory to sync to
        project: GCP project ID
        dry_run: If True, show what would be transferred without actually syncing
    """
    local_dir = Path(local_dir)

    # Ensure remote directory exists
    logger.info("Ensuring remote directory exists...")
    ssh_command(
        tpu_name,
        zone,
        f"mkdir -p {remote_dir}",
        project=project,
        worker="0",
    )

    # Fix permissions only on code directories (not checkpoints/datasets which are huge).
    # This handles files owned by other users and ensures binaries are executable.
    logger.info("Fixing file permissions on code directories...")
    code_dirs = ["src", "scripts", "packages", "examples", ".agent"]
    chmod_cmd = " && ".join(f"sudo chmod -R 777 {remote_dir}/{d} 2>/dev/null || true" for d in code_dirs)
    ssh_command(
        tpu_name,
        zone,
        chmod_cmd,
        project=project,
        worker="0",
    )

    # Build rsync command with gcloud as SSH transport
    ssh_cmd = f"gcloud compute tpus tpu-vm ssh {tpu_name} --zone={zone} --project={project} --worker=0 --"

    rsync_args = [
        "rsync",
        "-rltvz",  # recursive, links, times, verbose, compress (no perms/owner/group)
        "--progress",  # show progress for each file
        "--omit-dir-times",  # avoid "failed to set times" errors on NFS
        "--filter=:- .gitignore",  # respect .gitignore files
        "--exclude=.git",  # always exclude .git
        "--exclude=.venv",  # always exclude virtualenv
        "--exclude=__pycache__",  # always exclude pycache
        "--exclude=*.pyc",  # always exclude compiled python
        "--exclude=wandb",  # always exclude wandb logs
        "--exclude=.DS_Store",  # macOS metadata
        "--exclude=._*",  # macOS resource forks
        "--exclude=third_party/aloha",  # large third-party dir
        "--exclude=third_party/libero",  # large third-party dir
        "--exclude=.claude/worktrees",  # worktrees have permission issues on NFS
        "-e",
        ssh_cmd,  # use gcloud SSH as transport
        f"{local_dir}/",  # trailing slash = contents
        f":{remote_dir}",  # remote destination
    ]

    if dry_run:
        rsync_args.insert(1, "-n")

    logger.info("Syncing code from %s to %s:%s", local_dir, tpu_name, remote_dir)
    subprocess.run(rsync_args, check=True)
    logger.info("Code sync complete")


def install_deps(
    tpu_name: str,
    zone: str,
    remote_dir: str,
    project: str,
) -> None:
    """Install Python dependencies on TPU using uv.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        remote_dir: Remote directory containing pyproject.toml
        project: GCP project ID
    """
    logger.info("Installing dependencies on TPU %s", tpu_name)
    ssh_command(
        tpu_name,
        zone,
        (
            "source /nfs/aidm_nfs/saksham3/uv/vla/bin/activate && "
            'export PATH="/nfs/aidm_nfs/saksham3/uv/bin:$PATH" && '
            'export UV_PROJECT_ENVIRONMENT="/nfs/aidm_nfs/saksham3/uv/vla" && '
            f"cd {remote_dir} && uv sync --extra tpu --group rlds"
        ),
        project=project,
        worker="0",
    )
    logger.info("Dependency installation complete")


def sync_wandb_credentials(
    tpu_name: str,
    zone: str,
    project: str,
    local_netrc_path: str = "~/.netrc",
) -> None:
    """Sync wandb credentials from local machine to TPU.

    Copies the ~/.netrc file which contains wandb API key.
    On multi-host TPUs, syncs to ALL workers since any worker might be the
    primary JAX process that initializes wandb.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        project: GCP project ID
        local_netrc_path: Path to local netrc file
    """
    import os
    from pathlib import Path

    netrc_path = Path(os.path.expanduser(local_netrc_path))
    if not netrc_path.exists():
        logger.warning("No ~/.netrc found, skipping wandb credentials sync")
        return

    netrc_content = netrc_path.read_text()

    # Check if wandb credentials exist
    if "api.wandb.ai" not in netrc_content:
        logger.warning("No wandb credentials in ~/.netrc, skipping sync")
        return

    logger.info("Syncing wandb credentials to TPU %s (all workers)", tpu_name)

    # Write netrc content to TPU (escape for shell)
    # Use worker="all" because on multi-host TPUs, any worker might be the
    # primary JAX process (process index 0) that initializes wandb.
    escaped_content = netrc_content.replace("'", "'\\''")
    ssh_command(
        tpu_name,
        zone,
        f"echo '{escaped_content}' > ~/.netrc && chmod 600 ~/.netrc",
        project=project,
        worker="all",
    )

    logger.info("Wandb credentials synced to all workers")
