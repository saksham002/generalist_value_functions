"""Code synchronization utilities for TPU."""

from collections.abc import Sequence
import concurrent.futures
import logging
from pathlib import Path
import subprocess

from openpi.tpu.gcloud import _apply_ssh_user
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)


def sync_code(
    local_dir: str | Path,
    tpu_name: str,
    zone: str,
    remote_dir: str,
    project: str,
    *,
    workers: "Sequence[int] | None" = None,
    dry_run: bool = False,
) -> None:
    """Sync code to a TPU.

    With shared NFS one worker suffices, since every worker sees the same filesystem.
    On a local-disk pod each worker needs its own copy, so pass every worker index;
    they run concurrently because doing eight serially costs ~20 minutes.
    """
    targets = list(workers) if workers is not None else [0]
    if len(targets) == 1:
        sync_code_to_worker(local_dir, tpu_name, zone, remote_dir, project, worker=targets[0], dry_run=dry_run)
        return

    logger.info("Syncing code to %d workers of %s in parallel", len(targets), tpu_name)
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {
            pool.submit(
                sync_code_to_worker,
                local_dir,
                tpu_name,
                zone,
                remote_dir,
                project,
                worker=w,
                dry_run=dry_run,
            ): w
            for w in targets
        }
        for future in concurrent.futures.as_completed(futures):
            error = future.exception()
            if error is not None:
                errors.append(f"worker {futures[future]}: {error}")
    if errors:
        raise RuntimeError(f"Code sync failed on {len(errors)} worker(s): {'; '.join(errors)}")


def sync_code_to_worker(
    local_dir: str | Path,
    tpu_name: str,
    zone: str,
    remote_dir: str,
    project: str,
    *,
    worker: int = 0,
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
        worker=str(worker),
    )

    # Fix permissions only on code directories (not checkpoints/datasets which are huge).
    # This handles files owned by other users and ensures binaries are executable.
    logger.info("Fixing file permissions on code directories...")
    code_dirs = ["src", "scripts", "packages", "examples", ".agent"]
    chmod_parts = [f"sudo chmod -R 777 {remote_dir}/{d} 2>/dev/null || true" for d in code_dirs]
    # Also make the repo root dir + its top-level files writable. rsync writes each
    # file through a temp file in the target directory, so a root dir owned by a
    # different user blocks updates to root-level files (.gitignore, pyproject.toml,
    # uv.lock, ...). -maxdepth 1 keeps this off the huge gitignored data/ and
    # checkpoints/ dirs.
    chmod_parts.append(f"sudo chmod 777 {remote_dir} 2>/dev/null || true")
    chmod_parts.append(f"sudo find {remote_dir} -maxdepth 1 -type f -exec chmod 666 {{}} + 2>/dev/null || true")
    chmod_cmd = " ; ".join(chmod_parts)
    ssh_command(
        tpu_name,
        zone,
        chmod_cmd,
        project=project,
        worker=str(worker),
    )

    # Build rsync command with gcloud as SSH transport
    ssh_cmd = f"gcloud compute tpus tpu-vm ssh {_apply_ssh_user(tpu_name)} --zone={zone} --project={project} --worker={worker} --"

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
        "--exclude=.claude",  # local agent config; not needed on TPU and has permission issues on NFS
        "--exclude=logs",  # local debug logs; TPU-side copies often have stale uid/gid
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


def uv_location(nfs_mount_path: str | None, nfs_user: str = "saksham3") -> tuple[str, str]:
    """Where a pod's uv environment lives, and which workers must build it.

    The single source of truth for this decision: a shared filesystem means one environment
    built once, no filesystem means one per worker in its own home. Everything that installs
    or inspects the environment resolves it here so the two can never disagree.
    """
    if nfs_mount_path:
        root = f"{nfs_mount_path}/{nfs_user}/uv" if nfs_user else f"{nfs_mount_path}/uv"
        return root, "0"
    # No shared filesystem: every worker builds its own environment in its own home,
    # which also sidesteps the per-worker uid mismatch that breaks chmod on NFS.
    return "$HOME/uv", "all"


def uv_environment_ready(
    tpu_name: str,
    zone: str,
    project: str,
    *,
    nfs_mount_path: str | None,
    nfs_user: str = "saksham3",
) -> bool:
    """Whether the uv binary and the project venv are both already in place."""
    uv_root, worker = uv_location(nfs_mount_path, nfs_user)
    result = ssh_command(
        tpu_name,
        zone,
        f"test -x {uv_root}/bin/uv && test -f {uv_root}/vla/bin/activate",
        project=project,
        worker=worker,
        check=False,
        timeout=180,
    )
    return result.returncode == 0


def ensure_uv_environment(
    tpu_name: str,
    zone: str,
    remote_dir: str,
    project: str,
    *,
    nfs_mount_path: str | None,
    nfs_user: str = "saksham3",
    force: bool = False,
) -> None:
    """Provision the uv environment if it is not already there.

    A shared filesystem is not the same as a *provisioned* one: a filer that has never been
    used by this project has no uv and no venv, and assuming otherwise made an NFS pod fail
    at the first command that needed either. Presence is therefore what decides whether to
    build, and the build itself is the same work in both cases — only its location differs.
    """
    if not force and uv_environment_ready(tpu_name, zone, project, nfs_mount_path=nfs_mount_path, nfs_user=nfs_user):
        logger.info("uv environment already present on %s", tpu_name)
        return
    install_uv(tpu_name, zone, project, nfs_mount_path=nfs_mount_path, nfs_user=nfs_user)
    install_deps(tpu_name, zone, remote_dir, project, nfs_mount_path, nfs_user=nfs_user)


def install_deps(
    tpu_name: str,
    zone: str,
    remote_dir: str,
    project: str,
    nfs_mount_path: str | None = "/nfs/aidm_nfs",
    nfs_user: str = "saksham3",
) -> None:
    """Install Python dependencies on TPU using uv.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        remote_dir: Remote directory containing pyproject.toml
        project: GCP project ID
        nfs_mount_path: NFS mount path (e.g. /nfs/aidm_nfs)
        nfs_user: Owner directory on the NFS mount
    """
    logger.info("Installing dependencies on TPU %s", tpu_name)
    uv_root, worker = uv_location(nfs_mount_path, nfs_user)

    ssh_command(
        tpu_name,
        zone,
        (
            f'export PATH="{uv_root}/bin:$PATH" && '
            f'export UV_CACHE_DIR="{uv_root}/cache" && '
            f'export UV_PROJECT_ENVIRONMENT="{uv_root}/vla" && '
            # uv pip install ignores UV_PROJECT_ENVIRONMENT unless VIRTUAL_ENV is set too.
            f'export VIRTUAL_ENV="{uv_root}/vla" && '
            f"cd {remote_dir} && "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra tpu --group rlds && "
            "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."
        ),
        project=project,
        worker=worker,
    )
    logger.info("Dependency installation complete")


def install_uv(
    tpu_name: str,
    zone: str,
    project: str,
    *,
    nfs_mount_path: str | None = None,
    nfs_user: str = "saksham3",
) -> None:
    """Install uv where this pod's environment lives.

    On a local-disk pod that is each worker's own home; on a shared filesystem it is the one
    location every worker reads. The install itself is identical either way.
    """
    uv_root, worker = uv_location(nfs_mount_path, nfs_user)
    logger.info("Installing uv at %s on TPU %s (worker=%s)", uv_root, tpu_name, worker)
    ssh_command(
        tpu_name,
        zone,
        (
            f"mkdir -p {uv_root}/bin {uv_root}/cache && "
            f"test -x {uv_root}/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | "
            f"UV_INSTALL_DIR={uv_root}/bin sh"
        ),
        project=project,
        worker=worker,
    )


def stage_paligemma_weights(tpu_name: str, zone: str, project: str, source_uri: str) -> None:
    """Put pt_224.npz in each worker's own cache.

    PaliGemmaWeightLoader fetches from a Google-owned bucket that denies anonymous reads,
    and neither a symlink nor an NFS OPENPI_DATA_HOME works: download.py resolves the
    symlink then calls relative_to(cache_dir), and get_cache_dir() chmods the cache root
    on every startup, which fails under per-worker uids. A real file each worker owns is
    what works.
    """
    cache = "~/.cache/openpi/vertex-model-garden-paligemma-us/paligemma"
    logger.info("Staging PaliGemma weights on every worker of %s", tpu_name)
    ssh_command(
        tpu_name,
        zone,
        f"mkdir -p {cache} && test -s {cache}/pt_224.npz || gcloud storage cp {source_uri} {cache}/pt_224.npz",
        project=project,
        worker="all",
    )


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
