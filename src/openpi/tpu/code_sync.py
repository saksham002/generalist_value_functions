"""Getting code, dependencies and credentials onto a pod.

Every function takes the pod's :class:`~openpi.tpu.config.RemoteLayout`, so where things
go and which workers they go to is decided once, by the layout, rather than re-derived from
an NFS mount path and a defaulted user name in each signature.
"""

import concurrent.futures
import logging
from pathlib import Path
import subprocess
import time

from openpi.tpu.config import PodConfig
from openpi.tpu.config import RemoteLayout
from openpi.tpu.gcloud import _apply_ssh_user
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)

# Big gitignored trees rsync skips anyway (--filter=:- .gitignore) and which would take
# minutes to walk if the permission fix recursed into them.
_CHMOD_SKIP_DIRS = "checkpoints data wandb .venv __pycache__ logs"


def sync_code(
    local_dir: str | Path,
    tpu_name: str,
    config: PodConfig,
    layout: RemoteLayout,
    *,
    dry_run: bool = False,
) -> None:
    """Sync code to a TPU.

    With shared NFS one worker suffices, since every worker sees the same filesystem. On a
    local-disk pod each worker needs its own copy, so they run concurrently — doing eight
    serially costs ~20 minutes.
    """
    targets = layout.sync_workers if layout.sync_workers is not None else [0]
    if len(targets) == 1:
        sync_code_to_worker(local_dir, tpu_name, config, layout, worker=targets[0], dry_run=dry_run)
        return

    logger.info("Syncing code to %d workers of %s in parallel", len(targets), tpu_name)
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {
            pool.submit(sync_code_to_worker, local_dir, tpu_name, config, layout, worker=w, dry_run=dry_run): w
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
    config: PodConfig,
    layout: RemoteLayout,
    *,
    worker: int = 0,
    dry_run: bool = False,
) -> None:
    """Sync code to one worker using rsync over gcloud SSH, respecting .gitignore."""
    local_dir = Path(local_dir)
    remote_dir = layout.working_dir

    logger.info("Ensuring remote directory exists...")
    ssh_command(tpu_name, config.zone, f"mkdir -p {remote_dir}", project=config.project, worker=str(worker))

    # Make every directory rsync will write into world-writable first. The uid this login
    # maps to differs per worker and per filer, so a tree synced by one pod's worker is
    # routinely unwritable by the next pod's — rsync then fails to stat/mkdir inside it and
    # the launcher aborts. Enumerating code directories was worse: anything not on the list
    # broke on the first uid change.
    logger.info("Fixing file permissions on the sync target...")
    chmod_cmd = " ; ".join(
        [
            f"sudo chmod 777 {remote_dir} 2>/dev/null || true",
            # Top-level files (pyproject.toml, uv.lock, ...): rsync writes via a temp file in
            # the parent, so the parent being 777 is what matters, but open the files too.
            f"sudo find {remote_dir} -maxdepth 1 -type f -exec chmod 666 {{}} + 2>/dev/null || true",
            f"for d in $(ls -A {remote_dir} 2>/dev/null); do "
            f'  case " {_CHMOD_SKIP_DIRS} " in *" $d "*) ;; '
            f"*) [ -d {remote_dir}/$d ] && sudo chmod -R 777 {remote_dir}/$d 2>/dev/null;; esac; "
            "done; true",
        ]
    )
    ssh_command(tpu_name, config.zone, chmod_cmd, project=config.project, worker=str(worker))

    ssh_cmd = (
        f"gcloud compute tpus tpu-vm ssh {_apply_ssh_user(tpu_name)} "
        f"--zone={config.zone} --project={config.project} --worker={worker} --"
    )
    # No -t: preserving mtimes calls utime() on the destination, which requires
    # *ownership*, not write permission. Per-worker uid mappings mean a file synced
    # by an earlier pod is routinely owned by a different uid on this one, so the
    # copy succeeds and then rsync exits 23 ("some files/attrs were not
    # transferred") purely over timestamps. --omit-dir-times covered directories
    # only. Without mtimes, --checksum is what keeps the transfer incremental.
    rsync_args = [
        "rsync",
        "-rlvz",  # recursive, links, verbose, compress (no times/perms/owner/group)
        "--checksum",
        "--progress",
        "--omit-dir-times",
        "--filter=:- .gitignore",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=wandb",
        "--exclude=.DS_Store",
        "--exclude=._*",
        "--exclude=third_party/aloha",
        "--exclude=third_party/libero",
        "--exclude=.claude",  # local agent config; has permission issues on NFS
        "--exclude=logs",  # local debug logs; TPU-side copies often have stale uid/gid
        "-e",
        ssh_cmd,
        f"{local_dir}/",
        f":{remote_dir}",
    ]
    if dry_run:
        rsync_args.insert(1, "-n")

    logger.info("Syncing code from %s to %s:%s", local_dir, tpu_name, remote_dir)
    subprocess.run(rsync_args, check=True)
    logger.info("Code sync complete")


def sync_gemma_helper(local_dir: str | Path, tpu_name: str, config: PodConfig, layout: RemoteLayout) -> None:
    """Sync the local Gemma helper checkout to the pod.

    With NFS one worker suffices; without it every worker needs its own copy.
    """
    local_gemma_dir = Path(local_dir)
    if not local_gemma_dir.exists():
        raise FileNotFoundError(f"Local Gemma helper directory does not exist: {local_gemma_dir}")

    remote_dir = layout.gemma_dir
    targets = layout.sync_workers if layout.sync_workers is not None else [0]
    ssh_command(
        tpu_name,
        config.zone,
        f"mkdir -p {Path(remote_dir).parent!s}",
        project=config.project,
        worker=layout.shared_workers,
    )

    for worker in targets:
        ssh_cmd = (
            f"gcloud compute tpus tpu-vm ssh {_apply_ssh_user(tpu_name)} "
            f"--zone={config.zone} --project={config.project} --worker={worker} --"
        )
        rsync_args = [
            "rsync",
            # See sync_code_to_worker: -t fails on NFS under per-worker uid mappings.
            "-rlvz",
            "--checksum",
            "--omit-dir-times",
            "--exclude=.git",
            "--exclude=.venv",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "-e",
            ssh_cmd,
            f"{local_gemma_dir}/",
            f":{remote_dir}",
        ]
        logger.info("Syncing Gemma helper to %s worker %s:%s", tpu_name, worker, remote_dir)
        # gcloud ssh returns 255 on a transient connection failure, and one blip across
        # eight serial per-worker syncs would otherwise abort the whole launch.
        for attempt in range(1, 4):
            result = subprocess.run(rsync_args, check=False)
            if result.returncode == 0:
                break
            if attempt == 3:
                raise subprocess.CalledProcessError(result.returncode, rsync_args)
            logger.warning("Gemma sync to worker %s failed (rc=%s); retry %s/3", worker, result.returncode, attempt)
            time.sleep(10)


def install_gemma_helper(tpu_name: str, config: PodConfig, layout: RemoteLayout) -> None:
    """Install the synced Gemma helper into the environment the job will use.

    On NFS that is the one shared uv env; on a local-disk pod each worker has its own, so
    the install has to run everywhere.
    """
    ssh_command(
        tpu_name,
        config.zone,
        (
            f"source {layout.venv}/bin/activate && "
            f'export UV_PROJECT_ENVIRONMENT="{layout.venv}" && '
            f"SITE_PACKAGES=\"$({layout.venv}/bin/python -c 'import site; print(site.getsitepackages()[0])')\" && "
            f"sudo chmod -R 777 {layout.gemma_dir} && "
            'sudo chmod 777 "$SITE_PACKAGES" && '
            f"cd {layout.gemma_dir} && "
            f"{layout.uv_root}/bin/uv pip install --python {layout.venv}/bin/python -e . --no-deps && "
            'sudo chmod -R 777 "$SITE_PACKAGES"/gemma.pth "$SITE_PACKAGES"/gemma-*.dist-info'
        ),
        project=config.project,
        worker=layout.shared_workers,
    )


def uv_environment_ready(tpu_name: str, config: PodConfig, layout: RemoteLayout) -> bool:
    """Whether the uv binary and the project venv are both already in place."""
    result = ssh_command(
        tpu_name,
        config.zone,
        f"test -x {layout.uv_root}/bin/uv && test -f {layout.venv}/bin/activate",
        project=config.project,
        worker=layout.shared_workers,
        check=False,
        timeout=180,
    )
    return result.returncode == 0


def ensure_uv_environment(tpu_name: str, config: PodConfig, layout: RemoteLayout, *, force: bool = False) -> None:
    """Provision the uv environment if it is not already there.

    A shared filesystem is not the same as a *provisioned* one: a filer that has never been
    used by this project has no uv and no venv, and assuming otherwise made an NFS pod fail
    at the first command that needed either. Presence is therefore what decides whether to
    build, and the build itself is the same work in both cases — only its location differs.
    """
    if not force and uv_environment_ready(tpu_name, config, layout):
        logger.info("uv environment already present on %s", tpu_name)
        return
    install_uv(tpu_name, config, layout)
    install_deps(tpu_name, config, layout)


def install_uv(tpu_name: str, config: PodConfig, layout: RemoteLayout) -> None:
    """Install uv where this pod's environment lives."""
    logger.info("Installing uv at %s on TPU %s (worker=%s)", layout.uv_root, tpu_name, layout.shared_workers)
    ssh_command(
        tpu_name,
        config.zone,
        (
            f"mkdir -p {layout.uv_root}/bin {layout.uv_root}/cache && "
            f"test -x {layout.uv_root}/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | "
            f"UV_INSTALL_DIR={layout.uv_root}/bin sh"
        ),
        project=config.project,
        worker=layout.shared_workers,
    )


def install_deps(tpu_name: str, config: PodConfig, layout: RemoteLayout) -> None:
    """Install Python dependencies on TPU using uv."""
    logger.info("Installing dependencies on TPU %s", tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        (
            f'export PATH="{layout.uv_root}/bin:$PATH" && '
            f'export UV_CACHE_DIR="{layout.uv_root}/cache" && '
            f'export UV_PROJECT_ENVIRONMENT="{layout.venv}" && '
            # uv pip install ignores UV_PROJECT_ENVIRONMENT unless VIRTUAL_ENV is set too.
            f'export VIRTUAL_ENV="{layout.venv}" && '
            f"cd {layout.working_dir} && "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra tpu --group rlds && "
            "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."
        ),
        project=config.project,
        worker=layout.shared_workers,
    )
    logger.info("Dependency installation complete")


def stage_paligemma_weights(tpu_name: str, config: PodConfig, layout: RemoteLayout, source_uri: str) -> None:
    """Put pt_224.npz in each worker's own cache, from this pod's own continent.

    PaliGemmaWeightLoader fetches from a Google-owned bucket that denies anonymous reads,
    and neither a symlink nor an NFS OPENPI_DATA_HOME works: download.py resolves the
    symlink then calls relative_to(cache_dir), and get_cache_dir() chmods the cache root on
    every startup, which fails under per-worker uids. A real file each worker owns is what
    works.

    Idempotent — it skips when the cache file is already present — so it runs on every
    launch rather than only when the pod looks unprepared. Gating it behind a setup check
    meant a reused pod whose filer was mounted never had the weights staged at all.
    """
    cache = layout.paligemma_cache_dir
    logger.info("Staging PaliGemma weights from %s on every worker of %s", source_uri, tpu_name)
    ssh_command(
        tpu_name,
        config.zone,
        f"mkdir -p {cache} && test -s {cache}/pt_224.npz || gcloud storage cp {source_uri} {cache}/pt_224.npz",
        project=config.project,
        worker="all",
    )


def sync_wandb_credentials(tpu_name: str, config: PodConfig, local_netrc_path: str = "~/.netrc") -> None:
    """Copy the local ~/.netrc, which holds the wandb API key, to every worker.

    All workers rather than one: any of them might be the primary JAX process that
    initializes wandb.
    """
    netrc_path = Path(local_netrc_path).expanduser()
    if not netrc_path.exists():
        logger.warning("No ~/.netrc found, skipping wandb credentials sync")
        return

    netrc_content = netrc_path.read_text()
    if "api.wandb.ai" not in netrc_content:
        logger.warning("No wandb credentials in ~/.netrc, skipping sync")
        return

    logger.info("Syncing wandb credentials to TPU %s (all workers)", tpu_name)
    escaped_content = netrc_content.replace("'", "'\\''")
    ssh_command(
        tpu_name,
        config.zone,
        f"echo '{escaped_content}' > ~/.netrc && chmod 600 ~/.netrc",
        project=config.project,
        worker="all",
    )
    logger.info("Wandb credentials synced to all workers")
