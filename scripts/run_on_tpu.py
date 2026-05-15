#!/usr/bin/env python3
"""TPU job orchestration CLI.

This script automates the entire workflow for running jobs on TPUs:
1. Finds an available TPU or creates one
2. Sets up the TPU (NFS mount, uv install)
3. Syncs code and runs the user's command
4. Monitors for completion/preemption with Slack notifications
5. Optionally retries on preemption

Example usage:

    # Simple run on v6e-8
    uv run scripts/run_on_tpu.py --tpu-type v6e-8 \
      --command "uv run scripts/train_value_function.py icmmdit_image_libero_rlds"

    # With Slack notifications and retry
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    uv run scripts/run_on_tpu.py --tpu-type v6e-8 \
      --retry-on-preemption \
      --command "uv run scripts/train_value_function.py icmmdit_image_libero_rlds"

    # Use specific TPU
    uv run scripts/run_on_tpu.py --tpu-name v6e-0 --tpu-type v6e-8 \
      --command "uv run scripts/train_value_function.py icmmdit_image_libero_rlds"
"""

import dataclasses
import logging
from pathlib import Path
import subprocess
import sys
import time

import tyro

from openpi.tpu.code_sync import install_deps
from openpi.tpu.code_sync import sync_code
from openpi.tpu.code_sync import sync_wandb_credentials
from openpi.tpu.config import get_tpu_config
from openpi.tpu.config import get_worker_count
from openpi.tpu.gcloud import ssh_command
from openpi.tpu.job import JobRunner
from openpi.tpu.manager import cleanup_preempted
from openpi.tpu.manager import create_tpu
from openpi.tpu.manager import find_available_tpu
from openpi.tpu.manager import wait_for_tpu_ready
from openpi.tpu.setup import setup_tpu
from openpi.tpu.setup import verify_setup
from openpi.tpu.slack import SlackNotifier
from openpi.tpu.slack import _format_duration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TPUJobConfig:
    """Configuration for running a job on TPU."""

    tpu_type: str
    """TPU type (e.g., 'v6e-8', 'v5e-128')."""

    command: str
    """Command to run on the TPU."""

    tpu_name: str | None = None
    """Specific TPU name to use (optional). If not specified, finds or creates one."""

    nfs_user: str = "saksham3"
    """NFS username (e.g. 'saksham3', 'jeffyu'). Determines venv and default working_dir."""

    working_dir: str = ""
    """Working directory on the TPU. Defaults to /nfs/aidm_nfs/<nfs_user>/batch_value_learning."""

    sync_code: bool = True
    """Whether to sync code before running."""

    install_deps: bool = False
    """Whether to run uv sync on the TPU after syncing code."""

    local_code_dir: str = dataclasses.field(default_factory = lambda: str(Path(__file__).resolve().parents[1]))
    """Local code directory to sync from."""

    local_gemma_dir: str = dataclasses.field(
        default_factory = lambda: str(Path.home() / "projects/AIRe/robocoin/helper/gemma")
    )
    """Local Gemma helper checkout to sync from."""

    remote_gemma_dir: str = ""
    """Remote Gemma helper checkout path on NFS. Defaults to /nfs/aidm_nfs/<nfs_user>/helper/gemma."""

    retry_on_preemption: bool = False
    """Whether to retry the job if the TPU is preempted."""

    max_retries: int | None = None
    """Maximum number of retries (None = infinite)."""

    slack_webhook_url: str | None = None
    """Slack webhook URL for notifications."""

    poll_interval: int = 30
    """Seconds between job status checks."""

    wait_timeout: int = 600
    """Seconds to wait for TPU to become ready."""

    verbose: bool = False
    """Show all gcloud commands being executed."""

    local_tmux: bool = True
    """Create a local tmux session with one window per worker for viewing outputs."""


def sync_gemma_helper(config: TPUJobConfig, tpu_name: str, tpu_config) -> None:
    """Sync the local Gemma helper checkout to the TPU NFS path."""
    local_gemma_dir = Path(config.local_gemma_dir)
    if not local_gemma_dir.exists():
        raise FileNotFoundError(f"Local Gemma helper directory does not exist: {local_gemma_dir}")

    remote_parent = str(Path(config.remote_gemma_dir).parent)
    ssh_command(
        tpu_name,
        tpu_config.zone,
        f"mkdir -p {remote_parent}",
        project = tpu_config.project,
        worker = "0",
    )

    ssh_cmd = (
        f"gcloud compute tpus tpu-vm ssh {tpu_name} "
        f"--zone={tpu_config.zone} --project={tpu_config.project} --worker=0 --"
    )
    rsync_args = [
        "rsync",
        "-rltvz",
        "--progress",
        "--omit-dir-times",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "-e",
        ssh_cmd,
        f"{local_gemma_dir}/",
        f":{config.remote_gemma_dir}",
    ]

    logger.info("Syncing Gemma helper from %s to %s:%s", local_gemma_dir, tpu_name, config.remote_gemma_dir)
    subprocess.run(rsync_args, check = True)


def install_gemma_helper(config: TPUJobConfig, tpu_name: str, tpu_config) -> None:
    """Install the synced Gemma helper into the shared uv environment."""
    nfs = tpu_config.nfs_mount_path
    uv_bin = f"{nfs}/saksham3/uv/bin/uv"
    vla_env = f"{nfs}/saksham3/uv/vla"
    ssh_command(
        tpu_name,
        tpu_config.zone,
        (
            f"source {vla_env}/bin/activate && "
            f'export UV_PROJECT_ENVIRONMENT="{vla_env}" && '
            f'SITE_PACKAGES="$({vla_env}/bin/python -c \'import site; print(site.getsitepackages()[0])\')" && '
            f"sudo chmod -R 777 {config.remote_gemma_dir} && "
            'sudo chmod 777 "$SITE_PACKAGES" && '
            f"cd {config.remote_gemma_dir} && "
            f"{uv_bin} pip install --python {vla_env}/bin/python -e . --no-deps && "
            'sudo chmod -R 777 "$SITE_PACKAGES"/gemma.pth "$SITE_PACKAGES"/gemma-*.dist-info'
        ),
        project = tpu_config.project,
        worker = "0",
    )


def find_or_create_tpu(config: TPUJobConfig) -> str:
    """Find an available TPU or create a new one.

    Args:
        config: Job configuration

    Returns:
        Name of the TPU to use
    """
    if config.tpu_name:
        logger.info("Using specified TPU: %s", config.tpu_name)
        return config.tpu_name

    logger.info("Looking for available %s TPU...", config.tpu_type)
    tpu_name = find_available_tpu(config.tpu_type)

    if tpu_name:
        logger.info("Found available TPU: %s", tpu_name)
        return tpu_name

    logger.info("No available TPU found, creating new one...")
    tpu_name = create_tpu(config.tpu_type)
    logger.info("Created TPU: %s", tpu_name)

    tpu_config = get_tpu_config(config.tpu_type)
    logger.info("Waiting for TPU to become ready...")
    if not wait_for_tpu_ready(tpu_name, tpu_config.zone, tpu_config.project, timeout=config.wait_timeout):
        raise RuntimeError(f"TPU {tpu_name} did not become ready within {config.wait_timeout}s")

    return tpu_name


def run_job(config: TPUJobConfig) -> int:
    """Run a job on TPU with optional retry on preemption.

    Args:
        config: Job configuration

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    nfs_base = f"/nfs/aidm_nfs/{config.nfs_user}"
    config = dataclasses.replace(
        config,
        working_dir=config.working_dir or f"{nfs_base}/batch_value_learning",
        remote_gemma_dir=config.remote_gemma_dir or f"{nfs_base}/helper/gemma",
    )

    notifier = SlackNotifier(config.slack_webhook_url)
    tpu_config = get_tpu_config(config.tpu_type)
    retry_count = 0

    while True:
        try:
            tpu_name = find_or_create_tpu(config)
        except Exception as e:
            logger.error("Failed to find or create TPU: %s", e)
            notifier.notify_error("N/A", str(e), config.command)
            return 1

        if not verify_setup(tpu_name, tpu_config, nfs_user=config.nfs_user):
            logger.info("Setting up TPU %s...", tpu_name)
            try:
                setup_tpu(tpu_name, tpu_config)
            except Exception as e:
                logger.error("Failed to setup TPU: %s", e)
                notifier.notify_error(tpu_name, f"Setup failed: {e}", config.command)
                return 1

        if config.sync_code:
            logger.info("Syncing code to TPU %s...", tpu_name)
            try:
                sync_code(
                    config.local_code_dir,
                    tpu_name,
                    tpu_config.zone,
                    config.working_dir,
                    tpu_config.project,
                )
                sync_gemma_helper(config, tpu_name, tpu_config)
                install_gemma_helper(config, tpu_name, tpu_config)
                if config.install_deps:
                    install_deps(tpu_name, tpu_config.zone, config.working_dir, tpu_config.project, tpu_config.nfs_mount_path)
                sync_wandb_credentials(tpu_name, tpu_config.zone, tpu_config.project)
            except subprocess.CalledProcessError as e:
                logger.error("Failed to sync code: %s", e)
                if e.stdout:
                    logger.error("Command stdout:\n%s", e.stdout)
                if e.stderr:
                    logger.error("Command stderr:\n%s", e.stderr)
                notifier.notify_error(tpu_name, f"Code sync failed: {e}", config.command)
                return 1
            except Exception as e:
                logger.error("Failed to sync code: %s", e)
                notifier.notify_error(tpu_name, f"Code sync failed: {e}", config.command)
                return 1

        num_workers = get_worker_count(config.tpu_type)
        runner = JobRunner(
            tpu_name,
            tpu_config.zone,
            tpu_config.project,
            config.working_dir,
            notifier,
            num_workers = num_workers,
            nfs_mount_path = tpu_config.nfs_mount_path,
            nfs_user = config.nfs_user,
        )

        notifier.notify_started(tpu_name, config.tpu_type, config.command)
        start_time = time.time()

        try:
            runner.start_job(config.command)
        except Exception as e:
            logger.error("Failed to start job: %s", e)
            notifier.notify_error(tpu_name, f"Failed to start job: {e}", config.command)
            return 1

        if config.local_tmux and runner.create_local_tmux_session():
            logger.info(
                "Attach to local tmux session with: tmux attach-session -t %s",
                runner.local_session_name,
            )

        try:
            status = runner.monitor_job(poll_interval=config.poll_interval)
            duration = time.time() - start_time

            if status.state == "completed":
                notifier.notify_completion(tpu_name, config.command, duration, success=True)
                logger.info("Job completed successfully in %s", _format_duration(duration))
                return 0

            if status.state == "preempted":
                cleanup_preempted(config.tpu_type)

                can_retry = config.retry_on_preemption and (
                    config.max_retries is None or retry_count < config.max_retries
                )

                if can_retry:
                    retry_count += 1
                    notifier.notify_preemption(tpu_name, config.command, retry_count, config.max_retries)
                    logger.info("TPU preempted, retrying (attempt %s)...", retry_count)
                    config = dataclasses.replace(config, tpu_name=None)
                    continue
                notifier.notify_error(tpu_name, "Preempted, max retries exceeded", config.command)
                logger.error("TPU preempted, max retries exceeded")
                return 1

            # Check for TPU lockfile error and auto-recover
            if status.output_tail and "libtpu_lockfile" in status.output_tail:
                logger.warning("Detected TPU lockfile error, clearing lockfile and retrying...")
                try:
                    ssh_command(
                        tpu_name,
                        tpu_config.zone,
                        "sudo rm -f /tmp/libtpu_lockfile",
                        project=tpu_config.project,
                        worker="all",
                    )
                    logger.info("Cleared TPU lockfile on all workers")
                    # Retry the job
                    retry_count += 1
                    if config.max_retries is None or retry_count <= (config.max_retries or 3):
                        logger.info("Retrying job after lockfile cleanup (attempt %s)...", retry_count)
                        continue
                except Exception as e:
                    logger.error("Failed to clear lockfile: %s", e)

            notifier.notify_completion(
                tpu_name, config.command, duration, success=False, output_tail=status.output_tail
            )
            logger.error("Job failed with exit code %s", status.exit_code)
            if status.output_tail:
                logger.error("Last output:\n%s", status.output_tail)
            return status.exit_code or 1
        finally:
            if config.local_tmux:
                runner.cleanup_local_tmux_session()


if __name__ == "__main__":
    config = tyro.cli(TPUJobConfig)
    if config.verbose:
        logging.getLogger("openpi.tpu.gcloud").setLevel(logging.DEBUG)
    exit_code = run_job(config)
    sys.exit(exit_code)
