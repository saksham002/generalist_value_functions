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
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import tyro

from openpi.tpu import buckets
from openpi.tpu.buckets import localize_command
from openpi.tpu.code_sync import ensure_uv_environment
from openpi.tpu.code_sync import stage_paligemma_weights
from openpi.tpu.code_sync import sync_code
from openpi.tpu.code_sync import sync_wandb_credentials
from openpi.tpu.config import DEFAULT_PROJECT
from openpi.tpu.config import DEFAULT_TPU_USER
from openpi.tpu.config import get_tpu_user
from openpi.tpu.config import get_worker_count
from openpi.tpu.config import resolve_from_pod
from openpi.tpu.gcloud import ssh_command
from openpi.tpu.job import JobRunner
from openpi.tpu.job import JobStatus
from openpi.tpu.manager import TPUAllocation
from openpi.tpu.manager import allocate_spot_tpu
from openpi.tpu.manager import cleanup_preempted
from openpi.tpu.manager import find_available_tpu
from openpi.tpu.manager import is_tpu_preempted
from openpi.tpu.manager import running_process_owners
from openpi.tpu.setup import mark_setup_finished
from openpi.tpu.setup import mark_setup_started
from openpi.tpu.setup import setup_tpu
from openpi.tpu.setup import verify_setup
from openpi.tpu.slack import SlackNotifier
from openpi.tpu.slack import _format_duration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Google's own bucket denies anonymous reads, so pods pull from this project's copy.
PALIGEMMA_WEIGHTS_URI = "gs://saksham-euw4/base_checkpoints/paligemma/pt_224.npz"


@dataclasses.dataclass
class TPUJobConfig:
    """Configuration for running a job on TPU."""

    tpu_type: str
    """TPU type (e.g., 'v6e-8', 'v5e-128')."""

    command: str
    """Command to run on the TPU."""

    tpu_name: str | None = None
    """Specific TPU name to use (optional). If not specified, finds or creates one."""

    user: str = DEFAULT_TPU_USER
    """Key in TPU_USERS; determines resource names and the regional write bucket."""

    project: str = DEFAULT_PROJECT
    """GCP project searched for pods, quota and Filestore instances."""

    spot: bool = False
    """Race every zone with live spot quota instead of reusing a reserved pod."""

    done_marker: str | None = None
    """Empty GCS object the remote command creates on success. Its presence is how a fresh
    launcher learns the work already finished; it is deleted once detected, so the handshake
    is one-shot and cannot go stale."""

    carry_checkpoints_from: str | None = None
    """Checkpoint root to seed this run from on its FIRST launch, e.g. a previous run's
    region bucket. The newest committed checkpoint under it is copied into wherever this
    launch actually writes. Across preemption retries the carry happens automatically."""

    post_launch_hook: str = ""
    """Command run locally right after the job starts on a pod, and again after every
    preemption retry, since each retry is a fresh pod that needs the same treatment. It is
    launched in the background so job monitoring is never blocked, and receives the pod's
    identity through the environment: TPU_NAME, TPU_ZONE, TPU_PROJECT, TPU_WORKER_COUNT and
    TPU_COMMAND (the localized command, so the hook can read the flags the run actually
    uses). Anything the hook needs to know about the run is in those five variables."""

    nfs_user: str = "saksham3"
    """NFS username (e.g. 'saksham3', 'jeffyu'). Determines venv and default working_dir."""

    working_dir: str = ""
    """Working directory on the TPU. Defaults to /nfs/aidm_nfs/<nfs_user>/batch_value_learning."""

    sync_code: bool = True
    """Whether to sync code before running."""

    install_deps: bool = False
    """Whether to run uv sync on the TPU after syncing code."""

    local_code_dir: str = dataclasses.field(default_factory=lambda: str(Path(__file__).resolve().parents[1]))
    """Local code directory to sync from."""

    local_gemma_dir: str = dataclasses.field(
        default_factory=lambda: str(Path.home() / "projects/AIRe/robocoin/helper/gemma")
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

    race_timeout: int = 21600
    """Seconds a spot race may ride the queue. Stockouts last hours, so this is far larger
    than wait_timeout: timing out deletes every candidate, losing queue position."""

    rss_guard_interval: int = 300
    """Seconds between host-RSS checks while a job runs. 0 disables the guard."""

    verbose: bool = False
    """Show all gcloud commands being executed."""

    local_tmux: bool = True
    """Create a local tmux session with one window per worker for viewing outputs."""

    endpoint_file: str | None = None
    """Path to write the allocated pod's name and zone to, as JSON, once it is resolved.
    A spot race picks both at runtime, so a client that has to reach the job (an eval
    hitting a policy server, say) has no other way to learn where it landed. Rewritten on
    every preemption retry, so the reader always sees the pod currently serving."""


def sync_gemma_helper(config: TPUJobConfig, tpu_name: str, tpu_config, workers=None) -> None:
    """Sync the local Gemma helper checkout to the pod.

    With NFS one worker suffices; without it every worker needs its own copy.
    """
    local_gemma_dir = Path(config.local_gemma_dir)
    if not local_gemma_dir.exists():
        raise FileNotFoundError(f"Local Gemma helper directory does not exist: {local_gemma_dir}")

    remote_parent = str(Path(config.remote_gemma_dir).parent)
    targets = list(workers) if workers else [0]
    ssh_command(
        tpu_name,
        tpu_config.zone,
        f"mkdir -p {remote_parent}",
        project=tpu_config.project,
        worker="all" if workers else "0",
    )

    for worker in targets:
        ssh_cmd = (
            f"gcloud compute tpus tpu-vm ssh {tpu_name} "
            f"--zone={tpu_config.zone} --project={tpu_config.project} --worker={worker} --"
        )
        rsync_args = [
            "rsync",
            "-rltvz",
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
        logger.info("Syncing Gemma helper to %s worker %s:%s", tpu_name, worker, config.remote_gemma_dir)
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


def install_gemma_helper(config: TPUJobConfig, tpu_name: str, tpu_config, nfs_user: str = "") -> None:
    """Install the synced Gemma helper into the environment the job will use.

    On NFS that is the one shared uv env; on a local-disk pod each worker has its own,
    so the install has to run everywhere.
    """
    if tpu_config.uses_nfs:
        uv_root = f"{tpu_config.nfs_mount_path}/{nfs_user or tpu_config.nfs_directory}/uv"
        worker = "0"
    else:
        uv_root = "$HOME/uv"
        worker = "all"
    uv_bin = f"{uv_root}/bin/uv"
    vla_env = f"{uv_root}/vla"
    ssh_command(
        tpu_name,
        tpu_config.zone,
        (
            f"source {vla_env}/bin/activate && "
            f'export UV_PROJECT_ENVIRONMENT="{vla_env}" && '
            f"SITE_PACKAGES=\"$({vla_env}/bin/python -c 'import site; print(site.getsitepackages()[0])')\" && "
            f"sudo chmod -R 777 {config.remote_gemma_dir} && "
            'sudo chmod 777 "$SITE_PACKAGES" && '
            f"cd {config.remote_gemma_dir} && "
            f"{uv_bin} pip install --python {vla_env}/bin/python -e . --no-deps && "
            'sudo chmod -R 777 "$SITE_PACKAGES"/gemma.pth "$SITE_PACKAGES"/gemma-*.dist-info'
        ),
        project=tpu_config.project,
        worker=worker,
    )


def find_or_create_tpu(config: TPUJobConfig) -> TPUAllocation:
    """Resolve the pod to run on, and where it lives.

    A named pod must already exist and describes itself. An unnamed spot launch races
    every zone with live quota; an unnamed reserved launch reuses an idle pod of the
    right shape wherever it is.
    """
    if config.tpu_name:
        logger.info("Using specified TPU: %s", config.tpu_name)
        resolved = resolve_from_pod(config.tpu_name, user=config.user, project=config.project, tpu_type=config.tpu_type)
        # Naming a pod says which one to use, not that it is free. Pods are shared, and a
        # pod that was idle when the launch was prepared may have been claimed since; a
        # second job on the same TPU makes one of the two fail on the accelerator lock.
        owners = running_process_owners(config.tpu_name, resolved)
        if owners is None:
            raise RuntimeError(
                f"Cannot tell what is running on {config.tpu_name}; refusing to start a job on it. "
                "Check the pod by hand, or omit --tpu-name to pick an idle pod automatically."
            )
        if owners:
            raise RuntimeError(
                f"{config.tpu_name} is already busy with processes owned by {sorted(owners)}. "
                "Wait for it to free up, or omit --tpu-name to pick an idle pod automatically."
            )
        return TPUAllocation(name=config.tpu_name, config=resolved)

    if config.spot:
        logger.info("Allocating spot %s across every zone with quota...", config.tpu_type)
        return allocate_spot_tpu(config.tpu_type, user=config.user, project=config.project, timeout=config.race_timeout)

    logger.info("Looking for an available %s TPU...", config.tpu_type)
    # Scope to this user's own pods: matching on family prefix alone would adopt someone
    # else's reserved pod of the same shape.
    allocation = find_available_tpu(
        config.tpu_type,
        project=config.project,
        user=config.user,
        resource_owner=get_tpu_user(config.user).resource_owner,
    )
    if allocation is None:
        raise RuntimeError(
            f"No idle {config.tpu_type} TPU exists. Reserved launches target an existing pod; "
            "pass --tpu-name, create one, or use --spot to race for capacity."
        )
    logger.info("Found available TPU %s in %s", allocation.name, allocation.config.zone)
    return allocation


def checkpoint_root_of(command: str) -> str | None:
    """The per-run checkpoint directory a training command writes to.

    ``--checkpoint-base-dir`` is shared across runs; the run's own tree is
    ``<base>/<config>/<exp-name or config>``, matching TrainConfig.checkpoint_dir.
    """
    if not buckets.is_train_command(command):
        return None
    base = re.search(r"--checkpoint-base-dir[=\s]+(\S+)", command)
    if base is None:
        return None
    tokens = command.split()
    entry = next((i for i, t in enumerate(tokens) if t.endswith(".py")), None)
    if entry is None or entry + 1 >= len(tokens):
        return None
    config_name = tokens[entry + 1]
    if config_name.startswith("-"):
        return None
    exp = re.search(r"--exp-name[=\s]+(\S+)", command)
    exp_name = exp.group(1) if exp else config_name
    return f"{base.group(1).rstrip('/')}/{config_name}/{exp_name}"


# Matches the fine-tune selector but not its override flags: `--fine-tune <name>` and
# `--fine-tune=<name>` are separated by whitespace or '=', whereas
# `--fine-tune.data-factory.rlds-data-dir` continues with a '.'.
_FINE_TUNE_IN_COMMAND = re.compile(r"--fine-tune[=\s]+([\w.-]+)")


def fine_tune_name_of(command: str) -> str | None:
    """The fine-tune config a training command selects, or None for a plain run."""
    match = _FINE_TUNE_IN_COMMAND.search(command)
    return match.group(1) if match else None


def run_post_launch_hook(config: TPUJobConfig, tpu_name: str, tpu_config) -> None:
    """Start the post-launch hook for the pod the job was just started on.

    Runs per attempt rather than per launch: a preemption retry puts the job on a fresh
    pod whose local state starts empty, so whatever the hook sets up has to be redone
    there. Failure is logged and never propagated — a hook is an accompaniment to the run,
    not a precondition for it.
    """
    if not config.post_launch_hook:
        return
    env = {
        **os.environ,
        "TPU_NAME": tpu_name,
        "TPU_ZONE": tpu_config.zone,
        "TPU_PROJECT": tpu_config.project,
        "TPU_WORKER_COUNT": str(get_worker_count(config.tpu_type)),
        "TPU_COMMAND": config.command,
    }
    logger.info("Starting post-launch hook on %s: %s", tpu_name, config.post_launch_hook)
    try:
        # shell=True is deliberate: the hook is operator-supplied, exactly like the
        # training command this script already runs.
        subprocess.Popen(
            config.post_launch_hook,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        logger.warning("Could not start post-launch hook: %s", e)


def adjust_mesh_flags(command: str, tpu_config) -> str:
    """Make the FSDP axis expressible on the pod that was actually allocated.

    A pod's physical mesh is [4, hosts, 1] — four chips per host. An FSDP axis is only
    assignable if it is a product of a subset of those axis sizes, so a config written for
    a 16-host v5e-64 asks for 16 and fails on a v6e-32, whose mesh is [4, 8, 1]:

        NotImplementedError: Failed to find assignment for logical_axis_index 1 of
        size 16 with remaining assignable mesh [4, 8, 1]

    The host count is always an axis, so it is the safe choice. Only applied when the
    command does not already pin the value.
    """
    # Match any namespaced spelling (`--fsdp-devices`, `--critic.fsdp-devices`, ...): a
    # command that already pins the axis under a subcommand prefix must not have a second,
    # top-level flag appended, which its CLI would reject outright.
    if tpu_config.family != "v6e" or "fsdp-devices" in command:
        return command
    hosts = get_worker_count(tpu_config.tpu_type)
    logger.info("Setting --fsdp-devices=%d for %s (mesh [4, %d, 1])", hosts, tpu_config.tpu_type, hosts)
    return f"{command} --fsdp-devices={hosts}"


def worker_rss_over_limit(allocation: TPUAllocation, *, job_pattern: str = "scripts/train") -> str | None:
    """Return a description of the worst offending worker, or None if all are under.

    The ceiling is the family's host RAM less headroom: a worker that crosses it is about
    to be OOM-killed and will drop the JAX coordinator, so the run is better killed and
    reported than silently relaunched into the same memory profile.
    """
    limit_mb = int(allocation.config.host_ram_gb * 1024 * 0.95)
    awk = (
        "max=$(ps -eo rss=,comm=,args= | awk -v pat='" + job_pattern + "' "
        "'$2 ~ /^python/ && index($0, pat) > 0 { if ($1 > m) m = $1 } END { print m + 0 }'); "
        'echo "RSSMB $(hostname) $((max / 1024))"'
    )
    try:
        result = ssh_command(
            allocation.name,
            allocation.config.zone,
            awk,
            project=allocation.config.project,
            worker="all",
            check=False,
        )
    except Exception as e:
        logger.warning("RSS probe failed on %s: %s", allocation.name, e)
        return None

    worst: tuple[str, int] | None = None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "RSSMB" and parts[2].isdigit():
            megabytes = int(parts[2])
            if worst is None or megabytes > worst[1]:
                worst = (parts[1], megabytes)
    if worst is None:
        return None
    logger.info("worker RSS max: %s %d MB (limit %d MB)", worst[0], worst[1], limit_mb)
    return f"{worst[0]}={worst[1]}MB" if worst[1] > limit_mb else None


def _marker_exists(marker: str) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "ls", marker], capture_output=True, text=True, timeout=300, check=False
    )
    return result.returncode == 0


def consume_done_marker(marker: str | None) -> bool:
    """Return whether the work already completed, clearing the marker if so.

    Generic by construction: run_on_tpu knows nothing about checkpoints, only that the
    command it was given reported success by creating this object.
    """
    if not marker or not _marker_exists(marker):
        return False
    logger.info("Completion marker %s present: work already finished", marker)
    subprocess.run(["gcloud", "storage", "rm", marker], capture_output=True, text=True, timeout=300, check=False)
    return True


def command_with_done_marker(command: str, marker: str | None) -> str:
    """Append marker creation so only a successful command records completion.

    Written by one worker only. The command runs on every worker, and GCS rate-limits
    mutation of a *single* object, so N workers creating the same marker at the same
    instant means one succeeds and the rest get HTTP 429 — which fails their half of the
    chain and reports a finished run as a failure. The worker index comes from the
    hostname's ``-w-<index>`` suffix; TPU_WORKER_ID is not set in the ssh environment.
    A hostname without that suffix is a single host, which therefore writes it too.

    The marker tail is one brace group: ``&&`` binds only to the first command of a
    ``;``-separated list, so an ungrouped tail runs even when the command fails — a crash
    then reports itself finished. ``-n "$h"`` guards the single-host branch for the same
    reason: if the failed chain skipped the hostname assignment, both ``$w`` and ``$h``
    are empty and ``[ "$w" = "$h" ]`` alone would match.
    """
    if not marker:
        return command
    return (
        f"{command} && "
        f"{{ h=$(hostname); w=${{h##*-w-}}; "
        f'if [ -n "$h" ] && {{ [ "$w" = 0 ] || [ "$w" = "$h" ]; }}; then '
        f"gcloud storage cp /dev/null {marker}; fi; }}"
    )


def preemption_retry_available(config: TPUJobConfig, retry_count: int) -> bool:
    """Whether losing the pod right now may be answered by re-acquiring one.

    Only a spot launch can re-acquire: a reserved launch would go looking for an idle pod
    that the preemption just removed.
    """
    if not (config.retry_on_preemption and config.spot):
        return False
    return config.max_retries is None or retry_count < config.max_retries


def run_job(config: TPUJobConfig) -> int:
    """Run a job on TPU with optional retry on preemption.

    Args:
        config: Job configuration

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    notifier = SlackNotifier(config.slack_webhook_url)
    retry_count = 0
    # Where the previous attempt wrote, so a retry that lands elsewhere can carry it over.
    previous_checkpoint_root: str | None = None

    while True:
        # Lets a preemption-retry loop terminate instead of relaunching forever.
        if consume_done_marker(config.done_marker):
            return 0

        try:
            allocation = find_or_create_tpu(config)
        except Exception as e:
            logger.error("Failed to find or create TPU: %s", e)
            notifier.notify_error("N/A", str(e), config.command)
            return 1

        # Zone, NFS and the RSS ceiling all come from the pod that was actually allocated.
        tpu_name = allocation.name
        tpu_config = allocation.config

        if config.endpoint_file:
            # Written before setup rather than after the job starts: a client that blocks
            # on this file should be free to begin probing while the pod is still being
            # prepared, and setup on a fresh spot pod takes minutes.
            endpoint = {"name": tpu_name, "zone": tpu_config.zone, "project": tpu_config.project}
            Path(config.endpoint_file).parent.mkdir(parents=True, exist_ok=True)
            Path(config.endpoint_file).write_text(json.dumps(endpoint))
            logger.info("Wrote endpoint %s to %s", endpoint, config.endpoint_file)

        # Claim the pod the moment it is ours. Marking inside setup_tpu is too late: the
        # localize, carry and verify steps before it take minutes, during which the pod has
        # no processes and a second launcher's idle check reads it as free.
        mark_setup_started(tpu_name, tpu_config)

        # Remote paths depend on whether this pod has a shared filesystem, which is only
        # known once it is allocated: a local-disk pod keeps everything in each worker's
        # own home directory.
        if tpu_config.uses_nfs:
            remote_base = f"{tpu_config.nfs_mount_path}/{config.nfs_user}"
            sync_workers = None
        else:
            remote_base = "~"
            sync_workers = list(range(get_worker_count(config.tpu_type)))
        run_config = dataclasses.replace(
            config,
            working_dir=config.working_dir or f"{remote_base}/batch_value_learning",
            remote_gemma_dir=config.remote_gemma_dir or f"{remote_base}/helper/gemma",
        )
        config = run_config

        # A raced pod can land in any US/EU zone, so point the command's GCS paths at that
        # region before running it. Reserved launches never move and are left alone.
        if config.spot:
            try:
                localized, rewrites = localize_command(config.command, tpu_config)
            except Exception as e:
                logger.error("Failed to localize GCS paths for %s: %s", tpu_config.zone, e)
                notifier.notify_error(tpu_name, str(e), config.command)
                mark_setup_finished(tpu_name, tpu_config)
                return 1
            if rewrites:
                config = dataclasses.replace(config, command=localized)

        config = dataclasses.replace(config, command=adjust_mesh_flags(config.command, tpu_config))

        # A re-raced pod can land in a different region, where localization points writes
        # at that region's bucket — an empty one. Carry the newest committed checkpoint
        # over, or the run silently restarts from step 0 while its progress sits elsewhere.
        current_root = checkpoint_root_of(config.command)
        source_root = previous_checkpoint_root or config.carry_checkpoints_from
        if current_root and source_root:
            try:
                buckets.carry_checkpoints(source_root, current_root)
            except Exception as e:
                logger.error("Failed to carry checkpoints from %s: %s", source_root, e)
                notifier.notify_error(tpu_name, f"checkpoint carry failed: {e}", config.command)
                mark_setup_finished(tpu_name, tpu_config)
                return 1

            # A fine-tune writes under <root>/<ft-name>, which the carry above never
            # reaches: it only moves the newest step directly under the root it is given.
            # Carrying just the base means a re-raced pod restores the pretrained weights
            # and silently discards every fine-tune step taken so far — the run looks
            # healthy while repeating hours of work.
            fine_tune = fine_tune_name_of(config.command)
            if fine_tune:
                try:
                    buckets.carry_checkpoints(f"{source_root}/{fine_tune}", f"{current_root}/{fine_tune}")
                except NotImplementedError as e:
                    # Cross-continent: refusing the copy is correct, but it is not worth
                    # failing a launch that can still resume from the base checkpoint.
                    logger.warning("Could not carry fine-tune progress for %s: %s", fine_tune, e)
                except Exception as e:
                    logger.error("Failed to carry fine-tune checkpoints for %s: %s", fine_tune, e)
                    notifier.notify_error(tpu_name, f"fine-tune carry failed: {e}", config.command)
                    mark_setup_finished(tpu_name, tpu_config)
                    return 1
        previous_checkpoint_root = current_root or previous_checkpoint_root

        if not verify_setup(tpu_name, tpu_config, nfs_user=config.nfs_user):
            logger.info("Setting up TPU %s...", tpu_name)
            try:
                setup_tpu(tpu_name, tpu_config)
                if not tpu_config.uses_nfs:
                    # Without a shared filesystem each worker needs its own copy of the
                    # PaliGemma weights. uv is handled with the rest of the environment,
                    # after the code is synced, by the same path both pod types take.
                    stage_paligemma_weights(tpu_name, tpu_config.zone, tpu_config.project, PALIGEMMA_WEIGHTS_URI)
            except Exception as e:
                # A spot pod can vanish mid-setup, and every remaining ssh then hangs to its
                # timeout and surfaces as a setup error. Retrying is the whole point of
                # --retry-on-preemption, so ask the pod before giving up on the run.
                if preemption_retry_available(config, retry_count) and is_tpu_preempted(
                    tpu_name, tpu_config.zone, tpu_config.project
                ):
                    logger.info("%s was preempted during setup; re-acquiring", tpu_name)
                    retry_count += 1
                    cleanup_preempted(config.tpu_type, project=config.project)
                    mark_setup_finished(tpu_name, tpu_config)
                    continue
                logger.error("Failed to setup TPU: %s", e)
                notifier.notify_error(tpu_name, f"Setup failed: {e}", config.command)
                mark_setup_finished(tpu_name, tpu_config)
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
                    workers=sync_workers,
                )
                # Dependencies first: the gemma helper installs into the venv, so it has to
                # exist by now. Whether the pod has a shared filesystem decides only *where*
                # that environment lives — not whether this step runs. A filer that was
                # never provisioned has no uv and no venv, so presence is what is checked.
                ensure_uv_environment(
                    tpu_name,
                    tpu_config.zone,
                    config.working_dir,
                    tpu_config.project,
                    nfs_mount_path=tpu_config.nfs_mount_path if tpu_config.uses_nfs else None,
                    nfs_user=config.nfs_user,
                    force=config.install_deps,
                )
                sync_gemma_helper(config, tpu_name, tpu_config, workers=sync_workers)
                install_gemma_helper(config, tpu_name, tpu_config, nfs_user=config.nfs_user)
                sync_wandb_credentials(tpu_name, tpu_config.zone, tpu_config.project)
            except Exception as e:
                # Same exposure as setup: a pod lost mid-sync surfaces as an rsync/ssh error.
                if preemption_retry_available(config, retry_count) and is_tpu_preempted(
                    tpu_name, tpu_config.zone, tpu_config.project
                ):
                    logger.info("%s was preempted during code sync; re-acquiring", tpu_name)
                    retry_count += 1
                    cleanup_preempted(config.tpu_type, project=config.project)
                    mark_setup_finished(tpu_name, tpu_config)
                    continue
                logger.error("Failed to sync code: %s", e)
                if isinstance(e, subprocess.CalledProcessError):
                    if e.stdout:
                        logger.error("Command stdout:\n%s", e.stdout)
                    if e.stderr:
                        logger.error("Command stderr:\n%s", e.stderr)
                notifier.notify_error(tpu_name, f"Code sync failed: {e}", config.command)
                mark_setup_finished(tpu_name, tpu_config)
                return 1

        num_workers = get_worker_count(config.tpu_type)
        runner = JobRunner(
            tpu_name,
            tpu_config.zone,
            tpu_config.project,
            config.working_dir,
            notifier,
            num_workers=num_workers,
            nfs_mount_path=tpu_config.nfs_mount_path,
            nfs_user=config.nfs_user,
        )

        mark_setup_finished(tpu_name, tpu_config)
        notifier.notify_started(tpu_name, config.tpu_type, config.command)
        start_time = time.time()

        try:
            runner.start_job(command_with_done_marker(config.command, config.done_marker))
        except Exception as e:
            logger.error("Failed to start job: %s", e)
            notifier.notify_error(tpu_name, f"Failed to start job: {e}", config.command)
            return 1

        run_post_launch_hook(config, tpu_name, tpu_config)

        if config.local_tmux and runner.create_local_tmux_session():
            logger.info(
                "Attach to local tmux session with: tmux attach-session -t %s",
                runner.local_session_name,
            )

        rss_breach: list[str] = []
        stop_guard = threading.Event()

        # Bind every loop variable as a default: the closure outlives one iteration of the
        # retry loop, and late binding would let it police the previous pod.
        def _guard(
            stop=stop_guard,
            alloc=allocation,
            breach=rss_breach,
            name=tpu_name,
            cfg=tpu_config,
            interval=config.rss_guard_interval,
        ) -> None:
            while not stop.wait(interval):
                over = worker_rss_over_limit(alloc)
                if over:
                    breach.append(over)
                    logger.error("Host RSS over limit on %s (%s); killing the run", name, over)
                    ssh_command(name, cfg.zone, "sudo pkill -9 python", project=cfg.project, worker="all", check=False)
                    return

        guard_thread = None
        if config.rss_guard_interval > 0:
            guard_thread = threading.Thread(target=_guard, daemon=True)
            guard_thread.start()

        try:
            try:
                status = runner.monitor_job(poll_interval=config.poll_interval)
            except Exception as monitor_error:
                # Monitoring throws when the pod stops answering — a timed-out ssh, a
                # vanished resource. That is overwhelmingly a preemption, and treating it
                # as a generic failure skips the retry loop that exists for exactly it.
                stop_guard.set()
                logger.warning("Monitoring %s failed (%s); checking the pod", tpu_name, monitor_error)
                if not is_tpu_preempted(tpu_name, tpu_config.zone, tpu_config.project):
                    raise
                logger.info("%s is gone; treating as preemption", tpu_name)
                status = JobStatus(state="preempted")
            stop_guard.set()
            duration = time.time() - start_time

            if rss_breach:
                # Relaunching would reproduce the same memory profile, so stop instead.
                notifier.notify_error(tpu_name, f"host RSS over limit ({rss_breach[0]})", config.command)
                return 1

            if status.state == "completed":
                # Clear the handshake here as well: reaching this branch means the run
                # finished in *this* process, and a marker left behind would make the next
                # legitimate launch of the same command a no-op.
                consume_done_marker(config.done_marker)
                notifier.notify_completion(tpu_name, config.command, duration, success=True)
                logger.info("Job completed successfully in %s", _format_duration(duration))
                return 0

            if status.state == "preempted":
                cleanup_preempted(config.tpu_type, project=config.project)

                can_retry = config.retry_on_preemption and (
                    config.max_retries is None or retry_count < config.max_retries
                )
                # A reserved retry would look for an idle pod that preemption just removed
                # and raise "no idle TPU"; only a spot launch can actually re-acquire.
                if can_retry and not config.spot:
                    logger.error(
                        "--retry-on-preemption needs --spot: a reserved launch has no way to "
                        "re-acquire capacity after its pod is gone."
                    )
                    can_retry = False

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
