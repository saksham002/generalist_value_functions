"""Low-level gcloud command wrappers.

Every external call in this package funnels through :func:`run_gcloud`, which is what
makes retry policy, timeout policy and credential-expiry detection single-sourced — and
what makes the layers above it testable by substituting one function.
"""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "cmu-aidm-v2"


def _apply_ssh_user(tpu_name: str) -> str:
    """Prefix `<user>@` to tpu_name when OPENPI_TPU_SSH_USER is set.

    Lets you launch from a machine whose local Linux user differs from the
    user the TPU NFS dir is owned by (e.g. lab-PC `huzheyuan` -> NFS owned by
    `jeffyu`). Requires OS Login on the GCP project to allow that mapping.
    """
    user = os.environ.get("OPENPI_TPU_SSH_USER")
    return f"{user}@{tpu_name}" if user else tpu_name


# gcloud ssh retries internally and will sit on an unreachable or deleted pod
# indefinitely, which silently wedges a launcher. Every call gets a ceiling.
DEFAULT_GCLOUD_TIMEOUT_SECONDS = 900

_AUTH_FAILURE_MARKERS = (
    "reauthentication failed",
    "reauthentication required",
    "please run:\n\n  $ gcloud auth login",
    "credentials are no longer valid",
    "invalid_grant",
)
# One alert per process: once credentials lapse every later call fails identically, and a
# message per failed call would bury the one that matters.
_auth_alert_state = {"sent": False}

# Naming the account matters: with two credentialed accounts a bare `gcloud auth login`
# opens an interactive picker that hangs instead of prompting.
_REAUTH_COMMAND = "gcloud auth login saksham3@andrew.cmu.edu --force --no-browser"


def alert_on_auth_failure(stderr: str | None, *, context: str = "") -> bool:
    """Slack immediately if `stderr` shows gcloud credentials have lapsed.

    A launcher cannot recover from this on its own — every subsequent gcloud call fails the
    same way, so a run that is otherwise healthy dies unattended. The only useful response
    is to tell someone to re-authenticate while the job is still salvageable.
    """
    if not stderr or _auth_alert_state["sent"]:
        return False
    lowered = stderr.lower()
    if not any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
        return False

    _auth_alert_state["sent"] = True
    logger.error("gcloud credentials have expired%s; run: %s", f" ({context})" if context else "", _REAUTH_COMMAND)
    message = (
        f"gcloud auth EXPIRED while run_on_tpu was running{f' ({context})' if context else ''}. "
        f"TPU launcher cannot continue until you re-authenticate: {_REAUTH_COMMAND}"
    )
    try:
        subprocess.run(
            [sys.executable, str(Path.home() / "utils" / "slack.py"), message],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except Exception as e:
        logger.warning("Could not send auth-failure Slack alert: %s", e)
    return True


# Failures that clear on their own. `gcloud ssh` collapses every ssh-level problem into
# exit 255, and a `--worker=all` call fails as a whole if any single host blips — with 32
# workers that makes transient failure the common case rather than the exception.
_TRANSIENT_MARKERS = (
    "connection closed",
    "connection reset",
    "connection refused",
    "connection timed out",
    "broken pipe",
    "kex_exchange_identification",
    "ssh_exchange_identification",
    "no route to host",
    "temporary failure in name resolution",
    "could not resolve hostname",
    "operation timed out",
    "deadline exceeded",
    "backend error",
    "internal error",
    "service unavailable",
    "try again",
)
_SSH_FAILURE_RETURNCODE = 255
# 137 is SIGKILL. On a shared pod that is apt being killed under memory pressure or lock
# contention rather than a broken command, and it clears on a retry.
_KILLED_RETURNCODE = 137

# States a pod never returns from, so nothing is gained by trying it again. A deleted pod
# reports no state at all, which counts the same way.
_TERMINAL_TPU_STATES = (None, "PREEMPTED", "TERMINATED", "DELETING")

# "The API did not answer", which is not a statement about the pod. Deliberately outside
# _TERMINAL_TPU_STATES: an unreachable control plane once read as a deleted pod, so a
# lapsed credential looked exactly like a preemption and killed runs whose pods were
# healthy — including one that had already finished training.
TPU_STATE_UNKNOWN = "UNKNOWN"

# gcloud says this, and only this, when the resource genuinely is not there.
_NOT_FOUND_MARKERS = ("not_found", "was not found", "could not be found")


def _describe_failure_state(result: subprocess.CompletedProcess[str]) -> str | None:
    """Map a failed describe onto a state: absent (None) or unreachable (UNKNOWN)."""
    lowered = (result.stderr or "").lower()
    if any(marker in lowered for marker in _NOT_FOUND_MARKERS):
        return None
    return TPU_STATE_UNKNOWN


def is_transient_failure(returncode: int, stderr: str | None) -> bool:
    """Whether a failed gcloud call is worth simply running again."""
    if returncode in (_SSH_FAILURE_RETURNCODE, _KILLED_RETURNCODE):
        return True
    lowered = (stderr or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def run_gcloud(
    args: list[str],
    *,
    project: str = DEFAULT_PROJECT,
    check: bool = True,
    capture_output: bool = True,
    timeout: float | None = DEFAULT_GCLOUD_TIMEOUT_SECONDS,
    retries: int = 3,
    retry_delay: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    """Execute a gcloud command, retrying failures that are transient.

    Retrying lives here rather than at call sites because every gcloud and ssh call in this
    package funnels through this function: handling it per-call-site is what let the same
    class of failure reappear in setup, in state polling and in monitoring, each needing its
    own fix. A timeout counts as transient — `check=False` suppresses a non-zero exit but
    never a `TimeoutExpired`, so an unhandled one kills an otherwise healthy launcher.

    Expired credentials are deliberately not retried: they never clear on their own.

    Args:
        retries: Extra attempts after the first for a transient failure. 0 disables.
        retry_delay: Seconds between attempts, growing linearly with attempt number.
    """
    cmd = ["gcloud", "--project", project, *args]
    logger.debug("Running: %s", " ".join(cmd))
    label = " ".join(args[:3])

    for attempt in range(retries + 1):
        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=capture_output,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            result = subprocess.CompletedProcess(cmd, _SSH_FAILURE_RETURNCODE, "", f"timed out after {timeout}s")

        if result.returncode == 0:
            return result
        if alert_on_auth_failure(result.stderr, context=label):
            break
        if attempt < retries and is_transient_failure(result.returncode, result.stderr):
            delay = retry_delay * (attempt + 1)
            logger.info(
                "gcloud %s failed transiently (rc=%s%s); retry %d/%d in %.0fs",
                label,
                result.returncode,
                ", timeout" if timed_out else "",
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
            continue
        break

    if timed_out and check:
        raise subprocess.TimeoutExpired(cmd, timeout)
    if check:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def list_tpus(zone: str, project: str = DEFAULT_PROJECT) -> list[dict]:
    """List TPU VMs in a zone.

    Args:
        zone: GCP zone
        project: GCP project ID

    Returns:
        List of TPU VM dictionaries
    """
    result = run_gcloud(
        ["compute", "tpus", "tpu-vm", "list", "--zone", zone, "--format=json"],
        project=project,
    )
    if not result.stdout.strip():
        return []
    return json.loads(result.stdout)


def list_queued_resources(
    zone: str,
    project: str = DEFAULT_PROJECT,
    filter_expr: str | None = None,
) -> list[dict]:
    """List queued TPU resources in a zone.

    Args:
        zone: GCP zone
        project: GCP project ID
        filter_expr: Optional filter expression

    Returns:
        List of queued resource dictionaries
    """
    args = ["compute", "tpus", "queued-resources", "list", "--zone", zone, "--format=json"]
    if filter_expr:
        args.extend(["--filter", filter_expr])

    result = run_gcloud(args, project=project)
    if not result.stdout.strip():
        return []
    return json.loads(result.stdout)


def describe_queued_resource(name: str, zone: str, project: str = DEFAULT_PROJECT) -> dict | None:
    """Get queued resource details.

    Args:
        name: Queued resource name
        zone: GCP zone
        project: GCP project ID

    Returns:
        Queued resource dictionary, or None if not found
    """
    result = run_gcloud(
        ["compute", "tpus", "queued-resources", "describe", name, "--zone", zone, "--format=json"],
        project=project,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def get_tpu_state(name: str, zone: str, project: str = DEFAULT_PROJECT) -> str | None:
    """Get the state of a TPU VM.

    Args:
        name: TPU VM name
        zone: GCP zone
        project: GCP project ID

    Returns:
        TPU state string (e.g., "READY", "PREEMPTED"), None if the pod does not exist, or
        ``TPU_STATE_UNKNOWN`` if the call itself failed and the pod's state is unknown.
    """
    try:
        result = run_gcloud(
            ["compute", "tpus", "tpu-vm", "describe", name, "--zone", zone, "--format=value(state)"],
            project=project,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TPU_STATE_UNKNOWN
    if result.returncode != 0:
        return _describe_failure_state(result)
    return result.stdout.strip()


def create_queued_resource(
    name: str,
    zone: str,
    accelerator_type: str,
    runtime_version: str,
    *,
    project: str = DEFAULT_PROJECT,
    spot: bool = False,
    reserved: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Create a queued TPU resource.

    Args:
        name: Resource name
        zone: GCP zone
        accelerator_type: TPU accelerator type (e.g., "v6e-8", "v5litepod-128")
        runtime_version: TPU runtime version
        project: GCP project ID
        spot: Whether to use spot/preemptible instances
        reserved: Whether to use reserved capacity

    Returns:
        CompletedProcess result
    """
    args = [
        "compute",
        "tpus",
        "queued-resources",
        "create",
        name,
        "--node-id",
        name,
        "--zone",
        zone,
        "--accelerator-type",
        accelerator_type,
        "--runtime-version",
        runtime_version,
    ]

    if spot:
        args.append("--spot")
    if reserved:
        args.append("--reserved")

    return run_gcloud(args, project=project)


def delete_queued_resource(name: str, zone: str, project: str = DEFAULT_PROJECT) -> subprocess.CompletedProcess[str]:
    """Delete a queued TPU resource.

    Args:
        name: Resource name
        zone: GCP zone
        project: GCP project ID

    Returns:
        CompletedProcess result
    """
    return run_gcloud(
        ["compute", "tpus", "queued-resources", "delete", name, "--zone", zone, "--quiet"],
        project=project,
    )


def delete_tpu_vm(name: str, zone: str, project: str = DEFAULT_PROJECT) -> subprocess.CompletedProcess[str]:
    """Delete a TPU VM.

    Args:
        name: TPU VM name
        zone: GCP zone
        project: GCP project ID

    Returns:
        CompletedProcess result
    """
    return run_gcloud(
        ["compute", "tpus", "tpu-vm", "delete", name, "--zone", zone, "--quiet"],
        project=project,
    )


def get_tpu_state_and_health(tpu_name: str, zone: str, project: str) -> tuple[str | None, str | None]:
    """Return ``(state, health)`` for a TPU, or ``(None, None)`` if it cannot be determined.

    This is called in a polling loop, so a slow or failed describe must read as "no answer
    this round" rather than propagate: ``check=False`` suppresses a non-zero exit but not a
    timeout, and an unhandled one kills a launcher that is otherwise healthy.
    """
    try:
        result = run_gcloud(
            [
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                tpu_name,
                "--zone",
                zone,
                "--format=value(state,health)",
            ],
            project=project,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        logger.warning("describe %s in %s timed out; state is unknown", tpu_name, zone)
        return TPU_STATE_UNKNOWN, None
    if result.returncode != 0:
        state = _describe_failure_state(result)
        if state == TPU_STATE_UNKNOWN:
            logger.warning("describe %s in %s failed; state is unknown, not assuming the pod is gone", tpu_name, zone)
        return state, None
    parts = result.stdout.strip().split()
    state = parts[0] if parts else None
    health = parts[1] if len(parts) > 1 else None
    return state, health


def ssh_command(
    tpu_name: str,
    zone: str,
    command: str,
    *,
    project: str = DEFAULT_PROJECT,
    worker: str = "all",
    check: bool = True,
    timeout: float | None = DEFAULT_GCLOUD_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Execute a command on a TPU via SSH.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        command: Command to execute
        project: GCP project ID
        worker: Worker specification ("all", "0", etc.)
        check: Whether to raise on non-zero exit code

    Returns:
        CompletedProcess result
    """
    args = [
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        _apply_ssh_user(tpu_name),
        "--zone",
        zone,
        f"--worker={worker}",
        f"--command={command}",
    ]
    # Try once without internal retries first. ssh to a pod that no longer exists hangs
    # instead of refusing, so every attempt costs a full timeout; asking the API whether the
    # pod still exists is far cheaper than spending two more of those on a pod that cannot
    # come back.
    result = run_gcloud(args, project=project, check=False, timeout=timeout, retries=0)
    if result.returncode == 0:
        return result

    state, _ = get_tpu_state_and_health(tpu_name, zone, project)
    if state in _TERMINAL_TPU_STATES:
        logger.info("%s is %s; not retrying ssh against it", tpu_name, state)
        if check:
            raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
        return result

    # The pod is still there, so the failure is worth the usual retries.
    return run_gcloud(args, project=project, check=check, timeout=timeout)
