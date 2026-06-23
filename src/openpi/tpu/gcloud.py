"""Low-level gcloud command wrappers for TPU operations."""

import json
import logging
import os
from pathlib import Path
import subprocess

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


def run_gcloud(
    args: list[str],
    *,
    project: str = DEFAULT_PROJECT,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute a gcloud command.

    Args:
        args: Arguments to pass to gcloud
        project: GCP project ID
        check: Whether to raise on non-zero exit code
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess with stdout/stderr as strings
    """
    cmd = ["gcloud", "--project", project, *args]
    logger.debug("Running: %s", " ".join(cmd))

    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=True,
    )


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


def describe_tpu(name: str, zone: str, project: str = DEFAULT_PROJECT) -> dict | None:
    """Get TPU VM details.

    Args:
        name: TPU VM name
        zone: GCP zone
        project: GCP project ID

    Returns:
        TPU VM dictionary, or None if not found
    """
    result = run_gcloud(
        ["compute", "tpus", "tpu-vm", "describe", name, "--zone", zone, "--format=json"],
        project=project,
        check=False,
    )
    if result.returncode != 0:
        return None
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
        TPU state string (e.g., "READY", "PREEMPTED"), or None if not found
    """
    result = run_gcloud(
        ["compute", "tpus", "tpu-vm", "describe", name, "--zone", zone, "--format=value(state)"],
        project=project,
        check=False,
    )
    if result.returncode != 0:
        return None
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


def ssh_command(
    tpu_name: str,
    zone: str,
    command: str,
    *,
    project: str = DEFAULT_PROJECT,
    worker: str = "all",
    check: bool = True,
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
    return run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            _apply_ssh_user(tpu_name),
            "--zone",
            zone,
            f"--worker={worker}",
            f"--command={command}",
        ],
        project=project,
        check=check,
    )


def scp_to_tpu(
    local_path: str | Path,
    tpu_name: str,
    remote_path: str,
    zone: str,
    *,
    project: str = DEFAULT_PROJECT,
    worker: str = "0",
) -> subprocess.CompletedProcess[str]:
    """Copy a file to a TPU.

    Args:
        local_path: Local file path
        tpu_name: TPU VM name
        remote_path: Remote destination path
        zone: GCP zone
        project: GCP project ID
        worker: Worker to copy to (default "0" for shared NFS)

    Returns:
        CompletedProcess result
    """
    return run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            str(local_path),
            f"{_apply_ssh_user(tpu_name)}:{remote_path}",
            "--zone",
            zone,
            f"--worker={worker}",
        ],
        project=project,
    )


def scp_from_tpu(
    tpu_name: str,
    remote_path: str,
    local_path: str | Path,
    zone: str,
    *,
    project: str = DEFAULT_PROJECT,
    worker: str = "0",
) -> subprocess.CompletedProcess[str]:
    """Copy a file from a TPU.

    Args:
        tpu_name: TPU VM name
        remote_path: Remote file path
        local_path: Local destination path
        zone: GCP zone
        project: GCP project ID
        worker: Worker to copy from

    Returns:
        CompletedProcess result
    """
    return run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            f"{_apply_ssh_user(tpu_name)}:{remote_path}",
            str(local_path),
            "--zone",
            zone,
            f"--worker={worker}",
        ],
        project=project,
    )
