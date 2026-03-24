"""High-level TPU management functions."""

import logging
import time

from openpi.tpu.config import get_tpu_config
from openpi.tpu.config import get_tpu_type_prefix
from openpi.tpu.gcloud import create_queued_resource
from openpi.tpu.gcloud import delete_queued_resource
from openpi.tpu.gcloud import delete_tpu_vm
from openpi.tpu.gcloud import describe_queued_resource
from openpi.tpu.gcloud import get_tpu_state
from openpi.tpu.gcloud import list_queued_resources
from openpi.tpu.gcloud import list_tpus
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)


def _get_short_name(full_name: str) -> str:
    """Extract short name from full resource path.

    The gcloud API returns full paths like:
    "projects/cmu-aidm-v2/locations/us-central1-b/nodes/v6e-0"

    This extracts just "v6e-0".
    """
    return full_name.rsplit("/", 1)[-1]


def find_available_tpu(tpu_type: str) -> str | None:
    """Find an existing TPU that is ready and has no python processes running.

    Args:
        tpu_type: TPU type string like "v6e-8"

    Returns:
        Name of an available TPU, or None if none found
    """
    config = get_tpu_config(tpu_type)
    type_prefix = get_tpu_type_prefix(tpu_type)

    tpus = list_tpus(config.zone, config.project)

    for tpu in tpus:
        full_name = tpu.get("name", "")
        name = _get_short_name(full_name)
        state = tpu.get("state", "")
        accel_type = tpu.get("acceleratorType", "")

        if not name.startswith(type_prefix):
            continue

        if state != "READY":
            logger.debug("TPU %s is not READY (state=%s)", name, state)
            continue

        if config.accelerator_type not in accel_type and tpu_type not in accel_type:
            logger.debug("TPU %s has wrong accelerator type: %s", name, accel_type)
            continue

        if is_tpu_available(name, config.zone, config.project):
            logger.info("Found available TPU: %s", name)
            return name

    return None


def is_tpu_available(tpu_name: str, zone: str, project: str) -> bool:
    """Check if a TPU has no python processes running.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        project: GCP project ID

    Returns:
        True if no python processes are running
    """
    try:
        result = ssh_command(
            tpu_name,
            zone,
            "pgrep python || true",
            project=project,
            worker="0",
            check=False,
        )
        has_python = bool(result.stdout.strip())
        if has_python:
            logger.debug("TPU %s has python processes running", tpu_name)
        return not has_python
    except Exception as e:
        logger.warning("Failed to check TPU %s availability: %s", tpu_name, e)
        return False


def create_tpu(tpu_type: str) -> str:
    """Create a new TPU.

    First cleans up any preempted VMs or suspended queued resources,
    then creates a new queued resource.

    Args:
        tpu_type: TPU type string like "v6e-8"

    Returns:
        Name of the created TPU
    """
    config = get_tpu_config(tpu_type)
    type_prefix = get_tpu_type_prefix(tpu_type)

    cleanup_preempted(tpu_type)

    # Collect all existing names from both queued resources and TPU VMs
    existing_names: set[str] = set()

    queued = list_queued_resources(
        config.zone,
        config.project,
        filter_expr=f"state.state!=SUSPENDED AND name~'^{type_prefix}-'",
    )
    for qr in queued:
        existing_names.add(_get_short_name(qr.get("name", "")))

    tpus = list_tpus(config.zone, config.project)
    for tpu in tpus:
        name = _get_short_name(tpu.get("name", ""))
        if name.startswith(type_prefix):
            existing_names.add(name)

    # Find next available index
    next_index = 0
    while f"{type_prefix}-{next_index}" in existing_names:
        next_index += 1
    tpu_name = f"{type_prefix}-{next_index}"

    logger.info("Creating TPU %s with type %s", tpu_name, config.accelerator_type)

    create_queued_resource(
        name=tpu_name,
        zone=config.zone,
        accelerator_type=config.accelerator_type,
        runtime_version=config.runtime_version,
        project=config.project,
        spot=config.is_spot,
        reserved=not config.is_spot,
    )

    return tpu_name


def wait_for_tpu_ready(
    tpu_name: str,
    zone: str,
    project: str,
    timeout: float = 600,
    poll_interval: float = 10,
) -> bool:
    """Wait for a TPU to become READY.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        project: GCP project ID
        timeout: Maximum time to wait in seconds
        poll_interval: Time between status checks in seconds

    Returns:
        True if TPU is READY, False if timeout reached
    """
    start_time = time.time()
    last_state = None

    while time.time() - start_time < timeout:
        state = get_tpu_state(tpu_name, zone, project)

        if state != last_state:
            logger.info("TPU %s state: %s", tpu_name, state)
            last_state = state

        if state == "READY":
            return True

        if state == "PREEMPTED":
            logger.warning("TPU %s was preempted", tpu_name)
            return False

        if state is None:
            queued = describe_queued_resource(tpu_name, zone, project)
            if queued:
                qr_state = queued.get("state", {}).get("state", "UNKNOWN")
                if qr_state != last_state:
                    logger.info("Queued resource %s state: %s", tpu_name, qr_state)
                    last_state = qr_state
                if qr_state == "SUSPENDED":
                    logger.warning("Queued resource %s was suspended", tpu_name)
                    return False

        time.sleep(poll_interval)

    logger.warning("Timeout waiting for TPU %s to become READY", tpu_name)
    return False


def cleanup_preempted(tpu_type: str) -> None:
    """Delete preempted VMs and suspended queued resources.

    Args:
        tpu_type: TPU type string like "v6e-8"
    """
    config = get_tpu_config(tpu_type)
    type_prefix = get_tpu_type_prefix(tpu_type)

    tpus = list_tpus(config.zone, config.project)
    for tpu in tpus:
        full_name = tpu.get("name", "")
        name = _get_short_name(full_name)
        state = tpu.get("state", "")
        accel_type = tpu.get("acceleratorType", "")

        if not name.startswith(type_prefix):
            continue
        if type_prefix not in accel_type and config.accelerator_type not in accel_type:
            continue
        if state != "PREEMPTED":
            continue

        logger.info("Deleting preempted TPU VM: %s", name)
        try:
            delete_tpu_vm(name, config.zone, config.project)
        except Exception as e:
            logger.warning("Failed to delete preempted TPU %s: %s", name, e)

    queued = list_queued_resources(
        config.zone,
        config.project,
        filter_expr=f"state.state=SUSPENDED AND tpu.nodeSpec.node.acceleratorType~'{type_prefix}'",
    )
    for qr in queued:
        full_name = qr.get("name", "")
        name = _get_short_name(full_name)
        logger.info("Deleting suspended queued resource: %s", name)
        try:
            delete_queued_resource(name, config.zone, config.project)
        except Exception as e:
            logger.warning("Failed to delete suspended queued resource %s: %s", name, e)


def is_tpu_preempted(tpu_name: str, zone: str, project: str) -> bool:
    """Check if a TPU has been preempted.

    Args:
        tpu_name: TPU VM name
        zone: GCP zone
        project: GCP project ID

    Returns:
        True if TPU is in PREEMPTED state
    """
    state = get_tpu_state(tpu_name, zone, project)
    return state == "PREEMPTED"
