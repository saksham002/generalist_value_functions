"""Allocate TPUs: reuse an idle pod, or race spot capacity across every zone with quota.

Nothing here derives a zone from a TPU type. A pod either already exists — in which case
Cloud Asset Inventory finds it wherever it lives — or it is being created, in which case
the zone is chosen by the race and travels with the resulting :class:`TPUAllocation`.
"""

import concurrent.futures
import dataclasses
import logging
import threading
import time

from openpi.tpu import discovery
from openpi.tpu.config import TPUConfigWithType
from openpi.tpu.config import accelerator_type_for
from openpi.tpu.config import get_tpu_name_prefix
from openpi.tpu.config import get_tpu_type_prefix
from openpi.tpu.config import get_tpu_user
from openpi.tpu.config import resolve_from_pod
from openpi.tpu.config import spot_race_configs
from openpi.tpu.config import with_discovered_nfs
from openpi.tpu.gcloud import create_queued_resource
from openpi.tpu.gcloud import delete_queued_resource
from openpi.tpu.gcloud import delete_tpu_vm
from openpi.tpu.gcloud import describe_queued_resource
from openpi.tpu.gcloud import get_tpu_state
from openpi.tpu.gcloud import get_tpu_state_and_health
from openpi.tpu.gcloud import list_queued_resources
from openpi.tpu.gcloud import list_tpus
from openpi.tpu.gcloud import ssh_command
from openpi.tpu.setup import mark_setup_finished
from openpi.tpu.setup import try_claim_setup

logger = logging.getLogger(__name__)

# A zone that answers "no capacity" clears on its own and is worth retrying; a zone that
# answers "quota exceeded" never clears within a race and is dropped.
_STOCKOUT_MARKERS = ("no more capacity", "code 8", "resource_exhausted")
_QUOTA_MARKERS = ("quota limit", "code 429", "has been exceeded", "exhausted. limit")
_EXISTS_MARKERS = ("already exists", "code 6", "alreadyexists")
# Having quota for a shape in a zone does not mean the zone offers it, so this is only
# discoverable by asking. It never clears, which makes the zone worthless for this race.
_UNSUPPORTED_MARKERS = ("zonal accelerator configurations", "was not found in zone")


@dataclasses.dataclass(frozen=True)
class TPUAllocation:
    """A pod and the fully resolved config describing where it actually landed."""

    name: str
    config: TPUConfigWithType


def _get_short_name(full_name: str) -> str:
    """Extract short name from a full resource path.

    "projects/cmu-aidm-v2/locations/us-central1-b/nodes/v6e-0" -> "v6e-0"
    """
    return full_name.rsplit("/", 1)[-1]


def classify_create_failure(message: str) -> str:
    """Return 'exists', 'stockout', 'quota', 'unsupported' or 'unknown' for a failed create."""
    lowered = message.lower()
    if any(marker in lowered for marker in _QUOTA_MARKERS):
        return "quota"
    if any(marker in lowered for marker in _UNSUPPORTED_MARKERS):
        return "unsupported"
    if any(marker in lowered for marker in _STOCKOUT_MARKERS):
        return "stockout"
    if any(marker in lowered for marker in _EXISTS_MARKERS):
        return "exists"
    return "unknown"


def create_failure_reason(error: Exception) -> str:
    """The part of a failed gcloud call that says why it failed.

    ``CalledProcessError`` stringifies to the command line, so logging the front of it
    prints argv and hides the diagnosis behind a truncation.
    """
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    return " ".join(stderr.split())[-300:] if stderr else str(error)[-300:]


def is_tpu_available(tpu_name: str, zone: str, project: str, *, ssh_user: str | None = None) -> bool:
    """Whether a TPU is reachable AND has no python processes running.

    Reachability is checked first and deliberately: an ssh that fails produces empty
    stdout, which would otherwise read as "no python running" and mark an unreachable or
    already-deleted pod as idle. Cloud Asset Inventory lags deletions, so that phantom is
    a routine occurrence, not an edge case.
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
    except Exception as e:
        logger.warning("Failed to reach TPU %s: %s", tpu_name, e)
        return False
    if result.returncode != 0:
        logger.info("TPU %s is unreachable (ssh exit %s); not treating it as idle", tpu_name, result.returncode)
        return False
    has_python = bool(result.stdout.strip())
    if has_python:
        logger.debug("TPU %s has python processes running", tpu_name)
    return not has_python


def _is_tpu_usable(tpu_name: str, config: TPUConfigWithType) -> bool:
    """Whether a pod is ready to be handed a job.

    State and health from the API only. An ssh probe was tried and removed: it is what a
    launcher hangs on when a pod is deleted underneath it, and READY/HEALTHY is already
    the condition that matters.
    """
    state, health = get_tpu_state_and_health(tpu_name, config.zone, config.project)
    if state != "READY":
        logger.debug("TPU %s in %s is %s, not READY", tpu_name, config.zone, state)
        return False
    # A fresh pod reports empty health briefly; only an explicitly unhealthy one is rejected.
    if health and health != "HEALTHY":
        logger.info("TPU %s in %s is READY but %s", tpu_name, config.zone, health)
        return False
    return True


def running_process_owners(tpu_name: str, config: TPUConfigWithType) -> set[str] | None:
    """Usernames owning python processes on the pod, or None if that cannot be determined.

    Returning None rather than an empty set matters: an unreachable worker must not be read
    as "nobody is running anything", which would licence killing a live job.
    """
    try:
        result = ssh_command(
            tpu_name,
            config.zone,
            "ps -eo user=,comm= | awk '$2 ~ /^python/ {print $1}' | sort -u",
            project=config.project,
            worker="all",
            check=False,
            timeout=180,
        )
    except Exception as e:
        logger.warning("Could not read process owners on %s: %s", tpu_name, e)
        return None
    if result.returncode != 0:
        return None
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith(("SSH", "Using", "Warning"))
    }


def reclaim_tpu(tpu_name: str, config: TPUConfigWithType) -> None:
    """Clear leftover state from an owned pod so it can be reused.

    Kills stray training processes and removes the libtpu lockfile, which JAX leaves
    behind when a job dies without cleanup and which otherwise blocks the next run.
    """
    logger.warning("Reclaiming %s in %s: killing processes and clearing lockfile", tpu_name, config.zone)
    ssh_command(
        tpu_name,
        config.zone,
        "sudo pkill -9 python 2>/dev/null; sudo rm -f /tmp/libtpu_lockfile",
        project=config.project,
        worker="all",
        check=False,
        timeout=300,
    )


def find_available_tpu(
    tpu_type: str,
    *,
    project: str,
    user: str,
    resource_owner: str | None = None,
    is_spot: bool = False,
    reclaim_owned: bool = False,
) -> TPUAllocation | None:
    """Find an existing READY pod of this shape anywhere in the project.

    Searches every location via Cloud Asset Inventory rather than one configured zone, so
    a pod created in a zone this code has never heard of is still reused.

    Any READY, idle pod of the right shape is reused regardless of whose name it carries.
    ``reclaim_owned`` additionally takes back a *busy* pod, but only one whose name carries
    ``resource_owner`` — killing another user's processes is never acceptable, while
    borrowing a pod they are not using is.
    """
    wanted_accelerator = accelerator_type_for(tpu_type)
    type_prefix = get_tpu_type_prefix(tpu_type)

    # Borrowing someone else's idle pod is allowed, but this user's own pods are tried
    # first: taking a colleague's while one of yours sits free needlessly puts your work
    # on a pod they may want back.
    candidates = sorted(
        discovery.list_all_tpus(project),
        key=lambda pod: 0 if resource_owner and resource_owner in pod.name else 1,
    )

    for pod in candidates:
        if pod.state != "READY" or not pod.name.startswith(type_prefix):
            continue
        # Ownership does not gate *use*: an idle pod of the right shape is usable by
        # anyone. It gates only whether we may kill what is running on it.
        owned = resource_owner is not None and resource_owner in pod.name
        try:
            described = discovery.describe_pod(pod.name, pod.zone, project)
            if described.accelerator_type != wanted_accelerator:
                continue
            config = resolve_from_pod(pod.name, user=user, project=project, tpu_type=tpu_type)
        except Exception as e:
            # Almost always a pod deleted since the index was built; skip and keep looking.
            logger.info("Skipping %s: %s", pod.name, str(e)[:120])
            continue
        if is_spot:
            # Reusing a RESERVED pod for a spot launch silently defeats the request: the
            # caller asked for preemptible capacity, and stamping is_spot on a reserved
            # pod's config only relabels it. Skip anything GCP did not create as spot.
            if described.spot is not True:
                logger.info("Skipping %s in %s: not a spot pod", pod.name, pod.zone)
                continue
            config = dataclasses.replace(config, is_spot=True)
        # Same bar as the race: a pod that is not READY/HEALTHY is adopted only to fail
        # later during sync. Asset Inventory's state lags, so this re-checks it live.
        if not _is_tpu_usable(pod.name, config):
            logger.info("Skipping %s in %s: not READY/HEALTHY", pod.name, config.zone)
            continue
        # "No python running" is not the same as free: a pod another launcher is still
        # setting up has nothing running yet, and claiming it makes the two race on apt
        # and ssh until both fail.
        # Claim atomically rather than checking then claiming: two launchers sweeping in
        # the same second would both pass a check and both take the pod.
        if not try_claim_setup(pod.name, config):
            logger.info("Skipping %s in %s: another launcher claimed it", pod.name, config.zone)
            continue
        if is_tpu_available(pod.name, config.zone, project, ssh_user=config.ssh_user):
            logger.info("Reusing available TPU %s in %s", pod.name, config.zone)
            return TPUAllocation(name=pod.name, config=config)
        # The claim is taken before the pod is known to be idle, so a pod that turns out to
        # be busy has to have it handed back. Leaving it behind would make every later sweep
        # skip a pod this launcher never used.
        mark_setup_finished(pod.name, config)
        if reclaim_owned and owned:
            # Never kill this user's own work. Anything else on their pod is stale or
            # foreign and may be cleared. Undetermined ownership counts as "theirs".
            remote_login = config.remote_home.rstrip("/").rsplit("/", 1)[-1]
            owners = running_process_owners(pod.name, config)
            if owners is None:
                logger.info("%s is busy and process ownership is unknown; leaving it alone", pod.name)
                continue
            if remote_login in owners:
                logger.info("%s is running %s's own processes; leaving it alone", pod.name, remote_login)
                continue
            logger.info("%s is busy with foreign/stale processes %s", pod.name, sorted(owners))
            reclaim_tpu(pod.name, config)
            if is_tpu_available(pod.name, config.zone, project, ssh_user=config.ssh_user):
                logger.info("Reclaimed TPU %s in %s", pod.name, config.zone)
                return TPUAllocation(name=pod.name, config=config)
            logger.warning("%s still busy after reclaim; leaving it alone", pod.name)
    return None


def _next_free_name(prefix: str, config: TPUConfigWithType) -> str:
    """Name the next pod ``<prefix>-<highest existing index + 1>`` in one zone.

    Deliberately not the lowest unused index: reusing a gap left by a deleted pod makes
    names ambiguous across a pod's lifetime, so the counter only ever moves forward.
    Both queued resources and VMs are counted, since a queued resource holds the name
    before any VM exists.
    """
    existing: set[str] = set()
    for qr in list_queued_resources(
        config.zone, config.project, filter_expr=f"state.state!=SUSPENDED AND name~'^{prefix}-'"
    ):
        existing.add(_get_short_name(qr.get("name", "")))
    for tpu in list_tpus(config.zone, config.project):
        existing.add(_get_short_name(tpu.get("name", "")))

    highest = -1
    for name in existing:
        # Exact prefix only: "<prefix>-<digits>" and nothing else.
        if not name.startswith(f"{prefix}-"):
            continue
        suffix = name[len(prefix) + 1 :]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{prefix}-{highest + 1}"


def create_tpu(tpu_type: str, config: TPUConfigWithType, *, tpu_name: str | None = None) -> str:
    """Create one pod in ``config.zone``. Returns the name."""
    prefix = get_tpu_name_prefix(tpu_type, resource_owner=config.resource_owner, is_spot=config.is_spot)
    name = tpu_name or _next_free_name(prefix, config)

    logger.info("Creating TPU %s (%s) in %s", name, config.accelerator_type, config.zone)
    create_queued_resource(
        name=name,
        zone=config.zone,
        accelerator_type=config.accelerator_type,
        runtime_version=config.runtime_version,
        project=config.project,
        spot=config.is_spot,
        reserved=not config.is_spot,
    )
    return name


def wait_for_tpu_ready(
    tpu_name: str,
    zone: str,
    project: str,
    timeout: float = 600,
    poll_interval: float = 10,
) -> bool:
    """Wait for a TPU to become READY."""
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


def _delete_candidate(tpu_name: str, config: TPUConfigWithType) -> None:
    """Delete one candidate's VM and its queued resource, tolerating absence."""
    for delete, label in ((delete_tpu_vm, "VM"), (delete_queued_resource, "queued resource")):
        try:
            delete(tpu_name, config.zone, config.project)
        except Exception as e:
            logger.debug("Deleting %s %s in %s: %s", label, tpu_name, config.zone, e)


def delete_tpu_allocation(allocation: TPUAllocation) -> None:
    """Tear down one allocation's VM and queue."""
    _delete_candidate(allocation.name, allocation.config)


def cleanup_preempted(tpu_type: str, *, project: str) -> None:
    """Delete preempted VMs and suspended queued resources for a shape, in every zone."""
    type_prefix = get_tpu_type_prefix(tpu_type)
    wanted_accelerator = accelerator_type_for(tpu_type)

    zones_seen: set[str] = set()
    for pod in discovery.list_all_tpus(project):
        if not pod.name.startswith(type_prefix):
            continue
        zones_seen.add(pod.zone)
        if pod.state != "PREEMPTED":
            continue
        logger.info("Deleting preempted TPU VM %s in %s", pod.name, pod.zone)
        try:
            delete_tpu_vm(pod.name, pod.zone, project)
        except Exception as e:
            logger.warning("Failed to delete preempted TPU %s: %s", pod.name, e)

    for zone in sorted(zones_seen):
        queued = list_queued_resources(
            zone,
            project,
            filter_expr=(
                f"(state.state=SUSPENDED OR state.state=FAILED) AND "
                f"tpu.nodeSpec.node.acceleratorType~'{wanted_accelerator}'"
            ),
        )
        for qr in queued:
            name = _get_short_name(qr.get("name", ""))
            logger.info("Deleting suspended queued resource %s in %s", name, zone)
            try:
                delete_queued_resource(name, zone, project)
            except Exception as e:
                logger.warning("Failed to delete suspended queued resource %s: %s", name, e)


def is_tpu_preempted(tpu_name: str, zone: str, project: str) -> bool:
    """Whether a TPU is gone as far as a running job is concerned.

    A pod that has been deleted outright reports no state at all rather than PREEMPTED —
    GCP removes the resource, it does not leave a tombstone. Treating only the PREEMPTED
    string as preemption means a deleted pod falls through to the generic failure path and
    skips the retry that exists for exactly this case.
    """
    state = get_tpu_state(tpu_name, zone, project)
    if state is None:
        logger.info("TPU %s no longer exists in %s; treating as preempted", tpu_name, zone)
        return True
    return state == "PREEMPTED"


def _prepare_candidate(name: str, config: TPUConfigWithType, cancel: threading.Event) -> None:
    """Submit one race candidate, letting the caller drop zones that answer 'quota'."""
    if cancel.is_set():
        return
    try:
        create_tpu(config.tpu_type, config, tpu_name=name)
    except Exception as e:
        reason = create_failure_reason(e)
        kind = classify_create_failure(reason)
        logger.info("Candidate %s in %s failed (%s): %s", name, config.zone, kind, reason)
        # A queued resource already under this name in this zone is an earlier attempt for
        # the same pod still in flight. Racing it is the point, so keep the zone rather
        # than abandoning capacity that may be about to land.
        if kind == "exists":
            logger.info("Adopting in-flight queued resource %s in %s", name, config.zone)
            return
        raise


def _requeue_failed(name: str, config: TPUConfigWithType) -> bool:
    """Re-submit a candidate whose queued resource gave up. Returns whether it was re-submitted.

    A FAILED queued resource never becomes a node, so a race polling for that node waits out
    its entire timeout for capacity that was already refused.
    """
    queued = describe_queued_resource(name, config.zone, config.project)
    state = queued.get("state", {}).get("state") if queued else None
    # No resource at all is as dead as a FAILED one: the zone's create never landed, or a
    # previous re-submission deleted the old resource and could not replace it. Either way
    # nothing is in flight, so polling this candidate can only ever time out.
    if queued is not None and state != "FAILED":
        return False
    logger.info("Queued resource %s in %s is %s; re-submitting", name, config.zone, state or "absent")
    try:
        if queued is not None:
            delete_queued_resource(name, config.zone, config.project)
        create_tpu(config.tpu_type, config, tpu_name=name)
    except Exception as e:
        logger.info("Could not re-submit %s in %s: %s", name, config.zone, create_failure_reason(e))
        return False
    return True


def _cleanup_losers(
    candidates: dict[str, tuple[str, TPUConfigWithType]],
    *,
    keep: str | None,
) -> None:
    """Delete every candidate except the winning zone's."""
    for zone, (name, config) in candidates.items():
        if zone == keep:
            continue
        _delete_candidate(name, config)


def race_spot_tpu(
    tpu_type: str,
    *,
    user: str,
    project: str,
    timeout: float,
    poll_interval: float = 15,
) -> TPUAllocation:
    """Create in every quota-bearing zone at once; keep the first usable pod.

    Losing candidates are deleted as soon as a winner appears, and on any error or
    timeout every candidate is torn down so a failed race leaves nothing behind.
    """
    configs = spot_race_configs(tpu_type, user=user, project=project)
    prefix_of = {
        config.zone: get_tpu_name_prefix(tpu_type, resource_owner=config.resource_owner, is_spot=True)
        for config in configs
    }
    # Name each candidate from what is actually free in its own zone. Using the list
    # position instead would both collide with an existing pod and produce a meaningless
    # index (the 13th zone becoming "-12" rather than the next free number).
    #
    # A zone whose naming query fails is dropped rather than aborting the race: some zones
    # reject the queued-resources API outright, and one bad zone must not cost the other
    # eighteen.
    # Keyed by zone, not by name: TPU names are per-zone, so every empty zone yields the
    # same "<prefix>-0" and a name-keyed dict would silently collapse the whole race into
    # a single candidate.
    candidates: dict[str, tuple[str, TPUConfigWithType]] = {}
    for config in configs:
        try:
            candidates[config.zone] = (_next_free_name(prefix_of[config.zone], config), config)
        except Exception as e:
            logger.info("Dropping %s from the race: %s", config.zone, str(e)[:120])
    if not candidates:
        raise RuntimeError(f"No zone could be prepared for a {tpu_type} race")
    logger.info(
        "Racing %s across %d zones: %s",
        tpu_type,
        len(candidates),
        {z: n for z, (n, _) in candidates.items()},
    )

    cancel = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates))
    futures = {
        executor.submit(_prepare_candidate, name, config, cancel): (name, config)
        for name, config in candidates.values()
    }
    # Everything below is tracked by zone. Candidates share a name whenever their zones were
    # all empty, so a name-keyed set or dict silently collapses four candidates into one and
    # the loop only ever probes whichever survived.
    dropped: set[str] = set()
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            for future, (_, config) in futures.items():
                if not future.done() or config.zone in dropped:
                    continue
                error = future.exception()
                if error is None:
                    continue
                # Both verdicts are permanent for this race: more waiting cannot grant quota,
                # and a zone that does not offer the shape will never start offering it.
                kind = classify_create_failure(create_failure_reason(error))
                if kind in ("quota", "unsupported"):
                    logger.info("Dropping %s: %s, retrying will not clear it", config.zone, kind)
                    dropped.add(config.zone)

            live = [(name, config) for zone, (name, config) in candidates.items() if zone not in dropped]
            if not live:
                raise RuntimeError(f"No zone can supply {tpu_type}: every candidate is quota-capped or unsupported")

            for name, config in live:
                if not _is_tpu_usable(name, config):
                    _requeue_failed(name, config)
                    continue
                cancel.set()
                executor.shutdown(wait=False, cancel_futures=True)
                logger.info("Race winner: %s in %s", name, config.zone)
                _cleanup_losers(candidates, keep=config.zone)
                described = discovery.describe_pod(name, config.zone, project)
                return TPUAllocation(name=name, config=with_discovered_nfs(config, pod=described))
            time.sleep(poll_interval)
    except BaseException:
        cancel.set()
        executor.shutdown(wait=False, cancel_futures=True)
        _cleanup_losers(candidates, keep=None)
        raise

    cancel.set()
    executor.shutdown(wait=False, cancel_futures=True)
    _cleanup_losers(candidates, keep=None)
    raise RuntimeError(f"No spot TPU became usable within {timeout}s")


def allocate_spot_tpu(
    tpu_type: str,
    *,
    user: str,
    project: str,
    timeout: float,
) -> TPUAllocation:
    """Reuse an idle spot pod if one exists, otherwise race for capacity."""
    owner = get_tpu_user(user).resource_owner
    reused = find_available_tpu(
        tpu_type, project=project, user=user, resource_owner=owner, is_spot=True, reclaim_owned=True
    )
    if reused is not None:
        return reused

    # A preempted pod is not removed by GCP; it lingers holding its name and its share of
    # quota, and _next_free_name counts it, so indices climb every failed attempt. Clear
    # the family's carcasses before asking for new capacity.
    try:
        cleanup_preempted(tpu_type, project=project)
    except Exception as e:
        logger.warning("Could not clean up preempted %s resources: %s", tpu_type, e)
    return race_spot_tpu(tpu_type, user=user, project=project, timeout=timeout)
