"""Acquire a pod: resume our own, reuse an idle one, or race spot capacity.

Nothing here derives a zone from a TPU type. A pod either already exists — in which case
Cloud Asset Inventory finds it wherever it lives — or it is being created, in which case
the zone is chosen by the race and travels with the resulting :class:`Acquisition`.

Every knob that scopes an acquisition travels in one :class:`AllocationRequest` rather than
as optional keywords threaded through seven signatures, which is how a region filter came
to be honoured everywhere except preempted-resource cleanup.
"""

import concurrent.futures
import dataclasses
import hashlib
import logging
import threading
import time

from openpi.tpu import certificate
from openpi.tpu import discovery
from openpi.tpu.certificate import PodState
from openpi.tpu.config import DEFAULT_PROJECT
from openpi.tpu.config import DEFAULT_TPU_USER
from openpi.tpu.config import PodConfig
from openpi.tpu.config import accelerator_type_for
from openpi.tpu.config import get_tpu_name_prefix
from openpi.tpu.config import get_tpu_type_prefix
from openpi.tpu.config import get_tpu_user
from openpi.tpu.config import resolve_from_pod
from openpi.tpu.config import spot_race_configs
from openpi.tpu.config import with_discovered_nfs
from openpi.tpu.gcloud import _TERMINAL_TPU_STATES
from openpi.tpu.gcloud import TPU_STATE_UNKNOWN
from openpi.tpu.gcloud import create_queued_resource
from openpi.tpu.gcloud import delete_queued_resource
from openpi.tpu.gcloud import delete_tpu_vm
from openpi.tpu.gcloud import describe_queued_resource
from openpi.tpu.gcloud import get_tpu_state
from openpi.tpu.gcloud import get_tpu_state_and_health
from openpi.tpu.gcloud import list_queued_resources
from openpi.tpu.gcloud import list_tpus
from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)

# A zone that answers "no capacity" clears on its own and is worth retrying; a zone that
# answers "quota exceeded" clears when someone's pods are deleted, so it stays in the race.
_STOCKOUT_MARKERS = ("no more capacity", "code 8", "resource_exhausted")
_QUOTA_MARKERS = ("quota limit", "code 429", "has been exceeded", "exhausted. limit")
_EXISTS_MARKERS = ("already exists", "code 6", "alreadyexists")
# Having quota for a shape in a zone does not mean the zone offers it, so this is only
# discoverable by asking. It never clears, which makes the zone worthless for this race.
_UNSUPPORTED_MARKERS = ("zonal accelerator configurations", "was not found in zone")

# Shapes a spot launch may take when the caller does not care which. Racing them
# concurrently and keeping the first usable pod is the point: capacity for any one of them
# is scarce and bursty, so widening the request multiplies the odds.
DEFAULT_SPOT_TPU_TYPES = ("v4-64", "v5e-64", "v6e-32")

_RACE_POLL_SECONDS = 15


@dataclasses.dataclass(frozen=True, kw_only=True)
class AllocationRequest:
    """Everything that scopes which pod an acquisition may return."""

    run_id: str
    tpu_type: str | None = None
    tpu_name: str | None = None
    user: str = DEFAULT_TPU_USER
    project: str = DEFAULT_PROJECT
    spot: bool = False
    region: str | None = None
    only_my_pods: bool = False
    race_timeout: float = 21600
    max_race_zones: int | None = None

    @property
    def shapes(self) -> tuple[str, ...]:
        """Shapes this request will accept, widest-first for an untyped spot launch.

        Comma-separated so a launch can name the shapes it will take without having to
        accept every default. Capacity for any one shape is scarce and bursty, and the
        zones that can serve them barely overlap, so being able to say "v5e-64 or v6e-32"
        is the difference between racing two zones and racing six.
        """
        if not self.tpu_type:
            return DEFAULT_SPOT_TPU_TYPES
        return tuple(shape.strip() for shape in self.tpu_type.split(",") if shape.strip())

    @property
    def single_shape(self) -> str | None:
        """The one shape this request names, or None when it names none or several.

        Reserved and named acquisitions look a pod up *by* its shape, so they need exactly
        one; only the spot race can hold several in flight at once.
        """
        shapes = self.shapes
        return shapes[0] if self.tpu_type and len(shapes) == 1 else None

    @property
    def resource_owner(self) -> str:
        return get_tpu_user(self.user).resource_owner

    def in_region(self, zone: str | None) -> bool:
        return self.region is None or bool(zone and zone.startswith(self.region))

    @property
    def name_token(self) -> str:
        """The per-run component of a raced pod's name; see :func:`get_tpu_name_prefix`.

        The run id's trailing digest is used rather than its label: it is already the part
        that identifies the run, it is eight characters, and it is stable across a re-race
        into another region — so a relaunched run counts indices in the same namespace its
        earlier attempt did instead of opening a second one.
        """
        token = self.run_id.rsplit("-", 1)[-1][:8]
        return token if token.isalnum() else hashlib.sha256(self.run_id.encode()).hexdigest()[:8]


class RaceArbiter:
    """Decides which racing shape is allowed to keep its pod.

    ``cancelled`` alone is not enough. Setting an Event is not a test-and-set, so two shapes
    that find a usable pod in the same poll round would both return one and the loser's pod
    would simply be dropped on the floor, still running and still billing. Winning is
    therefore a single guarded transition: exactly one caller is told to keep its pod, and
    every other race learns it lost in time to tear its own candidates down.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._won = False
        self.cancelled = threading.Event()

    def win(self) -> bool:
        """Claim the race for this shape. True only for the first caller."""
        with self._lock:
            if self._won:
                return False
            self._won = True
            self.cancelled.set()
            return True

    def cancel(self) -> None:
        self.cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()


@dataclasses.dataclass(frozen=True)
class Acquisition:
    """A pod, the config describing where it landed, and how we came to hold it."""

    name: str
    config: PodConfig
    resumed: bool = False
    """True when this pod already carried our certificate: our own job, still in flight.
    The caller must attach to it rather than set it up and start a second copy."""


def _get_short_name(full_name: str) -> str:
    """ "projects/p/locations/z/nodes/v6e-0" -> "v6e-0"."""
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


def _is_tpu_usable(tpu_name: str, config: PodConfig) -> bool:
    """Whether a pod is ready to be handed a job.

    State and health from the API only. An ssh probe was tried and removed: it is what a
    launcher hangs on when a pod is deleted underneath it, and READY/HEALTHY is already the
    condition that matters.
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


def running_process_owners(tpu_name: str, config: PodConfig) -> set[str] | None:
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


def reclaim_tpu(tpu_name: str, config: PodConfig) -> None:
    """Clear leftover state from an owned pod so it can be reused.

    Kills stray training processes and removes the libtpu lockfile, which JAX leaves behind
    when a job dies without cleanup and which otherwise blocks the next run.
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


def is_tpu_preempted(tpu_name: str, zone: str, project: str) -> bool:
    """Whether a TPU is gone as far as a running job is concerned.

    A pod that has been deleted outright reports no state at all rather than PREEMPTED —
    GCP removes the resource, it does not leave a tombstone. The same applies to the states
    a pod passes through on its way out: a preemption is observed as DELETING for as long as
    the teardown takes.

    What does *not* count is a failure to ask. ``TPU_STATE_UNKNOWN`` means the control plane
    did not answer — expired credentials, a network blip — which says nothing about the pod.
    Reading that as "gone" declared preemptions against healthy pods and ended runs that
    were still training.
    """
    state = get_tpu_state(tpu_name, zone, project)
    if state == TPU_STATE_UNKNOWN:
        logger.warning(
            "TPU %s in %s: state unknown (control plane unreachable); not treating as preempted", tpu_name, zone
        )
        return False
    if state in _TERMINAL_TPU_STATES:
        logger.info("TPU %s in %s is %s; treating as preempted", tpu_name, zone, state or "gone")
        return True
    return False


# ---------------------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------------------


def acquire(request: AllocationRequest) -> Acquisition:
    """Return a pod this run may use, resuming its own work if any is still in flight.

    The order is deliberate and is the whole occupancy policy:

    1. **resume** — a pod anywhere carrying this run's certificate is our own job, still
       running. Attaching costs nothing and prevents a second copy of the same run writing
       into the same checkpoint directory;
    2. **named** — an explicit ``--tpu-name`` must be free, and refuses rather than waits;
    3. **reuse** — an idle pod of an acceptable shape, claimed atomically;
    4. **race** — spot capacity, only when the caller asked for it.
    """
    resumed = _find_our_pod(request)
    if resumed is not None:
        return resumed

    if request.tpu_name:
        return _acquire_named(request)

    if request.spot:
        return _acquire_spot(request)

    if not request.tpu_type:
        raise ValueError(
            "--tpu-type is required for a reserved (non-spot) launch: a reserved pod is looked "
            f"up by shape. Pass --tpu-type, or use --spot to race any of {DEFAULT_SPOT_TPU_TYPES}."
        )
    if request.single_shape is None:
        raise ValueError(
            f"A reserved launch takes exactly one --tpu-type, not {list(request.shapes)}: it looks a pod "
            "up by shape rather than creating one. Several shapes only make sense with --spot."
        )
    logger.info("Looking for an available %s TPU...", request.single_shape)
    reused = _find_idle_pod(request.single_shape, request)
    if reused is None:
        raise RuntimeError(
            f"No idle {request.tpu_type} TPU exists. Reserved launches target an existing pod; "
            "pass --tpu-name, create one, or use --spot to race for capacity."
        )
    return reused


def _candidate_pods(request: AllocationRequest) -> list[discovery.DiscoveredPod]:
    """READY pods in scope, this user's own first.

    Borrowing a colleague's idle pod is allowed, but taking one while yours sits free
    needlessly puts your work on a pod they may want back. The region filter is applied to
    the raw inventory, before anything is described or probed, so a pod elsewhere is never
    even looked at.
    """
    pods = [
        pod for pod in discovery.list_all_tpus(request.project) if pod.state == "READY" and request.in_region(pod.zone)
    ]
    return sorted(pods, key=lambda pod: 0 if request.resource_owner in pod.name else 1)


def _find_our_pod(request: AllocationRequest) -> Acquisition | None:
    """A pod already running THIS run, if any.

    Answers the question a restarted launcher has to ask before allocating: is my own job
    still in flight somewhere? Deliberately not filtered by shape — a launcher that was
    requeued after an untyped spot race has no idea which shape won.
    """
    for pod in _candidate_pods(request):
        if request.resource_owner not in pod.name:
            # Only this user's pods can carry this user's certificate, and probing every
            # pod in the project would cost an ssh round trip each.
            continue
        if certificate.read(pod.name, pod.zone, request.project) != request.run_id:
            continue
        try:
            config = resolve_from_pod(pod.name, user=request.user, project=request.project, zone=pod.zone)
        except Exception as e:
            logger.info("Could not resolve %s while resuming: %s", pod.name, str(e)[:120])
            continue
        logger.info(
            "Run %r is already in flight on %s (%s); attaching instead of allocating",
            request.run_id,
            pod.name,
            pod.zone,
        )
        return Acquisition(name=pod.name, config=config, resumed=True)
    return None


def _acquire_named(request: AllocationRequest) -> Acquisition:
    """Use the pod the caller named, or refuse.

    Naming a pod says which one to use, not that it is free. Pods are shared, and one that
    was idle when the launch was prepared may have been claimed since; a second job on the
    same TPU makes one of the two fail on the accelerator lock.
    """
    logger.info("Using specified TPU: %s", request.tpu_name)
    config = resolve_from_pod(
        request.tpu_name, user=request.user, project=request.project, tpu_type=request.single_shape
    )
    probe = certificate.probe_and_claim(request.tpu_name, config.zone, request.project, request.run_id)
    if probe.state is PodState.UNREACHABLE:
        raise RuntimeError(
            f"Cannot tell what is running on {request.tpu_name}; refusing to start a job on it. "
            "Check the pod by hand, or omit --tpu-name to pick an idle pod automatically."
        )
    if probe.state is PodState.BUSY:
        held = f"run {probe.certificate!r}" if probe.certificate else f"{probe.python_processes} python process(es)"
        raise RuntimeError(
            f"{request.tpu_name} is already busy ({held}). Wait for it to free up, or omit "
            "--tpu-name to pick an idle pod automatically."
        )
    return Acquisition(name=request.tpu_name, config=config, resumed=probe.state is PodState.OURS)


def _acquire_spot(request: AllocationRequest) -> Acquisition:
    """Reuse an idle spot pod if one exists, otherwise race for capacity."""
    for shape in request.shapes:
        reused = _find_idle_pod(shape, request, reclaim_owned=True, require_spot=True)
        if reused is not None:
            return reused

    # A preempted pod is not removed by GCP; it lingers holding its name and its share of
    # quota, and _next_free_name counts it, so indices climb every failed attempt. Clear
    # each shape's carcasses before asking for new capacity.
    for shape in request.shapes:
        try:
            cleanup_preempted(shape, project=request.project, region=request.region)
        except Exception as e:
            logger.warning("Could not clean up preempted %s resources: %s", shape, e)

    logger.info("Allocating spot %s across every zone with quota...", request.tpu_type or f"any of {request.shapes}")
    return race_spot_tpu_any(request)


def may_reclaim(probe: certificate.PodProbe, *, owned: bool, reclaim_owned: bool) -> bool:
    """Whether a busy pod may have its processes killed and be taken over.

    The certificate check is the important one. Reclaiming exists for *stale processes* left
    on a pod nobody holds. A pod carrying a certificate is held by a live launcher, and for
    the several minutes between a claim and the job actually starting there is no python
    running on it at all -- so judging by processes alone reads a pod mid-setup as idle and
    takes it from the run installing onto it, deleting that run's certificate and writing
    another. Two launchers then set up on one pod, which is the exact outcome the
    certificate exists to prevent.
    """
    if not reclaim_owned or not owned:
        return False
    if probe.state is certificate.PodState.UNREACHABLE:
        return False
    return probe.certificate is None


def _find_idle_pod(
    tpu_type: str,
    request: AllocationRequest,
    *,
    reclaim_owned: bool = False,
    require_spot: bool = False,
) -> Acquisition | None:
    """Find and claim an existing READY, idle pod of this shape anywhere in the project.

    Any idle pod of the right shape is reused regardless of whose name it carries, unless
    ``only_my_pods`` is set. ``reclaim_owned`` additionally takes back a *busy* pod, but only
    one whose name carries this user's resource owner and whose processes are not theirs —
    killing another user's work is never acceptable, while borrowing a pod they are not
    using is.
    """
    wanted_accelerator = accelerator_type_for(tpu_type)
    type_prefix = get_tpu_type_prefix(tpu_type)

    for pod in _candidate_pods(request):
        if not pod.name.startswith(type_prefix):
            continue
        # Ownership does not gate *use*: an idle pod of the right shape is usable by anyone.
        # It gates only whether we may kill what is running on it.
        owned = request.resource_owner in pod.name
        if request.only_my_pods and not owned:
            logger.info("Skipping %s in %s: not this user's pod (--only-my-pods)", pod.name, pod.zone)
            continue
        try:
            described = discovery.describe_pod(pod.name, pod.zone, request.project)
            if described.accelerator_type != wanted_accelerator:
                continue
            config = resolve_from_pod(
                pod.name, user=request.user, project=request.project, tpu_type=tpu_type, zone=pod.zone
            )
        except Exception as e:
            # Almost always a pod deleted since the index was built; skip and keep looking.
            logger.info("Skipping %s: %s", pod.name, str(e)[:120])
            continue
        if require_spot and not config.is_spot:
            # Reusing a RESERVED pod for a spot launch silently defeats the request: the
            # caller asked for preemptible capacity, and relabelling a reserved pod's config
            # does not make it one.
            logger.info("Skipping %s in %s: not a spot pod", pod.name, pod.zone)
            continue
        # Same bar as the race: a pod that is not READY/HEALTHY is adopted only to fail later
        # during sync. Asset Inventory's state lags, so this re-checks it live.
        if not _is_tpu_usable(pod.name, config):
            logger.info("Skipping %s in %s: not READY/HEALTHY", pod.name, config.zone)
            continue

        probe = certificate.probe_and_claim(pod.name, config.zone, request.project, request.run_id)
        if probe.state is PodState.CLAIMED:
            logger.info("Claimed idle TPU %s in %s", pod.name, config.zone)
            return Acquisition(name=pod.name, config=config)
        if probe.state is PodState.OURS:
            logger.info("TPU %s already carries this run's certificate; attaching", pod.name)
            return Acquisition(name=pod.name, config=config, resumed=True)
        if not may_reclaim(probe, owned=owned, reclaim_owned=reclaim_owned):
            if probe.certificate is not None:
                logger.info("%s carries run %r's certificate; leaving it alone", pod.name, probe.certificate)
            continue

        # Never kill this user's own work. Anything else on their pod is stale or foreign
        # and may be cleared. Undetermined ownership counts as theirs.
        owners = running_process_owners(pod.name, config)
        if owners is None:
            logger.info("%s is busy and process ownership is unknown; leaving it alone", pod.name)
            continue
        if config.user.remote_login in owners:
            logger.info("%s is running %s's own processes; leaving it alone", pod.name, config.user.remote_login)
            continue
        logger.info("%s is busy with foreign/stale processes %s", pod.name, sorted(owners))
        reclaim_tpu(pod.name, config)
        # The reclaim killed whatever was running; the certificate it left behind, if any, is
        # stale by definition, so clear it before trying to take the pod.
        certificate.release(pod.name, config.zone, request.project)
        retry = certificate.probe_and_claim(pod.name, config.zone, request.project, request.run_id)
        if retry.state is PodState.CLAIMED:
            logger.info("Reclaimed TPU %s in %s", pod.name, config.zone)
            return Acquisition(name=pod.name, config=config)
        logger.warning("%s still busy after reclaim; leaving it alone", pod.name)
    return None


# ---------------------------------------------------------------------------------------
# Creation and the spot race
# ---------------------------------------------------------------------------------------


def _next_free_name(prefix: str, config: PodConfig) -> str:
    """Name the next pod ``<prefix>-<highest existing index + 1>`` in one zone.

    Deliberately not the lowest unused index: reusing a gap left by a deleted pod makes
    names ambiguous across a pod's lifetime, so the counter only ever moves forward. Both
    queued resources and VMs are counted, since a queued resource holds the name before any
    VM exists.
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


def create_tpu(config: PodConfig, *, tpu_name: str) -> str:
    """Create one pod in ``config.zone``. Returns the name."""
    logger.info("Creating TPU %s (%s) in %s", tpu_name, config.accelerator_type, config.zone)
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


def _delete_candidate(tpu_name: str, config: PodConfig) -> None:
    """Delete one candidate's VM and its queued resource, tolerating absence."""
    for delete, label in ((delete_tpu_vm, "VM"), (delete_queued_resource, "queued resource")):
        try:
            delete(tpu_name, config.zone, config.project)
        except Exception as e:
            logger.debug("Deleting %s %s in %s: %s", label, tpu_name, config.zone, e)


def cleanup_preempted(tpu_type: str, *, project: str, region: str | None = None) -> None:
    """Delete preempted VMs and suspended queued resources for a shape, in every zone."""
    type_prefix = get_tpu_type_prefix(tpu_type)
    wanted_accelerator = accelerator_type_for(tpu_type)

    zones_seen: set[str] = set()
    for pod in discovery.list_all_tpus(project):
        if not pod.name.startswith(type_prefix):
            continue
        if region is not None and not (pod.zone or "").startswith(region):
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


def _prepare_candidate(name: str, config: PodConfig, arbiter: RaceArbiter) -> None:
    """Submit one race candidate, letting the caller drop zones that answer 'unsupported'."""
    if arbiter.is_cancelled:
        return
    try:
        create_tpu(config, tpu_name=name)
    except Exception as e:
        reason = create_failure_reason(e)
        kind = classify_create_failure(reason)
        logger.info("Candidate %s in %s failed (%s): %s", name, config.zone, kind, reason)
        # A queued resource already under this name in this zone is an earlier attempt for
        # the same pod still in flight. Racing it is the point.
        if kind == "exists":
            logger.info("Adopting in-flight queued resource %s in %s", name, config.zone)
            return
        raise


def _requeue_failed(name: str, config: PodConfig) -> bool:
    """Re-submit a candidate whose queued resource gave up.

    A FAILED queued resource never becomes a node, so a race polling for that node waits out
    its entire timeout for capacity that was already refused. No resource at all is as dead
    as a FAILED one: nothing is in flight, so polling can only ever time out.
    """
    queued = describe_queued_resource(name, config.zone, config.project)
    state = queued.get("state", {}).get("state") if queued else None
    if queued is not None and state != "FAILED":
        return False
    logger.info("Queued resource %s in %s is %s; re-submitting", name, config.zone, state or "absent")
    try:
        if queued is not None:
            delete_queued_resource(name, config.zone, config.project)
        create_tpu(config, tpu_name=name)
    except Exception as e:
        logger.info("Could not re-submit %s in %s: %s", name, config.zone, create_failure_reason(e))
        return False
    return True


def _cleanup_losers(candidates: dict[str, tuple[str, PodConfig]], *, keep: str | None) -> None:
    """Delete every candidate except the winning zone's."""
    for zone, (name, config) in candidates.items():
        if zone == keep:
            continue
        _delete_candidate(name, config)


def race_spot_tpu(
    tpu_type: str,
    request: AllocationRequest,
    *,
    arbiter: RaceArbiter | None = None,
) -> Acquisition:
    """Create in every quota-bearing zone at once; keep the first usable pod.

    Losing candidates are deleted as soon as a winner appears, and on any error, timeout or
    external cancellation every candidate is torn down so a failed race leaves nothing
    behind. The ``arbiter`` is what lets a multi-shape race stop the shapes that did not win
    instead of leaving them holding queued resources until their own timeout, and what
    guarantees exactly one of them keeps a pod.
    """
    arbiter = arbiter or RaceArbiter()
    configs = spot_race_configs(
        tpu_type,
        user=request.user,
        project=request.project,
        region=request.region,
        max_zones=request.max_race_zones,
    )
    prefix_of = {
        config.zone: get_tpu_name_prefix(
            tpu_type, resource_owner=config.user.resource_owner, is_spot=True, run_token=request.name_token
        )
        for config in configs
    }
    # Name each candidate from what is actually free in its own zone. Using the list
    # position instead would both collide with an existing pod and produce a meaningless
    # index. Keyed by zone, not by name: TPU names are per-zone, so every empty zone yields
    # the same "<prefix>-0" and a name-keyed dict would collapse the race into one candidate.
    #
    # A zone whose naming query fails is dropped rather than aborting the race: some zones
    # reject the queued-resources API outright, and one bad zone must not cost the others.
    candidates: dict[str, tuple[str, PodConfig]] = {}
    for config in configs:
        try:
            candidates[config.zone] = (_next_free_name(prefix_of[config.zone], config), config)
        except Exception as e:
            logger.info("Dropping %s from the race: %s", config.zone, str(e)[:120])
    if not candidates:
        raise RuntimeError(f"No zone could be prepared for a {tpu_type} race")
    logger.info("Racing %s across %d zones: %s", tpu_type, len(candidates), {z: n for z, (n, _) in candidates.items()})

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates))
    futures = {
        executor.submit(_prepare_candidate, name, config, arbiter): (name, config)
        for name, config in candidates.values()
    }
    dropped: set[str] = set()
    deadline = time.time() + request.race_timeout
    try:
        while time.time() < deadline:
            if arbiter.is_cancelled:
                raise RuntimeError(f"{tpu_type} race cancelled: another shape won")

            for future, (_, config) in futures.items():
                if not future.done() or config.zone in dropped:
                    continue
                error = future.exception()
                if error is None:
                    continue
                kind = classify_create_failure(create_failure_reason(error))
                # Only a zone that cannot offer the shape at all is permanently out: no
                # amount of waiting makes a v6e-32 appear where the accelerator is not
                # available. Quota is different — it is exceeded because someone's pods
                # currently hold it, and it frees the moment those are deleted.
                if kind == "unsupported":
                    logger.info("Dropping %s: %s, retrying will not clear it", config.zone, kind)
                    dropped.add(config.zone)
                elif kind == "quota":
                    logger.info("%s is quota-capped for now; keeping it in the race", config.zone)

            live = [(name, config) for zone, (name, config) in candidates.items() if zone not in dropped]
            if not live:
                raise RuntimeError(f"No zone can supply {tpu_type}: every candidate is unsupported")

            for name, config in live:
                if not _is_tpu_usable(name, config):
                    _requeue_failed(name, config)
                    continue
                if not arbiter.win():
                    # Another shape got there first in this same round. Fall through to the
                    # teardown below so this pod is deleted rather than left running unowned.
                    raise RuntimeError(f"{tpu_type} race lost: another shape claimed the win")
                executor.shutdown(wait=False, cancel_futures=True)
                logger.info("Race winner: %s in %s", name, config.zone)
                _cleanup_losers(candidates, keep=config.zone)
                described = discovery.describe_pod(name, config.zone, request.project)
                return Acquisition(name=name, config=with_discovered_nfs(config, pod=described))
            time.sleep(_RACE_POLL_SECONDS)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        _cleanup_losers(candidates, keep=None)
        raise

    executor.shutdown(wait=False, cancel_futures=True)
    _cleanup_losers(candidates, keep=None)
    raise RuntimeError(f"No spot TPU became usable within {request.race_timeout}s")


def race_spot_tpu_any(request: AllocationRequest) -> Acquisition:
    """Race several TPU shapes at once and keep the first usable pod of any of them.

    One independent single-type race per shape, each keeping its own zone-keyed candidates,
    its own dropped/quota bookkeeping and its own loser cleanup. A shared cancel event is
    what makes losing shapes stop promptly: they see it on their next poll and tear their
    candidates down, instead of holding queued resources until their own timeout expires.
    """
    shapes = request.shapes
    arbiter = RaceArbiter()
    if len(shapes) == 1:
        return race_spot_tpu(shapes[0], request, arbiter=arbiter)

    logger.info("Racing %d TPU shapes concurrently: %s", len(shapes), shapes)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(shapes))
    futures = {executor.submit(race_spot_tpu, shape, request, arbiter=arbiter): shape for shape in shapes}
    errors: dict[str, str] = {}
    try:
        for future in concurrent.futures.as_completed(futures):
            shape = futures[future]
            error = future.exception()
            if error is None:
                acquisition = future.result()
                logger.info("Multi-shape race winner: %s (%s in %s)", shape, acquisition.name, acquisition.config.zone)
                return acquisition
            errors[shape] = str(error)[:200]
            logger.info("Shape %s dropped out of the multi-shape race: %s", shape, errors[shape])
    finally:
        # Every losing race observes this and tears its own candidates down; waiting for them
        # to finish doing so keeps a launch from returning while pods it created still exist.
        arbiter.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    raise RuntimeError(f"No spot TPU of any shape {shapes} became usable: {errors}")
