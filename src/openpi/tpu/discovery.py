"""Discover a pod's own attributes instead of declaring them.

Zone, accelerator, runtime and NFS are facts about a live pod, so nothing here is
configured per family. Cloud Asset Inventory answers "where does this pod live" for every
location in one call, `describe` answers "what is it", and an ssh probe answers "what
filesystem does it have".

The only properties that genuinely cannot be discovered are the ones required to *create*
a pod that does not exist yet: its runtime version and accelerator string. Those live in
:mod:`openpi.tpu.config`.
"""

import dataclasses
import json
import logging

from openpi.tpu.gcloud import run_gcloud

logger = logging.getLogger(__name__)

_ASSET_TIMEOUT_SECONDS = 240
_DESCRIBE_TIMEOUT_SECONDS = 180
_PROBE_TIMEOUT_SECONDS = 240

# Where every pod mounts its filer. Not discoverable before the first mount, and uniform
# across the project; a live pod's actual mount point overrides it via probe_nfs.
DEFAULT_NFS_MOUNT_PATH = "/nfs/aidm_nfs"


@dataclasses.dataclass(frozen=True)
class DiscoveredPod:
    """What a live pod reports about itself."""

    name: str
    zone: str
    state: str
    accelerator_type: str | None = None
    runtime_version: str | None = None
    # Whether GCP created this pod as preemptible. Reported only by describe; the asset
    # index does not carry it, so it is None for pods that were only listed.
    spot: bool | None = None
    nfs_server: str | None = None
    nfs_mount_path: str | None = None

    @property
    def uses_nfs(self) -> bool:
        return self.nfs_server is not None and self.nfs_mount_path is not None


def list_all_tpus(project: str) -> tuple[DiscoveredPod, ...]:
    """List every TPU node in the project across all locations in one call.

    Uses Cloud Asset Inventory because ``gcloud compute tpus tpu-vm list`` requires an
    explicit ``--zone`` and would otherwise force a scan of every candidate zone.
    """
    result = run_gcloud(
        [
            "asset",
            "search-all-resources",
            f"--scope=projects/{project}",
            "--asset-types=tpu.googleapis.com/Node",
            "--format=json",
        ],
        project=project,
        check=False,
        timeout=_ASSET_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list TPUs in {project!r}: {result.stderr.strip()}")

    pods: list[DiscoveredPod] = []
    for asset in json.loads(result.stdout or "[]"):
        full_name = asset.get("name", "")
        short_name = full_name.rpartition("/nodes/")[2]
        if not short_name:
            continue
        pods.append(
            DiscoveredPod(
                name=short_name,
                zone=asset.get("location", ""),
                state=asset.get("state", ""),
            )
        )
    return tuple(pods)


def find_pod(tpu_name: str, *, project: str) -> DiscoveredPod | None:
    """Locate one pod by name without being told its zone."""
    for pod in list_all_tpus(project):
        if pod.name == tpu_name:
            return pod
    return None


def describe_pod(tpu_name: str, zone: str, project: str) -> DiscoveredPod:
    """Read a pod's accelerator, runtime and state straight off the resource."""
    result = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            tpu_name,
            f"--zone={zone}",
            "--format=json",
        ],
        project=project,
        check=False,
        timeout=_DESCRIBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to describe TPU {tpu_name!r} in {zone}: {result.stderr.strip()}")

    payload = json.loads(result.stdout)
    # The resource name carries its own location: .../locations/<zone>/nodes/<name>
    reported_zone = payload.get("name", "").partition("/locations/")[2].partition("/")[0] or zone
    return DiscoveredPod(
        name=tpu_name,
        zone=reported_zone,
        state=payload.get("state", ""),
        accelerator_type=payload.get("acceleratorType"),
        runtime_version=payload.get("runtimeVersion"),
        spot=bool(payload.get("schedulingConfig", {}).get("spot", False)),
    )


@dataclasses.dataclass(frozen=True)
class Filestore:
    """A Filestore instance available to mount."""

    zone: str
    ip_address: str
    share: str
    state: str

    @property
    def nfs_server(self) -> str:
        return f"{self.ip_address}:/{self.share}"


def _region_of(zone: str) -> str:
    region, separator, suffix = zone.rpartition("-")
    if not separator or len(suffix) != 1 or not suffix.isalpha():
        raise ValueError(f"Invalid zone {zone!r}; expected '<region>-<zone-letter>'")
    return region


def list_filestores(project: str) -> tuple[Filestore, ...]:
    """Enumerate every Filestore instance in the project.

    This is what makes NFS availability dynamic: a filer added in a new region later is
    picked up with no code change.
    """
    result = run_gcloud(
        ["filestore", "instances", "list", "--format=json"],
        project=project,
        check=False,
        timeout=_DESCRIBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list Filestore instances in {project!r}: {result.stderr.strip()}")

    stores: list[Filestore] = []
    for instance in json.loads(result.stdout or "[]"):
        location = instance.get("name", "").partition("/locations/")[2].partition("/")[0]
        networks = instance.get("networks") or [{}]
        addresses = networks[0].get("ipAddresses") or [None]
        shares = instance.get("fileShares") or [{}]
        if not location or not addresses[0] or not shares[0].get("name"):
            continue
        stores.append(
            Filestore(
                zone=location,
                ip_address=addresses[0],
                share=shares[0]["name"],
                state=instance.get("state", ""),
            )
        )
    return tuple(stores)


def nfs_for_zone(
    zone: str,
    project: str,
    *,
    disambiguate_with: DiscoveredPod | None = None,
    ssh_user: str | None = None,
) -> tuple[str | None, str | None]:
    """Decide what NFS, if any, a pod in ``zone`` should mount.

    Filestore is reachable across zones within a region, so candidates are scoped by
    region rather than zone. Returns ``(None, None)`` for a region with no filer, which
    is the signal to run in local-disk mode.

    When a region holds more than one filer the choice is settled empirically, by reading
    the mount table of an already-running pod there, rather than guessed: mounting the
    wrong share yields a pod that boots but fails much later in a way that looks
    unrelated.
    """
    region = _region_of(zone)
    candidates = [
        store for store in list_filestores(project) if _region_of(store.zone) == region and store.state == "READY"
    ]
    if not candidates:
        logger.info("No Filestore in %s; pod in %s runs in local-disk mode", region, zone)
        return None, None
    if len(candidates) == 1:
        only = candidates[0]
        return only.nfs_server, DEFAULT_NFS_MOUNT_PATH

    # A freshly created pod has mounted nothing yet, so it can never answer this itself;
    # the answer has to come from a pod that was already set up in the region.
    for pod in _mount_witnesses(region, project, preferred=disambiguate_with):
        mounted_server, mounted_path = probe_nfs(pod.name, pod.zone, project, ssh_user=ssh_user)
        if mounted_server is not None:
            logger.info("Region %s has %d filers; %s mounts %s", region, len(candidates), pod.name, mounted_server)
            return mounted_server, mounted_path

    raise ValueError(
        f"Region {region} has {len(candidates)} Filestore instances "
        f"({', '.join(store.nfs_server for store in candidates)}) and no running pod to disambiguate. "
        "Pass disambiguate_with= a live pod in that region."
    )


def _mount_witnesses(region: str, project: str, *, preferred: DiscoveredPod | None) -> tuple[DiscoveredPod, ...]:
    """Pods in ``region`` worth asking which filer this region uses, best bet first."""
    witnesses = [preferred] if preferred is not None else []
    try:
        others = list_all_tpus(project)
    except RuntimeError as error:
        logger.warning("Could not list pods to disambiguate NFS in %s: %s", region, error)
        others = ()
    witnesses.extend(
        pod
        for pod in others
        if _region_of(pod.zone) == region and pod.state == "READY" and pod.name not in {w.name for w in witnesses}
    )
    return tuple(witnesses)


def probe_nfs(tpu_name: str, zone: str, project: str, *, ssh_user: str | None = None) -> tuple[str | None, str | None]:
    """Ask a live pod what NFS it has mounted, if any.

    Returns ``(nfs_server, mount_path)``, or ``(None, None)`` when the pod runs off local
    disk. Reads worker 0's mount table rather than assuming a per-region NFS server.
    """
    target = f"{ssh_user}@{tpu_name}" if ssh_user else tpu_name
    result = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            target,
            f"--zone={zone}",
            "--worker=0",
            "--command=mount -t nfs,nfs4 2>/dev/null | head -1",
        ],
        project=project,
        timeout=_PROBE_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("NFS probe failed on %s (%s); assuming local disk", tpu_name, result.stderr.strip()[:120])
        return None, None

    for line in result.stdout.splitlines():
        # "10.155.154.42:/europe on /nfs/aidm_nfs type nfs4 (rw,...)"
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "on" and ":" in parts[0]:
            logger.info("Discovered NFS on %s: %s at %s", tpu_name, parts[0], parts[2])
            return parts[0], parts[2]
    return None, None
