"""TPU configuration: what a pod IS, and where a job puts things on it.

Almost nothing here is configuration. Zone, NFS, accelerator and runtime are facts about
a pod, so they are discovered (:mod:`openpi.tpu.discovery`) rather than declared, and the
zones a spot pod may be raced in come from live quota (:mod:`openpi.tpu.quota`).

Only two things must be declared, because they are inputs to *creating* a pod that does
not exist yet and so cannot be read off one:

- ``runtime_version`` per family — a pod built with the wrong runtime boots but cannot run
  the job, and fails much later looking unrelated;
- the accelerator string per family — v5e alone renames ``v5e-N`` to ``v5litepod-N``.

Three dataclasses, each with one lifetime:

- :class:`TPUUserConfig` — policy owned by a person: storage namespaces, home directory.
- :class:`PodConfig` — facts about one pod, holding its user by composition rather than
  copying the user's fields in. A function that needs a zone no longer receives a bucket
  registry.
- :class:`RemoteLayout` — every path a job touches on the pod, derived once from whether
  that pod has a shared filesystem. This is the single answer to "where does X live",
  which used to be re-derived independently in four modules.
"""

from collections.abc import Mapping
import dataclasses

from openpi.tpu import discovery
from openpi.tpu import quota
from openpi.tpu.gcloud import DEFAULT_PROJECT

_SUPPORTED_TPU_FAMILIES = frozenset({"v4", "v5e", "v6e"})

# Single-user repo: existing call sites pass only a TPU type, so resolution needs a default.
# DEFAULT_PROJECT is re-exported from gcloud rather than restated, so the two cannot drift.
DEFAULT_TPU_USER = "saksham"
__all__ = ["DEFAULT_PROJECT"]

# The one per-family fact that cannot be discovered, because it is an input to creating a
# pod that does not exist yet. A live pod reports its own runtime and that reading wins.
FAMILY_RUNTIME_VERSIONS: dict[str, str] = {
    "v4": "tpu-ubuntu2204-base",
    "v5e": "v2-alpha-tpuv5-lite",
    "v6e": "v2-alpha-tpuv6e",
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class TPUUserConfig:
    """Storage and resource namespaces owned by one TPU user."""

    resource_owner: str
    nfs_directory: str
    remote_home: str
    gcs_buckets_by_region: Mapping[str, str]
    ssh_user: str | None = None

    @property
    def remote_login(self) -> str:
        """The login name the pod knows this user by, taken from their home directory."""
        return self.remote_home.rstrip("/").rsplit("/", 1)[-1]


TPU_USERS: dict[str, TPUUserConfig] = {
    "saksham": TPUUserConfig(
        resource_owner="saksham",
        nfs_directory="saksham3",
        remote_home="/home/saksham3",
        # Regions absent here resolve to their canonical name and are created on demand.
        gcs_buckets_by_region={
            "europe-west4": "saksham-euw4",
            "us-central2": "saksham-usc2",
        },
    ),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class RemoteLayout:
    """Every path a job uses on one pod, decided by whether that pod has a filer.

    A pod in a region with a Filestore shares one tree across workers; a pod without one
    keeps everything in each worker's own home. That single boolean used to be consulted
    separately in ``run_on_tpu`` (working dir, sync fan-out), ``code_sync`` (uv root),
    ``setup`` (which steps run) and ``job`` (the preamble), each with its own default for
    the NFS user. It is answered once, here.
    """

    uses_nfs: bool
    nfs_mount_path: str | None
    nfs_user: str
    worker_count: int

    @property
    def base(self) -> str:
        """Root the job's own trees hang off."""
        return f"{self.nfs_mount_path}/{self.nfs_user}" if self.uses_nfs else "~"

    @property
    def working_dir(self) -> str:
        return f"{self.base}/batch_value_learning"

    @property
    def gemma_dir(self) -> str:
        return f"{self.base}/helper/gemma"

    @property
    def uv_root(self) -> str:
        # "~" does not expand inside every context this is interpolated into, and the uv
        # root is referenced from shells that do not go through a login path.
        return f"{self.base}/uv" if self.uses_nfs else "$HOME/uv"

    @property
    def venv(self) -> str:
        return f"{self.uv_root}/vla"

    @property
    def shared_workers(self) -> str:
        """Worker spec for work that only needs doing once on a shared filesystem."""
        return "0" if self.uses_nfs else "all"

    @property
    def sync_workers(self) -> list[int] | None:
        """Workers rsync must run against; None means "worker 0 is enough"."""
        return None if self.uses_nfs else list(range(self.worker_count))

    # Per-worker local paths. Identical on both pod kinds: each is deliberately outside the
    # shared tree, because every worker needs its own copy or its own answer.
    paligemma_cache_dir = "~/.cache/openpi/vertex-model-garden-paligemma-us/paligemma"
    log_file = "~/tpu_job_output.log"
    exit_code_file = "~/tpu_job_exit_code"
    certificate_file = "~/tpu_run_id"

    def localize_path(self, path: str) -> str:
        """Rewrite an NFS path for a pod that has no NFS, or return it unchanged.

        A command written for a filer pod carries absolute paths under the mount point —
        a validation cache directory, say. Landing that command on a local-disk pod would
        have it write into a directory that does not exist and cannot be created. The
        NFS-relative tail is preserved and re-rooted at the worker's own home, so the same
        command means the same thing on both pod kinds.
        """
        if self.uses_nfs or not path.startswith(discovery.DEFAULT_NFS_MOUNT_PATH):
            return path
        tail = path[len(discovery.DEFAULT_NFS_MOUNT_PATH) :].lstrip("/")
        # Drop the leading <nfs_user> component: on a local-disk pod the home directory
        # already scopes the path to this user, so keeping it nests a redundant level.
        head, separator, rest = tail.partition("/")
        if separator and head == self.nfs_user:
            tail = rest
        return f"~/{tail}" if tail else "~"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PodConfig:
    """One TPU pod: its shape, where it lives, what storage it has, and whose it is."""

    tpu_type: str
    family: str
    accelerator_type: str
    zone: str
    project: str
    is_spot: bool
    runtime_version: str
    nfs_server: str | None
    nfs_mount_path: str | None
    user_key: str
    user: TPUUserConfig

    @property
    def uses_nfs(self) -> bool:
        """Whether this pod has an NFS filesystem; False means local-disk mode."""
        return self.nfs_server is not None and self.nfs_mount_path is not None

    @property
    def region(self) -> str:
        return region_from_zone(self.zone)

    @property
    def gcs_bucket(self) -> str:
        """The bucket this pod's writes belong in, whether or not it exists yet."""
        return self.user.gcs_buckets_by_region.get(self.region) or canonical_bucket_for_region(
            self.region, resource_owner=self.user.resource_owner
        )

    @property
    def worker_count(self) -> int:
        return get_worker_count(self.tpu_type)

    def layout(self, nfs_user: str | None = None) -> RemoteLayout:
        """Where a job's files live on this pod."""
        return RemoteLayout(
            uses_nfs=self.uses_nfs,
            nfs_mount_path=self.nfs_mount_path,
            nfs_user=nfs_user or self.user.nfs_directory,
            worker_count=self.worker_count,
        )


def _parse_tpu_type(tpu_type: str) -> tuple[str, int]:
    """Parse the family and numeric shape component of a TPU type."""
    family, separator, size_text = tpu_type.partition("-")
    if family not in _SUPPORTED_TPU_FAMILIES or not separator or not size_text.isdigit():
        expected = ", ".join(f"{name}-*" for name in sorted(_SUPPORTED_TPU_FAMILIES))
        raise ValueError(f"Unknown TPU type: {tpu_type}. Expected one of {expected}")
    return family, int(size_text)


def runtime_version_for(family: str) -> str:
    try:
        return FAMILY_RUNTIME_VERSIONS[family]
    except KeyError:
        raise ValueError(f"Unknown TPU family {family!r}. Known: {sorted(FAMILY_RUNTIME_VERSIONS)}") from None


def accelerator_type_for(tpu_type: str) -> str:
    """Accelerator string for a shape. v5e is the only family that renames."""
    family, size = _parse_tpu_type(tpu_type)
    return f"v5litepod-{size}" if family == "v5e" else tpu_type


def region_from_zone(zone: str) -> str:
    """Return the GCP region containing ``zone``."""
    region, separator, zone_suffix = zone.rpartition("-")
    if not separator or not region or len(zone_suffix) != 1 or not zone_suffix.isalpha():
        raise ValueError(f"Invalid zone {zone!r}; expected '<region>-<zone-letter>'")
    return region


def region_abbreviation(region: str) -> str:
    """Abbreviate a region the way existing bucket names do.

    europe-west4 -> euw4, us-central2 -> usc2, us-east1 -> use1.
    """
    locality, separator, direction = region.partition("-")
    if not separator or not direction:
        raise ValueError(f"Invalid region {region!r}; expected '<locality>-<direction><number>'")
    continent = "eu" if locality.startswith("europe") else locality
    digits = "".join(character for character in direction if character.isdigit())
    return f"{continent}{direction[0]}{digits}"


def canonical_bucket_for_region(region: str, *, resource_owner: str) -> str:
    """Bucket a user's writes belong in for ``region``, whether or not it exists yet."""
    return f"{resource_owner}-{region_abbreviation(region)}"


def get_tpu_user(user: str) -> TPUUserConfig:
    """Return a registered TPU user configuration."""
    try:
        return TPU_USERS[user]
    except KeyError:
        raise ValueError(f"Unknown TPU user: {user!r}. Known users: {sorted(TPU_USERS)}") from None


def _build_config(
    *,
    tpu_type: str,
    zone: str,
    project: str,
    is_spot: bool,
    runtime_version: str,
    nfs_server: str | None,
    nfs_mount_path: str | None,
    user: str,
) -> PodConfig:
    family, _ = _parse_tpu_type(tpu_type)
    return PodConfig(
        tpu_type=tpu_type,
        family=family,
        accelerator_type=accelerator_type_for(tpu_type),
        zone=zone,
        project=project,
        is_spot=is_spot,
        runtime_version=runtime_version,
        nfs_server=nfs_server,
        nfs_mount_path=nfs_mount_path,
        user_key=user,
        user=get_tpu_user(user),
    )


def resolve_from_pod(
    tpu_name: str,
    *,
    user: str = DEFAULT_TPU_USER,
    project: str = DEFAULT_PROJECT,
    tpu_type: str | None = None,
    zone: str | None = None,
) -> PodConfig:
    """Resolve a config entirely from a pod that already exists.

    Pass ``zone`` whenever it is known. Pod names are unique only WITHIN a zone, and the
    same name can exist in several (spot races reuse low indices per zone). When it is not
    known, :func:`discovery.find_pod` raises on an ambiguous name rather than silently
    picking whichever zone the inventory happened to list first.

    Zone comes from Cloud Asset Inventory, accelerator, runtime and spot-ness from
    ``describe``, and NFS from the region's Filestore instances disambiguated against this
    pod's own mount table. Nothing is assumed from the pod's name.

    Raises:
        ValueError: If no pod of that name exists in the project, or the name is ambiguous.
    """
    if zone is None:
        located = discovery.find_pod(tpu_name, project=project)
        if located is None:
            raise ValueError(
                f"TPU {tpu_name!r} does not exist in project {project!r}. "
                "Non-spot launches target an existing pod; create it first or use a spot launch."
            )
        zone = located.zone

    described = discovery.describe_pod(tpu_name, zone, project)
    resolved_type = tpu_type or _tpu_type_from_accelerator(described.accelerator_type, tpu_name)
    nfs_server, nfs_mount_path = discovery.nfs_for_zone(
        described.zone, project, disambiguate_with=described, ssh_user=get_tpu_user(user).ssh_user
    )
    return _build_config(
        tpu_type=resolved_type,
        zone=described.zone,
        project=project,
        # Read off the pod rather than assumed: a spot pod adopted by a reserved launch used
        # to carry is_spot=False and be relabelled afterwards by whichever caller noticed.
        is_spot=bool(described.spot),
        runtime_version=described.runtime_version or runtime_version_for(_parse_tpu_type(resolved_type)[0]),
        nfs_server=nfs_server,
        nfs_mount_path=nfs_mount_path,
        user=user,
    )


def _tpu_type_from_accelerator(accelerator_type: str | None, tpu_name: str) -> str:
    """Recover a TPU type from the accelerator a live pod reports."""
    if not accelerator_type:
        raise ValueError(f"TPU {tpu_name!r} reports no acceleratorType; cannot infer its type")
    if accelerator_type.startswith("v5litepod-"):
        return f"v5e-{accelerator_type.partition('-')[2]}"
    _parse_tpu_type(accelerator_type)
    return accelerator_type


def spot_race_configs(
    tpu_type: str,
    *,
    user: str = DEFAULT_TPU_USER,
    project: str = DEFAULT_PROJECT,
    region: str | None = None,
) -> tuple[PodConfig, ...]:
    """Resolve one config per zone with live spot quota for ``tpu_type``.

    NFS is left unresolved: a pod that does not exist yet has no mount table, and probing
    every candidate zone would cost an ssh round trip per zone. Call
    :func:`with_discovered_nfs` on the winner instead.
    """
    family, size = _parse_tpu_type(tpu_type)
    # One Asset Inventory call establishes which zones actually hold TPUs today; that is
    # the evidence that distinguishes a usable zone from one the default quota merely
    # mentions.
    occupied = frozenset(
        pod.zone
        for pod in discovery.list_all_tpus(project)
        if pod.zone and (region is None or pod.zone.startswith(region))
    )
    zones = quota.spot_quota_zones(family, size, project=project, occupied_zones=occupied, region=region)
    return tuple(
        _build_config(
            tpu_type=tpu_type,
            zone=zone,
            project=project,
            is_spot=True,
            runtime_version=runtime_version_for(family),
            nfs_server=None,
            nfs_mount_path=None,
            user=user,
        )
        for zone in zones
    )


def with_discovered_nfs(config: PodConfig, *, pod: discovery.DiscoveredPod | None = None) -> PodConfig:
    """Attach the NFS a now-existing pod should mount, or leave it in local-disk mode."""
    nfs_server, nfs_mount_path = discovery.nfs_for_zone(
        config.zone, config.project, disambiguate_with=pod, ssh_user=config.user.ssh_user
    )
    return dataclasses.replace(config, nfs_server=nfs_server, nfs_mount_path=nfs_mount_path)


def get_tpu_name_prefix(tpu_type: str, *, resource_owner: str, is_spot: bool) -> str:
    """Return the user- and capacity-scoped prefix used for TPU resource names."""
    family, size = _parse_tpu_type(tpu_type)
    capacity_suffix = "-spot" if is_spot else ""
    return f"{family}-{resource_owner}{capacity_suffix}-{size}"


def get_tpu_type_prefix(tpu_type: str) -> str:
    """Extract the type prefix from a TPU type (e.g., 'v6e' from 'v6e-8')."""
    return tpu_type.split("-")[0]


def get_worker_count(tpu_type: str) -> int:
    """Return the number of TPU hosts represented by an accelerator shape.

    v5e and v6e shape numbers count chips, with four chips per host, except that a v6e
    slice of up to eight chips (v6e-1/-4/-8) is a single host. v4 shape numbers count
    TensorCores, with eight TensorCores per host.
    """
    family, size = _parse_tpu_type(tpu_type)
    if family == "v6e" and size <= 8:
        return 1
    units_per_host = 8 if family == "v4" else 4
    return max(1, size // units_per_host)
