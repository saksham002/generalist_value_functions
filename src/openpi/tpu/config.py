"""TPU configuration.

Almost nothing here is configuration. Zone, NFS, accelerator and runtime are facts about
a pod, so they are discovered (:mod:`openpi.tpu.discovery`) rather than declared, and the
zones a spot pod may be raced in come from live quota (:mod:`openpi.tpu.quota`).

Only three things must be declared, because they are inputs to *creating* a pod that does
not exist yet and so cannot be read off one:

- ``runtime_version`` per family — a pod built with the wrong runtime boots but cannot run
  the job, and fails much later looking unrelated;
- the accelerator string per family — v5e alone renames ``v5e-N`` to ``v5litepod-N``;
- ``host_ram_gb`` per family — the RSS guard's ceiling has to be per-family.

Resolution has two entry points, matching the two ways a pod comes to exist:

- :func:`resolve_from_pod` — the pod already exists (any reserved pod, or a spot pod that
  won a race). Everything is read off it. Raises if it does not exist.
- :func:`spot_race_configs` — the pod does not exist yet. Zones come from live quota; NFS
  is attached afterwards with :func:`with_discovered_nfs` once the winner is real.
"""

from collections.abc import Mapping
import dataclasses

from openpi.tpu import discovery
from openpi.tpu import quota

_SUPPORTED_TPU_FAMILIES = frozenset({"v4", "v5e", "v6e"})

# Single-user repo: existing call sites pass only a TPU type, so resolution needs a default.
DEFAULT_TPU_USER = "saksham"
DEFAULT_PROJECT = "cmu-aidm-v2"


@dataclasses.dataclass(frozen=True, kw_only=True)
class TPUFamilySpec:
    """The irreducible per-family facts needed to create and police a pod."""

    family: str
    runtime_version: str
    host_ram_gb: int
    accelerator_device_glob: str
    """Device nodes a running job holds; differs between v4/v6e and v5e."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class TPUUserConfig:
    """Storage and resource namespaces owned by one TPU user."""

    resource_owner: str
    nfs_directory: str
    remote_home: str
    gcs_buckets_by_region: Mapping[str, str]
    ssh_user: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class TPUConfigWithType:
    """A concrete TPU shape bound to a user, a zone, and whatever storage it has."""

    family: str
    zone: str
    project: str
    is_spot: bool
    runtime_version: str
    nfs_server: str | None
    nfs_mount_path: str | None
    user: str
    resource_owner: str
    nfs_directory: str
    remote_home: str
    gcs_bucket: str
    gcs_bucket_region: str
    ssh_user: str | None
    tpu_type: str
    accelerator_type: str

    @property
    def uses_nfs(self) -> bool:
        """Whether this pod has an NFS filesystem; False means local-disk mode."""
        return self.nfs_server is not None and self.nfs_mount_path is not None

    @property
    def gcloud_accelerator_type(self) -> str:
        """Accelerator string accepted by ``gcloud compute tpus``."""
        return self.accelerator_type

    @property
    def host_ram_gb(self) -> int:
        return TPU_FAMILY_SPECS[self.family].host_ram_gb

    @property
    def accelerator_device_glob(self) -> str:
        return TPU_FAMILY_SPECS[self.family].accelerator_device_glob


TPU_FAMILY_SPECS: dict[str, TPUFamilySpec] = {
    "v4": TPUFamilySpec(
        family="v4",
        runtime_version="tpu-ubuntu2204-base",
        host_ram_gb=400,
        accelerator_device_glob="/dev/accel*",
    ),
    "v5e": TPUFamilySpec(
        family="v5e",
        runtime_version="v2-alpha-tpuv5-lite",
        host_ram_gb=188,
        accelerator_device_glob="/dev/vfio/*",
    ),
    "v6e": TPUFamilySpec(
        family="v6e",
        runtime_version="v2-alpha-tpuv6e",
        host_ram_gb=708,
        accelerator_device_glob="/dev/accel*",
    ),
}


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


def _parse_tpu_type(tpu_type: str) -> tuple[str, int]:
    """Parse the family and numeric shape component of a TPU type."""
    family, separator, size_text = tpu_type.partition("-")
    if family not in _SUPPORTED_TPU_FAMILIES or not separator or not size_text.isdigit():
        expected = ", ".join(f"{name}-*" for name in sorted(_SUPPORTED_TPU_FAMILIES))
        raise ValueError(f"Unknown TPU type: {tpu_type}. Expected one of {expected}")
    return family, int(size_text)


def get_family_spec(family: str) -> TPUFamilySpec:
    try:
        return TPU_FAMILY_SPECS[family]
    except KeyError:
        raise ValueError(f"Unknown TPU family {family!r}. Known: {sorted(TPU_FAMILY_SPECS)}") from None


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
) -> TPUConfigWithType:
    family, _ = _parse_tpu_type(tpu_type)
    user_config = get_tpu_user(user)
    bucket_region = region_from_zone(zone)
    gcs_bucket = user_config.gcs_buckets_by_region.get(bucket_region) or canonical_bucket_for_region(
        bucket_region, resource_owner=user_config.resource_owner
    )
    return TPUConfigWithType(
        family=family,
        zone=zone,
        project=project,
        is_spot=is_spot,
        runtime_version=runtime_version,
        nfs_server=nfs_server,
        nfs_mount_path=nfs_mount_path,
        user=user,
        resource_owner=user_config.resource_owner,
        nfs_directory=user_config.nfs_directory,
        remote_home=user_config.remote_home,
        gcs_bucket=gcs_bucket,
        gcs_bucket_region=bucket_region,
        ssh_user=user_config.ssh_user,
        tpu_type=tpu_type,
        accelerator_type=accelerator_type_for(tpu_type),
    )


def resolve_from_pod(
    tpu_name: str,
    *,
    user: str = DEFAULT_TPU_USER,
    project: str = DEFAULT_PROJECT,
    tpu_type: str | None = None,
) -> TPUConfigWithType:
    """Resolve a config entirely from a pod that already exists.

    Zone comes from Cloud Asset Inventory, accelerator and runtime from ``describe``, and
    NFS from the region's Filestore instances disambiguated against this pod's own mount
    table. Nothing is assumed from the pod's name.

    Raises:
        ValueError: If no pod of that name exists in the project.
    """
    located = discovery.find_pod(tpu_name, project=project)
    if located is None:
        raise ValueError(
            f"TPU {tpu_name!r} does not exist in project {project!r}. "
            "Non-spot launches target an existing pod; create it first or use a spot launch."
        )

    described = discovery.describe_pod(tpu_name, located.zone, project)
    resolved_type = tpu_type or _tpu_type_from_accelerator(described.accelerator_type, tpu_name)
    nfs_server, nfs_mount_path = discovery.nfs_for_zone(
        described.zone, project, disambiguate_with=described, ssh_user=get_tpu_user(user).ssh_user
    )
    return _build_config(
        tpu_type=resolved_type,
        zone=described.zone,
        project=project,
        is_spot=False,
        runtime_version=described.runtime_version or get_family_spec(_parse_tpu_type(resolved_type)[0]).runtime_version,
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
) -> tuple[TPUConfigWithType, ...]:
    """Resolve one config per zone with live spot quota for ``tpu_type``.

    NFS is left unresolved: a pod that does not exist yet has no mount table, and probing
    every candidate zone would cost an ssh round trip per zone. Call
    :func:`with_discovered_nfs` on the winner instead.
    """
    family, size = _parse_tpu_type(tpu_type)
    spec = get_family_spec(family)
    # One Asset Inventory call establishes which zones actually hold TPUs today; that is
    # the evidence that distinguishes a usable zone from one the default quota merely
    # mentions.
    occupied = frozenset(pod.zone for pod in discovery.list_all_tpus(project) if pod.zone)
    zones = quota.spot_quota_zones(family, size, project=project, occupied_zones=occupied)
    return tuple(
        _build_config(
            tpu_type=tpu_type,
            zone=zone,
            project=project,
            is_spot=True,
            runtime_version=spec.runtime_version,
            nfs_server=None,
            nfs_mount_path=None,
            user=user,
        )
        for zone in zones
    )


def with_discovered_nfs(
    config: TPUConfigWithType,
    *,
    pod: discovery.DiscoveredPod | None = None,
) -> TPUConfigWithType:
    """Attach the NFS a now-existing pod should mount, or leave it in local-disk mode."""
    nfs_server, nfs_mount_path = discovery.nfs_for_zone(
        config.zone, config.project, disambiguate_with=pod, ssh_user=config.ssh_user
    )
    return dataclasses.replace(config, nfs_server=nfs_server, nfs_mount_path=nfs_mount_path)


def resolve_is_spot(*, config_is_spot: bool, spot_override: bool | None) -> bool:
    """Resolve a tri-state launch override against the deployment default."""
    return config_is_spot if spot_override is None else spot_override


def get_tpu_name_prefix(tpu_type: str, *, resource_owner: str, is_spot: bool) -> str:
    """Return the user- and capacity-scoped prefix used for TPU resource names."""
    family, size = _parse_tpu_type(tpu_type)
    capacity_suffix = "-spot" if is_spot else ""
    return f"{family}-{resource_owner}{capacity_suffix}-{size}"


def infer_tpu_type_from_name(tpu_name: str) -> str:
    """Recover a TPU type from a managed or historical resource name.

    Only a naming convenience: authoritative type comes from the pod's acceleratorType.
    """
    parts = tpu_name.split("-")
    if len(parts) in (2, 3) and parts[1].isdigit() and (len(parts) == 2 or parts[2].isdigit()):
        family, size = parts[:2]
        _parse_tpu_type(f"{family}-{size}")
        return f"{family}-{size}"

    if len(parts) < 3:
        raise ValueError(f"Could not infer TPU type from {tpu_name!r}")

    family, owner, *tail = parts
    if tail and tail[0] == "spot":
        tail = tail[1:]
    size = tail[0] if tail else ""
    index_tokens = tail[1:]
    valid_index = not index_tokens or (len(index_tokens) == 1 and index_tokens[0].isdigit())
    if not owner or not size.isdigit() or not valid_index:
        raise ValueError(f"Could not infer TPU type from {tpu_name!r}")

    tpu_type = f"{family}-{size}"
    _parse_tpu_type(tpu_type)
    return tpu_type


def get_tpu_type_prefix(tpu_type: str) -> str:
    """Extract the type prefix from a TPU type (e.g., 'v6e' from 'v6e-8')."""
    return tpu_type.split("-")[0]


def get_worker_count(tpu_type: str) -> int:
    """Return the number of TPU hosts represented by an accelerator shape.

    v5e and v6e shape numbers count chips, with four chips per host. v4 shape numbers
    count TensorCores, with eight TensorCores per host.
    """
    family, size = _parse_tpu_type(tpu_type)
    units_per_host = 8 if family == "v4" else 4
    return max(1, size // units_per_host)
