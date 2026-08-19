"""Live spot-quota discovery via the Cloud Quotas API.

Zones are not declared anywhere in this repo. The set of zones a spot pod may be raced
in is derived per launch from the project's actual quota, so a quota grant or revocation
takes effect without a code change.

The API returns three kinds of entry per quota metric:

- a per-zone entry carrying ``details.value`` — an explicit override for that zone;
- a per-zone entry with empty ``details`` — that zone is explicitly zeroed;
- one entry with no ``dimensions`` — the default, applying to every zone listed in its
  ``applicableLocations``.

Limits only: the API does not report usage, so a zone reported as having quota may still
be full. A create returning ``code 429`` is the authoritative signal and drops the zone
for the remainder of a race.
"""

import functools
import json
import logging

from openpi.tpu.gcloud import run_gcloud

logger = logging.getLogger(__name__)

# The one mapping that cannot be discovered: which quota metric governs each family.
SPOT_QUOTA_IDS: dict[str, str] = {
    "v4": "TPUV4sPreemptiblePodPerProjectPerZoneForTPUAPI",
    "v5e": "TPUV5sPreemptibleLitepodPerProjectPerZoneForTPUAPI",
    "v6e": "TPUV6EPreemptiblePerProjectPerZoneForTPUAPI",
}

# Pods are only ever placed on these continents, so cross-continent egress never applies.
ALLOWED_CONTINENT_PREFIXES: tuple[str, ...] = ("us-", "europe-")

_QUOTA_QUERY_TIMEOUT_SECONDS = 240


def _continent_allowed(zone: str) -> bool:
    return zone.startswith(ALLOWED_CONTINENT_PREFIXES)


def _parse_quota_payload(payload: dict) -> dict[str, int]:
    """Flatten one quotaInfo document into ``{zone: limit}``."""
    limits: dict[str, int] = {}
    default_value: int | None = None
    default_locations: list[str] = []

    for entry in payload.get("dimensionsInfos", []):
        dimensions = entry.get("dimensions") or {}
        raw_value = (entry.get("details") or {}).get("value")
        if not dimensions:
            default_value = int(raw_value) if raw_value is not None else 0
            default_locations = entry.get("applicableLocations", [])
            continue
        zone = dimensions.get("zone")
        if zone is None:
            continue
        # Empty details on a zone-scoped entry means the zone is zeroed, not defaulted.
        limits[zone] = int(raw_value) if raw_value is not None else 0

    if default_value is not None:
        for zone in default_locations:
            limits.setdefault(zone, default_value)
    return limits


@functools.cache
def _fetch_quota_overrides(family: str, project: str) -> frozenset[str]:
    """Zones carrying an explicit per-zone grant, as opposed to the project default.

    An override is a deliberate act by whoever provisioned the project; the default grant
    is applied to dozens of zones the project has never used and is not evidence that a
    zone can actually serve capacity.
    """
    payload = _fetch_quota_payload(family, project)
    return frozenset(
        entry["dimensions"]["zone"]
        for entry in payload.get("dimensionsInfos", [])
        if (entry.get("dimensions") or {}).get("zone") and (entry.get("details") or {}).get("value") is not None
    )


@functools.cache
def _fetch_quota_limits(family: str, project: str) -> dict[str, int]:
    """Return ``{zone: limit}`` for a family's spot quota. Cached per process."""
    return _parse_quota_payload(_fetch_quota_payload(family, project))


@functools.cache
def _fetch_quota_payload(family: str, project: str) -> dict:
    try:
        quota_id = SPOT_QUOTA_IDS[family]
    except KeyError:
        raise ValueError(f"No spot quota metric known for TPU family {family!r}") from None
    result = run_gcloud(
        [
            "alpha",
            "quotas",
            "info",
            "describe",
            quota_id,
            "--service=tpu.googleapis.com",
            "--format=json",
        ],
        project=project,
        check=False,
        timeout=_QUOTA_QUERY_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read {family} spot quota: {result.stderr.strip()}")
    return json.loads(result.stdout)


def spot_quota_zones(
    family: str,
    size: int,
    *,
    project: str,
    occupied_zones: "frozenset[str] | None" = None,
    region: str | None = None,
) -> tuple[str, ...]:
    """Return zones worth racing for a ``size``-core pod of ``family``.

    Quota alone over-reports badly: the project default grant covers dozens of zones it
    has never used, so racing all of them creates queued resources that wait forever. A
    zone qualifies when its quota fits the shape, it is in the US or EU, and either

    - it carries an **explicit** per-zone quota override (someone provisioned it), or
    - it currently holds a TPU (``occupied_zones``), which is direct evidence it works.

    ``region`` (e.g. ``"europe-west4"``) restricts the result to zones with that prefix.
    Use it when the job's data lives in one region and a pod anywhere else would read it
    across regions -- the race then never lands outside, rather than winning a distant
    zone and paying egress for the whole run.
    """
    limits = _fetch_quota_limits(family, project)
    overrides = _fetch_quota_overrides(family, project)
    permitted = overrides | (occupied_zones or frozenset())

    def _in_region(zone: str) -> bool:
        return region is None or zone.startswith(region + "-") or zone == region

    zones = tuple(
        sorted(
            zone
            for zone, limit in limits.items()
            if limit >= size and _continent_allowed(zone) and zone in permitted and _in_region(zone)
        )
    )
    if not zones:
        where = f"in region {region!r}" if region else "in US/EU"
        raise ValueError(
            f"No eligible zone for {family}-{size} {where}: quota fits in "
            f"{sum(1 for z, v in limits.items() if v >= size and _continent_allowed(z) and _in_region(z))} zones "
            f"there, but none is quota-overridden or currently occupied"
        )
    logger.info(
        "Spot quota for %s-%d%s: %d eligible zones %s",
        family,
        size,
        f" in {region}" if region else "",
        len(zones),
        zones,
    )
    return zones
