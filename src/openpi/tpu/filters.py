"""What a launch will accept as a pod, as one object.

The acquisition path has three ways to end up holding a pod -- a named pod, reuse of an
idle one, and the spot race -- and each used to apply its own subset of the restrictions.
The shape came from ``tpu_type``, the region from ``region``, ownership from
``only_my_pods``, and nothing carried them together. That is how a launch pinned to one
pod with ``--tpu-name v4-64-0`` ended up training on a colleague's ``v4-vansh-spot-64``:
the retry dropped the pin, the reserved path fell through to reuse, and reuse checked only
the shape prefix.

:class:`PodFilter` is the single answer to "may this launch use that pod". Every path
consults it, it explains its refusals so a skipped pod says why in the log, and adding a
new dimension means adding it here rather than in three call sites.
"""

import dataclasses

from openpi.tpu.config import accelerator_type_for
from openpi.tpu.config import get_tpu_type_prefix
from openpi.tpu.config import region_from_zone


def continent_of_zone(zone: str) -> str:
    """'europe-west4-a' -> 'eu', 'us-central2-b' -> 'us'."""
    locality = region_from_zone(zone).split("-", 1)[0]
    return "eu" if locality.startswith("europe") else locality


def _split(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated CLI value; empty and None both mean unrestricted."""
    return tuple(item.strip() for item in value.split(",") if item.strip()) if value else ()


@dataclasses.dataclass(frozen=True, kw_only=True)
class PodFilter:
    """Which pods a launch may use. Empty tuple means that dimension is unrestricted."""

    tpu_types: tuple[str, ...] = ()
    """Accepted shapes, e.g. ('v5e-64', 'v6e-32'). Matched on the accelerator a pod
    reports where that is known, and on the family prefix of its name otherwise."""

    regions: tuple[str, ...] = ()
    zones: tuple[str, ...] = ()
    continents: tuple[str, ...] = ()
    """'us' and/or 'eu'. Coarser than regions and useful on its own: what usually matters
    is that a pod is on the same side of the Atlantic as the data it will read."""

    only_my_pods: bool = False
    """Refuse a pod whose name does not carry ``resource_owner``. Borrowing a colleague's
    idle pod is a heuristic, and being wrong takes a machine they wanted for days."""

    resource_owner: str = ""

    @classmethod
    def parse(
        cls,
        *,
        tpu_type: str | None = None,
        region: str | None = None,
        zone: str | None = None,
        continent: str | None = None,
        only_my_pods: bool = False,
        resource_owner: str = "",
    ) -> "PodFilter":
        return cls(
            tpu_types=_split(tpu_type),
            regions=_split(region),
            zones=_split(zone),
            continents=_split(continent),
            only_my_pods=only_my_pods,
            resource_owner=resource_owner,
        )

    @property
    def families(self) -> tuple[str, ...]:
        """Family prefixes of the accepted shapes, e.g. ('v5e', 'v6e')."""
        return tuple(dict.fromkeys(get_tpu_type_prefix(t) for t in self.tpu_types))

    @property
    def accelerators(self) -> tuple[str, ...]:
        """Accelerator strings of the accepted shapes; v5e alone renames."""
        return tuple(dict.fromkeys(accelerator_type_for(t) for t in self.tpu_types))

    def rejects(self, *, name: str, zone: str | None, accelerator_type: str | None = None) -> str | None:
        """Why this pod is unusable for this launch, or None if it passes.

        A reason rather than a bool so the caller can log *why* a pod was skipped; silent
        filtering is what made the shape mismatch above hard to see.
        """
        if self.only_my_pods and self.resource_owner and self.resource_owner not in name:
            return f"not this user's pod (--only-my-pods, want {self.resource_owner!r})"
        if self.zones and not (zone and any(zone == z for z in self.zones)):
            return f"zone {zone} is not one of {list(self.zones)}"
        if self.regions and not (zone and any(zone.startswith(r + "-") or zone == r for r in self.regions)):
            return f"zone {zone} is not in {list(self.regions)}"
        if self.continents and not (zone and continent_of_zone(zone) in self.continents):
            return f"zone {zone} is not in {list(self.continents)}"
        if self.tpu_types:
            # An accelerator the pod reports is authoritative; a name prefix is the only
            # evidence available before a pod exists, so it is the fallback rather than
            # the primary test.
            if accelerator_type is not None:
                if accelerator_type not in self.accelerators:
                    return f"accelerator {accelerator_type} is not one of {list(self.accelerators)}"
            elif not any(name.startswith(family) for family in self.families):
                return f"name {name} is not one of the families {list(self.families)}"
        return None

    def accepts(self, *, name: str, zone: str | None, accelerator_type: str | None = None) -> bool:
        return self.rejects(name=name, zone=zone, accelerator_type=accelerator_type) is None

    def without_types(self) -> "PodFilter":
        """The same filter with the shape dimension dropped.

        The resume sweep needs it: a relaunched launcher looking for its own certificate
        has no idea which shape won last time, so filtering by shape there would make it
        miss its own running job and start a second one.
        """
        return dataclasses.replace(self, tpu_types=())

    def describe(self) -> str:
        """One line for the launch log, so what a run will accept is visible up front."""
        parts = []
        if self.tpu_types:
            parts.append(f"types={list(self.tpu_types)}")
        for label, values in (("zones", self.zones), ("regions", self.regions), ("continents", self.continents)):
            if values:
                parts.append(f"{label}={list(values)}")
        if self.only_my_pods:
            parts.append(f"only pods named for {self.resource_owner!r}")
        return ", ".join(parts) or "unrestricted"
