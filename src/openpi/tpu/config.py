"""TPU configuration constants."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class TPUConfig:
    """Configuration for a TPU type."""

    zone: str
    project: str
    is_spot: bool
    runtime_version: str
    nfs_server: str
    nfs_mount_path: str

    @property
    def gcloud_accelerator_type(self) -> str:
        """Get the gcloud accelerator type from the TPU type.

        This is set dynamically based on the tpu_type when getting the config.
        """
        raise NotImplementedError("Use get_tpu_config() to get a fully configured TPUConfig")


@dataclasses.dataclass(frozen=True)
class TPUConfigWithType(TPUConfig):
    """TPU config with the specific accelerator type."""

    tpu_type: str = ""
    accelerator_type: str = ""


TPU_CONFIGS: dict[str, TPUConfig] = {
    "v6e": TPUConfig(
        zone="us-central1-b",
        project="cmu-aidm-v2",
        is_spot=True,
        runtime_version="v2-alpha-tpuv6e",
        nfs_server="10.4.84.2:/nfs_us_central1_b",
        nfs_mount_path="/nfs/aidm_nfs",
    ),
    "v5e": TPUConfig(
        zone="europe-west4-b",
        project="cmu-aidm-v2",
        is_spot=False,
        runtime_version="v2-alpha-tpuv5-lite",
        nfs_server="10.155.154.42:/europe",
        nfs_mount_path="/nfs/aidm_nfs",
    ),
}


def get_tpu_config(tpu_type: str) -> TPUConfigWithType:
    """Get TPU config for a given TPU type.

    Args:
        tpu_type: TPU type string like "v6e-8" or "v5e-128"

    Returns:
        TPUConfigWithType with all configuration including accelerator type

    Raises:
        ValueError: If the TPU type is not recognized
    """
    if tpu_type.startswith("v6e"):
        base_config = TPU_CONFIGS["v6e"]
        accelerator_type = tpu_type
    elif tpu_type.startswith("v5e"):
        base_config = TPU_CONFIGS["v5e"]
        size = tpu_type.split("-")[1]
        accelerator_type = f"v5litepod-{size}"
    else:
        raise ValueError(f"Unknown TPU type: {tpu_type}. Expected v6e-* or v5e-*")

    return TPUConfigWithType(
        zone=base_config.zone,
        project=base_config.project,
        is_spot=base_config.is_spot,
        runtime_version=base_config.runtime_version,
        nfs_server=base_config.nfs_server,
        nfs_mount_path=base_config.nfs_mount_path,
        tpu_type=tpu_type,
        accelerator_type=accelerator_type,
    )


def get_tpu_type_prefix(tpu_type: str) -> str:
    """Extract the type prefix from a TPU type (e.g., 'v6e' from 'v6e-8')."""
    return tpu_type.split("-")[0]


def get_worker_count(tpu_type: str) -> int:
    """Get number of workers from TPU type.

    For v6e TPUs, each host has 8 chips, so worker count = max(1, chips / 8).
    Examples: v6e-8 -> 1 worker, v6e-64 -> 8 workers, v6e-256 -> 32 workers.

    Args:
        tpu_type: TPU type string like "v6e-8" or "v6e-64"

    Returns:
        Number of workers (hosts) for this TPU type
    """
    chips = int(tpu_type.split("-")[1])
    return max(1, chips // 4)
