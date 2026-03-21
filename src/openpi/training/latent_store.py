"""Latent Store: Storage and retrieval of pre-computed video latent sequences.

This module provides infrastructure for storing full trajectory latent sequences
(output of a VAE encoder) and slicing them at training time based on view
configurations (future-facing or past-facing).

Export time: Encode entire trajectory -> single latent sequence [C, T_lat, H_lat, W_lat] per camera
Training time: Slice into the latent sequence based on view config (future vs past, window size)

The latent store is stored as a TFDS dataset.
This ensures identical loading behavior (file shuffling, process sharding) as the source RLDS
dataset, enabling deterministic joining via `tfds.builder_from_directory()`.

A manifest.json at the root stores metadata about the source dataset, encoding parameters,
and latent dimensions.
"""

import dataclasses
import json
import logging
from typing import Any, Literal

from etils import epath
import numpy as np

logger = logging.getLogger(__name__)

# Dataset name and version for the latent store TFDS dataset
LATENT_STORE_DATASET_NAME = "latent_store"
LATENT_STORE_VERSION = "1.0.0"


def get_shard_filename(split: str, shard_idx: int) -> str:
    """Generate TFDS shard filename for latent store."""
    return f"{LATENT_STORE_DATASET_NAME}-{split}.tfrecord-{shard_idx:05d}"


def make_safe_latent_key(image_key: str) -> str:
    """Convert image key to safe latent feature key (replaces '/' with '_')."""
    return f"{image_key.replace('/', '_')}_latents"


# =============================================================================
# Data structures
# =============================================================================


@dataclasses.dataclass(frozen=True)
class LatentStoreManifest:
    """Manifest stored as manifest.json at latent store root.

    Contains metadata about the source RLDS dataset, the encoding process,
    and the resulting latent dimensions. Used to validate compatibility
    between the latent store and the source dataset at training time.
    """

    version: str = "1.0"

    # Source RLDS identity for validation
    source_rlds_data_dir: str = ""
    source_dataset_name: str = ""
    source_dataset_version: str = ""
    source_num_episodes: int = 0

    # Which cameras were encoded
    image_keys: tuple[str, ...] = ()

    # Latent properties
    latent_channels: int = 0
    latent_height: int = 0
    latent_width: int = 0
    temporal_compression: int = 4  # Factor (e.g., 4 means 4 raw frames -> 1 latent frame)

    # VAE metadata (for reproducibility)
    vae_params_path: str = ""
    vae_metadata_path: str = ""

    # FPS configuration (for target_fps support)
    source_fps: int = 0  # Original FPS of source data (e.g., 30)
    target_fps: int = 0  # Target FPS after downsampling (e.g., 10)
    vae_temporal_compression: int = 0  # Raw VAE compression (e.g., 4)
    # temporal_compression is the EFFECTIVE value: vae_tc * fps_ratio

    # Creation timestamp
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "source_rlds_data_dir": self.source_rlds_data_dir,
            "source_dataset_name": self.source_dataset_name,
            "source_dataset_version": self.source_dataset_version,
            "source_num_episodes": self.source_num_episodes,
            "image_keys": list(self.image_keys),
            "latent_channels": self.latent_channels,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "temporal_compression": self.temporal_compression,
            "vae_params_path": self.vae_params_path,
            "vae_metadata_path": self.vae_metadata_path,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "vae_temporal_compression": self.vae_temporal_compression,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LatentStoreManifest":
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            source_rlds_data_dir=data.get("source_rlds_data_dir", ""),
            source_dataset_name=data.get("source_dataset_name", ""),
            source_dataset_version=data.get("source_dataset_version", ""),
            source_num_episodes=data.get("source_num_episodes", 0),
            image_keys=tuple(data.get("image_keys", [])),
            latent_channels=data.get("latent_channels", 0),
            latent_height=data.get("latent_height", 0),
            latent_width=data.get("latent_width", 0),
            temporal_compression=data.get("temporal_compression", 4),
            vae_params_path=data.get("vae_params_path", ""),
            vae_metadata_path=data.get("vae_metadata_path", ""),
            source_fps=data.get("source_fps", 0),
            target_fps=data.get("target_fps", 0),
            vae_temporal_compression=data.get("vae_temporal_compression", 0),
            created_at=data.get("created_at", ""),
        )


@dataclasses.dataclass(frozen=True)
class LatentViewConfig:
    """Training-time configuration for slicing latent sequences.

    Determines how to extract a window from the full latent sequence
    for each training step.
    """

    direction: Literal["future", "past"]
    """Which direction to slice: "future" looks ahead, "past" looks back."""

    window_size: int
    """Number of latent frames to extract per step."""

    output_key: str
    """Key name prefix in the batch dict (e.g., "video_latents" or "observation_latents")."""

    image_keys: tuple[str, ...] | None = None
    """Which cameras to extract latents for. None means all cameras in the store."""

    stride: int = 1
    """Stride between extracted latent frames. stride=1 means contiguous frames,
    stride=6 means every 6th frame (e.g., 12 seconds apart at 2 FPS with temporal_compression=4)."""


# =============================================================================
# Manifest I/O
# =============================================================================


def load_manifest(latent_store_dir: str) -> LatentStoreManifest:
    """Load and parse manifest.json from a latent store directory.

    Args:
        latent_store_dir: Path to the latent store root directory.

    Returns:
        Parsed LatentStoreManifest.

    Raises:
        FileNotFoundError: If manifest.json doesn't exist.
        ValueError: If manifest is malformed.
    """
    manifest_path = epath.Path(latent_store_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Latent store manifest not found at {manifest_path}")

    with manifest_path.open("r") as f:
        data = json.load(f)

    return LatentStoreManifest.from_dict(data)


def save_manifest(manifest: LatentStoreManifest, latent_store_dir: str) -> None:
    """Save manifest.json to a latent store directory.

    Args:
        manifest: Manifest to save.
        latent_store_dir: Path to the latent store root directory.
    """
    manifest_path = epath.Path(latent_store_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def validate_manifest_against_rlds(
    manifest: LatentStoreManifest,
    rlds_data_dir: str,
    dataset_name: str,
    dataset_version: str,
) -> None:
    """Validate that the latent store manifest matches the source RLDS dataset.

    Args:
        manifest: The latent store manifest.
        rlds_data_dir: Path to the RLDS data directory.
        dataset_name: Name of the RLDS dataset.
        dataset_version: Version of the RLDS dataset.

    Raises:
        RuntimeError: If the manifest doesn't match the source dataset.
    """
    # Note: We do a relaxed validation that only checks dataset name and version,
    # not the exact data directory path (which may differ between export and training).
    if manifest.source_dataset_name != dataset_name:
        raise RuntimeError(
            f"Latent store was created from dataset '{manifest.source_dataset_name}' "
            f"but trying to use with dataset '{dataset_name}'."
        )
    if manifest.source_dataset_version != dataset_version:
        raise RuntimeError(
            f"Latent store was created from dataset version '{manifest.source_dataset_version}' "
            f"but trying to use with version '{dataset_version}'."
        )


# =============================================================================
# TFDS-based storage (for deterministic joining with RLDS)
# =============================================================================


def get_latent_store_tfds_feature_spec(manifest: LatentStoreManifest) -> Any:
    """Build TFDS feature spec for latent store dataset.

    Creates a FeaturesDict compatible with tfds.core.SequentialWriter that stores
    episode-level latent sequences with variable temporal dimensions.

    Args:
        manifest: Latent store manifest with dimension info.

    Returns:
        tfds.features.FeaturesDict with episode metadata and per-camera latent tensors.
    """
    import tensorflow_datasets as tfds

    features = {
        "episode_index": tfds.features.Scalar(dtype=np.int64),
        "num_steps": tfds.features.Scalar(dtype=np.int64),
        "num_latent_frames": tfds.features.Scalar(dtype=np.int64),
    }

    for image_key in manifest.image_keys:
        # Variable temporal dimension (None), fixed spatial dims
        features[make_safe_latent_key(image_key)] = tfds.features.Tensor(
            shape=(manifest.latent_channels, None, manifest.latent_height, manifest.latent_width),
            dtype=np.float32,
        )

    return tfds.features.FeaturesDict(features)


class LatentStoreTFDSShardWriter:
    """Writer that creates a single TFDS shard file matching an RLDS shard.

    This writer creates shard files with indices that exactly match the source RLDS
    dataset's shards. This ensures that when TFDS loads both datasets with the same
    shuffle settings, the file interleaving produces identical episode ordering.

    Each instance writes one shard; create multiple instances for multiple shards.
    """

    def __init__(
        self,
        output_dir: str,
        shard_idx: int,
        manifest: LatentStoreManifest,
        *,
        split: str = "train",
    ):
        """Initialize the shard writer.

        Args:
            output_dir: Root directory for the latent store (TFDS dataset will be at
                output_dir/{LATENT_STORE_DATASET_NAME}/{LATENT_STORE_VERSION}/).
            shard_idx: Index of this shard (must match source RLDS shard index).
            manifest: Manifest describing the latent store.
            split: Split name (e.g., "train", "val").
        """
        import tensorflow as tf

        self._output_dir = epath.Path(output_dir)
        self._shard_idx = shard_idx
        self._manifest = manifest
        self._split = split
        self._episode_count = 0
        self._total_bytes = 0

        # Create dataset directory structure
        self._dataset_dir = self._output_dir / LATENT_STORE_DATASET_NAME / LATENT_STORE_VERSION
        self._dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create TFDS feature spec for serialization
        self._features = get_latent_store_tfds_feature_spec(manifest)

        # Create shard file with TFDS naming convention
        self._shard_path = self._dataset_dir / get_shard_filename(split, shard_idx)
        self._writer = tf.io.TFRecordWriter(str(self._shard_path))

    def write_episode(
        self,
        episode_index: int,
        num_steps: int,
        latents_dict: dict[str, np.ndarray],
    ) -> None:
        """Write a single episode's latents to the shard.

        Args:
            episode_index: Index of the episode in the source dataset (RLDS episode_index).
            num_steps: Number of raw steps in the episode.
            latents_dict: Dict mapping image_key -> latent array [C, T_lat, H, W].
        """
        example: dict[str, Any] = {
            "episode_index": episode_index,
            "num_steps": num_steps,
        }

        num_latent_frames = None
        for image_key in self._manifest.image_keys:
            if image_key not in latents_dict:
                raise ValueError(f"Missing latents for image key '{image_key}'")

            latents = latents_dict[image_key]
            if num_latent_frames is None:
                num_latent_frames = latents.shape[1]
            elif latents.shape[1] != num_latent_frames:
                raise ValueError(
                    f"Inconsistent latent frame counts: expected {num_latent_frames}, "
                    f"got {latents.shape[1]} for key '{image_key}'"
                )

            example[make_safe_latent_key(image_key)] = latents.astype(np.float32)

        example["num_latent_frames"] = num_latent_frames or 0

        # Serialize using TFDS features
        serialized = self._features.serialize_example(example)
        self._writer.write(serialized)
        self._total_bytes += len(serialized)
        self._episode_count += 1

    def finalize(self) -> None:
        """Close the writer."""
        self._writer.close()
        logger.info(f"Wrote {self._episode_count} episodes to shard {self._shard_idx} at {self._shard_path}")

    @property
    def episode_count(self) -> int:
        """Number of episodes written to this shard."""
        return self._episode_count

    @property
    def shard_idx(self) -> int:
        """Index of this shard."""
        return self._shard_idx

    @property
    def total_bytes(self) -> int:
        """Total bytes written to this shard."""
        return self._total_bytes

    @property
    def dataset_dir(self) -> epath.Path:
        """Path to the TFDS dataset directory."""
        return self._dataset_dir


def get_latent_store_builder(latent_store_dir: str) -> Any:
    """Load latent store as a TFDS builder.

    Locates the TFDS dataset directory within the latent store and returns a builder
    that can be used with standard TFDS loading (as_dataset, split_for_jax_process, etc.).

    Args:
        latent_store_dir: Path to the latent store root directory.

    Returns:
        TFDS DatasetBuilder for the latent store.

    Raises:
        FileNotFoundError: If no TFDS dataset found in the latent store directory.
    """
    import tensorflow_datasets as tfds

    store_path = epath.Path(latent_store_dir)

    # Check for TFDS dataset at expected location
    expected_tfds_dir = store_path / LATENT_STORE_DATASET_NAME / LATENT_STORE_VERSION
    if expected_tfds_dir.exists() and (expected_tfds_dir / "dataset_info.json").exists():
        return tfds.builder_from_directory(str(expected_tfds_dir))

    # Fall back to searching for dataset_info.json
    candidates = list(store_path.glob("*/*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No TFDS dataset found in {latent_store_dir}. "
            f"Expected dataset at {expected_tfds_dir} or a dataset_info.json somewhere under {store_path}."
        )

    dataset_dir = candidates[0].parent
    logger.info(f"Found latent store TFDS dataset at {dataset_dir}")
    return tfds.builder_from_directory(str(dataset_dir))


# =============================================================================
# Slicing utilities
# =============================================================================


def slice_latents_for_step(
    latents: Any,  # tf.Tensor or np.ndarray
    step: int,
    temporal_compression: int,
    direction: Literal["future", "past"],
    window_size: int,
    stride: int = 1,
) -> Any:
    """Extract latent window for a given raw step.

    For "future" direction: extracts latents starting at the current frame.
    For "past" direction: extracts latents ending at the current frame.

    When stride > 1, extracts sparse frames (e.g., stride=6 for every 6th frame).
    Pads at episode boundaries using edge padding.

    Args:
        latents: Full latent sequence [C, T_lat, H_lat, W_lat].
        step: Raw step index in the original trajectory.
        temporal_compression: Temporal compression factor.
        direction: "future" or "past".
        window_size: Number of latent frames to extract.
        stride: Stride between extracted frames (1 = contiguous, >1 = sparse).

    Returns:
        Sliced latents [C, window_size, H_lat, W_lat].
    """
    import tensorflow as tf

    lat_idx = step // temporal_compression
    num_latent_frames = tf.shape(latents)[1]

    if stride == 1:
        # Original contiguous slicing logic
        if direction == "future":
            start = lat_idx
            end = lat_idx + window_size
        else:  # "past"
            start = lat_idx - window_size + 1
            end = lat_idx + 1

        # Handle out-of-bounds with padding
        pad_before = tf.maximum(0, -start)
        pad_after = tf.maximum(0, end - num_latent_frames)

        # Clamp indices to valid range
        start_clamped = tf.maximum(0, start)
        end_clamped = tf.minimum(num_latent_frames, end)

        # Slice the valid portion
        sliced = latents[:, start_clamped:end_clamped, :, :]

        # Repeat boundary frames to keep window size fixed at episode boundaries.
        first_frame = sliced[:, :1, :, :]
        last_frame = sliced[:, -1:, :, :]
        pad_before_frames = tf.repeat(first_frame, pad_before, axis=1)
        pad_after_frames = tf.repeat(last_frame, pad_after, axis=1)
        sliced = tf.concat([pad_before_frames, sliced, pad_after_frames], axis=1)

        # Ensure exact window size and return
        return sliced[:, :window_size, :, :]

    # Strided (sparse) slicing: extract frames at indices [start, start+stride, ...]
    if direction == "future":
        # Indices: lat_idx, lat_idx + stride, lat_idx + 2*stride, ...
        indices = lat_idx + tf.range(window_size) * stride
    else:  # "past"
        # Indices: ..., lat_idx - 2*stride, lat_idx - stride, lat_idx
        indices = lat_idx - (window_size - 1 - tf.range(window_size)) * stride

    # Clamp indices to valid range
    indices_clamped = tf.clip_by_value(indices, 0, num_latent_frames - 1)

    # Gather frames at strided indices
    # latents is [C, T, H, W], we need to gather along axis 1
    latents_transposed = tf.transpose(latents, [1, 0, 2, 3])  # [T, C, H, W]
    gathered = tf.gather(latents_transposed, indices_clamped)  # [window_size, C, H, W]
    return tf.transpose(gathered, [1, 0, 2, 3])  # [C, window_size, H, W]


# =============================================================================
# Batch slicing for training
# =============================================================================


def slice_latents_for_trajectory(
    latents: Any,  # tf.Tensor [C, T_lat, H_lat, W_lat]
    traj_len: int,
    temporal_compression: int,
    direction: Literal["future", "past"],
    window_size: int,
    stride: int = 1,
) -> Any:
    """Slice latent sequences into per-step windows for an entire trajectory.

    Args:
        latents: Full latent sequence [C, T_lat, H_lat, W_lat].
        traj_len: Number of raw steps in the trajectory.
        temporal_compression: Temporal compression factor.
        direction: "future" or "past".
        window_size: Number of latent frames to extract per step.
        stride: Stride between extracted frames (1 = contiguous, >1 = sparse).

    Returns:
        Per-step latent windows [traj_len, C, window_size, H_lat, W_lat].
    """
    import tensorflow as tf

    def slice_at_step(step):
        return slice_latents_for_step(
            latents,
            step,
            temporal_compression,
            direction,
            window_size,
            stride,
        )

    # Map over all steps and return
    return tf.map_fn(
        slice_at_step,
        tf.range(traj_len),
        fn_output_signature=tf.TensorSpec(
            shape=[None, window_size, None, None],  # [C, window_size, H, W]
            dtype=latents.dtype,
        ),
    )
