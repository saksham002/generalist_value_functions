"""Distributed GPU computation of Cosmos VAE latent stores for RLDS datasets.

Creates a latent store containing full trajectory latent sequences for each camera.
Training-time view configuration determines how to slice into these sequences.

Three tyro subcommands:
  launch  - Submit N sbatch worker jobs from the cluster head node.
  worker  - Process a contiguous chunk of episodes on a single GPU.
  merge   - Combine per-worker outputs into a single latent store.

Usage:
    # 1. Launch N workers
    uv run scripts/compute_latent_store.py launch \
        --config-name cosmos_robocoin_video_prediction \
        --output-dir gs://max-us-central1/ROBOCOIN_latent_store \
        --num-workers 16

    # 2. Worker mode (invoked by sbatch, not called directly)
    uv run scripts/compute_latent_store.py worker \
        --config-name cosmos_robocoin_video_prediction \
        --output-dir gs://max-us-central1/ROBOCOIN_latent_store \
        --num-workers 16 --worker-id 3

    # 3. Merge (after all workers finish)
    uv run scripts/compute_latent_store.py merge \
        --config-name cosmos_robocoin_video_prediction \
        --output-dir gs://max-us-central1/ROBOCOIN_latent_store \
        --num-workers 16
"""

import dataclasses
import datetime
import io
import json
import logging
import shlex
import subprocess
import time
from typing import Annotated, Any

from etils import epath
import numpy as np
from PIL import Image
import tqdm
import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Argument dataclasses
# =============================================================================


@dataclasses.dataclass
class CommonArgs:
    """Arguments shared across all subcommands."""

    config_name: str
    """Config name to resolve RLDS data source and VAE paths."""

    output_dir: str
    """Output latent store directory (local or GCS path)."""

    split: str = "train"
    """Split to process (e.g., 'train', 'val')."""

    num_workers: int = 8
    """Number of worker jobs to distribute across."""

    target_fps: int | None = None
    """Target FPS for frame downsampling. If None, uses source FPS (no downsampling)."""

    source_fps: int | None = None
    """Expected source FPS. If None, read from trajectory metadata."""


@dataclasses.dataclass
class LaunchArgs(CommonArgs):
    """Launch N sbatch worker jobs from the cluster head node."""

    partition: str = "preempt"
    """SLURM partition (used when partition_split is not specified)."""

    partition_split: str | None = None
    """Split workers across partitions. Format: 'partition1:count1,partition2:count2,...'.
    Example: 'general:8,preempt:24' assigns workers 0-7 to general, 8-31 to preempt.
    Total counts must equal num_workers. Overrides --partition when specified."""

    time_limit: str = "12:00:00"
    """SLURM time limit."""

    mem: str = "64GB"
    """SLURM memory limit."""

    gres: str = "gpu:L40S:1"
    """SLURM GPU resource spec."""

    encode_batch_size: int = 8
    """Number of frames per VAE forward pass."""

    max_episodes: int | None = None
    """Limit total episodes to process across all workers."""

    target_height: int | None = None
    """Optional target height override for video frames."""

    target_width: int | None = None
    """Optional target width override for video frames."""

    auto_merge: bool = False
    """Submit a dependent merge job after all workers."""

    dry_run: bool = False
    """Print sbatch commands without submitting."""

    uv_bin: str = "uv"
    """Full path to the uv binary on the worker nodes."""

    total_episodes: int | None = None
    """Total source episodes. If provided, skips querying the dataset."""

    ssh_host: str | None = None
    """SSH host to route sbatch through (e.g., 'babel')."""

    log_dir_base: str = "~/slurm_logs"
    """Base directory for SLURM log files."""


@dataclasses.dataclass
class WorkerArgs(CommonArgs):
    """Process a contiguous chunk of episodes on a single GPU."""

    worker_id: int = 0
    """This worker's index (0-based)."""

    encode_batch_size: int = 8
    """Number of frames per VAE forward pass."""

    max_episodes: int | None = None
    """Limit total episodes to process across all workers."""

    target_height: int | None = None
    """Optional target height override for video frames."""

    target_width: int | None = None
    """Optional target width override for video frames."""


@dataclasses.dataclass
class MergeArgs(CommonArgs):
    """Combine per-worker outputs into a single latent store."""

    source_dir: str | None = None
    """Directory containing worker outputs. Defaults to output_dir if not specified.
    Use this to read workers from one location (e.g., local disk) and write merged
    output to another (e.g., GCS)."""

    overwrite: bool = False
    """Overwrite existing merged output."""

    cleanup_workers: bool = False
    """Delete worker directories after successful merge."""


# =============================================================================
# Episode range computation
# =============================================================================


def compute_episode_ranges(
    total_episodes: int,
    num_workers: int,
    max_episodes: int | None = None,
) -> list[tuple[int, int]]:
    """Compute (start_episode, num_episodes) for each worker."""
    effective = min(total_episodes, max_episodes) if max_episodes is not None else total_episodes
    base = effective // num_workers
    remainder = effective % num_workers
    ranges = []
    offset = 0
    for worker_id in range(num_workers):
        count = base + (1 if worker_id < remainder else 0)
        ranges.append((offset, count))
        offset += count
    return ranges


# =============================================================================
# Config resolution helpers
# =============================================================================


def resolve_config(config_name: str):
    """Resolve openpi config and return (config, data_config, dataset_cfg, model_config)."""
    import openpi.training.config as _config

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is None:
        raise ValueError("Config must have rlds_data_dir set.")

    datasets = data_config.datasets
    if not datasets:
        raise ValueError("Config must have datasets configured.")
    if len(datasets) > 1:
        logger.warning("Multiple datasets configured, only processing first one.")

    return config, data_config, datasets[0], config.model


def get_total_episodes(data_dir: str, dataset_cfg, split: str) -> int:
    """Get total episode count from source dataset."""
    import tensorflow_datasets as tfds

    builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)
    return builder.info.splits[split].num_examples


def get_rlds_shard_info(data_dir: str, dataset_cfg, split: str) -> list[tuple[int, int]]:
    """Get (start_episode, num_episodes) for each RLDS shard.

    These are positional indices for split syntax, not the RLDS episode_index feature.
    Used to process shards independently while maintaining 1:1 correspondence.
    """
    import tensorflow_datasets as tfds

    builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)
    split_info = builder.info.splits[split]
    shard_lengths = split_info.shard_lengths

    ranges = []
    offset = 0
    for length in shard_lengths:
        ranges.append((offset, length))
        offset += length

    return ranges


def get_rlds_episode_index(episode: dict) -> int:
    """Extract the episode_index feature from RLDS episode.

    For RoboCOIN, this is in each step's metadata as 'episode_index' (int64).
    This is the ground-truth identifier, NOT the positional index.
    """
    first_step = episode["steps"][0]
    return int(first_step["episode_index"])


def get_episode_fps(episode: dict) -> int:
    """Extract FPS from episode metadata (RoboCOIN format)."""
    fps = episode["episode_metadata"]["fps"]
    if hasattr(fps, "numpy"):
        fps = fps.numpy()
    return int(fps)


# =============================================================================
# VAE loading
# =============================================================================


def load_cosmos_vae(
    vae_params_path: str,
    metadata_path: str,
) -> tuple[Any, np.ndarray, np.ndarray, dict]:
    """Load Cosmos VAE with pretrained weights."""
    from cosmos_predict2._src.predict2_jax.vae_wan2pt1 import Wan2pt1VAENNX
    from flax import nnx

    import openpi.models.model as _model
    import openpi.shared.download as _download
    import openpi.shared.nnx_utils as nnx_utils

    metadata_path = _download.maybe_download(metadata_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    vae_config = metadata["vae_config"]
    latent_mean = np.array(metadata["latent_normalization"]["mean"], dtype=np.float32)
    latent_inv_std = np.array(metadata["latent_normalization"]["inv_std"], dtype=np.float32)

    logger.info("Creating Cosmos VAE model")
    jax_vae = Wan2pt1VAENNX(**vae_config)

    vae_params_path = _download.maybe_download(vae_params_path)
    logger.info(f"Loading VAE params from {vae_params_path}")
    vae_params = _model.restore_params(vae_params_path, restore_type=np.ndarray)

    _, vae_state = nnx.split(jax_vae)
    nnx_utils.replace_state_from_pure_dict_numeric_key_compat(vae_state, vae_params)
    jax_vae = nnx.merge(nnx.graphdef(jax_vae), vae_state)

    return jax_vae, latent_mean, latent_inv_std, vae_config


# =============================================================================
# Image processing utilities
# =============================================================================


def decode_jpeg_image(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to numpy array."""
    return np.array(Image.open(io.BytesIO(jpeg_bytes)))


def normalize_image_for_encoding(value: Any) -> np.ndarray:
    """Normalize image value from RLDS step to uint8 [H, W, C]."""
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        value = value.numpy()

    if isinstance(value, bytes | bytearray | memoryview | np.bytes_):
        return decode_jpeg_image(bytes(value))

    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in ("S", "O", "U"):
        scalar_value = value.item()
        if isinstance(scalar_value, str):
            scalar_value = scalar_value.encode("utf-8")
        if isinstance(scalar_value, bytes | bytearray | memoryview | np.bytes_):
            return decode_jpeg_image(bytes(scalar_value))

    if isinstance(value, np.ndarray):
        if value.ndim != 3:
            raise ValueError(f"Expected image array with shape [H, W, C], got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image array, got {value.dtype}")
        return value

    raise TypeError(f"Unsupported image value type: {type(value)}")


def resize_image(img: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize image to target dimensions."""
    if img.shape[0] == target_height and img.shape[1] == target_width:
        return img
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_width, target_height), Image.BILINEAR)
    return np.array(pil_img)


def extract_all_frames(episode: dict, image_key: str) -> np.ndarray:
    """Extract all frames for a camera from an episode.

    Args:
        episode: RLDS episode dict with 'steps' containing observations.
        image_key: Key to extract (e.g., 'image', 'observation/image/cam_0').

    Returns:
        Array of shape [T, H, W, C] as uint8.
    """
    frames = []
    for step in episode["steps"]:
        # Try flat key first (RLDS datasets often use flat keys like "observation/image/cam_0")
        if image_key in step:
            value = step[image_key]
        elif "/" in image_key:
            # Fall back to nested access for nested dict structure
            parts = image_key.split("/")
            value = step
            for part in parts:
                value = value[part]
        else:
            value = step["observation"][image_key]

        frame = normalize_image_for_encoding(value)
        frames.append(frame)

    return np.stack(frames, axis=0)


def infer_image_keys(source_features) -> tuple[str, ...]:
    """Infer image observation keys from source RLDS feature schema."""
    import tensorflow_datasets as tfds

    if "steps" not in source_features:
        raise ValueError("Expected top-level 'steps' feature in source RLDS dataset.")

    steps_feature = source_features["steps"]
    if not hasattr(steps_feature, "feature"):
        raise ValueError("Expected source 'steps' feature to contain nested feature spec.")

    step_features = steps_feature.feature
    image_keys: list[str] = []

    def _is_image_feature(feature) -> bool:
        if isinstance(feature, tfds.features.Image):
            return True
        shape = getattr(feature, "shape", None)
        dtype = getattr(feature, "dtype", None)
        if dtype is None:
            dtype = getattr(feature, "np_dtype", None)
        if shape is not None and dtype is not None and len(shape) == 3 and shape[-1] in (1, 3, 4) and dtype == np.uint8:
            return True
        if shape is not None and len(shape) == 0:
            dtype_str = str(dtype) if dtype is not None else ""
            if "string" in dtype_str.lower() or dtype in (np.object_, np.bytes_):
                return True
        return False

    if "observation" in step_features:
        observation_features = step_features["observation"]
        if isinstance(observation_features, tfds.features.FeaturesDict):
            for key, feature in observation_features.items():
                if _is_image_feature(feature):
                    image_keys.append(key)
                elif isinstance(feature, tfds.features.FeaturesDict):
                    for sub_key, sub_feature in feature.items():
                        if _is_image_feature(sub_feature):
                            image_keys.append(f"{key}/{sub_key}")

    for key, feature in step_features.items():
        if key.startswith("observation/") and _is_image_feature(feature):
            image_keys.append(key)

    if not image_keys:
        raise ValueError("No image keys found in source dataset.")

    return tuple(sorted(set(image_keys)))


# =============================================================================
# Worker implementation
# =============================================================================


def run_worker(args: WorkerArgs) -> None:
    """Run a single worker that processes its assigned RLDS shards.

    Each worker writes a TFDS dataset using LatentStoreTFDSWriter. Episodes are
    processed in RLDS shard order (round-robin across workers) to maintain alignment.
    The merge step combines worker TFDS datasets into a single dataset with sequential
    shard numbering.
    """
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    import tensorflow_datasets as tfds

    import openpi.training.latent_store as latent_store

    tf.config.set_visible_devices([], "GPU")

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id={args.worker_id} out of range [0, {args.num_workers}).")

    config, data_config, dataset_cfg, model_config = resolve_config(args.config_name)

    total_episodes = get_total_episodes(data_config.rlds_data_dir, dataset_cfg, args.split)
    shard_info = get_rlds_shard_info(data_config.rlds_data_dir, dataset_cfg, args.split)
    num_shards = len(shard_info)

    # Distribute shards across workers (round-robin)
    my_shards = [i for i in range(num_shards) if i % args.num_workers == args.worker_id]
    my_episode_count = sum(shard_info[i][1] for i in my_shards)

    logger.info(
        f"Worker {args.worker_id}/{args.num_workers}: assigned {len(my_shards)} shards "
        f"({my_episode_count} episodes total)"
    )
    for shard_idx in my_shards:
        start_pos, num_eps = shard_info[shard_idx]
        logger.info(f"  Shard {shard_idx}: episodes [{start_pos}, {start_pos + num_eps})")

    output_dir = epath.Path(args.output_dir)
    worker_dir = output_dir / "_workers" / f"worker_{args.worker_id}"
    done_marker = worker_dir / "_DONE"

    if done_marker.exists():
        logger.info(f"Worker {args.worker_id}: already completed, skipping.")
        return

    # Load VAE
    vae_params_path = getattr(model_config, "vae_params_path", None)
    cosmos_kwargs_path = getattr(model_config, "cosmos_kwargs_path", None)
    if vae_params_path is None or cosmos_kwargs_path is None:
        raise ValueError("Model config must have vae_params_path and cosmos_kwargs_path.")

    logger.info(f"Loading Cosmos VAE from {vae_params_path}")
    vae, latent_mean, latent_inv_std, vae_config = load_cosmos_vae(vae_params_path, cosmos_kwargs_path)

    # Resolve target dimensions
    target_height = args.target_height or vae_config.get("in_height", 480)
    target_width = args.target_width or vae_config.get("in_width", 848)
    vae_temporal_compression = vae_config.get("temporal_compression", 4)

    # FPS configuration
    source_fps = args.source_fps
    target_fps = args.target_fps

    # Load source dataset info
    source_builder = tfds.builder(dataset_cfg.name, data_dir=data_config.rlds_data_dir, version=dataset_cfg.version)
    image_keys = infer_image_keys(source_builder.info.features)
    logger.info(f"Image keys: {image_keys}")

    if target_fps is not None and source_fps is None:
        first_episode = next(iter(source_builder.as_dataset(split=f"{args.split}[0:1]")), None)
        if first_episode is None:
            raise ValueError(f"Split '{args.split}' is empty; cannot infer source_fps.")
        source_fps = get_episode_fps(first_episode)

    # Compute effective temporal compression
    if source_fps is not None and target_fps is not None:
        if source_fps % target_fps != 0:
            raise ValueError(
                f"source_fps ({source_fps}) must be divisible by target_fps ({target_fps}). "
                f"E.g., 30->10 is valid, 30->20 is not."
            )
        fps_ratio = source_fps // target_fps
        effective_temporal_compression = vae_temporal_compression * fps_ratio
        logger.info(
            f"FPS downsampling: {source_fps} -> {target_fps} (ratio={fps_ratio}), "
            f"effective temporal compression: {vae_temporal_compression} * {fps_ratio} = {effective_temporal_compression}"
        )
    else:
        fps_ratio = 1
        effective_temporal_compression = vae_temporal_compression

    logger.info(
        f"Target dimensions: {target_height}x{target_width}, VAE temporal_compression={vae_temporal_compression}"
    )

    @jax.jit
    def encode_batch_jit(videos: jax.Array) -> jax.Array:
        """Encode a batch of video frames to latents."""
        video_float = videos.astype(jnp.float32) / 127.5 - 1.0
        video_float = jnp.transpose(video_float, (0, 4, 1, 2, 3))  # BTHWC -> BCTHW
        return vae.encode(
            video_float,
            scale_mean=latent_mean.reshape(1, -1, 1, 1, 1),
            scale_std=latent_inv_std.reshape(1, -1, 1, 1, 1),
        )

    # Infer latent shape from a test encode
    test_input = np.zeros((1, vae_temporal_compression, target_height, target_width, 3), dtype=np.uint8)
    test_latents = np.array(encode_batch_jit(jnp.array(test_input)))
    latent_shape = test_latents.shape  # [1, C, T_lat, H_lat, W_lat]
    latent_channels = latent_shape[1]
    latent_height = latent_shape[3]
    latent_width = latent_shape[4]
    logger.info(f"Latent shape: C={latent_channels}, H={latent_height}, W={latent_width}")

    # Create manifest
    manifest = latent_store.LatentStoreManifest(
        version="1.0",
        source_rlds_data_dir=data_config.rlds_data_dir,
        source_dataset_name=dataset_cfg.name,
        source_dataset_version=dataset_cfg.version,
        source_num_episodes=total_episodes,
        image_keys=image_keys,
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
        temporal_compression=effective_temporal_compression,
        vae_params_path=vae_params_path,
        vae_metadata_path=cosmos_kwargs_path,
        source_fps=source_fps or 0,
        target_fps=target_fps or 0,
        vae_temporal_compression=vae_temporal_compression,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    total_written = 0
    # Track shard metadata for merge: {shard_idx: {"episode_count": N, "num_bytes": M}}
    shard_metadata: dict[int, dict[str, int]] = {}

    # Process each RLDS shard assigned to this worker, creating matching latent store shards
    for shard_idx in my_shards:
        start_pos, num_episodes = shard_info[shard_idx]

        # Apply max_episodes limit across all workers
        if args.max_episodes is not None:
            global_end = min(args.max_episodes, total_episodes)
            shard_end = start_pos + num_episodes
            if start_pos >= global_end:
                logger.info(f"Shard {shard_idx}: skipping (past max_episodes limit)")
                continue
            if shard_end > global_end:
                num_episodes = global_end - start_pos
                logger.info(f"Shard {shard_idx}: truncating to {num_episodes} episodes due to max_episodes")

        split_spec = f"{args.split}[{start_pos}:{start_pos + num_episodes}]"
        dataset = source_builder.as_dataset(split=split_spec)

        # Create a shard writer that matches this RLDS shard index
        shard_writer = latent_store.LatentStoreTFDSShardWriter(
            output_dir=str(worker_dir),
            shard_idx=shard_idx,
            manifest=manifest,
            split=args.split,
        )

        for raw_episode in tqdm.tqdm(dataset, total=num_episodes, desc=f"Worker {args.worker_id} Shard {shard_idx}"):
            episode = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in raw_episode.items()}
            episode["steps"] = list(episode["steps"])

            # Get the ACTUAL episode_index from RLDS data (not positional)
            rlds_episode_index = get_rlds_episode_index(episode)

            # Determine FPS ratio for this episode
            episode_fps_ratio = fps_ratio

            num_steps = len(episode["steps"])
            latents_dict = {}

            for image_key in image_keys:
                # Extract all frames for this camera
                frames = extract_all_frames(episode, image_key)  # [T, H, W, C]

                # Downsample frames if target_fps is set
                if episode_fps_ratio > 1:
                    frames = frames[::episode_fps_ratio]

                num_downsampled_frames = len(frames)

                # Resize if needed
                if frames.shape[1] != target_height or frames.shape[2] != target_width:
                    resized_frames = np.stack([resize_image(f, target_height, target_width) for f in frames], axis=0)
                else:
                    resized_frames = frames

                # Pad to multiple of VAE temporal_compression
                pad_frames = (
                    vae_temporal_compression - (num_downsampled_frames % vae_temporal_compression)
                ) % vae_temporal_compression
                if pad_frames > 0:
                    padding = np.repeat(resized_frames[-1:], pad_frames, axis=0)
                    resized_frames = np.concatenate([resized_frames, padding], axis=0)

                # Encode in batches
                num_latent_frames = resized_frames.shape[0] // vae_temporal_compression
                all_latents = []

                for batch_start in range(0, num_latent_frames, args.encode_batch_size):
                    batch_end = min(batch_start + args.encode_batch_size, num_latent_frames)
                    batch_frames = resized_frames[
                        batch_start * vae_temporal_compression : batch_end * vae_temporal_compression
                    ]  # [batch*tc, H, W, C]

                    # Reshape to [batch, tc, H, W, C]
                    actual_batch_size = batch_end - batch_start
                    batch_frames = batch_frames.reshape(
                        actual_batch_size, vae_temporal_compression, target_height, target_width, 3
                    )

                    # Encode
                    batch_latents = np.array(encode_batch_jit(jnp.array(batch_frames)))  # [batch, C, 1, H, W]
                    all_latents.append(batch_latents)

                # Concatenate all latents [total_latent_frames, C, 1, H, W]
                all_latents = np.concatenate(all_latents, axis=0)
                # Reshape to [C, T_lat, H, W]
                all_latents = all_latents[:, :, 0, :, :]  # Remove temporal dim (should be 1 per chunk)
                latents_dict[image_key] = all_latents.transpose(1, 0, 2, 3)  # [C, T_lat, H, W]

            # Store with the RLDS episode_index feature (ground-truth identifier)
            shard_writer.write_episode(
                episode_index=rlds_episode_index,
                num_steps=num_steps,
                latents_dict=latents_dict,
            )

        shard_writer.finalize()
        total_written += shard_writer.episode_count

        # Record metadata for this shard
        shard_metadata[shard_idx] = {
            "episode_count": shard_writer.episode_count,
            "num_bytes": shard_writer.total_bytes,
        }

    # Save manifest at worker root (for merge to find)
    latent_store.save_manifest(manifest, str(worker_dir))

    # Save shard metadata for merge step
    shard_metadata_path = worker_dir / "shard_metadata.json"
    with shard_metadata_path.open("w") as f:
        json.dump(shard_metadata, f, indent=2)

    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"Worker {args.worker_id}: done. Wrote {total_written} episodes across {len(shard_metadata)} shards.")


# =============================================================================
# Launch implementation
# =============================================================================


def parse_partition_split(partition_split: str, num_workers: int) -> list[str]:
    """Parse partition_split string into per-worker partition assignments.

    Args:
        partition_split: Format 'partition1:count1,partition2:count2,...'
        num_workers: Total number of workers (must match sum of counts)

    Returns:
        List of partition names, one per worker_id.
    """
    assignments: list[str] = []
    for raw_part in partition_split.split(","):
        part = raw_part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid partition_split format: '{part}'. Expected 'partition:count'.")
        partition, count_str = part.split(":", 1)
        count = int(count_str)
        assignments.extend([partition] * count)

    if len(assignments) != num_workers:
        raise ValueError(
            f"partition_split counts sum to {len(assignments)}, but num_workers={num_workers}. These must match."
        )
    return assignments


def run_launch(args: LaunchArgs) -> None:
    """Submit N sbatch worker jobs from the cluster head node."""
    # Parse partition assignments
    if args.partition_split is not None:
        worker_partitions = parse_partition_split(args.partition_split, args.num_workers)
        logger.info(f"Partition split: {args.partition_split}")
    else:
        worker_partitions = [args.partition] * args.num_workers

    if args.total_episodes is not None:
        total_episodes = args.total_episodes
    else:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        _, data_config, dataset_cfg, _ = resolve_config(args.config_name)
        total_episodes = get_total_episodes(data_config.rlds_data_dir, dataset_cfg, args.split)

    ranges = compute_episode_ranges(total_episodes, args.num_workers, args.max_episodes)

    logger.info(f"Total source episodes: {total_episodes}")
    for worker_id, (start, count) in enumerate(ranges):
        partition = worker_partitions[worker_id]
        logger.info(f"  Worker {worker_id}: episodes [{start}, {start + count}) -> {partition}")

    log_dir = epath.Path(args.log_dir_base).expanduser() / args.config_name
    if args.ssh_host:
        subprocess.run(["ssh", args.ssh_host, f"mkdir -p {shlex.quote(str(log_dir))}"], check=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    def _submit_sbatch(cmd: list[str]) -> str:
        if args.ssh_host:
            remote_cmd = " ".join(shlex.quote(arg) for arg in cmd)
            result = subprocess.run(["ssh", args.ssh_host, remote_cmd], capture_output=True, text=True, check=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().split(";")[0]

    def _make_wrap_cmd(inner_cmd: str) -> str:
        """Create bash wrapper with SSL cert setup for cluster execution."""
        return (
            f'bash -lc "'
            f"source ~/.bashrc && "
            f"export CURL_CA_BUNDLE=\\$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || echo /etc/ssl/certs/ca-bundle.crt) && "
            f"export REQUESTS_CA_BUNDLE=\\$CURL_CA_BUNDLE && "
            f"export SSL_CERT_FILE=\\$CURL_CA_BUNDLE && "
            f"cd ~/projects/AIRe/robocoin/batch_value_learning && {inner_cmd}"
            f'"'
        )

    extra_worker_args = [
        "--encode-batch-size",
        str(args.encode_batch_size),
        "--split",
        args.split,
    ]
    if args.max_episodes is not None:
        extra_worker_args += ["--max-episodes", str(args.max_episodes)]
    if args.target_height is not None:
        extra_worker_args += ["--target-height", str(args.target_height)]
    if args.target_width is not None:
        extra_worker_args += ["--target-width", str(args.target_width)]
    if args.target_fps is not None:
        extra_worker_args += ["--target-fps", str(args.target_fps)]
    if args.source_fps is not None:
        extra_worker_args += ["--source-fps", str(args.source_fps)]

    extra_args_str = " ".join(extra_worker_args)

    job_ids = []
    for worker_id in range(args.num_workers):
        worker_partition = worker_partitions[worker_id]
        worker_cmd = (
            f"{args.uv_bin} run --no-sync scripts/compute_latent_store.py worker"
            f" --config-name {args.config_name}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
            f" --worker-id {worker_id}"
        )
        if extra_args_str:
            worker_cmd += f" {extra_args_str}"

        wrap_cmd = _make_wrap_cmd(worker_cmd)
        log_pattern = str(log_dir / f"latent_store_worker_{worker_id}.log")

        sbatch_cmd = [
            "sbatch",
            "--parsable",
            "-p",
            worker_partition,
            "--mem",
            args.mem,
            "--gres",
            args.gres,
            "--time",
            args.time_limit,
            "--output",
            log_pattern,
            f"--job-name=latent_w{worker_id}",
            "--wrap",
            wrap_cmd,
        ]

        if args.dry_run:
            logger.info(f"[DRY RUN] Worker {worker_id}: {' '.join(sbatch_cmd)}")
            job_ids.append(f"DRY_{worker_id}")
        else:
            job_id = _submit_sbatch(sbatch_cmd)
            job_ids.append(job_id)
            logger.info(f"Worker {worker_id}: submitted job {job_id} (log: {log_pattern})")

    if args.auto_merge and not args.dry_run:
        merge_cmd = (
            f"{args.uv_bin} run --no-sync scripts/compute_latent_store.py merge"
            f" --config-name {args.config_name}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
            f" --split {args.split}"
        )

        merge_wrap = _make_wrap_cmd(merge_cmd)
        dep_str = ":".join(job_ids)
        merge_log_pattern = str(log_dir / "latent_store_merge.log")

        # Use first partition from split (typically the more reliable one) or fallback to args.partition
        merge_partition = worker_partitions[0]
        merge_sbatch_cmd = [
            "sbatch",
            "--parsable",
            "-p",
            merge_partition,
            "--mem",
            "32GB",
            "--time",
            "04:00:00",
            "--output",
            merge_log_pattern,
            "--job-name=latent_merge",
            f"--dependency=afterok:{dep_str}",
            "--wrap",
            merge_wrap,
        ]
        merge_job_id = _submit_sbatch(merge_sbatch_cmd)
        logger.info(f"Merge job submitted: {merge_job_id} (depends on workers: {dep_str})")

    logger.info("All jobs submitted.")
    if not args.auto_merge:
        logger.info(
            "To merge after all workers finish, run:\n"
            f"  uv run scripts/compute_latent_store.py merge"
            f" --config-name {args.config_name}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
        )


# =============================================================================
# Merge implementation
# =============================================================================


def run_merge(args: MergeArgs) -> None:
    """Merge per-worker shard outputs into a single latent store TFDS dataset.

    Each worker creates shard files that EXACTLY match RLDS shard indices. This function:
    1. Reads shard_metadata.json from each worker to get per-shard episode counts and bytes
    2. Copies shard files to merged location PRESERVING their original indices
    3. Creates dataset_info.json with shard_lengths in shard index order

    This maintains 1:1 correspondence with RLDS shards for deterministic joining.
    """
    import concurrent.futures

    import tensorflow_datasets as tfds

    import openpi.training.latent_store as latent_store

    output_dir = epath.Path(args.output_dir)
    source_dir = epath.Path(args.source_dir) if args.source_dir else output_dir
    merged_dataset_dir = output_dir / latent_store.LATENT_STORE_DATASET_NAME / latent_store.LATENT_STORE_VERSION

    if source_dir != output_dir:
        logger.info(f"Reading workers from: {source_dir}")
        logger.info(f"Writing merged output to: {output_dir}")

    if merged_dataset_dir.exists():
        if args.overwrite:
            logger.info(f"Removing existing merged output: {merged_dataset_dir}")
            merged_dataset_dir.rmtree()
        else:
            raise FileExistsError(f"Merged output already exists: {merged_dataset_dir}. Use --overwrite.")

    # Verify all workers are done
    missing_workers = []
    for worker_id in range(args.num_workers):
        done_marker = source_dir / "_workers" / f"worker_{worker_id}" / "_DONE"
        if not done_marker.exists():
            missing_workers.append(worker_id)

    if missing_workers:
        raise RuntimeError(
            f"Missing _DONE markers for workers: {missing_workers}. "
            "Not all workers have completed. Wait for them or re-run failed workers."
        )

    # Collect shard metadata from all workers
    # Each worker creates shards with indices matching RLDS shards (round-robin assignment)
    # shard_data: {shard_idx: {"episode_count": N, "num_bytes": M, "source_path": Path}}
    shard_data: dict[int, dict] = {}
    reference_manifest: latent_store.LatentStoreManifest | None = None

    for worker_id in range(args.num_workers):
        worker_dir = source_dir / "_workers" / f"worker_{worker_id}"
        worker_dataset_dir = worker_dir / latent_store.LATENT_STORE_DATASET_NAME / latent_store.LATENT_STORE_VERSION

        # Load manifest if not already loaded
        if reference_manifest is None:
            manifest_path = worker_dir / "manifest.json"
            if manifest_path.exists():
                reference_manifest = latent_store.load_manifest(str(worker_dir))

        # Load shard metadata
        metadata_path = worker_dir / "shard_metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Worker {worker_id}: shard_metadata.json not found, skipping.")
            continue

        with metadata_path.open("r") as f:
            worker_metadata = json.load(f)

        for shard_idx_str, info in worker_metadata.items():
            shard_idx = int(shard_idx_str)
            if shard_idx in shard_data:
                raise RuntimeError(f"Duplicate shard {shard_idx} from multiple workers")

            # Find the shard file (should be at worker_dataset_dir with matching index)
            shard_path = worker_dataset_dir / latent_store.get_shard_filename(args.split, shard_idx)
            if not shard_path.exists():
                raise FileNotFoundError(f"Expected shard file not found: {shard_path}")

            shard_data[shard_idx] = {
                "episode_count": info["episode_count"],
                "num_bytes": info["num_bytes"],
                "source_path": shard_path,
            }

        logger.info(f"Worker {worker_id}: {len(worker_metadata)} shards")

    if not shard_data:
        raise RuntimeError("No shard data found across any workers.")

    if reference_manifest is None:
        raise RuntimeError("Could not load manifest from any worker.")

    # Sort shards by index to build shard_lengths array in correct order
    sorted_shard_indices = sorted(shard_data.keys())
    num_shards = len(sorted_shard_indices)
    max_shard_idx = max(sorted_shard_indices)

    # Verify we have all shards (no gaps)
    if sorted_shard_indices != list(range(max_shard_idx + 1)):
        missing = set(range(max_shard_idx + 1)) - set(sorted_shard_indices)
        raise RuntimeError(f"Missing shards: {sorted(missing)}. Cannot create valid TFDS dataset.")

    # Build shard_lengths array (in shard index order)
    shard_lengths = [shard_data[i]["episode_count"] for i in sorted_shard_indices]
    total_num_bytes = sum(shard_data[i]["num_bytes"] for i in sorted_shard_indices)
    total_episodes = sum(shard_lengths)

    logger.info(f"Merge plan: {total_episodes} episodes in {num_shards} shards.")

    # Create merged directory and copy shards in parallel (preserving indices)
    merged_dataset_dir.mkdir(parents=True, exist_ok=True)

    def _copy_shard(shard_idx: int, source_path: epath.Path) -> None:
        # Preserve the exact shard index in the filename
        dest_path = merged_dataset_dir / latent_store.get_shard_filename(args.split, shard_idx)
        logger.info(f"Copying shard {shard_idx}: {source_path} -> {dest_path}")
        source_path.copy(dest_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_copy_shard, shard_idx, shard_data[shard_idx]["source_path"])
            for shard_idx in sorted_shard_indices
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # Create TFDS feature spec from manifest
    features = latent_store.get_latent_store_tfds_feature_spec(reference_manifest)

    # Write dataset info and split metadata
    dataset_name = latent_store.LATENT_STORE_DATASET_NAME

    filename_template = tfds.core.ShardedFileTemplate(
        dataset_name=dataset_name,
        split=args.split,
        filetype_suffix="tfrecord",
        data_dir=str(merged_dataset_dir),
        template="{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
    )

    split_info = tfds.core.SplitInfo(
        name=args.split,
        shard_lengths=shard_lengths,
        num_bytes=total_num_bytes,
        filename_template=filename_template,
    )

    identity = tfds.core.DatasetIdentity(
        name=dataset_name,
        version=tfds.core.Version(latent_store.LATENT_STORE_VERSION),
        data_dir=str(merged_dataset_dir),
        module_name=__name__,
    )
    merged_info = tfds.core.DatasetInfo(
        builder=identity,
        description=f"Latent store for {reference_manifest.source_dataset_name}",
        features=features,
    )
    merged_info.set_splits(tfds.core.SplitDict([split_info]))
    merged_info.write_to_directory(merged_dataset_dir)

    # Save manifest at store root
    latent_store.save_manifest(reference_manifest, str(output_dir))

    # Verify
    verify_builder = tfds.builder_from_directory(str(merged_dataset_dir))
    verify_count = verify_builder.info.splits[args.split].num_examples
    if verify_count != total_episodes:
        raise RuntimeError(
            f"Verification failed: expected {total_episodes} episodes, got {verify_count} in merged dataset."
        )
    logger.info(f"Merge complete: {verify_count} episodes in {num_shards} shards written to {merged_dataset_dir}")

    if args.cleanup_workers:
        for worker_id in range(args.num_workers):
            worker_root = source_dir / "_workers" / f"worker_{worker_id}"
            if worker_root.exists():
                logger.info(f"Cleaning up worker {worker_id}: {worker_root}")
                worker_root.rmtree()
        workers_dir = source_dir / "_workers"
        try:
            remaining = list(workers_dir.iterdir())
            if not remaining:
                workers_dir.rmtree()
        except Exception:
            pass


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    args = tyro.cli(
        Annotated[
            Annotated[LaunchArgs, tyro.conf.subcommand("launch")]
            | Annotated[WorkerArgs, tyro.conf.subcommand("worker")]
            | Annotated[MergeArgs, tyro.conf.subcommand("merge")],
            tyro.conf.OmitSubcommandPrefixes,
        ],
    )

    if isinstance(args, LaunchArgs):
        run_launch(args)
    elif isinstance(args, WorkerArgs):
        run_worker(args)
    elif isinstance(args, MergeArgs):
        run_merge(args)
    else:
        raise ValueError(f"Unknown args type: {type(args)}")


if __name__ == "__main__":
    main()
