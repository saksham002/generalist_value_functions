#!/usr/bin/env python3
"""TPU wrapper for counterfactual action generation.

Runs the same episode/shard traversal as the GPU worker, but executes sampling in
lockstep across TPU hosts. Each host samples a fixed local slice of the global
sampling batch, host 0 gathers all sampled actions, applies the same output
transforms as the GPU path, and writes the final TFDS shards directly.
"""

from collections import defaultdict
from collections import deque
import dataclasses
import datetime
import logging
import time
from typing import Any

from compute_latent_store import get_rlds_episode_index
from compute_latent_store import resolve_config
from etils import epath
import numpy as np
import tqdm
import tyro

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    output_dir: str
    split: str = "train"
    num_samples: int = 32
    samples_per_batch: int = 32
    stride: int = 1
    max_episodes: int | None = None
    debug_metrics: bool = False
    num_workers: int | None = None
    overwrite: bool = False
    profile_log_dir: str | None = None
    profile_num_batches: int = 1
    profile_skip_batches: int = 0


@dataclasses.dataclass
class PendingRequest:
    step_idx: int
    strided_idx: int
    state: np.ndarray
    transformed: dict[str, Any]
    gt_actions: np.ndarray | None
    action_mask: np.ndarray | None
    encoded_images: Any
    written_samples: int = 0


def _build_policy_prompt(subtask_texts: list[str], first_null_index: int) -> str | None:
    valid_subtask_texts = []
    for text in subtask_texts[:first_null_index]:
        stripped = text.rstrip(". ").strip()
        lowered = stripped.lower()
        if lowered in {"static", "abnormal"}:
            continue
        valid_subtask_texts.append(stripped)
    if not valid_subtask_texts:
        return None
    return ", ".join(valid_subtask_texts)


def _repeat_tree(tree: Any, repeats: int) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree.map(
        lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (repeats, *np.shape(x))),
        tree,
    )


def _concat_tree(trees: list[Any]) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis = 0), *trees)


def _pad_batch_to_size(tree: Any, current_size: int, target_size: int) -> Any:
    import jax
    import jax.numpy as jnp

    if current_size >= target_size:
        return tree
    pad_size = target_size - current_size

    def pad_array(x: Any) -> Any:
        if x is None:
            return None
        pad_width = [(0, pad_size)] + [(0, 0)] * (len(x.shape) - 1)
        return jnp.pad(x, pad_width)

    return jax.tree.map(pad_array, tree)


def _slice_tree_axis0(tree: Any, start: int, end: int) -> Any:
    import jax

    return jax.tree.map(
        lambda x: None if x is None else x[start:end],
        tree,
    )


def _construct_eef_repr_np(action: np.ndarray, eef_action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            eef_action[..., :6],
            action[..., 6:7],
            eef_action[..., 6:12],
            action[..., 13:14],
        ],
        axis = -1,
    ).astype(np.float32)


def _read_existing_shard_metadata(shard_path: epath.Path) -> dict[str, int]:
    import tensorflow as tf

    episode_count = 0
    for _ in tf.data.TFRecordDataset([str(shard_path)]):
        episode_count += 1
    return {
        "episode_count": episode_count,
        "num_bytes": int(shard_path.stat().length),
    }


def _select_policy_subtask(
    step: dict[str, Any], subtask_texts: list[str], fps: int, action_horizon: int, max_subtasks: int
) -> tuple[str | None, int, np.ndarray]:
    first_null_index_val = int(step["first_null_index"])
    if first_null_index_val == 0:
        return None, 0, np.zeros(action_horizon, dtype = np.bool_)

    steps_all = np.asarray(step["steps_to_subtask_end"], dtype = np.int32)
    include_subtasks = np.zeros(max_subtasks, dtype = np.bool_)
    for idx, text in enumerate(subtask_texts[:first_null_index_val]):
        lowered = text.rstrip(". ").strip().lower()
        include_subtasks[idx] = lowered not in {"static", "abnormal"}

    policy_prompt = _build_policy_prompt(subtask_texts, first_null_index_val)
    if policy_prompt is None:
        return None, 0, np.zeros(action_horizon, dtype = np.bool_)

    masked_steps = np.where(include_subtasks, steps_all, np.iinfo(np.int32).max)
    sampled_idx = int(np.argmin(masked_steps))
    selected_steps = int(steps_all[sampled_idx])
    action_mask = np.arange(action_horizon, dtype = np.int32) <= selected_steps
    if fps == 30:
        valid_30fps_actions = 3 * action_horizon // 5
        action_mask &= np.arange(action_horizon, dtype = np.int32) < valid_30fps_actions

    return policy_prompt, sampled_idx, action_mask


def _format_timing(times: dict[str, float]) -> str:
    total_time = sum(times.values())
    if total_time == 0:
        return "no timed work"
    parts = []
    for key, value in sorted(times.items(), key = lambda item: item[1], reverse = True):
        parts.append(f"{key}={value:.2f}s ({100.0 * value / total_time:.1f}%)")
    return ", ".join(parts)


def _emit_progress_log(message: str, *, disable_tqdm: bool) -> None:
    if disable_tqdm:
        print(message, flush = True)
    else:
        tqdm.tqdm.write(message)


def _write_dataset_info(
    output_dir: epath.Path,
    split: str,
    manifest: Any,
    shard_metadata: dict[int, dict[str, int]],
) -> None:
    import tensorflow_datasets as tfds

    import openpi.training.counterfactual_action_store as ca_store

    sorted_shard_indices = sorted(shard_metadata.keys())
    shard_lengths = [shard_metadata[i]["episode_count"] for i in sorted_shard_indices]
    total_num_bytes = sum(shard_metadata[i]["num_bytes"] for i in sorted_shard_indices)
    total_episodes = sum(shard_lengths)
    merged_dataset_dir = output_dir / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / ca_store.VERSION

    features = ca_store.get_counterfactual_action_store_tfds_feature_spec(manifest)
    dataset_name = ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME
    filename_template = tfds.core.ShardedFileTemplate(
        dataset_name = dataset_name,
        split = split,
        filetype_suffix = "tfrecord",
        data_dir = str(merged_dataset_dir),
        template = "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
    )
    split_info = tfds.core.SplitInfo(
        name = split,
        shard_lengths = shard_lengths,
        num_bytes = total_num_bytes,
        filename_template = filename_template,
    )
    identity = tfds.core.DatasetIdentity(
        name = dataset_name,
        version = tfds.core.Version(ca_store.VERSION),
        data_dir = str(merged_dataset_dir),
        module_name = __name__,
    )
    merged_info = tfds.core.DatasetInfo(
        builder = identity,
        description = f"Counterfactual action store for {manifest.source_dataset_name}",
        features = features,
    )
    merged_info.set_splits(tfds.core.SplitDict([split_info]))
    merged_info.write_to_directory(merged_dataset_dir)

    verify_builder = tfds.builder_from_directory(str(merged_dataset_dir))
    verify_count = verify_builder.info.splits[split].num_examples
    if verify_count != total_episodes:
        raise RuntimeError(
            f"Verification failed: expected {total_episodes} episodes, got {verify_count} in merged dataset."
        )


def main() -> int:
    from flax import nnx
    import jax
    from jax.experimental import multihost_utils
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import tensorflow as tf
    import tensorflow_datasets as tfds

    import openpi.models.model as _model
    import openpi.policies.policy as _policy
    from openpi.robocoin_utils.utils import RLDS_TO_STANDARD_CAMERA_MAP
    from openpi.robocoin_utils.utils import extract_embodiment
    from openpi.shared import nnx_utils
    import openpi.shared.nnx_utils as _nnx_utils
    from openpi.training import checkpoints as _checkpoints
    import openpi.training.counterfactual_action_store as ca_store
    from openpi.training.time_utils import Timer
    import openpi.transforms as _transforms

    args = tyro.cli(Args)

    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required.")

    jax.distributed.initialize()
    process_index = jax.process_index()
    process_count = jax.process_count()

    if args.num_workers is not None and args.num_workers != process_count:
        raise ValueError(
            f"--num-workers={args.num_workers} does not match TPU host count {process_count}."
        )
    if args.samples_per_batch % process_count != 0:
        raise ValueError(
            f"samples_per_batch={args.samples_per_batch} must be divisible by process_count={process_count}."
        )

    local_samples_per_batch = args.samples_per_batch // process_count
    if local_samples_per_batch <= 0:
        raise ValueError(
            f"Per-host sampling batch size must be positive, got {local_samples_per_batch}."
        )
    assert local_samples_per_batch % args.num_samples == 0, (
        f"local_samples_per_batch={local_samples_per_batch} must be divisible by "
        f"num_samples={args.num_samples}. Increase --samples-per-batch to at least "
        f"{args.num_samples * process_count}."
    )

    tf.config.set_visible_devices([], "GPU")

    config, data_config, dataset_cfg, _ = resolve_config(args.config_name)
    source_builder = tfds.builder(dataset_cfg.name, data_dir = data_config.rlds_data_dir, version = dataset_cfg.version)
    split_info = source_builder.info.splits[args.split]
    total_episodes = split_info.num_examples
    shard_lengths = split_info.shard_lengths
    shard_info = []
    offset = 0
    for length in shard_lengths:
        shard_info.append((offset, length))
        offset += length
    num_shards = len(shard_info)

    logger.info(
        "Initialized TPU counterfactual run: process %d/%d, local_samples_per_batch=%d, num_shards=%d",
        process_index,
        process_count,
        local_samples_per_batch,
        num_shards,
    )

    policy_model_config = config.policy if config.policy is not None else config.model
    checkpoint_dir_path = epath.Path(args.checkpoint_dir)

    logger.info("Loading policy from %s", args.checkpoint_dir)
    all_params = _model.restore_params(checkpoint_dir_path / "params", dtype = jnp.bfloat16)
    if "policy" in all_params:
        policy_params = all_params["policy"]
        if "params" in policy_params and len(policy_params) == 1:
            policy_params = policy_params["params"]
        logger.info("Extracted policy params with keys: %s", list(policy_params.keys())[:10])
    else:
        policy_params = all_params

    model_shape = nnx.eval_shape(policy_model_config.create, jax.random.key(0))
    graphdef, state = nnx.split(model_shape)
    policy_params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), policy_params)
    _nnx_utils.replace_state_from_pure_dict_numeric_key_compat(state, policy_params)
    model = nnx.merge(graphdef, state)
    data_config_for_policy = config.data.create(config.assets_dirs, policy_model_config)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir_path / "assets", data_config_for_policy.asset_id)

    policy = _policy.Policy(
        model,
        transforms = [
            _transforms.InjectDefaultPrompt(None),
            *data_config_for_policy.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles = data_config_for_policy.use_quantile_norm),
            *(
                [_transforms.Clip(data_config_for_policy.clip_normalized_bounds)]
                if data_config_for_policy.clip_normalized_bounds is not None
                else []
            ),
            *data_config_for_policy.model_transforms.inputs,
        ],
        output_transforms = [
            *data_config_for_policy.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles = data_config_for_policy.use_quantile_norm),
            *data_config_for_policy.data_transforms.outputs,
        ],
        metadata = config.policy_metadata,
    )

    model = policy._model  # noqa: SLF001
    input_transform = policy._input_transform  # noqa: SLF001
    output_transform = policy._output_transform  # noqa: SLF001
    sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
    rng = policy._rng  # noqa: SLF001

    if hasattr(model, "config") and hasattr(model.config, "guidance"):
        object.__setattr__(model.config, "guidance", 0.0)
        logger.info("Disabled classifier-free guidance for sampling (guidance=0.0).")

    sample_actions_jit = nnx_utils.module_jit(model.sample_actions)
    has_prefix_cache = hasattr(model, "compute_prefix_cache")
    compute_prefix_cache_jit = nnx_utils.module_jit(model.compute_prefix_cache) if has_prefix_cache else None
    encode_images_jit = nnx_utils.module_jit(model.encode_images) if hasattr(model, "encode_images") else None
    profiled_batches = 0
    seen_sampling_batches = 0

    action_horizon = policy_model_config.action_horizon
    action_dim = policy_model_config.action_dim
    max_subtasks = 5

    manifest = ca_store.CounterfactualActionStoreManifest(
        version = "1.0",
        source_rlds_data_dir = data_config.rlds_data_dir,
        source_dataset_name = dataset_cfg.name,
        source_dataset_version = dataset_cfg.version,
        source_num_episodes = total_episodes,
        num_samples = args.num_samples,
        action_dim = action_dim,
        action_horizon = action_horizon,
        stride = args.stride,
        policy_config_name = args.config_name,
        policy_checkpoint_dir = args.checkpoint_dir,
        created_at = datetime.datetime.now(datetime.UTC).isoformat(),
    )

    output_dir = epath.Path(args.output_dir)
    shard_metadata: dict[int, dict[str, int]] = {}
    worker_times = defaultdict(float)
    timed_episode_count = 0

    for shard_idx in range(num_shards):
        start_pos, num_episodes = shard_info[shard_idx]

        if args.max_episodes is not None:
            global_end = min(args.max_episodes, total_episodes)
            shard_end = start_pos + num_episodes
            if start_pos >= global_end:
                multihost_utils.sync_global_devices(f"counterfactual_skip_past_limit_{shard_idx}")
                continue
            if shard_end > global_end:
                num_episodes = global_end - start_pos

        split_spec = f"{args.split}[{start_pos}:{start_pos + num_episodes}]"
        dataset = source_builder.as_dataset(split = split_spec)

        shard_path = (
            output_dir
            / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME
            / ca_store.VERSION
            / ca_store.get_shard_filename(args.split, shard_idx)
        )
        skip_shard = False
        existing_metadata = None
        if shard_path.exists() and not args.overwrite:
            existing_metadata = _read_existing_shard_metadata(shard_path)
            if existing_metadata["episode_count"] == num_episodes:
                skip_shard = True
                if process_index == 0:
                    shard_metadata[shard_idx] = existing_metadata
                    logger.info(
                        "Shard %d: skipping existing shard %s (%d/%d episodes, %d bytes)",
                        shard_idx,
                        shard_path,
                        existing_metadata["episode_count"],
                        num_episodes,
                        existing_metadata["num_bytes"],
                    )
            elif process_index == 0:
                logger.warning(
                    "Shard %d: existing shard %s has %d episodes, expected %d; recomputing shard.",
                    shard_idx,
                    shard_path,
                    existing_metadata["episode_count"],
                    num_episodes,
                )
        elif shard_path.exists() and args.overwrite and process_index == 0:
            logger.info("Shard %d: overwrite enabled, recomputing existing shard %s", shard_idx, shard_path)

        multihost_utils.sync_global_devices(f"counterfactual_shard_{shard_idx}_skip_check")
        if skip_shard:
            continue

        shard_writer = None
        if process_index == 0:
            shard_writer = ca_store.CounterfactualActionStoreTFDSShardWriter(
                output_dir = str(output_dir),
                shard_idx = shard_idx,
                manifest = manifest,
                split = args.split,
            )

        iterator = tqdm.tqdm(
            dataset,
            total = num_episodes,
            desc = f"TPU Shard {shard_idx}",
            disable = process_index != 0,
        )
        for raw_episode in iterator:
            episode_start_time = time.perf_counter()
            episode_timer = Timer()

            with episode_timer.context("materialize_episode"):
                episode = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in raw_episode.items()}
                episode["steps"] = list(episode["steps"])

            rlds_episode_index = get_rlds_episode_index(episode)
            num_steps = len(episode["steps"])

            repo_id = episode["episode_metadata"]["repo_id"]
            if hasattr(repo_id, "numpy"):
                repo_id = repo_id.numpy()
            embodiment = extract_embodiment(repo_id)
            fps = int(episode["episode_metadata"]["fps"])

            episode_actions = None
            if args.debug_metrics:
                logger.info("Episode %d fps=%d", rlds_episode_index, fps)
                episode_actions = []
                for step in episode["steps"]:
                    step_action = step["action"]
                    if hasattr(step_action, "numpy"):
                        step_action = step_action.numpy()
                    step_action = np.asarray(step_action, dtype = np.float32)
                    if data_config_for_policy.robocoin_use_eef:
                        step_eef_action = step["eef_sim_pose_action"]
                        if hasattr(step_eef_action, "numpy"):
                            step_eef_action = step_eef_action.numpy()
                        step_eef_action = np.asarray(step_eef_action, dtype = np.float32)
                        step_action = _construct_eef_repr_np(step_action, step_eef_action)
                    episode_actions.append(step_action)
                episode_actions = np.asarray(episode_actions, dtype = np.float32)

            num_strided_positions = (num_steps + args.stride - 1) // args.stride
            all_actions = None
            if process_index == 0:
                all_actions = np.zeros(
                    (num_strided_positions, args.num_samples, action_horizon, action_dim),
                    dtype = np.float32,
                )
            episode_valid_frames = 0
            episode_sampling_l1_sum = 0.0
            episode_sampling_mse_sum = 0.0
            episode_sampling_num_valid = 0.0
            episode_cov_trace_sum = 0.0
            episode_cov_trace_num_valid = 0.0

            if any(int(step["first_null_index"]) > 0 for step in episode["steps"]):
                pending_requests: deque[PendingRequest] = deque()
                step_requests: dict[int, list[PendingRequest]] = {}
                step_encoding_inputs: list[tuple[int, dict[str, Any]]] = []
                encode_batch_size = min(args.samples_per_batch, 32)

                def flush_pending_step_encodes(
                    step_encoding_inputs: list[tuple[int, dict[str, Any]]] = step_encoding_inputs,
                    step_requests: dict[int, list[PendingRequest]] = step_requests,
                    episode_timer: Timer = episode_timer,
                    encode_batch_size: int = encode_batch_size,
                ) -> None:
                    if not step_encoding_inputs:
                        return

                    actual_batch_size = len(step_encoding_inputs)
                    batched_inputs = _concat_tree(
                        [_repeat_tree(transformed, 1) for _, transformed in step_encoding_inputs]
                    )
                    batched_inputs = _pad_batch_to_size(batched_inputs, actual_batch_size, encode_batch_size)

                    encoded_observation = _model.Observation.from_dict(batched_inputs)
                    with episode_timer.context("encode_images"):
                        batch_encoded_images = encode_images_jit(encoded_observation.images)
                        batch_encoded_images = jax.block_until_ready(batch_encoded_images)
                    batch_encoded_images = batch_encoded_images[:actual_batch_size]

                    for encoded_idx, (encoded_step_idx, _) in enumerate(step_encoding_inputs):
                        step_encoded_images = batch_encoded_images[encoded_idx : encoded_idx + 1]
                        for request in step_requests[encoded_step_idx]:
                            request.encoded_images = step_encoded_images

                    step_encoding_inputs.clear()

                for strided_idx, step_idx in enumerate(range(0, num_steps, args.stride)):
                    step = episode["steps"][step_idx]

                    subtask_texts = []
                    for si in range(1, max_subtasks + 1):
                        text = step[f"subtask_{si}"]
                        if hasattr(text, "numpy"):
                            text = text.numpy()
                        if isinstance(text, bytes):
                            text = text.decode("utf-8")
                        subtask_texts.append(text)

                    policy_prompt, _, action_mask = _select_policy_subtask(
                        step,
                        subtask_texts,
                        fps,
                        action_horizon,
                        max_subtasks,
                    )
                    if policy_prompt is None:
                        continue

                    episode_valid_frames += 1

                    step_state = step["observation/state"]
                    if hasattr(step_state, "numpy"):
                        step_state = step_state.numpy()
                    step_state = np.array(step_state, dtype = np.float32)

                    decoded_images = {}
                    with episode_timer.context("image_decode"):
                        for cam_key in (
                            "observation/image/cam_0",
                            "observation/image/cam_1",
                            "observation/image/cam_2",
                        ):
                            img_data = step[cam_key]
                            if hasattr(img_data, "numpy"):
                                img_data = img_data.numpy()
                            cam_name = cam_key.split("/")[-1]
                            standard_cam_name = RLDS_TO_STANDARD_CAMERA_MAP[cam_name]
                            decoded_images[standard_cam_name] = np.asarray(
                                tf.io.decode_image(img_data, expand_animations = False, dtype = tf.uint8).numpy()
                            )

                    obs_dict: dict[str, Any] = {
                        "image": dict(decoded_images),
                        "state": step_state,
                        "prompt": policy_prompt,
                        "embodiment": embodiment,
                    }

                    transform_with_actions = dict(obs_dict)
                    transform_with_actions["action_mask"] = action_mask
                    if args.debug_metrics:
                        gt_action_indices = np.minimum(
                            step_idx + np.arange(action_horizon, dtype = np.int32),
                            num_steps - 1,
                        )
                        gt_actions = episode_actions[gt_action_indices].copy()
                        if data_config.rlds_kwargs["use_chunk_wise_delta"]:
                            gt_actions = gt_actions - gt_actions[:1, :]
                        transform_with_actions["actions"] = gt_actions

                    with episode_timer.context("input_transform"):
                        transformed = input_transform(transform_with_actions)

                    gt_actions_transformed = None
                    action_mask_transformed = np.asarray(transformed["action_mask"], dtype = np.bool_)
                    if args.debug_metrics:
                        gt_actions_transformed = np.asarray(transformed["actions"], dtype = np.float32)

                    transformed = {
                        k: v
                        for k, v in transformed.items()
                        if not isinstance(v, str) and k not in ("embodiment", "actions")
                    }

                    first_transformed_for_step = transformed
                    if encode_images_jit is not None:
                        transformed = {k: v for k, v in transformed.items() if k != "image"}

                    request = PendingRequest(
                        step_idx = step_idx,
                        strided_idx = strided_idx,
                        state = step_state,
                        transformed = transformed,
                        gt_actions = gt_actions_transformed,
                        action_mask = action_mask_transformed,
                        encoded_images = None,
                    )
                    pending_requests.append(request)
                    step_requests.setdefault(step_idx, []).append(request)

                    if encode_images_jit is not None:
                        step_encoding_inputs.append((step_idx, first_transformed_for_step))
                        if len(step_encoding_inputs) >= encode_batch_size:
                            flush_pending_step_encodes()

                if encode_images_jit is not None:
                    flush_pending_step_encodes()

                while pending_requests:
                    batch_requests: list[tuple[PendingRequest, int]] = []
                    remaining_capacity = args.samples_per_batch

                    while pending_requests and remaining_capacity > 0:
                        request = pending_requests.popleft()
                        num_to_take = min(args.num_samples - request.written_samples, remaining_capacity)
                        batch_requests.append((request, num_to_take))
                        request.written_samples += num_to_take
                        remaining_capacity -= num_to_take
                        if request.written_samples < args.num_samples:
                            pending_requests.appendleft(request)

                    actual_batch_size = sum(n for _, n in batch_requests)
                    batched_inputs = _concat_tree(
                        [_repeat_tree(request.transformed, n) for request, n in batch_requests]
                    )
                    batched_inputs = _pad_batch_to_size(batched_inputs, actual_batch_size, args.samples_per_batch)
                    row_ids = None
                    if args.debug_metrics:
                        row_ids = np.full(args.samples_per_batch, -1, dtype = np.int32)
                        row_ids[:actual_batch_size] = np.arange(actual_batch_size, dtype = np.int32)

                    # Compute prefix KV cache from unique observations, then repeat to match sample counts.
                    local_prefix_cache = None
                    if compute_prefix_cache_jit is not None:
                        unique_batch_size = args.samples_per_batch // args.num_samples
                        local_unique_batch_size = local_samples_per_batch // args.num_samples
                        unique_inputs = _concat_tree(
                            [_repeat_tree(request.transformed, 1) for request, _ in batch_requests]
                        )
                        unique_inputs = _pad_batch_to_size(unique_inputs, len(batch_requests), unique_batch_size)
                        unique_local_start = process_index * local_unique_batch_size
                        unique_local_end = unique_local_start + local_unique_batch_size
                        local_unique_inputs = _slice_tree_axis0(unique_inputs, unique_local_start, unique_local_end)
                        local_unique_obs = _model.Observation.from_dict(local_unique_inputs)
                        local_raw_cache = compute_prefix_cache_jit(local_unique_obs)
                        raw_kv_cache, raw_prefix_mask = local_raw_cache
                        repeated_kv_cache = jax.tree.map(
                            lambda x: jnp.repeat(x, args.num_samples, axis = 1),
                            raw_kv_cache,
                        )
                        repeated_prefix_mask = jnp.repeat(raw_prefix_mask, args.num_samples, axis = 0)
                        local_prefix_cache = (repeated_kv_cache, repeated_prefix_mask)

                    local_start = process_index * local_samples_per_batch
                    local_end = local_start + local_samples_per_batch
                    local_inputs = _slice_tree_axis0(batched_inputs, local_start, local_end)

                    observation = _model.Observation.from_dict(local_inputs)
                    transition = _model.wrap_observation_as_transition(observation)

                    rng, sample_rng = jax.random.split(rng)
                    local_sample_rng = jax.random.fold_in(sample_rng, process_index)
                    call_kwargs = sample_kwargs
                    if local_prefix_cache is not None:
                        call_kwargs = {**call_kwargs, "prefix_cache": local_prefix_cache}

                    if batch_requests[0][0].encoded_images is not None:
                        batch_encoded_images = jnp.concatenate(
                            [
                                jnp.broadcast_to(
                                    request.encoded_images,
                                    (n, *request.encoded_images.shape[1:]),
                                )
                                for request, n in batch_requests
                            ],
                            axis = 0,
                        )
                        batch_encoded_images = _pad_batch_to_size(
                            batch_encoded_images,
                            actual_batch_size,
                            args.samples_per_batch,
                        )
                        local_encoded_images = batch_encoded_images[local_start:local_end]
                        call_kwargs = {**call_kwargs, "encoded_images": local_encoded_images}

                    should_profile_batch = (
                        args.profile_log_dir is not None
                        and seen_sampling_batches >= args.profile_skip_batches
                        and profiled_batches < args.profile_num_batches
                    )
                    profile_log_dir = None
                    if args.profile_log_dir is not None:
                        profile_log_dir = str(epath.Path(args.profile_log_dir) / f"process_{process_index}")

                    if should_profile_batch:
                        with (
                            jax.profiler.trace(profile_log_dir, create_perfetto_link = False),
                            jax.profiler.StepTraceAnnotation(
                                "counterfactual_sample_batch",
                                step_num = profiled_batches,
                            ),
                            episode_timer.context("sample_actions"),
                        ):
                            local_actions_out = sample_actions_jit(local_sample_rng, transition, **call_kwargs)
                            local_actions_out = jax.block_until_ready(local_actions_out)
                        if process_index == 0:
                            logger.info(
                                "Wrote JAX trace for profiled batch %d/%d (sampling batch index %d) to %s",
                                profiled_batches + 1,
                                args.profile_num_batches,
                                seen_sampling_batches,
                                profile_log_dir,
                            )
                        profiled_batches += 1
                    else:
                        with episode_timer.context("sample_actions"):
                            local_actions_out = sample_actions_jit(local_sample_rng, transition, **call_kwargs)
                            local_actions_out = jax.block_until_ready(local_actions_out)
                    seen_sampling_batches += 1

                    local_actions_shape = tuple(local_actions_out.shape)
                    expected_local_shape = (local_samples_per_batch, action_horizon, action_dim)
                    if local_actions_shape != expected_local_shape:
                        raise AssertionError(
                            "Unexpected local sampled action shape: "
                            f"got {local_actions_shape}, expected {expected_local_shape}, "
                            f"process_index = {process_index}, actual_batch_size = {actual_batch_size}, "
                            f"local_samples_per_batch = {local_samples_per_batch}"
                        )

                    with episode_timer.context("sample_actions_gather"):
                        local_actions_host = np.asarray(jax.device_get(local_actions_out))
                        gathered_actions = multihost_utils.process_allgather(local_actions_host, tiled = True)
                        gathered_row_ids = None
                        if args.debug_metrics:
                            local_row_ids = row_ids[local_start:local_end]
                            gathered_row_ids = multihost_utils.process_allgather(local_row_ids, tiled = True)

                    if process_index == 0:
                        gathered_actions_shape = tuple(gathered_actions.shape)
                        if gathered_actions_shape[0] < actual_batch_size:
                            raise AssertionError(
                                "Gathered sampled action batch is smaller than expected: "
                                f"got {gathered_actions_shape}, expected first dim >= {actual_batch_size}, "
                                f"local_samples_per_batch = {local_samples_per_batch}, process_count = {process_count}"
                            )
                        gathered_actions = gathered_actions[:actual_batch_size]
                        actions_np = np.asarray(gathered_actions)
                        if args.debug_metrics:
                            gathered_row_ids = np.asarray(gathered_row_ids[:actual_batch_size])
                            expected_row_ids = np.arange(actual_batch_size, dtype = np.int32)
                            mismatch = gathered_row_ids != expected_row_ids
                            mismatch_sum = int(np.sum(mismatch))
                            logger.info(
                                "TPU shard %d episode_index=%d sampling batch %d row-order mismatch_sum=%d",
                                shard_idx,
                                rlds_episode_index,
                                seen_sampling_batches,
                                mismatch_sum,
                            )
                            if mismatch_sum != 0:
                                raise AssertionError(
                                    "Gathered action row order mismatch: "
                                    f"mismatch_sum = {mismatch_sum}, actual_batch_size = {actual_batch_size}"
                                )

                        batch_states = np.concatenate(
                            [
                                np.broadcast_to(request.state[None], (n, *request.state.shape))
                                for request, n in batch_requests
                            ],
                            axis = 0,
                        )
                        with episode_timer.context("output_transform"):
                            transformed_outputs = output_transform(
                                {
                                    "embodiment": embodiment,
                                    "state": batch_states,
                                    "actions": actions_np,
                                    "next_state": batch_states,
                                    "next_actions": actions_np,
                                }
                            )
                        output_actions = transformed_outputs["actions"]

                        offset = 0
                        for request, n in batch_requests:
                            request_actions = actions_np[offset : offset + n]
                            if args.debug_metrics:
                                gt_actions_broadcast = np.broadcast_to(
                                    request.gt_actions[None, ...],
                                    request_actions.shape,
                                )
                                abs_diff = np.abs(request_actions - gt_actions_broadcast)
                                sq_diff = np.square(request_actions - gt_actions_broadcast)

                                if getattr(model, "action_dim_mask", None) is not None:
                                    dim_mask = np.asarray(model.action_dim_mask, dtype = np.float32)[None, None, :]
                                    dim_mask_den = max(float(np.sum(dim_mask)), 1.0)
                                    l1_per_step = np.sum(abs_diff * dim_mask, axis = -1) / dim_mask_den
                                    mse_per_step = np.sum(sq_diff * dim_mask, axis = -1) / dim_mask_den
                                else:
                                    l1_per_step = np.mean(abs_diff, axis = -1)
                                    mse_per_step = np.mean(sq_diff, axis = -1)

                                step_mask = request.action_mask.astype(np.float32)[None, :]
                                episode_sampling_l1_sum += float(np.sum(l1_per_step * step_mask))
                                episode_sampling_mse_sum += float(np.sum(mse_per_step * step_mask))
                                episode_sampling_num_valid += float(np.sum(step_mask) * n)

                            centered = request_actions - np.mean(request_actions, axis = 0, keepdims = True)
                            cov_trace_per_timestep = np.sum(np.mean(np.square(centered), axis = 0), axis = -1)
                            episode_cov_trace_sum += float(np.mean(cov_trace_per_timestep))
                            episode_cov_trace_num_valid += 1.0

                            start = request.written_samples - n
                            end = request.written_samples
                            all_actions[request.strided_idx, start:end] = output_actions[offset : offset + n]
                            offset += n

            if process_index == 0:
                with episode_timer.context("write_episode"):
                    shard_writer.write_episode(
                        episode_index = rlds_episode_index,
                        num_steps = num_steps,
                        actions = all_actions,
                    )
                episode_elapsed = time.perf_counter() - episode_start_time
                episode_times = episode_timer.get_total_times(reset = True)
                for key, value in episode_times.items():
                    worker_times[key] += value
                timed_episode_count += 1
                _emit_progress_log(
                    (
                        "TPU shard %d episode_index=%d num_steps=%d valid_frames=%d "
                        "sampling_l1=%.6f sampling_mse=%.6f cov_trace_per_timestep=%.6f "
                        "debug_metrics=%s elapsed=%.2fs"
                    )
                    % (
                        shard_idx,
                        rlds_episode_index,
                        num_steps,
                        episode_valid_frames,
                        episode_sampling_l1_sum / max(episode_sampling_num_valid, 1.0),
                        episode_sampling_mse_sum / max(episode_sampling_num_valid, 1.0),
                        episode_cov_trace_sum / max(episode_cov_trace_num_valid, 1.0),
                        args.debug_metrics,
                        episode_elapsed,
                    ),
                    disable_tqdm = process_index != 0,
                )
                _emit_progress_log(
                    "TPU shard %d episode_index=%d timing: %s"
                    % (
                        shard_idx,
                        rlds_episode_index,
                        _format_timing(episode_times),
                    ),
                    disable_tqdm = process_index != 0,
                )
                logger.info(
                    "TPU shard %d episode_index=%d num_steps=%d valid_frames=%d "
                    "sampling_l1=%.6f sampling_mse=%.6f cov_trace_per_timestep=%.6f "
                    "debug_metrics=%s elapsed=%.2fs",
                    shard_idx,
                    rlds_episode_index,
                    num_steps,
                    episode_valid_frames,
                    episode_sampling_l1_sum / max(episode_sampling_num_valid, 1.0),
                    episode_sampling_mse_sum / max(episode_sampling_num_valid, 1.0),
                    episode_cov_trace_sum / max(episode_cov_trace_num_valid, 1.0),
                    args.debug_metrics,
                    episode_elapsed,
                )

            multihost_utils.sync_global_devices(f"counterfactual_episode_{shard_idx}_{rlds_episode_index}")

        if process_index == 0:
            shard_writer.finalize()
            shard_metadata[shard_idx] = {
                "episode_count": shard_writer.episode_count,
                "num_bytes": shard_writer.total_bytes,
            }

        multihost_utils.sync_global_devices(f"counterfactual_shard_{shard_idx}_done")

    if process_index == 0:
        ca_store.save_manifest(manifest, str(output_dir))
        _write_dataset_info(output_dir, args.split, manifest, shard_metadata)
        if timed_episode_count > 0:
            average_times = {key: value / timed_episode_count for key, value in worker_times.items()}
            logger.info(
                "TPU timing totals across %d episodes: %s",
                timed_episode_count,
                _format_timing(dict(worker_times)),
            )
            logger.info(
                "TPU timing averages per episode: %s",
                _format_timing(average_times),
            )
        logger.info("TPU counterfactual generation complete. Wrote %d shards.", len(shard_metadata))

    multihost_utils.sync_global_devices("counterfactual_tpu_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
