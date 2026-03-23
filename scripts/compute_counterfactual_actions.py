"""Distributed GPU computation of counterfactual action stores for RLDS datasets.

Pre-computes N action samples from a BC-trained policy for every state in the dataset,
using the same prompt construction as policy training. Follows the same shard-mirroring
TFDS pattern as the latent store.

Three tyro subcommands:
  launch  - Submit N sbatch worker jobs from the cluster head node.
  worker  - Process a contiguous chunk of episodes on a single GPU.
  merge   - Combine per-worker outputs into a single counterfactual action store.

Usage:
    # 1. Launch N workers
    uv run scripts/compute_counterfactual_actions.py launch \
        --config-name cosmos_robocoin_bc_flow \
        --checkpoint-dir gs://path/to/checkpoint \
        --output-dir gs://path/to/store \
        --num-workers 16

    # 2. Worker mode (invoked by sbatch, not called directly)
    uv run scripts/compute_counterfactual_actions.py worker \
        --config-name cosmos_robocoin_bc_flow \
        --checkpoint-dir gs://path/to/checkpoint \
        --output-dir gs://path/to/store \
        --num-workers 16 --worker-id 3

    # 3. Merge (after all workers finish)
    uv run scripts/compute_counterfactual_actions.py merge \
        --config-name cosmos_robocoin_bc_flow \
        --checkpoint-dir gs://path/to/checkpoint \
        --output-dir gs://path/to/store \
        --num-workers 16
"""

from collections import defaultdict
from collections import deque
import dataclasses
import datetime
import json
import logging
import shlex
import subprocess
import time
from typing import Annotated, Any

from compute_latent_store import get_rlds_episode_index
from compute_latent_store import get_total_episodes
from compute_latent_store import parse_partition_split
from compute_latent_store import resolve_config
from etils import epath
import numpy as np
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

    config_name: str = "cosmos_robocoin_bc_flow"
    """Config name to resolve RLDS data source and policy."""

    checkpoint_dir: str = ""
    """Path to the trained policy checkpoint (required)."""

    output_dir: str = ""
    """Output counterfactual action store directory (local or GCS path)."""

    split: str = "train"
    """Split to process (e.g., 'train', 'val')."""

    num_workers: int = 8
    """Number of worker jobs to distribute across."""

    num_samples: int = 32
    """Number of action samples to generate per state."""

    stride: int = 1
    """Stride for computing counterfactual actions. stride=5 means compute at steps 0, 5, 10, etc."""

    debug_metrics: bool = False
    """If True, compute extra normalized-space diagnostics such as sampling loss."""


@dataclasses.dataclass
class LaunchArgs(CommonArgs):
    """Launch N sbatch worker jobs from the cluster head node."""

    partition: str = "preempt"
    """SLURM partition."""

    partition_split: str | None = None
    """Split workers across partitions. Format: 'partition1:count1,partition2:count2,...'."""

    time_limit: str = "24:00:00"
    """SLURM time limit."""

    mem: str = "64GB"
    """SLURM memory limit."""

    gres: str = "gpu:L40S:1"
    """SLURM GPU resource spec."""

    max_episodes: int | None = None
    """Limit total episodes to process across all workers."""

    samples_per_batch: int = 32
    """Maximum total sampled actions per GPU forward pass across all observations."""

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

    max_episodes: int | None = None
    """Limit total episodes to process across all workers."""

    samples_per_batch: int = 32
    """Maximum total sampled actions per GPU forward pass across all observations."""

    profile_log_dir: str | None = None
    """If set, write a JAX trace for the first few sampling batches to this directory."""

    profile_num_batches: int = 1
    """Number of sampling batches to include in the trace when profile_log_dir is set."""

    profile_skip_batches: int = 0
    """Number of initial sampling batches to skip before starting JAX tracing."""


@dataclasses.dataclass
class MergeArgs(CommonArgs):
    """Combine per-worker outputs into a single counterfactual action store."""

    source_dir: str | None = None
    """Directory containing worker outputs. Defaults to output_dir if not specified."""

    overwrite: bool = False
    """Overwrite existing merged output."""

    cleanup_workers: bool = False
    """Delete worker directories after successful merge."""


# =============================================================================
# Worker implementation
# =============================================================================


def run_worker(args: WorkerArgs) -> None:
    """Run a single worker that processes its assigned RLDS shards.

    For each episode, for each step:
    - Construct observation dict and apply input transforms
    - Generate num_samples action samples via the policy model
    - Apply output transforms to unnormalize actions
    - Store as [num_steps, num_samples, action_horizon, action_dim]
    """
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    import tensorflow_datasets as tfds

    import openpi.models.model as _model
    from openpi.robocoin_utils.utils import extract_embodiment
    import openpi.training.counterfactual_action_store as ca_store
    from openpi.training.robocoin_data_loader import RLDS_TO_STANDARD_CAMERA_MAP
    from openpi.training.time_utils import Timer
    from openpi.value_functions.base_value_functions import Transition

    tf.config.set_visible_devices([], "GPU")

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id={args.worker_id} out of range [0, {args.num_workers}).")

    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required.")

    config, data_config, dataset_cfg, _ = resolve_config(args.config_name)

    # Create builder once and reuse for metadata queries and dataset loading
    source_builder = tfds.builder(dataset_cfg.name, data_dir=data_config.rlds_data_dir, version=dataset_cfg.version)
    split_info = source_builder.info.splits[args.split]
    total_episodes = split_info.num_examples
    shard_lengths = split_info.shard_lengths
    shard_info = []
    offset = 0
    for length in shard_lengths:
        shard_info.append((offset, length))
        offset += length
    num_shards = len(shard_info)

    # Distribute shards across workers (round-robin)
    my_shards = [i for i in range(num_shards) if i % args.num_workers == args.worker_id]
    my_episode_count = sum(shard_info[i][1] for i in my_shards)

    logger.info(
        f"Worker {args.worker_id}/{args.num_workers}: assigned {len(my_shards)} shards "
        f"({my_episode_count} episodes total)"
    )

    output_dir = epath.Path(args.output_dir)
    worker_dir = output_dir / "_workers" / f"worker_{args.worker_id}"
    done_marker = worker_dir / "_DONE"

    if done_marker.exists():
        logger.info(f"Worker {args.worker_id}: already completed, skipping.")
        return

    # Load policy. For value function training configs, the checkpoint stores params
    # as {"critic": ..., "policy": ...}. We need to extract the "policy" subtree and
    # load it into the policy model config (config.policy), not the value function
    # model (config.model). We replicate the relevant parts of create_trained_policy
    # here to handle this param extraction.
    policy_model_config = config.policy if config.policy is not None else config.model

    logger.info(f"Loading policy from {args.checkpoint_dir}")

    from openpi.training import checkpoints as _checkpoints

    checkpoint_dir_path = epath.Path(args.checkpoint_dir)
    all_params = _model.restore_params(checkpoint_dir_path / "params", dtype=jnp.bfloat16)

    # Extract BC policy params from the checkpoint. For actor-critic training configs,
    # checkpoint params are structured as {"policy": {"params": {<model keys>}}, "critic": ...}.
    # We need to unwrap to get the raw model params for CosmosFlow.load().
    if "policy" in all_params:
        policy_params = all_params["policy"]
        if "params" in policy_params and len(policy_params) == 1:
            policy_params = policy_params["params"]
        logger.info(f"Extracted policy params with keys: {list(policy_params.keys())[:10]}")
    else:
        policy_params = all_params

    # Load model, bypassing BaseModelConfig.load()'s strict pytree equality check
    # which fails on Orbax's string-serialized integer keys. We use
    # replace_state_from_pure_dict_numeric_key_compat which handles the mismatch.
    from flax import nnx
    import orbax.checkpoint as ocp

    import openpi.shared.nnx_utils as _nnx_utils

    _model_instance = nnx.eval_shape(policy_model_config.create, jax.random.key(0))
    _graphdef, _state = nnx.split(_model_instance)
    policy_params = ocp.transform_utils.intersect_trees(_state.to_pure_dict(), policy_params)
    _nnx_utils.replace_state_from_pure_dict_numeric_key_compat(_state, policy_params)
    model = nnx.merge(_graphdef, _state)
    data_config_for_policy = config.data.create(config.assets_dirs, policy_model_config)
    norm_stats = None
    if hasattr(config.data, "_load_robocoin_norm_stats") and config.data.norm_stats_path is not None:
        try:
            norm_stats = config.data._load_robocoin_norm_stats()
            logger.info(f"Loaded norm stats from config path {config.data.norm_stats_path}")
        except FileNotFoundError:
            logger.info(
                f"Norm stats not found at config path {config.data.norm_stats_path}, "
                "falling back to checkpoint assets."
            )
    if norm_stats is None:
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir_path / "assets", data_config_for_policy.asset_id)

    import openpi.policies.policy as _policy
    import openpi.transforms as _transforms

    policy = _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(None),
            *data_config_for_policy.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config_for_policy.use_quantile_norm),
            *(
                [_transforms.Clip(data_config_for_policy.clip_normalized_bounds)]
                if data_config_for_policy.clip_normalized_bounds is not None
                else []
            ),
            *data_config_for_policy.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config_for_policy.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles = data_config_for_policy.use_quantile_norm),
            *data_config_for_policy.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
    )

    # Extract model and transforms for batched inference.
    # We access private members because the Policy.infer() API only handles single
    # observations, and we need batched inference for efficiency.
    model = policy._model  # noqa: SLF001
    input_transform = policy._input_transform  # noqa: SLF001
    output_transform = policy._output_transform  # noqa: SLF001
    sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
    rng = policy._rng  # noqa: SLF001

    # Disable classifier-free guidance for counterfactual action generation.
    # Set it on the model config so the branch stays static under JIT.
    if hasattr(model, "config") and hasattr(model.config, "guidance"):
        object.__setattr__(model.config, "guidance", 0.0)
        logger.info("Disabled classifier-free guidance for sampling (guidance=0.0).")

    # JIT-compile sample_actions for the model
    from openpi.shared import nnx_utils

    sample_actions_jit = nnx_utils.module_jit(model.sample_actions)
    encode_images_jit = nnx_utils.module_jit(model.encode_images) if hasattr(model, "encode_images") else None
    profiled_batches = 0
    seen_sampling_batches = 0

    @dataclasses.dataclass
    class PendingRequest:
        step_idx: int
        strided_idx: int
        state: np.ndarray
        transformed: dict[str, Any]
        gt_actions: np.ndarray | None
        action_mask: np.ndarray | None
        encoded_images: jax.Array | None
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
        return jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (repeats, *np.shape(x))),
            tree,
        )

    def _concat_tree(trees: list[Any]) -> Any:
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)

    def _pad_batch_to_size(tree: Any, current_size: int, target_size: int) -> Any:
        """Pad arrays in tree along axis 0 to target_size to avoid JIT recompilation."""
        if current_size >= target_size:
            return tree
        pad_size = target_size - current_size

        def pad_array(x: jax.Array | None) -> jax.Array | None:
            if x is None:
                return None
            pad_width = [(0, pad_size)] + [(0, 0)] * (len(x.shape) - 1)
            return jnp.pad(x, pad_width)

        return jax.tree.map(pad_array, tree)

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

    def _select_policy_subtask(
        step: dict[str, Any], subtask_texts: list[str], fps: int
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

    # Get action dimensions from policy model config
    action_horizon = policy_model_config.action_horizon
    action_dim = policy_model_config.action_dim
    max_subtasks = 5

    logger.info(f"Action horizon={action_horizon}, action_dim={action_dim}, num_samples={args.num_samples}")

    # Create manifest
    manifest = ca_store.CounterfactualActionStoreManifest(
        version="1.0",
        source_rlds_data_dir=data_config.rlds_data_dir,
        source_dataset_name=dataset_cfg.name,
        source_dataset_version=dataset_cfg.version,
        source_num_episodes=total_episodes,
        num_samples=args.num_samples,
        action_dim=action_dim,
        action_horizon=action_horizon,
        stride=args.stride,
        policy_config_name=args.config_name,
        policy_checkpoint_dir=args.checkpoint_dir,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )

    total_written = 0
    shard_metadata: dict[int, dict[str, int]] = {}
    worker_times = defaultdict(float)
    timed_episode_count = 0

    def format_timing(times: dict[str, float]) -> str:
        total_time = sum(times.values())
        if total_time == 0:
            return "no timed work"
        parts = []
        for key, value in sorted(times.items(), key=lambda item: item[1], reverse=True):
            parts.append(f"{key}={value:.2f}s ({100.0 * value / total_time:.1f}%)")
        return ", ".join(parts)

    for shard_idx in my_shards:
        start_pos, num_episodes = shard_info[shard_idx]

        # Apply max_episodes limit
        if args.max_episodes is not None:
            global_end = min(args.max_episodes, total_episodes)
            shard_end = start_pos + num_episodes
            if start_pos >= global_end:
                logger.info(f"Shard {shard_idx}: skipping (past max_episodes limit)")
                continue
            if shard_end > global_end:
                num_episodes = global_end - start_pos

        split_spec = f"{args.split}[{start_pos}:{start_pos + num_episodes}]"
        dataset = source_builder.as_dataset(split=split_spec)

        shard_writer = ca_store.CounterfactualActionStoreTFDSShardWriter(
            output_dir=str(worker_dir),
            shard_idx=shard_idx,
            manifest=manifest,
            split=args.split,
        )

        for raw_episode in tqdm.tqdm(dataset, total=num_episodes, desc=f"Worker {args.worker_id} Shard {shard_idx}"):
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

            # Allocate output array (strided dimensions if stride > 1)
            num_strided_positions = (num_steps + args.stride - 1) // args.stride
            all_actions = np.zeros(
                (num_strided_positions, args.num_samples, action_horizon, action_dim),
                dtype=np.float32,
            )
            episode_valid_frames = 0
            episode_sampling_l1_sum = 0.0
            episode_sampling_mse_sum = 0.0
            episode_sampling_num_valid = 0.0
            episode_cov_trace_sum = 0.0
            episode_cov_trace_num_valid = 0.0

            if not any(int(step["first_null_index"]) > 0 for step in episode["steps"]):
                # No valid subtasks for this episode, leave all zeros
                pass
            else:
                pending_requests: deque[PendingRequest] = deque()
                step_requests: dict[int, list[PendingRequest]] = {}
                step_encoding_inputs: list[tuple[int, dict[str, Any]]] = []
                encode_batch_size = min(args.samples_per_batch, 32)

                def flush_pending_step_encodes(
                    step_encoding_inputs: list = step_encoding_inputs,
                    step_requests: dict = step_requests,
                    episode_timer: Timer = episode_timer,
                    encode_batch_size: int = encode_batch_size,
                ) -> None:
                    if not step_encoding_inputs:
                        return

                    actual_batch_size = len(step_encoding_inputs)
                    batched_inputs = _concat_tree(
                        [_repeat_tree(transformed, 1) for _, transformed in step_encoding_inputs]
                    )

                    # Pad to fixed size to avoid JIT recompilation for partial batches
                    batched_inputs = _pad_batch_to_size(batched_inputs, actual_batch_size, encode_batch_size)

                    encoded_observation = _model.Observation.from_dict(batched_inputs)
                    with episode_timer.context("encode_images"):
                        batch_encoded_images = encode_images_jit(encoded_observation.images)
                        batch_encoded_images = jax.block_until_ready(batch_encoded_images)

                    # Slice back to actual size (discard padding)
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

                    policy_prompt, sampled_idx, action_mask = _select_policy_subtask(step, subtask_texts, fps)
                    if policy_prompt is None:
                        continue
                    episode_valid_frames += 1

                    first_transformed_for_step = None

                    # Extract observation data once per step (shared across subtasks)
                    step_state = step["observation/state"]
                    if hasattr(step_state, "numpy"):
                        step_state = step_state.numpy()
                    step_state = np.array(step_state, dtype=np.float32)

                    # Decode images once per step (not per subtask)
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
                    if args.debug_metrics:
                        gt_action_indices = np.minimum(
                            step_idx + np.arange(action_horizon, dtype = np.int32),
                            num_steps - 1,
                        )
                        gt_actions = episode_actions[gt_action_indices].copy()
                        if data_config.rlds_kwargs["use_chunk_wise_delta"]:
                            gt_actions = gt_actions - gt_actions[:1, :]
                        transform_with_actions["actions"] = gt_actions
                        transform_with_actions["action_mask"] = action_mask

                    with episode_timer.context("input_transform"):
                        transformed = input_transform(transform_with_actions)

                    gt_actions_transformed = None
                    action_mask_transformed = None
                    if args.debug_metrics:
                        gt_actions_transformed = np.asarray(transformed["actions"], dtype = np.float32)
                        action_mask_transformed = np.asarray(transformed["action_mask"], dtype = np.bool_)

                    transformed = {
                        k: v
                        for k, v in transformed.items()
                        if not isinstance(v, str) and k not in ("embodiment", "actions")
                    }
                    first_transformed_for_step = transformed
                    if encode_images_jit is not None:
                        transformed = {k: v for k, v in transformed.items() if k != "image"}

                    request = PendingRequest(
                        step_idx=step_idx,
                        strided_idx=strided_idx,
                        state=step_state,
                        transformed=transformed,
                        gt_actions=gt_actions_transformed,
                        action_mask=action_mask_transformed,
                        encoded_images=None,
                    )
                    pending_requests.append(request)
                    step_requests.setdefault(step_idx, []).append(request)

                    if first_transformed_for_step is not None and encode_images_jit is not None:
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

                    # Pad to fixed size to avoid JIT recompilation for partial batches
                    batched_inputs = _pad_batch_to_size(batched_inputs, actual_batch_size, args.samples_per_batch)

                    observation = _model.Observation.from_dict(batched_inputs)
                    transition = _model.wrap_observation_as_transition(observation)

                    rng, sample_rng = jax.random.split(rng)
                    call_kwargs = sample_kwargs
                    if batch_requests[0][0].encoded_images is not None:
                        batch_encoded_images = jnp.concatenate(
                            [
                                jnp.broadcast_to(
                                    request.encoded_images,
                                    (n, *request.encoded_images.shape[1:]),
                                )
                                for request, n in batch_requests
                            ],
                            axis=0,
                        )
                        # Pad encoded images to fixed size as well
                        batch_encoded_images = _pad_batch_to_size(
                            batch_encoded_images, actual_batch_size, args.samples_per_batch
                        )
                        call_kwargs = {**sample_kwargs, "encoded_images": batch_encoded_images}

                    should_profile_batch = (
                        args.profile_log_dir is not None
                        and seen_sampling_batches >= args.profile_skip_batches
                        and profiled_batches < args.profile_num_batches
                    )
                    if should_profile_batch:
                        with (
                            jax.profiler.trace(args.profile_log_dir, create_perfetto_link=False),
                            jax.profiler.StepTraceAnnotation("counterfactual_sample_batch", step_num=profiled_batches),
                            episode_timer.context("sample_actions"),
                        ):
                            actions_out = sample_actions_jit(sample_rng, transition, **call_kwargs)
                            actions_out = jax.block_until_ready(actions_out)
                        logger.info(
                            "Wrote JAX trace for profiled batch %d/%d (sampling batch index %d) to %s",
                            profiled_batches + 1,
                            args.profile_num_batches,
                            seen_sampling_batches,
                            args.profile_log_dir,
                        )
                        profiled_batches += 1
                    else:
                        with episode_timer.context("sample_actions"):
                            actions_out = sample_actions_jit(sample_rng, transition, **call_kwargs)
                            actions_out = jax.block_until_ready(actions_out)
                    seen_sampling_batches += 1

                    # Slice back to actual size (discard padding)
                    actions_out = actions_out[:actual_batch_size]

                    with episode_timer.context("sample_actions_to_host"):
                        actions_np = np.asarray(actions_out)
                    batch_states = np.concatenate(
                        [
                            np.broadcast_to(request.state[None], (n, *request.state.shape))
                            for request, n in batch_requests
                        ],
                        axis=0,
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
                        all_actions[request.strided_idx, start:end] = output_actions[
                            offset : offset + n
                        ]
                        offset += n

            with episode_timer.context("write_episode"):
                shard_writer.write_episode(
                    episode_index=rlds_episode_index,
                    num_steps=num_steps,
                    actions=all_actions,
                )
            episode_elapsed = time.perf_counter() - episode_start_time
            episode_times = episode_timer.get_total_times(reset=True)
            for key, value in episode_times.items():
                worker_times[key] += value
            timed_episode_count += 1
            logger.info(
                "Worker %d shard %d episode_index=%d num_steps=%d valid_frames=%d "
                "sampling_l1=%.6f sampling_mse=%.6f cov_trace_per_timestep=%.6f "
                "debug_metrics=%s elapsed=%.2fs",
                args.worker_id,
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
            logger.info(
                "Worker %d shard %d episode_index=%d timing: %s",
                args.worker_id,
                shard_idx,
                rlds_episode_index,
                format_timing(episode_times),
            )

        shard_writer.finalize()
        total_written += shard_writer.episode_count

        shard_metadata[shard_idx] = {
            "episode_count": shard_writer.episode_count,
            "num_bytes": shard_writer.total_bytes,
        }

    # Save manifest
    ca_store.save_manifest(manifest, str(worker_dir))

    # Save shard metadata
    shard_metadata_path = worker_dir / "shard_metadata.json"
    with shard_metadata_path.open("w") as f:
        json.dump(shard_metadata, f, indent=2)

    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if timed_episode_count > 0:
        average_times = {key: value / timed_episode_count for key, value in worker_times.items()}
        logger.info(
            "Worker %d timing totals across %d episodes: %s",
            args.worker_id,
            timed_episode_count,
            format_timing(dict(worker_times)),
        )
        logger.info(
            "Worker %d timing averages per episode: %s",
            args.worker_id,
            format_timing(average_times),
        )

    logger.info(f"Worker {args.worker_id}: done. Wrote {total_written} episodes across {len(shard_metadata)} shards.")


# =============================================================================
# Launch implementation
# =============================================================================


def run_launch(args: LaunchArgs) -> None:
    """Submit N sbatch worker jobs from the cluster head node."""
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required.")

    if args.partition_split is not None:
        worker_partitions = parse_partition_split(args.partition_split, args.num_workers)
    else:
        worker_partitions = [args.partition] * args.num_workers

    if args.total_episodes is not None:
        total_episodes = args.total_episodes
    else:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        _, data_config, dataset_cfg, _ = resolve_config(args.config_name)
        total_episodes = get_total_episodes(data_config.rlds_data_dir, dataset_cfg, args.split)

    logger.info(f"Total source episodes: {total_episodes}")
    for worker_id in range(args.num_workers):
        logger.info(f"  Worker {worker_id} -> {worker_partitions[worker_id]}")

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
        "--split",
        args.split,
        "--samples-per-batch",
        str(args.samples_per_batch),
        "--num-samples",
        str(args.num_samples),
        "--stride",
        str(args.stride),
    ]
    if args.max_episodes is not None:
        extra_worker_args += ["--max-episodes", str(args.max_episodes)]
    if args.debug_metrics:
        extra_worker_args += ["--debug-metrics"]

    extra_args_str = " ".join(extra_worker_args)

    job_ids = []
    for worker_id in range(args.num_workers):
        worker_partition = worker_partitions[worker_id]
        worker_cmd = (
            f"{args.uv_bin} run --no-sync scripts/compute_counterfactual_actions.py worker"
            f" --config-name {args.config_name}"
            f" --checkpoint-dir {args.checkpoint_dir}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
            f" --worker-id {worker_id}"
        )
        if extra_args_str:
            worker_cmd += f" {extra_args_str}"

        wrap_cmd = _make_wrap_cmd(worker_cmd)
        log_pattern = str(log_dir / f"ca_store_worker_{worker_id}.log")

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
            f"--job-name=ca_w{worker_id}",
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
            f"{args.uv_bin} run --no-sync scripts/compute_counterfactual_actions.py merge"
            f" --config-name {args.config_name}"
            f" --checkpoint-dir {args.checkpoint_dir}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
            f" --split {args.split}"
        )

        merge_wrap = _make_wrap_cmd(merge_cmd)
        dep_str = ":".join(job_ids)
        merge_log_pattern = str(log_dir / "ca_store_merge.log")

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
            "--job-name=ca_merge",
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
            f"  uv run scripts/compute_counterfactual_actions.py merge"
            f" --config-name {args.config_name}"
            f" --checkpoint-dir {args.checkpoint_dir}"
            f" --output-dir {args.output_dir}"
            f" --num-workers {args.num_workers}"
        )


# =============================================================================
# Merge implementation
# =============================================================================


def run_merge(args: MergeArgs) -> None:
    """Merge per-worker shard outputs into a single counterfactual action store TFDS dataset."""
    import concurrent.futures

    import tensorflow_datasets as tfds

    import openpi.training.counterfactual_action_store as ca_store

    output_dir = epath.Path(args.output_dir)
    source_dir = epath.Path(args.source_dir) if args.source_dir else output_dir
    merged_dataset_dir = output_dir / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / ca_store.VERSION

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
        raise RuntimeError(f"Missing _DONE markers for workers: {missing_workers}. Not all workers have completed.")

    # Collect shard metadata from all workers
    shard_data: dict[int, dict] = {}
    reference_manifest: ca_store.CounterfactualActionStoreManifest | None = None

    for worker_id in range(args.num_workers):
        worker_dir = source_dir / "_workers" / f"worker_{worker_id}"
        worker_dataset_dir = worker_dir / ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / ca_store.VERSION

        if reference_manifest is None:
            manifest_path = worker_dir / "manifest.json"
            if manifest_path.exists():
                reference_manifest = ca_store.load_manifest(str(worker_dir))

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

            shard_path = worker_dataset_dir / ca_store.get_shard_filename(args.split, shard_idx)
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

    sorted_shard_indices = sorted(shard_data.keys())
    max_shard_idx = max(sorted_shard_indices)

    if sorted_shard_indices != list(range(max_shard_idx + 1)):
        missing = set(range(max_shard_idx + 1)) - set(sorted_shard_indices)
        raise RuntimeError(f"Missing shards: {sorted(missing)}. Cannot create valid TFDS dataset.")

    shard_lengths = [shard_data[i]["episode_count"] for i in sorted_shard_indices]
    total_num_bytes = sum(shard_data[i]["num_bytes"] for i in sorted_shard_indices)
    total_episodes = sum(shard_lengths)
    num_shards = len(sorted_shard_indices)

    logger.info(f"Merge plan: {total_episodes} episodes in {num_shards} shards.")

    merged_dataset_dir.mkdir(parents=True, exist_ok=True)

    def _copy_shard(shard_idx: int, source_path: epath.Path) -> None:
        dest_path = merged_dataset_dir / ca_store.get_shard_filename(args.split, shard_idx)
        logger.info(f"Copying shard {shard_idx}: {source_path} -> {dest_path}")
        source_path.copy(dest_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_copy_shard, shard_idx, shard_data[shard_idx]["source_path"])
            for shard_idx in sorted_shard_indices
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    features = ca_store.get_counterfactual_action_store_tfds_feature_spec(reference_manifest)

    dataset_name = ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME

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
        version=tfds.core.Version(ca_store.VERSION),
        data_dir=str(merged_dataset_dir),
        module_name=__name__,
    )
    merged_info = tfds.core.DatasetInfo(
        builder=identity,
        description=f"Counterfactual action store for {reference_manifest.source_dataset_name}",
        features=features,
    )
    merged_info.set_splits(tfds.core.SplitDict([split_info]))
    merged_info.write_to_directory(merged_dataset_dir)

    ca_store.save_manifest(reference_manifest, str(output_dir))

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
