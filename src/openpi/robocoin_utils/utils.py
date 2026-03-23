import logging
import os
import pickle

logger = logging.getLogger(__name__)

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.value_functions.base_value_functions as _value_fn
from openpi.models import model as _model
from openpi.value_functions.networks.base_networks import BaseValueNetwork


RLDS_TO_STANDARD_CAMERA_MAP = {
    "cam_0": "base_0_rgb",
    "cam_1": "left_wrist_0_rgb",
    "cam_2": "right_wrist_0_rgb",
}


def extract_embodiment(repo_id: str | bytes) -> str:
    """Extract embodiment name from a RoboCOIN repo identifier."""
    if isinstance(repo_id, bytes):
        repo_id = repo_id.decode("utf-8")
    return "_".join(repo_id.split("/")[-1].split("_", 2)[:2])


def detokenize_prompt(token_ids: np.ndarray, mask: np.ndarray) -> str:
    """Decode a token ID vector back to text using the PaliGemma SentencePiece model.

    Args:
        token_ids: 1D array of token IDs.
        mask: Boolean mask of valid (non-padding) positions.

    Returns:
        Decoded text string.
    """
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer()
    ids = np.asarray(token_ids)[np.asarray(mask, dtype = bool)]
    return tokenizer._tokenizer.decode(ids.tolist())


@nnx.jit
def _jitted_compute_value(
    model_to_use: _value_fn.BaseValueFunction,
    obs: _model.Observation,
    act: _model.Actions | None,
) -> jnp.ndarray:
    return model_to_use.compute_value(obs, act, take_min_over_ensemble = True)


def stack_frames(frame_dicts: list[dict], key: str) -> jax.Array | None:
    """Stack a single key across all frame dicts into a JAX array.

    Args:
        frame_dicts: List of frame dictionaries.
        key: Key to stack from each frame dict.

    Returns:
        JAX array of stacked values, or None if key not present.
    """
    if key not in frame_dicts[0]:
        return None

    values = []
    for f in frame_dicts:
        val = f[key]
        if hasattr(val, "device"):
            val = np.asarray(val)
        values.append(val)
    return jnp.asarray(np.stack(values, axis=0))


def stack_images(frame_dicts: list[dict], image_key: str) -> tuple[dict, dict]:
    """Stack images from frame dicts with proper preprocessing.

    Args:
        frame_dicts: List of frame dictionaries.
        image_key: Key for the image dict (e.g., "image", "mirror_image").

    Returns:
        Tuple of (images_dict, image_masks_dict) as JAX arrays.
    """
    if image_key not in frame_dicts[0]:
        return {}, {}

    batch_size = len(frame_dicts)
    images_dict = {}
    image_masks_dict = {}

    for cam_key in frame_dicts[0][image_key].keys():
        cam_images = []
        for f in frame_dicts:
            img = f[image_key][cam_key]
            if hasattr(img, "device"):
                img = np.asarray(img)
            # Convert uint8 [0, 255] to float32 [-1, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 127.5 - 1.0
            cam_images.append(img)
        images_dict[cam_key] = jnp.asarray(np.stack(cam_images, axis=0))
        image_masks_dict[cam_key] = jnp.ones((batch_size,), dtype=jnp.bool_)

    return images_dict, image_masks_dict


def get_obs_and_action(
    frame_dicts: list[dict],
    prefix: str,
    action_conditioned: bool,
) -> tuple[_model.Observation, jax.Array | None]:
    """Build an Observation and action from frame dicts with a given prefix.

    Args:
        frame_dicts: List of frame dictionaries.
        prefix: Key prefix (e.g., "", "mirror_", "negative_").
                For prefix="", uses keys like "state", "image", "actions".
                For prefix="mirror_", uses keys like "mirror_state", "mirror_image", "mirror_actions".
                For prefix="negative_", only changes tokenized_prompt keys.
        action_conditioned: Whether to include actions and action_mask.

    Returns:
        Tuple of (Observation, action) where action is None if not action_conditioned.
    """
    if prefix == "negative_":
        # Negative only changes the prompt, uses same state/image/action as default
        state_key = "state"
        image_key = "image"
        actions_key = "actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_negative_prompt"
        prompt_mask_key = "tokenized_negative_prompt_mask"
    elif prefix == "mirror_":
        state_key = "mirror_state"
        image_key = "mirror_image"
        actions_key = "mirror_actions"
        action_mask_key = "action_mask"  # Same mask applies to mirrored actions
        prompt_key = "mirror_tokenized_prompt"
        prompt_mask_key = "mirror_tokenized_prompt_mask"
    else:
        # Default (empty prefix)
        state_key = "state"
        image_key = "image"
        actions_key = "actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"

    state = stack_frames(frame_dicts, state_key)
    if state is None:
        raise ValueError(f"Missing required key '{state_key}' in frame dicts")

    images_dict, image_masks_dict = stack_images(frame_dicts, image_key)

    tokenized_prompt = stack_frames(frame_dicts, prompt_key)
    tokenized_prompt_mask = stack_frames(frame_dicts, prompt_mask_key)

    action = None
    action_mask = None
    if action_conditioned:
        action = stack_frames(frame_dicts, actions_key)
        action_mask = stack_frames(frame_dicts, action_mask_key)

    obs = _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        action_mask=action_mask,
    )

    return obs, action


def count_subtask_segments(frames: list[dict], prefix: str = "") -> tuple[int, int, list[str]]:
    """Count distinct subtask segments in a trajectory and find the midpoint split.

    Args:
        frames: List of frame dictionaries, each containing subtask text keys.
        prefix: Key prefix to use (e.g., "", "negative_", "mirror_").

    Returns:
        Tuple of (num_segments, split_frame_idx, segments) where split_frame_idx is the
        frame index of the subtask boundary closest to the midpoint.
    """
    key = prefix + "subtask_1_text"
    count = 0
    prev = None
    boundaries = []
    segments = []
    segment_starts = []
    for i, f in enumerate(frames):
        st = f[key]
        if hasattr(st, "item"):
            st = st.item()
        if st != prev:
            if st.lower() not in ("null", "static", "abnormal"):
                count += 1
                segments.append(st)
                segment_starts.append(i)
                if prev is not None:
                    boundaries.append(i)
            prev = st

    labeled_segments = []
    for idx, (text, start) in enumerate(zip(segments, segment_starts)):
        end = segment_starts[idx + 1] if idx + 1 < len(segment_starts) else len(frames)
        labeled_segments.append(f"{text} - [{start}, {end})")

    split_frame_idx = boundaries[count // 2 - 1] if boundaries else len(frames) // 2
    return count, split_frame_idx, labeled_segments


def cache_val_episodes(
    val_dataloader,
    num_val_trajectories: int,
    cache_dir: str | None,
    include_repos: tuple[str, ...],
    save_only: bool,
) -> dict[int, list[dict]]:
    """Load or collect validation episodes, optionally caching them to disk.

    On first call (no cache), iterates the dataloader, collects frames for
    num_val_trajectories unique-repo-id trajectories, and optionally saves them
    to per-trajectory pickle files under cache_dir.  On subsequent calls the
    cache is loaded instead of re-iterating the dataloader.

    Args:
        val_dataloader: RoboCOIN dataloader with repeat=False, shuffle=False.
        num_val_trajectories: Number of unique-repo trajectories to collect.
        cache_dir: Directory to save/load cached trajectories. None disables caching.
        include_repos: Repo IDs that must be included (up to len(include_repos) slots reserved).
        save_only: If True, collect and cache episodes then return {}.

    Returns:
        Dict mapping traj_idx -> sorted list of frame dicts, or {} if save_only=True.
    """
    traj_frames: dict[int, list[dict]] = {}

    cache_exists = (
        cache_dir is not None
        and os.path.exists(cache_dir)
        and any(f.startswith("traj_") and f.endswith(".pkl") for f in os.listdir(cache_dir))
    )

    if cache_exists and not save_only:
        logger.info(f"Loading cached validation episodes from {cache_dir}")
        for filename in os.listdir(cache_dir):
            if filename.startswith("traj_") and filename.endswith(".pkl"):
                traj_idx = int(filename.replace("traj_", "").replace(".pkl", ""))
                cache_file = os.path.join(cache_dir, filename)
                with open(cache_file, "rb") as f:
                    traj_frames[traj_idx] = pickle.load(f)
        logger.info(f"Loaded {len(traj_frames)} trajectories from cache")
    elif not cache_exists:
        active_trajs: set[int] = set()
        saved_traj_count = 0
        seen_repo_ids: set[str] = set()
        seen_required_repo_ids: set[str] = set()
        num_non_required_slots = num_val_trajectories - len(include_repos)
        logger.info(f"Collecting validation frames for first {num_val_trajectories} unique repo_id trajectories")

        if cache_dir:
            os.makedirs(cache_dir, exist_ok = True)

        for batch in val_dataloader:
            traj_indices = batch.get("_traj_index", None)
            if traj_indices is None:
                logger.warning("Batch missing _traj_index, cannot identify trajectories")
                continue

            if hasattr(traj_indices, "device"):
                traj_indices = np.asarray(traj_indices)

            unique_batch_trajs = set(int(t) for t in traj_indices)

            completed_trajs = active_trajs - unique_batch_trajs
            for traj_idx in completed_trajs:
                if traj_idx in traj_frames:
                    frames = traj_frames[traj_idx]
                    if frames:
                        frames.sort(key=lambda f: f["_frame_index"])

                    if cache_dir:
                        cache_file = os.path.join(cache_dir, f"traj_{traj_idx}.pkl")
                        with open(cache_file, "wb") as f:
                            pickle.dump(frames, f)
                        logger.info(f"Saved traj {traj_idx} ({len(frames)} frames) to {cache_file}")

                    del traj_frames[traj_idx]
                    saved_traj_count += 1

                active_trajs.discard(traj_idx)

            if saved_traj_count >= num_val_trajectories:
                logger.info(f"Saved {saved_traj_count} trajectories, breaking early")
                break

            batch_size = traj_indices.shape[0]
            for i in range(batch_size):
                traj_idx = int(traj_indices[i])

                if traj_idx not in traj_frames and traj_idx not in active_trajs:
                    sample_repo_id = batch["repo_id"][i]
                    if isinstance(sample_repo_id, bytes):
                        sample_repo_id = sample_repo_id.decode("utf-8")
                    if sample_repo_id in seen_repo_ids:
                        continue

                    is_required = sample_repo_id in include_repos and sample_repo_id not in seen_required_repo_ids
                    if not is_required and num_non_required_slots == 0:
                        continue

                    traj_frames[traj_idx] = []
                    active_trajs.add(traj_idx)
                    seen_repo_ids.add(sample_repo_id)
                    if is_required:
                        seen_required_repo_ids.add(sample_repo_id)
                    else:
                        num_non_required_slots -= 1

                if traj_idx not in active_trajs:
                    continue

                frame = {}
                for key, value in batch.items():
                    if isinstance(value, dict):
                        frame[key] = {}
                        for sub_key, sub_value in value.items():
                            frame[key][sub_key] = np.asarray(sub_value[i])
                    else:
                        frame[key] = np.asarray(value[i])

                traj_frames[traj_idx].append(frame)

        # Save any remaining active trajectories that were still in-progress when the dataloader ended
        for traj_idx in list(active_trajs):
            if saved_traj_count >= num_val_trajectories:
                break
            if traj_idx in traj_frames:
                frames = traj_frames[traj_idx]
                if frames:
                    frames.sort(key = lambda f: f["_frame_index"])
                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"traj_{traj_idx}.pkl")
                    with open(cache_file, "wb") as f:
                        pickle.dump(frames, f)
                    logger.info(f"Saved remaining traj {traj_idx} ({len(frames)} frames) to {cache_file}")
                del traj_frames[traj_idx]
                saved_traj_count += 1

        logger.info(f"Total trajectories saved: {saved_traj_count}, unique repo_ids: {len(seen_repo_ids)}")

    if save_only:
        return {}

    return traj_frames


def predict_values(
    model: _value_fn.BaseValueFunction,
    all_frames: list[tuple],
    ep_mc_returns: dict,
    action_conditioned: bool,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], dict[str, list[np.ndarray]]]:
    """Run batched value function inference on collected validation frames.

    Performs three forward passes per batch where applicable: default prompt,
    negative (counterfactual) prompt, and mirrored demonstration.

    When the network's compute_value returns (val, attn_scores) (e.g. PaliGemma at
    inference), attention scores are collected alongside predictions and returned as
    the fourth element. Otherwise the fourth element is an empty dict.

    Args:
        model: The value function model.
        all_frames: List of (traj_idx, frame_idx_in_ep, frame_dict) tuples in order.
        ep_mc_returns: Dict mapping traj_idx -> list of mc_return values (used for keying output).
        action_conditioned: Whether the model expects actions as input.

    Returns:
        Tuple of (all_predictions, all_predictions_neg, all_predictions_mirror, all_attn_scores).
    """
    BATCH_SIZE = 64
    all_predictions: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_predictions_neg: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_predictions_mirror: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_attn_scores: dict[str, list[np.ndarray]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}

    for batch_start in range(0, len(all_frames), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_frames))
        batch_frames = all_frames[batch_start:batch_end]

        frame_dicts = [f[2] for f in batch_frames]

        obs, act = get_obs_and_action(frame_dicts, prefix="", action_conditioned=action_conditioned)

        if batch_start == 0:
            logger.info(f"  Batch obs.state: shape={obs.state.shape}, dtype={obs.state.dtype}")
            if obs.images:
                for k, v in obs.images.items():
                    logger.info(f"  Batch obs.images[{k}]: shape={v.shape}, dtype={v.dtype}")
            if obs.tokenized_prompt is not None:
                logger.info(f"  Batch obs.tokenized_prompt: shape={obs.tokenized_prompt.shape}")

        pred_values_np, attn_np = jax.device_get(_jitted_compute_value(model, obs, act))

        pred_values_neg_np = None
        if "tokenized_negative_prompt" in frame_dicts[0]:
            obs_neg, act_neg = get_obs_and_action(frame_dicts, prefix="negative_", action_conditioned=action_conditioned)
            pred_values_neg_np, _ = jax.device_get(_jitted_compute_value(model, obs_neg, act_neg))

        pred_values_mirror_np = None
        if "mirror_state" in frame_dicts[0]:
            obs_mirror, act_mirror = get_obs_and_action(frame_dicts, prefix="mirror_", action_conditioned=action_conditioned)
            pred_values_mirror_np, _ = jax.device_get(_jitted_compute_value(model, obs_mirror, act_mirror))

        for i, (ep_idx, _, _) in enumerate(batch_frames):
            all_predictions[ep_idx].append(float(pred_values_np[i]))
            all_attn_scores[ep_idx].append(attn_np[i])
            if pred_values_neg_np is not None:
                all_predictions_neg[ep_idx].append(float(pred_values_neg_np[i]))
            if pred_values_mirror_np is not None:
                all_predictions_mirror[ep_idx].append(float(pred_values_mirror_np[i]))

    total_predictions = sum(len(preds) for preds in all_predictions.values())
    logger.info(f"Computed {total_predictions} predictions")

    return all_predictions, all_predictions_neg, all_predictions_mirror, all_attn_scores


@nnx.jit
def _jitted_compute_features(
    network: BaseValueNetwork,
    obs: _model.Observation,
    act: _model.Actions | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return network.compute_features(obs, act)


def extract_embeddings(
    network: BaseValueNetwork,
    all_frames: list[tuple],
    ep_keys: set,
    action_conditioned: bool,
    batch_size: int = 64,
) -> dict[str, list[np.ndarray]]:
    """Batched feature extraction, returning per-episode lists of embedding vectors.

    Args:
        network: The value network (before the head).
        all_frames: List of (ep_key, frame_idx_in_ep, frame_dict) tuples.
        ep_keys: Set of episode keys to collect embeddings for.
        action_conditioned: Whether the network expects actions as input.
        batch_size: Number of frames per forward pass.

    Returns:
        Dict mapping ep_key -> list of embedding numpy arrays [embed_dim].
    """
    all_embeddings: dict[str, list[np.ndarray]] = {k: [] for k in ep_keys}

    for batch_start in range(0, len(all_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(all_frames))
        batch_frames = all_frames[batch_start:batch_end]

        frame_dicts = [f[2] for f in batch_frames]
        obs, act = get_obs_and_action(frame_dicts, prefix = "", action_conditioned = action_conditioned)

        result = _jitted_compute_features(network, obs, act)
        if isinstance(result, tuple):
            features = result[0]
        else:
            features = result
        current_batch_size = len(batch_frames)
        features = jax.experimental.multihost_utils.process_allgather(features, tiled = True)
        assert features.shape == (current_batch_size, network.feature_dim), (
            f"Expected ({current_batch_size}, {network.feature_dim}), got {features.shape}"
        )
        features_np = jax.device_get(features)

        for i, (ep_key, _, _) in enumerate(batch_frames):
            all_embeddings[ep_key].append(features_np[i])

    total = sum(len(v) for v in all_embeddings.values())
    logger.info(f"Extracted {total} embeddings across {len(ep_keys)} episodes")

    return all_embeddings
