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


def decode_text(text: str | bytes | np.ndarray | np.generic) -> str:
    """Decode a scalar text-like value to a Python string."""
    if hasattr(text, "item"):
        text = text.item()
    if isinstance(text, bytes):
        return text.decode("utf-8")
    return str(text)


def swap_left_right_text(text: str) -> str:
    """Swap left/right mentions in a task string."""
    swapped = text.replace("left", "TEMP_LEFT_MARKER")
    swapped = swapped.replace("right", "left")
    return swapped.replace("TEMP_LEFT_MARKER", "right")


def generate_negative_subtask_text(subtask_text: str) -> str:
    """Generate the negative counterfactual text used for RoboCOIN validation."""
    if "Place the plate" in subtask_text:
        return "Place the plate on the dish rack"
    if "Grab the knife" in subtask_text:
        return "Grab the banana with your right hand"
    if "Place the knife" in subtask_text:
        return "Place the knife on the board"
    if "Pass the plate" in subtask_text:
        return "Rotate the plate with the right gripper"
    return swap_left_right_text(subtask_text)


def sample_random_actions(
    actions: np.ndarray,
    *,
    use_quantile_norm: bool,
    _traj_index: int | None = None,
    frame_index: int | None = None,
) -> np.ndarray:
    """Sample deterministic per-frame random normalized actions for validation."""
    clip_bound = 1.25 if use_quantile_norm else 5.0
    if _traj_index is None or frame_index is None:
        seed = 86
    else:
        seed = ((_traj_index + 1) * 1_000_003 + frame_index) % (2**32)
    rng = np.random.default_rng(seed = seed)
    return rng.uniform(-clip_bound, clip_bound, size = actions.shape).astype(actions.dtype, copy = False)


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
        prefix: Key prefix (e.g., "", "negative_", "random_").
                For prefix="", uses keys like "state", "image", "actions".
                For prefix="negative_", only changes tokenized_prompt keys.
                For prefix="random_" or prefix="counterfactual_", only changes the action key.
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
    elif prefix == "random_":
        state_key = "state"
        image_key = "image"
        actions_key = "random_actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"
    elif prefix == "counterfactual_":
        state_key = "state"
        image_key = "image"
        actions_key = "counterfactual_actions"
        action_mask_key = "action_mask"
        prompt_key = "tokenized_prompt"
        prompt_mask_key = "tokenized_prompt_mask"
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
        if action is not None and action.ndim == 4:
            action = action[:, 0]
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


_CACHE_KEYS = {
    "state", "image", "image_mask", "actions", "action_mask",
    "tokenized_prompt", "tokenized_prompt_mask",
    "tokenized_negative_prompt", "tokenized_negative_prompt_mask",
    "negative_subtask_1_text", "random_actions", "counterfactual_actions",
    "mc_return", "include_subtask", "fps",
    "repo_id", "episode_index", "_frame_index",
}


def _unstack_trajectory(traj: dict, t: int) -> dict:
    """Extract a single frame at timestep t from a stacked trajectory dict."""
    frame: dict = {}
    for key, value in traj.items():
        if isinstance(value, dict):
            frame[key] = {k: np.asarray(v[t]) for k, v in value.items()}
        else:
            frame[key] = np.asarray(value[t])
    return frame


def _extract_cache_frame(frame: dict, prompt_text: str) -> dict:
    """Keep only plotting-relevant keys and add subtask_1_text from the original prompt."""
    cached: dict = {}
    for key in _CACHE_KEYS:
        if key in frame:
            cached[key] = frame[key]
    cached["subtask_1_text"] = prompt_text
    return cached


def _decode_repo_id(raw) -> str:
    if isinstance(raw, np.ndarray):
        if raw.ndim != 0:
            raise ValueError(f"Expected scalar repo_id, got shape {raw.shape}")
        raw = raw.item()
    elif isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _decode_and_resize_image_bytes(
    image_bytes: bytes | np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    """Decode a single JPEG/PNG image to uint8 and resize to target_size.

    Used at prediction time so that cache_val_episodes can store the raw
    compressed bytes (≈10–30x smaller than decoded uint8) and we only
    materialize the decoded array for one episode at a time.
    """
    import tensorflow as tf

    if isinstance(image_bytes, np.ndarray):
        image_bytes = image_bytes.item() if image_bytes.shape == () else bytes(image_bytes)
    image = tf.io.decode_image(image_bytes, expand_animations = False, dtype = tf.uint8)
    image = tf.image.resize(
        image, target_size, method = tf.image.ResizeMethod.BILINEAR, antialias = True
    )
    image = tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8)
    return image.numpy()


def decode_episode_images(frames: list[dict], target_size: tuple[int, int]) -> None:
    """In-place decode + resize of all `image` entries in a list of cached frames.

    Frames cached via cache_val_episodes (with the upstream dataset configured
    with decode_images=False) hold raw compressed bytes per camera. This
    converts them to uint8 (target_size, 3) arrays so downstream stack_images /
    plotting can consume them. Only operates on frames whose `image[cam]` is
    bytes / scalar object array; already-decoded frames are passed through.
    """
    for frame in frames:
        if "image" not in frame:
            continue
        image_dict = frame["image"]
        for cam_key, value in list(image_dict.items()):
            if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3:
                continue  # already decoded
            image_dict[cam_key] = _decode_and_resize_image_bytes(value, target_size)


def cache_val_episodes(
    trajectory_iter,
    num_val_trajectories: int,
    cache_dir: str | None,
    include_repos: tuple[str, ...],
    save_only: bool,
    input_transform = None,
) -> dict[str, list[dict]]:
    """Load or collect validation episodes, optionally caching them to disk.

    Iterates an RLDS trajectory dataset (return_trajectories=True), applies
    input_transform to each frame, and caches the transformed frames as
    per-trajectory pickle files keyed by ``repo_key`` (``repo_index`` when the
    underlying dataset exposes it, else ``repo_id``; sanitized: ``/`` → ``__``).
    Using ``repo_index`` lets a single-task fine-tune cache multiple distinct
    trajectories from the same ``repo_id``.
    On subsequent calls the cache is loaded instead of re-iterating the dataset.

    Multi-worker safe: every host calls this concurrently. Worker 0 is the
    full-coverage worker (caches all ``num_val_trajectories`` repos including
    non-required ones); workers > 0 only cache repos in ``include_repos``. Each
    worker checks ``<sanitized_repo_key>.pkl`` existence at two points
    — when deciding whether to claim the repo and again right before writing
    — and skips if another worker already produced that file.

    Args:
        trajectory_iter: RLDS dataset yielding full trajectories (stacked dicts),
            or None when loading from an existing cache.
        num_val_trajectories: Number of unique-repo trajectories to collect.
        cache_dir: Directory to save/load cached trajectories. None disables caching.
        include_repos: Repo IDs that must be included (up to len(include_repos) slots reserved).
        save_only: If True, collect and cache episodes then return {}.
        input_transform: Composed transform pipeline (Normalize + ResizeImages + TokenizePrompt)
            applied to each frame before caching. Required when collecting, ignored when loading.

    Returns:
        Dict mapping ``<sanitized_repo_key>`` (str) -> sorted list of frame dicts,
        or {} if save_only=True.
    """
    import jax  # local import to keep this util usable from non-JAX callers

    traj_frames: dict[str, list[dict]] = {}
    process_index = jax.process_index() if jax is not None else 0
    include_set = set(include_repos)

    def _sanitize(repo_id: str) -> str:
        return repo_id.replace("/", "__")

    cache_exists = (
        cache_dir is not None
        and os.path.exists(cache_dir)
        and any(f.endswith(".pkl") for f in os.listdir(cache_dir))
    )

    # Fast-path: if save_only=True and every .pkl this worker would have
    # written already exists, skip the trajectory_iter entirely. Iterating
    # the val tf.data pipeline a second time after the cache is full
    # accumulates host RAM (prefetch buffers + per-traj pickle bursts) and
    # has OOM-killed workers caching the largest trajectories
    # (tableware_cleaning, 2930 frames → 1.3 GB pickle).
    if save_only and cache_dir is not None and os.path.exists(cache_dir):
        existing_pkls = {f for f in os.listdir(cache_dir) if f.endswith(".pkl")}
        required_pkls = {f"{_sanitize(repo)}.pkl" for repo in include_repos}
        all_required = required_pkls.issubset(existing_pkls)
        # Worker 0 is the only worker that fills the non-required slots.
        if process_index == 0:
            all_required = all_required and len(existing_pkls) >= num_val_trajectories
        if all_required:
            logger.info(
                f"[pidx={process_index}] save_only fast-path: cache already "
                f"contains required .pkls ({len(existing_pkls)} present), "
                f"skipping trajectory iteration."
            )
            return {}

    if cache_exists and not save_only:
        logger.info(f"[pidx={process_index}] Loading cached validation episodes from {cache_dir}")
        for filename in os.listdir(cache_dir):
            if filename.endswith(".pkl"):
                key = filename[: -len(".pkl")]
                cache_file = os.path.join(cache_dir, filename)
                with open(cache_file, "rb") as f:
                    traj_frames[key] = pickle.load(f)
        logger.info(f"[pidx={process_index}] Loaded {len(traj_frames)} trajectories from cache")
    else:
        # Collect path. Runs for save_only=True regardless of cache_exists, so
        # that workers arriving after the first .pkl is written still iterate
        # their shard and contribute their include_repos. The per-file
        # os.path.exists + O_CREAT|O_EXCL checks below guard against
        # double-writes.
        assert trajectory_iter is not None, "No cache found and no trajectory iterator provided."
        assert input_transform is not None, "input_transform is required when collecting trajectories."

        seen_repo_keys: set[int | str] = set()
        seen_required_repo_ids: set[str] = set()
        num_non_required_slots = num_val_trajectories - len(include_repos)
        collected_count = 0
        logger.info(
            f"[pidx={process_index}] Collecting validation trajectories "
            f"({'all repos up to ' + str(num_val_trajectories) if process_index == 0 else 'include_repos only'})"
        )

        if cache_dir:
            os.makedirs(cache_dir, exist_ok = True)
            # TPU workers within the same pod can be assigned different uids
            # for the same user, so a dir created by one worker often
            # isn't writable by the others (mode 0775, group-mismatched).
            # Best-effort chmod to 0o777 — silently ignore if we don't own it.
            try:
                os.chmod(cache_dir, 0o777)
            except (PermissionError, OSError):
                pass

        for traj in trajectory_iter:
            # Some shards/transforms can yield zero-frame trajectories (all
            # frames filtered out by include_subtask / mask_50fps / etc.). Skip
            # rather than crash on traj["repo_id"][0].
            if len(traj["repo_id"]) == 0:
                continue

            repo_id = _decode_repo_id(traj["repo_id"][0])
            repo_key: int | str = int(traj["repo_index"][0]) if "repo_index" in traj else repo_id

            if repo_key in seen_repo_keys:
                continue

            # Worker > 0: only claim repos in include_repos.
            if process_index != 0 and repo_id not in include_set:
                continue

            is_required = repo_id in include_repos and repo_id not in seen_required_repo_ids
            if process_index == 0 and not is_required and num_non_required_slots <= 0:
                continue

            sanitized = _sanitize(str(repo_key))
            cache_file = (
                os.path.join(cache_dir, f"{sanitized}.pkl") if cache_dir else None
            )
            # Skip if another worker already produced this file.
            if cache_file is not None and os.path.exists(cache_file):
                logger.info(
                    f"[pidx={process_index}] {sanitized}.pkl already exists in {cache_dir}, skipping"
                )
                seen_repo_keys.add(repo_key)
                if is_required:
                    seen_required_repo_ids.add(repo_id)
                elif process_index == 0:
                    num_non_required_slots -= 1
                collected_count += 1
                if process_index == 0 and collected_count >= num_val_trajectories:
                    break
                if process_index != 0 and len(seen_required_repo_ids) >= len(include_set):
                    break
                continue

            # Un-stack trajectory, apply transforms, keep only cache keys
            traj_len = len(traj["repo_id"])
            frames = []
            for t in range(traj_len):
                frame = _unstack_trajectory(traj, t)
                prompt_text = frame["prompt"]
                if isinstance(prompt_text, bytes):
                    prompt_text = prompt_text.decode("utf-8")
                elif hasattr(prompt_text, "item"):
                    prompt_text = prompt_text.item()
                    if isinstance(prompt_text, bytes):
                        prompt_text = prompt_text.decode("utf-8")
                frame = input_transform(frame)
                frames.append(_extract_cache_frame(frame, prompt_text))

            frames.sort(key = lambda f: int(f["_frame_index"]))

            if cache_file is not None:
                # Re-check existence right before writing to handle the race
                # where another worker finished caching the same repo while we
                # were transforming frames.
                if os.path.exists(cache_file):
                    logger.info(
                        f"[pidx={process_index}] {sanitized}.pkl appeared during transform, skipping write"
                    )
                else:
                    # Use O_CREAT|O_EXCL for an atomic create-or-fail. Catching
                    # FileExistsError + PermissionError treats both as "another
                    # worker won the race, skip".
                    try:
                        fd = os.open(cache_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                    except (FileExistsError, PermissionError) as exc:
                        logger.info(
                            f"[pidx={process_index}] race on {sanitized}.pkl ({type(exc).__name__}), skipping write"
                        )
                    else:
                        with os.fdopen(fd, "wb") as f:
                            pickle.dump(frames, f)
                        # chmod 0o777 so other workers in this pod (which may
                        # have inconsistent uids for the same user) can
                        # subsequently read/write this file.
                        try:
                            os.chmod(cache_file, 0o777)
                        except (PermissionError, OSError):
                            pass
                        logger.info(
                            f"[pidx={process_index}] Cached {sanitized} (repo {repo_id}, {traj_len} frames) to {cache_file}"
                        )

            if not save_only:
                traj_frames[sanitized] = frames

            seen_repo_keys.add(repo_key)
            if is_required:
                seen_required_repo_ids.add(repo_id)
            elif process_index == 0:
                num_non_required_slots -= 1
            collected_count += 1

            if process_index == 0 and collected_count >= num_val_trajectories:
                break

        logger.info(
            f"[pidx={process_index}] Collected {collected_count} trajectories, "
            f"unique repos: {len(seen_repo_keys)}"
        )

    if save_only:
        return {}

    return traj_frames


def predict_values(
    model: _value_fn.BaseValueFunction,
    all_frames: list[tuple],
    ep_mc_returns: dict,
    action_conditioned: bool,
    batch_size: int = 64,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[np.ndarray]],
]:
    """Run batched value function inference on collected validation frames.

    Performs three forward passes per batch where applicable: default prompt,
    negative (counterfactual) prompt, and random actions.

    When the network's compute_value returns (val, attn_scores) (e.g. PaliGemma at
    inference), attention scores are collected alongside predictions and returned as
    the fourth element. Otherwise the fourth element is an empty dict.

    Args:
        model: The value function model.
        all_frames: List of (traj_idx, frame_idx_in_ep, frame_dict) tuples in order.
        ep_mc_returns: Dict mapping traj_idx -> list of mc_return values (used for keying output).
        action_conditioned: Whether the model expects actions as input.

    Returns:
        Tuple of (
            all_predictions,
            all_predictions_neg,
            all_predictions_random,
            all_predictions_counterfactual,
            all_attn_scores,
        ).
    """
    all_predictions: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_predictions_neg: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_predictions_random: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_predictions_counterfactual: dict[str, list[float]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}
    all_attn_scores: dict[str, list[np.ndarray]] = {ep_idx: [] for ep_idx in ep_mc_returns.keys()}

    for batch_start in range(0, len(all_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(all_frames))
        batch_frames = all_frames[batch_start:batch_end]

        frame_dicts = [f[2] for f in batch_frames]
        # Pad partial last batches with copies of the final frame so every
        # ``_jitted_compute_value`` call sees the same leading-axis size. Without
        # this, each unique trailing-batch length specialises a new compiled
        # program in the XLA cache, and that cache (per-traj × multiple
        # variants) eats enough HBM that the next ptrain_step can OOM.
        if len(frame_dicts) < batch_size:
            frame_dicts = frame_dicts + [frame_dicts[-1]] * (batch_size - len(frame_dicts))

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

        pred_values_random_np = None
        if "random_actions" in frame_dicts[0]:
            obs_random, act_random = get_obs_and_action(frame_dicts, prefix="random_", action_conditioned=action_conditioned)
            pred_values_random_np, _ = jax.device_get(_jitted_compute_value(model, obs_random, act_random))

        pred_values_counterfactual_np = None
        if "counterfactual_actions" in frame_dicts[0]:
            obs_counterfactual, act_counterfactual = get_obs_and_action(
                frame_dicts, prefix = "counterfactual_", action_conditioned = action_conditioned
            )
            pred_values_counterfactual_np, _ = jax.device_get(
                _jitted_compute_value(model, obs_counterfactual, act_counterfactual)
            )

        for i, (ep_idx, _, _) in enumerate(batch_frames):
            all_predictions[ep_idx].append(float(pred_values_np[i]))
            all_attn_scores[ep_idx].append(attn_np[i])
            if pred_values_neg_np is not None:
                all_predictions_neg[ep_idx].append(float(pred_values_neg_np[i]))
            if pred_values_random_np is not None:
                all_predictions_random[ep_idx].append(float(pred_values_random_np[i]))
            if pred_values_counterfactual_np is not None:
                all_predictions_counterfactual[ep_idx].append(float(pred_values_counterfactual_np[i]))

    total_predictions = sum(len(preds) for preds in all_predictions.values())
    logger.info(f"Computed {total_predictions} predictions")

    return all_predictions, all_predictions_neg, all_predictions_random, all_predictions_counterfactual, all_attn_scores




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
