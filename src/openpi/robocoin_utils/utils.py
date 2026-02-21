import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model


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
    for i, f in enumerate(frames):
        st = f[key]
        if hasattr(st, "item"):
            st = st.item()
        if st != prev:
            if st != "null":
                count += 1
                segments.append(st)
                if prev is not None:
                    boundaries.append(i)
            prev = st
    split_frame_idx = boundaries[count // 2 - 1] if boundaries else len(frames) // 2
    return count, split_frame_idx, segments
