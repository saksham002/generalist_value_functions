"""DLIMP-based data loader for the RoboCOIN TFDS dataset.

This module provides a data loader for training value functions and policies on the
RoboCOIN dataset, which contains robot manipulation trajectories with images and
proprioceptive state observations.

Output format follows the standard convention from openpi/models/model.py:
    {
        "image": {"base_0_rgb": ..., "left_wrist_0_rgb": ..., ...},
        "image_mask": {"base_0_rgb": ..., ...},
        "state": [state_dim],
        "next_state": [state_dim],
        "actions": [action_horizon, action_dim],
        "next_actions": [action_horizon, action_dim],
        "action_mask": [action_horizon],
        "next_action_mask": [action_horizon],
        "tokenized_prompt": [max_token_len],
        "tokenized_prompt_mask": [max_token_len],
        ...
    }
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import logging
from typing import Any

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from openpi import transforms as _transforms
from openpi.models.model import IMAGE_KEYS
from openpi.models.tokenizer import PaligemmaTokenizer

logger = logging.getLogger(__name__)
# Disable GPU for TensorFlow (we only use it for data loading)
# tf.config.experimental.set_visible_devices([], "GPU")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATA_DIR = "/data/group_data/rl/saksham3/"
DEFAULT_DATASET_NAME = "robocoin:1.0.0"
DEFAULT_MAX_CAMERAS = 3
DEFAULT_MAX_STATE_DIM = 14
DEFAULT_MAX_ACTION_DIM = 14
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MAX_TOKEN_LEN = 48
DEFAULT_ACTION_HORIZON = 30

RLDS_TO_STANDARD_CAMERA_MAP = {
    f"cam_{i}": IMAGE_KEYS[i] for i in range(min(DEFAULT_MAX_CAMERAS, len(IMAGE_KEYS)))
}

# =============================================================================
# Configuration
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RoboCOINDataLoaderConfig:
    """Configuration for the RoboCOIN data loader."""

    data_dir: str = DEFAULT_DATA_DIR
    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "train"
    batch_size: int = 64
    shuffle: bool = True
    local_shuffle_buffer_size: int = 250000
    seed: int = 86
    prefetch_buffer_size: int = 4
    max_cameras: int = DEFAULT_MAX_CAMERAS
    max_state_dim: int = DEFAULT_MAX_STATE_DIM
    max_action_dim: int = DEFAULT_MAX_ACTION_DIM
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    drop_remainder: bool = False
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN
    num_batches: int | None = None
    state_norm_stats: dict[str, Any] | None = None
    use_quantile_norm: bool = False
    td_n: int | None = None
    repeat: bool = True
    use_eef: bool = False
    action_horizon: int = DEFAULT_ACTION_HORIZON


# =============================================================================
# TF Transforms (used with tf.data.Dataset.map)
# =============================================================================


class AddTrajectoryKeys:
    """Trajectory-level transform to add next_* keys and action chunks.

    Must run as traj_map (before flatten) to access sequential timesteps.
    
    Creates:
    - next_observation/state, next_observation/image/cam_X: at t+1 (MC) or t+td_n (TD)
    - action_chunk: [ep_len, action_horizon, action_dim] starting at current frame
    - next_action_chunk: [ep_len, action_horizon, action_dim] starting at next frame
    - action_mask: [ep_len, action_horizon] valid actions in chunk
    - next_action_mask: [ep_len, action_horizon] valid actions in next chunk
    """

    def __init__(
        self,
        td_n: int | None = None,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        use_eef: bool = False,
    ):
        self.td_n = td_n
        self.max_cameras = max_cameras
        self.action_horizon = action_horizon
        self.use_eef = use_eef

    def map(self, episode: dict[str, Any]) -> dict[str, Any]:
        ep_len = episode["_len"][0]
        frame_indices = episode["_frame_index"]

        # Validate required keys
        if "observation/state" not in episode:
            raise ValueError("Missing required key 'observation/state' in episode")
        if "action" not in episode:
            raise ValueError("Missing required key 'action' in episode")
        
        state = episode["observation/state"]
        action = episode["action"]
        
        # Validate EEF keys if use_eef is enabled
        if self.use_eef:
            if "eef_sim_pose_state" not in episode:
                raise ValueError("use_eef=True but 'eef_sim_pose_state' not found in episode")
            if "eef_sim_pose_action" not in episode:
                raise ValueError("use_eef=True but 'eef_sim_pose_action' not found in episode")
            eef_state = episode["eef_sim_pose_state"]
            eef_action = episode["eef_sim_pose_action"]
        
        cam_keys = [f"observation/image/cam_{i}" for i in range(self.max_cameras)]

        next_offset = 1 if self.td_n is None else self.td_n
        next_indices = tf.minimum(frame_indices + next_offset, ep_len - 1)

        episode["next_observation/state"] = tf.gather(state, next_indices)
        
        if self.use_eef:
            episode["next_eef_sim_pose_state"] = tf.gather(eef_state, next_indices)
            
        for cam_key in cam_keys:
            if cam_key in episode:
                episode[f"next_{cam_key}"] = tf.gather(episode[cam_key], next_indices)

        offsets = tf.range(self.action_horizon)
        
        chunk_indices = frame_indices[:, None] + offsets[None, :]
        chunk_indices = tf.minimum(chunk_indices, ep_len - 1)
        episode["action_chunk"] = tf.gather(action, chunk_indices)
        
        next_chunk_indices = next_indices[:, None] + offsets[None, :]
        next_chunk_indices = tf.minimum(next_chunk_indices, ep_len - 1)
        episode["next_action_chunk"] = tf.gather(action, next_chunk_indices)

        # Create per-subtask action masks: [ep_len, 5, action_horizon]
        steps_to_subtask_end = episode.get("steps_to_subtask_end")  # [ep_len, 5]
        subtask_mask = offsets[None, None, :] <= steps_to_subtask_end[:, :, None]  # [ep_len, 5, action_horizon]
        episode["action_mask"] = subtask_mask  # [ep_len, 5, action_horizon]
        
        next_steps = tf.gather(steps_to_subtask_end, next_indices)  # [ep_len, 5]
        next_subtask_mask = offsets[None, None, :] <= next_steps[:, :, None]
        episode["next_action_mask"] = next_subtask_mask

        if self.use_eef:
            episode["eef_action_chunk"] = tf.gather(eef_action, chunk_indices)
            episode["next_eef_action_chunk"] = tf.gather(eef_action, next_chunk_indices)

        return episode


class ImageResizeTransform:
    """Decode JPEG images and resize to target size."""

    def __init__(
        self,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
    ):
        self.target_size = target_size
        self.max_cameras = max_cameras
        self._decode_and_resize_fn = tf.function(
            self._decode_and_resize_impl,
            input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)],
            reduce_retracing=True,
        )

    def _decode_and_resize_impl(self, jpeg_bytes: tf.Tensor) -> tf.Tensor:
        is_empty = tf.equal(tf.strings.length(jpeg_bytes), 0)

        def decode_resize():
            image = tf.io.decode_jpeg(jpeg_bytes, channels=3, ratio=2)
            image = tf.image.resize(image, self.target_size, method="bilinear")
            return tf.cast(image, tf.uint8)

        def return_zeros():
            return tf.zeros((*self.target_size, 3), dtype=tf.uint8)

        return tf.cond(is_empty, return_zeros, decode_resize)

    def map(self, example: dict[str, Any]) -> dict[str, Any]:
        cam_keys = [f"observation/image/cam_{i}" for i in range(self.max_cameras)]
        next_cam_keys = [f"next_observation/image/cam_{i}" for i in range(self.max_cameras)]
        all_keys = cam_keys + next_cam_keys

        images = tf.stack([example.get(k, tf.constant(b"", dtype=tf.string)) for k in all_keys])
        resized = tf.map_fn(
            self._decode_and_resize_fn,
            images,
            parallel_iterations=len(all_keys),
            fn_output_signature=tf.TensorSpec(shape=(*self.target_size, 3), dtype=tf.uint8),
        )

        for i, cam_key in enumerate(all_keys):
            if cam_key in example:
                example[cam_key] = resized[i]
        return example


# =============================================================================
# Main Transform (applied per-frame via frame_map)
# =============================================================================


class MainTransform:
    """TensorFlow transform for state/action/image restructuring.
    
    Applied per-frame. Handles state/action processing, image
    restructuring, subtask sampling, tokenization, and normalization.
    """

    def __init__(
        self,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
        use_eef: bool = False,
    ):
        self.max_cameras = max_cameras
        self.use_eef = use_eef

    @staticmethod
    def _construct_eef_repr(data: tf.Tensor, eef_data: tf.Tensor) -> tf.Tensor:
        """Construct 14-D representation from EEF pose and gripper values."""
        return tf.concat(
            [
                eef_data[..., :6],
                data[..., 6:7],
                eef_data[..., 6:12],
                data[..., 13:14],
            ],
            axis=-1,
        )

    def _process_state(self, raw_frame: dict) -> dict:
        """Process state and next_state with optional EEF representation."""
        if "observation/state" not in raw_frame:
            raise ValueError("Missing required key 'observation/state' in frame")
        
        state = tf.cast(raw_frame.pop("observation/state"), tf.float32)

        if self.use_eef:
            if "eef_sim_pose_state" not in raw_frame:
                raise ValueError("use_eef=True but 'eef_sim_pose_state' not found in frame")
            eef_state = tf.cast(raw_frame.pop("eef_sim_pose_state"), tf.float32)
            processed_state = self._construct_eef_repr(state, eef_state)
        else:
            processed_state = state

        if "next_observation/state" not in raw_frame:
            raise ValueError("Missing required key 'next_observation/state' in frame")
        
        next_state = tf.cast(raw_frame.pop("next_observation/state"), tf.float32)
        
        if self.use_eef:
            if "next_eef_sim_pose_state" not in raw_frame:
                raise ValueError("use_eef=True but 'next_eef_sim_pose_state' not found in frame")
            next_eef = tf.cast(raw_frame.pop("next_eef_sim_pose_state"), tf.float32)
            processed_next_state = self._construct_eef_repr(next_state, next_eef)
        else:
            processed_next_state = next_state

        return {"state": processed_state, "next_state": processed_next_state}

    def _process_actions(self, raw_frame: dict) -> dict:
        """Process action chunks and masks."""
        if "action_chunk" not in raw_frame:
            raise ValueError("Missing required key 'action_chunk' in frame")
        
        action_chunk = tf.cast(raw_frame.pop("action_chunk"), tf.float32)

        if self.use_eef:
            if "eef_action_chunk" not in raw_frame:
                raise ValueError("use_eef=True but 'eef_action_chunk' not found in frame")
            eef_chunk = tf.cast(raw_frame.pop("eef_action_chunk"), tf.float32)
            actions = self._construct_eef_repr(action_chunk, eef_chunk)
        else:
            actions = action_chunk

        if "action_mask" not in raw_frame:
            raise ValueError("Missing required key 'action_mask' in frame")
        action_mask = raw_frame.pop("action_mask")

        if "next_action_chunk" not in raw_frame:
            raise ValueError("Missing required key 'next_action_chunk' in frame")
        
        next_chunk = tf.cast(raw_frame.pop("next_action_chunk"), tf.float32)
        
        if self.use_eef:
            if "next_eef_action_chunk" not in raw_frame:
                raise ValueError("use_eef=True but 'next_eef_action_chunk' not found in frame")
            next_eef_chunk = tf.cast(raw_frame.pop("next_eef_action_chunk"), tf.float32)
            next_actions = self._construct_eef_repr(next_chunk, next_eef_chunk)
        else:
            next_actions = next_chunk

        if "next_action_mask" not in raw_frame:
            raise ValueError("Missing required key 'next_action_mask' in frame")
        next_action_mask = raw_frame.pop("next_action_mask")

        return {
            "actions": actions,
            "action_mask": action_mask,
            "next_actions": next_actions,
            "next_action_mask": next_action_mask,
        }

    def _restructure_images(self, raw_frame: dict, prefix: str = "") -> dict:
        """Extract camera images from flat RLDS keys to nested dict format."""
        images = {}
        masks = {}
        for cam_idx in range(self.max_cameras):
            rlds_key = f"{prefix}observation/image/cam_{cam_idx}"
            rlds_cam_name = f"cam_{cam_idx}"
            standard_key = RLDS_TO_STANDARD_CAMERA_MAP.get(rlds_cam_name, rlds_cam_name)

            if rlds_key in raw_frame:
                img_data = raw_frame.pop(rlds_key)
                images[standard_key] = img_data
                masks[standard_key] = True
        return {"images": images, "masks": masks}

    def map(self, raw_frame: dict[str, Any]) -> dict[str, Any]:
        """Transform a raw frame to intermediate format (TF tensors only)."""
        frame = {}

        # Process state/actions
        frame.update(self._process_state(raw_frame))
        frame.update(self._process_actions(raw_frame))

        # Process images
        current_imgs = self._restructure_images(raw_frame, prefix="")
        next_imgs = self._restructure_images(raw_frame, prefix="next_")
        
        if current_imgs["images"]:
            frame["image"] = current_imgs["images"]
            frame["image_mask"] = current_imgs["masks"]
        if next_imgs["images"]:
            frame["next_image"] = next_imgs["images"]
            frame["next_image_mask"] = next_imgs["masks"]

        # Pass through metadata and subtask info (will be processed post-batch in NumPy)
        for key in ["first_null_index", "steps_to_subtask_end", "episode_index", "_frame_index",
                    "subtask_1", "subtask_2", "subtask_3", "subtask_4", "subtask_5", "_traj_index"]:
            if key in raw_frame:
                frame[key] = raw_frame[key]

        return frame


# =============================================================================
# Post-Batch NumPy Transform
# =============================================================================


class PostBatchTransform:
    """NumPy transform applied after batching for tokenization and normalization."""

    def __init__(
        self,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
        state_norm_stats: dict[str, Any] | None = None,
        use_quantile_norm: bool = False,
        td_n: int | None = None,
        use_eef: bool = False,
        split: str = "train",
    ):
        self.discount = discount
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.td_n = td_n
        self.use_eef = use_eef
        self.split = split

        self._tokenizer: PaligemmaTokenizer | None = None
        self._max_token_len = max_token_len

        self._normalize_fn: _transforms.Normalize | None = None
        self._normalize_fn = _transforms.Normalize(state_norm_stats, use_quantiles=use_quantile_norm)

    @property
    def tokenizer(self) -> PaligemmaTokenizer:
        if self._tokenizer is None:
            self._tokenizer = PaligemmaTokenizer(max_len=self._max_token_len)
        return self._tokenizer

    @staticmethod
    def _generate_negative_subtask_text(subtask_text: str) -> str:
        if "Place the plate" in subtask_text:
            return "Place the plate on the table"
        if "Grab the knife" in subtask_text:
            return "Grab the banana with your right hand"
        if "Place the knife" in subtask_text:
            return "Place the knife on the board"
        if "Pass the plate" in subtask_text:
            return "Rotate the plate with the right gripper"
        return PostBatchTransform._swap_left_right_text(subtask_text)

    @staticmethod
    def _swap_left_right_text(text: str) -> str:
        swapped = text.replace("left", "TEMP_LEFT_MARKER")
        swapped = swapped.replace("right", "left")
        swapped = swapped.replace("TEMP_LEFT_MARKER", "right")
        return swapped

    @staticmethod
    def _decode_text(text: Any) -> str:
        if isinstance(text, bytes):
            return text.decode("utf-8")
        return str(text)

    @staticmethod
    def _create_mirror_images(images: dict, image_masks: dict) -> tuple[dict, dict]:
        """Create horizontally flipped images with swapped left/right wrist keys."""
        mirror_images = {}
        mirror_masks = {}
        rng = jax.random.PRNGKey(0)

        for key, img in images.items():
            sub_rngs = jax.random.split(rng, img.shape[0])
            flipped = jax.vmap(augmax.HorizontalFlip(p=1.0))(sub_rngs, jnp.asarray(img))
            flipped = np.asarray(flipped)

            mirror_key = key
            if key == "left_wrist_0_rgb":
                mirror_key = "right_wrist_0_rgb"
            elif key == "right_wrist_0_rgb":
                mirror_key = "left_wrist_0_rgb"

            mirror_images[mirror_key] = flipped
            mirror_masks[mirror_key] = image_masks.get(key, np.ones(img.shape[0], dtype=np.bool_))
        return mirror_images, mirror_masks

    @staticmethod
    def _mirror_14d_array(arr: np.ndarray) -> np.ndarray:
        """Mirror a 14D array by swapping left/right arms and flipping appropriate signs.    
        """
        if arr.shape[-1] != 14:
            raise ValueError(f"Expected last dimension to be 14, got {arr.shape[-1]}")
        
        mirror = np.empty_like(arr)
        # Swap left (0:7) and right (7:14) arms
        mirror[..., 0:7] = arr[..., 7:14]
        mirror[..., 7:14] = arr[..., 0:7]
        # Negate y-axis components (indices 1, 3, 5 for left arm, 8, 10, 12 for right arm)
        for idx in [1, 3, 5, 8, 10, 12]:
            mirror[..., idx] = -mirror[..., idx]
        return mirror


    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply post-batch processing: subtask sampling, tokenization, normalization."""
        first_null_index = batch.get("first_null_index")
        steps_to_subtask_end = batch.get("steps_to_subtask_end")

        # if first_null_index is None or steps_to_subtask_end is None:
        #     return batch

        first_null_index = first_null_index.astype(np.int32)
        steps_to_subtask_end = steps_to_subtask_end.astype(np.int32)
        batch_size = first_null_index.shape[0]

        subtask_texts_all = [batch[f"subtask_{i}"] for i in range(1, 6)]
        texts_stacked = np.stack(subtask_texts_all, axis=0)

        loss_masks = first_null_index > 0

        if self.split == "val":
            sampled_indices = np.zeros(batch_size, dtype=np.int32)
        else:
            safe_upper_bound = np.maximum(first_null_index, 1)
            sampled_indices = (np.random.rand(batch_size) * safe_upper_bound).astype(np.int32)

        selected_texts = texts_stacked[sampled_indices, np.arange(batch_size)]
        selected_steps = steps_to_subtask_end[np.arange(batch_size), sampled_indices]

        mc_returns = np.power(self.discount, selected_steps.astype(np.float32))
        mc_returns = np.where(loss_masks, mc_returns, 0.0)

        batch["mc_return"] = mc_returns.astype(np.float32)
        batch["loss_mask"] = loss_masks.astype(np.bool_)
        batch["sampled_indices"] = sampled_indices

        if self.td_n is not None:
            within_horizon = selected_steps < self.td_n
            batch["termination"] = within_horizon
            td_reward = np.power(self.discount, selected_steps.astype(np.float32))
            batch["reward"] = np.where(within_horizon, td_reward, 0.0).astype(np.float32)
        else:
            batch["termination"] = (selected_steps == 0)
            batch["reward"] = batch["termination"].astype(np.float32)

        batch["truncation"] = np.zeros(batch_size, dtype=np.bool_)

        # Select action_mask for the sampled subtask from per-subtask masks [B, 5, action_horizon]
        if "action_mask" in batch:
            batch["action_mask"] = batch["action_mask"][np.arange(batch_size), sampled_indices]
        
        if "next_action_mask" in batch:
            batch["next_action_mask"] = batch["next_action_mask"][np.arange(batch_size), sampled_indices]

        # Tokenize prompts
        tokenized_prompts = []
        tokenized_masks = []
        for i in range(batch_size):
            text = self._decode_text(selected_texts[i])
            tokens, mask = self.tokenizer.tokenize(text, state=None)
            tokenized_prompts.append(tokens)
            tokenized_masks.append(mask)

        batch["tokenized_prompt"] = np.stack(tokenized_prompts, axis=0)
        batch["tokenized_prompt_mask"] = np.stack(tokenized_masks, axis=0)

        if self.split == "val":
            self._process_validation_extras(batch, subtask_texts_all, batch_size)

        # Clean up subtask keys
        for i in range(1, 6):
            batch.pop(f"subtask_{i}", None)

        # Apply normalization
        batch = self._normalize_fn(batch)

        # Create mirror state and action after normalization (for counterfactual validation)
        if "mirror_image" in batch:
            batch_state = batch.get("state")
            if batch_state is not None and batch_state.shape[-1] == 14:
                batch["mirror_state"] = self._mirror_14d_array(batch_state)
            
            # Mirror actions for Q(s,a) models: actions have shape [B, action_horizon, 14]
            batch_actions = batch.get("actions")
            if batch_actions is not None and batch_actions.shape[-1] == 14:
                batch["mirror_actions"] = self._mirror_14d_array(batch_actions)

        return batch

    def _process_validation_extras(self, batch: dict, subtask_texts_all: list, batch_size: int) -> None:
        """Process validation-specific extras."""
        subtask_1_texts = [self._decode_text(subtask_texts_all[0][i]) for i in range(batch_size)]
        batch["subtask_1_text"] = subtask_1_texts

        negative_texts = [self._generate_negative_subtask_text(t) for t in subtask_1_texts]
        batch["negative_subtask_1_text"] = negative_texts

        mirror_texts = [self._swap_left_right_text(t) for t in subtask_1_texts]
        batch["mirror_subtask_1_text"] = mirror_texts

        neg_tokens, neg_masks, mirror_tokens, mirror_masks = [], [], [], []
        for i in range(batch_size):
            nt, nm = self.tokenizer.tokenize(negative_texts[i], state=None)
            neg_tokens.append(nt)
            neg_masks.append(nm)

            mt, mm = self.tokenizer.tokenize(mirror_texts[i], state=None)
            mirror_tokens.append(mt)
            mirror_masks.append(mm)

        batch["tokenized_negative_prompt"] = np.stack(neg_tokens, axis=0)
        batch["tokenized_negative_prompt_mask"] = np.stack(neg_masks, axis=0)
        batch["mirror_tokenized_prompt"] = np.stack(mirror_tokens, axis=0)
        batch["mirror_tokenized_prompt_mask"] = np.stack(mirror_masks, axis=0)

        images = batch.get("image")
        image_masks = batch.get("image_mask")
        if images and self.use_eef:
            mirror_images, mirror_masks_dict = self._create_mirror_images(images, image_masks)
            batch["mirror_image"] = mirror_images
            batch["mirror_image_mask"] = mirror_masks_dict



# =============================================================================
# Main Factory Function
# =============================================================================


def create_robocoin_data_loader(config: RoboCOINDataLoaderConfig) -> Iterator[dict[str, Any]]:
    """Create a DLIMP-based data loader for the RoboCOIN dataset.

    Args:
        config: Data loader configuration.

    Returns:
        NumPy iterator yielding batches compatible with the training pipeline.
    """
    import tensorflow_datasets as tfds

    try:
        import dlimp as dl
    except ImportError:
        raise ImportError("dlimp is required for RoboCOIN data loading. Install with: pip install dlimp")

    logger.info(f"Config: {config}")
    logger.info(f"Building DLIMP dataset: {config.dataset_name}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Batch size: {config.batch_size}")

    builder = tfds.builder(config.dataset_name, data_dir=config.data_dir)
    dataset = dl.DLataset.from_rlds(builder, split=config.split, shuffle=True, num_parallel_reads=-1)

    def _drop_episode_metadata(episode: Any) -> Any:
        if isinstance(episode, dict):
            if "steps" in episode:
                return {"steps": episode["steps"]}
            return {k: v for k, v in episode.items() if k not in ("traj_metadata", "episode_metadata")}
        return episode

    dataset = dataset.map(_drop_episode_metadata)

    if config.repeat:
        dataset = dataset.repeat()

    dataset = dataset.traj_map(
        AddTrajectoryKeys(
            td_n=config.td_n,
            max_cameras=config.max_cameras,
            action_horizon=config.action_horizon,
            use_eef=config.use_eef,
        ).map
    )

    dataset = dataset.flatten()

    dataset = dataset.frame_map(
        ImageResizeTransform(
            target_size=config.image_size,
            max_cameras=config.max_cameras,
        ).map
    )

    dataset = dataset.frame_map(
        MainTransform(
            max_cameras=config.max_cameras,
            use_eef=config.use_eef,
        ).map
    )

    if config.shuffle:
        dataset = dataset.shuffle(config.local_shuffle_buffer_size, seed=config.seed)

    dataset = dataset.batch(config.batch_size, drop_remainder=config.drop_remainder)
    dataset = dataset.with_ram_budget(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # return dataset.as_numpy_iterator()

    # Create post-batch transform
    post_batch_transform = PostBatchTransform(
        discount=config.discount,
        reward_scale=config.reward_scale,
        reward_bias=config.reward_bias,
        max_token_len=config.max_token_len,
        state_norm_stats=config.state_norm_stats,
        use_quantile_norm=config.use_quantile_norm,
        td_n=config.td_n,
        use_eef=config.use_eef,
        split=config.split,
    )

    # Wrap iterator to apply post-batch transform
    class TransformedIterator:
        def __init__(self, dataset, transform):
            self._dataset = dataset
            self._transform = transform
            self._iterator = self._dataset.as_numpy_iterator()

        def __iter__(self):
            return self

        def __next__(self):
            batch = next(self._iterator)
            return self._transform(batch)

    return TransformedIterator(dataset, post_batch_transform)


# =============================================================================
# Wrapper Class for Compatibility
# =============================================================================


class RoboCOINDataLoader:
    """Data loader wrapper compatible with the openpi data loader interface.
    
    Handles sharding similarly to RLDSDataLoader for multi-device training.
    """

    def __init__(
        self,
        config: RoboCOINDataLoaderConfig,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self.config = config
        self._num_batches = num_batches if num_batches is not None else config.num_batches

        if sharding is None:
            # Use data parallel sharding by default (same as RLDSDataLoader)
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding

    def _to_sharded_array_or_passthrough(self, x: Any) -> Any:
        """Convert numeric arrays to sharded JAX arrays; keep unsupported dtypes as-is."""
        arr = np.asarray(x)
        if arr.dtype.kind in {"O", "U", "S", "V", "M", "m"}:
            return x
        return jax.make_array_from_process_local_data(self._sharding, arr)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        num_items = 0
        while True:
            data_iter = create_robocoin_data_loader(self.config)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # Exhausted the dataset, create new iterator
                num_items += 1
                yield jax.tree.map(
                    self._to_sharded_array_or_passthrough, batch
                )
