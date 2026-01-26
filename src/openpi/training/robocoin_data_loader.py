"""DLIMP-based data loader for the RoboCOIN TFDS dataset.

This module provides a data loader for training V(s) value functions on the RoboCOIN
dataset, which contains robot manipulation trajectories with images and proprioceptive
state observations.

The loader is adapted from survey_scripts/test_dlimp_dataloader.py and yields batches
compatible with the batch_value_learning training pipeline.

Output format follows the standard convention from openpi/models/model.py:
    {
        "image": {"base_0_rgb": ..., "left_wrist_0_rgb": ..., ...},  # Nested dict
        "image_mask": {"base_0_rgb": ..., ...},
        "state": ...,
        "next_state": ...,
        "actions": ...,
        "next_actions": ...,
        "tokenized_prompt": [B, max_token_len],  # From subtask_1 text
        "tokenized_prompt_mask": [B, max_token_len],
        ...
    }
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
from typing import Any, Iterator
import jax

import numpy as np
import tensorflow as tf

from openpi.models.model import IMAGE_KEYS
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import sharding as _sharding
from openpi import transforms as _transforms

logger = logging.getLogger(__name__)

# Disable GPU for TensorFlow (we only use it for data loading)
tf.config.experimental.set_visible_devices([], "GPU")


# Default configuration constants
DEFAULT_DATA_DIR = "/data/group_data/rl/saksham3/"
DEFAULT_DATASET_NAME = "robocoin:1.0.0"
DEFAULT_MAX_CAMERAS = 3
DEFAULT_MAX_STATE_DIM = 14
DEFAULT_MAX_ACTION_DIM = 54
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MAX_TOKEN_LEN = 48  # Max token length for subtask text tokenization

# Mapping from RLDS cam_X keys to standard IMAGE_KEYS
# RLDS: observation/image/cam_0, cam_1, cam_2
# Standard: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
RLDS_TO_STANDARD_CAMERA_MAP = {
    f"cam_{i}": IMAGE_KEYS[i] for i in range(min(DEFAULT_MAX_CAMERAS, len(IMAGE_KEYS)))
}

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class RoboCOINDataLoaderConfig:
    """Configuration for the RoboCOIN data loader.

    Attributes:
        data_dir: Directory containing the TFDS dataset.
        dataset_name: Name of the dataset (e.g., "robocoin:1.0.0").
        split: Dataset split ("train", "test", etc.).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data.
        shuffle_buffer_size: Size of shuffle buffer for shuffling.
        seed: Random seed for shuffling.
        prefetch_buffer_size: Number of batches to prefetch asynchronously.
        max_cameras: Maximum number of camera views.
        max_state_dim: Maximum state dimension (for padding).
        max_action_dim: Maximum action dimension (for padding).
        image_size: Target image size (height, width).
        drop_remainder: Whether to drop the last incomplete batch.
        discount: Discount factor for MC return computation.
        reward_scale: Reward scaling factor.
        reward_bias: Reward bias.
        max_token_len: Maximum token length for subtask text tokenization.
        num_batches: Number of batches to yield (None for infinite).
        sharding: JAX sharding for distributed training (None for default).
        state_norm_stats: Normalization stats for state (NormStats or None).
        use_quantile_norm: Whether to use quantile normalization for state.
    """

    data_dir: str = DEFAULT_DATA_DIR
    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "train"
    batch_size: int = 64
    shuffle: bool = True
    shuffle_buffer_size: int = 250000
    seed: int = 86
    prefetch_buffer_size: int = 4
    max_cameras: int = DEFAULT_MAX_CAMERAS
    max_state_dim: int = DEFAULT_MAX_STATE_DIM          # Currently unused
    max_action_dim: int = DEFAULT_MAX_ACTION_DIM        # Currently unused
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    drop_remainder: bool = False
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN
    # Number of batches to yield (None for infinite)
    num_batches: int | None = None
    # JAX sharding for distributed training (None for default batch sharding)
    sharding: Any = None
    # Normalization stats for state (dict with 'state' key -> NormStats)
    state_norm_stats: dict[str, Any] | None = None
    # Whether to use quantile (min-max) normalization for state
    use_quantile_norm: bool = False
    # TD-n parameter: None for MC learning, int for TD-n learning
    td_n: int | None = None
    # Whether to repeat the dataset infinitely (True for training, False for validation)
    repeat: bool = True
    # If True, this host uses the full batch_size without splitting across hosts.
    # Useful for validation cache collection where only worker 0 needs the data.
    single_host_batch: bool = False


class AddBatchKeys:
    """Trajectory-level transform to add next_* keys and episode boundary info.
    
    This transform must run as a traj_map (before flatten) to have access to
    sequential timesteps within an episode.
    
    For MC learning (td_n=None):
    - next_* keys are set to current values (not used in loss)
    - termination = is_terminal, truncation = False
    
    For TD-n learning (td_n=int):
    - next_* keys are set to values at timestep t + td_n (clamped to ep_len - 1)
    - termination = True for t >= ep_len - td_n (i.e., last td_n steps)
    - truncation = False always
    - td_reward = gamma ** (ep_len - 1 - t) for terminated steps
    
    Output keys follow RLDS flattened format (will be transformed to standard format by prefetcher):
    - next_observation/state
    - next_observation/image/cam_X
    - next_action
    - termination
    - truncation
    - td_reward (TD mode only)
    """
    
    def __init__(
        self,
        td_n: int | None = None,
        gamma: float = 0.99,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
    ):
        """Initialize the transform.
        
        Args:
            td_n: Number of steps for TD learning. None for MC learning.
            gamma: Discount factor for reward computation.
            max_cameras: Number of camera views.
            reward and termination overwritten in AsyncBatchPrefetcher
        """
        self.td_n = td_n
        self.gamma = gamma
        self.max_cameras = max_cameras
    
    def map(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Add next_* keys and termination/reward fields to the episode.
        
        Args:
            episode: Episode dict with keys like observation/state, observation/image/cam_X, etc.
                     Must have _len key indicating episode length.
        
        Returns:
            Episode dict with added next_* and termination fields.
        """
        # Get episode length
        ep_len = episode["_len"][0]  # Scalar tensor
        frame_indices = episode["_frame_index"]  # (ep_len,)
        
        # Get current observation keys
        state = episode.get("observation/state", None)
        action = episode.get("action", None)
        
        # Camera image keys (RLDS format)
        cam_keys = [f"observation/image/cam_{i}" for i in range(self.max_cameras)]
        
        if self.td_n is None:
            # MC learning: next_* keys set to immediate next timestep (t+1)
            # This provides actual next-state data while still using MC returns for loss
            t_plus_1 = tf.minimum(frame_indices + 1, ep_len - 1)
            
            if state is not None:
                episode["next_observation/state"] = tf.gather(state, t_plus_1)
            if action is not None:
                episode["next_action"] = tf.gather(action, t_plus_1)
            for cam_key in cam_keys:
                if cam_key in episode:
                    episode[f"next_{cam_key}"] = tf.gather(episode[cam_key], t_plus_1)
            
            # Termination: True only at last step, truncation: False always
            is_terminal = episode.get("is_terminal", tf.zeros_like(frame_indices, dtype=tf.bool))
            episode["reward"] = tf.where(is_terminal, 1.0, 0.0)
            episode["termination"] = tf.cast(is_terminal, tf.bool)
            episode["truncation"] = tf.zeros_like(frame_indices, dtype=tf.bool)
            
        else:
            # TD-n learning: next_* keys at t + td_n
            td_n = self.td_n
            
            # Compute t + td_n indices, clamped to ep_len - 1
            t_plus_n = tf.minimum(frame_indices + td_n, ep_len - 1)
            
            # Gather next_* values at t + td_n
            if state is not None:
                episode["next_observation/state"] = tf.gather(state, t_plus_n)
            if action is not None:
                episode["next_action"] = tf.gather(action, t_plus_n)
            for cam_key in cam_keys:
                if cam_key in episode:
                    episode[f"next_{cam_key}"] = tf.gather(episode[cam_key], t_plus_n)
            
            # Termination: True for t >= ep_len - td_n (last td_n steps)
            termination = frame_indices >= ep_len - td_n
            episode["termination"] = termination
            episode["truncation"] = tf.zeros_like(frame_indices, dtype=tf.bool)
            
            # TD reward: gamma^(ep_len - 1 - t) for terminated steps, 0 otherwise
            steps_to_end = tf.cast(ep_len - 1 - frame_indices, tf.float32)
            td_reward = tf.pow(self.gamma, steps_to_end)
            episode["reward"] = tf.where(termination, td_reward, 0.0)
        
        return episode


class ImageResizeTransform:
    """Decode JPEG images and resize to target size.

    Applied element-wise (per-frame) using TF functions.
    Does NOT normalize.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
    ):
        self.target_size = target_size
        self.max_cameras = max_cameras

    def _decode_and_resize(self, jpeg_bytes: tf.Tensor) -> tf.Tensor:
        """Decode a JPEG image and resize it."""
        is_empty = tf.equal(tf.strings.length(jpeg_bytes), 0)

        def decode_resize():
            image = tf.io.decode_jpeg(jpeg_bytes, channels=3, ratio=2)
            image = tf.image.resize(image, self.target_size, method="bilinear")
            image = tf.cast(image, tf.uint8)
            return image

        def return_zeros():
            return tf.zeros((*self.target_size, 3), dtype=tf.uint8)

        return tf.cond(is_empty, return_zeros, decode_resize)

    def map(self, example: dict[str, Any]) -> dict[str, Any]:
        """Decode and resize all camera images in the example."""
        cam_keys = [f"observation/image/cam_{i}" for i in range(self.max_cameras)]
        next_cam_keys = [f"next_observation/image/cam_{i}" for i in range(self.max_cameras)]
        all_keys = cam_keys + next_cam_keys

        # Stack all camera images into a single tensor
        images = tf.stack(
            [example.get(k, tf.constant(b"", dtype=tf.string)) for k in all_keys]
        )

        # Process all cameras using tf.map_fn
        resized = tf.map_fn(
            self._decode_and_resize,
            images,
            parallel_iterations=len(all_keys),
            fn_output_signature=tf.TensorSpec(
                shape=(*self.target_size, 3), dtype=tf.uint8
            ),
        )

        # Unstack back to dict
        for i, cam_key in enumerate(all_keys):
            if cam_key in example:
                example[cam_key] = resized[i]

        return example


class AsyncBatchPrefetcher:
    """Asynchronously prefetch batches in a background thread.

    Performs state normalization and JAX sharding for distributed training.
    Images are kept as uint8 for memory efficiency - PaliGemma normalizes internally.
    
    Handles:
    - State normalization (zscore or minmax)
    - JAX sharding for distributed training
    - Tokenization of subtask text using PaligemmaTokenizer
    """

    def __init__(
        self,
        iterator: Iterator,
        buffer_size: int = 2,
        max_cameras: int = DEFAULT_MAX_CAMERAS,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
        num_batches: int | None = None,
        sharding: Any = None,
        state_norm_stats: dict[str, Any] | None = None,
        use_quantile_norm: bool = False,
        td_n: int | None = None,
        single_host_batch: bool = False,
        split: str = "train",
    ):
        self.iterator = iterator
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.max_cameras = max_cameras
        self.discount = discount
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.max_token_len = max_token_len
        self.num_batches = num_batches
        self.sharding = sharding
        self.single_host_batch = single_host_batch
        self.split = split
        self.thread = None
        self.stop_event = threading.Event()
        self.exception = None
        self._batch_count = 0
        self._crossed = 0
        self.td_n = td_n
        
        # Lazy-initialized tokenizer (to avoid download during import)
        self._tokenizer: PaligemmaTokenizer | None = None
        
        # Lazy-initialized default sharding
        self._default_sharding = None
        
        # State normalization transform (reuses existing transforms.Normalize)
        self._state_normalize_fn: _transforms.Normalize | None = None
        if state_norm_stats is not None:
            self._state_normalize_fn = _transforms.Normalize(
                state_norm_stats, 
                use_quantiles=use_quantile_norm
            )
    
    @property
    def tokenizer(self) -> PaligemmaTokenizer:
        """Lazily initialize the PaliGemma tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = PaligemmaTokenizer(max_len=self.max_token_len)
        return self._tokenizer
    
    def _get_sharding(self):
        """Get the sharding to use for JAX arrays."""
        if self.sharding is not None:
            return self.sharding
        if self._default_sharding is None:
            # Use make_mesh from sharding module with single device for default batch sharding
            mesh = _sharding.make_mesh(num_fsdp_devices = 1)
            self._default_sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS)
            )
        return self._default_sharding
    
    def _normalize_state_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply state normalization using the configured Normalize transform."""
        if self._state_normalize_fn is None:
            return batch
        return self._state_normalize_fn(batch)

    def _transform_batch(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        """Transform raw TFDS batch to standard training pipeline format.

        Input format (from TFDS/DLIMP - flat RLDS keys):
            - observation/state: [B, state_dim]
            - observation/image/cam_X: [B, H, W, C] for each camera
            - action: [B, action_dim]
            - reward: [B]
            - is_terminal: [B]
            - is_first: [B]

        Output format (standard nested dict from model.py):
            - image: {"base_0_rgb": [B, H, W, C], ...}  # Nested dict
            - image_mask: {"base_0_rgb": [B], ...}
            - state: [B, state_dim]
            - next_image: {"base_0_rgb": [B, H, W, C], ...}
            - next_image_mask: {"base_0_rgb": [B], ...}
            - next_state: [B, state_dim]
            - actions: [B, action_dim]
            - next_actions: [B, action_dim]
            - reward: [B]
            - mc_return: [B]
            - termination: [B]
            - truncation: [B]
        """
        batch = {}
        self._crossed = 0

        # State (normalization applied at end of method)
        state = raw_batch.pop("observation/state", None)
        if state is not None:
            batch["state"] = state.astype(np.float32)

        self._crossed += 1
        
        # Next state (set by AddBatchKeys transform)
        next_state = raw_batch.pop("next_observation/state", None)
        if next_state is not None:
            batch["next_state"] = next_state.astype(np.float32)

        self._crossed += 1

        # Actions
        action = raw_batch.pop("action", None)
        if action is not None:
            batch["actions"] = action

        self._crossed += 1
        
        # Next actions (set by AddBatchKeys transform)
        next_action = raw_batch.pop("next_action", None)
        if next_action is not None:
            batch["next_actions"] = next_action

        self._crossed += 1

        # Reward with scaling/bias
        reward = raw_batch.pop("reward", None)
        if reward is not None:
            reward = reward.astype(np.float32) * self.reward_scale + self.reward_bias
            batch["reward"] = reward

        self._crossed += 1

        # Episode boundaries (set by AddBatchKeys transform)
        termination = raw_batch.pop("termination", None)
        truncation = raw_batch.pop("truncation", None)
        
        if termination is not None:
            batch["termination"] = termination.astype(np.bool_)
        else:
            batch["termination"] = np.zeros(state.shape[0], dtype=np.bool_)
        
        if truncation is not None:
            batch["truncation"] = truncation.astype(np.bool_)
        else:
            batch["truncation"] = np.zeros(state.shape[0], dtype=np.bool_)

        # Camera images - output as nested dict with standard keys
        # Maps RLDS cam_X -> standard IMAGE_KEYS (base_0_rgb, etc.)
        images = {}
        image_masks = {}
        next_images = {}
        next_image_masks = {}

        for cam_idx in range(self.max_cameras):
            rlds_key = f"observation/image/cam_{cam_idx}"
            next_rlds_key = f"next_observation/image/cam_{cam_idx}"
            rlds_cam_name = f"cam_{cam_idx}"
            
            # Map to standard key if available, otherwise keep original
            standard_key = RLDS_TO_STANDARD_CAMERA_MAP.get(rlds_cam_name, rlds_cam_name)
            
            # Pop to prevent duplicate storage
            img_data = raw_batch.pop(rlds_key, None)
            if img_data is not None and img_data.size > 0:
                # Keep images as uint8 - PaliGemma normalizes internally
                images[standard_key] = img_data
                image_masks[standard_key] = np.ones(img_data.shape[0], dtype=np.bool_)
                self._crossed += 1
            
            # Next images (set by AddBatchKeys transform)
            next_img_data = raw_batch.pop(next_rlds_key, None)
            if next_img_data is not None and next_img_data.size > 0:
                # Keep images as uint8 - PaliGemma normalizes internally
                next_images[standard_key] = next_img_data
                next_image_masks[standard_key] = np.ones(next_img_data.shape[0], dtype=np.bool_)
                self._crossed += 1

        # Store as nested dicts (standard format from model.py)
        if images:
            batch["image"] = images
            batch["image_mask"] = image_masks
        if next_images:
            batch["next_image"] = next_images
            batch["next_image_mask"] = next_image_masks

        # V(s, l) training: For each datapoint, randomly sample one valid subtask.
        # Use first_null_index to determine valid subtask range [0, first_null_index).
        # Use that subtask's text as tokenized_prompt and compute mc_return as
        # gamma ** steps_to_subtask_end[sampled_idx].
        first_null_index = raw_batch.get("first_null_index", None)
        steps_to_subtask_end = raw_batch.get("steps_to_subtask_end", None)
        subtask_is_last = raw_batch.get("subtask_is_last", None)
        
        if first_null_index is not None and steps_to_subtask_end is not None:
            first_null_index = first_null_index.astype(np.int32)
            steps_to_subtask_end = steps_to_subtask_end.astype(np.int32)
            batch_size = first_null_index.shape[0]
            
            # Collect all subtask texts into a list of arrays
            # subtask_texts_all[subtask_idx][batch_idx] -> text
            subtask_texts_all = []
            for subtask_idx in range(1, 6):  # subtask_1 through subtask_5
                subtask_key = f"subtask_{subtask_idx}"
                subtask_texts = raw_batch[subtask_key]
                subtask_texts_all.append(subtask_texts)
            
            # Vectorized sampling: sample random index in [0, first_null_index) for each batch element
            # If first_null_index is 0, there are no valid subtasks
            loss_masks = first_null_index > 0  # [B] - True if there's at least one valid subtask
            
            # For validation, always use subtask_1 (index 0); for training, sample randomly
            if self.split == "val":
                sampled_indices = np.zeros(batch_size, dtype=np.int32)  # Always subtask_1
            else:
                safe_upper_bound = np.maximum(first_null_index, 1)
                sampled_indices = (np.random.rand(batch_size) * safe_upper_bound).astype(np.int32)  # [B]
            
            # Gather the selected subtask text for each batch element
            # Stack texts into shape [num_subtasks, B]
            texts_stacked = np.stack(subtask_texts_all, axis = 0)  # [5, B]
            
            # Use advanced indexing: texts_stacked[sampled_indices[i], i] for each i
            selected_texts = texts_stacked[sampled_indices, np.arange(batch_size)]  # [B]
            
            # Gather steps_to_subtask_end for the sampled indices
            # steps_to_subtask_end is [B, 5], we want steps_to_subtask_end[i, sampled_indices[i]]
            selected_steps = steps_to_subtask_end[np.arange(batch_size), sampled_indices]  # [B]
            
            mc_returns = np.power(self.discount, selected_steps.astype(np.float32))  # [B]
            self._crossed += 1
            mc_returns = np.where(loss_masks, mc_returns, 0.0)
            
            # Tokenize all selected texts
            tokenized_prompts = []
            tokenized_masks = []
            for i in range(batch_size):
                text = selected_texts[i]
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                tokens, mask = self.tokenizer.tokenize(text, state = None)
                tokenized_prompts.append(tokens)
                tokenized_masks.append(mask)
            
            batch["tokenized_prompt"] = np.stack(tokenized_prompts, axis = 0)
            batch["tokenized_prompt_mask"] = np.stack(tokenized_masks, axis = 0)
            batch["mc_return"] = mc_returns.astype(np.float32)
            self._crossed += 1
            batch["loss_mask"] = loss_masks.astype(np.bool_)
            
            # Compute termination and reward based on td_n mode
            # TD-n: termination = True if within TD horizon (use MC reward, no bootstrap)
            #       reward = gamma ** selected_steps if within horizon, else 0
            # MC (td_n=None): termination = True only at subtask completion (selected_steps == 0)
            #                 reward = 1 if terminal, else 0
            if self.td_n is not None:
                # Within TD horizon: use MC reward (gamma^steps), mark as terminal (no bootstrap)
                # Beyond TD horizon: no reward, bootstrap with V(next)
                within_horizon = selected_steps <= self.td_n
                batch["termination"] = within_horizon
                td_reward = np.power(self.discount, selected_steps.astype(np.float32))
                batch["reward"] = np.where(within_horizon, td_reward, 0.0).astype(np.float32)
            else:
                # MC mode: terminal only at subtask completion
                batch["termination"] = (selected_steps == 0)
                batch["reward"] = batch["termination"].astype(np.float32)

            batch["sampled_indices"] = sampled_indices
            
            # For validation, store subtask_1 text for plotting labels
            if self.split == "val":
                subtask_1_texts = []
                for i in range(batch_size):
                    text = subtask_texts_all[0][i]  # subtask_1 is at index 0
                    if isinstance(text, bytes):
                        text = text.decode("utf-8")
                    subtask_1_texts.append(text)
                batch["subtask_1_text"] = subtask_1_texts
        
        # Store metadata for debugging/analysis
        if first_null_index is not None:
            batch["first_null_index"] = first_null_index.astype(np.int32)
        if steps_to_subtask_end is not None:
            batch["steps_to_subtask_end"] = steps_to_subtask_end.astype(np.int32)

        # Preserve episode_index and frame indices for validation episode identification
        episode_index = raw_batch.get("episode_index", None)
        if episode_index is not None:
            batch["episode_index"] = episode_index.astype(np.int32)
        
        frame_index = raw_batch.get("_frame_index", None)
        if frame_index is not None:
            batch["_frame_index"] = frame_index.astype(np.int32)

        # Apply state normalization using transforms.Normalize
        batch = self._normalize_state_batch(batch)

        return batch

    def _prefetch_worker(self):
        """Worker function that runs in a separate thread to prefetch batches."""
        try:
            for raw_batch in self.iterator:
                if self.stop_event.is_set():
                    break

                # Transform batch to training format
                batch = self._transform_batch(raw_batch)
                self.buffer.put(batch)
        except Exception as e:
            logger.error(f"Error in prefetch worker: {e}, crossed: {self._crossed}")
            self.exception = e
            self.buffer.put(None)

    def start(self):
        """Start the prefetching thread."""
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

    def __iter__(self):
        """Make this object iterable."""
        return self

    def __next__(self) -> dict[str, Any]:
        """Get the next prefetched batch with JAX sharding applied."""
        # Check num_batches limit
        if self.num_batches is not None and self._batch_count >= self.num_batches:
            raise StopIteration
        
        if self.exception:
            raise self.exception

        item = self.buffer.get()
        if item is None:
            raise StopIteration
        
        self._batch_count += 1
        
        # Skip JAX sharding if single_host_batch is True (for validation cache collection)
        if self.single_host_batch:
            # Convert to JAX arrays, but skip text keys that can't be converted
            def maybe_to_jax(key, val):
                # Skip text-based keys that can't be JAX arrays
                if key == "subtask_1_text":
                    return val
                # Recursively handle nested dicts (e.g., "image", "image_mask")
                if isinstance(val, dict):
                    return {k: maybe_to_jax(k, v) for k, v in val.items()}
                return jax.numpy.asarray(val)
            
            return {k: maybe_to_jax(k, v) for k, v in item.items()}
        
        # Apply JAX sharding for distributed training
        sharding = self._get_sharding()
        return jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(sharding, x),
            item
        )

    def stop(self):
        """Stop the prefetching thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)


def create_robocoin_data_loader(
    config: RoboCOINDataLoaderConfig,
) -> AsyncBatchPrefetcher:
    """Create a DLIMP-based data loader for the RoboCOIN dataset.

    Args:
        config: Data loader configuration.

    Returns:
        AsyncBatchPrefetcher iterator yielding batches compatible with
        the batch_value_learning training pipeline.
    """
    import tensorflow_datasets as tfds

    try:
        import dlimp as dl
    except ImportError:
        raise ImportError(
            "dlimp is required for RoboCOIN data loading. "
            "Install with: pip install dlimp"
        )

    print(f"Building DLIMP dataset: {config.dataset_name}")
    print(f"Data directory: {config.data_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Final RoboCOINDataLoaderConfig: {config}")

    # Build TFDS dataset
    builder = tfds.builder(config.dataset_name, data_dir=config.data_dir)

    # Create DLIMP dataset from RLDS format
    # Always shuffle episodes for randomization (episode-level shuffle)
    # The config.shuffle controls frame-level shuffling after flatten
    dataset = dl.DLataset.from_rlds(
        builder, split=config.split, shuffle=True, num_parallel_reads=-1
    )

    # Drop episode-level metadata that does not align with per-step length
    def _drop_episode_metadata(episode: Any) -> Any:
        if isinstance(episode, dict):
            if "steps" in episode:
                return {"steps": episode["steps"]}
            filtered = {
                key: value
                for key, value in episode.items()
                if key not in ("traj_metadata", "episode_metadata")
            }
            if filtered:
                return filtered
        return episode

    dataset = dataset.map(_drop_episode_metadata)

    # Repeat for continuous iteration (only for training, not validation)
    if config.repeat:
        dataset = dataset.repeat()

    # Apply trajectory-level transform to add next_* keys for TD/MC learning
    # This must run before flatten() to have access to sequential timesteps
    logger.info(f"TD-n mode: {'MC (td_n=None)' if config.td_n is None else f'TD-{config.td_n}'}")
    dataset = dataset.traj_map(
        AddBatchKeys(
            td_n=config.td_n,
            gamma=config.discount,
            max_cameras=config.max_cameras,
        ).map
    )
    
    # Flatten episodes to individual frames
    dataset = dataset.flatten()

    # Apply per-frame transforms: decode and resize images
    dataset = dataset.frame_map(
        ImageResizeTransform(
            target_size=config.image_size,
            max_cameras=config.max_cameras,
        ).map
    )

    # Frame-level shuffle if requested (separate from episode-level shuffle above)
    if config.shuffle:
        dataset = dataset.shuffle(config.shuffle_buffer_size, seed=config.seed)

    # Batch the data
    dataset = dataset.batch(config.batch_size, drop_remainder=config.drop_remainder)

    # Set RAM budget
    dataset.with_ram_budget(1)

    # Create numpy iterator
    numpy_iterator = dataset.as_numpy_iterator()

    # Wrap with async prefetcher (handles transform, state normalization, and JAX sharding)
    prefetcher = AsyncBatchPrefetcher(
        numpy_iterator,
        buffer_size=config.prefetch_buffer_size,
        max_cameras=config.max_cameras,
        discount=config.discount,
        reward_scale=config.reward_scale,
        reward_bias=config.reward_bias,
        max_token_len=config.max_token_len,
        num_batches=config.num_batches,
        sharding=config.sharding,
        state_norm_stats=config.state_norm_stats,
        use_quantile_norm=config.use_quantile_norm,
        td_n=config.td_n,
        single_host_batch=config.single_host_batch,
        split=config.split,
    )
    prefetcher.start()

    return prefetcher


class RoboCOINDataLoader:
    """Data loader wrapper compatible with the batch_value_learning interface.

    Provides an iterator interface that yields batches for value function training.
    """

    def __init__(self, config: RoboCOINDataLoaderConfig):
        self.config = config
        self._iterator: AsyncBatchPrefetcher | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Create and return the data iterator."""
        if self._iterator is not None:
            self._iterator.stop()
        self._iterator = create_robocoin_data_loader(self.config)
        return self._iterator

    def __del__(self):
        """Cleanup the iterator when the loader is destroyed."""
        if self._iterator is not None:
            self._iterator.stop()
