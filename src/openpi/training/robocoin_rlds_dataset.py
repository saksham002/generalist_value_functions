"""RLDS-based data loader for RoboCOIN.

RoboCOIN dataset structure:
    observation/state: 14D float32 (joint positions and gripper)
    observation/image/cam_0: base camera JPEG
    observation/image/cam_1: left wrist camera JPEG
    observation/image/cam_2: right wrist camera JPEG
    action: 14D float32 (absolute joint actions)
    eef_sim_pose_state: 14D EEF pose (optional, for use_eef=True)
    eef_sim_pose_action: 14D EEF actions (optional)
    subtask_1..subtask_5: subtask text strings
    steps_to_subtask_end: [5] int32, steps until each subtask completes
    first_null_index: int32, number of valid subtasks (0-5)

Key features:
    - Multi-subtask episodes: up to 5 subtasks per episode
    - Per-frame subtask sampling during training
    - Next-state indexing controlled by td_n with fps-aware offsets
    - Optional EEF action representation (14D) with joint-angle state
"""

from collections.abc import Sequence
import logging
from typing import Any

import openpi.training.rlds_dataset as rlds_dataset


class RoboCoinRldsDataset(rlds_dataset.BaseRldsDataset):
    """RoboCOIN-specific RLDS dataset loader.

    Handles RoboCOIN-specific trajectory key mapping and frame-level subtask selection.
    Image decoding is handled by BaseRldsDataset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[rlds_dataset.RLDSDataset],
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 42,
        action_chunk_size: int = 25,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        critic_mode: bool = True,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        use_eef: bool = False,
        td_n: int | None = None,
        filter_n: int | None = None,
        mask_50fps: bool = False,
        use_chunk_wise_delta: bool = False,
        include_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        max_num_demos: int | None = None,
        latent_store_dir: str | None = None,
        latent_views: Sequence[rlds_dataset.latent_store.LatentViewConfig] = (),
        counterfactual_action_store_dir: str | None = None,
    ):
        if td_n is not None and td_n % 5 != 0:
            raise ValueError(f"td_n must be a multiple of 5, got {td_n}")
        if action_chunk_size % 5 != 0:
            raise ValueError(f"action_chunk_size must be a multiple of 5, got {action_chunk_size}")
        if filter_n is not None and filter_n % 5 != 0:
            raise ValueError(f"filter_n must be a multiple of 5, got {filter_n}")

        self._use_eef = use_eef
        self._td_n = td_n
        self._filter_n = filter_n
        self._mask_50fps = mask_50fps
        self._use_chunk_wise_delta = use_chunk_wise_delta
        logging.info(
            f"RoboCoinRldsDataset: critic_mode={critic_mode}, discount={discount}, "
            f"reward_scale={reward_scale}, reward_bias={reward_bias}, use_eef={use_eef}, "
            f"td_n={td_n}, filter_n={filter_n}, mask_50fps={mask_50fps}, "
            f"use_chunk_wise_delta={use_chunk_wise_delta}"
        )

        super().__init__(
            data_dir = data_dir,
            batch_size = batch_size,
            datasets = datasets,
            split = split,
            shuffle = shuffle,
            shuffle_seed = shuffle_seed,
            action_chunk_size = action_chunk_size,
            shuffle_buffer_size = shuffle_buffer_size,
            num_parallel_reads = num_parallel_reads,
            num_parallel_calls = num_parallel_calls,
            critic_mode = critic_mode,
            discount = discount,
            reward_scale = reward_scale,
            reward_bias = reward_bias,
            image_obs_keys = ("cam_0", "cam_1", "cam_2"),
            include_images = include_images,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            max_num_demos = max_num_demos,
            latent_store_dir = latent_store_dir,
            latent_views = latent_views,
            counterfactual_action_store_dir = counterfactual_action_store_dir,
        )

    def trajectory_transforms(self, traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Extract trajectory fields used by the frame-level transform."""
        del dataset_cfg
        import tensorflow as tf

        if self._use_eef:
            actions = self._construct_eef_repr(
                tf.cast(traj["action"], tf.float32),
                tf.cast(traj["eef_sim_pose_action"], tf.float32),
            )
        else:
            actions = tf.cast(traj["action"], tf.float32)
        state = tf.cast(traj["observation/state"], tf.float32)
        traj_len = tf.shape(state)[0]
        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
        repo_id = traj["traj_metadata"]["episode_metadata"]["repo_id"]
        embodiment = tf.repeat(self._extract_embodiment(repo_id[0])[None], traj_len)

        observation = {"state": state}
        if self._include_images:
            observation["cam_0"] = traj["observation/image/cam_0"]
            observation["cam_1"] = traj["observation/image/cam_1"]
            observation["cam_2"] = traj["observation/image/cam_2"]

        result = {
            "actions": actions,
            "observation": observation,
            "subtask_1": traj["subtask_1"],
            "subtask_2": traj["subtask_2"],
            "subtask_3": traj["subtask_3"],
            "subtask_4": traj["subtask_4"],
            "subtask_5": traj["subtask_5"],
            "steps_to_subtask_end": traj["steps_to_subtask_end"],
            "first_null_index": traj["first_null_index"],
            "fps": fps,
            "repo_id": repo_id,
            "embodiment": embodiment,
        }
        for key in ("index", "episode_index", "_frame_index", "_traj_index"):
            if key in traj:
                result[key] = traj[key]

        # Include video latents if present (for video prediction mode)
        # Cast to bfloat16 to reduce shuffle buffer memory (~400KB -> ~200KB per sample)
        if "video_latents" in traj:
            result["video_latents"] = tf.cast(traj["video_latents"], tf.bfloat16)

        return result

    @staticmethod
    def _construct_eef_repr(data, eef_data):
        """Construct 14D EEF representation from EEF pose and gripper values."""
        import tensorflow as tf

        return tf.concat(
            [
                eef_data[..., :6],
                data[..., 6:7],
                eef_data[..., 6:12],
                data[..., 13:14],
            ],
            axis = -1,
        )

    @staticmethod
    def _construct_eef_state(state, eef_state):
        """Compatibility alias for the 14D EEF representation helper."""
        return RoboCoinRldsDataset._construct_eef_repr(state, eef_state)

    @staticmethod
    def _extract_embodiment(repo_id):
        """Extract the embodiment name from a RoboCOIN repo identifier."""
        import tensorflow as tf

        repo_name = tf.strings.split(repo_id, sep = "/")[-1]
        parts = tf.strings.split(repo_name, sep = "_")
        return tf.strings.reduce_join(parts[:2], separator = "_")

    def _compute_next_indices(self, traj_len, fps):
        import tensorflow as tf

        frame_indices = tf.range(traj_len, dtype = tf.int32)
        if self._td_n is None:
            next_offset = 1
        else:
            next_offset = tf.where(tf.equal(fps, 30), 3 * self._td_n // 5, self._td_n)
        return tf.minimum(frame_indices + next_offset, traj_len - 1)

    def _prepare_trajectory(self, raw_traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Attach next-step fields and subtask-conditioned action masks before flattening."""
        import tensorflow as tf

        mapped_traj = self.trajectory_transforms(raw_traj, dataset_cfg)
        for key, value in raw_traj.items():
            if key.startswith(("latents/", "_latent")):
                mapped_traj[key] = value
        if "counterfactual_actions" in raw_traj:
            mapped_traj["counterfactual_actions"] = raw_traj["counterfactual_actions"]
        for key in ("_ca_episode_index",):
            if key in raw_traj:
                mapped_traj[key] = raw_traj[key]

        mapped_traj = self._apply_latent_views(mapped_traj)

        traj_len = tf.shape(mapped_traj["actions"])[0]
        fps = tf.cast(mapped_traj["fps"][0], tf.int32)
        next_indices = self._compute_next_indices(traj_len, fps)

        mapped_traj["next_observation"] = {
            key: tf.gather(value, next_indices) for key, value in mapped_traj["observation"].items()
        }
        mapped_traj["next_actions_raw"] = tf.gather(mapped_traj["actions"], next_indices)

        offsets = tf.range(self._action_chunk_size, dtype = tf.int32)
        steps = tf.cast(mapped_traj["steps_to_subtask_end"], tf.int32)
        mapped_traj["action_mask"] = offsets[None, None, :] <= steps[:, :, None]
        next_steps = tf.gather(steps, next_indices)
        mapped_traj["next_action_mask"] = offsets[None, None, :] <= next_steps[:, :, None]

        if "counterfactual_actions" in mapped_traj:
            mapped_traj["counterfactual_next_actions"] = tf.gather(mapped_traj["counterfactual_actions"], next_indices)

        if self._latent_views and self._latent_manifest is not None:
            for view_config in self._latent_views:
                if view_config.direction == "past":
                    image_keys = view_config.image_keys or self._latent_manifest.image_keys
                    for image_key in image_keys:
                        key = f"{view_config.output_key}_{image_key}"
                        if key in mapped_traj:
                            mapped_traj[f"next_{key}"] = tf.gather(mapped_traj[key], next_indices)

        return mapped_traj

    def _restructure_images(self, frame: dict[str, Any], prefix: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
        """Move camera images from observation dicts into the standard image/image_mask format."""
        images = {}
        masks = {}
        image_name_map = {
            "cam_0": "base_0_rgb",
            "cam_1": "left_wrist_0_rgb",
            "cam_2": "right_wrist_0_rgb",
        }
        observation_key = "observation" if prefix == "" else "next_observation"
        if observation_key not in frame:
            return images, masks

        for key, value in frame[observation_key].items():
            if key == "state":
                continue
            image_key = image_name_map[key]
            images[image_key] = value
            masks[image_key] = True

        return images, masks

    def frame_transforms(self, frame: dict) -> dict:
        """Decode images, select a subtask, and produce the final per-frame batch keys."""
        import tensorflow as tf

        frame = super().frame_transforms(frame)

        images, image_masks = self._restructure_images(frame)
        next_images, next_image_masks = self._restructure_images(frame, prefix = "next_")

        frame["state"] = tf.cast(frame["observation"]["state"], tf.float32)
        frame["next_state"] = tf.cast(frame["next_observation"]["state"], tf.float32)
        del frame["observation"]
        del frame["next_observation"]

        frame["actions"] = tf.cast(frame["actions"], tf.float32)
        frame["next_actions"] = tf.cast(frame["next_actions"], tf.float32)
        if self._use_chunk_wise_delta:
            frame["actions"] = frame["actions"] - frame["actions"][:1, :]
            frame["next_actions"] = frame["next_actions"] - frame["next_actions"][:1, :]

        if images:
            frame["image"] = images
            frame["image_mask"] = image_masks
        if next_images:
            frame["next_image"] = next_images
            frame["next_image_mask"] = next_image_masks

        first_null_index = tf.cast(frame["first_null_index"], tf.int32)
        steps_all = tf.cast(frame["steps_to_subtask_end"], tf.int32)

        if self._return_trajectories:
            sampled_idx = tf.constant(0, dtype = tf.int32)
        else:
            safe_upper = tf.maximum(first_null_index, 1)
            sampled_idx = tf.random.uniform([], minval = 0, maxval = safe_upper, dtype = tf.int32)

        selected_steps = steps_all[sampled_idx]
        selected_steps_f = tf.cast(selected_steps, tf.float32)
        loss_mask = first_null_index > 0
        if self._mask_50fps:
            loss_mask = tf.logical_and(loss_mask, tf.equal(tf.cast(frame["fps"], tf.int32), 30))

        subtask_texts = tf.stack(
            [
                frame["subtask_1"],
                frame["subtask_2"],
                frame["subtask_3"],
                frame["subtask_4"],
                frame["subtask_5"],
            ]
        )
        stripped_subtask_texts = tf.strings.regex_replace(subtask_texts, r"\.\s*$", "")
        lower_subtask_texts = tf.strings.lower(stripped_subtask_texts)
        valid_subtasks = tf.range(5) < first_null_index
        include_subtasks = (
            valid_subtasks
            & tf.not_equal(lower_subtask_texts, b"static")
            & tf.not_equal(lower_subtask_texts, b"abnormal")
        )

        if self._critic_mode:
            frame["prompt"] = subtask_texts[sampled_idx]
        else:
            selected_texts = tf.boolean_mask(stripped_subtask_texts, include_subtasks)
            frame["prompt"] = tf.strings.reduce_join(selected_texts, separator = ", ")
            masked_steps = tf.where(include_subtasks, steps_all, tf.int32.max)
            sampled_idx = tf.argmin(masked_steps, output_type = tf.int32)
            selected_steps = steps_all[sampled_idx]
            selected_steps_f = tf.cast(selected_steps, tf.float32)

        loss_mask = tf.logical_and(loss_mask, include_subtasks[sampled_idx])
        frame["steps_to_subtask_end"] = selected_steps
        frame["sampled_index"] = sampled_idx
        frame["loss_mask"] = loss_mask

        frame["action_mask"] = frame["action_mask"][sampled_idx]
        frame["next_action_mask"] = frame["next_action_mask"][sampled_idx]

        fps = tf.cast(frame["fps"], tf.int32)
        exponent_per_step = tf.cast(tf.where(tf.equal(fps, 30), 5, 3), tf.float32)
        frame["mc_return"] = tf.pow(self._discount, exponent_per_step * selected_steps_f)

        td_n = self._td_n if self._td_n is not None else 0
        frame["td_discount"] = tf.pow(self._discount, 3.0 * tf.cast(td_n, tf.float32))

        if self._td_n is not None:
            td_n_native = tf.where(tf.equal(fps, 30), 3 * td_n // 5, td_n)
            termination = selected_steps < td_n_native
            td_reward = tf.pow(self._discount, exponent_per_step * selected_steps_f)
            frame["termination"] = termination
            frame["reward"] = tf.where(termination, td_reward, 0.0)
        else:
            frame["termination"] = tf.equal(selected_steps, 0)
            frame["reward"] = tf.cast(frame["termination"], tf.float32)

        frame["truncation"] = tf.constant(False)

        is_30fps = tf.equal(fps, 30)
        action_horizon = tf.shape(frame["action_mask"])[0]
        valid_30fps_actions = 3 * action_horizon // 5
        fps_mask_30 = tf.sequence_mask(valid_30fps_actions, action_horizon)
        frame["action_mask"] = tf.where(is_30fps, tf.logical_and(frame["action_mask"], fps_mask_30), frame["action_mask"])
        frame["next_action_mask"] = tf.where(
            is_30fps,
            tf.logical_and(frame["next_action_mask"], fps_mask_30),
            frame["next_action_mask"],
        )

        return frame

    def _apply_frame_transforms_to_trajectory(self, traj: dict) -> dict:
        """Apply frame transforms to a trajectory and then apply the same frame filter."""
        import tensorflow as tf

        traj = super()._apply_frame_transforms_to_trajectory(traj)
        if self._filter_n is None:
            return traj

        fps = tf.cast(traj["fps"], tf.int32)
        filter_n_native = tf.where(tf.equal(fps, 30), 3 * self._filter_n // 5, self._filter_n)
        mask = traj["steps_to_subtask_end"] >= filter_n_native
        return tf.nest.map_structure(lambda x: tf.boolean_mask(x, mask), traj)

    def frame_filter(self, frame: dict) -> bool:
        """Filter frames by the selected subtask horizon when filter_n is configured."""
        import tensorflow as tf

        if self._filter_n is None:
            return tf.constant(value = True)
        fps = tf.cast(frame["fps"], tf.int32)
        filter_n_native = tf.where(tf.equal(fps, 30), 3 * self._filter_n // 5, self._filter_n)
        return frame["steps_to_subtask_end"] >= filter_n_native
