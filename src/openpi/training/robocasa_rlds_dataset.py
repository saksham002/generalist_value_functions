"""RLDS-based data loader for RoboCasa.

Provides fast RLDS-based data loading for RoboCasa datasets converted via
scripts/convert_robocasa_to_rlds.py. Uses the generic BaseRldsDataset with
RoboCasa-specific trajectory transforms, FPS interpolation (20 → 30 Hz when
configured), and an RL-fields override that mirrors RoboCOIN's 30-fps pattern
so the value head can transfer cleanly during fine-tuning.

RoboCasa RLDS dataset structure (per step):
    observation/robot0_agentview_left:  JPEG bytes (256x256x3)
    observation/robot0_eye_in_hand:     JPEG bytes (256x256x3)
    observation/robot0_agentview_right: JPEG bytes (256x256x3)
    observation/state:                  float32[16]
    action:                             float32[12]
    language_instruction:               text
    reward:                             float32
    is_terminal:                        bool
    is_last:                            bool

Raw state layout (modality.json):
    [0:3]   base_position
    [3:7]   base_rotation                          (quaternion, xyzw)
    [7:10]  end_effector_position_relative
    [10:14] end_effector_rotation_relative         (quaternion, xyzw)
    [14:16] gripper_qpos

Converted state layout (13D, after trajectory_transforms):
    [0:3]   base_position
    [3:6]   base_rotation                          (extrinsic-xyz Euler)
    [6:9]   eef_position
    [9:12]  eef_rotation                           (extrinsic-xyz Euler)
    [12:13] gripper                                (mean of finger positions)

Dataset naming convention: {split}__{category}__{snake_case_task} (e.g.,
target__atomic__pick_place_counter_to_cabinet, target__composite__load_dishwasher).
Data dir points to the root: robocasa_rlds/ (e.g., /data/group_data/rl/datasets/robocasa_rlds).
"""

from collections.abc import Sequence
import dataclasses
import logging

import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.state_action_spaces as state_action_spaces


class RoboCasaRldsDataset(rlds_dataset.BaseRldsDataset):
    """RoboCasa-specific RLDS dataset loader.

    Keeps RoboCasa-specific trajectory key mapping, rotation handling, FPS
    interpolation, and RoboCOIN-style RL-fields semantics. Action chunking and
    image decoding are handled by BaseRldsDataset.

    Decodes 3 cameras: robot0_agentview_left/right and robot0_eye_in_hand,
    remapped to cam_0/cam_1/cam_2 for joint-pipeline compatibility.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[rlds_dataset.RLDSDataset],
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 25_000,
        num_parallel_reads: int = 4,
        num_parallel_calls: int = 4,
        critic_mode: bool = False,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        include_images: bool = True,
        decode_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        max_num_demos: int | None = None,
        image_size: tuple[int, int] | None = None,
        # FPS interpolation kwargs (scoped to RoboCasa). When `interpolation_config` is
        # None, the dataset returns native 20 Hz trajectories unchanged.
        interpolation_config: state_action_spaces.InterpolationConfig | None = None,
        action_space_spec: state_action_spaces.StateActionSpaceSpec | None = None,
        state_space_spec: state_action_spaces.StateActionSpaceSpec | None = None,
        native_fps: float | None = None,
        mask_boundary_actions: bool = True,
        td_n: int = 50,
        **kwargs,
    ):
        # Validation-only kwargs (val_split, val_latent_store_dir) are consumed by the
        # validation dataset factory; accept and ignore them here.
        del kwargs
        if critic_mode:
            logging.info(
                f"RoboCasaRldsDataset: critic mode enabled (discount={discount}, "
                f"reward_scale={reward_scale}, reward_bias={reward_bias})"
            )

        if interpolation_config is not None:
            if action_space_spec is None or state_space_spec is None:
                raise ValueError(
                    "interpolation_config requires action_space_spec and state_space_spec to be set."
                )
            if native_fps is None:
                raise ValueError(
                    "interpolation_config requires native_fps (data has no per-trajectory FPS metadata)."
                )
            logging.info(
                f"RoboCasaRldsDataset: FPS interpolation {native_fps} → "
                f"{interpolation_config.target_fps} Hz (action_horizon_seconds="
                f"{interpolation_config.action_horizon_seconds})"
            )

        if interpolation_config is None or interpolation_config.target_fps != 30.0:
            raise ValueError("RoboCasaRldsDataset only supports target_fps = 30.0")

        # BaseRldsDataset asserts sum(weights) == 1.0 with strict equality.
        # Floating-point accumulation breaks this for many equal-weight datasets
        # (e.g., sum([0.02] * 50) = 1.0000000000000002). Fix the last weight
        # so the sum is exactly 1.0.
        datasets = list(datasets)
        total_weight = sum(d.weight for d in datasets)
        if total_weight != 1.0:
            partial_sum = sum(d.weight for d in datasets[:-1])
            datasets[-1] = dataclasses.replace(datasets[-1], weight = 1.0 - partial_sum)

        if td_n % 5 != 0:
            raise ValueError(f"td_n must be a multiple of 5 (matches RoboCOIN convention), got {td_n}")
        if action_chunk_size % 5 != 0:
            raise ValueError(
                f"action_chunk_size must be a multiple of 5 (matches RoboCOIN convention), got {action_chunk_size}"
            )

        self._interpolation_config = interpolation_config
        self._action_space_spec = action_space_spec
        self._state_space_spec = state_space_spec
        self._native_fps = native_fps
        self._mask_boundary_actions = mask_boundary_actions
        self._td_n = td_n

        super().__init__(
            data_dir = data_dir,
            batch_size = batch_size,
            datasets = datasets,
            shuffle = shuffle,
            action_chunk_size = action_chunk_size,
            shuffle_buffer_size = shuffle_buffer_size,
            num_parallel_reads = num_parallel_reads,
            num_parallel_calls = num_parallel_calls,
            critic_mode = critic_mode,
            discount = discount,
            reward_scale = reward_scale,
            reward_bias = reward_bias,
            image_obs_keys = ("cam_0", "cam_1", "cam_2"),
            image_size = image_size,
            include_images = include_images,
            decode_images = decode_images,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            max_num_demos = max_num_demos,
        )

    def trajectory_transforms(self, traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        import tensorflow as tf

        # Convert raw 16D state to 13D state with extrinsic-xyz Euler rotations.
        # Raw layout: base_pos[0:3], base_quat[3:7] (xyzw), eef_pos[7:10],
        #             eef_quat[10:14] (xyzw), gripper[14:16].
        # Converted: base_pos[0:3], base_euler[3:6], eef_pos[6:9], eef_euler[9:12], gripper[12:13].
        raw_state = traj["observation"]["state"]  # [T, 16]
        base_position = raw_state[:, 0:3]
        base_quat = raw_state[:, 3:7]
        eef_position = raw_state[:, 7:10]
        eef_quat = raw_state[:, 10:14]
        gripper_qpos = raw_state[:, 14:16]

        base_euler = state_action_spaces.quaternion_to_euler_xyz_tf(base_quat)
        eef_euler = state_action_spaces.quaternion_to_euler_xyz_tf(eef_quat)
        gripper = tf.reduce_mean(gripper_qpos, axis = -1, keepdims = True)

        converted_state = tf.concat(
            [base_position, base_euler, eef_position, eef_euler, gripper],
            axis = -1,
        )  # [T, 13]

        # Compose RoboCasa's per-step delta eef action with the current state to get
        # an absolute target in the base-relative frame.
        # Raw action layout: base_motion[0:4], control_mode[4:5], eef_pos[5:8] (delta),
        #                    eef_rot[8:11] (axis-angle delta), gripper_close[11:12].
        # Position composes by addition. Rotation composes via quaternion math:
        #     R_abs = R_delta · R_state, then convert back to extrinsic-xyz Euler.
        action = traj["action"]  # [T, 12]
        action_delta_aa = action[:, 8:11]
        action_delta_quat = state_action_spaces.axis_angle_to_quaternion_tf(action_delta_aa)
        absolute_eef_quat = state_action_spaces.quat_multiply_tf(action_delta_quat, eef_quat)
        absolute_eef_euler = state_action_spaces.quaternion_to_euler_xyz_tf(absolute_eef_quat)

        absolute_action = tf.concat(
            [
                action[:, 0:5],  # base_motion, control_mode (passed through; ignored downstream
                                 # in the bimanual-EEF mapping but retained so the 12D layout
                                 # stays intact for the supervised RoboCasa-native path).
                action[:, 5:8] + eef_position,  # eef_pos: delta + state = absolute
                absolute_eef_euler,             # eef_rot: composed quaternion → euler-xyz
                action[:, 11:12],               # gripper: stays absolute
            ],
            axis = -1,
        )

        observation = {"state": converted_state}
        if self._include_images:
            # Remap to unified cam_* names so frame_transforms / latent store can use them.
            observation["cam_0"] = traj["observation"]["robot0_agentview_left"]
            observation["cam_1"] = traj["observation"]["robot0_agentview_right"]
            observation["cam_2"] = traj["observation"]["robot0_eye_in_hand"]

        mapped_traj = {
            "actions": absolute_action,
            "observation": observation,
            "prompt": traj["language_instruction"],
        }

        # Forward dlimp metadata; cache_val_episodes' frames.sort + AddValidationVariants'
        # seeding both require these.
        for metadata_key in ("_frame_index", "_traj_index"):
            if metadata_key in traj:
                mapped_traj[metadata_key] = traj[metadata_key]

        if self._interpolation_config is not None:
            self._interpolate_trajectory(traj, mapped_traj)

        # Emit per-step repo_id from the dataset name. Used by the validation-cache helper
        # `cache_val_episodes` (robocoin_utils/utils.py:483) to identify trajectories — the
        # helper is RoboCOIN-shaped and assumes this field exists on every trajectory.
        # Built AFTER interpolation so the length matches the (possibly resampled) actions.
        post_traj_len = tf.shape(mapped_traj["actions"])[0]
        mapped_traj["repo_id"] = tf.fill([post_traj_len], dataset_cfg.name)

        mapped_traj["action_mask"] = self._build_action_mask(post_traj_len)

        return mapped_traj

    def _interpolate_trajectory(self, raw_traj: dict, mapped_traj: dict) -> None:
        """Resample mapped_traj (and selected raw_traj fields) from native_fps to target_fps.

        Mutates both dicts in place. Per-dim interpolation semantics come from
        `self._action_space_spec` / `self._state_space_spec` (linear for positions,
        euler-xyz via quat-slerp for rotations, step-hold for grippers). Cameras are
        resampled via nearest-neighbour (no pixel interpolation). raw_traj's done flags
        are gathered along the same nearest_indices so any downstream `_apply_rl_fields`
        path that reads them sees a self-consistent target-FPS trajectory.
        """
        import tensorflow as tf

        source_fps = tf.constant(self._native_fps, dtype = tf.float32)
        target_fps = tf.constant(self._interpolation_config.target_fps, dtype = tf.float32)

        traj_len = tf.shape(mapped_traj["actions"])[0]
        source_times = tf.cast(tf.range(traj_len), tf.float32) / source_fps
        duration = tf.cast(traj_len - 1, tf.float32) / source_fps
        target_len = tf.cast(tf.round(duration * target_fps), tf.int32) + 1
        target_times = tf.cast(tf.range(target_len), tf.float32) / target_fps

        new_actions = state_action_spaces.interpolate_trajectory_tf(
            tf.cast(mapped_traj["actions"], tf.float32),
            source_times,
            target_times,
            self._action_space_spec,
        )
        observation = mapped_traj["observation"]
        new_state = state_action_spaces.interpolate_trajectory_tf(
            tf.cast(observation["state"], tf.float32),
            source_times,
            target_times,
            self._state_space_spec,
        )

        nearest_indices = state_action_spaces.get_nearest_indices(source_times, target_times)
        new_images = {}
        for image_key in self._image_obs_keys:
            if image_key in observation:
                new_images[image_key] = tf.gather(observation[image_key], nearest_indices)

        mapped_traj["actions"] = new_actions
        observation["state"] = new_state
        for image_key, image_val in new_images.items():
            observation[image_key] = image_val

        # Resample done flags and reward in raw_traj. Our overridden `_apply_rl_fields`
        # does not depend on these (it derives termination from steps_to_subtask_end), but
        # keep them consistent for any path that still reads from raw_traj.
        for key in ("is_terminal", "is_last", "termination", "truncation"):
            if key in raw_traj:
                raw_traj[key] = tf.gather(raw_traj[key], nearest_indices)
        if "reward" in raw_traj:
            raw_traj["reward"] = state_action_spaces.interpolate_sparse_rewards_tf(
                tf.cast(raw_traj["reward"], tf.float32), source_times, target_times,
            )

        # Resample per-step prompt if present (scalar prompts left untouched).
        if "prompt" in mapped_traj:
            prompt = mapped_traj["prompt"]
            if len(prompt.shape) > 0:
                mapped_traj["prompt"] = tf.gather(prompt, nearest_indices)

        if "_frame_index" in mapped_traj:
            mapped_traj["_frame_index"] = tf.range(target_len, dtype = mapped_traj["_frame_index"].dtype)
        if "_traj_index" in mapped_traj:
            value = mapped_traj["_traj_index"]
            mapped_traj["_traj_index"] = tf.fill([target_len], value[0]) if len(value.shape) > 0 else value

    def _build_action_mask(self, traj_len, *, td_n_offset = 0):
        """Per-step (T, action_chunk_size) bool mask. Same logic for policy and critic;
        critic passes td_n_offset to shift the boundary for next_action_mask.
        """
        import tensorflow as tf
        i = tf.range(traj_len)
        steps_to_end = traj_len - 1 - i - td_n_offset
        offsets = tf.range(self._action_chunk_size, dtype = tf.int32)
        if self._mask_boundary_actions:
            mask = offsets[None, :] <= steps_to_end[:, None]
        else:
            mask = tf.ones([traj_len, self._action_chunk_size], dtype = tf.bool)
        if self._interpolation_config is not None and self._interpolation_config.target_fps == 30.0:
            valid = 3 * self._action_chunk_size // 5
            fps_mask = tf.sequence_mask(valid, self._action_chunk_size)
            mask = tf.logical_and(mask, fps_mask[None, :])
        return mask

    def _apply_rl_fields(self, raw_traj: dict, mapped_traj: dict, action_chunk_size: int) -> dict:
        """Compute MC return / reward / termination / td_discount RoboCOIN-style.

        Overrides BaseRldsDataset's reverse-scan MC return with the closed-form
        γ^(exp_per_step · steps_to_end) used by RoboCOIN at 30 fps
        (robocoin_rlds_dataset.py:614-650).

        steps_to_subtask_end is steps-from-current-frame to end-of-episode (RoboCasa
        does not have multi-subtask episodes); termination fires inside the n-step
        lookahead window. Truncation is False — assumes success-only training data.

        Next-state machinery (`next_observation`, `next_actions_raw`) follows the
        base class's gather-along-`next_indices` pattern.
        """
        import tensorflow as tf

        traj_len = tf.shape(mapped_traj["actions"])[0]
        i = tf.range(traj_len)
        steps_to_subtask_end = traj_len - 1 - i  # [T] int32
        selected_steps_f = tf.cast(steps_to_subtask_end, tf.float32)

        exp_per_step = tf.constant(5.0, dtype = tf.float32)
        discount = tf.constant(self._discount, dtype = tf.float32)

        mc_return = tf.pow(discount, exp_per_step * selected_steps_f)

        # Mirror robocoin's 30fps branch: td_n is in canonical 50Hz units, so the
        # 30fps-native step count is 3*td_n/5 (robocoin_rlds_dataset.py:637-641).
        td_n_native = 3 * self._td_n // 5
        termination = steps_to_subtask_end < td_n_native
        td_reward = tf.pow(discount, exp_per_step * selected_steps_f)
        reward = tf.where(termination, td_reward, tf.zeros_like(td_reward))

        td_discount_scalar = tf.pow(discount, exp_per_step * tf.cast(td_n_native, tf.float32))
        td_discount = tf.fill([traj_len], td_discount_scalar)

        truncation = tf.zeros([traj_len], dtype = tf.bool)

        # action_mask is built in trajectory_transforms (applies to both modes).
        # next_action_mask shifts the boundary by td_n_native (critic-only).
        next_action_mask = self._build_action_mask(traj_len, td_n_offset = td_n_native)

        if self._interpolation_config is not None:
            fps_value = self._interpolation_config.target_fps
        else:
            fps_value = self._native_fps if self._native_fps is not None else 20.0
        fps_array = tf.fill([traj_len], tf.cast(fps_value, tf.float32))

        # next_observation / next_actions_raw start at i + td_n_native (robocoin
        # uses td_n_native; using action_chunk_size here is wrong when td_n_native != chunk).
        next_indices = tf.minimum(i + td_n_native, traj_len - 1)
        observation = mapped_traj.get("observation", {})
        if not isinstance(observation, dict):
            raise ValueError("trajectory_transforms must produce observation as a dict when critic_mode=True.")
        next_observation = {key: tf.gather(value, next_indices) for key, value in observation.items()}
        next_actions_raw = tf.gather(mapped_traj["actions"], next_indices)

        mapped_traj["reward"] = reward
        mapped_traj["mc_return"] = mc_return
        mapped_traj["termination"] = termination
        mapped_traj["truncation"] = truncation
        mapped_traj["td_discount"] = td_discount
        mapped_traj["steps_to_subtask_end"] = steps_to_subtask_end
        mapped_traj["next_action_mask"] = next_action_mask
        mapped_traj["fps"] = fps_array
        mapped_traj["next_observation"] = next_observation
        mapped_traj["next_actions_raw"] = next_actions_raw
        return mapped_traj

    def _get_per_step_done_flags(self, raw_traj: dict, mapped_traj: dict, per_step_rewards):
        """Kept for back-compat with non-critic paths that may still call into the
        BaseRldsDataset machinery. Critic-mode training uses the `_apply_rl_fields`
        override above and never reads these flags.
        """
        del mapped_traj
        del per_step_rewards
        import tensorflow as tf

        per_step_terminations = tf.cast(raw_traj["is_terminal"], tf.bool)
        is_last = tf.cast(raw_traj["is_last"], tf.bool)
        per_step_truncations = tf.logical_and(is_last, tf.logical_not(per_step_terminations))
        return per_step_terminations, per_step_truncations

    def frame_transforms(self, frame: dict) -> dict:
        """Apply per-frame transforms with latent key remapping for unified camera names."""
        import tensorflow as tf

        # Call base class for image decoding
        frame = super().frame_transforms(frame)

        # Filter out frames with invalid prompts ('null', empty, etc.)
        if "prompt" in frame:
            prompt = frame["prompt"]
            if isinstance(prompt, tf.Tensor):
                is_valid = tf.logical_not(
                    tf.reduce_any(
                        [
                            tf.equal(tf.strings.lower(prompt), "null"),
                            tf.equal(tf.strings.length(prompt), 0),
                        ]
                    )
                )
                frame["_filter_mask"] = is_valid

        # Remap latent keys from original RoboCasa names to unified cam_0/1/2 names.
        latent_key_mapping = {
            "robot0_agentview_left": "cam_0",
            "robot0_agentview_right": "cam_1",
            "robot0_eye_in_hand": "cam_2",
        }
        keys_to_remap = list(frame.keys())
        for key in keys_to_remap:
            for original, unified in latent_key_mapping.items():
                if original in key:
                    new_key = key.replace(original, unified)
                    frame[new_key] = frame.pop(key)
                    break

        return frame
