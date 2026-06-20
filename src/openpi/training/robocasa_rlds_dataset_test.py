"""Tests for RoboCasaRldsDataset trajectory transforms and done flag computation."""

import numpy as np
import tensorflow as tf

import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.robocasa_rlds_dataset as robocasa_rlds_dataset


def _make_robocasa_dataset_for_unit_tests():
    dataset = robocasa_rlds_dataset.RoboCasaRldsDataset.__new__(robocasa_rlds_dataset.RoboCasaRldsDataset)
    dataset._discount = 0.99  # noqa: SLF001
    dataset._reward_scale = 1.0  # noqa: SLF001
    dataset._reward_bias = 0.0  # noqa: SLF001
    dataset._critic_mode = True  # noqa: SLF001
    dataset._action_chunk_size = 3  # noqa: SLF001
    dataset._include_images = True  # noqa: SLF001
    dataset._image_obs_keys = ("robot0_agentview_left", "robot0_eye_in_hand")  # noqa: SLF001
    dataset._prompt_mode = "subtask"  # noqa: SLF001
    # Interpolation kwargs default to None for unit tests (no FPS resampling).
    dataset._interpolation_config = None  # noqa: SLF001
    dataset._action_space_spec = None  # noqa: SLF001
    dataset._state_space_spec = None  # noqa: SLF001
    dataset._native_fps = None  # noqa: SLF001
    return dataset


class TestRoboCasaDoneFlags:
    """Tests for RoboCasa-specific done flag logic.

    RoboCasa RLDS datasets have explicit is_terminal and is_last flags:
    - is_terminal = True when the task completed successfully (done=True on last step).
    - Truncation = last step AND NOT terminated.
    """

    def test_successful_episode_termination(self):
        """Episode where task was completed: last step has is_terminal=True, is_last=True."""
        dataset = _make_robocasa_dataset_for_unit_tests()
        raw = {
            "is_terminal": tf.constant([False, False, False, True]),
            "is_last": tf.constant([False, False, False, True]),
        }

        terminations, truncations = dataset._get_per_step_done_flags(  # noqa: SLF001
            raw_traj = raw,
            mapped_traj = {},
            per_step_rewards = tf.constant([0.0, 0.0, 0.0, 1.0]),
        )

        np.testing.assert_array_equal(terminations.numpy(), [False, False, False, True])
        np.testing.assert_array_equal(truncations.numpy(), [False, False, False, False])

    def test_failed_episode_truncation(self):
        """Episode where task was NOT completed: last step has is_terminal=False, is_last=True."""
        dataset = _make_robocasa_dataset_for_unit_tests()
        raw = {
            "is_terminal": tf.constant([False, False, False, False, False]),
            "is_last": tf.constant([False, False, False, False, True]),
        }

        terminations, truncations = dataset._get_per_step_done_flags(  # noqa: SLF001
            raw_traj = raw,
            mapped_traj = {},
            per_step_rewards = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0]),
        )

        np.testing.assert_array_equal(terminations.numpy(), [False, False, False, False, False])
        np.testing.assert_array_equal(truncations.numpy(), [False, False, False, False, True])

    def test_early_termination(self):
        """Episode where success happens before the last step (possible in variable-length episodes)."""
        dataset = _make_robocasa_dataset_for_unit_tests()
        raw = {
            "is_terminal": tf.constant([False, False, True, False, False]),
            "is_last": tf.constant([False, False, True, False, False]),
        }

        terminations, truncations = dataset._get_per_step_done_flags(  # noqa: SLF001
            raw_traj = raw,
            mapped_traj = {},
            per_step_rewards = tf.constant([0.0, 0.0, 1.0, 0.0, 0.0]),
        )

        np.testing.assert_array_equal(terminations.numpy(), [False, False, True, False, False])
        # is_last AND NOT terminated at step 2 is False (it IS terminated), so no truncation
        np.testing.assert_array_equal(truncations.numpy(), [False, False, False, False, False])


def _make_valid_robocasa_state(batch_size: int) -> tf.Tensor:
    """Create valid 16D RoboCasa state with unit quaternions (xyzw)."""
    # State layout: base_pos[0:3], base_quat[3:7], eef_pos[7:10], eef_quat[10:14], gripper[14:16]
    state = np.zeros((batch_size, 16), dtype = np.float32)
    state[:, 0:3] = np.random.rand(batch_size, 3)  # base_pos
    state[:, 3:7] = [0.0, 0.0, 0.0, 1.0]  # base_quat (xyzw identity)
    state[:, 7:10] = np.random.rand(batch_size, 3)  # eef_pos
    state[:, 10:14] = [0.0, 0.0, 0.0, 1.0]  # eef_quat (xyzw identity)
    state[:, 14:16] = np.random.rand(batch_size, 2)  # gripper
    return tf.constant(state)


class TestRoboCasaTrajectoryTransforms:
    def test_trajectory_transforms_output_keys(self):
        """Verify trajectory_transforms produces the expected key structure with unified camera names."""
        dataset = _make_robocasa_dataset_for_unit_tests()
        raw = {
            "action": tf.constant([[1.0] * 12, [2.0] * 12]),
            "observation": {
                "robot0_agentview_left": tf.constant([b"jpeg1", b"jpeg2"]),
                "robot0_agentview_right": tf.constant([b"jpeg3", b"jpeg4"]),
                "robot0_eye_in_hand": tf.constant([b"jpeg5", b"jpeg6"]),
                "state": _make_valid_robocasa_state(2),
            },
            "language_instruction": tf.constant("pick up the cup"),
        }

        dataset_cfg = rlds_dataset.RLDSDataset(name = "target__atomic__dummy", version = "1.0.0", weight = 1.0)
        mapped = dataset.trajectory_transforms(raw, dataset_cfg = dataset_cfg)

        assert set(mapped.keys()) == {"actions", "observation", "prompt"}
        # Camera keys are remapped to unified names: cam_0, cam_1, cam_2
        assert set(mapped["observation"].keys()) == {"cam_0", "cam_1", "cam_2", "state"}
        # State is converted from 16D (quaternion) to 13D (axis-angle)
        assert mapped["observation"]["state"].shape == (2, 13)

    def test_rl_mode_output_keys_regression(self):
        """Verify all expected keys are present after trajectory_transforms + RL fields + chunking."""
        dataset = _make_robocasa_dataset_for_unit_tests()
        raw = {
            "action": tf.cast(tf.range(10), tf.float32)[:, None] * tf.ones([1, 12]),
            "observation": {
                "robot0_agentview_left": tf.cast(tf.range(10) + 100, tf.float32),
                "robot0_eye_in_hand": tf.cast(tf.range(10) + 200, tf.float32),
                "robot0_agentview_right": tf.cast(tf.range(10) + 300, tf.float32),
                "state": _make_valid_robocasa_state(10),
            },
            "language_instruction": "task",
            "reward": tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "is_terminal": tf.constant([False, False, False, False, False, False, False, True, False, False]),
            "is_last": tf.constant([False, False, False, False, False, False, False, True, False, False]),
        }

        dataset_cfg = rlds_dataset.RLDSDataset(name = "target__atomic__dummy", version = "1.0.0", weight = 1.0)
        mapped = dataset.trajectory_transforms(raw, dataset_cfg = dataset_cfg)
        mapped = dataset._apply_rl_fields(raw, mapped, action_chunk_size = 3)  # noqa: SLF001
        mapped = dataset._chunk_actions(mapped, action_chunk_size = 3)  # noqa: SLF001

        expected_keys = {
            "actions",
            "observation",
            "prompt",
            "reward",
            "mc_return",
            "termination",
            "truncation",
            "td_discount",                # added by RoboCOIN-style _apply_rl_fields override
            "steps_to_subtask_end",       # added by RoboCOIN-style _apply_rl_fields override
            "next_observation",
            "next_actions",
        }
        assert set(mapped.keys()) == expected_keys
        assert "next_actions_raw" not in mapped
        # Camera keys are remapped to unified names
        assert set(mapped["next_observation"].keys()) == {"cam_0", "cam_1", "cam_2", "state"}
        assert mapped["actions"].shape == (10, 3, 12)
        assert mapped["next_actions"].shape == (10, 3, 12)
        assert mapped["reward"].shape == (10,)
        assert mapped["mc_return"].shape == (10,)
        assert mapped["termination"].shape == (10,)
        assert mapped["truncation"].shape == (10,)

    def test_trajectory_transforms_excludes_images_when_disabled(self):
        """Verify trajectory_transforms can emit state-only observations when images are disabled."""
        dataset = _make_robocasa_dataset_for_unit_tests()
        dataset._include_images = False  # noqa: SLF001
        raw = {
            "action": tf.constant([[1.0] * 12, [2.0] * 12]),
            "observation": {
                "robot0_agentview_left": tf.constant([b"jpeg1", b"jpeg2"]),
                "robot0_agentview_right": tf.constant([b"jpeg3", b"jpeg4"]),
                "robot0_eye_in_hand": tf.constant([b"jpeg5", b"jpeg6"]),
                "state": _make_valid_robocasa_state(2),
            },
            "language_instruction": tf.constant("pick up the cup"),
        }

        dataset_cfg = rlds_dataset.RLDSDataset(name = "target__atomic__dummy", version = "1.0.0", weight = 1.0)
        mapped = dataset.trajectory_transforms(raw, dataset_cfg = dataset_cfg)

        assert set(mapped["observation"].keys()) == {"state"}
        assert mapped["observation"]["state"].shape == (2, 13)
