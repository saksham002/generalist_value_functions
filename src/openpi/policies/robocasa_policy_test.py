"""Tests for RoboCasa policy input/output transforms."""

import numpy as np

from openpi.models import model as _model
from openpi.policies import robocasa_policy


def _make_converted_example() -> dict:
    """Create test data with 13D converted state (as if from RLDS pipeline)."""
    return {
        "observation/state": np.random.rand(robocasa_policy.ROBOCASA_STATE_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
        "observation/wrist_image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
        "prompt": "do something",
    }


class TestStateConversion:
    def test_quaternion_to_euler_xyz(self):
        """Test quaternion → extrinsic-xyz Euler conversion (xyzw storage)."""
        # Identity quaternion [x=0, y=0, z=0, w=1] -> zero Euler [0, 0, 0]
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
        result = robocasa_policy.quaternion_to_euler_xyz(identity_quat)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0], decimal = 6)

        # 90° rotation about z-axis: [x=0, y=0, z=sin(45°), w=cos(45°)] -> [0, 0, π/2]
        angle = np.pi / 2
        z_rot_quat = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])
        result = robocasa_policy.quaternion_to_euler_xyz(z_rot_quat)
        expected = np.array([0.0, 0.0, angle])
        np.testing.assert_array_almost_equal(result, expected, decimal = 6)

        # 90° rotation about x-axis: [x=sin(45°), y=0, z=0, w=cos(45°)] -> [π/2, 0, 0]
        x_rot_quat = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])
        result = robocasa_policy.quaternion_to_euler_xyz(x_rot_quat)
        expected = np.array([angle, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal = 6)

    def test_convert_raw_state_shape(self):
        """Test that raw 16D state is converted to 13D."""
        raw_state = np.random.rand(16).astype(np.float32)
        # Set quaternions to valid unit quaternions (xyzw identity)
        raw_state[3:7] = [0.0, 0.0, 0.0, 1.0]  # base rotation
        raw_state[10:14] = [0.0, 0.0, 0.0, 1.0]  # eef rotation

        converted = robocasa_policy.convert_raw_state_to_model_state(raw_state)
        assert converted.shape == (13,)

    def test_convert_raw_state_batched(self):
        """Test batched conversion."""
        raw_state = np.random.rand(10, 16).astype(np.float32)
        raw_state[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
        raw_state[:, 10:14] = [0.0, 0.0, 0.0, 1.0]

        converted = robocasa_policy.convert_raw_state_to_model_state(raw_state)
        assert converted.shape == (10, 13)


class TestRoboCasaInputs:
    def test_state_padding_pi0(self):
        """13D converted state should be padded to action_dim (32) for π₀."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0)
        data = _make_converted_example()
        result = transform(data)

        assert result["state"].shape == (32,)
        # First 13 dims should be from the converted state
        np.testing.assert_array_equal(result["state"][:13], data["observation/state"])
        # Remaining dims should be zero-padded
        np.testing.assert_array_equal(result["state"][13:], 0.0)

    def test_camera_mapping(self):
        """Check that cameras are mapped to the correct model image slots."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0)
        data = _make_converted_example()
        result = transform(data)

        assert set(result["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        # base_0_rgb should be the agentview_left image
        np.testing.assert_array_equal(result["image"]["base_0_rgb"], data["observation/image"])
        # left_wrist_0_rgb should be the wrist image
        np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], data["observation/wrist_image"])
        # right_wrist_0_rgb should be zeros
        np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros_like(data["observation/image"]))

    def test_image_mask_pi0(self):
        """For π₀, right_wrist_0_rgb mask should be False (it's a padding image)."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0)
        data = _make_converted_example()
        result = transform(data)

        assert result["image_mask"]["base_0_rgb"] == np.True_
        assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
        assert result["image_mask"]["right_wrist_0_rgb"] == np.False_

    def test_image_mask_pi0_fast(self):
        """For π₀-FAST, all image masks should be True."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0_FAST)
        data = _make_converted_example()
        result = transform(data)

        assert result["image_mask"]["right_wrist_0_rgb"] == np.True_

    def test_action_padding(self):
        """Actions (12D) should be padded to action_dim (32)."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0)
        data = _make_converted_example()
        data["actions"] = np.random.rand(50, 12).astype(np.float32)
        result = transform(data)

        assert result["actions"].shape == (50, 32)
        np.testing.assert_array_equal(result["actions"][:, :12], data["actions"])
        np.testing.assert_array_equal(result["actions"][:, 12:], 0.0)

    def test_prompt_passthrough(self):
        """Prompt should be passed through unchanged."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0)
        data = _make_converted_example()
        result = transform(data)

        assert result["prompt"] == data["prompt"]

    def test_float_image_conversion(self):
        """Float32 (C,H,W) images should be converted to uint8 (H,W,C)."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32, model_type = _model.ModelType.PI0)
        data = {
            "observation/state": np.random.rand(13).astype(np.float32),  # 13D converted state
            "observation/image": np.random.rand(3, 224, 224).astype(np.float32),
            "observation/wrist_image": np.random.rand(3, 224, 224).astype(np.float32),
            "prompt": "test",
        }
        result = transform(data)

        assert result["image"]["base_0_rgb"].dtype == np.uint8
        assert result["image"]["base_0_rgb"].shape == (224, 224, 3)


class TestRoboCasaOutputs:
    def test_action_extraction(self):
        """Output should extract first 12 actions from padded model output."""
        transform = robocasa_policy.RoboCasaOutputs()
        data = {"actions": np.random.rand(50, 32).astype(np.float32)}
        result = transform(data)

        assert result["actions"].shape == (50, 12)
        np.testing.assert_array_equal(result["actions"], data["actions"][:, :12])


class TestRoboCasaEefRotEulerXYZToAxisAngle:
    def test_identity_passes_through(self):
        """Zero euler-xyz on eef_rot maps to zero axis-angle. Other slots untouched."""
        transform = robocasa_policy.RoboCasaEefRotEulerXYZToAxisAngle()
        actions = np.zeros((30, 12), dtype = np.float32)
        actions[:, 0:4] = [0.1, 0.2, 0.3, 0.4]   # base_motion (untouched)
        actions[:, 4:5] = -1.0                    # control_mode (untouched)
        actions[:, 5:8] = 0.5                     # eef_pos (untouched)
        actions[:, 11:12] = 1.0                   # gripper (untouched)
        # eef_rot at [8:11] left at zeros = identity rotation
        result = transform({"actions": actions})["actions"]
        assert result.shape == (30, 12)
        np.testing.assert_array_equal(result[:, 0:4], actions[:, 0:4])
        np.testing.assert_array_equal(result[:, 4:5], actions[:, 4:5])
        np.testing.assert_array_equal(result[:, 5:8], actions[:, 5:8])
        np.testing.assert_array_almost_equal(result[:, 8:11], 0.0, decimal = 6)
        np.testing.assert_array_equal(result[:, 11:12], actions[:, 11:12])

    def test_z_axis_90deg_roundtrip(self):
        """Euler-xyz [0, 0, π/2] -> axis-angle [0, 0, π/2] (rotation about world z)."""
        transform = robocasa_policy.RoboCasaEefRotEulerXYZToAxisAngle()
        actions = np.zeros((5, 12), dtype = np.float32)
        actions[:, 8:11] = [0.0, 0.0, np.pi / 2]
        result = transform({"actions": actions})["actions"]
        np.testing.assert_array_almost_equal(result[:, 8:11], [[0.0, 0.0, np.pi / 2]] * 5, decimal = 6)


class TestRoboCasaAbsoluteToDelta:
    def test_subtracts_state_from_eef_slots(self):
        """Should subtract state's right_eef_position/rotation from unified action slots."""
        transform = robocasa_policy.RoboCasaAbsoluteToDelta()
        state = np.zeros(20, dtype = np.float32)
        state[7:10] = [0.5, 0.2, 0.1]  # right_eef_position
        state[10:13] = [0.3, -0.1, 0.4]  # right_eef_rotation
        actions = np.zeros((30, 19), dtype = np.float32)
        actions[:, 7:10] = [1.5, 1.2, 1.1]  # absolute eef_pos
        actions[:, 10:13] = [0.4, 0.0, 0.5]  # absolute eef_rot
        actions[:, 13:14] = 1.0  # gripper (unchanged)
        actions[:, 14:18] = [0.1, 0.0, 0.0, 0.0]  # base_motion (unchanged)
        actions[:, 18:19] = -1.0  # control_mode (unchanged)

        result = transform({"state": state, "actions": actions})

        horizon = result["actions"].shape[0]
        # eef slots: absolute - state = delta
        np.testing.assert_allclose(result["actions"][:, 7:10], np.tile([1.0, 1.0, 1.0], (horizon, 1)))
        np.testing.assert_allclose(result["actions"][:, 10:13], np.tile([0.1, 0.1, 0.1], (horizon, 1)))
        # Other slots unchanged
        np.testing.assert_allclose(result["actions"][:, 13:14], np.full((horizon, 1), 1.0))
        np.testing.assert_allclose(result["actions"][:, 14:18], np.tile([0.1, 0.0, 0.0, 0.0], (horizon, 1)))
        np.testing.assert_allclose(result["actions"][:, 18:19], np.full((horizon, 1), -1.0))

    # NOTE: The on-main test_roundtrip_with_chunkwise_delta test is omitted because it
    # depends on `openpi.training.unified_state_action_spaces`, which is part of the
    # joint-training stack and not present on this branch.


class TestRoboCasaBimanualEEFInputs:
    """Tests for the supervised bimanual-EEF transform.

    Maps RoboCasa single-arm + base data into the RoboCOIN bimanual 14D layout with
    left-arm zeros, base/control_mode dropped, and bimanual 3-cam layout (left wrist masked).
    """

    def test_state_layout(self):
        """13D RoboCasa state -> 14D bimanual-EEF arm-first (arm in [0:7], padding in [7:14]=0)."""
        transform = robocasa_policy.RoboCasaBimanualEEFInputs(action_dim = 32)
        state = np.arange(13, dtype = np.float32)  # base[0:6], eef_pos[6:9], eef_rot[9:12], grip[12:13]
        data = {
            "observation/state": state,
            "observation/image_right": np.zeros((224, 224, 3), dtype = np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype = np.uint8),
        }
        result = transform(data)

        assert result["state"].shape == (14,)
        np.testing.assert_array_equal(result["state"][0:3], state[6:9])    # eef_pos
        np.testing.assert_array_equal(result["state"][3:6], state[9:12])   # eef_rot
        np.testing.assert_array_equal(result["state"][6:7], state[12:13])  # gripper
        np.testing.assert_array_equal(result["state"][7:14], 0.0)          # padding

    def test_state_accepts_raw_16d(self):
        """Raw 16D state should be auto-converted to 13D, then mapped to 14D bimanual."""
        transform = robocasa_policy.RoboCasaBimanualEEFInputs(action_dim = 32)
        raw_state = np.zeros(16, dtype = np.float32)
        raw_state[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
        raw_state[10:14] = [1.0, 0.0, 0.0, 0.0]
        data = {
            "observation/state": raw_state,
            "observation/image_right": np.zeros((224, 224, 3), dtype = np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype = np.uint8),
        }
        result = transform(data)
        assert result["state"].shape == (14,)

    def test_action_layout(self):
        """12D RoboCasa action -> arm-first 14D bimanual-EEF, dropping base_motion + control_mode."""
        transform = robocasa_policy.RoboCasaBimanualEEFInputs(action_dim = 32)
        # action: [base_motion:4, control_mode:1, eef_pos:3, eef_rot:3, gripper:1]
        actions = np.arange(12 * 50, dtype = np.float32).reshape(50, 12)
        data = {
            "observation/state": np.zeros(13, dtype = np.float32),
            "observation/image_right": np.zeros((224, 224, 3), dtype = np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype = np.uint8),
            "actions": actions,
        }
        result = transform(data)

        assert result["actions"].shape == (50, 14)
        np.testing.assert_array_equal(result["actions"][:, 0:3], actions[:, 5:8])    # eef_pos
        np.testing.assert_array_equal(result["actions"][:, 3:6], actions[:, 8:11])   # eef_rot
        np.testing.assert_array_equal(result["actions"][:, 6:7], actions[:, 11:12])  # gripper
        np.testing.assert_array_equal(result["actions"][:, 7:14], 0.0)               # padding

    def test_camera_mapping_and_masks(self):
        """agentview_right -> base_0_rgb, eye_in_hand -> left_wrist_0_rgb, right_wrist_0_rgb masked."""
        transform = robocasa_policy.RoboCasaBimanualEEFInputs(action_dim = 32)
        right = np.full((224, 224, 3), 7, dtype = np.uint8)
        wrist = np.full((224, 224, 3), 9, dtype = np.uint8)
        data = {
            "observation/state": np.zeros(13, dtype = np.float32),
            "observation/image_right": right,
            "observation/wrist_image": wrist,
        }
        result = transform(data)

        assert set(result["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        np.testing.assert_array_equal(result["image"]["base_0_rgb"], right)
        np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], wrist)
        np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros_like(right))
        assert result["image_mask"]["base_0_rgb"] == np.True_
        assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
        assert result["image_mask"]["right_wrist_0_rgb"] == np.False_


class TestRoboCasaBimanualEEFOutputs:
    def test_inverse_layout(self):
        """Arm-first 14D bimanual action -> 12D RoboCasa native (base/control_mode = 0)."""
        transform = robocasa_policy.RoboCasaBimanualEEFOutputs()
        actions = np.zeros((50, 14), dtype = np.float32)
        actions[:, 0:3] = 1.0  # eef_pos
        actions[:, 3:6] = 2.0  # eef_rot
        actions[:, 6:7] = 3.0  # gripper
        result = transform({"actions": actions})

        assert result["actions"].shape == (50, 12)
        np.testing.assert_array_equal(result["actions"][:, 0:4], 0.0)  # base_motion
        np.testing.assert_array_equal(result["actions"][:, 4:5], 0.0)  # control_mode
        np.testing.assert_array_equal(result["actions"][:, 5:8], 1.0)  # eef_pos
        np.testing.assert_array_equal(result["actions"][:, 8:11], 2.0)  # eef_rot
        np.testing.assert_array_equal(result["actions"][:, 11:12], 3.0)  # gripper


class TestRoboCasaBimanualEEFInputsCriticMode:
    def test_full_pipeline(self):
        """Same transform handles critic mode by passing through next_* and RL fields."""
        transform = robocasa_policy.RoboCasaBimanualEEFInputs(action_dim = 14)
        state = np.arange(13, dtype = np.float32)
        next_state = np.arange(13, 26, dtype = np.float32)
        actions = np.arange(50 * 12, dtype = np.float32).reshape(50, 12)
        next_actions = actions + 1
        right = np.full((224, 224, 3), 7, dtype = np.uint8)
        wrist = np.full((224, 224, 3), 9, dtype = np.uint8)
        data = {
            "observation/state": state,
            "observation/image_right": right,
            "observation/wrist_image": wrist,
            "next_observation/state": next_state,
            "next_observation/image_right": right,
            "next_observation/wrist_image": wrist,
            "actions": actions,
            "next_actions": next_actions,
            "reward": 1.0,
            "mc_return": 0.99,
            "termination": True,
            "truncation": False,
            "td_discount": 0.86,
            "action_mask": np.ones(50, dtype = np.bool_),
            "next_action_mask": np.ones(50, dtype = np.bool_),
            "fps": 30.0,
            "prompt": "stack the bowls",
        }
        result = transform(data)

        # Arm-first state/action layout for current and next.
        assert result["state"].shape == (14,)
        assert result["next_state"].shape == (14,)
        np.testing.assert_array_equal(result["state"][0:7], state[6:13])
        np.testing.assert_array_equal(result["state"][7:14], 0.0)
        np.testing.assert_array_equal(result["next_state"][0:7], next_state[6:13])
        np.testing.assert_array_equal(result["next_state"][7:14], 0.0)

        assert result["actions"].shape == (50, 14)
        assert result["next_actions"].shape == (50, 14)
        np.testing.assert_array_equal(result["actions"][:, 0:3], actions[:, 5:8])
        np.testing.assert_array_equal(result["actions"][:, 3:6], actions[:, 8:11])
        np.testing.assert_array_equal(result["actions"][:, 6:7], actions[:, 11:12])
        np.testing.assert_array_equal(result["actions"][:, 7:14], 0.0)

        # Cameras current + next.
        np.testing.assert_array_equal(result["image"]["base_0_rgb"], right)
        np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], wrist)
        np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros_like(right))
        assert result["image_mask"]["right_wrist_0_rgb"] == np.False_
        np.testing.assert_array_equal(result["next_image"]["base_0_rgb"], right)
        assert result["next_image_mask"]["right_wrist_0_rgb"] == np.False_

        # RL field passthrough.
        assert result["reward"] == np.float32(1.0)
        assert result["mc_return"] == np.float32(0.99)
        assert result["termination"] == np.True_
        assert result["truncation"] == np.False_
        assert result["td_discount"] == np.float32(0.86)
        assert result["action_mask"].shape == (50,)
        assert result["next_action_mask"].shape == (50,)
        assert result["fps"] == np.float32(30.0)
        assert result["prompt"] == "stack the bowls"


class TestRoboCasaInputsCriticMode:
    def test_next_observation_and_rl_fields(self):
        """Same transform handles critic mode by processing next_observation/* and RL fields."""
        transform = robocasa_policy.RoboCasaInputs(action_dim = 32)
        data = {
            "observation/state": np.random.rand(13).astype(np.float32),
            "observation/image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
            "observation/image_right": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
            "observation/wrist_image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
            "next_observation/state": np.random.rand(13).astype(np.float32),
            "next_observation/image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
            "next_observation/image_right": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
            "next_observation/wrist_image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
            "reward": 1.0,
            "mc_return": 0.99,
            "termination": True,
            "truncation": False,
            "actions": np.random.rand(12).astype(np.float32),
            "next_actions": np.random.rand(12).astype(np.float32),
            "prompt": "pick up the cup",
        }
        result = transform(data)

        assert result["state"].shape == (32,)
        assert "base_0_rgb" in result["image"]
        assert result["next_state"].shape == (32,)
        assert "base_0_rgb" in result["next_image"]
        assert result["reward"] == np.float32(1.0)
        assert result["mc_return"] == np.float32(0.99)
        assert result["termination"] == np.True_
        assert result["truncation"] == np.False_
        assert result["actions"].shape == (32,)
        assert result["next_actions"].shape == (32,)
