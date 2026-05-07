"""RoboCasa policy input/output transforms.

Maps RoboCasa observations and actions to the standardized model input format
expected by π₀ / π₀-FAST.

Raw state from environment (16D, from modality.json):
    [0:3]   base_position
    [3:7]   base_rotation (quaternion [x,y,z,w])
    [7:10]  end_effector_position_relative
    [10:14] end_effector_rotation_relative (quaternion [x,y,z,w])
    [14:16] gripper_qpos (2D finger positions)

Converted state for model (13D, extrinsic-xyz Euler rotations):
    [0:3]   base_position
    [3:6]   base_rotation (extrinsic-xyz Euler — roll/pitch/yaw)
    [6:9]   end_effector_position
    [9:12]  end_effector_rotation (extrinsic-xyz Euler — roll/pitch/yaw)
    [12:13] gripper (mean of finger positions)

The conversion (quaternion→euler-xyz, 2D→1D gripper) is applied:
- During training: in RoboCasaRldsDataset.trajectory_transforms()
- During inference: in RoboCasaInputs.__call__()

Euler-xyz matches `ROBOCOIN_EEF_STATE_SPEC` so the bimanual fine-tune is drop-in
weight-compatible. `transforms.DeltaActions` / `AbsoluteActions` operate natively
on extrinsic-xyz Euler blocks (`rpy_index_start`).

Camera mapping (matches RoboCasaRldsDataset + JointRldsDataConfig training):
    robot0_agentview_left  → base_0_rgb
    robot0_agentview_right → left_wrist_0_rgb
    robot0_eye_in_hand     → right_wrist_0_rgb
"""

import dataclasses
from typing import Any

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

# Raw state dimension from environment
ROBOCASA_RAW_STATE_DIM = 16
# Converted state dimension (after quaternion→euler-xyz conversion)
ROBOCASA_STATE_DIM = 13
ROBOCASA_ACTION_DIM = 12


def quaternion_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] quaternion to extrinsic-xyz Euler angles (3D radian roll/pitch/yaw).

    Storage order is xyzw (scipy / robosuite / ROS convention) — `w = quat[..., 3]`.

    Args:
        quat: [..., 4] array with quaternion components [x, y, z, w]

    Returns:
        [..., 3] array with extrinsic-xyz Euler angles in radians.
    """
    flat = quat.reshape(-1, 4)
    eulers = Rotation.from_quat(flat).as_euler("xyz")
    return eulers.reshape(*quat.shape[:-1], 3)


def convert_raw_state_to_model_state(raw_state: np.ndarray) -> np.ndarray:
    """Convert raw 16D RoboCasa state to 13D model state.

    Converts quaternion rotations to extrinsic-xyz Euler and reduces gripper 2D → 1D.

    Args:
        raw_state: [..., 16] array with raw state from environment

    Returns:
        [..., 13] array with converted state for model
    """
    base_position = raw_state[..., 0:3]
    base_quat = raw_state[..., 3:7]  # [x, y, z, w]
    eef_position = raw_state[..., 7:10]
    eef_quat = raw_state[..., 10:14]  # [x, y, z, w]
    gripper_qpos = raw_state[..., 14:16]

    # Convert quaternions to extrinsic-xyz Euler.
    base_euler = quaternion_to_euler_xyz(base_quat)
    eef_euler = quaternion_to_euler_xyz(eef_quat)

    # Reduce gripper from 2D to 1D (mean of finger positions).
    gripper = np.mean(gripper_qpos, axis = -1, keepdims = True)

    return np.concatenate(
        [base_position, base_euler, eef_position, eef_euler, gripper],
        axis = -1,
    )


def make_robocasa_example() -> dict:
    """Creates a random input example for the RoboCasa policy.

    Returns raw 16D state as would come from the environment.
    """
    return {
        "observation/state": np.random.rand(ROBOCASA_RAW_STATE_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
        "observation/wrist_image": np.random.randint(256, size = (224, 224, 3), dtype = np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen = True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Convert RoboCasa observations to model input format.

    Expects 13D converted state (from RLDS trajectory_transforms for training,
    or from convert_raw_state_to_model_state for inference).
    Pads to action_dim and maps cameras to the 3-slot image interface.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    # If set, map the 13D converted state to unified space (e.g., 20D) using this mapping
    # instead of zero-padding to action_dim. Used for unified-action-space inference.
    # Annotated as Any because the unified_state_action_spaces module is not present
    # on this branch; pass an instance (with apply_mapping_np) if/when it is ported.
    unified_state_mapping: Any | None = None

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        state = np.asarray(data["observation/state"], dtype = np.float32)
        if state.shape[-1] == ROBOCASA_RAW_STATE_DIM:
            state = convert_raw_state_to_model_state(state)
        assert state.shape[-1] == ROBOCASA_STATE_DIM, (
            f"Expected 13D converted state or 16D raw state, got {state.shape[-1]}D."
        )
        if self.unified_state_mapping is not None:
            state = self.unified_state_mapping.apply_mapping_np(state)
        else:
            state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        if "observation/image_right" in data:
            image_right = _parse_image(data["observation/image_right"])
            inputs = {
                "state": state,
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": image_right,
                    "right_wrist_0_rgb": wrist_image,
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                },
            }
        else:
            inputs = {
                "state": state,
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                },
            }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen = True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """Convert model outputs back to RoboCasa action format.

    Extracts the first 12 action dimensions from the padded model output.
    Used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROBOCASA_ACTION_DIM])}


@dataclasses.dataclass(frozen = True)
class RoboCasaEefRotEulerXYZToAxisAngle(transforms.DataTransformFn):
    """Convert the eef_rot slot of a RoboCasa-native action chunk from extrinsic-xyz
    Euler angles to axis-angle (rotvec form).

    Designed to run as the LAST output transform after `RoboCasaBimanualEEFOutputs`
    (or `RoboCasaOutputs`) has produced a 12D RoboCasa-native action layout
    `[base_motion:4, control_mode:1, eef_pos:3, eef_rot:3, gripper:1]`. The model
    trains with euler-xyz rotations (matching `ROBOCOIN_EEF_STATE_SPEC`); the env's
    OSC controller expects axis-angle. This transform bridges the two reps without
    touching any other slot.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"]).copy()
        eef_rot_euler = actions[..., 8:11]
        flat = eef_rot_euler.reshape(-1, 3)
        eef_rot_aa = Rotation.from_euler("xyz", flat).as_rotvec()
        actions[..., 8:11] = eef_rot_aa.reshape(*eef_rot_euler.shape)
        return {**data, "actions": actions}


# Unified action space dimension (bimanual + mobile base)
UNIFIED_ACTION_DIM = 19


@dataclasses.dataclass(frozen = True)
class RoboCasaAbsoluteToDelta(transforms.DataTransformFn):
    """Convert unified absolute right-arm eef pose back to RoboCasa-native delta actions.

    Inverse of the delta-to-absolute conversion applied in
    ``RoboCasaRldsDataset.trajectory_transforms``. Subtracts the unified state's
    ``right_eef_position`` / ``right_eef_rotation`` from the corresponding slots
    of the unified 19D action. ``state`` is expected in the unnormalized unified
    coordinate frame (i.e., this transform must run after ``Unnormalize`` and
    after ``InverseChunkWiseDeltaActions``).
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"]).copy()
        state = np.asarray(data["state"])
        actions[..., 7:10] -= state[..., 7:10]  # eef_pos: absolute -> delta
        actions[..., 10:13] -= state[..., 10:13]  # eef_rot: absolute -> delta
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen = True)
class RoboCasaUnifiedOutputs(transforms.DataTransformFn):
    """Convert unified 19D actions back to RoboCasa 12D format.

    Maps from the unified action space (used in joint RoboCOIN+RoboCasa training)
    back to RoboCasa's native action format for simulator evaluation.

    Unified 19D layout:
        [0:3]   left_eef_position (unused for RoboCasa)
        [3:6]   left_eef_rotation (unused for RoboCasa)
        [6:7]   left_gripper (unused for RoboCasa)
        [7:10]  right_eef_position -> eef_position
        [10:13] right_eef_rotation -> eef_rotation
        [13:14] right_gripper -> gripper_close
        [14:18] base_motion -> base_motion
        [18:19] control_mode -> control_mode

    RoboCasa 12D layout:
        [0:4]   base_motion
        [4:5]   control_mode
        [5:8]   eef_position
        [8:11]  eef_rotation
        [11:12] gripper_close
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        # actions shape: [action_horizon, unified_action_dim]

        robocasa_actions = np.concatenate(
            [
                actions[:, 14:18],  # base_motion [0:4]
                actions[:, 18:19],  # control_mode [4:5]
                actions[:, 7:10],  # eef_position [5:8]
                actions[:, 10:13],  # eef_rotation [8:11]
                actions[:, 13:14],  # gripper_close [11:12]
            ],
            axis = -1,
        )
        return {"actions": robocasa_actions}


# Bimanual-EEF action layout dimension (RoboCOIN convention: [left:7, right:7]).
BIMANUAL_EEF_ACTION_DIM = 14


def _to_bimanual_right_arm_state(state: np.ndarray) -> np.ndarray:
    """Map a 13D RoboCasa converted state into the 14D bimanual-EEF layout (right arm only).

    Layout produced (14D):
        [0:7]   = 0                       # left arm placeholder (zeros)
        [7:10]  = state[..., 6:9]         # right eef_position
        [10:13] = state[..., 9:12]        # right eef_rotation
        [13:14] = state[..., 12:13]       # right gripper
    Drops state[..., 0:6] (base_position, base_rotation).
    """
    assert state.shape[-1] == ROBOCASA_STATE_DIM, (
        f"Expected 13D converted state, got {state.shape[-1]}D"
    )
    zeros_left = np.zeros((*state.shape[:-1], 7), dtype = state.dtype)
    right_arm = state[..., 6:13]  # eef_pos:3, eef_rot:3, gripper:1
    return np.concatenate([zeros_left, right_arm], axis = -1)


def _to_bimanual_right_arm_action(action: np.ndarray) -> np.ndarray:
    """Map a 12D RoboCasa native action into the 14D bimanual-EEF layout (right arm only).

    Drops action[..., 0:5] (base_motion, control_mode). Output layout (14D):
        [0:7]   = 0                       # left arm placeholder
        [7:10]  = action[..., 5:8]        # right eef_position
        [10:13] = action[..., 8:11]       # right eef_rotation
        [13:14] = action[..., 11:12]      # right gripper
    """
    assert action.shape[-1] == ROBOCASA_ACTION_DIM, (
        f"Expected 12D RoboCasa action, got {action.shape[-1]}D"
    )
    zeros_left = np.zeros((*action.shape[:-1], 7), dtype = action.dtype)
    right_arm = np.concatenate(
        [
            action[..., 5:8],   # eef_position
            action[..., 8:11],  # eef_rotation
            action[..., 11:12], # gripper
        ],
        axis = -1,
    )
    return np.concatenate([zeros_left, right_arm], axis = -1)


@dataclasses.dataclass(frozen = True)
class RoboCasaBimanualEEFInputs(transforms.DataTransformFn):
    """Map RoboCasa observations into the RoboCOIN bimanual-EEF layout for supervised
    fine-tuning of bimanual configs (e.g. pi-0.5 with action_dim_offset=14).

    Drops base_motion + control_mode from action and base_position + base_rotation from
    state, placing the single arm into the right-arm slot of the 14D bimanual-EEF
    convention. Cameras follow the RoboCOIN bimanual 3-cam layout with the left wrist
    masked out (RoboCasa is single-arm).

    Action: 12D RoboCasa native -> 14D bimanual EEF (right arm only). NOT padded to
    model action_dim — downstream PadStatesAndActions handles the offset insertion.

    Camera mapping:
        base_0_rgb        <- observation/image_right    (agentview_right)   mask=True
        left_wrist_0_rgb  <- observation/wrist_image    (eye_in_hand)       mask=True
        right_wrist_0_rgb <- zeros_like(base_0_rgb)                          mask=False
    """

    action_dim: int  # Unused (kept for API parity with RoboCasaInputs).
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype = np.float32)
        if state.shape[-1] == ROBOCASA_RAW_STATE_DIM:
            state = convert_raw_state_to_model_state(state)
        bimanual_state = _to_bimanual_right_arm_state(state)

        base_image = _parse_image(data["observation/image_right"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": bimanual_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _to_bimanual_right_arm_action(
                np.asarray(data["actions"], dtype = np.float32)
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen = True)
class RoboCasaBimanualEEFOutputs(transforms.DataTransformFn):
    """Convert model outputs from bimanual-EEF layout back to RoboCasa-native 12D actions.

    The model emits a 14D bimanual action ([left:7, right:7]); we drop the left arm
    slot and reassemble the RoboCasa-native layout. Base motion and control mode are
    zero-filled — bimanual-EEF fine-tunes intentionally do not predict them.

    RoboCasa native 12D layout:
        [0:4]   base_motion       <- 0
        [4:5]   control_mode      <- 0
        [5:8]   eef_position      <- bimanual[..., 7:10]
        [8:11]  eef_rotation      <- bimanual[..., 10:13]
        [11:12] gripper           <- bimanual[..., 13:14]
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        # actions shape: [..., action_horizon, 14] (bimanual EEF) or larger if upstream
        # padding was not stripped — slice the last dim defensively.
        eef_pos = actions[..., 7:10]
        eef_rot = actions[..., 10:13]
        gripper = actions[..., 13:14]
        zeros = np.zeros(
            (*actions.shape[:-1], 5),  # base_motion (4) + control_mode (1)
            dtype = actions.dtype,
        )
        robocasa_actions = np.concatenate([zeros, eef_pos, eef_rot, gripper], axis = -1)
        return {"actions": robocasa_actions}


@dataclasses.dataclass(frozen = True)
class RoboCasaRLInputs(transforms.DataTransformFn):
    """Transform RoboCasa observations and RL fields for value function training.

    Handles both current observation (state, image) and next_observation
    (next_state, next_image) for TD learning with image-based value functions.

    Expects 13D converted state from RLDS pipeline (after trajectory_transforms).

    Expected input keys from RLDS dataset (after repack):
        - observation/state (13D), observation/image, observation/image_right, observation/wrist_image
        - next_observation/state (13D), next_observation/image, next_observation/image_right, next_observation/wrist_image
        - actions, next_actions
        - reward, mc_return, termination, truncation
        - prompt
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Process current observation
        base_image = _parse_image(data["observation/image"])
        base_image_right = _parse_image(data["observation/image_right"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        state = np.asarray(data["observation/state"], dtype = np.float32)
        assert state.shape[-1] == ROBOCASA_STATE_DIM, (
            f"Expected 13D converted state from RLDS pipeline, got {state.shape[-1]}D"
        )
        inputs = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": {
                "base_0_rgb": base_image,
                "base_1_rgb": base_image_right,
                "left_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
        }

        # Process next observation
        next_base_image = _parse_image(data["next_observation/image"])
        next_base_image_right = _parse_image(data["next_observation/image_right"])
        next_wrist_image = _parse_image(data["next_observation/wrist_image"])

        next_state = np.asarray(data["next_observation/state"], dtype = np.float32)
        assert next_state.shape[-1] == ROBOCASA_STATE_DIM, (
            f"Expected 13D converted state from RLDS pipeline, got {next_state.shape[-1]}D"
        )
        inputs["next_state"] = transforms.pad_to_dim(next_state, self.action_dim)
        inputs["next_image"] = {
            "base_0_rgb": next_base_image,
            "base_1_rgb": next_base_image_right,
            "left_wrist_0_rgb": next_wrist_image,
        }
        inputs["next_image_mask"] = {
            "base_0_rgb": np.True_,
            "base_1_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
        }

        # Pass through actions
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "next_actions" in data:
            inputs["next_actions"] = transforms.pad_to_dim(data["next_actions"], self.action_dim)

        # Pass through RL fields
        if "reward" in data:
            inputs["reward"] = np.asarray(data["reward"], dtype = np.float32)
        if "mc_return" in data:
            inputs["mc_return"] = np.asarray(data["mc_return"], dtype = np.float32)
        if "termination" in data:
            inputs["termination"] = np.asarray(data["termination"], dtype = np.bool_)
        if "truncation" in data:
            inputs["truncation"] = np.asarray(data["truncation"], dtype = np.bool_)

        # Pass through prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen = True)
class RoboCasaBimanualEEFRLInputs(transforms.DataTransformFn):
    """Critic-mode counterpart of RoboCasaBimanualEEFInputs.

    Maps RoboCasa observations + RL fields (current + next) into the RoboCOIN
    bimanual-EEF layout for value-function fine-tuning (e.g.
    `robocasa_paligemma_q_sarsa_finetune`). Drops base_motion + control_mode from
    actions and base_position + base_rotation from state. The single arm goes to
    the right-arm slot; left-arm slots are zero-filled.

    Actions stay absolute (NO chunk-wise delta is applied in the RoboCasa pipeline;
    `RoboCasaRldsDataset.trajectory_transforms` already converted RoboCasa's native
    delta eef actions to absolute base-frame poses, matching the RoboCOIN bimanual
    EEF convention).

    Camera mapping (matches `RoboCasaBimanualEEFInputs`):
        base_0_rgb        <- observation/image_right    (agentview_right)   mask=True
        left_wrist_0_rgb  <- observation/wrist_image    (eye_in_hand)       mask=True
        right_wrist_0_rgb <- zeros_like(base_0_rgb)                         mask=False
    """

    action_dim: int  # Unused (kept for API parity with RoboCasaRLInputs).

    def __call__(self, data: dict) -> dict:
        # Current observation
        base_image = _parse_image(data["observation/image_right"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype = np.float32)

        inputs = {
            "state": _to_bimanual_right_arm_state(state),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Next observation
        next_base_image = _parse_image(data["next_observation/image_right"])
        next_wrist_image = _parse_image(data["next_observation/wrist_image"])
        next_state = np.asarray(data["next_observation/state"], dtype = np.float32)
        inputs["next_state"] = _to_bimanual_right_arm_state(next_state)
        inputs["next_image"] = {
            "base_0_rgb": next_base_image,
            "left_wrist_0_rgb": next_wrist_image,
            "right_wrist_0_rgb": np.zeros_like(next_base_image),
        }
        inputs["next_image_mask"] = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }

        # Actions
        if "actions" in data:
            inputs["actions"] = _to_bimanual_right_arm_action(
                np.asarray(data["actions"], dtype = np.float32)
            )
        if "next_actions" in data:
            inputs["next_actions"] = _to_bimanual_right_arm_action(
                np.asarray(data["next_actions"], dtype = np.float32)
            )

        # RL fields
        if "reward" in data:
            inputs["reward"] = np.asarray(data["reward"], dtype = np.float32)
        if "mc_return" in data:
            inputs["mc_return"] = np.asarray(data["mc_return"], dtype = np.float32)
        if "termination" in data:
            inputs["termination"] = np.asarray(data["termination"], dtype = np.bool_)
        if "truncation" in data:
            inputs["truncation"] = np.asarray(data["truncation"], dtype = np.bool_)
        if "td_discount" in data:
            inputs["td_discount"] = np.asarray(data["td_discount"], dtype = np.float32)
        if "action_mask" in data:
            inputs["action_mask"] = np.asarray(data["action_mask"], dtype = np.bool_)
        if "next_action_mask" in data:
            inputs["next_action_mask"] = np.asarray(data["next_action_mask"], dtype = np.bool_)
        if "fps" in data:
            inputs["fps"] = np.asarray(data["fps"], dtype = np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for metadata_key in ("_frame_index", "_traj_index", "repo_id"):
            if metadata_key in data:
                inputs[metadata_key] = data[metadata_key]

        return inputs
