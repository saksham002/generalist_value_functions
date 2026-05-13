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


def _robocasa_observation(
    data: dict,
    *,
    prefix: str,
    action_dim: int,
    model_type: _model.ModelType,
    unified_state_mapping: Any | None,
) -> tuple[np.ndarray, dict, dict]:
    """Shared shape-up for current/next observation in the non-bimanual layout.

    Returns (state_padded, image_dict, image_mask_dict). `prefix` is "" for current,
    "next_" for critic next-state.
    """
    mask_padding = model_type == _model.ModelType.PI0
    state_key = f"{prefix}observation/state"
    base_image_key = f"{prefix}observation/image"
    image_right_key = f"{prefix}observation/image_right"
    wrist_image_key = f"{prefix}observation/wrist_image"

    state = np.asarray(data[state_key], dtype = np.float32)
    if state.shape[-1] == ROBOCASA_RAW_STATE_DIM:
        state = convert_raw_state_to_model_state(state)
    assert state.shape[-1] == ROBOCASA_STATE_DIM, (
        f"Expected 13D converted state or 16D raw state, got {state.shape[-1]}D."
    )
    if unified_state_mapping is not None:
        state = unified_state_mapping.apply_mapping_np(state)
    else:
        state = transforms.pad_to_dim(state, action_dim)

    base_image = _parse_image(data[base_image_key])
    wrist_image = _parse_image(data[wrist_image_key])

    if image_right_key in data:
        image_right = _parse_image(data[image_right_key])
        image = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": image_right,
            "right_wrist_0_rgb": wrist_image,
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
    else:
        image = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
        }
    return state, image, image_mask


@dataclasses.dataclass(frozen = True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Convert RoboCasa observations to model input format.

    Expects 13D converted state (from RLDS trajectory_transforms for training, or from
    `convert_raw_state_to_model_state` for inference). Pads to action_dim and maps
    cameras to the 3-slot image interface. `next_observation/*` and RL fields (reward,
    mc_return, termination, truncation, td_discount, action_mask, next_action_mask, fps)
    are passed through when present — that's the only difference between supervised and
    critic-mode usage.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    # If set, map the 13D converted state to unified space (e.g., 20D) using this mapping
    # instead of zero-padding to action_dim. Used for unified-action-space inference.
    # Annotated as Any because the unified_state_action_spaces module is not present
    # on this branch; pass an instance (with apply_mapping_np) if/when it is ported.
    unified_state_mapping: Any | None = None

    def __call__(self, data: dict) -> dict:
        state, image, image_mask = _robocasa_observation(
            data, prefix = "",
            action_dim = self.action_dim,
            model_type = self.model_type,
            unified_state_mapping = self.unified_state_mapping,
        )
        inputs: dict = {"state": state, "image": image, "image_mask": image_mask}

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)

        if "next_observation/state" in data:
            next_state, next_image, next_image_mask = _robocasa_observation(
                data, prefix = "next_",
                action_dim = self.action_dim,
                model_type = self.model_type,
                unified_state_mapping = self.unified_state_mapping,
            )
            inputs["next_state"] = next_state
            inputs["next_image"] = next_image
            inputs["next_image_mask"] = next_image_mask
        if "next_actions" in data:
            inputs["next_actions"] = transforms.pad_to_dim(data["next_actions"], self.action_dim)

        for rl_key, rl_dtype in (
            ("reward", np.float32),
            ("mc_return", np.float32),
            ("termination", np.bool_),
            ("truncation", np.bool_),
            ("td_discount", np.float32),
            ("action_mask", np.bool_),
            ("next_action_mask", np.bool_),
            ("fps", np.float32),
            ("steps_to_subtask_end", np.int32),
        ):
            if rl_key in data:
                inputs[rl_key] = np.asarray(data[rl_key], dtype = rl_dtype)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for metadata_key in ("_frame_index", "_traj_index", "repo_id"):
            if metadata_key in data:
                inputs[metadata_key] = data[metadata_key]

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


def _to_bimanual_arm_first_state(state: np.ndarray) -> np.ndarray:
    """Map 13D RoboCasa state to arm-first 14D bimanual-EEF layout.

    Output layout (14D):
        [0:3]   = state[..., 6:9]         # eef_position
        [3:6]   = state[..., 9:12]        # eef_rotation (extrinsic-xyz Euler)
        [6:7]   = state[..., 12:13]       # gripper
        [7:14]  = 0                       # padding for the absent second arm
    Drops state[..., 0:6] (base_position, base_rotation).
    """
    assert state.shape[-1] == ROBOCASA_STATE_DIM, (
        f"Expected 13D converted state, got {state.shape[-1]}D"
    )
    arm = state[..., 6:13]  # eef_pos:3, eef_rot:3, gripper:1
    zeros = np.zeros((*state.shape[:-1], 7), dtype = state.dtype)
    return np.concatenate([arm, zeros], axis = -1)


def _to_bimanual_arm_first_action(action: np.ndarray) -> np.ndarray:
    """Map 12D RoboCasa native action to arm-first 14D bimanual-EEF layout.

    Output layout (14D):
        [0:3]   = action[..., 5:8]        # eef_position
        [3:6]   = action[..., 8:11]       # eef_rotation
        [6:7]   = action[..., 11:12]      # gripper
        [7:14]  = 0                       # padding
    Drops action[..., 0:5] (base_motion, control_mode).
    """
    assert action.shape[-1] == ROBOCASA_ACTION_DIM, (
        f"Expected 12D RoboCasa action, got {action.shape[-1]}D"
    )
    arm = np.concatenate(
        [
            action[..., 5:8],   # eef_position
            action[..., 8:11],  # eef_rotation
            action[..., 11:12], # gripper
        ],
        axis = -1,
    )
    zeros = np.zeros((*action.shape[:-1], 7), dtype = action.dtype)
    return np.concatenate([arm, zeros], axis = -1)


def _bimanual_eef_observation(data: dict, prefix: str) -> tuple[np.ndarray, dict, dict]:
    """Shared shape-up for current/next observation in the bimanual-EEF layout.

    Returns (state_14d, image_dict, image_mask_dict). `prefix` is "" for the current
    observation and "next_" for next-state critic inputs.
    """
    state_key = f"{prefix}observation/state"
    base_image_key = f"{prefix}observation/image_right"
    wrist_image_key = f"{prefix}observation/wrist_image"

    state = np.asarray(data[state_key], dtype = np.float32)
    if state.shape[-1] == ROBOCASA_RAW_STATE_DIM:
        state = convert_raw_state_to_model_state(state)
    state_14d = _to_bimanual_arm_first_state(state)

    base_image = _parse_image(data[base_image_key])
    wrist_image = _parse_image(data[wrist_image_key])
    image = {
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": wrist_image,
        "right_wrist_0_rgb": np.zeros_like(base_image),
    }
    image_mask = {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    return state_14d, image, image_mask


@dataclasses.dataclass(frozen = True)
class RoboCasaBimanualEEFInputs(transforms.DataTransformFn):
    """Map RoboCasa observations to arm-first 14D bimanual-EEF layout.

    Drops base_motion + control_mode from actions and base_position + base_rotation from
    state, placing the single arm in the FIRST 7 slots of the 14D vector and zero-padding
    the remaining 7. Same layout for supervised pi-0.5 fine-tuning (matches
    `action_dim_mask = (T)*7 + (F)*25`) and for value-function fine-tuning.

    Output layout (state and action):
        [0:7]   = arm (eef_pos:3, eef_rot:3, gripper:1)
        [7:14]  = 0 (padding)

    Camera mapping (same for current + next observations):
        base_0_rgb        <- observation/image_right    mask=True
        left_wrist_0_rgb  <- observation/wrist_image    mask=True
        right_wrist_0_rgb <- zeros_like(base_0_rgb)     mask=False

    `next_observation/*` and RL fields (reward, mc_return, termination, truncation,
    td_discount, action_mask, next_action_mask, fps) are passed through when present
    — that's the only difference between supervised and critic-mode usage.
    """

    action_dim: int  # Unused (kept for API parity with RoboCasaInputs).
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state, image, image_mask = _bimanual_eef_observation(data, prefix = "")
        inputs: dict = {"state": state, "image": image, "image_mask": image_mask}

        if "actions" in data:
            inputs["actions"] = _to_bimanual_arm_first_action(
                np.asarray(data["actions"], dtype = np.float32)
            )

        if f"next_observation/state" in data:
            next_state, next_image, next_image_mask = _bimanual_eef_observation(data, prefix = "next_")
            inputs["next_state"] = next_state
            inputs["next_image"] = next_image
            inputs["next_image_mask"] = next_image_mask
        if "next_actions" in data:
            inputs["next_actions"] = _to_bimanual_arm_first_action(
                np.asarray(data["next_actions"], dtype = np.float32)
            )

        for rl_key, rl_dtype in (
            ("reward", np.float32),
            ("mc_return", np.float32),
            ("termination", np.bool_),
            ("truncation", np.bool_),
            ("td_discount", np.float32),
            ("action_mask", np.bool_),
            ("next_action_mask", np.bool_),
            ("fps", np.float32),
            ("steps_to_subtask_end", np.int32),
        ):
            if rl_key in data:
                inputs[rl_key] = np.asarray(data[rl_key], dtype = rl_dtype)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for metadata_key in ("_frame_index", "_traj_index", "repo_id"):
            if metadata_key in data:
                inputs[metadata_key] = data[metadata_key]

        return inputs


@dataclasses.dataclass(frozen = True)
class RoboCasaBimanualEEFOutputs(transforms.DataTransformFn):
    """Convert model outputs from arm-first 14D bimanual-EEF layout back to RoboCasa-native 12D actions.

    The model emits a 14D action where the right arm sits in the first 7 slots
    (mirror of `RoboCasaBimanualEEFInputs`); we drop the trailing 7 zero placeholders
    and reassemble the RoboCasa-native layout. Base motion stays zero-filled —
    bimanual-EEF fine-tunes intentionally do not predict it. Control mode is
    set to -1, which signals arm-only mode to the RoboCasa env: the controller
    ignores base_motion and executes only the eef + gripper sub-action. All of
    the current target tasks (close_blender_lid, etc.) are arm-only manipulation,
    so -1 is the correct value to emit here.

    RoboCasa native 12D layout:
        [0:4]   base_motion       <- 0
        [4:5]   control_mode      <- -1   (arm-only mode)
        [5:8]   eef_position      <- arm_first[..., 0:3]
        [8:11]  eef_rotation      <- arm_first[..., 3:6]
        [11:12] gripper           <- arm_first[..., 6:7]
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        # actions shape: [..., action_horizon, 14] (arm-first bimanual EEF) or larger if
        # upstream padding was not stripped — slice the last dim defensively.
        eef_pos = actions[..., 0:3]
        eef_rot = actions[..., 3:6]
        gripper = actions[..., 6:7]
        # base_motion stays zero; control_mode = -1 means arm-only mode
        # (current target tasks are all arm-only manipulation).
        base_motion = np.zeros((*actions.shape[:-1], 4), dtype = actions.dtype)
        control_mode = -np.ones((*actions.shape[:-1], 1), dtype = actions.dtype)
        robocasa_actions = np.concatenate(
            [base_motion, control_mode, eef_pos, eef_rot, gripper], axis = -1,
        )
        return {"actions": robocasa_actions}


