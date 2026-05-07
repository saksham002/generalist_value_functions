"""FPS-aware interpolation for RLDS datasets (RoboCasa-scoped port).

Stripped port of `origin/main:src/openpi/training/unified_state_action_spaces.py` —
keeps the FPS-aware interpolation core and adds rotation helpers (axis-angle,
euler-xyz, quaternion) needed for the RoboCasa pipeline. Unified-action-space
machinery (`UNIFIED_ACTION_SPACE`, `ROBOCOIN_*_TO_UNIFIED`, `ActionSpaceMapping`)
is intentionally not ported — this branch does not do joint training.

Quaternion convention: scipy / robosuite / ROS xyzw (`quat[..., 0:3]` = (x, y, z)
axis components, `quat[..., 3]` = w scalar). Identity = `(0, 0, 0, 1)`.
"""

import dataclasses
from typing import Literal

import numpy as np
import tensorflow as tf

# =============================================================================
# Data structures (verbatim from origin/main)
# =============================================================================


@dataclasses.dataclass(frozen = True)
class DimensionSpec:
    """Specification for a contiguous range of dimensions.

    Defines how a subset of action/state dimensions should be interpolated.
    """

    name: str
    start_idx: int
    end_idx: int  # exclusive
    interpolation_type: Literal["linear", "slerp", "step", "angular_linear", "euler_xyz"]
    chunk_delta: bool

    def __post_init__(self):
        if self.start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {self.start_idx}")
        if self.end_idx <= self.start_idx:
            raise ValueError(f"end_idx ({self.end_idx}) must be > start_idx ({self.start_idx})")
        if self.interpolation_type == "slerp" and (self.end_idx - self.start_idx) != 4:
            raise ValueError(
                f"SLERP interpolation requires exactly 4 dimensions (quaternion), got {self.end_idx - self.start_idx}"
            )
        if self.interpolation_type == "euler_xyz" and (self.end_idx - self.start_idx) != 3:
            raise ValueError(
                f"euler_xyz dims must be 3-wide (extrinsic xyz Euler angles), got {self.end_idx - self.start_idx}"
            )

    @property
    def dim(self) -> int:
        return self.end_idx - self.start_idx


@dataclasses.dataclass(frozen = True)
class StateActionSpaceSpec:
    """Defines semantic structure of action/state vector."""

    total_dim: int
    dimensions: tuple[DimensionSpec, ...]
    representation: Literal["absolute", "delta"] = "absolute"

    def __post_init__(self):
        covered = [False] * self.total_dim
        for dim_spec in self.dimensions:
            if dim_spec.end_idx > self.total_dim:
                raise ValueError(
                    f"DimensionSpec '{dim_spec.name}' end_idx ({dim_spec.end_idx}) exceeds total_dim ({self.total_dim})"
                )
            for i in range(dim_spec.start_idx, dim_spec.end_idx):
                if covered[i]:
                    raise ValueError(f"Dimension {i} is covered by multiple DimensionSpecs")
                covered[i] = True
        uncovered = [i for i, c in enumerate(covered) if not c]
        if uncovered:
            raise ValueError(f"Dimensions {uncovered} are not covered by any DimensionSpec")

    def get_chunk_delta_dims(self) -> list[tuple[int, int]]:
        return [(d.start_idx, d.end_idx) for d in self.dimensions if d.chunk_delta]


@dataclasses.dataclass(frozen = True)
class InterpolationConfig:
    """Configuration for FPS-aware interpolation."""

    target_fps: float
    action_horizon_seconds: float

    def __post_init__(self):
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")
        if self.action_horizon_seconds <= 0:
            raise ValueError(f"action_horizon_seconds must be > 0, got {self.action_horizon_seconds}")

    @property
    def action_horizon_steps(self) -> int:
        """Effective action chunk length in target_fps frames."""
        return round(self.action_horizon_seconds * self.target_fps)


# =============================================================================
# Quaternion / rotation helpers (xyzw convention throughout)
# =============================================================================


def quaternion_to_axis_angle_tf(quat: tf.Tensor) -> tf.Tensor:
    """Convert [x, y, z, w] quaternion to 3D axis-angle (rotvec form).

    Args:
        quat: [..., 4] tensor with quaternion components [x, y, z, w].

    Returns:
        [..., 3] tensor with axis-angle (axis · angle) representation.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    angle = 2.0 * tf.math.acos(tf.clip_by_value(w, -1.0, 1.0))
    sin_half_angle = tf.math.sin(angle / 2.0)
    safe_sin = tf.where(tf.abs(sin_half_angle) < 1e-8, tf.ones_like(sin_half_angle), sin_half_angle)
    axis = tf.stack([x, y, z], axis = -1) / safe_sin[..., None]
    return axis * angle[..., None]


def axis_angle_to_quaternion_tf(axis_angle: tf.Tensor) -> tf.Tensor:
    """Convert 3D axis-angle (rotvec form) to [x, y, z, w] quaternion.

    Args:
        axis_angle: [..., 3] tensor where direction = unit axis and magnitude = angle.

    Returns:
        [..., 4] unit quaternion in [x, y, z, w] order.
    """
    angle = tf.linalg.norm(axis_angle, axis = -1, keepdims = True)  # [..., 1]
    safe_angle = tf.where(angle < 1e-8, tf.ones_like(angle), angle)
    axis = axis_angle / safe_angle  # [..., 3]; direction is meaningless for zero-angle so set safely
    # For zero-angle, return identity (0,0,0,1).
    half = angle / 2.0
    sin_half = tf.sin(half)
    cos_half = tf.cos(half)
    xyz = axis * sin_half  # [..., 3]
    quat = tf.concat([xyz, cos_half], axis = -1)  # [..., 4]
    # Replace zero-angle entries with exact identity to avoid NaN propagation.
    is_zero = tf.squeeze(angle < 1e-8, axis = -1)  # [...]
    identity = tf.constant([0.0, 0.0, 0.0, 1.0], dtype = quat.dtype)
    identity_broadcast = tf.broadcast_to(identity, tf.shape(quat))
    return tf.where(is_zero[..., None], identity_broadcast, quat)


def quat_multiply_tf(q1: tf.Tensor, q2: tf.Tensor) -> tf.Tensor:
    """Hamilton product of [x, y, z, w] quaternions: q1 · q2 (q1 applied after q2 if used as R).

    Args:
        q1, q2: [..., 4] tensors in xyzw order.

    Returns:
        [..., 4] tensor in xyzw order representing the composed rotation.
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return tf.stack([x, y, z, w], axis = -1)


def quaternion_to_euler_xyz_tf(quat: tf.Tensor) -> tf.Tensor:
    """Convert [x, y, z, w] quaternion to extrinsic-xyz Euler angles (roll, pitch, yaw).

    Output: [..., 3] in radians, where index 0 = rotation about world x, 1 = world y, 2 = world z.
    Matches `scipy.spatial.transform.Rotation.as_euler("xyz", ...)` for unit quaternions.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # Standard formulas for extrinsic xyz (ZYX intrinsic-equivalent) Euler from quaternion.
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = tf.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = tf.clip_by_value(sinp, -1.0, 1.0)
    pitch = tf.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = tf.atan2(siny_cosp, cosy_cosp)
    return tf.stack([roll, pitch, yaw], axis = -1)


def euler_xyz_to_quaternion_tf(euler: tf.Tensor) -> tf.Tensor:
    """Convert extrinsic-xyz Euler angles (roll, pitch, yaw) to [x, y, z, w] quaternion."""
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr = tf.cos(roll * 0.5), tf.sin(roll * 0.5)
    cp, sp = tf.cos(pitch * 0.5), tf.sin(pitch * 0.5)
    cy, sy = tf.cos(yaw * 0.5), tf.sin(yaw * 0.5)
    # Extrinsic xyz: q = qz · qy · qx (note: scipy as_euler("xyz") inverts this).
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return tf.stack([x, y, z, w], axis = -1)


# =============================================================================
# TensorFlow Interpolation Functions (verbatim port from origin/main)
# =============================================================================


def get_nearest_indices(source_times: tf.Tensor, target_times: tf.Tensor) -> tf.Tensor:
    """For each target timestep, find index of nearest source timestep."""
    diffs = tf.abs(target_times[:, None] - source_times[None, :])
    return tf.argmin(diffs, axis = 1, output_type = tf.int32)


def interpolate_linear_tf(
    values: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
) -> tf.Tensor:
    """Linear interpolation."""
    source_len = tf.shape(source_times)[0]
    right_indices = tf.searchsorted(source_times, target_times, side = "right")
    right_indices = tf.minimum(right_indices, source_len - 1)
    left_indices = tf.maximum(right_indices - 1, 0)
    left_values = tf.gather(values, left_indices)
    right_values = tf.gather(values, right_indices)
    left_times = tf.gather(source_times, left_indices)
    right_times = tf.gather(source_times, right_indices)
    time_diff = tf.maximum(right_times - left_times, 1e-8)
    t = tf.clip_by_value((target_times - left_times) / time_diff, 0.0, 1.0)
    t = t[:, None]
    return left_values + t * (right_values - left_values)


def interpolate_step_tf(
    values: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
) -> tf.Tensor:
    """Step (zero-order-hold) interpolation."""
    source_len = tf.shape(source_times)[0]
    indices = tf.searchsorted(source_times, target_times, side = "right") - 1
    indices = tf.clip_by_value(indices, 0, source_len - 1)
    return tf.gather(values, indices)


def interpolate_angular_linear_tf(
    values: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
) -> tf.Tensor:
    """Linear interpolation with [-pi, pi] wraparound (for joint angles)."""
    import math

    source_len = tf.shape(source_times)[0]
    right_indices = tf.searchsorted(source_times, target_times, side = "right")
    right_indices = tf.minimum(right_indices, source_len - 1)
    left_indices = tf.maximum(right_indices - 1, 0)
    left_values = tf.gather(values, left_indices)
    right_values = tf.gather(values, right_indices)
    left_times = tf.gather(source_times, left_indices)
    right_times = tf.gather(source_times, right_indices)
    time_diff = tf.maximum(right_times - left_times, 1e-8)
    t = tf.clip_by_value((target_times - left_times) / time_diff, 0.0, 1.0)
    t = t[:, None]
    diff = right_values - left_values
    diff = tf.math.floormod(diff + math.pi, 2 * math.pi) - math.pi
    result = left_values + t * diff
    return tf.math.floormod(result + math.pi, 2 * math.pi) - math.pi


def interpolate_slerp_tf(
    quaternions: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
) -> tf.Tensor:
    """SLERP interpolation for quaternions (works for either xyzw or wxyz; just composes
    the same antipodal-flip + great-circle interpolation regardless of component order).
    """
    source_len = tf.shape(source_times)[0]
    right_indices = tf.searchsorted(source_times, target_times, side = "right")
    right_indices = tf.minimum(right_indices, source_len - 1)
    left_indices = tf.maximum(right_indices - 1, 0)
    q1 = tf.gather(quaternions, left_indices)
    q2 = tf.gather(quaternions, right_indices)
    left_times = tf.gather(source_times, left_indices)
    right_times = tf.gather(source_times, right_indices)
    time_diff = tf.maximum(right_times - left_times, 1e-8)
    t = tf.clip_by_value((target_times - left_times) / time_diff, 0.0, 1.0)

    dot = tf.reduce_sum(q1 * q2, axis = -1, keepdims = True)
    q2 = tf.where(dot < 0, -q2, q2)
    dot = tf.abs(dot)

    threshold = 0.9995
    linear_mask = dot > threshold

    theta = tf.acos(tf.clip_by_value(dot, -1.0, 1.0))
    sin_theta = tf.sin(theta)
    sin_theta = tf.maximum(sin_theta, 1e-8)

    t = t[:, None]
    s1 = tf.sin((1 - t) * theta) / sin_theta
    s2 = tf.sin(t * theta) / sin_theta
    result_slerp = s1 * q1 + s2 * q2

    result_linear = (1 - t) * q1 + t * q2
    result_linear = result_linear / tf.norm(result_linear, axis = -1, keepdims = True)

    result = tf.where(linear_mask, result_linear, result_slerp)
    return result / tf.norm(result, axis = -1, keepdims = True)


def interpolate_euler_xyz_tf(
    eulers: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
) -> tf.Tensor:
    """Interpolate extrinsic-xyz Euler angles via quaternion SLERP.

    Direct interpolation in Euler space mishandles wraparound and gimbal-lock; convert
    to quaternion (xyzw), SLERP on the unit-quaternion stream, then convert back.
    """
    quats = euler_xyz_to_quaternion_tf(eulers)  # [T_src, 4]
    interp_quats = interpolate_slerp_tf(quats, source_times, target_times)  # [T_tgt, 4]
    return quaternion_to_euler_xyz_tf(interp_quats)  # [T_tgt, 3]


def interpolate_trajectory_tf(
    trajectory: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
    spec: StateActionSpaceSpec,
) -> tf.Tensor:
    """Interpolate full trajectory using appropriate method per dimension.

    Difference from origin/main: the `euler_xyz` path is implemented (main raises
    NotImplementedError) — required because RoboCasa's converted state and action
    use euler-xyz rotations.
    """
    result_parts = []
    sorted_dims = sorted(spec.dimensions, key = lambda d: d.start_idx)

    for dim_spec in sorted_dims:
        dim_values = trajectory[:, dim_spec.start_idx : dim_spec.end_idx]
        if dim_spec.interpolation_type == "linear":
            interp_values = interpolate_linear_tf(dim_values, source_times, target_times)
        elif dim_spec.interpolation_type == "step":
            interp_values = interpolate_step_tf(dim_values, source_times, target_times)
        elif dim_spec.interpolation_type == "angular_linear":
            interp_values = interpolate_angular_linear_tf(dim_values, source_times, target_times)
        elif dim_spec.interpolation_type == "slerp":
            interp_values = interpolate_slerp_tf(dim_values, source_times, target_times)
        elif dim_spec.interpolation_type == "euler_xyz":
            interp_values = interpolate_euler_xyz_tf(dim_values, source_times, target_times)
        else:
            raise ValueError(f"Unknown interpolation type: {dim_spec.interpolation_type}")
        result_parts.append(interp_values)

    return tf.concat(result_parts, axis = -1)


def interpolate_sparse_rewards_tf(
    rewards: tf.Tensor,
    source_times: tf.Tensor,
    target_times: tf.Tensor,
) -> tf.Tensor:
    """Snap each source-step reward > 0 onto the nearest target step (preserves sparsity)."""
    target_len = tf.shape(target_times)[0]
    target_rewards = tf.zeros(target_len, dtype = tf.float32)
    reward_mask = rewards > 0
    reward_indices = tf.where(reward_mask)[:, 0]
    reward_times = tf.gather(source_times, reward_indices)
    diffs = tf.abs(reward_times[:, None] - target_times[None, :])
    nearest_target_indices = tf.argmin(diffs, axis = 1, output_type = tf.int32)
    updates = tf.ones(tf.shape(nearest_target_indices)[0], dtype = tf.float32)
    indices = nearest_target_indices[:, None]
    return tf.tensor_scatter_nd_update(target_rewards, indices, updates)


# =============================================================================
# RoboCasa specs used by `_interpolate_trajectory` (RoboCasa-native shape)
# =============================================================================
#
# Interpolation runs INSIDE `RoboCasaRldsDataset.trajectory_transforms`, BEFORE the
# bimanual-EEF mapping in `data_transforms.inputs`. So at interpolation time the
# action is still 12D RoboCasa-native (`[base_motion:4, control_mode:1, eef_pos:3,
# eef_rot_eulerXYZ:3, gripper:1]`) and the state is the 13D converted layout. The
# specs below match those shapes. Bimanual-EEF specs (separate, below) are kept for
# any future use that wants the 14D layout.

ROBOCASA_NATIVE_STATE_SPEC = StateActionSpaceSpec(
    total_dim = 13,
    dimensions = (
        DimensionSpec("base_position", 0, 3, "linear", chunk_delta = False),
        DimensionSpec("base_rotation", 3, 6, "euler_xyz", chunk_delta = False),
        DimensionSpec("eef_position", 6, 9, "linear", chunk_delta = False),
        DimensionSpec("eef_rotation", 9, 12, "euler_xyz", chunk_delta = False),
        DimensionSpec("gripper", 12, 13, "step", chunk_delta = False),
    ),
)

ROBOCASA_NATIVE_ACTION_SPEC = StateActionSpaceSpec(
    total_dim = 12,
    dimensions = (
        DimensionSpec("base_motion", 0, 4, "linear", chunk_delta = False),
        DimensionSpec("control_mode", 4, 5, "step", chunk_delta = False),
        DimensionSpec("eef_position", 5, 8, "linear", chunk_delta = True),
        DimensionSpec("eef_rotation", 8, 11, "euler_xyz", chunk_delta = True),
        DimensionSpec("gripper", 11, 12, "step", chunk_delta = True),
    ),
    representation = "absolute",  # trajectory_transforms already composed delta → absolute
)


# Bimanual-EEF specs (14D, left:7 + right:7) — kept for downstream uses that operate
# on the post-mapping layout (e.g. `RoboCasaBimanualEEFRLInputs` consumers). Not used
# by `_interpolate_trajectory` directly; see ROBOCASA_NATIVE_*_SPEC above.

ROBOCASA_BIMANUAL_EEF_STATE_SPEC = StateActionSpaceSpec(
    total_dim = 14,
    dimensions = (
        DimensionSpec("left_eef_position", 0, 3, "linear", chunk_delta = False),
        DimensionSpec("left_eef_rotation", 3, 6, "euler_xyz", chunk_delta = False),
        DimensionSpec("left_gripper", 6, 7, "step", chunk_delta = False),
        DimensionSpec("right_eef_position", 7, 10, "linear", chunk_delta = False),
        DimensionSpec("right_eef_rotation", 10, 13, "euler_xyz", chunk_delta = False),
        DimensionSpec("right_gripper", 13, 14, "step", chunk_delta = False),
    ),
)

ROBOCASA_BIMANUAL_EEF_ACTION_SPEC = StateActionSpaceSpec(
    total_dim = 14,
    dimensions = (
        DimensionSpec("left_eef_position", 0, 3, "linear", chunk_delta = True),
        DimensionSpec("left_eef_rotation", 3, 6, "euler_xyz", chunk_delta = True),
        DimensionSpec("left_gripper", 6, 7, "step", chunk_delta = True),
        DimensionSpec("right_eef_position", 7, 10, "linear", chunk_delta = True),
        DimensionSpec("right_eef_rotation", 10, 13, "euler_xyz", chunk_delta = True),
        DimensionSpec("right_gripper", 13, 14, "step", chunk_delta = True),
    ),
    representation = "absolute",
)
