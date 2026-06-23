"""Cross-check scipy vs pure-JAX implementations of AbsoluteActions.

Loads norm stats from the RoboCOIN 60Hz dataset, builds a chunk-wise-delta
action tensor from `actions.mean` (mirroring the data pipeline layout for the
14-D EEF action: [left_pos(3), left_rpy(3), left_grip(1), right_pos(3),
right_rpy(3), right_grip(1)]), then runs both AbsoluteActions transforms and
asserts the outputs match.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

import openpi.transforms as _transforms

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pure-JAX rotation helpers (extrinsic-xyz Euler convention, matching
# scipy.spatial.transform.Rotation.from_euler("xyz", ...).
# -----------------------------------------------------------------------------


def _euler_xyz_to_matrix(rpy: jnp.ndarray) -> jnp.ndarray:
    """rpy [..., 3] -> R [..., 3, 3] with R = Rz(c) @ Ry(b) @ Rx(a)."""
    a, b, c = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    ca, sa = jnp.cos(a), jnp.sin(a)
    cb, sb = jnp.cos(b), jnp.sin(b)
    cc, sc = jnp.cos(c), jnp.sin(c)
    row0 = jnp.stack([cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa], axis = -1)
    row1 = jnp.stack([sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa], axis = -1)
    row2 = jnp.stack([-sb,     cb * sa,                 cb * ca],                axis = -1)
    return jnp.stack([row0, row1, row2], axis = -2)


def _matrix_to_euler_xyz(R: jnp.ndarray) -> jnp.ndarray:
    """R [..., 3, 3] -> rpy [..., 3] for extrinsic-xyz Tait-Bryan angles."""
    b = jnp.arcsin(jnp.clip(-R[..., 2, 0], -1.0, 1.0))
    a = jnp.arctan2(R[..., 2, 1], R[..., 2, 2])
    c = jnp.arctan2(R[..., 1, 0], R[..., 0, 0])
    return jnp.stack([a, b, c], axis = -1)


def _apply_rpy_absolute_jax(state: jnp.ndarray, actions: jnp.ndarray, rpy_index_start) -> jnp.ndarray:
    out = actions
    for s in rpy_index_start:
        state_rpy = state[..., s : s + 3]
        action_rpy = out[..., s : s + 3]
        state_rpy_b = jnp.broadcast_to(state_rpy[..., None, :], action_rpy.shape)
        R_state = _euler_xyz_to_matrix(state_rpy_b)
        R_delta = _euler_xyz_to_matrix(action_rpy)
        R_abs = R_delta @ R_state
        out = out.at[..., s : s + 3].set(_matrix_to_euler_xyz(R_abs))
    return out


def absolute_actions_jax(
    state: jnp.ndarray,
    actions: jnp.ndarray,
    mask,
    rpy_index_start,
) -> jnp.ndarray:
    """Pure-JAX equivalent of openpi.transforms.AbsoluteActions for one (state, actions) pair.

    Args:
        state: [..., D] absolute state at chunk start.
        actions: [..., chunk, D] chunk-wise delta actions.
        mask: 1-D bool sequence; True dims add `state` back. Length <= D.
        rpy_index_start: starts of 3-wide extrinsic-xyz euler blocks composed via R_delta @ R_state.
    """
    mask_arr = jnp.asarray(mask, dtype = bool)
    if rpy_index_start is not None:
        for s in rpy_index_start:
            mask_arr = mask_arr.at[s : s + 3].set(False)
    dims = mask_arr.shape[-1]
    out = actions.at[..., :dims].add(
        jnp.expand_dims(jnp.where(mask_arr, state[..., :dims], 0.0), axis = -2)
    )
    if rpy_index_start is not None:
        out = _apply_rpy_absolute_jax(state, out, rpy_index_start)
    return out


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


NORM_STATS_PATH = "gs://saksham-euw4/hdf5/real_hang_60_Hz/norm_stats/norm_stats.json"
DELTA_MASK = _transforms.make_bool_mask(6, -1, 6, -1)
RPY_INDEX_START = (3, 10)


def _load_robocoin_chunkwise_action_mean(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load chunk-wise-delta action mean (14D EEF) and EEF state mean from a flat RoboCOIN
    norm_stats.json, mirroring RoboCoinRldsDataConfig._convert_robocoin_stats with
    use_eef=True, use_chunk_wise_delta=True, subsample=False (no chunk subsampling here).
    """
    import json
    from etils import epath

    raw = json.loads(epath.Path(path).read_text())
    assert "observation.state" in raw, "expected flat single-embodiment RoboCOIN norm_stats"

    eef_action_diff = raw["eef_sim_pose_action_diff"]  # chunk-wise delta, [chunk, 12]
    abs_action = raw["action"]  # absolute, [16] (joint+gripper)
    eef_state = raw["eef_sim_pose_state"]  # absolute EEF state, [12]
    joint_state = raw["observation.state"]  # absolute joint+gripper state, [16]

    eef_diff_mean = np.asarray(eef_action_diff["mean"], dtype = np.float32)
    abs_action_mean = np.asarray(abs_action["mean"], dtype = np.float32)

    # 16D abs action -> grippers at indices 7 (left) and 15 (right) per RoboCoinRldsDataConfig
    raw_action_dim = abs_action_mean.shape[-1]
    left_grip_idx = raw_action_dim // 2 - 1
    right_grip_idx = raw_action_dim - 1
    left_grip = abs_action_mean[..., left_grip_idx : left_grip_idx + 1]
    right_grip = abs_action_mean[..., right_grip_idx : right_grip_idx + 1]
    # Broadcast gripper across chunk axis if eef_diff has a leading chunk dim
    if eef_diff_mean.ndim == 2:
        chunk = eef_diff_mean.shape[0]
        left_grip = np.broadcast_to(left_grip, (chunk, 1)).copy()
        right_grip = np.broadcast_to(right_grip, (chunk, 1)).copy()
        actions_mean = np.concatenate(
            [eef_diff_mean[..., :6], left_grip, eef_diff_mean[..., 6:12], right_grip], axis = -1
        )
    else:
        actions_mean = np.concatenate(
            [eef_diff_mean[:6], left_grip, eef_diff_mean[6:12], right_grip], axis = -1
        )

    eef_state_mean = np.asarray(eef_state["mean"], dtype = np.float32)
    joint_state_mean = np.asarray(joint_state["mean"], dtype = np.float32)
    # 14D EEF state: eef_pose(6) + left_grip + eef_pose(6) + right_grip
    state_left_grip = joint_state_mean[..., joint_state_mean.shape[-1] // 2 - 1 : joint_state_mean.shape[-1] // 2]
    state_right_grip = joint_state_mean[..., -1:]
    state_mean = np.concatenate(
        [eef_state_mean[:6], state_left_grip, eef_state_mean[6:12], state_right_grip], axis = -1
    )
    return actions_mean, state_mean


def main() -> None:
    logger.info(f"Loading raw RoboCOIN norm_stats from {NORM_STATS_PATH} ...")
    actions_mean, state_mean = _load_robocoin_chunkwise_action_mean(NORM_STATS_PATH)
    logger.info(f"actions.mean shape (chunk-wise delta, 14D): {actions_mean.shape}")
    logger.info(f"state.mean shape (14D EEF): {state_mean.shape}")
    assert actions_mean.shape[-1] == 14 and state_mean.shape[-1] == 14

    if actions_mean.ndim == 1:
        action_horizon = 50
        actions_mean = np.broadcast_to(actions_mean, (action_horizon, 14)).copy()
    action_horizon = actions_mean.shape[0]
    logger.info(f"Using action_horizon = {action_horizon}")

    batch_size = 2
    state_batch = np.broadcast_to(state_mean, (batch_size, 14)).copy()
    actions_batch = np.broadcast_to(
        actions_mean, (batch_size, action_horizon, 14)
    ).copy()

    logger.info(f"Inputs: state {state_batch.shape}, actions {actions_batch.shape}")
    logger.info(f"DELTA_MASK = {list(DELTA_MASK)}, rpy_index_start = {RPY_INDEX_START}")

    # ---- scipy/numpy reference path ----
    abs_transform = _transforms.AbsoluteActions(mask = DELTA_MASK, rpy_index_start = RPY_INDEX_START)
    out_scipy = abs_transform({"state": state_batch.copy(), "actions": actions_batch.copy()})["actions"]
    logger.info(f"scipy output shape: {out_scipy.shape}")

    # ---- pure-jax path ----
    out_jax = absolute_actions_jax(
        jnp.asarray(state_batch),
        jnp.asarray(actions_batch),
        DELTA_MASK,
        RPY_INDEX_START,
    )
    out_jax_np = np.asarray(out_jax)
    logger.info(f"jax output shape: {out_jax_np.shape}")

    # ---- compare ----
    diff = np.abs(out_scipy - out_jax_np)
    max_abs = diff.max()
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    logger.info(f"max |scipy - jax| = {max_abs:.3e} at index {max_idx}")
    logger.info(f"per-dim max |diff|: {diff.max(axis = (0, 1))}")

    rtol, atol = 1e-5, 1e-5
    np.testing.assert_allclose(out_scipy, out_jax_np, rtol = rtol, atol = atol)
    logger.info(f"PASSED: scipy and pure-jax outputs match within rtol={rtol}, atol={atol}.")

    # ---- also run inside jax.jit to confirm the jax path is jit-compatible ----
    jit_fn = jax.jit(
        lambda s, a: absolute_actions_jax(s, a, DELTA_MASK, RPY_INDEX_START)
    )
    out_jit = np.asarray(jit_fn(jnp.asarray(state_batch), jnp.asarray(actions_batch)))
    np.testing.assert_allclose(out_scipy, out_jit, rtol = rtol, atol = atol)
    logger.info("PASSED: jit-compiled jax output matches scipy.")


if __name__ == "__main__":
    main()
