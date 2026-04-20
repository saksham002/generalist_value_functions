"""Eval script for shirt-hang task with local policy inference.

Loads a trained policy checkpoint, connects to a remote robot environment
server, and runs episodes by querying the policy locally and sending actions
to the robot.

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run eval/xarm_scripts/eval_shirt_hang.py \
        --args.config-name robocoin_bimanual_pi05 \
        --args.checkpoint-dir /path/to/checkpoint \
        --args.robot-host robot-machine
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import requests
from scipy.spatial.transform import Rotation
import tyro

import openpi.models.model as _model
from openpi.models.best_of_n import BestOfNWrapper
from openpi.shared.normalize import NormStats
from openpi.value_functions import base_value_functions as _base_vf

import ipdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================


@dataclasses.dataclass
class Args:
    config_name: str
    """TrainConfig name registered in config.py."""

    checkpoint_dir: str
    """Path to the policy checkpoint directory."""

    robot_host: str = "localhost"
    """Host running robot_environment_server.py."""

    robot_port: int = 8081
    """Port for the robot environment server."""

    fine_tune_config: str | None = None
    """Optional FineTuneConfig name from config.py. If set, applies overrides to the base config."""

    step: int | None = None
    """Checkpoint step number. If None, loads the latest checkpoint."""

    num_episodes: int = 1
    """Number of episodes to run."""

    control_freq: int = 30
    """Frequency (Hz) at which the robot server expects actions. Passed to the
    RemoteEnvironmentAdapter so the server initializes its control loop accordingly."""

    query_freq: int = 15
    """How many env steps between policy replans."""

    max_steps: int = 3600
    """Maximum env steps per episode."""

    real_action_start: int = 0
    """Index into the policy action vector where real actions begin."""

    real_action_dim: int = 14
    """Number of real action dimensions."""

    camera_names: tuple[str, ...] = ("right/top", "left/wrist", "right/wrist")
    """Camera names as returned by the robot environment server."""


    debug: bool = False
    """Debug mode: skip policy loading, save camera images to disk and print state instead."""

    debug_output_dir: str = ""
    """Directory to save debug images when --debug is set. Defaults to eval/xarm_scripts/debug/."""

    critic_config: str | None = None
    """TrainConfig name for an optional Q-function critic. If set, BestOfN value-guided action selection is used."""

    critic_checkpoint: str | None = None
    """Path to the critic checkpoint directory. Required when critic_config is set."""

    critic_fine_tune_config: str | None = None
    """Optional FineTuneConfig name for the critic."""

    critic_step: int | None = None
    """Critic checkpoint step number. If None, loads the latest checkpoint."""

    num_samples: int = 8
    """Number of action candidates sampled per replan step when using BestOfN."""


# =============================================================================
# Critic loading
# =============================================================================


def load_critic(
    config_name: str,
    checkpoint_path: str,
    fine_tune_config: str | None,
    step: int | None,
) -> tuple[nnx.Module, dict[str, NormStats]]:
    """Load a value-function (critic) checkpoint for BestOfN action selection.

    Mirrors the loading pattern in scripts/evaluate_value_function.py: imports
    train_value_function.py to build the train state shape, restores params with
    explicit shardings, and merges into an nnx module. Norm stats are loaded
    from the checkpoint's saved assets directory.
    """
    from openpi.robocoin_utils.load_model_utils import load_critic as _load_critic

    critic_model, critic_norm_stats, _, _ = _load_critic(
        config_name,
        checkpoint_path,
        fine_tune = fine_tune_config,
        step = step,
    )
    return critic_model, critic_norm_stats


# =============================================================================
# Local policy
# =============================================================================


class LocalPolicy:
    """Loads a policy checkpoint and runs inference locally."""

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str,
        step: int | None = None,
        fine_tune_config: str | None = None,
        critic_model: nnx.Module | None = None,
        critic_norm_stats: dict[str, NormStats] | None = None,
        num_samples: int = 8,
    ) -> None:
        import openpi.policies.policy as _policy_module
        import openpi.shared.nnx_utils as nnx_utils
        import openpi.transforms as _transforms
        from openpi.robocoin_utils.load_model_utils import LoadPolicyConfig, load_policy
        from openpi.training import checkpoints as _checkpoints

        logger.info(f"Loading checkpoint from {checkpoint_dir} (step={step})...")
        load_config = LoadPolicyConfig(
            config_name=config_name,
            checkpoint_path=checkpoint_dir,
            fine_tune=fine_tune_config,
            step=step,
        )
        model, config = load_policy(load_config)
        logger.info("Checkpoint restored successfully.")

        policy_model_config = config.policy if config.policy is not None else config.model
        logger.info(f"discrete_state_input: {getattr(policy_model_config, 'discrete_state_input', 'N/A')}")

        logger.info("Loading normalization statistics from checkpoint assets...")
        data_config = config.data.create(config.assets_dirs, policy_model_config)
        asset_id = data_config.asset_id
        if step is not None:
            step_dir = str(step)
        else:
            # Find the latest step directory.
            step_dirs = sorted(
                (d for d in os.listdir(checkpoint_dir) if d.isdigit()),
                key = int,
            )
            step_dir = step_dirs[-1]
            logger.info(f"No step specified, using latest: {step_dir}")
        norm_stats_dir = os.path.join(checkpoint_dir, step_dir, "assets", asset_id)
        logger.info(f"Norm stats directory: {norm_stats_dir}")
        from openpi.shared import normalize as _normalize
        all_norm_stats = _normalize.load(norm_stats_dir)
        _INFERENCE_KEYS = {"state", "actions", "next_state", "next_actions"}
        norm_stats = {k: v for k, v in all_norm_stats.items() if k in _INFERENCE_KEYS}
        logger.info("Norm stats loaded. Building policy transforms...")

        policy = _policy_module.Policy(
            model,
            transforms=[
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *(
                    [_transforms.Clip(data_config.clip_normalized_bounds)]
                    if data_config.clip_normalized_bounds is not None
                    else []
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            metadata=config.policy_metadata,
        )

        self._model = policy._model  # noqa: SLF001
        self._input_transform = policy._input_transform  # noqa: SLF001
        self._output_transform = policy._output_transform  # noqa: SLF001
        self._sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
        self._rng = policy._rng  # noqa: SLF001

        if hasattr(self._model, "config") and hasattr(self._model.config, "guidance"):
            object.__setattr__(self._model.config, "guidance", 0.0)
            logger.info("Disabled classifier-free guidance (guidance=0.0).")

        self._sample_actions_jit = nnx_utils.module_jit(self._model.sample_actions)

        # Derive the EEF action layout from the loaded policy config so the slice
        # in predict() doesn't depend on hard-coded integers.
        self._action_dim_offset = policy_model_config.action_dim_offset
        self._eef_action_dim = norm_stats["actions"].mean.shape[-1]

        # TODO: Store the VF inside BestOfN and set self._model = BestOfN when a critic
        # is supplied, eliminating the branch in predict() and the @nnx.jit wrapper.
        # Both modules are frozen during eval so module_jit works uniformly.
        # Plan: ~/.claude-personal/plans/starry-dazzling-meteor.md
        self._critic_model = critic_model
        self._bestofn_sample = None
        self._bestofn = None
        if critic_model is not None:
            assert critic_norm_stats is not None, "critic_norm_stats required when critic_model is provided"
            logger.info(
                f"Building BestOfN wrapper with num_samples={num_samples}, "
                f"action_dim_offset={self._action_dim_offset}"
            )
            self._bestofn = BestOfNWrapper(
                action_dim = self._model.action_dim,
                action_horizon = self._model.action_horizon,
                max_token_len = self._model.max_token_len,
                base_model = self._model,
                num_samples = num_samples,
                take_min_over_ensemble = True,
                use_target_value = False,
                convert_to_global = True,
                selection_mode = "argmax",
                softmax_temperature = 1.0,
                policy_norm_stats = norm_stats,
                critic_norm_stats = critic_norm_stats,
                critic_action_dim_offset = self._action_dim_offset,
            )

            @nnx.jit
            def _bestofn_sample(bon, vf, rng, transition):
                return bon.sample_actions(
                    rng, transition, compute_next_action = False, value_function = vf,
                )

            self._bestofn_sample = _bestofn_sample

        logger.info("Policy loaded and JIT-compiled successfully.")

    def predict(self, obs_dict: dict[str, Any], initial_eef_pose: np.ndarray) -> np.ndarray:
        """Run policy inference on a single observation.

        Args:
            obs_dict: Raw observation with keys: image (dict of camera name -> RGB uint8 array),
                state, prompt, embodiment.
            initial_eef_pose: Current 14-D EEF pose in euler format from extract_eef_pose().

        Returns:
            actions: float32 array of shape [30, 14] at the policy's native 30 Hz.
        """
        raw_state = np.asarray(obs_dict["state"], dtype=np.float32)
        transformed = self._input_transform(obs_dict)

        batched = {}
        for k, v in transformed.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                batched[k] = {dk: jnp.asarray(dv)[None, ...] for dk, dv in v.items()}
            else:
                batched[k] = jnp.asarray(v)[None, ...]

        # Mask out the last 20 action positions so the model only attends to the first 30.
        action_horizon = self._model.action_horizon
        action_mask = jnp.concatenate([
            jnp.ones(action_horizon - 20, dtype=jnp.bool_),
            jnp.zeros(20, dtype=jnp.bool_),
        ])[None, :]  # (1, action_horizon)
        batched["action_mask"] = action_mask
        batched["image_mask"] = {k: jnp.array([True]) for k in batched.get("image", {})}

        observation = _model.Observation.from_dict(batched)

        self._rng, sample_rng = jax.random.split(self._rng)

        if self._critic_model is not None:
            # BestOfN path: build a Transition manually with the 14-d initial EEF
            # pose as transition.action so the wrapper can convert chunk-wise-delta
            # candidates back to global before passing them to the critic.
            init_pose = jnp.asarray(initial_eef_pose, dtype = jnp.float32)[None, None, :]
            transition = _base_vf.Transition(observation = observation, action = init_pose)
            actions_out = self._bestofn_sample(
                self._bestofn, self._critic_model, sample_rng, transition,
            )
        else:
            transition = _model.wrap_observation_as_transition(observation)
            actions_out = self._sample_actions_jit(sample_rng, transition, **self._sample_kwargs)
        actions_out = jax.block_until_ready(actions_out)

        start = self._action_dim_offset
        actions_np = np.asarray(
            actions_out[0, :, start : start + self._eef_action_dim]
        )  # [action_horizon, eef_action_dim]
        decoded = self._output_transform({
            "state": raw_state,
            "actions": actions_np,
            "next_state": raw_state,
            "next_actions": actions_np,
        })
        
        # Extract the 14-D EEF action subset, first 30 steps only.
        actions = np.asarray(decoded["actions"], dtype=np.float32)[:30]

        # Policy predicts global actions relative to current state; add current pose to get absolute base frame targets.
        actions += initial_eef_pose

        # ipdb.set_trace()

        return actions


# =============================================================================
# Subtask tracker
# =============================================================================

_SUBTASK_PROMPTS = [
    "Grasp the hanger",
    "Lift the hanger off the rod",
    "Pass hanger from right to left arm",
    "Hook one side of the shirt onto the hanger",
    "Hook the other side of the shirt onto the hanger",
    "Place the hanger on the rod",
]

_GRIP_THRESH = 400
_MIN_BOUNDARY_GAP = 75


class SubtaskTracker:
    """Real-time subtask detector mirroring the heuristics in solve_subtask_boundaries.py.

    Maintains a state machine over the 6 shirt-hang subtasks and advances to the next
    subtask when the corresponding sensor transition is detected from the live observation
    stream. Call update() every env step; read prompt to get the current subtask string.

    Signals used (all from obs["state"]):
        left/gripper_pos, right/gripper_pos  — gripper open/close state
        right/tcp_pose[2]                    — absolute right-arm TCP Z height (metres)
    """

    def __init__(self) -> None:
        self._subtask = 0
        self._steps_in_subtask = 0
        self._step = 0
        self._last_boundary_step = -_MIN_BOUNDARY_GAP - 1

        self._prev_lg_open: bool | None = None
        self._prev_rg_open: bool | None = None

        self._t01_step: int | None = None
        self._t12_step: int | None = None
        self._t23_pending_step: int | None = None
        self._t23_left_close_streak: int = 0
        self._t23_step: int | None = None
        self._t34_step: int | None = None
        self._t45_candidate_step: int | None = None
        self._t45_steps_since_candidate: int = 0
        self._t45_step: int | None = None

    @property
    def subtask(self) -> int:
        return self._subtask

    @property
    def prompt(self) -> str:
        return _SUBTASK_PROMPTS[self._subtask]

    def update(self, obs: dict[str, Any]) -> None:
        state = obs["state"]

        def _scalar(key: str) -> float:
            val = state.get(key)
            if val is None:
                return 0.0
            arr = np.asarray(val, dtype=np.float32)
            return float(arr[-1].flat[0] if arr.ndim > 1 else arr.flat[0])

        def _vec(key: str, dim: int = 3) -> np.ndarray:
            val = state.get(key)
            if val is None:
                return np.zeros(dim, dtype=np.float32)
            arr = np.asarray(val, dtype=np.float32)
            return arr[-1] if arr.ndim > 1 else arr

        lg = _scalar("left/gripper_pos")
        rg = _scalar("right/gripper_pos")
        if "right/tcp_pose" not in state and self._step == 0:
            logger.warning(
                "SubtaskTracker: 'right/tcp_pose' not found in obs['state']. "
                "T0->1 and T1->2 transitions require absolute TCP Z and will not fire. "
                "Available keys: %s", list(state.keys())
            )
        rz = float(_vec("right/tcp_pose", dim=7)[2])

        lg_open = lg > _GRIP_THRESH
        rg_open = rg > _GRIP_THRESH

        if self._subtask < len(_SUBTASK_PROMPTS) - 1:
            self._check_transition(lg, rg, rz, lg_open, rg_open)

        self._prev_lg_open = lg_open
        self._prev_rg_open = rg_open

        self._step += 1
        self._steps_in_subtask += 1

    def _can_accept_boundary(self, boundary_step: int) -> bool:
        return boundary_step - self._last_boundary_step > _MIN_BOUNDARY_GAP

    def _advance(self, boundary_step: int) -> None:
        logger.info(f"SubtaskTracker: subtask {self._subtask} -> {self._subtask + 1} "
                    f"({_SUBTASK_PROMPTS[self._subtask + 1]!r}) at step {self._step} "
                    f"(boundary_step={boundary_step})")
        self._subtask += 1
        self._steps_in_subtask = 0
        self._last_boundary_step = boundary_step
        self._t23_pending_step = None
        self._t23_left_close_streak = 0
        self._t45_candidate_step = None
        self._t45_steps_since_candidate = 0

    def _check_transition(self, lg: float, rg: float, rz: float, lg_open: bool, rg_open: bool) -> None:
        if self._subtask == 0:
            self._check_t01(rz, rg_open)
        elif self._subtask == 1:
            self._check_t12(rz)
        elif self._subtask == 2:
            self._check_t23(lg_open)
        elif self._subtask == 3:
            self._check_t34(rg, lg)
        elif self._subtask == 4:
            self._check_t45(lg_open)

    def _check_t01(self, rz: float, rg_open: bool) -> None:
        right_close_edge = self._prev_rg_open is True and not rg_open
        if right_close_edge and rz > 0.30 and self._can_accept_boundary(self._step):
            self._t01_step = self._step
            self._advance(self._step)

    def _check_t12(self, rz: float) -> None:
        if rz < 0.22 and self._can_accept_boundary(self._step):
            self._t12_step = self._step
            self._advance(self._step)

    def _check_t23(self, lg_open: bool) -> None:
        left_close_edge = self._prev_lg_open is True and not lg_open
        assert self._t01_step is not None

        if self._t23_pending_step is None:
            if left_close_edge and self._step > self._t01_step + 50:
                self._t23_pending_step = self._step
                self._t23_left_close_streak = 1
        elif not lg_open:
            self._t23_left_close_streak += 1
            if self._t23_left_close_streak >= 40:
                if self._can_accept_boundary(self._t23_pending_step):
                    self._t23_step = self._t23_pending_step
                    self._advance(self._t23_pending_step)
        else:
            self._t23_pending_step = None
            self._t23_left_close_streak = 0

    def _check_t34(self, rg: float, lg: float) -> None:
        if rg < 200 and lg > _GRIP_THRESH and self._can_accept_boundary(self._step):
            self._t34_step = self._step
            self._advance(self._step)

    def _check_t45(self, lg_open: bool) -> None:
        left_open_edge = self._prev_lg_open is False and lg_open

        if left_open_edge:
            self._t45_candidate_step = self._step
            self._t45_steps_since_candidate = 1
        elif self._t45_candidate_step is not None and lg_open:
            self._t45_steps_since_candidate += 1
        else:
            self._t45_candidate_step = None
            self._t45_steps_since_candidate = 0

        if self._t45_candidate_step is not None and self._t45_steps_since_candidate >= 30:
            if self._can_accept_boundary(self._t45_candidate_step):
                self._t45_step = self._t45_candidate_step
                self._advance(self._t45_candidate_step)


# =============================================================================
# Observation extraction
# =============================================================================

_CAMERA_MAP = {
    "right/top": "base_0_rgb",
    "left/wrist": "left_wrist_0_rgb",
    "right/wrist": "right_wrist_0_rgb",
}

_GRIPPER_MIN = 70.0
_GRIPPER_MAX = 850.0

def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).

    Matches the convention in dexterous_hang_config.py exactly.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype = np.float32)


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).

    Exact inverse of _quat_to_euler. Uses the same intrinsic XYZ convention.
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return np.array([x, y, z, w], dtype=np.float32)


def actions_euler_to_quat(actions: np.ndarray) -> np.ndarray:
    """Convert a 14-D EEF action array (or batch) from euler to quaternion format.

    Input layout (14-D):
        [left_pos(3), left_euler(3), left_gripper(1),
         right_pos(3), right_euler(3), right_gripper(1)]

    Output layout (16-D):
        [left_pos(3), left_quat(4), left_gripper(1),
         right_pos(3), right_quat(4), right_gripper(1)]

    Supports both single actions (14,) and batches (N, 14).
    """
    single = actions.ndim == 1
    if single:
        actions = actions[None, :]

    left_pos = actions[:, :3]
    left_quat = np.stack([euler_to_quat(e) for e in actions[:, 3:6]], axis=0)
    left_grip = actions[:, 6:7]
    right_pos = actions[:, 7:10]
    right_quat = np.stack([euler_to_quat(e) for e in actions[:, 10:13]], axis=0)
    right_grip = actions[:, 13:14]

    result = np.concatenate(
        [left_pos, left_quat, left_grip, right_pos, right_quat, right_grip],
        axis=-1,
    )

    return result[0] if single else result

def extract_state(obs: dict[str, Any]) -> np.ndarray:
    """Extract 16-D joint-angle state and 14-D EEF state in euler format matching the training norm stats.

    Layout: [left_pos(3), left_euler(3), left_gripper(1),
             right_pos(3), right_euler(3), right_gripper(1)]
    """
    state_dict = obs["state"]

    def _last(arr: np.ndarray) -> np.ndarray:
        return arr[-1] if arr.ndim > 1 else arr

    def _get(key: str, dim: int) -> np.ndarray:
        val = state_dict.get(key)
        return _last(val).astype(np.float32) if val is not None else np.zeros(dim, dtype=np.float32)

    left_tcp = _get("left/tcp_pose", 7)
    right_tcp = _get("right/tcp_pose", 7)

    parts = [
        left_tcp[:3],
        _quat_to_euler(left_tcp[3:7]),
        _get("left/gripper_pos", 1),
        right_tcp[:3],
        _quat_to_euler(right_tcp[3:7]),
        _get("right/gripper_pos", 1),
    ]
    eef_state = np.concatenate(parts, axis=-1)
    if eef_state.ndim > 1:
        eef_state = eef_state.flatten()

    state = np.concatenate([_get("left/joint_qpos", 7), _get("left/gripper_pos", 1), _get("right/joint_qpos", 7), _get("right/gripper_pos", 1)], axis = -1)
    return state, eef_state


def extract_images_rgb(obs: dict[str, Any], camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract RGB images from observation, mapped to canonical policy names."""
    images = {}
    for cam_name in camera_names:
        frames = obs["images"].get(cam_name)
        if frames is None:
            logger.warning(f"Camera {cam_name!r} not found in observation, skipping.")
            continue
        frame = frames[-1] if frames.ndim == 4 else frames
        canonical = _CAMERA_MAP.get(cam_name, cam_name)
        # Robot server returns BGR; convert to RGB for the policy.
        images[canonical] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not images:
        raise RuntimeError(f"No cameras found. Expected {camera_names}, got {list(obs['images'].keys())}.")
    return images


# =============================================================================
# Eval loop
# =============================================================================


def run_episode(
    env: Any,
    policy: LocalPolicy,
    args: Args,
    episode_idx: int,
) -> None:
    logger.info(f"Starting episode {episode_idx}")
    obs, _ = env.reset(seed=episode_idx)

    tracker = SubtaskTracker()

    action_plan: np.ndarray | None = None
    t = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and t < args.max_steps:
        tracker.update(obs)

        if t % args.query_freq == 0:
            prompt = tracker.prompt
            state, initial_eef_pose = extract_state(obs)
            images_rgb = extract_images_rgb(obs, args.camera_names)

            obs_dict = {
                "image": images_rgb,
                "state": state,
                "prompt": prompt,
            }

            t0 = time.perf_counter()
            full_actions = policy.predict(obs_dict, initial_eef_pose)
            elapsed = time.perf_counter() - t0

            action_plan = full_actions[
                :, args.real_action_start : args.real_action_start + args.real_action_dim
            ]
            logger.info(
                f"Episode {episode_idx} step {t}: prompt={prompt!r}, "
                f"action_plan shape={action_plan.shape}, inference={elapsed:.3f}s"
            )
            ipdb.set_trace()
            

        plan_idx = min(t % args.query_freq, action_plan.shape[0] - 1)
        # ipdb.set_trace()
        action = action_plan[plan_idx]
        # action = np.zeros_like(action)
    
        obs, reward, terminated, truncated, _ = env.step(action)
        t += 1

    logger.info(f"Episode {episode_idx} finished after {t} steps (terminated={terminated}, truncated={truncated})")


# =============================================================================
# Debug mode
# =============================================================================


def run_debug_episode(env: Any, args: Args) -> None:
    """Reset the robot, save camera images, and print the state. No policy needed."""
    debug_dir = args.debug_output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
    os.makedirs(debug_dir, exist_ok=True)

    logger.info("Debug mode: resetting environment...")
    obs, _ = env.reset(seed=0)

    state = extract_state(obs)
    print(f"State vector (dim={state.shape[0]}):\n{state}")

    for cam_name in args.camera_names:
        frames = obs["images"].get(cam_name)
        if frames is None:
            logger.warning(f"Camera {cam_name!r} not found in observation, skipping.")
            continue
        frame = frames[-1] if frames.ndim == 4 else frames
        safe_name = cam_name.replace("/", "_")
        path = os.path.join(debug_dir, f"{safe_name}.png")
        cv2.imwrite(path, frame)
        logger.info(f"Saved {cam_name} image ({frame.shape}) to {path}")

    logger.info(f"Debug output saved to {debug_dir}")


# =============================================================================
# Entrypoint
# =============================================================================


def _check_connection(url: str, name: str, timeout: float = 5.0) -> None:
    """Send a GET to the health endpoint and raise if it fails."""
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        r.raise_for_status()
        logger.info(f"{name} health check passed ({url}/health).")
    except requests.RequestException as e:
        raise RuntimeError(f"{name} not reachable at {url}/health: {e}") from e


def main(args: Args) -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from remote_environment_adapter import RemoteEnvironmentAdapter

    if args.debug:
        robot_url = f"http://{args.robot_host}:{args.robot_port}/api"
        _check_connection(robot_url, "Robot server")
        env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port, control_freq=args.control_freq)
        run_debug_episode(env, args)
        env.close()
        return

    critic_model = None
    critic_norm_stats = None
    if args.critic_config is not None:
        assert args.critic_checkpoint is not None, "--args.critic-checkpoint required when --args.critic-config is set"
        logger.info(
            f"Loading critic: config={args.critic_config}, checkpoint={args.critic_checkpoint}, "
            f"fine_tune={args.critic_fine_tune_config}, step={args.critic_step}"
        )
        critic_model, critic_norm_stats = load_critic(
            config_name = args.critic_config,
            checkpoint_path = args.critic_checkpoint,
            fine_tune_config = args.critic_fine_tune_config,
            step = args.critic_step,
        )

    logger.info(f"Loading policy: config={args.config_name}, checkpoint={args.checkpoint_dir}, "
                f"fine_tune={args.fine_tune_config}")
    policy = LocalPolicy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        fine_tune_config=args.fine_tune_config,
        critic_model = critic_model,
        critic_norm_stats = critic_norm_stats,
        num_samples = args.num_samples,
    )

    logger.info(f"Connecting to robot environment at {args.robot_host}:{args.robot_port}")
    env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port, control_freq=args.control_freq)
    logger.info("Connected to robot environment.")

    for episode_idx in range(args.num_episodes):
        run_episode(env, policy, args, episode_idx)

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
