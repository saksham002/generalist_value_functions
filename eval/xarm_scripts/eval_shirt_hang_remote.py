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
import jax
import jax.numpy as jnp
import numpy as np
import requests
from scipy.spatial.transform import Rotation
import tyro

import openpi.models.model as _model

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

    query_freq: int = 30
    """How many env steps between policy replans."""

    max_steps: int = 1800
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


# =============================================================================
# Local policy
# =============================================================================


class LocalPolicy:
    """Loads a policy checkpoint and runs inference locally."""

    def __init__(self, config_name: str, checkpoint_dir: str, step: int | None = None, fine_tune_config: str | None = None) -> None:
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
        logger.info("Policy loaded and JIT-compiled successfully.")

    def predict(self, obs_dict: dict[str, Any], initial_eef_pose: np.ndarray) -> np.ndarray:
        """Run policy inference on a single observation.

        Args:
            obs_dict: Raw observation with keys: image (dict of camera name -> RGB uint8 array),
                state, prompt, embodiment.
            initial_eef_pose: Current 14-D EEF pose in euler format from extract_eef_pose().

        Returns:
            actions: float32 array of shape [60, 14] (upsampled 30->60 Hz).
        """
        raw_state = np.asarray(obs_dict["state"], dtype=np.float32)
        # import pdb; pdb.set_trace()
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
        transition = _model.wrap_observation_as_transition(observation)

        self._rng, sample_rng = jax.random.split(self._rng)

        actions_out = self._sample_actions_jit(sample_rng, transition, **self._sample_kwargs)
        actions_out = jax.block_until_ready(actions_out)

        actions_np = np.asarray(actions_out[0, :, 14 : 28])  # [action_horizon, 32]
        # import ipdb; ipdb.set_trace()
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

        # Policy outputs 30 Hz actions; robot runs at 60 Hz.
        actions = np.repeat(actions, 2, axis=0)

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

        self._t01_pending: bool = False
        self._t01_steps_watching: int = 0
        self._t23_left_close_streak: int = 0
        self._t45_open_streak: int = 0

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

        if self._steps_in_subtask >= _MIN_BOUNDARY_GAP and self._subtask < len(_SUBTASK_PROMPTS) - 1:
            self._check_transition(lg, rg, rz)

        self._step += 1
        self._steps_in_subtask += 1

    def _advance(self) -> None:
        logger.info(f"SubtaskTracker: subtask {self._subtask} -> {self._subtask + 1} "
                    f"({_SUBTASK_PROMPTS[self._subtask + 1]!r}) at step {self._step}")
        self._subtask += 1
        self._steps_in_subtask = 0
        self._t01_pending = False
        self._t01_steps_watching = 0
        self._t23_left_close_streak = 0
        self._t45_open_streak = 0

    def _check_transition(self, lg: float, rg: float, rz: float) -> None:
        if self._subtask == 0:
            self._check_t01(rg, rz)
        elif self._subtask == 1:
            self._check_t12(rz)
        elif self._subtask == 2:
            self._check_t23(lg)
        elif self._subtask == 3:
            self._check_t34(rg, lg)
        elif self._subtask == 4:
            self._check_t45(lg)

    def _check_t01(self, rg: float, rz: float) -> None:
        rg_open = rg > _GRIP_THRESH
        if not self._t01_pending:
            if not rg_open and rz > 0.30:
                self._t01_pending = True
                self._t01_steps_watching = 0
        else:
            self._t01_steps_watching += 1
            if rg_open:
                self._t01_pending = False
                self._t01_steps_watching = 0
            elif self._t01_steps_watching > 500:
                self._t01_pending = False
                self._t01_steps_watching = 0
            elif rz < 0.20:
                self._advance()

    def _check_t12(self, rz: float) -> None:
        if 0.0 < rz < 0.22:
            self._advance()

    def _check_t23(self, lg: float) -> None:
        if lg <= _GRIP_THRESH:
            self._t23_left_close_streak += 1
            if self._t23_left_close_streak >= 40:
                self._advance()
        else:
            self._t23_left_close_streak = 0

    def _check_t34(self, rg: float, lg: float) -> None:
        if rg < 200 and lg > _GRIP_THRESH:
            self._advance()

    def _check_t45(self, lg: float) -> None:
        if lg > _GRIP_THRESH:
            self._t45_open_streak += 1
            if self._t45_open_streak >= 30:
                self._advance()
        else:
            self._t45_open_streak = 0


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
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).

    Matches the convention in dexterous_hang_config.py exactly.
    """
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype = np.float32)

def extract_state(obs: dict[str, Any]) -> np.ndarray:
    """Extract 14-D EEF state in euler format matching the training norm stats.

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
    state = np.concatenate(parts, axis=-1)
    if state.ndim > 1:
        state = state.flatten()
    return state


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
            state = extract_state(obs)
            images_rgb = extract_images_rgb(obs, args.camera_names)

            obs_dict = {
                "image": images_rgb,
                "state": state,
                "prompt": prompt,
            }

            t0 = time.perf_counter()
            full_actions = policy.predict(obs_dict, state)
            elapsed = time.perf_counter() - t0

            action_plan = full_actions[
                :, args.real_action_start : args.real_action_start + args.real_action_dim
            ]
            logger.info(
                f"Episode {episode_idx} step {t}: prompt={prompt!r}, "
                f"action_plan shape={action_plan.shape}, inference={elapsed:.3f}s"
            )

        plan_idx = min(t % args.query_freq, action_plan.shape[0] - 1)
        #import ipdb; ipdb.set_trace()
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
        env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port)
        run_debug_episode(env, args)
        env.close()
        return

    logger.info(f"Loading policy: config={args.config_name}, checkpoint={args.checkpoint_dir}, "
                f"fine_tune={args.fine_tune_config}")
    policy = LocalPolicy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        fine_tune_config=args.fine_tune_config,
    )

    logger.info(f"Connecting to robot environment at {args.robot_host}:{args.robot_port}")
    env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port)
    logger.info("Connected to robot environment.")

    for episode_idx in range(args.num_episodes):
        run_episode(env, policy, args, episode_idx)

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
