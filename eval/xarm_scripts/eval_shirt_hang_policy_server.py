"""Basic eval script for shirt-hang task using the serve_policy HTTP server.

Connects to a remote robot environment server and a remote policy server,
then runs episodes by sending observations to the policy server and executing
the returned action chunks on the robot.

Usage:
    uv run eval/xarm_scripts/eval_shirt_hang_policy_server.py \
        --policy-host babel-gpu-node \
        --robot-host robot-machine
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import time
from typing import Any

import cv2
import numpy as np
import requests
import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================


@dataclasses.dataclass
class Args:
    policy_host: str = "localhost"
    """Host running serve_policy.py."""

    policy_port: int = 8080
    """Port for the policy server."""

    robot_host: str = "localhost"
    """Host running robot_environment_server.py."""

    robot_port: int = 8081
    """Port for the robot environment server."""

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

    embodiment: str = "dual_xarms"
    """Embodiment string passed to the policy. Must match what the model was trained on."""

    policy_timeout: float = 300.0
    """Timeout in seconds for policy server requests. Should be long enough to cover JIT compilation on first call."""


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

        # T0→1: pending right-close event (gripper closed at high Z, waiting for descent)
        self._t01_pending: bool = False
        self._t01_steps_watching: int = 0

        # T2→3: consecutive frames of stable left close
        self._t23_left_close_streak: int = 0

        # T4→5: track consecutive frames of left open (final release confirmation)
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
            # Take last frame if obs is stacked, then flatten to scalar.
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
                "T0→1 and T1→2 transitions require absolute TCP Z and will not fire. "
                "Available keys: %s", list(state.keys())
            )
        rz = float(_vec("right/tcp_pose", dim=7)[2])

        if self._steps_in_subtask >= _MIN_BOUNDARY_GAP and self._subtask < len(_SUBTASK_PROMPTS) - 1:
            self._check_transition(lg, rg, rz)

        self._step += 1
        self._steps_in_subtask += 1

    def _advance(self) -> None:
        logger.info(f"SubtaskTracker: subtask {self._subtask} → {self._subtask + 1} "
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
        # Right gripper closes while Z > 0.30, then descends to Z < 0.20 without reopening.
        rg_open = rg > _GRIP_THRESH
        if not self._t01_pending:
            if not rg_open and rz > 0.30:
                self._t01_pending = True
                self._t01_steps_watching = 0
        else:
            self._t01_steps_watching += 1
            if rg_open:
                # Gripper reopened before descent confirmed — failed grasp attempt.
                self._t01_pending = False
                self._t01_steps_watching = 0
            elif self._t01_steps_watching > 500:
                # Arm stayed closed for 500+ steps but never descended — invalid grasp.
                self._t01_pending = False
                self._t01_steps_watching = 0
            elif rz < 0.20:
                self._advance()

    def _check_t12(self, rz: float) -> None:
        # Right TCP Z drops below handoff height. Guard against rz=0.0 (missing key).
        if 0.0 < rz < 0.22:
            self._advance()

    def _check_t23(self, lg: float) -> None:
        # Left gripper closes and stays closed for 40+ consecutive frames.
        if lg <= _GRIP_THRESH:
            self._t23_left_close_streak += 1
            if self._t23_left_close_streak >= 40:
                self._advance()
        else:
            self._t23_left_close_streak = 0

    def _check_t34(self, rg: float, lg: float) -> None:
        # Right firmly closed AND left open simultaneously.
        if rg < 200 and lg > _GRIP_THRESH:
            self._advance()

    def _check_t45(self, lg: float) -> None:
        # Left gripper opens and stays open for 30+ consecutive frames (final release).
        if lg > _GRIP_THRESH:
            self._t45_open_streak += 1
            if self._t45_open_streak >= 30:
                self._advance()
        else:
            self._t45_open_streak = 0


# =============================================================================
# Policy server client
# =============================================================================

# Maps robot environment camera names to canonical policy names.
_CAMERA_MAP = {
    "right/top": "base_0_rgb",
    "left/wrist": "left_wrist_0_rgb",
    "right/wrist": "right_wrist_0_rgb",
}


class PolicyServerClient:
    """Sends observations to serve_policy.py and returns action chunks."""

    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        self._base_url = f"http://{host}:{port}/api"
        self._timeout = timeout
        self._wait_for_server()

    def _wait_for_server(self, max_wait: float = 120.0) -> None:
        start = time.time()
        while time.time() - start < max_wait:
            try:
                r = requests.get(f"{self._base_url}/health", timeout=5.0)
                if r.status_code == 200:
                    logger.info("Policy server is ready.")
                    return
            except requests.RequestException:
                pass
            logger.info("Waiting for policy server...")
            time.sleep(2.0)
        raise RuntimeError(f"Policy server at {self._base_url} not responding after {max_wait}s")

    def predict(
        self,
        state: np.ndarray,
        images_bgr: dict[str, np.ndarray],
        prompt: str,
        embodiment: str,
    ) -> np.ndarray:
        """Send one observation to the policy server and return the action chunk.

        Args:
            state: Robot state vector, shape (state_dim,), float32.
            images_bgr: Dict of camera name -> BGR uint8 image (H, W, 3).
            prompt: Subtask text prompt.
            embodiment: Embodiment string.

        Returns:
            actions: float32 array of shape (action_horizon, action_dim).
        """
        encoded_images = {}
        for cam_name, img_bgr in images_bgr.items():
            canonical = _CAMERA_MAP.get(cam_name, cam_name)
            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            encoded_images[canonical] = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "state": state.tolist(),
            "images": encoded_images,
            "prompt": prompt,
            "embodiment": embodiment,
        }
        response = requests.post(f"{self._base_url}/predict", json=payload, timeout=self._timeout)
        response.raise_for_status()
        return np.array(response.json()["actions"], dtype=np.float32)


# =============================================================================
# Observation extraction
# =============================================================================


def extract_state(obs: dict[str, Any]) -> np.ndarray:
    """Extract the 40-d relative-frame state vector from a RMPDualXArmsEnv observation."""
    state_dict = obs["state"]

    def _last(arr: np.ndarray) -> np.ndarray:
        return arr[-1] if arr.ndim > 1 else arr

    def _get(key: str, dim: int) -> np.ndarray:
        val = state_dict.get(key)
        return _last(val).astype(np.float32) if val is not None else np.zeros(dim, dtype=np.float32)

    parts = [
        _get("left/relative2_tcp_pose", 7),
        _get("left/relative2_tcp_vel", 6),
        _get("left/wrist_tcp_vel", 6),
        _get("left/gripper_pos", 1),
        _get("right/relative2_tcp_pose", 7),
        _get("right/relative2_tcp_vel", 6),
        _get("right/wrist_tcp_vel", 6),
        _get("right/gripper_pos", 1),
    ]
    state = np.concatenate(parts, axis=-1)
    if state.ndim > 1:
        state = state.flatten()
    return state


def extract_images_bgr(obs: dict[str, Any], camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract BGR images from observation, taking the last frame if stacked."""
    images = {}
    for cam_name in camera_names:
        frames = obs["images"].get(cam_name)
        if frames is None:
            logger.warning(f"Camera {cam_name!r} not found in observation, skipping.")
            continue
        images[cam_name] = frames[-1] if frames.ndim == 4 else frames
    if not images:
        raise RuntimeError(f"No cameras found. Expected {camera_names}, got {list(obs['images'].keys())}.")
    return images


# =============================================================================
# Eval loop
# =============================================================================


def run_episode(
    env: Any,
    policy_client: PolicyServerClient,
    embodiment: str,
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
            images_bgr = extract_images_bgr(obs, args.camera_names)
            full_actions = policy_client.predict(state, images_bgr, prompt, embodiment)
            action_plan = full_actions[
                :, args.real_action_start : args.real_action_start + args.real_action_dim
            ]
            logger.info(
                f"Episode {episode_idx} step {t}: prompt={prompt!r}, "
                f"action_plan shape={action_plan.shape}"
            )

        plan_idx = min(t % args.query_freq, action_plan.shape[0] - 1)
        action = action_plan[plan_idx]

        obs, reward, terminated, truncated, _ = env.step(action)
        t += 1

    logger.info(f"Episode {episode_idx} finished after {t} steps (terminated={terminated}, truncated={truncated})")


# =============================================================================
# Entrypoint
# =============================================================================


def main(args: Args) -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from remote_environment_adapter import RemoteEnvironmentAdapter

    logger.info(f"embodiment={args.embodiment!r}")

    policy_client = PolicyServerClient(host=args.policy_host, port=args.policy_port, timeout=args.policy_timeout)

    logger.info(f"Connecting to robot environment at {args.robot_host}:{args.robot_port}")
    env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port)
    logger.info("Connected to robot environment.")

    for episode_idx in range(args.num_episodes):
        run_episode(env, policy_client, args.embodiment, args, episode_idx)

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
