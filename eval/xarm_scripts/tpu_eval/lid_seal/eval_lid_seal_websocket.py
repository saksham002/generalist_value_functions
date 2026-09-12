"""Eval script for lid-seal task via WebSocket policy server.

Connects to a remote policy server (serve_policy.py on a TPU pod) over WebSocket
and a remote robot environment server, then runs episodes by querying the server
for actions and sending them to the physical robot.

Usage:
    # Start the policy server on the TPU pod first:
    ./eval/xarm_scripts/tpu_eval/lid_seal/serve_policy_lid_seal.sh

    # Then run the eval client:
    uv run eval/xarm_scripts/tpu_eval/lid_seal/eval_lid_seal_websocket.py \
        --args.policy-host <tpu-worker-0-ip> \
        --args.robot-host xarmpc.pc.cs.cmu.edu
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import queue
import sys
import threading
import time
from typing import Any

import cv2
#import flax.nnx as nnx
import imageio
#import jax
#import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe under SLURM/headless eval
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.spatial.transform import Rotation
import tyro

from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import eval_image_helper as _eval_image_helper

# The client runs as a plain script, so the repo root is not on sys.path.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from eval.helpers import pause_timing as _pause_timing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


# Queue of Enter-key events used by --manual mode to advance subtasks.
_advance_q: "queue.Queue[None]" = queue.Queue()


def _start_key_listener() -> None:
    """Start a daemon thread that pushes one event per Enter key press onto _advance_q."""
    def _listen() -> None:
        while True:
            try:
                input()
            except EOFError:
                return
            _advance_q.put(None)

    threading.Thread(target = _listen, daemon = True).start()


def _wait_for_enter(message: str, *, manual: bool) -> bool:
    """Block until the operator presses Enter. Returns False once stdin is closed.

    In manual mode the key-listener thread owns stdin, so calling input() here would race it
    for the keypress and the prompt could hang forever. Wait on the listener's queue instead.
    Presses queued during the episode (or during video encoding) are drained first, so only a
    fresh press satisfies the prompt.
    """
    while not _advance_q.empty():
        _advance_q.get_nowait()
    logger.info(message)
    if manual:
        _advance_q.get()
        return True
    try:
        input()
    except EOFError:
        return False
    return True


# =============================================================================
# CLI
# =============================================================================


@dataclasses.dataclass
class Args:
    policy_host: str
    """Host running serve_policy.py (TPU worker 0 IP)."""

    policy_port: int = 8000
    """Port the policy server is listening on."""

    robot_host: str = "localhost"
    """Host running robot_environment_server.py."""

    robot_port: int = 8081
    """Port for the robot environment server."""

    num_episodes: int = 1
    """Number of episodes to run."""

    start_episode_idx: int = 0
    """Index of the first episode. The loop runs for `num_episodes` indices starting here,
    so resumed runs (e.g. after a crash at episode 2) keep the seed / log lines / video
    filenames aligned and don't overwrite earlier output."""

    control_freq: int = 60
    """Frequency (Hz) at which the robot server expects actions. Passed to the
    RemoteEnvironmentAdapter so the server initializes its control loop accordingly."""

    query_freq: int = 30
    """How many env steps between policy replans."""

    max_steps: int = 7200
    """Maximum env steps per episode."""

    real_action_start: int = 0
    """Index into the policy action vector where real actions begin."""

    real_action_dim: int = 14
    """Number of real action dimensions."""

    camera_names: tuple[str, ...] = ("right/top", "left/wrist", "right/wrist")
    """Camera names as returned by the robot environment server."""

    debug: bool = False
    """Debug mode: skip policy connection, save camera images to disk and print state instead."""

    debug_output_dir: str = ""
    """Directory to save debug images when --debug is set. Defaults to eval/xarm_scripts/tpu_eval/lid_seal/debug/."""

    manual: bool = False
    """If set, advance subtasks manually by pressing Enter (auto heuristic is disabled)."""

    debug_values: bool = False
    """If set, log the per-candidate Q-values inline with each replan log line."""

    num_samples: int = 8
    """Number of action candidates sampled per replan step (used for VideoLogger sizing)."""

    has_critic: bool = True
    """Whether the policy server is running with a critic (BestOfN). Controls Q-value plot in videos."""

    log_videos: bool = False
    """If set, save a per-episode 4-panel mp4 (wrist + base + Q-value plot)
    to eval/xarm_scripts/tpu_eval/lid_seal/videos/episode_<n>.mp4 at FPS = control_freq / query_freq."""

    video_subdir: str = ""
    """Optional subdirectory under eval/xarm_scripts/tpu_eval/lid_seal/videos/ in which to store
    this run's mp4s. Empty string saves directly into videos/."""

    log_pause_timing: bool = False
    """If set, record the wall-clock time at which every env step was sent and returned, and write
    episode_<n>_pause_timing.json into the same directory as the videos. cut_pauses.py uses it to
    remove the inference pauses from an external camera recording."""

    sync_tone: bool = True
    """With log_pause_timing, play a short tone through the speaker at the first and last step so
    the camera audio can be aligned to the log. Needs `aplay`; silently disabled without it."""

    use_critic_subtasks: bool = True
    """If True, send the per-step subtask prompt to the critic via obs["prompt"]. If False,
    send task_description instead, so the critic is conditioned on the same prompt the policy
    receives. Only affects the critic — the policy prompt is pinned server-side by the
    serve_policy --task-description override. Requires a critic whose prompt_mode reads the
    client prompt (not task_description_predict_current_subtask, which ignores obs["prompt"])."""

    task_description: str = ""
    """Prompt sent to the critic when use_critic_subtasks is False; should match the server's
    --task-description (the prompt the policy is conditioned on)."""

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

class SubtaskTracker:
    """Manual subtask tracker: advances only when force_advance() is called (--manual mode).

    Holds the current subtask index and exposes the corresponding prompt string, which is
    sent to the critic when --use-critic-subtasks is set. There is no automatic transition
    detection from the observation stream — the caller drives progression explicitly (e.g.
    on an Enter keypress).
    """

    def __init__(self) -> None:
        self._subtask = 0

    @property
    def subtask(self) -> int:
        return self._subtask

    @property
    def prompt(self) -> str:
        return _SUBTASK_PROMPTS[self._subtask]

    def force_advance(self) -> None:
        """Manually move to the next subtask (used by --manual mode)."""
        if self._subtask < len(_SUBTASK_PROMPTS) - 1:
            logger.info(f"SubtaskTracker: subtask {self._subtask} -> {self._subtask + 1} "
                        f"({_SUBTASK_PROMPTS[self._subtask + 1]!r})")
            self._subtask += 1


# =============================================================================
# Video logging
# =============================================================================


# Each panel is a square. 256 satisfies libx264's "divisible by 16" requirement,
# and the final 2x2 mosaic is 512x512 — small enough to keep encoding fast.
_VIDEO_PANEL_SIZE = 256


class VideoLogger:
    """Buffers per-replan observations + Q-values and writes a 2x2 mp4 at episode end.

    Layout (each panel _VIDEO_PANEL_SIZE x _VIDEO_PANEL_SIZE):

        +-------------------+-------------------+
        |  left wrist RGB   |  Q-value plot     |
        +-------------------+-------------------+
        |  right wrist RGB  |  base RGB         |
        +-------------------+-------------------+

    The Q-value plot is the only animated panel: lines = Q-values per candidate over
    replan ticks (static across frames), blue verticals = manual subtask advances
    (static), red vertical = current frame's tick (moves frame to frame).
    """

    def __init__(
        self,
        output_dir: str,
        fps: float,
        num_samples: int,
        *,
        has_critic: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.num_samples = num_samples
        # When False, the Q-value plot panel is replaced by a blank panel and the
        # rest of the mosaic shows only the three camera feeds.
        self.has_critic = has_critic
        os.makedirs(output_dir, exist_ok = True)
        self._reset()

    def _reset(self) -> None:
        self._images: list[dict[str, np.ndarray]] = []
        self._q_values: list[np.ndarray] = []
        self._steps: list[int] = []
        self._advance_steps: list[int] = []
        # Per-replan critic-predicted subtask (carried forward from the last AR-decode
        # step); used as the Q-value plot title.
        self._predicted_subtasks: list[str | None] = []
        self._episode_idx: int | None = None

    def start_episode(self, episode_idx: int) -> None:
        self._reset()
        self._episode_idx = episode_idx

    def record_predict(
        self,
        images: dict[str, np.ndarray],
        q_values: np.ndarray | None,
        t: int,
        predicted_subtask: str | None = None,
    ) -> None:
        """Snapshot the latest cameras + Q-values at the env step where predict() ran."""
        self._images.append({k: np.asarray(v).copy() for k, v in images.items()})
        if q_values is None:
            self._q_values.append(np.zeros(self.num_samples, dtype = np.float32))
        else:
            self._q_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
        self._steps.append(t)
        # The critic only AR-decodes a subtask every N steps; carry the last decoded
        # value forward so every frame's plot title reflects the current prediction.
        if predicted_subtask is None and self._predicted_subtasks:
            predicted_subtask = self._predicted_subtasks[-1]
        self._predicted_subtasks.append(predicted_subtask)

    def record_advance(self, t: int) -> None:
        """Mark an env step at which the user pressed Enter (manual subtask switch)."""
        self._advance_steps.append(t)

    def finish_episode(self) -> None:
        if self._episode_idx is None or not self._images:
            return
        q_matrix = np.stack(self._q_values, axis = 0)  # (T_replans, N)
        steps = np.asarray(self._steps, dtype = np.int64)
        # Persisting the exact per-replan Q-values plotted in the video is disabled.
        # qval_path = os.path.join(self.output_dir, f"Final Video q values_episode_{self._episode_idx}.npz")
        # np.savez(qval_path, steps = steps, q_values = q_matrix)
        # logger.info(f"Saved episode Q-values: {qval_path}")
        frames = [self._render_frame(i, q_matrix, steps) for i in range(len(self._images))]
        out_path = os.path.join(self.output_dir, f"episode_{self._episode_idx}.mp4")
        # imageio bundles its own ffmpeg with libx264; system ffmpeg on this cluster
        # lacks libx264 (see CLAUDE.md > "Saving Videos on HPC").
        imageio.mimsave(
            out_path,
            frames,
            format = "mp4",
            fps = self.fps,
            codec = "libx264",
            quality = 8,
        )
        logger.info(f"Saved episode video: {out_path}")

    def _render_frame(self, frame_idx: int, q_matrix: np.ndarray, steps: np.ndarray) -> np.ndarray:
        images = self._images[frame_idx]
        current_step = int(steps[frame_idx])
        size = _VIDEO_PANEL_SIZE

        def _panel(img: np.ndarray | None) -> np.ndarray:
            if img is None:
                return np.zeros((size, size, 3), dtype = np.uint8)
            return cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)

        left_wrist = _panel(images.get("left_wrist_0_rgb"))
        right_wrist = _panel(images.get("right_wrist_0_rgb"))
        base_rgb = _panel(images.get("base_0_rgb"))
        if self.has_critic:
            predicted_subtask = (
                self._predicted_subtasks[frame_idx] if frame_idx < len(self._predicted_subtasks) else None
            )
            value_panel = self._render_value_plot(size, q_matrix, steps, current_step, predicted_subtask)
        else:
            # No critic → no Q-values to plot; show a blank panel so the layout
            # and mp4 dimensions stay constant.
            value_panel = np.zeros((size, size, 3), dtype = np.uint8)

        top = np.concatenate([left_wrist, value_panel], axis = 1)
        bottom = np.concatenate([right_wrist, base_rgb], axis = 1)
        return np.concatenate([top, bottom], axis = 0)

    def _render_value_plot(
        self,
        size: int,
        q_matrix: np.ndarray,
        steps: np.ndarray,
        current_step: int,
        predicted_subtask: str | None = None,
    ) -> np.ndarray:
        dpi = 100
        figsize = (size / dpi, size / dpi)
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        for sample_idx in range(q_matrix.shape[1]):
            ax.plot(steps, q_matrix[:, sample_idx], linewidth = 0.8)
        for adv_step in self._advance_steps:
            ax.axvline(x = adv_step, color = "blue", linewidth = 1.0, alpha = 0.7)
        ax.axvline(x = current_step, color = "red", linewidth = 1.5)
        x_lo = int(steps[0]) if len(steps) > 0 else 0
        x_hi = int(steps[-1]) if len(steps) > 0 else 1
        if x_hi == x_lo:
            x_hi = x_lo + 1
        ax.set_xlim(x_lo, x_hi)
        if predicted_subtask:
            title = str(predicted_subtask)
            if len(title) > 42:
                title = title[:39] + "..."
            ax.set_title(title, fontsize = 6)
        ax.set_xlabel("env step", fontsize = 6)
        ax.set_ylabel("Q value", fontsize = 6)
        ax.tick_params(labelsize = 5)
        fig.tight_layout(pad = 0.5)
        fig.canvas.draw()
        # buffer_rgba is the matplotlib 3.x+ way; tostring_rgb was removed.
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        if buf.shape[0] != size or buf.shape[1] != size:
            buf = cv2.resize(buf, (size, size), interpolation = cv2.INTER_AREA)
        return buf


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

    #state = np.concatenate([_get("left/joint_qpos", 7), _get("left/gripper_pos", 1), _get("right/joint_qpos", 7), _get("right/gripper_pos", 1)], axis = -1)
    return eef_state, eef_state


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
    client: _websocket_client_policy.WebsocketClientPolicy,
    image_helper: _eval_image_helper.EvalImageHelper,
    args: Args,
    episode_idx: int,
    obs: dict[str, Any],
) -> None:
    logger.info(f"Starting episode {episode_idx}")

    # The robot was reset before the operator staged the scene, so the observation handed in
    # predates the objects being placed. Refresh it so the first policy query sees the real scene.
    obs = env.get_observation()

    tracker = SubtaskTracker()
    if args.manual:
        logger.info("Manual subtask switching enabled — press Enter to advance subtask.")

    # Per-episode video logger. With a critic, the Q-value plot panel is animated;
    # without one, that panel is left blank and the video shows only the 3 cameras.
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    if args.video_subdir:
        output_dir = os.path.join(output_dir, args.video_subdir)

    video_logger: VideoLogger | None = None
    if args.log_videos:
        video_fps = args.control_freq / args.query_freq
        video_logger = VideoLogger(
            output_dir = output_dir,
            fps = video_fps,
            num_samples = args.num_samples,
            has_critic = args.has_critic,
        )
        video_logger.start_episode(episode_idx)

    pause_logger: _pause_timing.PauseTimingLogger | None = None
    if args.log_pause_timing:
        pause_logger = _pause_timing.PauseTimingLogger(
            output_dir = output_dir,
            control_freq = args.control_freq,
            query_freq = args.query_freq,
            play_tones = args.sync_tone,
        )
        pause_logger.start_episode(episode_idx)

    action_plan: np.ndarray | None = None
    t = 0
    terminated = False
    truncated = False

    # Per-episode q-value npz logging disabled.
    # q_log_steps: list[int] = []
    # q_log_values: list[np.ndarray] = []
    # qval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qvalues")

    try:
        while not (terminated or truncated) and t < args.max_steps:
            if args.manual:
                while not _advance_q.empty():
                    _advance_q.get_nowait()
                    tracker.force_advance()
                    logger.info(f"Manual advance → subtask {tracker.subtask} ({tracker.prompt!r})")
                    if video_logger is not None:
                        video_logger.record_advance(t)

            if t % args.query_freq == 0:
                # obs["prompt"] is consumed by the critic; the policy prompt is pinned
                # server-side by serve_policy's --task-description override.
                critic_prompt = tracker.prompt if args.use_critic_subtasks else args.task_description
                state, initial_eef_pose = extract_state(obs)
                images_rgb = extract_images_rgb(obs, args.camera_names)
                element_images = image_helper.process_images({
                    "base_0_rgb": images_rgb.get("base_0_rgb"),
                    "left_wrist_0_rgb": images_rgb.get("left_wrist_0_rgb"),
                    "right_wrist_0_rgb": images_rgb.get("right_wrist_0_rgb"),
                })
                obs_dict = {
                    **element_images,
                    "state": state,
                    "prompt": critic_prompt,
                }

                t0 = time.perf_counter()

                infer_result = client.infer(obs_dict)
                full_actions = np.asarray(infer_result["actions"], dtype=np.float32)
                q_values = infer_result.get("q_values")
                server_ms = infer_result.get("server_timing", {}).get("infer_ms")

                # Per-episode .npz q-value persistence disabled.
                # if q_values is not None:
                #     q_log_steps.append(t)
                #     q_log_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
                #     os.makedirs(qval_dir, exist_ok = True)
                #     np.savez(
                #         os.path.join(qval_dir, f"episode_{episode_idx}.npz"),
                #         steps = np.asarray(q_log_steps, dtype = np.int64),
                #         q_values = np.stack(q_log_values, axis = 0),
                #     )

                elapsed = time.perf_counter() - t0
                if pause_logger is not None:
                    pause_logger.mark_replan(t, elapsed, server_ms)

                action_plan = full_actions[
                    :, args.real_action_start : args.real_action_start + args.real_action_dim
                ]
                log_line = (
                    f"Episode {episode_idx} step {t}: subtask={tracker.subtask} "
                    f"critic_prompt={critic_prompt!r}, inference={elapsed:.3f}s"
                )
                if q_values is not None:
                    # B = 1 in the eval flow; flatten and format.
                    values_str = ", ".join(f"{v:.4f}" for v in np.asarray(q_values).reshape(-1).tolist())
                    log_line += f", q_values=[{values_str}]"
                logger.info(log_line)

                # When the server AR-decodes the current subtask (returns
                # 'predicted_subtask' on decode steps), log it next to the tracker's
                # current subtask (the annotation) so alignment can be checked, and
                # dump the right/top frame the critic saw.
                predicted_subtask = infer_result.get("predicted_subtask")
                if predicted_subtask is not None:
                    logger.info(
                        f"[subtask decode] step {t}: predicted={predicted_subtask!r}  "
                        #f"annotation(tracker)=subtask{tracker.subtask}:{tracker.prompt!r}"
                    )
                    decode_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "decoded_subtask_images"
                    )
                    os.makedirs(decode_dir, exist_ok = True)
                    right_top_rgb = images_rgb.get("base_0_rgb")  # "right/top" → base_0_rgb
                    if right_top_rgb is not None:
                        safe_subtask = "".join(
                            c if c.isalnum() else "_" for c in str(predicted_subtask)
                        )[:60]
                        out_path = os.path.join(
                            decode_dir, f"ep{episode_idx}_step{t}_ann{tracker.subtask}_pred_{safe_subtask}.png"
                        )
                        imageio.imwrite(out_path, right_top_rgb)

                if video_logger is not None:
                    video_logger.record_predict(images_rgb, q_values, t, predicted_subtask)


            plan_idx = min(t % args.query_freq, action_plan.shape[0] - 1)
            # ipdb.set_trace()
            action = action_plan[plan_idx]
            # action = np.zeros_like(action)

            if pause_logger is not None:
                pause_logger.mark_step_sent(t)
            obs, reward, terminated, truncated, _ = env.step(action)
            if pause_logger is not None:
                pause_logger.mark_step_returned(t)
            t += 1

        logger.info(f"Episode {episode_idx} finished after {t} steps (terminated={terminated}, truncated={truncated})")
    finally:
        # The end tone is played here, before the (slow) video render, so it lands right after
        # the last step on the camera recording.
        if pause_logger is not None:
            pause_logger.finish_episode()
        if video_logger is not None:
            video_logger.finish_episode()


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

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from remote_environment_adapter import RemoteEnvironmentAdapter

    if args.manual:
        _start_key_listener()

    if args.debug:
        robot_url = f"http://{args.robot_host}:{args.robot_port}/api"
        _check_connection(robot_url, "Robot server")
        env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port, control_freq=args.control_freq)
        run_debug_episode(env, args)
        env.close()
        return

    logger.info(f"Connecting to robot environment at {args.robot_host}:{args.robot_port}")
    env = RemoteEnvironmentAdapter(host=args.robot_host, port=args.robot_port, control_freq=args.control_freq)
    logger.info("Connected to robot environment.")

    client = None
    for episode_idx in range(args.start_episode_idx, args.start_episode_idx + args.num_episodes):
        # Two separate Enters bracket every episode: the first homes the robot, the second
        # starts it moving. Splitting them leaves the scene-staging window entirely after the
        # reset, so the previous episode's video finishes encoding while the operator is still
        # being asked to reset rather than while the robot is already homed and waiting.
        try:
            if not _wait_for_enter(
                f"Press Enter to reset the robot for episode {episode_idx}...", manual=args.manual
            ):
                break
            logger.info(f"Resetting environment for episode {episode_idx}...")
            obs, _ = env.reset(seed=episode_idx)

            # Reconnect per episode so a response left buffered by a Ctrl-C'd inference in the
            # previous episode can't carry over. A stale buffered response would be returned by
            # the next episode's first infer() call, desyncing every request/response by one.
            # Done after the reset so the fresh socket sits idle for as little as possible.
            if client is not None:
                logger.info(
                    f"Reconnecting policy client to {args.policy_host}:{args.policy_port} for episode {episode_idx}"
                )
                try:
                    client._ws.close()
                except Exception as e:
                    logger.warning(f"Error closing previous client websocket (ignored, reconnecting anyway): {e}")
            else:
                logger.info(f"Connecting policy client to {args.policy_host}:{args.policy_port} for episode {episode_idx}")
            client = _websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
            image_helper = _eval_image_helper.EvalImageHelper.from_client(client)
            logger.info(
                f"EvalImageHelper: policy={image_helper.policy_image_size}, "
                f"critic={image_helper.critic_image_size}, "
                f"expect_critic_images={image_helper.expect_critic_images}"
            )

            if not _wait_for_enter(
                f"Reset complete. Stage the scene, then press Enter to start episode {episode_idx}...",
                manual=args.manual,
            ):
                break
        except KeyboardInterrupt:
            logger.info("Ctrl+C at the episode prompt; stopping.")
            break

        try:
            run_episode(env, client, image_helper, args, episode_idx, obs)
        except KeyboardInterrupt:
            # run_episode's finally has already written this episode's video by the time the
            # interrupt surfaces here.
            logger.info(f"Episode {episode_idx} interrupted by Ctrl+C")

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
