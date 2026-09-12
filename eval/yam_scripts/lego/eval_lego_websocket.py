"""Eval client for the lego task on the YAM bimanual robot via a WebSocket policy server.

Thin API caller: reads observations from the local ``YAMBimanualEnv`` (gello ZMQ stack),
forwards them to a remote ``serve_policy.py`` (TPU pod) and executes the returned joint
position chunks. All normalization, delta-to-absolute conversion and action slicing run
server-side; the client only reshapes the raw observation into the websocket element
(14D state, three RGB cameras, subtask prompt).

Runs in this repo's uv env with ``yam_teleop`` (the gello ZMQ robot env) installed into it.

Usage:
    # Start the policy server on the TPU pod first:
    ./eval/yam_scripts/lego/serve_policy_lego.sh

    # Then run the eval client (see run_eval_lego_websocket.sh):
    python eval/yam_scripts/lego/eval_lego_websocket.py \
        --args.policy-host <tpu-worker-0-ip> \
        --args.env-config ~/gello-yam-teleop/yam_teleop/configs/env.yaml \
        --args.subtasks "Take apart the lego pieces"
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import threading
import time
from typing import Any

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; the client runs headless alongside the robot nodes
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi_client import eval_image_helper as _eval_image_helper
from openpi_client import websocket_client_policy as _websocket_client_policy
from yam_teleop.env import YAMBimanualEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)


# Queue of Enter-key events used to advance subtasks during an episode.
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


def _wait_for_enter(message: str) -> None:
    """Block until the next Enter press. Goes through the listener queue rather than a second
    input() so only one thread ever reads stdin; any presses queued earlier are discarded."""
    while not _advance_q.empty():
        _advance_q.get_nowait()
    print(message, flush = True)
    _advance_q.get()


# =============================================================================
# CLI
# =============================================================================


@dataclasses.dataclass
class Args:
    policy_host: str
    """Host running serve_policy.py (TPU worker 0 external IP)."""

    env_config: str
    """Path to the gello env.yaml (YAMBimanualEnv config)."""

    policy_port: int = 8000
    """Port the policy server is listening on."""

    tasks_file: str = ""
    """Path to the lego tasks JSON. Empty string defaults to lego_tasks.json next to this script.
    When the file exists, each episode loads the task at its own episode_idx and takes its
    subtask prompts and task description from there, overriding `subtasks` and
    `task_description`."""

    subtasks: tuple[str, ...] = ("Take apart the blocks",)
    """Ordered subtask prompts for the episode when no tasks file is used. The policy
    (prompt_mode="subtask") is conditioned on the current one; press Enter during an
    episode to advance to the next."""

    task_description: str = ""
    """Full task description sent per call as obs["task_description"]. Only the critic reads it
    (task_description_predict_current_subtask prompt mode); empty sends nothing and the
    server's --task-description applies."""

    num_episodes: int = 1
    """Number of episodes to run."""

    start_episode_idx: int = 0
    """Index of the first episode, so resumed runs keep their log lines aligned."""

    query_freq: int = 30
    """Env steps between policy replans (30 = 0.5 s at the env's 60 Hz)."""

    max_steps: int = 10800
    """Maximum env steps per episode (10800 = 3 min at 60 Hz)."""

    real_action_dim: int = 14
    """Leading action dims executed on the robot: [left_arm(6), left_gripper, right_arm(6), right_gripper]."""

    no_reset: bool = False
    """Skip sending the robot home at episode start (continuous prompting)."""

    control_freq: int = 60
    """Env control rate (Hz) from the gello env.yaml; with query_freq it sets the video FPS."""

    num_samples: int = 8
    """Number of action candidates the server samples per replan (sizes the Q-value plot)."""

    has_critic: bool = False
    """Whether the policy server runs with a critic (BestOfN). Controls the Q-value panel in videos."""

    log_videos: bool = True
    """If set, save a per-episode 4-panel mp4 (wrist + top + Q-value plot) to
    eval/yam_scripts/lego/videos/<video_subdir>/episode_<n>.mp4 at FPS = control_freq / query_freq."""

    video_subdir: str = "BC"
    """Subdirectory under eval/yam_scripts/lego/videos/ for this run's mp4s (e.g. BC, BoN)."""


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
        |  right wrist RGB  |  top RGB          |
        +-------------------+-------------------+

    The Q-value plot is the only animated panel: lines = Q-values per candidate over
    replan ticks (static across frames), blue verticals = subtask advances (static),
    red vertical = current frame's tick (moves frame to frame).
    """

    def __init__(self, output_dir: str, fps: float, num_samples: int, *, has_critic: bool) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.num_samples = num_samples
        # When False, the Q-value panel is left blank so the layout and mp4
        # dimensions stay constant between BC and BoN runs.
        self.has_critic = has_critic
        os.makedirs(output_dir, exist_ok = True)
        self._reset()

    def _reset(self) -> None:
        self._images: list[dict[str, np.ndarray]] = []
        self._q_values: list[np.ndarray] = []
        self._steps: list[int] = []
        self._advance_steps: list[int] = []
        self._episode_idx: int | None = None

    def start_episode(self, episode_idx: int) -> None:
        self._reset()
        self._episode_idx = episode_idx

    def record_predict(self, images: dict[str, np.ndarray], q_values: np.ndarray | None, t: int) -> None:
        """Snapshot the latest cameras + Q-values at the env step where infer() ran."""
        self._images.append({k: np.asarray(v).copy() for k, v in images.items()})
        if q_values is None:
            self._q_values.append(np.zeros(self.num_samples, dtype = np.float32))
        else:
            self._q_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
        self._steps.append(t)

    def record_advance(self, t: int) -> None:
        """Mark an env step at which Enter advanced the subtask."""
        self._advance_steps.append(t)

    def finish_episode(self) -> None:
        if self._episode_idx is None or not self._images:
            return
        q_matrix = np.stack(self._q_values, axis = 0)  # (T_replans, N)
        steps = np.asarray(self._steps, dtype = np.int64)
        frames = [self._render_frame(i, q_matrix, steps) for i in range(len(self._images))]
        out_path = os.path.join(self.output_dir, f"episode_{self._episode_idx}.mp4")
        # imageio bundles its own ffmpeg with libx264, so no system ffmpeg is needed.
        imageio.mimsave(out_path, frames, format = "mp4", fps = self.fps, codec = "libx264", quality = 8)
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
        top_rgb = _panel(images.get("base_0_rgb"))
        if self.has_critic:
            value_panel = self._render_value_plot(size, q_matrix, steps, current_step)
        else:
            value_panel = np.zeros((size, size, 3), dtype = np.uint8)

        top = np.concatenate([left_wrist, value_panel], axis = 1)
        bottom = np.concatenate([right_wrist, top_rgb], axis = 1)
        return np.concatenate([top, bottom], axis = 0)

    def _render_value_plot(self, size: int, q_matrix: np.ndarray, steps: np.ndarray, current_step: int) -> np.ndarray:
        dpi = 100
        fig, ax = plt.subplots(figsize = (size / dpi, size / dpi), dpi = dpi)
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
# Observation formatting
# =============================================================================

# camera_node name -> model-side image key. Matches the LeRobot RLDS layout
# (cam_0=top -> base_0_rgb, cam_1=left_wrist, cam_2=right_wrist).
_CAMERA_MAP = {
    "top": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}


def extract_state(raw_obs: dict[str, Any]) -> np.ndarray:
    """14D joint state in the training layout: left joints, left gripper, right joints, right gripper."""
    return np.concatenate(
        [
            raw_obs["robot"]["left/joint_pos"],
            raw_obs["robot"]["left/gripper_pos"],
            raw_obs["robot"]["right/joint_pos"],
            raw_obs["robot"]["right/gripper_pos"],
        ]
    ).astype(np.float32)


def extract_images_bgr(raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Full-resolution HWC BGR frames (as published by camera_node) keyed by model-side camera name."""
    return {model_key: raw_obs["images"][cam_name] for cam_name, model_key in _CAMERA_MAP.items()}


def prepare_images(
    raw_obs: dict[str, Any], image_helper: _eval_image_helper.EvalImageHelper
) -> dict[str, dict[str, np.ndarray]]:
    """Resize first, then convert BGR -> RGB on the small frames.

    The channel flip is a full copy, so doing it on three 1080p frames costs ~45 ms of GIL time
    per replan and starves the env's camera-drain thread; on 224x224 it is negligible. The stretch
    resize is channel-agnostic, so ordering it first is exact. EvalImageHelper may alias the same
    dict under "image" and "critic_image", hence the fresh dicts instead of in-place flips.
    """
    resized_bgr = image_helper.process_images(extract_images_bgr(raw_obs))
    converted: dict[int, dict[str, np.ndarray]] = {}
    out = {}
    for stream_key, cams in resized_bgr.items():
        if id(cams) not in converted:
            converted[id(cams)] = {k: np.ascontiguousarray(v[:, :, ::-1]) for k, v in cams.items()}
        out[stream_key] = converted[id(cams)]
    return out


# =============================================================================
# Eval loop
# =============================================================================


def run_episode(
    env: YAMBimanualEnv,
    obs: dict[str, Any],
    client: _websocket_client_policy.WebsocketClientPolicy,
    image_helper: _eval_image_helper.EvalImageHelper,
    args: Args,
    episode_idx: int,
) -> None:
    """Run one episode from `obs` (the env was already reset by the caller). A Ctrl-C ends the
    episode early; the video is still written because the logger is flushed in `finally`."""
    logger.info(f"Starting episode {episode_idx}")

    subtask_idx = 0
    logger.info(f"Subtask 0/{len(args.subtasks) - 1}: {args.subtasks[0]!r} (press Enter to advance)")

    video_logger: VideoLogger | None = None
    if args.log_videos:
        video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos", args.video_subdir)
        video_logger = VideoLogger(
            output_dir = video_dir,
            fps = args.control_freq / args.query_freq,
            num_samples = args.num_samples,
            has_critic = args.has_critic,
        )
        video_logger.start_episode(episode_idx)

    action_plan: np.ndarray | None = None
    t = 0
    try:
        while t < args.max_steps:
            while not _advance_q.empty():
                _advance_q.get_nowait()
                if subtask_idx < len(args.subtasks) - 1:
                    subtask_idx += 1
                    logger.info(f"Subtask {subtask_idx}/{len(args.subtasks) - 1}: {args.subtasks[subtask_idx]!r}")
                    if video_logger is not None:
                        video_logger.record_advance(t)
                else:
                    logger.info("Already on the last subtask; Enter ignored.")

            if t % args.query_freq == 0:
                element_images = prepare_images(obs, image_helper)
                element = {
                    **element_images,
                    "state": extract_state(obs),
                    "prompt": args.subtasks[subtask_idx],
                }
                if args.task_description:
                    element["task_description"] = args.task_description

                t0 = time.perf_counter()
                infer_result = client.infer(element)
                elapsed = time.perf_counter() - t0

                full_actions = np.asarray(infer_result["actions"], dtype = np.float32)
                action_plan = full_actions[:, : args.real_action_dim]

                log_line = (
                    f"Episode {episode_idx} step {t}: subtask={subtask_idx} "
                    f"prompt={args.subtasks[subtask_idx]!r}, inference={elapsed:.3f}s"
                )
                q_values = infer_result.get("q_values")
                if q_values is not None:
                    values_str = ", ".join(f"{v:.4f}" for v in np.asarray(q_values).reshape(-1).tolist())
                    log_line += f", q_values=[{values_str}]"
                logger.info(log_line)

                predicted_subtask = infer_result.get("predicted_subtask")
                if predicted_subtask is not None:
                    logger.info(f"[subtask decode] step {t}: predicted={predicted_subtask!r}")

                if video_logger is not None:
                    # Snapshot the policy-sized frames: the mosaic panels are 256 px, so the
                    # full 1080p frames would only be copied to be shrunk again.
                    video_logger.record_predict(element_images["image"], q_values, t)

            plan_idx = min(t % args.query_freq, action_plan.shape[0] - 1)
            # YAMBimanualEnv.step blocks on the next broker frame, so this loop runs at the env's rate.
            obs, _, terminated, truncated, _ = env.step(action_plan[plan_idx])
            t += 1
            if terminated or truncated:
                break

        logger.info(f"Episode {episode_idx} finished after {t} steps")
    finally:
        if video_logger is not None:
            video_logger.finish_episode()


def load_lego_task(tasks_file: str, task_index: int) -> tuple[str, tuple[str, ...]]:
    """Task description and ordered subtask prompts for the task at `task_index`."""
    with open(tasks_file) as f:
        tasks = json.load(f)["tasks"]
    if not 0 <= task_index < len(tasks):
        raise IndexError(f"task_index {task_index} out of range for {tasks_file} ({len(tasks)} tasks).")
    entry = tasks[task_index]
    if entry["task_index"] != task_index:
        raise ValueError(f"{tasks_file} entry {task_index} carries task_index {entry['task_index']}.")
    subtasks = tuple(s["prompt"] for s in sorted(entry["subtasks"], key = lambda s: s["index"]))
    return entry["task_description"], subtasks


def main(args: Args) -> None:
    tasks_file = args.tasks_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lego_tasks.json")
    tasks_available = os.path.exists(tasks_file)
    if not tasks_available:
        logger.warning(f"tasks_file {tasks_file!r} not found; using --args.subtasks for every episode.")
    if not tasks_available and len(args.subtasks) == 0:
        raise ValueError("At least one subtask prompt is required (--args.subtasks).")

    _start_key_listener()

    logger.info(f"Connecting to YAM environment ({args.env_config})")
    env = YAMBimanualEnv(args.env_config)

    client = None
    try:
        for episode_idx in range(args.start_episode_idx, args.start_episode_idx + args.num_episodes):
            # Reconnect per episode so a response left buffered by a Ctrl-C'd inference in the
            # previous episode can't desync every request/response of the next one by one.
            if client is not None:
                try:
                    client._ws.close()
                except Exception as e:
                    logger.warning(f"Error closing previous client websocket (ignored, reconnecting anyway): {e}")
            logger.info(f"Connecting policy client to {args.policy_host}:{args.policy_port} for episode {episode_idx}")
            client = _websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
            image_helper = _eval_image_helper.EvalImageHelper.from_client(client)
            logger.info(
                f"EvalImageHelper: policy={image_helper.policy_image_size}, "
                f"critic={image_helper.critic_image_size}, "
                f"expect_critic_images={image_helper.expect_critic_images}"
            )

            if tasks_available:
                try:
                    args.task_description, args.subtasks = load_lego_task(tasks_file, episode_idx)
                except IndexError as e:
                    logger.error(f"No task available for episode {episode_idx}: {e}")
                    break
                logger.info(f"Loaded task #{episode_idx} from {os.path.basename(tasks_file)}")

            # Reset first so the arms are home while the scene is set up; the episode
            # itself starts on Enter. A Ctrl-C at this prompt ends the run.
            try:
                logger.info(f"Resetting the robot before episode {episode_idx}")
                obs = env._get_obs() if args.no_reset else env.reset()[0]
                logger.info(f"=== Episode {episode_idx} task: {args.task_description!r} ===")
                logger.info(f"=== Episode {episode_idx} subtasks: {list(args.subtasks)} ===")
                _wait_for_enter(
                    f"Set up the scene, then press Enter to start episode {episode_idx} "
                    f"(Enter during the episode advances the subtask; Ctrl-C ends it and saves the video)..."
                )
            except KeyboardInterrupt:
                logger.info("Ctrl+C at the start prompt; ending the run.")
                break
            try:
                run_episode(env, obs, client, image_helper, args, episode_idx)
            except KeyboardInterrupt:
                logger.info(f"Episode {episode_idx} interrupted by Ctrl+C; video saved, moving to the next episode.")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
