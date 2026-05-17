"""Standalone RoboCasa evaluation using a remote policy server.

Evaluates a served policy on RoboCasa tasks via the openpi WebSocket protocol.
The policy is served separately (e.g., via scripts/serve_policy.py) and this
script communicates with it over a WebSocket connection.

Usage:
    # Serve the policy on a GPU machine:
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_robocasa_rlds \
        --policy.dir=<checkpoint_path>

    # Run evaluation (same or different machine):
    uv run examples/robocasa/main.py \
        --task-set target_atomic_seen \
        --split target \
        --num-trials 50 \
        --log-dir data/robocasa
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


import collections
import dataclasses
from datetime import UTC
from datetime import datetime
import json
import logging
import os
import pathlib
import time

import gymnasium as gym
import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def _interpolate_action_chunk(
    action_chunk: np.ndarray, source_fps: float, target_fps: float,
) -> np.ndarray:
    """Resample a (T_src, 12) RoboCasa-native action chunk source_fps -> target_fps.
    Mirrors `state_action_spaces.interpolate_trajectory_tf` on `ROBOCASA_NATIVE_ACTION_SPEC`
    but treats slot [8:11] as axis-angle (rotvec) with quaternion SLERP."""
    if abs(source_fps - target_fps) < 1e-6:
        return action_chunk
    chunk = np.asarray(action_chunk, dtype=np.float32)
    T_src = chunk.shape[0]
    duration = (T_src - 1) / source_fps
    T_tgt = int(round(duration * target_fps)) + 1
    src_t = np.arange(T_src) / source_fps
    tgt_t = np.clip(np.arange(T_tgt) / target_fps, src_t[0], src_t[-1])

    out = np.zeros((T_tgt, 12), dtype=np.float32)
    for d in (0, 1, 2, 3, 5, 6, 7):
        out[:, d] = np.interp(tgt_t, src_t, chunk[:, d])
    nearest = np.clip(np.searchsorted(src_t, tgt_t, side="right") - 1, 0, T_src - 1)
    out[:, 4] = chunk[nearest, 4]
    out[:, 11] = chunk[nearest, 11]
    aa = chunk[:, 8:11]
    src_rots = Rotation.from_rotvec(aa)
    interp_rots = Slerp(src_t, src_rots)(tgt_t)
    out[:, 8:11] = interp_rots.as_rotvec().astype(np.float32)
    return out
import tqdm
import tyro

# 256 -> 512x512 final, satisfies libx264's "divisible by 16".
_VIDEO_PANEL_SIZE = 256


class VideoLogger:
    """Buffers per-replan camera frames + Q-values and writes a 2x2 mosaic mp4.

    Layout (panels are _VIDEO_PANEL_SIZE x _VIDEO_PANEL_SIZE):

        +-------------------+-------------------+
        |  agentview left   |  Q-value plot     |
        +-------------------+-------------------+
        |  agentview right  |  eye-in-hand RGB  |
        +-------------------+-------------------+

    Animated panel: Q-value plot (lines = each candidate's Q over replan ticks,
    static across frames; red vertical = current frame's tick).

    Ported from batch_value_learning_eval/eval/xarm_scripts/eval_shirt_hang_remote.py
    `VideoLogger`, simplified for the RoboCasa camera set (no manual-subtask
    advance markers).
    """

    def __init__(self, output_dir: str, fps: float, num_samples: int) -> None:
        # Lazy imports keep the rest of main.py runnable when matplotlib/cv2
        # are absent (default no-video path).
        import cv2 as _cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        self._cv2 = _cv2
        self._plt = _plt
        self.output_dir = output_dir
        self.fps = fps
        self.num_samples = num_samples
        os.makedirs(output_dir, exist_ok = True)
        self._reset()

    def _reset(self) -> None:
        self._images: list[dict[str, np.ndarray]] = []
        self._q_values: list[np.ndarray] = []
        self._steps: list[int] = []
        self._episode_idx: int | None = None

    def start_episode(self, episode_idx: int) -> None:
        self._reset()
        self._episode_idx = episode_idx

    def record_predict(self, images: dict[str, np.ndarray], q_values: np.ndarray | None, t: int) -> None:
        self._images.append({k: np.asarray(v).copy() for k, v in images.items()})
        if q_values is None:
            self._q_values.append(np.zeros((0,), dtype = np.float32))
        else:
            self._q_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
        self._steps.append(t)

    def finish_episode(self, success: bool = False) -> None:
        if self._episode_idx is None or not self._images:
            return
        if self.num_samples > 0:
            q_matrix = np.stack(self._q_values, axis = 0)  # (T_replans, N)
        else:
            q_matrix = np.zeros((len(self._steps), 0), dtype = np.float32)
        steps = np.asarray(self._steps, dtype = np.int64)
        frames = [self._render_frame(i, q_matrix, steps) for i in range(len(self._images))]
        # Group output mp4s by episode outcome so successes/ vs failures/ can
        # be inspected separately.
        sub = "successes" if success else "failures"
        sub_dir = os.path.join(self.output_dir, sub)
        os.makedirs(sub_dir, exist_ok = True)
        out_path = os.path.join(sub_dir, f"episode_{self._episode_idx}.mp4")
        # imageio bundles its own ffmpeg with libx264; system ffmpeg on this
        # cluster lacks libx264 (see CLAUDE.md > "Saving Videos on HPC").
        imageio.mimsave(out_path, frames, format = "mp4", fps = self.fps, codec = "libx264", quality = 8)
        logging.info(f"Saved episode video: {out_path}")

    def _render_frame(self, frame_idx: int, q_matrix: np.ndarray, steps: np.ndarray) -> np.ndarray:
        images = self._images[frame_idx]
        current_step = int(steps[frame_idx])
        size = _VIDEO_PANEL_SIZE

        def _panel(img):
            if img is None:
                return np.zeros((size, size, 3), dtype = np.uint8)
            return self._cv2.resize(img, (size, size), interpolation = self._cv2.INTER_AREA)

        agentview_left = _panel(images.get("agentview_left"))
        agentview_right = _panel(images.get("agentview_right"))
        eye_in_hand = _panel(images.get("eye_in_hand"))
        # In BC mode (no Q-values) the top-right panel is just an empty
        # placeholder so the 2x2 mosaic still renders.
        if self.num_samples > 0 and q_matrix.shape[1] > 0:
            value_panel = self._render_value_plot(size, q_matrix, steps, current_step)
        else:
            value_panel = np.zeros((size, size, 3), dtype = np.uint8)

        top = np.concatenate([agentview_left, value_panel], axis = 1)
        bottom = np.concatenate([agentview_right, eye_in_hand], axis = 1)
        return np.concatenate([top, bottom], axis = 0)

    def _render_value_plot(self, size: int, q_matrix: np.ndarray, steps: np.ndarray, current_step: int) -> np.ndarray:
        plt = self._plt
        dpi = 100
        figsize = (size / dpi, size / dpi)
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        for sample_idx in range(q_matrix.shape[1]):
            ax.plot(steps, q_matrix[:, sample_idx], linewidth = 0.8)
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
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        if buf.shape[0] != size or buf.shape[1] != size:
            buf = self._cv2.resize(buf, (size, size), interpolation = self._cv2.INTER_AREA)
        return buf


@dataclasses.dataclass
class Args:
    # Model-emit fps -> env-step fps. Equal => pass-through. Required (no
    # defaults). Listed first so the dataclass ordering rule
    # (non-default fields must precede default fields) is satisfied.
    model_action_fps: float
    env_action_fps: float

    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # RoboCasa evaluation parameters
    split: str = "target"
    num_trials: int = 50
    task_set: list[str] = dataclasses.field(default_factory=lambda: ["target_atomic_seen"])

    # Logging
    log_dir: str | None = None
    seed: int = 7
    # Save per-episode 2x2 mosaic mp4 (cameras + Q-value plot if available).
    log_videos: bool = False

    # Hard cap on env steps per episode. Default = 1200 (~1 min @ 20 Hz, the
    # robocasa native env_action_fps). The per-task horizon from RoboCasa
    # (`get_task_horizon(env_name) * 1.5`) is also enforced — whichever is
    # smaller wins, so max_steps acts as a ceiling for tasks with very long
    # horizons.
    max_steps: int = 1200


def eval_main(args: Args) -> None:
    np.random.seed(args.seed)

    all_env_names = []
    for task in args.task_set:
        if task in TASK_SET_REGISTRY:
            all_env_names.extend(TASK_SET_REGISTRY[task])
        else:
            # Treat as a direct environment name.
            all_env_names.append(task)

    logging.info(f"Evaluating {len(all_env_names)} environments from task sets: {args.task_set}")

    for env_name in all_env_names:
        eval_env(
            env_name,
            split=args.split,
            log_dir=args.log_dir,
            num_trials=args.num_trials,
            resize_size=args.resize_size,
            replan_steps=args.replan_steps,
            host=args.host,
            port=args.port,
            seed=args.seed,
            log_videos=args.log_videos,
            model_action_fps=args.model_action_fps,
            env_action_fps=args.env_action_fps,
            max_steps=args.max_steps,
        )


def eval_env(
    env_name: str,
    split: str,
    log_dir: str | None,
    num_trials: int,
    resize_size: int,
    replan_steps: int,
    host: str,
    port: int,
    seed: int,
    log_videos: bool = False,
    model_action_fps: float = ...,
    env_action_fps: float = ...,
    max_steps: int = 1200,
) -> None:
    task_horizon = get_task_horizon(env_name)
    # Cap the per-episode horizon at `max_steps` so tasks with long horizons
    # don't run unbounded; default 1200 ≈ 1 min at 20 Hz.
    horizon = min(int(task_horizon * 1.5), max_steps)

    # Set up logging directory
    log_path = None
    file_handler: logging.FileHandler | None = None
    if log_dir is not None:
        now_formatted = datetime.now(tz=UTC).strftime("%Y-%m-%d-%H-%M")
        log_path = pathlib.Path(log_dir) / "evals" / split / env_name / now_formatted

        # Skip if already evaluated
        for _root, _dirs, files in os.walk(os.path.dirname(str(log_path))):
            if "stats.json" in files:
                logging.info(f"{env_name}/{split}, stats path exists, skipping.")
                return

        log_path.mkdir(parents=True, exist_ok=True)

        # Mirror INFO+ records to <log_path>/eval.log so per-step inference
        # timings + episode outcomes persist alongside the video / stats.json.
        file_handler = logging.FileHandler(log_path / "eval.log", mode = "w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(file_handler)

    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)

    video_logger = None
    if log_videos and log_path is not None:
        # Lazy init after first infer (need num_samples from q_values shape; 0 in BC mode).
        video_logger = "pending"

    total_episodes, total_successes = 0, 0
    # Flat per-replan-call variance buffers, tagged by episode outcome.
    # (1) cross-sample q-value variance per call; (2) within-chunk per-dim
    # action variance (pre-interpolation, averaged over dims) per call.
    # Single mean at the end across all calls in success/failure episodes.
    call_q_var_succ: list[float] = []
    call_q_var_fail: list[float] = []
    call_act_var_succ: list[float] = []
    call_act_var_fail: list[float] = []
    for episode_idx in tqdm.tqdm(range(num_trials), desc=env_name):
        # Per-episode reset seed: deterministic across runs (same `seed` →
        # same starting configuration for episode_idx), and spaced enough
        # apart to keep adjacent episodes' RNG streams disjoint.
        obs, info = env.reset(seed = seed + 100 * episode_idx)
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()
        done = False
        # Per-call variance buffers for this episode.
        ep_call_q_vars: list[float] = []
        ep_call_act_vars: list[float] = []

        if isinstance(video_logger, VideoLogger):
            video_logger.start_episode(episode_idx)

        for t in range(horizon):
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            # Stretch resize matches tf.image.resize used by the RLDS dataset
            # builder during training (src/openpi/training/rlds_dataset.py:805).
            # resize_with_pad would letterbox these frames and the model has
            # never seen black bars on top/bottom.
            img = image_tools.convert_to_uint8(image_tools.resize_stretch(img, resize_size, resize_size))
            img_right = image_tools.convert_to_uint8(image_tools.resize_stretch(img_right, resize_size, resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_stretch(wrist_img, resize_size, resize_size))

            if not action_plan:
                # Construct state in modality.json order
                state = np.concatenate(
                    (
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.gripper_qpos"],
                    ),
                    axis=0,
                )

                element = {
                    "observation/image": img,
                    "observation/image_right": img_right,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                infer_t0 = time.perf_counter()
                infer_result = client.infer(element)
                roundtrip_ms = (time.perf_counter() - infer_t0) * 1000.0
                action_chunk = infer_result["actions"]
                # Pre-interpolation chunk variance: var across time axis per
                # dim, then mean over dims. Captures how variable the
                # selected action sequence is across the model horizon.
                pre_interp_chunk = np.asarray(action_chunk, dtype = np.float32)
                if pre_interp_chunk.shape[0] >= 2:
                    ep_call_act_vars.append(
                        float(np.mean(np.var(pre_interp_chunk, axis = 0)))
                    )
                # Resample model fps -> env fps. No-op when source==target.
                action_chunk = _interpolate_action_chunk(
                    np.asarray(action_chunk, dtype=np.float32),
                    source_fps=model_action_fps,
                    target_fps=env_action_fps,
                )
                q_values = infer_result.get("q_values")
                if q_values is not None:
                    q_arr_for_var = np.asarray(q_values, dtype = np.float32).reshape(-1)
                    if q_arr_for_var.size >= 2:
                        ep_call_q_vars.append(float(np.var(q_arr_for_var)))
                # Server-reported compute time (set by BestOfNPolicy.infer / Policy.infer).
                server_ms = infer_result.get("policy_timing", {}).get("infer_ms")
                server_str = f"{server_ms:.1f}ms" if server_ms is not None else "n/a"
                q_str = ""
                if q_values is not None:
                    q_arr = np.asarray(q_values).reshape(-1)
                    q_str = ", q_values=[" + ", ".join(f"{v:.4f}" for v in q_arr.tolist()) + "]"
                logging.info(
                    f"[t={t}] infer: server={server_str}, round-trip={roundtrip_ms:.1f}ms{q_str}"
                )
                assert len(action_chunk) >= replan_steps, (
                    f"Policy predicts {len(action_chunk)} steps but replan_steps={replan_steps}"
                )
                action_plan.extend(action_chunk[:replan_steps])

                if video_logger == "pending":
                    q_arr = np.asarray(q_values).reshape(-1) if q_values is not None else None
                    video_logger = VideoLogger(
                        output_dir = str(log_path / "videos"),
                        fps = max(1.0, env_action_fps / replan_steps),
                        num_samples = int(q_arr.shape[0]) if q_arr is not None else 0,
                    )
                    video_logger.start_episode(episode_idx)
                if isinstance(video_logger, VideoLogger):
                    video_logger.record_predict(
                        images = {
                            "agentview_left": img,
                            "agentview_right": img_right,
                            "eye_in_hand": wrist_img,
                        },
                        q_values = (np.asarray(q_values).reshape(-1) if q_values is not None else None),
                        t = t,
                    )

            abs_action = action_plan.popleft()
            # 12D RoboCasa-native: [0:4] base, [4:5] mode, [5:8] eef_pos_abs, [8:11] eef_rot_aa, [11:12] gripper.
            # OSC needs deltas from the live eef state (only env knows state[t+k]).
            cur_eef_pos = np.asarray(obs["state.end_effector_position_relative"], dtype = np.float32)
            cur_eef_quat_xyzw = np.asarray(obs["state.end_effector_rotation_relative"], dtype = np.float32)
            abs_eef_pos = abs_action[5:8].astype(np.float32)
            abs_eef_aa = abs_action[8:11].astype(np.float32)
            delta_eef_pos = abs_eef_pos - cur_eef_pos
            r_abs = Rotation.from_rotvec(abs_eef_aa)
            r_cur = Rotation.from_quat(cur_eef_quat_xyzw)
            delta_eef_aa = (r_abs * r_cur.inv()).as_rotvec().astype(np.float32)

            # convert_action expects:
            #   [0:3] ee_pos, [3:6] ee_rot, [6:7] gripper, [7:11] base_motion, [11:12] control_mode
            action = np.concatenate(
                [
                    delta_eef_pos,           # end_effector_position (delta)
                    delta_eef_aa,            # end_effector_rotation (axis-angle delta)
                    abs_action[11:12],       # gripper_close (absolute, untouched)
                    abs_action[0:4],         # base_motion (untouched; bimanual fine-tune
                                             # always emits zeros here per RoboCasaBimanualEEFOutputs)
                    abs_action[4:5],         # control_mode (untouched; same as above)
                ]
            )
            action = convert_action(action)

            obs, reward, terminated, truncated, info = env.step(action)
            # Sparse robocasa reward: 1.0 iff _check_success. Terminate on first hit.
            if float(reward) >= 1.0:
                done = True
                total_successes += 1
                break

        total_episodes += 1

        if isinstance(video_logger, VideoLogger):
            video_logger.finish_episode(success = bool(done))

        logging.info(f"Episode {total_episodes}: {'success' if done else 'failure'}")
        logging.info(f"Running: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)")

        ep_q_var_mean = float(np.mean(ep_call_q_vars)) if ep_call_q_vars else float("nan")
        ep_act_var_mean = float(np.mean(ep_call_act_vars)) if ep_call_act_vars else float("nan")
        logging.info(
            f"Episode {total_episodes} variance (per-call mean): q={ep_q_var_mean:.6f}, "
            f"action_pre_interp={ep_act_var_mean:.6f}"
        )

        if done:
            call_q_var_succ.extend(ep_call_q_vars)
            call_act_var_succ.extend(ep_call_act_vars)
        else:
            call_q_var_fail.extend(ep_call_q_vars)
            call_act_var_fail.extend(ep_call_act_vars)

    logging.info(
        f"[{env_name}] Final: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)"
    )

    def _mean_or_nan(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    q_var_succ_mean = _mean_or_nan(call_q_var_succ)
    q_var_fail_mean = _mean_or_nan(call_q_var_fail)
    act_var_succ_mean = _mean_or_nan(call_act_var_succ)
    act_var_fail_mean = _mean_or_nan(call_act_var_fail)
    logging.info(
        f"[{env_name}] q_value variance over n samples — success: {q_var_succ_mean:.6f} "
        f"(n_calls={len(call_q_var_succ)}), failure: {q_var_fail_mean:.6f} "
        f"(n_calls={len(call_q_var_fail)})"
    )
    logging.info(
        f"[{env_name}] action pre-interp per-dim variance (avg over dims) — success: "
        f"{act_var_succ_mean:.6f} (n_calls={len(call_act_var_succ)}), failure: "
        f"{act_var_fail_mean:.6f} (n_calls={len(call_act_var_fail)})"
    )

    if log_path is not None:
        with open(log_path / "stats.json", "w") as f:
            json.dump(
                {
                    "num_episodes": total_episodes,
                    "success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
                    "q_value_variance_success_mean": q_var_succ_mean,
                    "q_value_variance_failure_mean": q_var_fail_mean,
                    "action_pre_interp_variance_success_mean": act_var_succ_mean,
                    "action_pre_interp_variance_failure_mean": act_var_fail_mean,
                    "num_success_calls": len(call_q_var_succ),
                    "num_failure_calls": len(call_q_var_fail),
                },
                f,
                indent=4,
            )

    env.env.close()
    del env.env
    del env

    if file_handler is not None:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_main(tyro.cli(Args))
