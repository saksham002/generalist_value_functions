"""Standalone sim_bimanual_assembly evaluation using a remote policy server.

Evaluates a served Pi-0.5 policy (config `sim_bimanual_assembly_pi05`) on the
`DoubleInsertDualXarmsGymEnv` MuJoCo env via the openpi WebSocket protocol.
The policy is served separately (e.g., via scripts/serve_policy.py) and this
script communicates with it over a WebSocket connection.

Both policy and env run at 60 Hz natively, so no fps resampling is performed
on the action chunk. The policy emits 14-D absolute EEF actions in the
dataset's frame (R_state * R_delta for rotation; 1 - state_grip - 0.1*delta for
gripper). The client converts each chunked step to env-space deltas before
calling env.step() (env expects left-mul rotation deltas and a per-step
gripper residual scaled by 0.1, mirroring how the bc_eval / data-collection
pipeline drives the sim).

Subtask prompt switching: the inference prompt is the current subtask string,
advanced monotonically by reward thresholds (1.0 → subtask 2; 2.0 → subtask 3).
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


import collections
import dataclasses
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation
import tqdm
import tyro

from dual_xarms_sim.fix_dual_xarms_sim import DoubleInsertDualXarmsGymEnv


_VIDEO_PANEL_SIZE = 256
# Header band above the 2x2 mosaic for the current subtask prompt. Keep the
# combined frame height (2 * _VIDEO_PANEL_SIZE + _VIDEO_HEADER_HEIGHT) divisible
# by 16 so libx264 doesn't auto-resize.
_VIDEO_HEADER_HEIGHT = 32
_RESIZE_SIZE = 224
_ENV_ACTION_FPS = 60.0


# Per user-supplied annotations file (subtask_definitions). Index in this list
# is the prompt_idx latched by reward thresholds.
SUBTASKS = [
    "Insert the white block into the pink one",
    "Insert the right end of this combination into the blue block",
    "Place the combination on the wooden platform",
]


class VideoLogger:
    """2x2 mosaic mp4: right/top, Q-value plot, left/wrist, right/wrist.

    Layout (each panel _VIDEO_PANEL_SIZE x _VIDEO_PANEL_SIZE):

        +-------------------+-------------------+
        |   right/top       |  Q-value plot     |
        +-------------------+-------------------+
        |   left/wrist      |  right/wrist      |
        +-------------------+-------------------+
    """

    def __init__(self, output_dir: str, fps: float, num_samples: int) -> None:
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
        self._subtasks: list[str] = []
        self._episode_idx: int | None = None

    def start_episode(self, episode_idx: int) -> None:
        self._reset()
        self._episode_idx = episode_idx

    def record_predict(
        self,
        images: dict[str, np.ndarray],
        q_values: np.ndarray | None,
        t: int,
        subtask: str = "",
    ) -> None:
        self._images.append({k: np.asarray(v).copy() for k, v in images.items()})
        if q_values is None:
            self._q_values.append(np.zeros((0,), dtype = np.float32))
        else:
            self._q_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
        self._steps.append(t)
        self._subtasks.append(subtask)

    def finish_episode(self, success: bool = False) -> None:
        if self._episode_idx is None or not self._images:
            return
        if self.num_samples > 0:
            q_matrix = np.stack(self._q_values, axis = 0)
        else:
            q_matrix = np.zeros((len(self._steps), 0), dtype = np.float32)
        steps = np.asarray(self._steps, dtype = np.int64)
        frames = [self._render_frame(i, q_matrix, steps) for i in range(len(self._images))]
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

        right_top = _panel(images.get("right/top"))
        left_wrist = _panel(images.get("left/wrist"))
        right_wrist = _panel(images.get("right/wrist"))
        if self.num_samples > 0 and q_matrix.shape[1] > 0:
            value_panel = self._render_value_plot(size, q_matrix, steps, current_step)
        else:
            value_panel = np.zeros((size, size, 3), dtype = np.uint8)

        top = np.concatenate([right_top, value_panel], axis = 1)
        bottom = np.concatenate([left_wrist, right_wrist], axis = 1)
        mosaic = np.concatenate([top, bottom], axis = 0)
        subtask = self._subtasks[frame_idx] if frame_idx < len(self._subtasks) else ""
        header = self._render_subtask_header(mosaic.shape[1], subtask)
        return np.concatenate([header, mosaic], axis = 0)

    def _render_subtask_header(self, width: int, subtask: str) -> np.ndarray:
        header = np.zeros((_VIDEO_HEADER_HEIGHT, width, 3), dtype = np.uint8)
        if not subtask:
            return header
        # Shrink the font until the text fits, since the subtask strings vary in
        # length (e.g., 'Place the combination on the wooden platform' vs the
        # shorter first subtask). 0.5 fits ~50 chars at width 512.
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        scale = 0.55
        while scale > 0.3:
            (text_w, text_h), _baseline = self._cv2.getTextSize(subtask, font, scale, thickness)
            if text_w <= width - 8:
                break
            scale -= 0.05
        x = max(4, (width - text_w) // 2)
        y = (_VIDEO_HEADER_HEIGHT + text_h) // 2
        self._cv2.putText(header, subtask, (x, y), font, scale, (255, 255, 255), thickness, self._cv2.LINE_AA)
        return header

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
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 30

    # Eval parameters
    split: str = "target"
    num_trials: int = 100
    # Used only for log-path grouping; the env class is hard-coded.
    task_set: list[str] = dataclasses.field(default_factory = lambda: ["DoubleInsertDualXarms"])

    # Logging
    log_dir: str | None = None
    # Sub-directory under <log_dir>/<env> selecting the eval method
    # ("bc" | "bestofn8" | "bestofn64"); set by the launching bash script to
    # match the served critic config. Empty => no method subdir.
    method: str = ""
    # Leaf run-tag sub-dir under <log_dir>/<env>/<method> (a date-time stamp);
    # set by the launching bash script. Empty => no run-tag subdir.
    run_tag: str = ""
    seed: int = 86
    log_videos: bool = False

    # Hard cap on env steps per episode. Default matches the env's internal
    # MAX_STEPS = control_freq(60) * time_limit(120 s) = 7200.
    max_steps: int = 7200

    # Resume support. Skip the first `start_episode_idx` indices in the
    # per-episode loop, preserving the deterministic per-index reset seed
    # (`seed + 100 * episode_idx`). If `resume_from_dir` is set, the client
    # writes into that existing run-date subdir instead of generating a new
    # timestamp; prior successes/failures are recovered by counting
    # `videos/successes/*.mp4` and `videos/failures/*.mp4`.
    start_episode_idx: int = 0
    resume_from_dir: str | None = None


def _build_state14(obs: dict) -> np.ndarray:
    """14-D EEF state: [lxyz(3), lrpy(3), 1-lgrip, rxyz(3), rrpy(3), 1-rgrip].

    Matches `_construct_eef_state` from the HDF5 RLDS dataset (use_eef=True,
    state_dim=14): the dataset builder writes `1 - gripper_pos` for each
    gripper slot (see sim_bimanual_assembly_config.py:199-206).
    """
    left_pose = np.asarray(obs["state"]["left/tcp_pose"], dtype = np.float32)
    right_pose = np.asarray(obs["state"]["right/tcp_pose"], dtype = np.float32)
    # env returns scalar-LAST quat (xyzw); scipy default is scalar-last.
    left_rpy = Rotation.from_quat(left_pose[3:7]).as_euler("xyz").astype(np.float32)
    right_rpy = Rotation.from_quat(right_pose[3:7]).as_euler("xyz").astype(np.float32)
    left_g = float(obs["state"]["left/gripper_pos"][0])
    right_g = float(obs["state"]["right/gripper_pos"][0])
    return np.concatenate([
        left_pose[:3], left_rpy, [1.0 - left_g],
        right_pose[:3], right_rpy, [1.0 - right_g],
    ]).astype(np.float32)


def _abs_action_to_env_delta(abs_action: np.ndarray, obs: dict, env) -> np.ndarray:
    """Convert a single 14-D absolute action from the policy to env-space delta.

    Layout matches the HDF5 builder (sim_bimanual_assembly_config.py):
      abs_action[0:3]   = left target xyz (absolute, dataset frame)
      abs_action[3:6]   = left target rpy (R_state * R_delta_train, dataset)
      abs_action[6]     = left absolute gripper field = 1 - state_grip - 0.1*delta
      abs_action[7:10]  = right target xyz
      abs_action[10:13] = right target rpy
      abs_action[13]    = right absolute gripper field

    env.step (fix_dual_xarms_sim.py:411-469) integrates the pos/rot deltas
    against the *mocap IK target*, not the observed TCP:
      new_pos  = mocap_pos[i]  + pos_delta
      new_quat = R(rpy_delta) * R(mocap_quat[i])      # left-multiply
    so to land exactly on the absolute target we invert against the current
    mocap reference (i=0 left, i=1 right): pos_delta = abs - mocap_pos and
    rpy_delta = (R_abs * R_mocap^-1). Read just before env.step, so the mocap
    arrays still hold the target the previous step/reset set (what env.step
    integrates from). mocap_quat is MuJoCo scalar-first (wxyz).

    Gripper has no mocap: env reads `ctrl/255` (== obs `*/gripper_pos`) and
    applies `new_g = cur_g + 0.1*g_env`; builder's absolute field is
    `1 - cur_g - 0.1*delta`, so `g_env = (1 - cur_g - abs_grip) / 0.1`.
    """
    data = env.unwrapped._data
    left_mocap_pos = np.asarray(data.mocap_pos[0], dtype = np.float32)
    right_mocap_pos = np.asarray(data.mocap_pos[1], dtype = np.float32)
    r_mocap_l = Rotation.from_quat(np.asarray(data.mocap_quat[0]), scalar_first = True)
    r_mocap_r = Rotation.from_quat(np.asarray(data.mocap_quat[1]), scalar_first = True)
    cur_lg = float(obs["state"]["left/gripper_pos"][0])
    cur_rg = float(obs["state"]["right/gripper_pos"][0])

    lpos_delta = abs_action[0:3].astype(np.float32) - left_mocap_pos
    r_abs_l = Rotation.from_euler("xyz", abs_action[3:6])
    lrpy_delta = (r_abs_l * r_mocap_l.inv()).as_euler("xyz").astype(np.float32)
    lgrip_env = np.float32((1.0 - cur_lg - float(abs_action[6])) / 0.1)

    rpos_delta = abs_action[7:10].astype(np.float32) - right_mocap_pos
    r_abs_r = Rotation.from_euler("xyz", abs_action[10:13])
    rrpy_delta = (r_abs_r * r_mocap_r.inv()).as_euler("xyz").astype(np.float32)
    rgrip_env = np.float32((1.0 - cur_rg - float(abs_action[13])) / 0.1)

    return np.concatenate([
        lpos_delta, lrpy_delta, [lgrip_env],
        rpos_delta, rrpy_delta, [rgrip_env],
    ]).astype(np.float32)


def eval_main(args: Args) -> None:
    np.random.seed(args.seed)

    env_names = list(args.task_set) if args.task_set else ["DoubleInsertDualXarms"]
    logging.info(f"Evaluating {len(env_names)} environments: {env_names}")

    for env_name in env_names:
        eval_env(
            env_name,
            split = args.split,
            log_dir = args.log_dir,
            method = args.method,
            run_tag = args.run_tag,
            num_trials = args.num_trials,
            replan_steps = args.replan_steps,
            host = args.host,
            port = args.port,
            seed = args.seed,
            log_videos = args.log_videos,
            max_steps = args.max_steps,
            start_episode_idx = args.start_episode_idx,
            resume_from_dir = args.resume_from_dir,
        )


def eval_env(
    env_name: str,
    split: str,
    log_dir: str | None,
    method: str,
    run_tag: str,
    num_trials: int,
    replan_steps: int,
    host: str,
    port: int,
    seed: int,
    log_videos: bool = False,
    max_steps: int = 7200,
    start_episode_idx: int = 0,
    resume_from_dir: str | None = None,
) -> None:
    horizon = max_steps

    log_path = None
    file_handler: logging.FileHandler | None = None
    resuming = resume_from_dir is not None
    if resuming:
        log_path = pathlib.Path(resume_from_dir)
        assert log_path.exists(), f"resume_from_dir does not exist: {log_path}"
        assert not (log_path / "stats.json").exists(), (
            f"stats.json already exists at {log_path}; that run already finished. "
            "Refusing to resume."
        )
        file_handler = logging.FileHandler(log_path / "eval.log", mode = "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logging.info(
            f"Resuming eval in {log_path} from start_episode_idx={start_episode_idx}"
        )
    elif log_dir is not None:
        # log_dir is a pure base (no task — the gym env names the task).
        # Final tree: <log_dir>/<env>/<method>/<run_tag>/ , where method and
        # run_tag (a date-time stamp) are passed by the launching bash script.
        log_path = pathlib.Path(log_dir) / env_name
        if method:
            log_path = log_path / method
        if run_tag:
            log_path = log_path / run_tag

        if (log_path / "stats.json").exists():
            logging.info(f"stats.json exists at {log_path}, skipping.")
            return

        log_path.mkdir(parents = True, exist_ok = True)

        file_handler = logging.FileHandler(log_path / "eval.log", mode = "w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(file_handler)

    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    env = DoubleInsertDualXarmsGymEnv(
        control_freq = 60, time_limit = 120, render_mode = "rgb_array", image_obs = True,
    )

    video_logger = None
    if log_videos and log_path is not None:
        video_logger = "pending"

    total_episodes, total_successes = 0, 0
    if resuming and log_path is not None:
        prior_succ = sorted((log_path / "videos" / "successes").glob("episode_*.mp4"))
        prior_fail = sorted((log_path / "videos" / "failures").glob("episode_*.mp4"))
        total_successes = len(prior_succ)
        total_episodes = total_successes + len(prior_fail)
        logging.info(
            f"Recovered prior counts from {log_path}: "
            f"{total_successes} successes, {len(prior_fail)} failures, "
            f"{total_episodes} total."
        )
    call_q_var_succ: list[float] = []
    call_q_var_fail: list[float] = []
    call_act_var_succ: list[float] = []
    call_act_var_fail: list[float] = []
    # Per-episode mean ACS (average cosine similarity across the BestOfN
    # candidate pool), bucketed by episode outcome.
    episode_acs_succ: list[float] = []
    episode_acs_fail: list[float] = []
    for episode_idx in tqdm.tqdm(range(start_episode_idx, num_trials), desc = env_name, initial = start_episode_idx, total = num_trials):
        obs, info = env.reset(seed = seed + 100 * episode_idx)
        action_plan = collections.deque()
        done = False
        prompt_idx = 0
        ep_call_q_vars: list[float] = []
        ep_call_act_vars: list[float] = []
        ep_call_acs: list[float] = []

        if isinstance(video_logger, VideoLogger):
            video_logger.start_episode(episode_idx)

        for t in range(horizon):
            right_top = np.ascontiguousarray(obs["images"]["right/top"])
            left_wrist = np.ascontiguousarray(obs["images"]["left/wrist"])
            right_wrist = np.ascontiguousarray(obs["images"]["right/wrist"])
            base_img = image_tools.convert_to_uint8(image_tools.resize_stretch(right_top, _RESIZE_SIZE, _RESIZE_SIZE))
            left_wrist_img = image_tools.convert_to_uint8(image_tools.resize_stretch(left_wrist, _RESIZE_SIZE, _RESIZE_SIZE))
            right_wrist_img = image_tools.convert_to_uint8(image_tools.resize_stretch(right_wrist, _RESIZE_SIZE, _RESIZE_SIZE))

            if not action_plan:
                state14 = _build_state14(obs)
                element = {
                    "image": {
                        "base_0_rgb": base_img,
                        "left_wrist_0_rgb": left_wrist_img,
                        "right_wrist_0_rgb": right_wrist_img,
                    },
                    "state": state14,
                    "prompt": SUBTASKS[prompt_idx],
                }

                infer_t0 = time.perf_counter()
                infer_result = client.infer(element)
                roundtrip_ms = (time.perf_counter() - infer_t0) * 1000.0
                action_chunk = np.asarray(infer_result["actions"], dtype = np.float32)
                if action_chunk.shape[0] >= 2:
                    ep_call_act_vars.append(
                        float(np.mean(np.var(action_chunk, axis = 0)))
                    )
                q_values = infer_result.get("q_values")
                if q_values is not None:
                    q_arr_for_var = np.asarray(q_values, dtype = np.float32).reshape(-1)
                    if q_arr_for_var.size >= 2:
                        ep_call_q_vars.append(float(np.var(q_arr_for_var)))
                acs_value = infer_result.get("acs")
                if acs_value is not None and np.isfinite(acs_value):
                    ep_call_acs.append(float(acs_value))
                server_ms = infer_result.get("policy_timing", {}).get("infer_ms")
                server_str = f"{server_ms:.1f}ms" if server_ms is not None else "n/a"
                q_str = ""
                if q_values is not None:
                    q_arr = np.asarray(q_values).reshape(-1)
                    q_str = ", q_values=[" + ", ".join(f"{v:.4f}" for v in q_arr.tolist()) + "]"
                logging.info(
                    f"[t={t}] prompt_idx={prompt_idx} infer: server={server_str}, round-trip={roundtrip_ms:.1f}ms{q_str}"
                )
                assert len(action_chunk) >= replan_steps, (
                    f"Policy predicts {len(action_chunk)} steps but replan_steps={replan_steps}"
                )
                action_plan.extend(action_chunk[:replan_steps])

                if video_logger == "pending":
                    q_arr = np.asarray(q_values).reshape(-1) if q_values is not None else None
                    video_logger = VideoLogger(
                        output_dir = str(log_path / "videos"),
                        fps = max(1.0, _ENV_ACTION_FPS / replan_steps),
                        num_samples = int(q_arr.shape[0]) if q_arr is not None else 0,
                    )
                    video_logger.start_episode(episode_idx)
                if isinstance(video_logger, VideoLogger):
                    video_logger.record_predict(
                        images = {
                            "right/top": right_top,
                            "left/wrist": left_wrist,
                            "right/wrist": right_wrist,
                        },
                        q_values = (np.asarray(q_values).reshape(-1) if q_values is not None else None),
                        t = t,
                        subtask = SUBTASKS[prompt_idx],
                    )

            abs_action = action_plan.popleft()
            env_action = _abs_action_to_env_delta(abs_action, obs, env)
            obs, reward, terminated, truncated, info = env.step(env_action)
            r = float(reward)
            # Monotonic subtask latch: advance the prompt as soon as the
            # corresponding reward threshold is observed at ANY step, even if
            # reward subsequently drops back (env reward is non-monotonic).
            if r >= 2.0:
                prompt_idx = max(prompt_idx, 2)
            elif r >= 1.0:
                prompt_idx = max(prompt_idx, 1)
            if r >= 3.0:
                done = True
                total_successes += 1
                break
            if truncated:
                break

        total_episodes += 1

        if isinstance(video_logger, VideoLogger):
            video_logger.finish_episode(success = bool(done))

        logging.info(f"Episode {total_episodes}: {'success' if done else 'failure'}")
        logging.info(f"Running: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)")
        logging.info(f"Episode {total_episodes} final prompt_idx={prompt_idx}")

        ep_q_var_mean = float(np.mean(ep_call_q_vars)) if ep_call_q_vars else float("nan")
        ep_act_var_mean = float(np.mean(ep_call_act_vars)) if ep_call_act_vars else float("nan")
        ep_acs_mean = float(np.mean(ep_call_acs)) if ep_call_acs else float("nan")
        logging.info(
            f"Episode {total_episodes} variance (per-call mean): q={ep_q_var_mean:.6f}, "
            f"action_pre_interp={ep_act_var_mean:.6f}, acs={ep_acs_mean:.6f}"
        )

        if done:
            call_q_var_succ.extend(ep_call_q_vars)
            call_act_var_succ.extend(ep_call_act_vars)
            if ep_call_acs:
                episode_acs_succ.append(ep_acs_mean)
        else:
            call_q_var_fail.extend(ep_call_q_vars)
            call_act_var_fail.extend(ep_call_act_vars)
            if ep_call_acs:
                episode_acs_fail.append(ep_acs_mean)

    logging.info(
        f"[{env_name}] Final: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)"
    )

    def _mean_or_nan(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    q_var_succ_mean = _mean_or_nan(call_q_var_succ)
    q_var_fail_mean = _mean_or_nan(call_q_var_fail)
    act_var_succ_mean = _mean_or_nan(call_act_var_succ)
    act_var_fail_mean = _mean_or_nan(call_act_var_fail)
    acs_succ_mean = _mean_or_nan(episode_acs_succ)
    acs_fail_mean = _mean_or_nan(episode_acs_fail)
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
    logging.info(
        f"[{env_name}] ACS (avg cosine sim across BestOfN pool, per-episode mean) — success: "
        f"{acs_succ_mean:.6f} (n_episodes={len(episode_acs_succ)}), failure: "
        f"{acs_fail_mean:.6f} (n_episodes={len(episode_acs_fail)})"
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
                    "acs_success_mean": acs_succ_mean,
                    "acs_failure_mean": acs_fail_mean,
                    "num_success_episodes_acs": len(episode_acs_succ),
                    "num_failure_episodes_acs": len(episode_acs_fail),
                    "num_success_calls": len(call_q_var_succ),
                    "num_failure_calls": len(call_q_var_fail),
                },
                f,
                indent = 4,
            )

    env.close()
    del env

    if file_handler is not None:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    eval_main(tyro.cli(Args))
