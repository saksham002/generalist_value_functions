"""Standalone sim_xarm_packing evaluation using a remote policy server.

Evaluates a served Pi-0.5 policy (config `sim_xarm_packing_pi05_subtask`) on the
`PackingEnv` MuJoCo env (kshitiz-1225/vla_exploration, dual_xarms_sim) via the
openpi WebSocket protocol. The policy is served separately (via
scripts/serve_policy_sim_xarm_packing.sh) and this script communicates with it
over a WebSocket connection.

Set the PACKING_RAC_PATH env var to the `RAC/dual_xarms/dual_xarms_sim`
directory of a vla_exploration checkout so `dual_xarms_sim.packing_env`
resolves to the packing version of the package (it shadows any installed
dual_xarms_sim because PYTHONPATH/sys.path entries precede site-packages).

Action semantics (verified against the sim_xarm_packing RLDS dataset:
action[t] == state[t+1] exactly): the policy emits 14-D ABSOLUTE next-step EEF
targets in the base frame, [x, y, z, roll, pitch, yaw (scipy 'xyz' euler),
gripper] per arm, with the gripper as a raw absolute position in [0, 1] (same
scale as the env's `ctrl/255` — no `1 - g` inversion, unlike
sim_bimanual_assembly). The client inverts each absolute target against the
current MOCAP IK target (what env.step integrates from, mirroring the
sim_bimanual_assembly client) before env.step() (pos: `abs - mocap_pos`,
rot: `R_abs * R_mocap^-1` matching env.step's left-multiply integration,
gripper: `(abs - cur_g) / 0.04` against the observed gripper position).

Prompt / subtask flow: the policy was trained with prompt_mode="subtask" and
the critic with prompt_mode="task_description_predict_current_subtask" +
predict_subtask_ar=True. Each infer call sends the episode's task instruction
as `task_description` (consumed by the critic + its AR subtask decoder) and a
client-computed GROUND-TRUTH subtask as `prompt` (the policy's language
conditioning): the first not-yet-completed entry of the episode's subtask
schedule, latched MONOTONICALLY off the env's own event stream ("pack"
events for pack subtasks, "clutter_removed" events for remove subtasks —
the same signals the env's reward/done logic emits, so no completion
geometry is reimplemented; unpack/clutter_returned regressions are ignored).
Serving with --critic.policy-use-decoded-subtask instead conditions the
policy on the critic's AR-decoded subtask server-side, ignoring the client
prompt.

Episode specification: --episodes-json points to a JSON list with one entry per
episode (at least --num-trials entries). Each entry must carry the full scene
spec — a task instruction alone cannot initialize PackingEnv.reset(), which
needs trays/objects/assignments/initial_tray_contents:
    [
      {"goal_id": "SP-B1", "instruction": "..."},          # lookup in GOAL_LOOKUP
      {"goal": {"goal_id": "custom-0", "trays": [...], "objects": [...],
                "description": "...", "assignments": {...},
                "initial_tray_contents": {...}, "clutter_objects": [...]},
       "instruction": "..."}                                # inline PackingGoalSpec
    ]
`instruction` is optional and defaults to the goal's `description`.
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


import collections
import copy
import dataclasses
import json
import logging
import pathlib
import re
import sys
import time

import imageio
import numpy as np
from openpi_client import eval_image_helper as _eval_image_helper
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation
import tqdm
import tyro

if packing_rac_path := os.environ.get("PACKING_RAC_PATH"):
    sys.path.insert(0, packing_rac_path)

from dual_xarms_sim.packing_env import PackingEnv
from dual_xarms_sim.packing_goals_config import GOAL_LOOKUP
from dual_xarms_sim.packing_goals_config import PackingGoalSpec


_VIDEO_PANEL_SIZE = 256
# Header band above the 2x2 mosaic: two lines (critic-decoded subtask +
# simulator-ground-truth subtask). Keep the combined frame height
# (2 * _VIDEO_PANEL_SIZE + _VIDEO_HEADER_HEIGHT) divisible by 16 so libx264
# doesn't auto-resize (2*256 + 48 = 560 = 16*35).
_VIDEO_HEADER_HEIGHT = 48

# PackingEnv gripper delta scale: step() applies new_g = clip(cur_g + 0.04 * a).
_GRIPPER_DELTA_SCALE = 0.04


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
        self._gt_subtasks: list[str] = []
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
        gt_subtask: str = "",
    ) -> None:
        self._images.append({k: np.asarray(v).copy() for k, v in images.items()})
        if q_values is None:
            self._q_values.append(np.zeros((0,), dtype = np.float32))
        else:
            self._q_values.append(np.asarray(q_values, dtype = np.float32).reshape(-1))
        self._steps.append(t)
        self._subtasks.append(subtask)
        self._gt_subtasks.append(gt_subtask)

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
        """Render the header band; `subtask` may hold multiple '\\n'-separated
        lines (critic-decoded subtask + ground-truth subtask)."""
        header = np.zeros((_VIDEO_HEADER_HEIGHT, width, 3), dtype = np.uint8)
        if not subtask:
            return header
        lines = subtask.split("\n")
        font = self._cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        line_height = _VIDEO_HEADER_HEIGHT // len(lines)
        for line_idx, line in enumerate(lines):
            if not line:
                continue
            # Shrink the font until the text fits, since the subtask strings
            # vary in length. 0.5 fits ~50 chars at width 512.
            scale = 0.55
            while scale > 0.3:
                (text_w, text_h), _baseline = self._cv2.getTextSize(line, font, scale, thickness)
                if text_w <= width - 8:
                    break
                scale -= 0.05
            x = max(4, (width - text_w) // 2)
            y = line_idx * line_height + (line_height + text_h) // 2
            self._cv2.putText(header, line, (x, y), font, scale, (255, 255, 255), thickness, self._cv2.LINE_AA)
        return header

    def _render_value_plot(self, size: int, q_matrix: np.ndarray, steps: np.ndarray, current_step: int) -> np.ndarray:
        plt = self._plt
        dpi = 100
        figsize = (size / dpi, size / dpi)
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        for sample_idx in range(q_matrix.shape[1]):
            ax.plot(steps, q_matrix[:, sample_idx], linewidth = 0.8)
        # Ground-truth subtask boundaries: a vertical black line wherever the
        # simulator-derived subtask changed between consecutive infer calls.
        for frame_idx in range(1, len(self._gt_subtasks)):
            if self._gt_subtasks[frame_idx] != self._gt_subtasks[frame_idx - 1]:
                ax.axvline(x = int(steps[frame_idx]), color = "black", linewidth = 1.0)
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

    # Path to the eval episode spec JSON (see module docstring for the schema).
    # Must contain at least `num_trials` entries; episode i uses entry i.
    episodes_json: str = ""

    # Eval parameters
    num_trials: int = 50
    # Episode wall-clock budget in sim seconds (max_steps = control_freq * time_limit_s).
    time_limit_s: int = 400
    # Truncate the episode after this many env steps without a reward increase
    # (packing episodes are long; a stalled policy shouldn't burn the full budget).
    no_progress_timeout_steps: int = 14400
    # Must match the scene variant the training data was rendered with.
    scene_variant: str = "ycb_visual"
    # Max zero-action env steps after reset to let the arms converge to the
    # mocap home pose before the first policy observation (loop exits early
    # once the TCP stops moving).
    settle_steps: int = 120
    # If True, compute env deltas against the MEASURED TCP pose instead of the
    # mocap IK target (A/B knob; the dataset's actions are next-step mocap
    # targets, so mocap-relative is the training-consistent default).
    state_relative_deltas: bool = False

    # Logging
    log_dir: str | None = None
    # Multi-segment path under <log_dir>/<env> selecting the eval method.
    # Layouts derived by the launching bash script:
    #   policy-only: "<policy_config>/bc[_noise<level>]"
    #   best-of-N:   "<policy_config>/<critic_ft_config>_<critic_step>/best_of_n_<N>[_noise<level>]"
    # Empty => no method subdir (results go straight under <log_dir>/<env>).
    method: str = ""
    seed: int = 86
    log_videos: bool = False
    # If True, dump one JSON per episode under <log_path>/predictions/ holding
    # the per-infer-call critic values and subtask predictions returned by the
    # server (q_value(s), predicted_subtask, perplexity).
    save_predictions: bool = False

    # Resume support. Skip the first `start_episode_idx` indices in the
    # per-episode loop, preserving the deterministic per-index reset seed
    # (`seed + 100 * episode_idx`). If `resume_from_dir` is set, the client
    # writes into that existing run dir instead of a fresh one; prior
    # successes/failures are recovered by counting videos/*/episode_*.mp4.
    # Env cadence (env_control_freq / replan_steps) is derived from the
    # server's policy.metadata after connect — no CLI override.
    start_episode_idx: int = 0
    resume_from_dir: str | None = None


def _load_episodes(path: str) -> list[tuple[PackingGoalSpec, str, list[str] | None]]:
    """Parse the episodes JSON into (goal_spec, instruction, subtasks) triples.

    `subtasks` is the entry's ordered ground-truth subtask strings when the
    JSON provides them, else None (schedule is then derived from the goal).
    """
    with open(path) as f:
        entries = json.load(f)
    assert isinstance(entries, list), f"episodes JSON must be a top-level list, got {type(entries)}"

    episodes: list[tuple[PackingGoalSpec, str, list[str] | None]] = []
    for i, entry in enumerate(entries):
        if "goal_id" in entry:
            goal_id = entry["goal_id"]
            assert goal_id in GOAL_LOOKUP, f"entry {i}: unknown goal_id {goal_id!r}"
            goal = GOAL_LOOKUP[goal_id]
        elif "goal" in entry:
            goal = PackingGoalSpec(**entry["goal"])
        else:
            raise ValueError(f"entry {i} must have either 'goal_id' or 'goal': {entry}")
        instruction = entry.get("instruction") or goal.description
        episodes.append((goal, instruction, entry.get("subtasks")))
    return episodes


def _build_subtask_schedule(goal: PackingGoalSpec, subtask_texts: list[str] | None) -> list[dict]:
    """Ordered per-object subtasks: {"object", "tray", "kind", "text"}.

    When the episodes JSON provides `subtasks`, each string is parsed (kind
    from the leading verb, object by name match, tray from "the <name> tray")
    so the ground-truth completion check is tied to the exact string the
    policy is prompted with, in the JSON's order. Otherwise the schedule is
    derived from the goal in training-data phrasing: clutter removals first
    (they block tray space), then packs in `goal.objects` order.
    """
    all_objects = list(goal.objects) + list(goal.clutter_objects or [])
    if subtask_texts:
        # Longest-name-first so multi-word names win over any substring overlap.
        candidates = sorted(all_objects, key = lambda o: -len(o))
        schedule = []
        for text in subtask_texts:
            kind = "remove" if text.strip().lower().startswith("remove") else "pack"
            obj = next(o for o in candidates if o.replace("_", " ") in text)
            tray_match = re.search(r"the (small|medium|large) tray", text)
            assert tray_match is not None, f"no tray name in subtask {text!r}"
            schedule.append({
                "object": obj, "tray": tray_match.group(1), "kind": kind, "text": text,
            })
        return schedule

    schedule = []
    clutter_source_tray: dict[str, str] = {}
    for tray_name, items in (goal.initial_tray_contents or {}).items():
        for item in items:
            clutter_source_tray[item["object"]] = tray_name
    for obj in goal.clutter_objects or []:
        tray = clutter_source_tray[obj]
        schedule.append({
            "object": obj, "tray": tray, "kind": "remove",
            "text": f"Remove the {obj.replace('_', ' ')} from the {tray} tray.",
        })
    assignments = goal.assignments or {}
    for obj in goal.objects:
        tray = assignments[obj]
        schedule.append({
            "object": obj, "tray": tray, "kind": "pack",
            "text": f"Pack the {obj.replace('_', ' ')} into the {tray} tray.",
        })
    return schedule


def _latch_completed_subtasks(completed: set, events: list[dict]) -> None:
    """Monotonically latch subtask completion off the env's OWN event stream.

    The env's reward logic emits a "pack" event when an object becomes packed
    (stable in its ASSIGNED tray for the stability hold — the reward
    increment) and a "clutter_removed" event when a clutter object leaves a
    tray (the env's clutter-clear completion signal). Latching on these events
    reuses the env's completion scheme verbatim (no reimplemented geometry)
    and is monotonic by construction: the later "unpack" / "clutter_returned"
    events are deliberately ignored, so the walk never moves backwards even
    if the env's own (non-monotonic) status regresses.
    """
    for event in events:
        if event["type"] == "pack":
            completed.add((event["object"], "pack"))
        elif event["type"] == "clutter_removed":
            completed.add((event["object"], "remove"))


def _current_subtask(schedule: list[dict], completed: set) -> str:
    """First not-yet-latched subtask's text (the last one once all latched)."""
    for entry in schedule:
        if (entry["object"], entry["kind"]) not in completed:
            return entry["text"]
    return schedule[-1]["text"]


def _build_state14(obs: dict, env: PackingEnv) -> np.ndarray:
    """14-D EEF state: [xyz(3), rpy(3), grip, ...] per arm from the MOCAP
    IK-target pose — matching the sim_xarm_packing dataset exactly, whose
    state is the reconstructed commanded mocap target (openpi-test
    convert_xarm_sim_data_to_lerobot.py), NOT the measured TCP. rpy = scipy
    'xyz' euler of the mocap rotation (wxyz scalar-first quat); gripper is
    the raw setpoint `gripper_pos` (== ctrl/255) in [0, 1], no inversion.
    """
    data = env.data
    left_pos = np.asarray(data.mocap_pos[0], dtype = np.float32)
    right_pos = np.asarray(data.mocap_pos[1], dtype = np.float32)
    left_rpy = Rotation.from_quat(np.asarray(data.mocap_quat[0]), scalar_first = True).as_euler("xyz").astype(np.float32)
    right_rpy = Rotation.from_quat(np.asarray(data.mocap_quat[1]), scalar_first = True).as_euler("xyz").astype(np.float32)
    left_g = float(obs["state"]["left/gripper_pos"][0])
    right_g = float(obs["state"]["right/gripper_pos"][0])
    return np.concatenate([
        left_pos, left_rpy, [left_g],
        right_pos, right_rpy, [right_g],
    ]).astype(np.float32)


def _abs_action_to_env_delta(
    abs_action: np.ndarray, obs: dict, env: PackingEnv, *, state_relative: bool = False,
) -> np.ndarray:
    """Convert a single 14-D absolute action from the policy to env-space delta.

    Layout (matching the sim_xarm_packing dataset, action[t] == state[t+1]):
      abs_action[0:3]   = left target xyz (absolute, base frame)
      abs_action[3:6]   = left target rpy (scipy 'xyz' euler)
      abs_action[6]     = left absolute gripper position in [0, 1]
      abs_action[7:10]  = right target xyz
      abs_action[10:13] = right target rpy
      abs_action[13]    = right absolute gripper position

    Mirrors the sim_bimanual_assembly client: PackingEnv.step integrates the
    pos/rot deltas against the *mocap IK target*, not the observed TCP
    (new_pos = mocap_pos + delta; new_quat = R(delta) * R(mocap) left-multiply),
    so the deltas invert against the current mocap reference (i=0 left, i=1
    right): pos_delta = abs - mocap_pos, rpy_delta = (R_abs * R_mocap^-1).
    Read just before env.step, so the mocap arrays still hold the target the
    previous step/reset set. mocap_quat is MuJoCo scalar-first (wxyz).

    Gripper has no mocap: env reads `ctrl/255` (== obs `*/gripper_pos`) and
    applies `new_g = clip(cur_g + 0.04 * a, 0, 1)`, so `a = (abs - cur_g) / 0.04`,
    clipped to the env action_space range [-1, 1] (the per-step rate the
    teleop data was collected under).

    With `state_relative=True` the pos/rot reference is instead the MEASURED
    TCP pose from the observation (A/B experiment knob).
    """
    if state_relative:
        left_pose = np.asarray(obs["state"]["left/tcp_pose"], dtype = np.float32)
        right_pose = np.asarray(obs["state"]["right/tcp_pose"], dtype = np.float32)
        left_mocap_pos = left_pose[:3]
        right_mocap_pos = right_pose[:3]
        # env obs quat is scalar-LAST (xyzw).
        r_mocap_l = Rotation.from_quat(left_pose[3:7])
        r_mocap_r = Rotation.from_quat(right_pose[3:7])
    else:
        data = env.data
        left_mocap_pos = np.asarray(data.mocap_pos[0], dtype = np.float32)
        right_mocap_pos = np.asarray(data.mocap_pos[1], dtype = np.float32)
        r_mocap_l = Rotation.from_quat(np.asarray(data.mocap_quat[0]), scalar_first = True)
        r_mocap_r = Rotation.from_quat(np.asarray(data.mocap_quat[1]), scalar_first = True)
    cur_lg = float(obs["state"]["left/gripper_pos"][0])
    cur_rg = float(obs["state"]["right/gripper_pos"][0])

    lpos_delta = abs_action[0:3].astype(np.float32) - left_mocap_pos
    r_abs_l = Rotation.from_euler("xyz", abs_action[3:6])
    lrpy_delta = (r_abs_l * r_mocap_l.inv()).as_euler("xyz").astype(np.float32)
    lgrip_env = np.float32(np.clip((float(abs_action[6]) - cur_lg) / _GRIPPER_DELTA_SCALE, -1.0, 1.0))

    rpos_delta = abs_action[7:10].astype(np.float32) - right_mocap_pos
    r_abs_r = Rotation.from_euler("xyz", abs_action[10:13])
    rrpy_delta = (r_abs_r * r_mocap_r.inv()).as_euler("xyz").astype(np.float32)
    rgrip_env = np.float32(np.clip((float(abs_action[13]) - cur_rg) / _GRIPPER_DELTA_SCALE, -1.0, 1.0))

    return np.concatenate([
        lpos_delta, lrpy_delta, [lgrip_env],
        rpos_delta, rrpy_delta, [rgrip_env],
    ]).astype(np.float32)


def eval_main(args: Args) -> None:
    np.random.seed(args.seed)
    assert args.episodes_json, "--episodes-json is required"
    episodes = _load_episodes(args.episodes_json)
    assert len(episodes) >= args.num_trials, (
        f"episodes JSON has {len(episodes)} entries but --num-trials={args.num_trials}; "
        f"provide at least num_trials entries"
    )

    env_name = "Packing"
    log_path = None
    file_handler: logging.FileHandler | None = None
    resuming = args.resume_from_dir is not None
    if resuming:
        log_path = pathlib.Path(args.resume_from_dir)
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
            f"Resuming eval in {log_path} from start_episode_idx={args.start_episode_idx}"
        )
    elif args.log_dir is not None:
        log_path = pathlib.Path(args.log_dir) / env_name
        if args.method:
            log_path = log_path / args.method

        # Refuse to overwrite a prior run's outputs. To resume an interrupted
        # eval, set --resume-from-dir + --start-episode-idx instead.
        if log_path.exists():
            for marker in ("stats.json", "eval.log", "videos"):
                if (log_path / marker).exists():
                    raise FileExistsError(
                        f"Eval output already exists at {log_path} ({marker} present). "
                        "Delete it or use --resume-from-dir to continue."
                    )

        log_path.mkdir(parents = True, exist_ok = True)

        file_handler = logging.FileHandler(log_path / "eval.log", mode = "w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(file_handler)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    image_helper = _eval_image_helper.EvalImageHelper.from_client(client)
    logging.info(
        f"EvalImageHelper: policy={image_helper.policy_image_size}, "
        f"critic={image_helper.critic_image_size}, "
        f"expect_critic_images={image_helper.expect_critic_images}"
    )

    # Single source of truth for env cadence: the server's policy.metadata
    # exposes `policy_subsample` (True iff the policy was trained at half
    # cadence). Subsample → 30 Hz / replan 15, else 60 Hz / replan 30.
    server_meta = client.get_server_metadata() or {}
    policy_subsample = bool(server_meta.get("policy_subsample", False))
    if policy_subsample:
        env_control_freq = 30
        replan_steps = 15
    else:
        env_control_freq = 60
        replan_steps = 30
    max_steps = env_control_freq * args.time_limit_s
    logging.info(
        f"Eval cadence (from server policy_subsample={policy_subsample}): "
        f"env_control_freq={env_control_freq} Hz, replan_steps={replan_steps}, "
        f"max_steps={max_steps}"
    )

    env = PackingEnv(
        control_freq = env_control_freq,
        time_limit = args.time_limit_s,
        render_mode = "rgb_array",
        image_obs = True,
        scene_variant = args.scene_variant,
    )

    video_logger = None
    if args.log_videos and log_path is not None:
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
    packed_ratios: list[float] = []
    # Subtask-level success: for these goals each instruction clause is one
    # object into its assigned tray, and env reward counts assignment-
    # respecting placements only — so subtasks completed == max_reward.
    total_subtasks_completed = 0
    total_subtasks = 0
    per_goal_stats: dict[str, dict[str, int]] = {}
    for episode_idx in tqdm.tqdm(
        range(args.start_episode_idx, args.num_trials),
        desc = env_name, initial = args.start_episode_idx, total = args.num_trials,
    ):
        goal, instruction, subtask_texts = episodes[episode_idx]
        n_objects = max(1, len(goal.objects))
        obs, info = env.reset(seed = args.seed + 100 * episode_idx, options = {"goal": copy.deepcopy(goal)})
        logging.info(
            f"Episode {episode_idx}: goal={goal.goal_id}, instruction={instruction!r}"
        )

        # PackingEnv.reset leaves the arms ~20 cm below the mocap home target
        # (the reset IK only runs 5 steps); the dataset's episode-start state is
        # the settled home pose. Step zero-delta actions until the TCP stops
        # moving so the first policy observation matches training.
        # Ground-truth subtask conditioning: the policy prompt is the first
        # not-yet-completed subtask, latched monotonically off env events
        # (harvested after EVERY env.step — including settling — so pack
        # events from pre-placed objects are captured). The critic still
        # AR-decodes its own subtask for the value prompt (server-side); the
        # client's task_description feeds that.
        subtask_schedule = _build_subtask_schedule(goal, subtask_texts)
        completed_subtasks: set = set()
        logging.info(f"Subtask schedule: {[s['text'] for s in subtask_schedule]}")

        # Settle convergence is checked on the MEASURED TCP (the mocap-derived
        # policy state is constant under zero actions, so it can't signal
        # convergence); settling still matters so the arms/images match the
        # training frames before the first policy observation.
        def _measured_tcp_xyz(current_obs: dict) -> np.ndarray:
            return np.concatenate([
                np.asarray(current_obs["state"]["left/tcp_pose"][:3], dtype = np.float32),
                np.asarray(current_obs["state"]["right/tcp_pose"][:3], dtype = np.float32),
            ])

        prev_tcp = _measured_tcp_xyz(obs)
        for _settle_step in range(args.settle_steps):
            obs, _, _, _, info = env.step(np.zeros(14, dtype = np.float32))
            _latch_completed_subtasks(completed_subtasks, env.get_events())
            settled_tcp = _measured_tcp_xyz(obs)
            if np.abs(settled_tcp - prev_tcp).max() < 1e-5:
                break
            prev_tcp = settled_tcp
        logging.info(f"Settled to home in {_settle_step + 1} steps")

        action_plan = collections.deque()
        done = False
        max_reward = 0.0
        last_progress_reward = 0.0
        last_progress_step = 0
        ep_prediction_records: list[dict] = []

        def _make_element(current_obs: dict, prompt: str, task_instruction: str = instruction) -> dict:
            element_images = image_helper.process_images({
                "base_0_rgb": np.ascontiguousarray(current_obs["images"]["right/top"]),
                "left_wrist_0_rgb": np.ascontiguousarray(current_obs["images"]["left/wrist"]),
                "right_wrist_0_rgb": np.ascontiguousarray(current_obs["images"]["right/wrist"]),
            })
            return {
                **element_images,
                "state": _build_state14(current_obs, env),
                # Policy conditioning: the client-computed ground-truth subtask.
                "prompt": prompt,
                # Per-call task_description: routed into the critic prompt and
                # the AR subtask decoder (critic prompt_mode =
                # "task_description_predict_current_subtask").
                "task_description": task_instruction,
            }

        if isinstance(video_logger, VideoLogger):
            video_logger.start_episode(episode_idx)

        for t in range(max_steps):
            if not action_plan:
                policy_prompt = _current_subtask(subtask_schedule, completed_subtasks)
                element = _make_element(obs, policy_prompt)

                infer_t0 = time.perf_counter()
                infer_result = client.infer(element)
                roundtrip_ms = (time.perf_counter() - infer_t0) * 1000.0
                action_chunk = np.asarray(infer_result["actions"], dtype = np.float32)
                q_values = infer_result.get("q_values")

                # Video header: line 1 = the subtask the critic's value prompt
                # conditioned on this call (AR-decoded), line 2 = the
                # simulator-ground-truth subtask the policy was prompted with.
                header_subtask = (
                    f"critic: {infer_result.get('critic_subtask') or '(none)'}\n"
                    f"gt: {policy_prompt}"
                )

                if args.save_predictions:
                    ep_prediction_records.append({
                        "t": int(t),
                        "prompt": element["prompt"],
                        "q_value": (float(infer_result["q_value"]) if "q_value" in infer_result else None),
                        "q_values": (
                            np.asarray(q_values, dtype = np.float32).reshape(-1).tolist()
                            if q_values is not None else None
                        ),
                        "predicted_subtask": infer_result.get("predicted_subtask"),
                        "subtask_perplexity": (
                            float(infer_result["subtask_perplexity"])
                            if "subtask_perplexity" in infer_result else None
                        ),
                    })

                server_ms = infer_result.get("policy_timing", {}).get("infer_ms")
                server_str = f"{server_ms:.1f}ms" if server_ms is not None else "n/a"
                q_str = ""
                if q_values is not None:
                    q_arr = np.asarray(q_values).reshape(-1)
                    q_str = ", q_values=[" + ", ".join(f"{v:.4f}" for v in q_arr.tolist()) + "]"
                logging.info(
                    f"[t={t}] prompt={element['prompt']!r} infer: server={server_str}, "
                    f"round-trip={roundtrip_ms:.1f}ms{q_str}"
                )
                assert len(action_chunk) >= replan_steps, (
                    f"Policy predicts {len(action_chunk)} steps but replan_steps={replan_steps}"
                )
                action_plan.extend(action_chunk[:replan_steps])

                if video_logger == "pending":
                    q_arr = np.asarray(q_values).reshape(-1) if q_values is not None else None
                    video_logger = VideoLogger(
                        output_dir = str(log_path / "videos"),
                        fps = max(1.0, float(env_control_freq) / replan_steps),
                        num_samples = int(q_arr.shape[0]) if q_arr is not None else 0,
                    )
                    video_logger.start_episode(episode_idx)
                if isinstance(video_logger, VideoLogger):
                    video_logger.record_predict(
                        images = {
                            "right/top": np.ascontiguousarray(obs["images"]["right/top"]),
                            "left/wrist": np.ascontiguousarray(obs["images"]["left/wrist"]),
                            "right/wrist": np.ascontiguousarray(obs["images"]["right/wrist"]),
                        },
                        q_values = (np.asarray(q_values).reshape(-1) if q_values is not None else None),
                        t = t,
                        subtask = header_subtask,
                        gt_subtask = policy_prompt,
                    )

            abs_action = action_plan.popleft()
            env_action = _abs_action_to_env_delta(
                abs_action, obs, env, state_relative = args.state_relative_deltas,
            )
            obs, reward, done, truncated, info = env.step(env_action)
            _latch_completed_subtasks(completed_subtasks, env.get_events())
            reward = float(reward)
            max_reward = max(max_reward, reward)
            if reward > last_progress_reward:
                last_progress_reward = reward
                last_progress_step = t
            if done:
                total_successes += 1
                break
            if truncated:
                break
            if t - last_progress_step > args.no_progress_timeout_steps:
                logging.info(
                    f"Episode {episode_idx}: no reward progress for "
                    f"{args.no_progress_timeout_steps} steps at t={t}, truncating"
                )
                break

        total_episodes += 1
        packed_ratio = max_reward / n_objects
        packed_ratios.append(packed_ratio)
        total_subtasks_completed += int(max_reward)
        total_subtasks += n_objects
        goal_stats = per_goal_stats.setdefault(goal.goal_id, {"episodes": 0, "successes": 0})
        goal_stats["episodes"] += 1
        goal_stats["successes"] += int(done)

        if isinstance(video_logger, VideoLogger):
            video_logger.finish_episode(success = bool(done))

        if args.save_predictions and log_path is not None:
            pred_dir = log_path / "predictions"
            pred_dir.mkdir(parents = True, exist_ok = True)
            pred_path = pred_dir / f"episode_{episode_idx}.json"
            with open(pred_path, "w") as f:
                json.dump(
                    {
                        "episode_idx": int(episode_idx),
                        "goal_id": goal.goal_id,
                        "instruction": instruction,
                        "success": bool(done),
                        "max_reward": float(max_reward),
                        "packed_ratio": float(packed_ratio),
                        "records": ep_prediction_records,
                    },
                    f,
                    indent = 4,
                )
            logging.info(f"Saved {len(ep_prediction_records)} prediction records: {pred_path}")

        logging.info(
            f"Episode {episode_idx} ({goal.goal_id}): {'success' if done else 'failure'}, "
            f"packed {int(max_reward)}/{n_objects}"
        )
        logging.info(
            f"Running: {total_successes}/{total_episodes} "
            f"({total_successes / total_episodes * 100:.1f}%), "
            f"subtasks: {total_subtasks_completed}/{total_subtasks} "
            f"({total_subtasks_completed / total_subtasks * 100:.1f}%)"
        )

    logging.info(
        f"[{env_name}] Final: {total_successes}/{total_episodes} "
        f"({total_successes / total_episodes * 100:.1f}%), "
        f"mean packed_ratio={np.mean(packed_ratios):.3f}"
    )
    for goal_id, stats in sorted(per_goal_stats.items()):
        logging.info(
            f"[{env_name}] {goal_id}: {stats['successes']}/{stats['episodes']}"
        )

    if log_path is not None:
        with open(log_path / "stats.json", "w") as f:
            json.dump(
                {
                    "num_episodes": total_episodes,
                    "success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
                    "mean_packed_ratio": float(np.mean(packed_ratios)) if packed_ratios else 0.0,
                    "subtask_success_rate": (
                        total_subtasks_completed / total_subtasks if total_subtasks > 0 else 0.0
                    ),
                    "num_subtasks_completed": total_subtasks_completed,
                    "num_subtasks": total_subtasks,
                    "per_goal": per_goal_stats,
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
