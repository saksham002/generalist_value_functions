#!/usr/bin/env python3
"""Evaluate pi0-based policies with clean environment-policy decoupling via adapters."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
import dataclasses
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Annotated

# Additional imports for video recording
import cv2
import numpy as np
from tqdm import tqdm
import tyro
import wandb

from openpi.policies import policy_config as _policy_config
from openpi.training import config_shirt_hang as _config

# MuJoCo version check
try:
    import mujoco

    MUJOCO_VERSION = getattr(mujoco, "__version__", "unknown")
except ImportError:
    MUJOCO_VERSION = "not_installed"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
possible_rac_roots: list[Path] = []
if env_path := os.environ.get("RAC_PATH"):
    possible_rac_roots.append(Path(env_path))
possible_rac_roots.append(PROJECT_ROOT.parent / "RAC")
possible_rac_roots.append(Path.cwd().parent / "RAC")

RAC_ROOT: Path | None = None
for candidate_root in possible_rac_roots:
    if candidate_root.exists():
        RAC_ROOT = candidate_root
        for candidate in (candidate_root, candidate_root / "dual_xarms"):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        break

try:
    from dual_xarms_sim.relative_frame import RelativeFrame
    from dual_xarms_sim.relative_frame import WristRelativeTo
    from dual_xarms_sim.rmp_env import RMPDualXArmsEnv
    from dual_xarms_sim.utils.network import get_oculus_reading

    try:
        from flow_match.bc_flowmatch.env_wrappers.frame_stack_wrapper import FrameStackWrapperEnv
    except ImportError:
        FrameStackWrapperEnv = None
    try:
        from dual_xarms_sim.oculus_intervention import OculusIntervention

        OCULUS_AVAILABLE = True
    except ImportError:
        OCULUS_AVAILABLE = False
except ImportError as exc:
    raise ImportError("Could not import RMPDualXArmsEnv. Check RAC_PATH.") from exc


# ============================================================================
# VIDEO RECORDING UTILITIES
# ============================================================================


def video_writer_loop(frame_queue: queue.Queue, filename: str, fps: int, video_wh: tuple[int, int], image_crop_fn=None):
    """Thread-safe video writer loop that consumes frames from a queue.

    Args:
        frame_queue: Queue containing (frames_list, text_overlay) tuples
        filename: Output video file path
        fps: Frames per second
        video_wh: Video dimensions (width, height)
        image_crop_fn: Optional function to crop/resize frames
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, video_wh)

    while True:
        item = frame_queue.get()
        if item is None:  # Sentinel signals end-of-stream
            break

        frames, text = item

        # Stack frames horizontally (multi-camera view)
        # frames is list of [H, W, 3] arrays in RGB format
        frame_array = np.stack(frames, axis=0)  # (K, H, W, 3)

        # Apply crop/resize if provided
        if image_crop_fn is not None:
            # Note: OpenPI uses channels-last (H, W, C) format for images
            # No need to convert for JAX-based policies
            processed_frames = []
            for f in frame_array:
                # Convert RGB to BGR for OpenCV
                f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                processed_frames.append(f_bgr)
            frame_array = np.stack(processed_frames, axis=0)
        else:
            # Convert all frames from RGB to BGR for OpenCV
            frame_array = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frame_array])

        # Concatenate horizontally: (H, K*W, 3)
        frame = np.concatenate(frame_array, axis=1)

        # Overlay text
        position = (10, video_wh[1] - 10)
        frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    logging.info(f"Video writer thread finished writing to {filename}")


# ============================================================================
# ADAPTER INTERFACES - Clean decoupling between env and policy
# ============================================================================


class ObservationAdapter(ABC):
    """Adapter interface: Environment observation → Policy input format."""

    @abstractmethod
    def env_to_policy(self, env_obs: dict) -> dict:
        """Convert environment observation to policy input format."""


class ActionAdapter(ABC):
    """Adapter interface: Policy output → Environment action format."""

    @abstractmethod
    def policy_to_env(self, policy_actions: np.ndarray, config: dict) -> np.ndarray:
        """Convert policy action output to environment action format.

        Args:
            policy_actions: Raw policy output (may be padded)
            config: Action configuration (e.g., real_action_start, real_action_dim)

        Returns:
            Environment-ready action vector
        """


# ============================================================================
# CONCRETE ADAPTERS - Specific to shirt_hang_rac task
# ============================================================================


class ShirtHangObservationAdapter(ObservationAdapter):
    """Adapter for RMPDualXArmsEnv observations → Pi0 policy format."""

    CAMERA_MAP = {
        "right/top": "base_0_rgb",
        "left/wrist": "left_wrist_0_rgb",
        "right/wrist": "right_wrist_0_rgb",
    }

    def __init__(self, camera_names: tuple[str, ...]):
        self.camera_names = camera_names

    def env_to_policy(self, env_obs: dict) -> dict:
        """Convert RMPDualXArmsEnv observation to Pi0 policy input.

        OpenPI JAX format expects:
        - image: dict with keys "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
        - Each image: np.ndarray with shape (H, W, 3), uint8 [0, 255], channels-last
        - state: np.ndarray with shape (state_dim,)
        """
        images = self._extract_images(env_obs)

        # Map user-friendly names to OpenPI canonical names
        # OpenPI expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        policy_images = {}
        policy_masks = {}

        for user_cam_name in self.camera_names:
            canonical_name = self.CAMERA_MAP.get(user_cam_name, user_cam_name)

            if user_cam_name in images:
                img = images[user_cam_name]

                # Validate
                if not isinstance(img, np.ndarray):
                    raise TypeError(f"Image for camera '{user_cam_name}' is {type(img)}, expected np.ndarray")
                if img.ndim != 3 or img.shape[-1] != 3:
                    raise ValueError(f"Image for camera '{user_cam_name}' has shape {img.shape}, expected (H, W, 3)")

                policy_images[canonical_name] = img
                policy_masks[canonical_name] = np.array(True, dtype=bool)
            else:
                logging.warning(f"Camera '{user_cam_name}' not found, will be masked out")
                # Create dummy image with correct shape
                policy_images[canonical_name] = np.zeros((224, 224, 3), dtype=np.uint8)
                policy_masks[canonical_name] = np.array(False, dtype=bool)

        return {
            "image": policy_images,
            "image_mask": policy_masks,
            "state": self._extract_state(env_obs),
        }

    def _extract_state(self, obs: dict) -> np.ndarray:
        """Extract state vector (40-d) from observation.

        Returns 1D array with shape (40,) for JAX/OpenPI format.
        """

        def _last(array: np.ndarray) -> np.ndarray:
            return array[-1] if array.ndim > 1 else array

        state_dict = obs["state"]

        def _extract(key: str, fallback_dim: int) -> np.ndarray:
            value = state_dict.get(key)
            if value is None:
                return np.zeros(fallback_dim, dtype=np.float32)
            arr = _last(value)
            return arr.astype(np.float32)

        parts = [
            _extract("left/relative2_tcp_pose", 7),
            _extract("left/relative2_tcp_vel", 6),
            _extract("left/wrist_tcp_vel", 6),
            _extract("left/gripper_pos", 1),
            _extract("right/relative2_tcp_pose", 7),
            _extract("right/relative2_tcp_vel", 6),
            _extract("right/wrist_tcp_vel", 6),
            _extract("right/gripper_pos", 1),
        ]
        state = np.concatenate(parts, axis=-1)

        # Ensure 1D array
        if state.ndim > 1:
            state = state.flatten()

        return state

    def _extract_images(self, obs: dict) -> dict[str, np.ndarray]:
        """Extract and convert images: BGR (env) → RGB (policy).

        Training: image_bgr=True → loader converts BGR→RGB → model trained on RGB
        Environment: Returns BGR (real cameras)
        Conversion: Must convert BGR→RGB to match training
        """
        import cv2

        # Debug: print available cameras
        available_cameras = list(obs.get("images", {}).keys())
        logging.info(f"Available cameras in observation: {available_cameras}")

        image_dict = {}
        for cam_name in self.camera_names:
            cam_key = self.CAMERA_MAP.get(cam_name, cam_name)

            # Try to find the camera in the observation
            found_key = None
            if cam_key in obs["images"]:
                found_key = cam_key
            elif cam_name in obs["images"]:
                found_key = cam_name

            if found_key is None:
                logging.warning(
                    f"Camera {cam_name} (tried keys: {cam_key}, {cam_name}) not found in {available_cameras}"
                )
                continue

            cam_frames = obs["images"][found_key]

            # Handle frame stacking: take last frame
            if cam_frames.ndim == 4:
                cam_frame = cam_frames[-1]  # Shape: (H, W, C)
            else:
                cam_frame = cam_frames  # Shape: (H, W, C)

            # Convert BGR to RGB
            if cam_frame.shape[-1] == 3:
                cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)

            # Ensure uint8 [0, 255]
            if cam_frame.dtype != np.uint8:
                cam_frame = np.clip(cam_frame * 255, 0, 255).astype(np.uint8)

            logging.debug(f"Extracted image for {cam_name}: shape={cam_frame.shape}, dtype={cam_frame.dtype}")
            image_dict[cam_name] = cam_frame

        if not image_dict:
            raise RuntimeError(
                f"No cameras found! Expected {self.camera_names} but got {available_cameras}. "
                f"Please check your robot server camera configuration."
            )

        return image_dict


class ShirtHangActionAdapter(ActionAdapter):
    """Adapter for Pi0 actions → RMPDualXArmsEnv format."""

    def policy_to_env(self, policy_actions: np.ndarray, config: dict) -> np.ndarray:
        """Extract real actions from padded policy output.

        Pi0 output: 40-d actions (14 real at indices 14-27, rest are padding)
        Environment expects: 14-d actions (end-effector commands)
        """
        real_action_start = config["real_action_start"]
        real_action_dim = config["real_action_dim"]

        # Validate action shape
        if policy_actions.ndim != 2:
            raise ValueError(f"Expected (T, A) actions, got {policy_actions.shape}")

        plan_T, plan_A = policy_actions.shape
        required = real_action_start + real_action_dim

        if plan_A < required:
            raise ValueError(
                f"Action dim {plan_A} < required {required} (start={real_action_start}, dim={real_action_dim})"
            )

        # Extract real actions
        real_actions = policy_actions[:, real_action_start : real_action_start + real_action_dim]
        return real_actions


class AdaptedPolicy:
    """Policy wrapper with observation/action adapters for clean decoupling."""

    def __init__(
        self,
        policy,
        obs_adapter: ObservationAdapter,
        action_adapter: ActionAdapter,
        action_config: dict,
    ):
        self.policy = policy
        self.obs_adapter = obs_adapter
        self.action_adapter = action_adapter
        self.action_config = action_config

    def predict(self, env_obs: dict) -> np.ndarray:
        """End-to-end prediction: env obs → policy → env actions.

        Args:
            env_obs: Raw environment observation

        Returns:
            Environment-ready action sequence (T, action_dim)
        """
        # Env obs → Policy input
        policy_input = self.obs_adapter.env_to_policy(env_obs)

        # Policy inference
        policy_output = self.policy.infer(policy_input)
        policy_actions = np.asarray(policy_output["actions"]).astype(np.float32)

        # Policy actions → Env actions
        env_actions = self.action_adapter.policy_to_env(policy_actions, self.action_config)

        return env_actions


# ============================================================================
# EVALUATION LOGIC - Environment-agnostic
# ============================================================================


@dataclasses.dataclass
class EvalArgs:
    config: Annotated[str, tyro.conf.Positional] = "pi0_shirt_hang_rac_lora"
    checkpoint_root: str = "checkpoints/pi0_shirt_hang_rac_lora"
    steps: list[int] | None = None
    num_episodes: int = 100
    query_freq: int = 30
    horizon: int = 60
    control_hz: int = 60
    time_limit_s: float | None = None
    max_env_steps: int = 7200
    video_dir: str = "videos"
    metrics_dir: str = "eval_metrics"
    video_fps: int = 60
    camera_names: tuple[str, ...] = ("right/top", "left/wrist", "right/wrist")
    seed: int = 0
    enable_intervention: bool = True
    forced_replan_after_intervention: bool = True
    overlay_heatmap: str | None = None
    log_wandb: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    trace_replans: bool = True
    trace_steps: bool = False
    dump_episode_jsonl: bool = True

    record: bool = False  # If True, prompt to save episode data as npz after each episode
    save_data_dir: str = "data/openpi_shirt_hang_correction"  # Directory to save npz files

    # Manual labeling (shirt hanging task evaluation)
    manual_labeling: bool = True  # Prompt user to rate each episode (0-5 scale)
    auto_continue: bool = False  # If True, don't pause between episodes

    # Wait for Oculus button press to start episodes (like real_hang_intervention.py)
    wait_for_button: bool = True  # If True, wait for Oculus A button to start each episode

    # Remote robot execution
    remote_robot: bool = False  # Changed default to False for local simulation
    robot_host: str = "localhost"
    robot_port: int = 8080
    robot_timeout: float = 30.0


def rollout_episode(
    adapted_policy: AdaptedPolicy,
    env,
    args: EvalArgs,
    episode_idx: int,
    checkpoint_step: int,
    step_progress_bar: tqdm,
) -> tuple[float, float, dict[str, Path], dict, list]:
    """Environment-agnostic rollout using adapted policy.

    Returns:
        total_reward: Total reward accumulated
        max_reward: Maximum reward seen
        video_paths: Dictionary of video file paths
        episode_metrics: Dictionary of episode statistics
        recorded_data: List of recorded data (obses, actions, rews, dones, truncateds, infos) for saving
    """
    # For remote robot, video recording happens on server
    is_remote = args.remote_robot

    # Camera name mapping for remote robot (server uses canonical names)
    camera_name_map = (
        {
            "right/top": "base_0_rgb",
            "left/wrist": "left_wrist_0_rgb",
            "right/wrist": "right_wrist_0_rgb",
        }
        if is_remote
        else {}
    )

    # Reset environment
    obs, info = env.reset(seed=args.seed + episode_idx)
    query_freq = max(1, args.query_freq)
    step_progress_bar.reset()

    # Tracking variables
    total_reward = 0.0
    max_reward = 0.0
    action_norms: list[float] = []
    replan_steps: list[int] = []
    human_steps = 0
    prev_intervention_active = False
    intervened = False
    forced_query = False

    # Data recording (for npz saving)
    obses, actions, rews, dones, truncateds, infos = [], [], [], [], [], []
    intervention_step_counter = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    # Setup video recording (threaded)
    frame_queue = None
    writer_thread = None
    video_paths = {}

    if not is_remote:
        frame_queue = queue.Queue(maxsize=500)
        uuid = time.strftime("%Y%m%d-%H%M%S")
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        file_name = video_dir / f"{args.config}_{uuid}_ep{episode_idx}.mp4"
        writer_thread = threading.Thread(
            target=video_writer_loop,
            args=(
                frame_queue,
                str(file_name),
                args.video_fps,
                (224 * len(args.camera_names), 224),
                None,  # image_crop_fn
            ),
            daemon=True,
        )
        writer_thread.start()
        video_paths["multiview"] = file_name

    # Initialize action_plan with zeros to avoid None access
    action_plan = np.zeros((query_freq, 14), dtype=np.float32)
    t = 0
    terminated = False
    truncated = False

    # Initial frame overlay
    human_percentage = 0.0
    texts_overlay = f"human intervention: {human_steps}/{t} ({human_percentage:.1f}%)"
    if frame_queue is not None:
        # Extract frames from observation (env returns BGR, convert to RGB for video)
        frames_to_record = []
        for cam_name in args.camera_names:
            # Map to canonical name if using remote robot
            actual_cam_name = camera_name_map.get(cam_name, cam_name)
            if actual_cam_name in obs["images"]:
                cam_frames = obs["images"][actual_cam_name]
                cam_frame = cam_frames[-1] if cam_frames.ndim == 4 else cam_frames
                # Convert BGR to RGB for video writer (it will convert back to BGR for OpenCV)
                if cam_frame.shape[-1] == 3:
                    cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                frames_to_record.append(cam_frame)
        if frames_to_record:
            frame_queue.put((frames_to_record, texts_overlay))

    while not (terminated or truncated):
        # Replan: scheduled or forced
        if t % query_freq == 0 or forced_query:
            action_plan = adapted_policy.predict(obs)
            replan_steps.append(t)

            if args.trace_replans:
                marker = " [FORCED]" if forced_query else ""
                logging.info(f"Episode {episode_idx} step {t}: Replan{marker}, plan shape={action_plan.shape}")

            forced_query = False

        # Extract action from plan
        plan_idx = min(t % query_freq, action_plan.shape[0] - 1)
        action = action_plan[plan_idx]

        # Debug: Log action details periodically
        if t % 60 == 0:  # Every second at 60Hz
            logging.info(
                f"Episode {episode_idx} step {t}: action shape={action.shape}, "
                f"left_arm={action[:7]}, right_arm={action[7:14]}, "
                f"norm={np.linalg.norm(action):.6f}"
            )

        action_norms.append(float(np.linalg.norm(action)))

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        max_reward = max(max_reward, reward)

        # Intervention handling
        intervention_active = "intervene_action" in info
        if intervention_active != prev_intervention_active:
            prev_intervention_active = intervention_active
            if hasattr(env.unwrapped, "_displayer") and env.unwrapped._displayer:
                env.unwrapped._displayer.set_overlay_visible(intervention_active)

        if intervention_active:
            action = info["intervene_action"]
            intervened = True
            intervention_step_counter += 1
            human_steps += 1
        else:
            if intervened and args.forced_replan_after_intervention:
                forced_query = True
            intervened = False

        # Record data (for npz saving like real_hang_intervention.py)
        if args.record:
            obs_copy = deepcopy(obs)
            # Encode images as JPEG
            for name, img in obs_copy["images"].items():
                single_frame = img[-1] if img.ndim == 4 else img  # Get latest frame
                # Images from env are already in BGR format, no conversion needed for JPEG encoding
                result, encoded_image = cv2.imencode(".jpg", single_frame, encode_param)
                obs_copy["images"][name] = encoded_image
            # Extract latest state
            for name, state in obs_copy["state"].items():
                obs_copy["state"][name] = state[-1] if state.ndim > 1 else state

            obses.append(obs_copy)
            actions.append(action)
            rews.append(reward)
            dones.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        t += 1
        step_progress_bar.update(1)

        # Update video overlay
        human_percentage = (human_steps / t * 100) if t > 0 else 0.0
        texts_overlay = f"human intervention: {human_steps}/{t} ({human_percentage:.1f}%)"

        # Add frame to video
        if frame_queue is not None:
            frames_to_record = []
            for cam_name in args.camera_names:
                # Map to canonical name if using remote robot
                actual_cam_name = camera_name_map.get(cam_name, cam_name)
                if actual_cam_name in obs["images"]:
                    cam_frames = obs["images"][actual_cam_name]
                    cam_frame = cam_frames[-1] if cam_frames.ndim == 4 else cam_frames
                    # Convert BGR to RGB for video writer (it will convert back to BGR for OpenCV)
                    if cam_frame.shape[-1] == 3:
                        cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                    frames_to_record.append(cam_frame)
            if frames_to_record:
                frame_queue.put((frames_to_record, texts_overlay))

        if args.trace_steps:
            marker = " [INTERVENTION]" if intervention_active else ""
            logging.info(f"Episode {episode_idx} step {t}: reward={reward:.4f}, total={total_reward:.4f}{marker}")

        if terminated:
            step_progress_bar.set_description("Task Completed")

        if truncated:
            step_progress_bar.set_description("Task Truncated")

        if t >= args.max_env_steps:
            break

    # Stop video recording
    if frame_queue is not None:
        frame_queue.put(None)  # Signal end
        writer_thread.join()

    # Metrics
    episode_metrics = {
        "total_reward": total_reward,
        "max_reward": max_reward,
        "episode_length": t,
        "terminated": terminated,
        "truncated": truncated,
        "replan_count": len(replan_steps),
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "human_steps": human_steps,
        "human_percentage": human_percentage,
        "intervention_step_counter": intervention_step_counter,
        "manual_score": None,
    }

    recorded_data = [obses, actions, rews, dones, truncateds, infos] if args.record else []

    return total_reward, max_reward, video_paths, episode_metrics, recorded_data


def evaluate_checkpoint(
    args: EvalArgs,
    checkpoint_dir: Path,
    config: _config.TrainConfig,
    run: wandb.sdk.wandb_run.Run | None = None,
) -> None:
    """Evaluate checkpoint with decoupled policy-environment interface."""
    logging.info(f"Evaluating checkpoint: {checkpoint_dir}")

    # Load norm stats from checkpoint to avoid loading training datasets
    # This is critical for real robot evaluation where dataset files may not be accessible
    from openpi.training import checkpoints as _checkpoints

    # Create a minimal data config just to get asset_id without loading datasets
    # We'll pass norm_stats directly to bypass dataset loading
    try:
        # Try to load norm stats from checkpoint
        # Use repo_id for XArmDataConfig, asset_id for other configs
        asset_id = getattr(config.data, "asset_id", None) or getattr(config.data, "repo_id", None)
        if asset_id is None:
            raise ValueError("config.data must have either 'asset_id' or 'repo_id' attribute")

        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", asset_id)
        logging.info(
            f"Loaded normalization stats from checkpoint using asset_id='{asset_id}' (no dataset access needed)"
        )
    except Exception as e:
        logging.warning(f"Could not load norm stats from checkpoint: {e}")
        logging.warning("Will attempt to load policy with default method (requires dataset access)")
        norm_stats = None

    # Load policy
    policy = _policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_dir,
        default_prompt=getattr(config.data, "default_prompt", None),
        norm_stats=norm_stats,  # Pass pre-loaded norm stats to skip dataset loading
    )

    # Action configuration
    action_config = {
        "real_action_start": getattr(config.model, "real_action_start", 0),
        "real_action_dim": getattr(config.model, "real_action_dim", config.model.action_dim),
    }

    # Create adapters
    obs_adapter = ShirtHangObservationAdapter(camera_names=args.camera_names)
    action_adapter = ShirtHangActionAdapter()

    # Wrap policy with adapters
    adapted_policy = AdaptedPolicy(
        policy=policy,
        obs_adapter=obs_adapter,
        action_adapter=action_adapter,
        action_config=action_config,
    )

    # Create environment (local simulation OR remote robot)
    if args.remote_robot:
        # Remote robot mode - connect to robot server
        logging.info(f"Connecting to remote robot at {args.robot_host}:{args.robot_port}")

        # CRITICAL: Robot server frame stacking config must match training config!
        # Check config.data for frame stacking settings
        obs_history_len = getattr(config.data, "obs_history_len", 1)
        history_gap = getattr(config.data, "history_gap", 29)
        logging.warning("=" * 80)
        logging.warning("FRAME STACKING CONFIGURATION CHECK:")
        logging.warning(f"  Training config: obs_history_len={obs_history_len}, history_gap={history_gap}")
        logging.warning("  Robot server MUST be configured with the SAME values!")
        logging.warning(f"  Check run_robot_server.sh: OBS_HISTORY_LEN={obs_history_len}, HISTORY_GAP={history_gap}")
        logging.warning("=" * 80)

        # Import remote environment adapter
        from pathlib import Path
        import sys

        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from remote_environment_adapter import RemoteEnvironmentAdapter

        env = RemoteEnvironmentAdapter(
            host=args.robot_host,
            port=args.robot_port,
            timeout=args.robot_timeout,
        )
        logging.info("Connected to remote robot server successfully")
    else:
        # Local simulation mode
        logging.info("Using local simulation environment")
        env_kwargs = {
            "control_freq": args.control_hz,
            "time_limit": args.time_limit_s or (args.max_env_steps / args.control_hz),
        }
        if args.overlay_heatmap:
            env_kwargs["overlay_heatmap"] = args.overlay_heatmap

        env = RMPDualXArmsEnv(**env_kwargs)

        if args.enable_intervention:
            if not OCULUS_AVAILABLE:
                raise RuntimeError("OculusIntervention not available")
            env = OculusIntervention(env, freq=args.control_hz)

        env = RelativeFrame(env)
        env = WristRelativeTo(env)

        if FrameStackWrapperEnv is not None:
            obs_history_len = getattr(config.data, "obs_history_len", 1)
            history_gap = getattr(config.data, "history_gap", 29)
            env = FrameStackWrapperEnv(env, n_frames=obs_history_len, gap=history_gap)

    # Run episodes with Oculus button press to start (like real_hang_intervention.py)
    all_rewards = []
    all_max_rewards = []
    all_metrics = []
    total_recorded_steps = 0

    # Setup progress bars
    episodes_progress_bar = tqdm(range(args.num_episodes), desc="Episodes")
    step_progress_bar = tqdm(range(args.max_env_steps), desc="Steps", smoothing=1)

    ep = 0
    while episodes_progress_bar.n < episodes_progress_bar.total:
        try:
            # Wait for Oculus button press if enabled
            if args.wait_for_button and not args.remote_robot:
                oculus_data = get_oculus_reading(timeout=0.01)
                if oculus_data is None:
                    # Silently continue waiting for Oculus data
                    continue

                # Check if A button is pressed to start episode
                if not (oculus_data["left_a_button"] or oculus_data["right_a_button"]):
                    continue

                logging.info("Starting episode (A button pressed)...")

            episode_idx = episodes_progress_bar.n

            total_reward, max_reward, video_paths, episode_metrics, recorded_data = rollout_episode(
                adapted_policy=adapted_policy,
                env=env,
                args=args,
                episode_idx=episode_idx,
                checkpoint_step=int(checkpoint_dir.name),
                step_progress_bar=step_progress_bar,
            )

            all_rewards.append(total_reward)
            all_max_rewards.append(max_reward)
            episode_metrics["episode_idx"] = episode_idx
            episode_metrics["video_paths"] = {k: str(v) for k, v in video_paths.items()}

            # Manual labeling for shirt hanging task
            if args.manual_labeling:
                logging.info("\n" + "=" * 80)
                logging.info(f"Episode {episode_idx} completed:")
                logging.info(f"  Episode length: {episode_metrics['episode_length']} steps")
                logging.info(
                    f"  Human intervention: {episode_metrics['human_percentage']:.1f}% ({episode_metrics['human_steps']} steps)"
                )
                logging.info(
                    f"  Terminated: {episode_metrics['terminated']}, Truncated: {episode_metrics['truncated']}"
                )
                if video_paths:
                    logging.info(f"  Video: {video_paths.get('multiview', 'N/A')}")
                logging.info("=" * 80)

                # Prompt for manual score (0-5 scale for shirt hanging)
                while True:
                    try:
                        score_input = input("\nRate this episode (0-5, where 5 = full task success): ").strip()
                        manual_score = float(score_input)
                        if 0 <= manual_score <= 5:
                            episode_metrics["manual_score"] = manual_score
                            logging.info(f"Manual score recorded: {manual_score}")
                            break
                        logging.warning("Score must be between 0 and 5. Please try again.")
                    except ValueError:
                        logging.warning("Invalid input. Please enter a number between 0 and 5.")
                    except KeyboardInterrupt:
                        logging.warning("\nInterrupted. Setting score to 0 and continuing...")
                        episode_metrics["manual_score"] = 0.0
                        break

            # Save episode data if recording mode is enabled
            if args.record and recorded_data:
                obses, actions, rews, dones, truncateds, infos = recorded_data
                is_save_data = input("Finished episode. Save data? (y/n): ")
                if "y" in is_save_data.lower():
                    task_name = args.config
                    file_name = f"{task_name}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
                    save_dir = Path(args.save_data_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    file_path = save_dir / file_name

                    logging.info(
                        f"Saving data to {file_path}, with {episode_metrics['intervention_step_counter']} steps of intervention"
                    )
                    with open(file_path, "wb") as f:
                        np.savez(
                            f,
                            obses=obses,
                            actions=actions,
                            rews=rews,
                            dones=dones,
                            truncateds=truncateds,
                            infos=infos,
                            allow_pickle=True,
                        )
                    total_recorded_steps += episode_metrics["episode_length"]
                    logging.info(f"Total recorded steps: {total_recorded_steps}")
                else:
                    logging.info("Data discarded!!!")

            all_metrics.append(episode_metrics)

            logging.info(
                f"Episode {episode_idx}: reward={total_reward:.4f}, "
                f"max_reward={max_reward:.4f}, length={episode_metrics['episode_length']}, "
                f"human={episode_metrics['human_percentage']:.1f}%, "
                f"manual_score={episode_metrics.get('manual_score', 'N/A')}"
            )

            if args.dump_episode_jsonl:
                metrics_dir = checkpoint_dir / args.metrics_dir
                metrics_dir.mkdir(parents=True, exist_ok=True)
                with (metrics_dir / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(episode_metrics, default=lambda o: float(o)) + "\n")

            episodes_progress_bar.update(1)

            # Pause before next episode unless auto_continue is enabled
            if not args.auto_continue and episode_idx < args.num_episodes - 1:
                if args.wait_for_button:
                    logging.info("Waiting for A button press to start next episode...")
                else:
                    input("\nPress Enter to start next episode...")

            ep += 1

        except Exception as e:
            logging.error(f"Error in episode {ep}: {e}", exc_info=True)
            if args.wait_for_button:
                logging.info("Waiting for A button press to retry...")
                time.sleep(1)
            else:
                raise

    # Aggregate metrics
    aggregate_metrics = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "mean_max_reward": float(np.mean(all_max_rewards)),
        "mean_episode_length": float(np.mean([m["episode_length"] for m in all_metrics])),
        "mean_human_percentage": float(np.mean([m["human_percentage"] for m in all_metrics])),
    }

    # Add manual score metrics if manual labeling was used
    if args.manual_labeling:
        manual_scores = [m.get("manual_score", 0.0) for m in all_metrics if m.get("manual_score") is not None]
        if manual_scores:
            aggregate_metrics["mean_manual_score"] = float(np.mean(manual_scores))
            aggregate_metrics["std_manual_score"] = float(np.std(manual_scores))
            aggregate_metrics["min_manual_score"] = float(np.min(manual_scores))
            aggregate_metrics["max_manual_score"] = float(np.max(manual_scores))
            # Full task success = score of 5.0 (shirt hanging task)
            aggregate_metrics["full_task_success_rate"] = float(np.mean([s >= 5.0 for s in manual_scores]))
            # Partial success = score >= 3.0 (some progress made)
            aggregate_metrics["partial_success_rate"] = float(np.mean([s >= 3.0 for s in manual_scores]))
            aggregate_metrics["num_scored_episodes"] = len(manual_scores)
        else:
            logging.warning("No manual scores recorded!")
    else:
        # Legacy: use environment rewards (will always be 0.0 for RMPDualXArmsEnv)
        aggregate_metrics["mean_reward_based_success"] = float(np.mean([m["max_reward"] >= 3.0 for m in all_metrics]))

    metrics_dir = checkpoint_dir / args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate_metrics, indent=2))

    # Save manual scores to a text file for easy review
    if args.manual_labeling:
        scores_file = metrics_dir / "manual_scores.txt"
        with scores_file.open("w") as f:
            f.write("Manual Scores for Shirt Hanging Task Evaluation\n")
            f.write("=" * 80 + "\n")
            f.write(f"Checkpoint: {checkpoint_dir.name}\n")
            f.write(f"Total episodes: {len(all_metrics)}\n")
            f.write("\n")
            f.write("Episode-by-Episode Scores:\n")
            f.write("-" * 80 + "\n")
            for m in all_metrics:
                score = m.get("manual_score", "N/A")
                f.write(
                    f"Episode {m['episode_idx']:3d}: Score = {score:4.1f}  "
                    f"(Length: {m['episode_length']:4d} steps, "
                    f"Human: {m['human_percentage']:5.1f}%)\n"
                )
            f.write("-" * 80 + "\n")
            if "mean_manual_score" in aggregate_metrics:
                f.write("\nAggregate Statistics:\n")
                f.write(
                    f"  Mean score: {aggregate_metrics['mean_manual_score']:.2f} ± {aggregate_metrics['std_manual_score']:.2f}\n"
                )
                f.write(
                    f"  Score range: [{aggregate_metrics['min_manual_score']:.1f}, {aggregate_metrics['max_manual_score']:.1f}]\n"
                )
                f.write(f"  Full task success rate (≥ 5.0): {aggregate_metrics['full_task_success_rate']:.2%}\n")
                f.write(f"  Partial success rate (≥ 3.0): {aggregate_metrics['partial_success_rate']:.2%}\n")
        logging.info(f"Manual scores saved to: {scores_file}")

    if run is not None:
        step = int(checkpoint_dir.name)
        wandb_metrics = {f"eval/{k}": v for k, v in aggregate_metrics.items()}
        wandb_metrics["eval/checkpoint_step"] = step
        run.log(wandb_metrics, step=step)

    logging.info("=" * 80)
    logging.info(f"Checkpoint {checkpoint_dir.name} evaluation complete:")
    logging.info(f"  Episodes evaluated: {len(all_metrics)}")
    logging.info(f"  Mean episode length: {aggregate_metrics['mean_episode_length']:.1f} steps")
    logging.info(f"  Mean human intervention: {aggregate_metrics['mean_human_percentage']:.1f}%")

    if args.manual_labeling and "mean_manual_score" in aggregate_metrics:
        logging.info("\n  Manual Scoring Results:")
        logging.info(
            f"    Mean score: {aggregate_metrics['mean_manual_score']:.2f} ± {aggregate_metrics['std_manual_score']:.2f}"
        )
        logging.info(
            f"    Score range: [{aggregate_metrics['min_manual_score']:.1f}, {aggregate_metrics['max_manual_score']:.1f}]"
        )
        logging.info(f"    Full task success rate (score ≥ 5.0): {aggregate_metrics['full_task_success_rate']:.2%}")
        logging.info(f"    Partial success rate (score ≥ 3.0): {aggregate_metrics['partial_success_rate']:.2%}")
        logging.info(f"    Scored episodes: {aggregate_metrics['num_scored_episodes']}/{len(all_metrics)}")
    else:
        logging.info(f"  Mean reward: {aggregate_metrics['mean_reward']:.4f} ± {aggregate_metrics['std_reward']:.4f}")

    logging.info("=" * 80)

    env.close()


def gather_checkpoints(root: Path, steps: list[int] | None) -> list[Path]:
    """Gather checkpoint directories."""
    if not root.exists():
        return []

    if steps is None:
        checkpoints = [item for item in root.iterdir() if item.is_dir() and item.name.isdigit()]
        return sorted(checkpoints, key=lambda p: int(p.name))

    checkpoints = []
    for step in steps:
        checkpoint_dir = root / str(step)
        if checkpoint_dir.exists():
            checkpoints.append(checkpoint_dir)
        else:
            logging.warning(f"Checkpoint {checkpoint_dir} not found")

    return sorted(checkpoints, key=lambda p: int(p.name))


def main(args: EvalArgs) -> None:
    """Main evaluation function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = _config.get_config(args.config)

    run = None
    if args.log_wandb:
        project = args.wandb_project or f"{config.project_name}_eval"
        run_name = args.wandb_run_name or f"{args.config}_{Path(args.checkpoint_root).name}"
        run = wandb.init(project=project, name=run_name, config=dataclasses.asdict(args))

    checkpoint_root = Path(args.checkpoint_root)
    checkpoints = gather_checkpoints(checkpoint_root, args.steps)

    if not checkpoints:
        logging.error(f"No checkpoints found in {checkpoint_root}")
        return

    logging.info(f"Found {len(checkpoints)} checkpoints to evaluate")

    for checkpoint_dir in checkpoints:
        try:
            evaluate_checkpoint(args, checkpoint_dir, config, run)
        except Exception as e:
            logging.error(f"Error evaluating {checkpoint_dir}: {e}", exc_info=True)
            continue

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
