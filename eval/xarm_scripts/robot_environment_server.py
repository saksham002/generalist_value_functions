#!/usr/bin/env python3
"""
Robot Environment Server - Server-side REST API for remote robot control.

This server runs in the robot_env conda environment and exposes the robot
environment (RMPDualXArmsEnv with wrappers) via REST API.

Usage:
    python robot_environment_server.py --port 8080 --enable_intervention

Architecture:
    - Server (this file): Manages environment, exposes REST API
    - Client (openpi/scripts/remote_environment_adapter.py): Sends actions from policy
"""

import argparse
import base64
import json
import logging
import queue
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from einops import rearrange
from flask import Flask, jsonify, request
from flask_cors import CORS
from torchvision.transforms.v2 import CenterCrop, Compose, Resize

# Import robot environment dependencies
try:
    from dual_xarms_sim.rmp_env import RMPDualXArmsEnv
    from dual_xarms_sim.relative_frame import RelativeFrame, WristRelativeTo
    from dual_xarms_sim.oculus_intervention import OculusIntervention
    from bc_flowmatch.env_wrappers.frame_stack_wrapper import FrameStackWrapperEnv
    from bc_flowmatch.utils.eval_utils import video_writer_loop
except ImportError as e:
    raise ImportError(
        "Failed to import robot environment dependencies. "
        "Make sure you're running in the robot_env conda environment."
    ) from e

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RobotEnvironmentServer:
    """Manages robot environment and handles API requests."""

    def __init__(
        self,
        control_freq: int = 60,
        time_limit: float = 300.0,  # 5 minutes
        overlay_heatmap: Optional[str] = None,
        enable_intervention: bool = False,
        obs_history_len: int = 2,
        history_gap: int = 29,
        camera_names: tuple = ("right/top", "left/wrist", "right/wrist"),
        record_videos: bool = True,
        video_dir: str = "./robot_server_videos",
    ):
        """Initialize robot environment server."""
        self.control_freq = control_freq
        self.time_limit = time_limit
        self.overlay_heatmap = overlay_heatmap
        self.enable_intervention = enable_intervention
        self.obs_history_len = obs_history_len
        self.history_gap = history_gap
        self.camera_names = camera_names
        self.record_videos = record_videos
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Environment state
        self.env = None
        self.episode_active = False
        self.episode_idx = 0
        self.current_obs = None
        self.step_count = 0

        # Recording state
        self.recording_active = False
        self.recorded_data = {
            "obses": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "truncateds": [],
            "infos": [],
        }

        # Video recording
        self.video_queue = None
        self.video_thread = None

        # Create environment
        self._create_environment()

        logger.info("Robot environment server initialized")

    def _create_environment(self):
        """Create robot environment with all wrappers."""
        logger.info("Creating robot environment...")

        # Base environment
        env_kwargs = {
            "control_freq": self.control_freq,
            "time_limit": self.time_limit,
        }
        if self.overlay_heatmap:
            env_kwargs["overlay_heatmap"] = self.overlay_heatmap

        self.env = RMPDualXArmsEnv(**env_kwargs)

        # Intervention wrapper
        if self.enable_intervention:
            logger.info("Enabling Oculus intervention")
            self.env = OculusIntervention(self.env, freq=self.control_freq)

        # Relative frame wrappers
        self.env = RelativeFrame(self.env)
        self.env = WristRelativeTo(self.env)

        # Frame stacking wrapper
        if self.obs_history_len > 1:
            logger.info(f"Enabling frame stacking: {self.obs_history_len} frames, gap={self.history_gap}")
            self.env = FrameStackWrapperEnv(
                self.env,
                n_frames=self.obs_history_len,
                gap=self.history_gap
            )

        logger.info("Robot environment created successfully")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment and start new episode."""
        logger.info(f"Resetting environment (episode {self.episode_idx}, seed={seed})")

        # Reset environment
        obs, info = self.env.reset(seed=seed)
        self.current_obs = obs
        self.episode_active = True
        self.step_count = 0

        # Start video recording
        if self.record_videos:
            self._start_video_recording()

        # Add first frame to video
        if self.video_queue is not None:
            self._add_video_frame(obs, "Episode started")

        logger.info("Environment reset complete")
        return {"observation": self._serialize_observation(obs), "info": info}

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """Execute action in environment."""
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")

        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)

        # Step environment
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_obs = obs
        self.step_count += 1

        # Record data if active
        if self.recording_active:
            self._record_step(obs, action, reward, done, truncated, info)

        # Add frame to video
        if self.video_queue is not None:
            human_steps = len([i for i in self.recorded_data["infos"] if "intervene_action" in i])
            human_pct = (human_steps / self.step_count * 100) if self.step_count > 0 else 0
            self._add_video_frame(obs, f"Step {self.step_count}, Human: {human_pct:.1f}%")

        # Check if episode ended
        if done or truncated:
            self.episode_active = False
            self.episode_idx += 1
            logger.info(f"Episode ended: done={done}, truncated={truncated}, steps={self.step_count}")

            # Stop video recording
            if self.video_queue is not None:
                self._stop_video_recording()

        return {
            "observation": self._serialize_observation(obs),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": self._serialize_info(info),
        }

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation without stepping."""
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")

        return {"observation": self._serialize_observation(self.current_obs)}

    def close(self):
        """Close environment."""
        logger.info("Closing environment")

        # Stop video if active
        if self.video_queue is not None:
            self._stop_video_recording()

        if self.env is not None:
            self.env.close()

        self.episode_active = False

    def get_metadata(self) -> Dict[str, Any]:
        """Get environment metadata."""
        return {
            "control_freq": self.control_freq,
            "camera_names": list(self.camera_names),
            "obs_history_len": self.obs_history_len,
            "history_gap": self.history_gap,
            "enable_intervention": self.enable_intervention,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return {
            "episode_active": self.episode_active,
            "episode_idx": self.episode_idx,
            "step_count": self.step_count,
            "recording_active": self.recording_active,
        }

    def start_recording(self):
        """Start recording episode data."""
        self.recording_active = True
        self.recorded_data = {
            "obses": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "truncateds": [],
            "infos": [],
        }
        logger.info("Recording started")

    def stop_recording(self, save: bool = True) -> Optional[str]:
        """Stop recording and optionally save data."""
        self.recording_active = False

        if save and len(self.recorded_data["actions"]) > 0:
            filename = f"robot_episode_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            filepath = self.video_dir / filename

            with open(filepath, "wb") as f:
                np.savez(
                    f,
                    obses=self.recorded_data["obses"],
                    actions=self.recorded_data["actions"],
                    rews=self.recorded_data["rewards"],
                    dones=self.recorded_data["dones"],
                    truncateds=self.recorded_data["truncateds"],
                    infos=self.recorded_data["infos"],
                    allow_pickle=True,
                )

            logger.info(f"Recording saved to {filepath}")
            return str(filepath)

        logger.info("Recording stopped (not saved)")
        return None

    def _record_step(self, obs, action, reward, done, truncated, info):
        """Record a single step."""
        # Compress images for storage
        obs_copy = deepcopy(obs)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        for name, img in obs_copy["images"].items():
            single_frame = img[-1] if img.ndim == 4 else img
            _, encoded_image = cv2.imencode('.jpg', single_frame, encode_param)
            obs_copy["images"][name] = encoded_image

        for name, state in obs_copy["state"].items():
            obs_copy["state"][name] = state[-1] if state.ndim > 1 else state

        self.recorded_data["obses"].append(obs_copy)
        self.recorded_data["actions"].append(action)
        self.recorded_data["rewards"].append(reward)
        self.recorded_data["dones"].append(done)
        self.recorded_data["truncateds"].append(truncated)
        self.recorded_data["infos"].append(info)

    def _start_video_recording(self):
        """Start video recording thread."""
        if self.video_queue is not None:
            self._stop_video_recording()

        self.video_queue = queue.Queue(maxsize=500)
        uuid = time.strftime("%Y%m%d-%H%M%S")
        filename = self.video_dir / f"episode_{self.episode_idx}_{uuid}.mp4"

        # Calculate video dimensions based on number of cameras
        video_width = 224 * len(self.camera_names)
        video_height = 224

        # Image crop function to resize frames to 224x224 (same as real_hang_intervention.py)
        image_crop = Compose([
            CenterCrop((360, 360)),
            Resize((256, 256)),
            CenterCrop((224, 224)),
        ])

        # Wrapper to catch exceptions in video writer thread
        def video_writer_wrapper():
            try:
                logger.debug(f"Starting video writer: {filename}, size=({video_width}, {video_height}), fps={self.control_freq}")
                video_writer_loop(
                    self.video_queue,
                    str(filename),
                    self.control_freq,
                    (video_width, video_height),
                    image_crop,  # Apply same cropping as intervention script
                )
                logger.info(f"Video writer completed successfully: {filename}")
            except Exception as e:
                logger.error(f"Video writer thread crashed: {e}", exc_info=True)
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

        self.video_thread = threading.Thread(
            target=video_writer_wrapper,
            daemon=True,
        )
        self.video_thread.start()
        logger.info(f"Video recording started: {filename}")

    def _stop_video_recording(self):
        """Stop video recording thread."""
        if self.video_queue is not None:
            self.video_queue.put(None)  # Signal to stop
            if self.video_thread is not None:
                self.video_thread.join(timeout=5.0)
            self.video_queue = None
            self.video_thread = None
            logger.info("Video recording stopped")

    def _add_video_frame(self, obs, text_overlay: str = ""):
        """Add frame to video queue."""
        if self.video_queue is None:
            logger.warning("Video queue is None, cannot add frame")
            return

        try:
            frames = []
            available_cameras = list(obs.get("images", {}).keys())

            for cam_name in self.camera_names:
                if cam_name in obs["images"]:
                    img = obs["images"][cam_name]
                    frame = img[-1] if img.ndim == 4 else img

                    # Ensure frame is uint8 and has correct shape
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)

                    # Ensure frame is BGR (3 channels)
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[-1] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    frames.append(frame)
                else:
                    logger.warning(f"Camera '{cam_name}' not found in observation. Available: {available_cameras}")

            if frames:
                # Validate all frames have same shape
                first_shape = frames[0].shape
                if not all(f.shape == first_shape for f in frames):
                    logger.error(f"Frame shape mismatch! First: {first_shape}, Others: {[f.shape for f in frames[1:]]}")

                logger.debug(f"Adding {len(frames)} frames to video queue (shape: {first_shape}, dtype: {frames[0].dtype})")
                self.video_queue.put((frames, text_overlay), block=False)
            else:
                logger.warning(f"No frames to add! Expected cameras: {self.camera_names}, Available: {available_cameras}")
        except queue.Full:
            logger.warning("Video queue full, dropping frame")
        except Exception as e:
            logger.error(f"Error adding video frame: {e}", exc_info=True)

    def _serialize_observation(self, obs: Dict) -> Dict[str, Any]:
        """Serialize observation for JSON transmission."""
        serialized = {
            "images": {},
            "state": {},
        }

        # Serialize images as JPEG base64
        for cam_name, img in obs["images"].items():
            # Take last frame if frame-stacked
            frame = img[-1] if img.ndim == 4 else img

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            serialized["images"][cam_name] = {
                "data": img_base64,
                "shape": list(frame.shape),
            }

        # Serialize state
        for state_name, state in obs["state"].items():
            # Take last state if frame-stacked
            state_array = state[-1] if state.ndim > 1 else state
            serialized["state"][state_name] = state_array.tolist()

        return serialized

    def _serialize_info(self, info: Dict) -> Dict[str, Any]:
        """Serialize info dict for JSON transmission."""
        serialized = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized


# Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global server instance
server: Optional[RobotEnvironmentServer] = None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get environment metadata."""
    if server is None:
        return jsonify({"success": False, "error": "Server not initialized"}), 500
    return jsonify(server.get_metadata())


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status."""
    if server is None:
        return jsonify({"success": False, "error": "Server not initialized"}), 500
    return jsonify(server.get_status())


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset environment."""
    try:
        data = request.get_json() or {}
        seed = data.get('seed', None)

        result = server.reset(seed=seed)
        return jsonify({"success": True, **result})

    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/step', methods=['POST'])
def step():
    """Step environment with action."""
    try:
        data = request.get_json()
        action = np.array(data['action'], dtype=np.float32)

        result = server.step(action)
        return jsonify({"success": True, **result})

    except Exception as e:
        logger.error(f"Step failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/observation', methods=['GET'])
def get_observation():
    """Get current observation."""
    try:
        result = server.get_observation()
        return jsonify({"success": True, **result})

    except Exception as e:
        logger.error(f"Get observation failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/close', methods=['POST'])
def close():
    """Close environment."""
    try:
        server.close()
        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Close failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start recording episode data."""
    try:
        server.start_recording()
        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Start recording failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording episode data."""
    try:
        data = request.get_json() or {}
        save = data.get('save', True)

        saved_file = server.stop_recording(save=save)
        return jsonify({"success": True, "saved_file": saved_file})

    except Exception as e:
        logger.error(f"Stop recording failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Robot Environment Server - Exposes robot environment via REST API"
    )
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--control_freq', type=int, default=60, help='Control frequency (Hz)')
    parser.add_argument('--time_limit', type=float, default=300.0, help='Episode time limit (seconds)')
    parser.add_argument('--overlay_heatmap', type=str, default=None, help='Path to overlay heatmap image')
    parser.add_argument('--enable_intervention', action='store_true', help='Enable Oculus intervention')
    parser.add_argument('--obs_history_len', type=int, default=2, help='Observation history length')
    parser.add_argument('--history_gap', type=int, default=29, help='History gap between frames')
    parser.add_argument('--video_dir', type=str, default='./robot_server_videos', help='Video output directory')
    parser.add_argument('--no_videos', action='store_true', help='Disable video recording')

    args = parser.parse_args()

    # Create global server instance
    global server
    logger.info("Initializing robot environment server...")
    server = RobotEnvironmentServer(
        control_freq=args.control_freq,
        time_limit=args.time_limit,
        overlay_heatmap=args.overlay_heatmap,
        enable_intervention=args.enable_intervention,
        obs_history_len=args.obs_history_len,
        history_gap=args.history_gap,
        camera_names=("right/top", "left/wrist", "right/wrist"),
        record_videos=not args.no_videos,
        video_dir=args.video_dir,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")

    # Run Flask server
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

