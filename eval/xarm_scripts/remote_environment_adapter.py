#!/usr/bin/env python3
"""
Remote Environment Adapter - Client-side wrapper for remote robot environment.

This adapter provides a Gym-like interface that communicates with a remote
robot environment server via REST API. It allows policy inference to run on
a separate machine from the robot hardware.

Usage:
    env = RemoteEnvironmentAdapter(host="192.168.1.100", port=8080)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
    env.close()
"""

import base64
import logging
import time
from typing import Any

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class RemoteEnvironmentAdapter:
    """Gym-like interface for remote robot environment via REST API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize remote environment adapter.

        Args:
            host: Robot server hostname or IP address
            port: Robot server port
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.base_url = f"http://{host}:{port}/api"
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self.metadata = None
        self.episode_active = False

        logger.info(f"RemoteEnvironmentAdapter connecting to {self.base_url}")

        # Wait for server to be ready
        self._wait_for_server()

        # Get metadata
        self.metadata = self._get_metadata()
        logger.info(f"Connected to robot server. Metadata: {self.metadata}")

    def _wait_for_server(self, max_wait: float = 60.0):
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("Robot server is ready")
                    return
            except requests.RequestException:
                pass

            logger.info("Waiting for robot server...")
            time.sleep(2.0)

        raise RuntimeError(f"Robot server at {self.base_url} not responding after {max_wait}s")

    def _request_with_retry(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.retry_attempts):
            try:
                response = requests.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt < self.retry_attempts - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Request failed after {self.retry_attempts} attempts: {e}")
                    raise

    def _get_metadata(self) -> dict[str, Any]:
        """Get environment metadata from server."""
        response = self._request_with_retry("GET", "metadata")
        return response.json()

    def get_status(self) -> dict[str, Any]:
        """Get server status."""
        response = self._request_with_retry("GET", "status")
        return response.json()

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """
        Reset environment.

        Args:
            seed: Random seed for episode

        Returns:
            obs: Observation dictionary
            info: Info dictionary
        """
        logger.info(f"Resetting environment (seed={seed})")

        data = {}
        if seed is not None:
            data["seed"] = seed

        response = self._request_with_retry("POST", "reset", json=data)
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(f"Reset failed: {result.get('error', 'Unknown error')}")

        self.episode_active = True
        obs_data = result["observation"]
        obs = self._deserialize_observation(obs_data)
        info = obs_data.get("info", {})

        logger.info("Environment reset complete")
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute action in environment.

        Args:
            action: Action array

        Returns:
            obs: Observation dictionary
            reward: Reward value
            done: Done flag
            truncated: Truncated flag
            info: Info dictionary
        """
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")

        # Send action
        response = self._request_with_retry("POST", "step", json={"action": action.tolist()})
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(f"Step failed: {result.get('error', 'Unknown error')}")

        # Parse response
        obs = self._deserialize_observation(result["observation"])
        reward = result["reward"]
        done = result["done"]
        truncated = result["truncated"]
        info = result["info"]

        # Check if episode ended
        if done or truncated:
            self.episode_active = False
            logger.info(f"Episode ended: done={done}, truncated={truncated}")

        return obs, reward, done, truncated, info

    def get_observation(self) -> dict:
        """Get current observation without stepping."""
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")

        response = self._request_with_retry("GET", "observation")
        result = response.json()

        if not result.get("success", False):
            raise RuntimeError(f"Get observation failed: {result.get('error', 'Unknown error')}")

        return self._deserialize_observation(result["observation"])

    def close(self):
        """Close environment."""
        logger.info("Closing remote environment")
        try:
            response = self._request_with_retry("POST", "close")
            result = response.json()
            if not result.get("success", False):
                logger.warning(f"Close failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"Error closing environment: {e}")

        self.episode_active = False

    def start_recording(self):
        """Start recording episode data on server."""
        response = self._request_with_retry("POST", "start_recording")
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"Start recording failed: {result.get('error', 'Unknown error')}")
        logger.info("Recording started on server")

    def stop_recording(self, save: bool = True) -> str | None:
        """Stop recording episode data on server."""
        response = self._request_with_retry("POST", "stop_recording", json={"save": save})
        result = response.json()
        if not result.get("success", False):
            raise RuntimeError(f"Stop recording failed: {result.get('error', 'Unknown error')}")

        saved_file = result.get("saved_file")
        logger.info(f"Recording stopped on server (saved: {saved_file})")
        return saved_file

    def _deserialize_observation(self, obs_data: dict) -> dict:
        """
        Deserialize observation from JSON format.

        Args:
            obs_data: Serialized observation from server

        Returns:
            Deserialized observation dictionary with numpy arrays
        """
        obs = {"images": {}, "state": {}}

        # Deserialize images
        for cam_name, img_data in obs_data["images"].items():
            # Decode base64 JPEG
            img_bytes = base64.b64decode(img_data["data"])
            img_array = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Reconstruct frame stack shape if needed
            # Server sends last frame, we need to add temporal dimension
            obs["images"][cam_name] = img_array[np.newaxis, ...]  # Add time dimension

        # Deserialize state
        for state_name, state_data in obs_data["state"].items():
            state_array = np.array(state_data, dtype=np.float32)
            # Add temporal dimension for frame-stacked state
            obs["state"][state_name] = state_array[np.newaxis, ...]

        return obs

    @property
    def action_space(self):
        """Mock action space (not used by policy)."""

        class MockSpace:
            def sample(self):
                return np.zeros(14, dtype=np.float32)

        return MockSpace()

    @property
    def observation_space(self):
        """Mock observation space (not used by policy)."""
        return None

    @property
    def unwrapped(self):
        """Return self for compatibility."""
        return self


class RemoteEnvironmentWithWrappers(RemoteEnvironmentAdapter):
    """
    Extended remote environment adapter that mimics local wrapper behavior.

    This version handles frame stacking on the client side to match the exact
    interface expected by eval scripts.
    """

    def __init__(
        self, host: str = "localhost", port: int = 8080, obs_history_len: int = 1, history_gap: int = 29, **kwargs
    ):
        """
        Initialize with frame stacking support.

        Args:
            host: Robot server hostname
            port: Robot server port
            obs_history_len: Number of frames to stack
            history_gap: Gap between stacked frames
            **kwargs: Additional arguments for RemoteEnvironmentAdapter
        """
        super().__init__(host=host, port=port, **kwargs)

        self.obs_history_len = obs_history_len
        self.history_gap = history_gap

        # Frame buffers
        self.image_buffer = {}
        self.state_buffer = {}

        logger.info(
            f"Remote environment with frame stacking: obs_history_len={obs_history_len}, history_gap={history_gap}"
        )

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Reset environment and frame buffers."""
        obs, info = super().reset(seed=seed)

        # Initialize frame buffers
        self.image_buffer = {cam: [obs["images"][cam]] for cam in obs["images"]}
        self.state_buffer = {state: [obs["state"][state]] for state in obs["state"]}

        # Return stacked observation
        return self._get_stacked_observation(), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Step and update frame buffers."""
        obs, reward, done, truncated, info = super().step(action)

        # Update buffers
        for cam in obs["images"]:
            self.image_buffer[cam].append(obs["images"][cam])
        for state in obs["state"]:
            self.state_buffer[state].append(obs["state"][state])

        # Return stacked observation
        return self._get_stacked_observation(), reward, done, truncated, info

    def _get_stacked_observation(self) -> dict:
        """Get frame-stacked observation."""
        obs = {"images": {}, "state": {}}

        # Stack images
        for cam, buffer in self.image_buffer.items():
            # Get frames at specified intervals
            indices = self._get_stack_indices(len(buffer))
            frames = [buffer[i] for i in indices]
            # Stack along first dimension: (T, H, W, C)
            obs["images"][cam] = np.concatenate(frames, axis=0)

        # Stack states
        for state, buffer in self.state_buffer.items():
            indices = self._get_stack_indices(len(buffer))
            states = [buffer[i] for i in indices]
            # Stack along first dimension: (T, D)
            obs["state"][state] = np.concatenate(states, axis=0)

        return obs

    def _get_stack_indices(self, buffer_len: int) -> list:
        """Get indices for frame stacking."""
        if self.obs_history_len == 1:
            return [buffer_len - 1]  # Just return last frame

        # Calculate indices with gap
        indices = []
        for i in range(self.obs_history_len):
            idx = buffer_len - 1 - i * (self.history_gap + 1)
            idx = max(idx, 0)  # Use first frame if not enough history
            indices.append(idx)

        return list(reversed(indices))  # Oldest to newest


# For convenience
def create_remote_env(
    host: str = "localhost", port: int = 8080, use_frame_stacking: bool = False, **kwargs
) -> RemoteEnvironmentAdapter:
    """
    Factory function to create remote environment adapter.

    Args:
        host: Robot server hostname
        port: Robot server port
        use_frame_stacking: Whether to handle frame stacking client-side
        **kwargs: Additional arguments

    Returns:
        Remote environment adapter instance
    """
    if use_frame_stacking:
        return RemoteEnvironmentWithWrappers(host=host, port=port, **kwargs)
    return RemoteEnvironmentAdapter(host=host, port=port, **kwargs)


if __name__ == "__main__":
    # Test connection
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Create environment
    env = RemoteEnvironmentAdapter(host=args.host, port=args.port)

    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Image cameras: {list(obs['images'].keys())}")
    print(f"State keys: {list(obs['state'].keys())}")

    # Test a few steps
    print("\nTesting steps...")
    for i in range(3):
        action = np.zeros(14, dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, done={done}, truncated={truncated}")

        if done or truncated:
            break

    # Close
    env.close()
    print("\nTest complete!")
