"""Policy evaluation utilities for Gymnasium environments.

This module provides utilities for evaluating learned policies in Gymnasium
environments during training. Supports recovering environments from Minari
datasets for seamless integration with offline RL training.

Uses vectorized environments for efficient parallel evaluation.
"""

from collections.abc import Callable
import dataclasses
import logging
from typing import TYPE_CHECKING

import gymnasium
import minari
import mujoco
import numpy as np

import openpi.shared.array_typing as at

if TYPE_CHECKING:
    from openpi.training import config as _config


@dataclasses.dataclass
class EvalResults:
    """Results from policy evaluation."""

    episode_returns: list[float]
    episode_lengths: list[int]
    # Observation statistics from all evaluation steps
    obs_min: float = 0.0
    obs_max: float = 0.0
    obs_mean: float = 0.0
    obs_std: float = 0.0
    # Action statistics from all evaluation steps
    action_min: float = 0.0
    action_max: float = 0.0
    action_mean: float = 0.0
    action_std: float = 0.0
    # Video frames from first episode (if recorded)
    # Shape: [num_frames, height, width, channels] or None
    video_frames: np.ndarray | None = None

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.episode_returns))

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths))

    def to_dict(self, prefix: str = "eval") -> dict[str, float]:
        """Convert results to a dictionary for logging."""
        return {
            f"{prefix}/mean_return": self.mean_return,
            f"{prefix}/std_return": self.std_return,
            f"{prefix}/mean_length": self.mean_length,
            f"{prefix}/min_return": float(np.min(self.episode_returns)),
            f"{prefix}/max_return": float(np.max(self.episode_returns)),
            f"{prefix}/obs_min": self.obs_min,
            f"{prefix}/obs_max": self.obs_max,
            f"{prefix}/obs_mean": self.obs_mean,
            f"{prefix}/obs_std": self.obs_std,
            f"{prefix}/action_min": self.action_min,
            f"{prefix}/action_max": self.action_max,
            f"{prefix}/action_mean": self.action_mean,
            f"{prefix}/action_std": self.action_std,
        }


def create_eval_env(
    eval_config: "_config.EvalEnvConfig",
    data_config: "_config.DataConfig",
) -> gymnasium.Env:
    """Create a single Gymnasium environment for evaluation.

    For MinariEvalEnvConfig, recovers the environment from the Minari dataset.
    Falls back to using the dataset ID from data_config if not specified.

    Args:
        eval_config: Evaluation environment configuration.
        data_config: Data configuration (used for fallback dataset ID).

    Returns:
        A Gymnasium environment ready for evaluation.
    """
    from openpi.training import config as _config

    if isinstance(eval_config, _config.MinariEvalEnvConfig):
        dataset_id = eval_config.minari_dataset_id or data_config.minari_dataset_id
        if dataset_id is None:
            raise ValueError(
                "MinariEvalEnvConfig requires minari_dataset_id to be set, either in eval_config or data_config."
            )
        logging.info(f"Recovering evaluation environment from Minari dataset: {dataset_id}")
        dataset = minari.load_dataset(dataset_id, download=True)
        env = dataset.recover_environment()

        # Apply max_episode_steps wrapper if specified
        if eval_config.max_episode_steps > 0:
            env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=eval_config.max_episode_steps)

        return env

    raise NotImplementedError(f"Unsupported eval config type: {type(eval_config)}")


def create_vector_eval_env(
    eval_config: "_config.EvalEnvConfig",
    data_config: "_config.DataConfig",
    num_envs: int,
    *,
    render_mode: str | None = None,
) -> gymnasium.vector.VectorEnv:
    """Create a vectorized Gymnasium environment for parallel evaluation.

    Args:
        eval_config: Evaluation environment configuration.
        data_config: Data configuration (used for fallback dataset ID).
        num_envs: Number of parallel environments.
        render_mode: Render mode for environments (e.g., "rgb_array" for video recording).

    Returns:
        A vectorized Gymnasium environment.
    """
    from openpi.training import config as _config

    if isinstance(eval_config, _config.MinariEvalEnvConfig):
        dataset_id = eval_config.minari_dataset_id or data_config.minari_dataset_id
        if dataset_id is None:
            raise ValueError(
                "MinariEvalEnvConfig requires minari_dataset_id to be set, either in eval_config or data_config."
            )

        dataset = minari.load_dataset(dataset_id, download=True)

        def make_env():
            env = dataset.recover_environment(render_mode=render_mode)
            if eval_config.max_episode_steps > 0:
                env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=eval_config.max_episode_steps)
            return env

        logging.info(f"Creating {num_envs} async vectorized eval environments from: {dataset_id}")
        return gymnasium.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])

    raise NotImplementedError(f"Unsupported eval config type: {type(eval_config)}")


@at.typecheck
def evaluate_policy_vectorized(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    vec_env: gymnasium.vector.VectorEnv,
    num_episodes: int,
    seed: int = 42,
    *,
    record_video: bool = False,
) -> EvalResults:
    """Run policy in vectorized environment and collect episode returns.

    Uses vectorized environments to evaluate multiple episodes in parallel.

    Args:
        policy_fn: A function that takes a batch of flattened observations
            with shape [num_envs, obs_dim] and returns actions [num_envs, action_dim].
        vec_env: Vectorized Gymnasium environment.
        num_episodes: Total number of episodes to collect.
        seed: Random seed for environment resets.
        record_video: If True, record frames from the first completed episode.

    Returns:
        EvalResults containing episode returns, lengths, observation/action statistics,
        and optionally video frames.
    """
    num_envs = vec_env.num_envs
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    # Track per-env statistics
    current_returns = np.zeros(num_envs, dtype=np.float32)
    current_lengths = np.zeros(num_envs, dtype=np.int32)

    # Track observation and action statistics
    all_obs = []
    all_actions = []

    video_frames: list[np.ndarray] = []
    first_episode_done = False

    # Reset all environments
    obs, _ = vec_env.reset(seed=seed)
    obs_flat = _flatten_obs_batch(obs)

    if record_video:
        try:
            frames = vec_env.call("render")
        except mujoco.FatalError as e:
            logging.warning(
                f"Failed to render frame for video: {e}. Try adding MUJOCO_GL=EGL to your environment variables."
            )
            raise
        if frames[0] is not None:
            video_frames.append(np.asarray(frames[0], dtype=np.uint8))

    while len(episode_returns) < num_episodes:
        # Get actions for all environments
        actions = policy_fn(obs_flat)

        all_obs.append(obs_flat)
        all_actions.append(actions)

        # Step all environments
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        obs_flat = _flatten_obs_batch(obs)

        if record_video and not first_episode_done:
            frames = vec_env.call("render")
            if frames[0] is not None:
                video_frames.append(np.asarray(frames[0], dtype=np.uint8))

        # Update running statistics
        current_returns += rewards
        current_lengths += 1

        # Check for completed episodes
        dones = np.logical_or(terminateds, truncateds)
        for i in np.where(dones)[0]:
            if len(episode_returns) < num_episodes:
                episode_returns.append(float(current_returns[i]))
                episode_lengths.append(int(current_lengths[i]))

            if i == 0 and not first_episode_done:
                first_episode_done = True

            current_returns[i] = 0.0
            current_lengths[i] = 0

    # Compute final statistics
    all_obs_arr = np.concatenate(all_obs)
    all_actions_arr = np.concatenate(all_actions)

    logging.debug(
        f"Vectorized eval complete: {len(episode_returns)} episodes, mean_return={np.mean(episode_returns):.2f}"
    )

    video_array = None
    if video_frames:
        video_array = np.stack(video_frames, axis=0)  # [T, H, W, C]
        logging.info(f"Recorded video with {len(video_frames)} frames, shape: {video_array.shape}")

    return EvalResults(
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        obs_min=float(np.min(all_obs_arr)),
        obs_max=float(np.max(all_obs_arr)),
        obs_mean=float(np.mean(all_obs_arr)),
        obs_std=float(np.std(all_obs_arr)),
        action_min=float(np.min(all_actions_arr)),
        action_max=float(np.max(all_actions_arr)),
        action_mean=float(np.mean(all_actions_arr)),
        action_std=float(np.std(all_actions_arr)),
        video_frames=video_array,
    )


def _flatten_obs_batch(obs: dict | np.ndarray) -> np.ndarray:
    """Flatten a batch of observations to shape [num_envs, obs_dim].

    Handles both dict observations (concatenates all values sorted by key)
    and array observations.
    """
    if isinstance(obs, dict):
        # Each value has shape [num_envs, ...], flatten per-env and concatenate
        arrays = []
        for k in sorted(obs.keys()):
            arr = np.asarray(obs[k])
            # Reshape to [num_envs, -1]
            arr = arr.reshape(arr.shape[0], -1)
            arrays.append(arr)
        return np.concatenate(arrays, axis=1).astype(np.float32)
    # Already shape [num_envs, obs_dim] or [num_envs, ...]
    arr = np.asarray(obs)
    return arr.reshape(arr.shape[0], -1).astype(np.float32)
