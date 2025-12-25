"""Utilities for working with Minari/D4RL datasets."""

import minari
import numpy as np

from openpi.models import mlp_config


def get_minari_dims(dataset_id: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Get observation and action dimensions from a Minari dataset.

    Args:
        dataset_id: Minari dataset ID (e.g., 'D4RL/antmaze/large-diverse-v1')

    Returns:
        Tuple of (observation_dim, action_dim, action_low, action_high)
    """
    dataset = minari.load_dataset(dataset_id, download=True)

    # Handle observation space
    # Note: We only use the 'observation' key from Dict spaces, ignoring goal components
    obs_space = dataset.observation_space
    if hasattr(obs_space, "spaces"):  # Dict space (e.g., antmaze with achieved_goal, desired_goal)
        # Only use 'observation' key to match LeRobot conversion
        if "observation" not in obs_space.spaces:
            raise ValueError(
                f"Dict observation space must have an 'observation' key, got: {list(obs_space.spaces.keys())}"
            )
        obs_dim = int(np.prod(obs_space.spaces["observation"].shape))
    elif hasattr(obs_space, "shape"):  # Box space
        obs_dim = int(np.prod(obs_space.shape))
    else:
        raise ValueError(f"Unknown observation space type: {type(obs_space)}")

    # Handle action space
    action_space = dataset.action_space
    action_dim = int(np.prod(action_space.shape))
    action_low = action_space.low.astype(np.float32)
    action_high = action_space.high.astype(np.float32)

    return obs_dim, action_dim, action_low, action_high


def create_mlp_config_from_minari(
    dataset_id: str,
    *,
    action_horizon: int = 1,
    hidden_dims: tuple[int, ...] = (256, 256),
    dtype: str = "float32",
):
    """Create an MLPConfig with dimensions auto-detected from a Minari dataset.

    Args:
        dataset_id: Minari dataset ID (e.g., 'D4RL/antmaze/large-diverse-v1')
        action_horizon: Number of actions to predict at once.
        hidden_dims: Hidden layer dimensions.
        dtype: Data type for model parameters.

    Returns:
        MLPConfig with state_dim, action_dim, and action bounds set from the dataset.
    """
    state_dim, action_dim, action_low, action_high = get_minari_dims(dataset_id)

    return mlp_config.MLPConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dims=hidden_dims,
        dtype=dtype,
        action_low=tuple(action_low.tolist()),
        action_high=tuple(action_high.tolist()),
    )
