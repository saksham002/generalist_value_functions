"""Utilities for working with Minari/D4RL datasets."""

try:
    import minari
except ImportError:
    minari = None  # type: ignore
import numpy as np


def get_minari_dims(dataset_id: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Get observation and action dimensions from a Minari dataset.

    Args:
        dataset_id: Minari dataset ID (e.g., 'D4RL/antmaze/large-diverse-v1')

    Returns:
        Tuple of (observation_dim, action_dim, action_low, action_high)
    """
    if minari is None:
        raise ImportError("minari is required for this function but is not installed. Install with: pip install minari")
    dataset = minari.load_dataset(dataset_id, download=True)

    # Handle observation space - concatenate all spaces sorted by key
    obs_space = dataset.observation_space
    if hasattr(obs_space, "spaces"):  # Dict space
        obs_dim = sum(int(np.prod(obs_space.spaces[k].shape)) for k in sorted(obs_space.spaces.keys()))
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
