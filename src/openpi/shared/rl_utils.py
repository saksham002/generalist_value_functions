"""Shared RL utilities for value function training and data processing."""

import numpy as np


def compute_mc_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    discount: float = 0.99,
) -> np.ndarray:
    """Compute Monte-Carlo returns for a single trajectory.

    Args:
        rewards: Array of rewards, shape [T].
        dones: Array of done flags, shape [T].
        discount: Discount factor gamma.

    Returns:
        Array of MC returns, shape [T]. Each entry is the discounted sum of
        future rewards from that timestep until the end of the episode.
    """
    num_steps = len(rewards)
    mc_returns = np.zeros(num_steps, dtype=np.float32)

    # Compute returns backwards
    running_return = 0.0
    for t in range(num_steps - 1, -1, -1):
        running_return = rewards[t] if dones[t] else rewards[t] + discount * running_return
        mc_returns[t] = running_return

    return mc_returns
