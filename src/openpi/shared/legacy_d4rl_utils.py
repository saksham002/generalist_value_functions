"""Utilities for loading legacy D4RL datasets.

This module provides an alternative to Minari for loading D4RL datasets,
using the original d4rl library directly. Includes special handling for
sparse reward environments (antmaze) where failed trajectories use
reward_neg / (1 - gamma) as the return-to-go.

Based on the implementation from PolicyAgnosticRL.

Requirements:
    - gym (legacy gym, not gymnasium)
    - d4rl
    - mujoco-py (for mujoco environments)
"""

import collections
import logging

import d4rl
import gym
import gymnasium
import numpy as np

LEGACY_D4RL_ENV_CONFIG = {
    "antmaze": {
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
}


def get_legacy_d4rl_dims(env_name: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Get observation and action dimensions from a legacy D4RL environment.

    Args:
        env_name: D4RL environment name (e.g., 'antmaze-large-diverse-v2')

    Returns:
        Tuple of (observation_dim, action_dim, action_low, action_high)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)
    env.close()
    return obs_dim, action_dim, action_low, action_high


def calc_return_to_go_sparse(
    env_name: str,
    rewards: np.ndarray,
    masks: np.ndarray,
    gamma: float,
    reward_scale: float,
    reward_bias: float,
    *,
    is_sparse_reward: bool,
) -> np.ndarray:
    """Compute MC returns with special handling for sparse reward environments.

    For entirely failed trajectories (all rewards equal to reward_neg), uses
    reward_neg / (1 - gamma) as the return-to-go. This provides a more
    informative signal than simple discounted sums for sparse reward tasks.

    Args:
        env_name: Environment name (used to determine reward_neg).
        rewards: Array of rewards for the episode (after scaling/bias).
        masks: Array of masks (1 - terminals). Used to reset return accumulation.
        gamma: Discount factor.
        reward_scale: Scale factor used for reward transformation.
        reward_bias: Bias used for reward transformation.
        is_sparse_reward: Whether this is a sparse reward environment.

    Returns:
        Array of return-to-go values with the same shape as rewards.
    """
    if len(rewards) == 0:
        return np.array([], dtype=np.float32)

    if "antmaze" in env_name:
        reward_neg = LEGACY_D4RL_ENV_CONFIG["antmaze"]["reward_neg"] * reward_scale + reward_bias
    else:
        assert not is_sparse_reward, (
            "If you want to try on a sparse reward env, "
            "please add the reward_neg value in the LEGACY_D4RL_ENV_CONFIG dict."
        )
        reward_neg = 0.0

    if is_sparse_reward and np.all(np.isclose(rewards, reward_neg)):
        # All negative rewards -> entirely failed trajectory.
        # Use r / (1-gamma) for negative trajectory.
        # For example, if gamma = 0.99 and reward_neg = -1,
        # then return_to_go = [-100, -100, -100, ...]
        return np.full(len(rewards), reward_neg / (1 - gamma), dtype=np.float32)

    # Standard discounted return computation
    return_to_go = np.zeros(len(rewards), dtype=np.float32)
    prev_return = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        return_to_go[i] = rewards[i] + gamma * prev_return * masks[i]
        prev_return = return_to_go[i]

    return return_to_go


def load_legacy_d4rl_dataset(
    env_name: str,
    discount: float = 0.99,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: float = 0.999,
) -> dict[str, np.ndarray]:
    """Load a legacy D4RL dataset with MC return computation.

    This function:
    1. Loads the dataset using d4rl.qlearning_dataset()
    2. Detects episode boundaries via observation discontinuity
    3. Applies reward scaling/bias
    4. Computes MC returns with sparse reward handling for antmaze
    5. Clips actions to avoid boundary issues

    Args:
        env_name: D4RL environment name (e.g., 'antmaze-large-diverse-v2')
        discount: Discount factor for MC return computation.
        reward_scale: Scale factor for reward transformation (r' = scale * r + bias).
        reward_bias: Bias for reward transformation.
        clip_action: Action clipping margin (clips to [-clip_action, clip_action]).

    Returns:
        Dictionary with keys: observations, actions, next_observations, next_actions,
        rewards, mc_returns, terminals, episode_starts, episode_ends
    """
    logging.info(f"Loading legacy D4RL dataset: {env_name}")
    if reward_scale != 1.0 or reward_bias != 0.0:
        logging.info(f"Applying reward transformation: r' = {reward_scale} * r + {reward_bias}")

    # Determine if this is a sparse reward environment
    is_sparse_reward = "antmaze" in env_name or "maze2d" in env_name

    # Load the raw dataset using d4rl
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env.unwrapped)
    env.close()

    # Detect timeouts via observation discontinuity
    timeouts = np.zeros(len(dataset["rewards"]), dtype=bool)
    for i in range(len(dataset["terminals"]) - 1):
        obs_diff = np.linalg.norm(dataset["observations"][i + 1] - dataset["next_observations"][i])
        if obs_diff > 1e-6 or dataset["terminals"][i] == 1.0:
            timeouts[i] = True
    timeouts[-1] = True

    # Process episodes
    num_total_transitions = len(dataset["rewards"])
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    episode_step = 0
    for i in range(num_total_transitions):
        done_bool = bool(dataset["terminals"][i])
        is_final_timestep = timeouts[i]

        # Skip final timestep transitions (don't include in dataset)
        if not is_final_timestep or i == num_total_transitions - 1:
            for k in [
                "actions",
                "next_observations",
                "observations",
                "rewards",
                "terminals",
            ]:
                if k in dataset:
                    data_[k].append(dataset[k][i])
            if "next_observations" not in dataset:
                data_["next_observations"].append(dataset["observations"][i + 1])
            episode_step += 1

        if (done_bool or is_final_timestep) and episode_step > 0:
            episode_step = 0
            episode_data = {k: np.array(v) for k, v in data_.items()}

            # Apply reward transformation
            episode_data["rewards"] = episode_data["rewards"] * reward_scale + reward_bias

            # Compute MC returns with sparse reward handling
            masks = 1 - episode_data["terminals"].astype(np.float32)
            episode_data["mc_returns"] = calc_return_to_go_sparse(
                env_name,
                episode_data["rewards"],
                masks,
                discount,
                reward_scale,
                reward_bias,
                is_sparse_reward=is_sparse_reward,
            )

            # Clip actions
            episode_data["actions"] = np.clip(episode_data["actions"], -clip_action, clip_action)

            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)

    # Concatenate all episodes
    all_observations = []
    all_actions = []
    all_next_observations = []
    all_next_actions = []
    all_rewards = []
    all_mc_returns = []
    all_terminals = []
    episode_starts = []
    episode_ends = []

    current_idx = 0
    for ep_idx, episode in enumerate(episodes_dict_list):
        if ep_idx % 100 == 0:
            logging.info(f"Processing episode {ep_idx}/{len(episodes_dict_list)}")

        num_transitions = len(episode["actions"])
        episode_starts.append(current_idx)

        for t in range(num_transitions):
            all_observations.append(episode["observations"][t].astype(np.float32))
            all_actions.append(episode["actions"][t].astype(np.float32))
            all_next_observations.append(episode["next_observations"][t].astype(np.float32))

            # next_action: use next timestep's action, or last action for final step
            next_action = episode["actions"][t + 1] if t + 1 < num_transitions else episode["actions"][-1]
            all_next_actions.append(np.clip(next_action.astype(np.float32), -clip_action, clip_action))

            all_rewards.append(episode["rewards"][t])
            all_mc_returns.append(episode["mc_returns"][t])
            all_terminals.append(episode["terminals"][t])

        current_idx += num_transitions
        episode_ends.append(current_idx)

    logging.info(f"Loaded {len(episodes_dict_list)} episodes, {current_idx} transitions")

    return {
        "observations": np.stack(all_observations),
        "actions": np.stack(all_actions),
        "next_observations": np.stack(all_next_observations),
        "next_actions": np.stack(all_next_actions),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "mc_returns": np.array(all_mc_returns, dtype=np.float32),
        "terminals": np.array(all_terminals, dtype=bool),
        "episode_starts": np.array(episode_starts, dtype=np.int64),
        "episode_ends": np.array(episode_ends, dtype=np.int64),
    }


def _convert_gym_space(space: gym.Space) -> gymnasium.Space:
    """Convert a legacy gym Space to a gymnasium Space."""
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(n=space.n)
    if isinstance(space, gym.spaces.Dict):
        return gymnasium.spaces.Dict({k: _convert_gym_space(v) for k, v in space.spaces.items()})
    raise NotImplementedError(f"Unsupported space type for conversion: {type(space)}")


class GymnasiumBridgeWrapper(gymnasium.Env):
    """Bridge legacy gym env to gymnasium Env.

    Handles differences in space types, reset returns, and step returns.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.observation_space = _convert_gym_space(env.observation_space)
        self.action_space = _convert_gym_space(env.action_space)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        try:
            obs = self.env.reset(seed=seed)
        except TypeError:
            # Legacy gym envs might not support seed in reset
            if seed is not None:
                self.env.seed(seed)
            obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = info.get("TimeLimit.truncated", False)
        terminated = done and not truncated
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_legacy_d4rl_env(
    env_name: str,
    max_episode_steps: int = 1000,
    seed: int = 0,
) -> gymnasium.Env:
    """Create a legacy D4RL environment with proper wrappers for evaluation.

    Args:
        env_name: D4RL environment name (e.g., 'antmaze-large-diverse-v2')
        max_episode_steps: Maximum steps per episode.
        seed: Random seed for the environment.

    Returns:
        A wrapped gymnasium environment ready for evaluation.
    """
    try:
        env = gym.make(env_name, seed=seed)
    except TypeError:
        env = gym.make(env_name)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return GymnasiumBridgeWrapper(env)
