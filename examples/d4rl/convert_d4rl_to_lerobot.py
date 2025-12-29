"""
Convert Minari datasets (D4RL successor) to LeRobot format.

Supports any Minari dataset (antmaze, locomotion, etc.)
Includes all RL fields needed for value function training:
- state, next_state: Current and next state observations
- action, next_action: Current and next actions (for SARSA)
- reward: Immediate reward
- mc_return: Monte-Carlo returns (discounted sum of future rewards)
- termination: True if episode ended naturally (goal reached, failure, etc.)
- truncation: True if episode was cut off (time limit)

Note: termination vs truncation matters for TD learning. Only terminated states
should mask out the bootstrap term; truncated states should still bootstrap.

Usage:
    uv run examples/d4rl/convert_d4rl_to_lerobot.py --dataset_id D4RL/antmaze/large-diverse-v1

To list available datasets:
    python -c "import minari; print(minari.list_remote_datasets())"

Note: Install minari first: `uv pip install minari`
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import minari
import numpy as np
import tyro


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


def main(
    dataset_id: str,
    *,
    repo_name: str | None = None,
    hf_username: str = "local",
    max_episodes_to_save: int | None = None,
    push_to_hub: bool = False,
    discount: float = 0.99,
):
    """Convert a Minari dataset to LeRobot format.

    Args:
        dataset_id: Minari dataset ID (e.g., 'D4RL/antmaze/large-diverse-v1')
        repo_name: Output repository name. If not set, uses '{hf_username}/minari_{dataset_id}'.
        hf_username: Hugging Face username for the repo. Defaults to 'local'.
        max_episodes_to_save: If set, only save up to this many episodes.
        push_to_hub: Whether to push to Hugging Face Hub.
        discount: Discount factor for MC return computation. Defaults to 0.99.
    """
    if repo_name is None:
        # Convert dataset_id to valid repo name
        safe_name = dataset_id.replace("/", "_").replace("-", "_")
        repo_name = f"{hf_username}/minari_{safe_name}"

    # Load minari dataset (download if not available locally)
    print(f"Loading dataset: {dataset_id}")
    dataset = minari.load_dataset(dataset_id, download=True)

    # Get dimensions from first episode
    first_episode = dataset[0]
    obs_sample = first_episode.observations
    action_sample = first_episode.actions

    # Handle dict observations (common in some envs like antmaze)
    if isinstance(obs_sample, dict):
        # Only use 'observation' key, ignore goal spaces (achieved_goal, desired_goal)
        if "observation" in obs_sample:
            obs_dim = int(np.prod(obs_sample["observation"].shape[1:]))
        else:
            raise ValueError(f"Dict observations must have 'observation' key, got: {list(obs_sample.keys())}")
    else:
        obs_dim = int(obs_sample.shape[1] if len(obs_sample.shape) > 1 else obs_sample.shape[0])

    action_dim = int(action_sample.shape[1] if len(action_sample.shape) > 1 else action_sample.shape[0])

    print(f"Dataset: {dataset_id}")
    print(f"Number of episodes: {dataset.total_episodes}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Discount factor: {discount}")

    # Clean up any existing dataset
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset with all RL fields
    dataset_lerobot = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="minari",
        fps=20,  # Most D4RL envs use 20Hz
        features={
            "state": {
                "dtype": "float32",
                "shape": (obs_dim,),
                "names": ["state"],
            },
            "next_state": {
                "dtype": "float32",
                "shape": (obs_dim,),
                "names": ["next_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
            "next_actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["next_actions"],
            },
            "reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "mc_return": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["mc_return"],
            },
            "termination": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["termination"],
            },
            "truncation": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["truncation"],
            },
        },
    )

    # Convert episodes
    num_episodes = dataset.total_episodes
    if max_episodes_to_save is not None:
        num_episodes = min(num_episodes, max_episodes_to_save)

    for ep_idx in range(num_episodes):
        if ep_idx % 100 == 0:
            print(f"Processing episode {ep_idx}/{num_episodes}")

        episode = dataset[ep_idx]
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        truncations = episode.truncations

        # Handle dict observations - only use 'observation' key
        if isinstance(observations, dict):
            if "observation" in observations:
                observations = observations["observation"]
            else:
                raise ValueError(f"Dict observations must have 'observation' key, got: {list(observations.keys())}")

        # Check if this is an antmaze dataset
        is_antmaze = "antmaze" in dataset_id.lower()

        if is_antmaze:
            # For antmaze: truncate episode after first positive reward
            # The original data has invalid transitions after reaching the goal
            positive_reward_idx = np.where(rewards > 0)[0]
            if len(positive_reward_idx) > 0:
                # Keep data up to and including the first positive reward
                end_idx = positive_reward_idx[0] + 1
                observations = observations[: end_idx + 1]  # T+1 observations
                actions = actions[:end_idx]
                rewards = rewards[:end_idx]
                terminations = terminations[:end_idx]
                truncations = truncations[:end_idx]

            # For antmaze: termination = (reward == 1), truncation = False
            # The original flags are incorrect; reward=1 means goal reached (termination)
            terminations = rewards > 0
            truncations = np.zeros_like(truncations, dtype=bool)

        # Compute MC returns using combined done (termination OR truncation)
        # For MC returns, both end the episode trajectory
        dones = np.logical_or(terminations, truncations)
        mc_returns = compute_mc_returns(rewards, dones, discount)

        # Minari episodes have T+1 observations and T actions/rewards/etc.
        for t in range(len(actions)):
            obs = observations[t]
            next_obs = observations[t + 1] if t + 1 < len(observations) else observations[-1]
            action = actions[t]
            # next_action: action at t+1, or repeat last action if at end
            next_action = actions[t + 1] if t + 1 < len(actions) else actions[-1]

            dataset_lerobot.add_frame(
                {
                    "state": obs.astype(np.float32),
                    "next_state": next_obs.astype(np.float32),
                    "actions": action.astype(np.float32),
                    "next_actions": next_action.astype(np.float32),
                    "reward": np.array([rewards[t]], dtype=np.float32),
                    "mc_return": np.array([mc_returns[t]], dtype=np.float32),
                    "termination": np.array([terminations[t]], dtype=bool),
                    "truncation": np.array([truncations[t]], dtype=bool),
                    "task": dataset_id,
                }
            )
        dataset_lerobot.save_episode()

    print(f"Dataset saved to {output_path}")

    # Optionally push to hub
    if push_to_hub:
        dataset_lerobot.push_to_hub(
            tags=["minari", "d4rl", dataset_id.split("/")[0]],
            private=False,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
