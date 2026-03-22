from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
try:
    import minari
except ImportError:
    minari = None  # type: ignore
import numpy as np
import tensorflow as tf
import torch

import openpi.models.model as _model
try:
    import openpi.shared.legacy_d4rl_utils as legacy_d4rl_utils
except Exception:
    legacy_d4rl_utils = None  # type: ignore
import openpi.shared.rl_utils as rl_utils
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.samplers as samplers
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


@dataclasses.dataclass
class NumpyDataset(Dataset):
    """In-memory dataset storing all data as numpy arrays."""

    states: np.ndarray  # [N, state_dim]
    actions: np.ndarray  # [N, action_dim]
    next_states: np.ndarray  # [N, state_dim]
    next_actions: np.ndarray  # [N, action_dim]
    rewards: np.ndarray  # [N]
    mc_returns: np.ndarray  # [N]
    terminations: np.ndarray  # [N] bool
    truncations: np.ndarray  # [N] bool
    episode_starts: np.ndarray  # [num_episodes] - start index of each episode
    episode_ends: np.ndarray  # [num_episodes] - end index of each episode (exclusive)

    def __getitem__(self, index: SupportsIndex | Sequence[int] | np.ndarray | slice) -> dict:
        if isinstance(index, int | np.integer):
            idx = index
            return {
                "state": self.states[idx],
                "actions": self.actions[idx],
                "next_state": self.next_states[idx],
                "next_actions": self.next_actions[idx],
                "reward": np.float32(self.rewards[idx]),
                "mc_return": np.float32(self.mc_returns[idx]),
                "termination": self.terminations[idx],
                "truncation": self.truncations[idx],
            }
        return self.get_items_by_indices(index)

    def __len__(self) -> int:
        return len(self.states)

    @property
    def num_episodes(self) -> int:
        return len(self.episode_starts)

    def get_episode_frames(self, episode_idx: int) -> list[dict]:
        """Get all frames for a specific episode."""
        start = self.episode_starts[episode_idx]
        end = self.episode_ends[episode_idx]
        return [self[i] for i in range(start, end)]

    def get_items_by_indices(self, indices: np.ndarray) -> dict:
        """Get multiple items by indices, returning stacked arrays."""
        return {
            "state": self.states[indices],
            "actions": self.actions[indices],
            "next_state": self.next_states[indices],
            "next_actions": self.next_actions[indices],
            "reward": self.rewards[indices].astype(np.float32),
            "mc_return": self.mc_returns[indices].astype(np.float32),
            "termination": self.terminations[indices],
            "truncation": self.truncations[indices],
        }


class MultiTransitionDataset(Dataset):
    """Dataset that returns multiple transitions per sample using a sampler.

    Each sample contains num_transitions_per_sample transitions.
    The output shape is [num_transitions_per_sample, ...] for each field.

    Supports both single-index access (returns [n, ...]) and batched access
    when given a list of indices from BatchSampler (returns [batch, n, ...]).
    """

    def __init__(
        self,
        dataset: NumpyDataset,
        sampler: samplers.Sampler,
        num_samples: int | None = None,
    ):
        self._dataset = dataset
        self._sampler = sampler
        self._num_samples = num_samples if num_samples is not None else len(dataset)

    def __getitem__(self, index: SupportsIndex | Sequence[int]) -> dict:
        # Handle batched access (list of indices from BatchSampler)
        if isinstance(index, list | np.ndarray):
            batch_size = len(index)
            samples = [self._sample_transitions() for _ in range(batch_size)]
            return {k: np.stack([s[k] for s in samples], axis=0) for k in samples[0]}
        # Single-index access
        return self._sample_transitions()

    def _sample_transitions(self) -> dict:
        """Sample a single multi-transition group."""
        indices = self._sampler.sample()
        return self._dataset.get_items_by_indices(indices)

    def __len__(self) -> int:
        return self._num_samples


def create_numpy_dataset_from_minari(
    minari_dataset_id: str,
    discount: float = 0.99,
    max_episodes: int | None = None,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
) -> NumpyDataset:
    """Create a NumpyDataset by loading data directly from a Minari dataset.

    This computes MC returns on-the-fly and applies antmaze-specific corrections
    (same logic as convert_d4rl_to_lerobot.py).

    Args:
        minari_dataset_id: Minari dataset ID (e.g., 'D4RL/antmaze/large-diverse-v1')
        discount: Discount factor for MC return computation
        max_episodes: If set, only load up to this many episodes
        reward_scale: Scale factor for reward transformation (r' = scale * r + bias)
        reward_bias: Bias for reward transformation (r' = scale * r + bias)

    Returns:
        NumpyDataset with all data in memory
    """
    if minari is None:
        raise ImportError("minari is required for this function but is not installed. Install with: pip install minari")
    logging.info(f"Loading Minari dataset: {minari_dataset_id}")
    if reward_scale != 1.0 or reward_bias != 0.0:
        logging.info(f"Applying reward transformation: r' = {reward_scale} * r + {reward_bias}")
    dataset = minari.load_dataset(minari_dataset_id, download=True)

    # Get action bounds for clipping (avoid boundary values that cause NaN in log_prob)
    action_space = dataset.action_space
    action_low = action_space.low.astype(np.float32)
    action_high = action_space.high.astype(np.float32)
    clip_margin = 1e-4 * (action_high - action_low)
    action_low_clipped = action_low + clip_margin
    action_high_clipped = action_high - clip_margin

    is_antmaze = "antmaze" in minari_dataset_id.lower()
    is_pointmaze = "pointmaze" in minari_dataset_id.lower()

    # Collect all transitions
    all_states = []
    all_actions = []
    all_next_states = []
    all_next_actions = []
    all_rewards = []
    all_mc_returns = []
    all_terminations = []
    all_truncations = []
    episode_starts = []
    episode_ends = []

    num_episodes = dataset.total_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    current_idx = 0
    for ep_idx in range(num_episodes):
        if ep_idx % 100 == 0:
            logging.info(f"Processing episode {ep_idx}/{num_episodes}")

        episode = dataset[ep_idx]
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        truncations = episode.truncations

        if isinstance(observations, dict):
            obs_arrays = [observations[k] for k in sorted(observations.keys())]
            observations = np.concatenate(obs_arrays, axis=-1)

        if is_antmaze:
            # # For antmaze: truncate episode after first positive reward
            # positive_reward_idx = np.where(rewards > 0)[0]
            # if len(positive_reward_idx) > 0:
            #     end_idx = positive_reward_idx[0] + 1
            #     observations = observations[: end_idx + 1]
            #     actions = actions[:end_idx]
            #     rewards = rewards[:end_idx]
            #     terminations = terminations[:end_idx]
            #     truncations = truncations[:end_idx]

            # For antmaze: termination = (reward == 1), truncation = False
            terminations = rewards > 0
            truncations = np.zeros_like(truncations, dtype=bool)

        if is_pointmaze:
            # For pointmaze: termination = (reward == 1), same as antmaze
            terminations = rewards > 0

        # Apply reward transformation before MC return computation
        transformed_rewards = rewards * reward_scale + reward_bias

        # Compute MC returns using transformed rewards
        dones = np.logical_or(terminations, truncations)
        mc_returns = rl_utils.compute_mc_returns(transformed_rewards, dones, discount)

        # Process each transition (T+1 observations, T actions)
        num_transitions = len(actions)
        episode_starts.append(current_idx)

        for t in range(num_transitions):
            obs = observations[t].astype(np.float32)
            next_obs = (
                observations[t + 1].astype(np.float32)
                if t + 1 < len(observations)
                else observations[-1].astype(np.float32)
            )
            action = np.clip(actions[t].astype(np.float32), action_low_clipped, action_high_clipped)
            next_action = np.clip(
                actions[t + 1].astype(np.float32) if t + 1 < len(actions) else actions[-1].astype(np.float32),
                action_low_clipped,
                action_high_clipped,
            )

            all_states.append(obs)
            all_actions.append(action)
            all_next_states.append(next_obs)
            all_next_actions.append(next_action)
            all_rewards.append(transformed_rewards[t])
            all_mc_returns.append(mc_returns[t])
            all_terminations.append(terminations[t])
            all_truncations.append(truncations[t])

        current_idx += num_transitions
        episode_ends.append(current_idx)

    logging.info(f"Loaded {num_episodes} episodes, {current_idx} transitions")

    return NumpyDataset(
        states=np.stack(all_states),
        actions=np.stack(all_actions),
        next_states=np.stack(all_next_states),
        next_actions=np.stack(all_next_actions),
        rewards=np.array(all_rewards, dtype=np.float32),
        mc_returns=np.array(all_mc_returns, dtype=np.float32),
        terminations=np.array(all_terminations, dtype=bool),
        truncations=np.array(all_truncations, dtype=bool),
        episode_starts=np.array(episode_starts, dtype=np.int64),
        episode_ends=np.array(episode_ends, dtype=np.int64),
    )


def create_numpy_dataset_from_legacy_d4rl(
    env_name: str,
    discount: float = 0.99,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: float = 0.999,
) -> NumpyDataset:
    """Create a NumpyDataset by loading data from legacy D4RL.

    This function uses the d4rl library directly and includes special handling
    for sparse reward environments (antmaze) where failed trajectories use
    reward_neg / (1 - gamma) as return-to-go.

    Args:
        env_name: D4RL environment name (e.g., 'antmaze-large-diverse-v2')
        discount: Discount factor for MC return computation
        reward_scale: Scale factor for reward transformation (r' = scale * r + bias)
        reward_bias: Bias for reward transformation
        clip_action: Action clipping margin

    Returns:
        NumpyDataset with all data in memory
    """
    # Load the dataset using legacy D4RL utilities
    if legacy_d4rl_utils is None:
        raise ImportError("legacy D4RL utils are required but not available (missing d4rl, gym, or mujoco_py)")
    data = legacy_d4rl_utils.load_legacy_d4rl_dataset(
        env_name=env_name,
        discount=discount,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        clip_action=clip_action,
    )

    # Convert to NumpyDataset format
    # Legacy D4RL doesn't have truncations separate from terminals
    truncations = np.zeros_like(data["terminals"], dtype=bool)

    return NumpyDataset(
        states=data["observations"],
        actions=data["actions"],
        next_states=data["next_observations"],
        next_actions=data["next_actions"],
        rewards=data["rewards"],
        mc_returns=data["mc_returns"],
        terminations=data["terminals"],
        truncations=truncations,
        episode_starts=data["episode_starts"],
        episode_ends=data["episode_ends"],
    )


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split: str = "train",
    shuffle: bool = False,
    return_trajectories: bool = False,
    max_trajectories: int | None = None,
) -> Dataset:
    if data_config.rlds_dataset_class == "robocoin":
        from openpi.training.robocoin_rlds_dataset import RoboCoinRldsDataset

        return RoboCoinRldsDataset(
            data_dir = data_config.rlds_data_dir,
            batch_size = batch_size,
            split = split,
            shuffle = shuffle,
            action_chunk_size = action_horizon,
            datasets = data_config.datasets,
            critic_mode = data_config.critic_mode,
            discount = data_config.discount,
            reward_scale = data_config.reward_scale,
            reward_bias = data_config.reward_bias,
            use_eef = data_config.robocoin_use_eef,
            latent_store_dir = data_config.latent_store_dir,
            latent_views = data_config.latent_views,
            counterfactual_action_store_dir = data_config.counterfactual_action_store_dir,
            max_num_demos = data_config.max_num_demos,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            **data_config.rlds_kwargs,
        )

    return DroidRldsDataset(
        data_dir = data_config.rlds_data_dir,
        batch_size = batch_size,
        shuffle = shuffle,
        action_chunk_size = action_horizon,
        action_space = data_config.action_space,
        datasets = data_config.datasets,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    """Transform the dataset by applying the data transforms.

    Args:
        dataset: The dataset to transform.
        data_config: The data configuration.
        skip_norm_stats: Whether to skip data normalization.
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build the transform pipeline
    input_transforms = list(data_config.repack_transforms.inputs)
    input_transforms.extend(data_config.data_transforms.inputs)
    input_transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    if data_config.clip_normalized_bounds is not None:
        input_transforms.append(_transforms.Clip(data_config.clip_normalized_bounds))
    input_transforms.extend(data_config.model_transforms.inputs)

    return TransformedDataset(dataset, input_transforms)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.action_horizon
    if action_horizon is None:
        action_horizon = config.model.action_horizon
    if action_horizon is None:
        raise ValueError("Action horizon must be set on either TrainConfig or the model config.")
    config_fields = {f.name: getattr(data_config, f.name) for f in dataclasses.fields(data_config)}
    if "norm_stats" in config_fields and config_fields["norm_stats"]:
        config_fields["norm_stats"] = f"<{len(config_fields['norm_stats'])} keys>"
    logging.info(f"data_config: {config_fields}")

    # Check for minari dataset first (fastest option for in-memory datasets)
    if data_config.minari_dataset_id is not None:
        return create_numpy_data_loader(
            data_config,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            seed=config.seed,
            framework=framework,
        )

    # Check for legacy D4RL dataset (alternative to Minari)
    if data_config.legacy_d4rl_env_name is not None:
        return create_numpy_data_loader(
            data_config,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            seed=config.seed,
            framework=framework,
        )

    # Check for RoboCOIN dataset (DLIMP-based image+text+state loader)
    if data_config.robocoin_data_config is not None:
        import dataclasses as dc

        from openpi.models.tokenizer import create_tokenizer

        robocoin_config = dc.replace(
            data_config.robocoin_data_config,
            action_horizon = action_horizon,
        )
        data_config = dc.replace(data_config, robocoin_data_config = robocoin_config)

        # num_images unused for PaliGemma, used for Gemma3
        num_images = robocoin_config.max_cameras if config.backbone_variant == "gemma3" else 0
        tokenizer = create_tokenizer(config.backbone_variant, robocoin_config.max_token_len, num_images = num_images)
        return create_robocoin_data_loader(
            data_config,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            seed=config.seed,
            framework=framework,
            tokenizer=tokenizer,
        )

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, split = "train", shuffle = shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_robocoin_data_loader(
    data_config: _config.DataConfig,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    seed: int = 0,
    framework: str = "jax",
    tokenizer = None,
) -> DataLoader:
    """Create a DLIMP-based data loader for RoboCOIN image+text+state data.

    This loader is optimized for the RoboCOIN dataset which contains:
    - Camera images (up to 3 views)
    - Text prompts (subtask descriptions)
    - Proprioceptive state
    - Actions and rewards

    Args:
        data_config: The data configuration (must have robocoin_data_config set).
        batch_size: The batch size.
        sharding: The sharding to use for the data loader.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        seed: Random seed for shuffling.
        framework: The framework to use ("jax" or "pytorch").

    Returns:
        DataLoader wrapping the RoboCOIN DLIMP dataset.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RoboCOIN data loader is not supported yet")

    from openpi.training.robocoin_data_loader import RoboCOINDataLoader

    # Get the RoboCOIN loader config from data_config
    robocoin_config = data_config.robocoin_data_config

    # For distributed training, divide batch_size by the number of hosts.
    # local_shuffle_buffer_size is already per-host (specified directly in config).
    process_count = jax.process_count()
    local_batch_size = batch_size // process_count
    
    if process_count > 1:
        # Set TensorFlow random seed per host
        tf.random.set_seed(seed + jax.process_index())
        logging.info(
            f"Distributed training: {process_count} hosts, "
            f"local_batch_size={local_batch_size} (global={batch_size}), "
            f"local_shuffle_buffer_size={robocoin_config.local_shuffle_buffer_size}"
        )

    # Update config with batch_size, shuffle, and normalization settings
    import dataclasses as dc
    robocoin_config = dc.replace(
        robocoin_config,
        batch_size=local_batch_size,
        shuffle=shuffle,
        seed=seed + jax.process_index(),  # Different seed per host
        state_norm_stats=data_config.norm_stats,
        use_quantile_norm=data_config.use_quantile_norm,
    )

    # Create the RoboCOIN data loader with sharding.
    # Pass model_transforms for per-sample application (e.g. TokenizePrompt, PadStatesAndActions)
    # before sharding — same split-transform-restack pattern as RLDS/DROID pipelines.
    model_transforms_list = list(data_config.model_transforms.inputs) if data_config.model_transforms.inputs else None
    robocoin_loader = RoboCOINDataLoader(
        robocoin_config,
        sharding=sharding,
        num_batches=num_batches,
        tokenizer=tokenizer,
        model_transforms=model_transforms_list,
    )

    return DataLoaderImpl(data_config, robocoin_loader)


def create_numpy_data_loader(
    data_config: _config.DataConfig,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader:
    """Create a numpy-based data loader for fast in-memory data loading.

    This is optimized for state-only datasets (like D4RL/Minari) that fit in memory.
    It bypasses disk I/O by loading all data into numpy arrays upfront.

    Args:
        data_config: The data configuration (must have minari_dataset_id or legacy_d4rl_env_name set).
        batch_size: The batch size.
        sharding: The sharding to use for the data loader.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        seed: Random seed for shuffling.
        framework: The framework to use ("jax" or "pytorch").

    Returns:
        DataLoader wrapping the in-memory numpy dataset.
    """
    # Create the in-memory dataset from either Minari or legacy D4RL
    if data_config.minari_dataset_id is not None:
        base_dataset = create_numpy_dataset_from_minari(
            data_config.minari_dataset_id,
            discount=data_config.discount,
            reward_scale=data_config.reward_scale,
            reward_bias=data_config.reward_bias,
        )
    elif data_config.legacy_d4rl_env_name is not None:
        base_dataset = create_numpy_dataset_from_legacy_d4rl(
            data_config.legacy_d4rl_env_name,
            discount=data_config.discount,
            reward_scale=data_config.reward_scale,
            reward_bias=data_config.reward_bias,
        )
    else:
        raise ValueError("Either minari_dataset_id or legacy_d4rl_env_name must be set to use numpy data loader")

    # Wrap with multi-transition sampler if configured
    if data_config.num_transitions_per_sample is not None:
        num_transitions = data_config.num_transitions_per_sample
        rng = np.random.default_rng(seed)

        # Create appropriate sampler
        if data_config.multi_transition_sampler_type == "uniform":
            sampler_config = samplers.UniformRandomSamplerConfig(num_transitions_per_sample=num_transitions)
        elif data_config.multi_transition_sampler_type == "trajectory_uniform":
            sampler_config = samplers.TrajectoryUniformSamplerConfig(num_transitions_per_sample=num_transitions)
        elif data_config.multi_transition_sampler_type == "trajectory_ordered":
            sampler_config = samplers.TrajectoryOrderedSamplerConfig(num_transitions_per_sample=num_transitions)
        elif data_config.multi_transition_sampler_type == "trajectory_consecutive":
            sampler_config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=num_transitions)
        else:
            raise ValueError(f"Unknown multi_transition_sampler_type: {data_config.multi_transition_sampler_type}")

        sampler = sampler_config.create(base_dataset, rng)
        dataset = MultiTransitionDataset(base_dataset, sampler)
        logging.info(f"Multi-transition mode: n={num_transitions}, sampler={data_config.multi_transition_sampler_type}")
    else:
        dataset = base_dataset

    # Apply transforms.
    # We apply repack transforms (usually empty for numpy loader), data transforms, and normalization.
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Compute local batch size
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"numpy data loader: local_batch_size={local_batch_size}, total_samples={len(dataset)}")

    # Create weighted sampler if reward_1_upsample_weight > 1.0
    sampler = None
    if data_config.reward_1_upsample_weight > 1.0:
        # Compute sample weights based on reward (original reward=1 transitions get higher weight)
        # After transformation: r' = scale * r + bias, so original r=1 becomes scale * 1 + bias
        transformed_reward_1 = data_config.reward_scale * 1.0 + data_config.reward_bias
        reward_1_mask = np.isclose(base_dataset.rewards, transformed_reward_1)
        num_reward_1 = np.sum(reward_1_mask)
        num_other = len(base_dataset) - num_reward_1
        logging.info(
            f"Creating weighted sampler: {num_reward_1} reward=1 samples (transformed to {transformed_reward_1}), "
            f"{num_other} other samples, weight={data_config.reward_1_upsample_weight}"
        )

        # Weights: reward=1 transitions get higher weight
        weights = np.where(reward_1_mask, data_config.reward_1_upsample_weight, 1.0)
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights_tensor,
            num_samples=len(dataset),
            replacement=True,
        )

    # Create default sampler if no specific sampler was created
    if sampler is None:
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    # Wrap in BatchSampler for vectorized loading
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=local_batch_size, drop_last=True)

    # Use TorchDataLoader for batching (reuses existing infrastructure)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=None,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=False,  # Handled by sampler
        sampler=batch_sampler,  # Pass BatchSampler as sampler to disable auto-collation
        batch_sampler=None,
        num_batches=num_batches,
        num_workers=0,  # No workers needed for in-memory data
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int | None,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process. Can be None if using custom sampler/batching.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if local_batch_size is not None and len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)

        # Select collate_fn based on batching mode
        # If we are batching (via batch_size > 0 or batch_sampler), we use _collate_fn to stack samples.
        # If batching is disabled (local_batch_size is None), we use identity (pass through what dataset returns).
        # This supports both standard loading (batching enabled) and vectorized loading (batching handled by dataset).
        should_collate = (local_batch_size is not None) or (batch_sampler is not None)

        loader_kwargs = {
            "dataset": dataset,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "multiprocessing_context": mp_context,
            "generator": generator,
            "collate_fn": _collate_fn if should_collate else lambda x: x,
            "worker_init_fn": _worker_init_fn,
        }

        if batch_sampler is not None:
            loader_kwargs["batch_sampler"] = batch_sampler
        else:
            loader_kwargs["batch_size"] = local_batch_size
            loader_kwargs["shuffle"] = shuffle if sampler is None else False
            loader_kwargs["sampler"] = sampler
            loader_kwargs["drop_last"] = sampler is None

        self._data_loader = torch.utils.data.DataLoader(**loader_kwargs)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around RLDS data loaders to make them compatible with openpi.

    All batching already happens in the RLDS dataset, so we don't need to do anything here.
    Supports multi-host training - each process loads its shard and batches are combined.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset | rlds_dataset.BaseRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        dataset_size: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches
        self._dataset_size = dataset_size

        if sharding is None:
            # Use data parallel sharding by default across all devices (including multi-host).
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    @property
    def dataset_size(self) -> int | None:
        """Return the number of samples in the dataset, or None if not precomputed."""
        return self._dataset_size

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                batch = {
                    key: value
                    for key, value in batch.items()
                    if not np.issubdtype(np.asarray(value).dtype, np.str_)
                    and not np.issubdtype(np.asarray(value).dtype, np.bytes_)
                }
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def dataset_size(self) -> int | None:
        """Return the number of samples in the dataset, or None if unknown."""
        return self._data_loader.dataset_size

    def __iter__(self):
        for batch in self._data_loader:
            if self._data_config.critic_mode:
                # Critic mode: yield raw batch dict for value function training
                yield batch
            else:
                # Policy mode: yield (Observation, Actions) tuple
                yield _model.Observation.from_dict(batch), batch["actions"]
