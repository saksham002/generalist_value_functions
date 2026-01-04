"""Multi-transition sampling strategies for batch value learning.

Samplers return indices for multiple transitions that form a single sample.
This enables value functions that jointly predict values for multiple states.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openpi.training.data_loader import NumpyDataset


class Sampler(abc.ABC):
    """Base sampler interface."""

    @abc.abstractmethod
    def sample(self) -> np.ndarray:
        """Return indices for num_transitions_per_sample transitions."""


@dataclasses.dataclass(frozen=True)
class SamplerConfig(abc.ABC):
    """Base configuration for multi-transition samplers."""

    num_transitions_per_sample: int

    @abc.abstractmethod
    def create(self, dataset: NumpyDataset, rng: np.random.Generator) -> Sampler:
        """Create a sampler instance."""


class UniformRandomSampler(Sampler):
    """Sample n random transitions uniformly from the entire buffer."""

    def __init__(
        self,
        dataset: NumpyDataset,
        num_transitions_per_sample: int,
        rng: np.random.Generator,
    ):
        self._dataset = dataset
        self._num_transitions_per_sample = num_transitions_per_sample
        self._rng = rng
        self._dataset_size = len(dataset)

    def sample(self) -> np.ndarray:
        return self._rng.choice(
            self._dataset_size,
            size=self._num_transitions_per_sample,
            replace=False,
        )


@dataclasses.dataclass(frozen=True)
class UniformRandomSamplerConfig(SamplerConfig):
    """Sample n random transitions from entire buffer."""

    def create(self, dataset: NumpyDataset, rng: np.random.Generator) -> UniformRandomSampler:
        return UniformRandomSampler(dataset, self.num_transitions_per_sample, rng)


class TrajectoryUniformSampler(Sampler):
    """Sample n random transitions from a single randomly-chosen episode."""

    def __init__(
        self,
        dataset: NumpyDataset,
        num_transitions_per_sample: int,
        rng: np.random.Generator,
    ):
        self._dataset = dataset
        self._num_transitions_per_sample = num_transitions_per_sample
        self._rng = rng
        self._num_episodes = dataset.num_episodes

        # Pre-compute episode lengths for rejection sampling
        self._episode_lengths = dataset.episode_ends - dataset.episode_starts
        # Only consider episodes with enough transitions
        self._valid_episodes = np.where(self._episode_lengths >= num_transitions_per_sample)[0]
        if len(self._valid_episodes) == 0:
            raise ValueError(
                f"No episodes have at least {num_transitions_per_sample} transitions. "
                f"Max episode length: {self._episode_lengths.max()}"
            )

    def sample(self) -> np.ndarray:
        # Sample a valid episode
        episode_idx = self._rng.choice(self._valid_episodes)
        start = self._dataset.episode_starts[episode_idx]
        end = self._dataset.episode_ends[episode_idx]

        # Sample n transitions within this episode
        episode_indices = np.arange(start, end)
        return self._rng.choice(
            episode_indices,
            size=self._num_transitions_per_sample,
            replace=False,
        )


@dataclasses.dataclass(frozen=True)
class TrajectoryUniformSamplerConfig(SamplerConfig):
    """Sample n random transitions from a single episode."""

    def create(self, dataset: NumpyDataset, rng: np.random.Generator) -> TrajectoryUniformSampler:
        return TrajectoryUniformSampler(dataset, self.num_transitions_per_sample, rng)


class TrajectoryOrderedSampler(Sampler):
    """Sample n random transitions from a single episode, sorted by timestep."""

    def __init__(
        self,
        dataset: NumpyDataset,
        num_transitions_per_sample: int,
        rng: np.random.Generator,
    ):
        self._dataset = dataset
        self._num_transitions_per_sample = num_transitions_per_sample
        self._rng = rng

        # Pre-compute episode lengths for rejection sampling
        self._episode_lengths = dataset.episode_ends - dataset.episode_starts
        self._valid_episodes = np.where(self._episode_lengths >= num_transitions_per_sample)[0]
        if len(self._valid_episodes) == 0:
            raise ValueError(
                f"No episodes have at least {num_transitions_per_sample} transitions. "
                f"Max episode length: {self._episode_lengths.max()}"
            )

    def sample(self) -> np.ndarray:
        # Sample a valid episode
        episode_idx = self._rng.choice(self._valid_episodes)
        start = self._dataset.episode_starts[episode_idx]
        end = self._dataset.episode_ends[episode_idx]

        # Sample n transitions and sort by index (which equals timestep order)
        episode_indices = np.arange(start, end)
        selected = self._rng.choice(
            episode_indices,
            size=self._num_transitions_per_sample,
            replace=False,
        )
        return np.sort(selected)


@dataclasses.dataclass(frozen=True)
class TrajectoryOrderedSamplerConfig(SamplerConfig):
    """Sample n transitions from one episode, sorted by timestep."""

    def create(self, dataset: NumpyDataset, rng: np.random.Generator) -> TrajectoryOrderedSampler:
        return TrajectoryOrderedSampler(dataset, self.num_transitions_per_sample, rng)
