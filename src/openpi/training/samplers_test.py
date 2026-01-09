"""Tests for multi-transition samplers."""

import numpy as np
import pytest

from openpi.training import data_loader as _data_loader
from openpi.training import samplers


def make_test_dataset(num_episodes: int = 5, transitions_per_episode: int = 20) -> _data_loader.NumpyDataset:
    """Create a test NumpyDataset with known structure."""
    total_transitions = num_episodes * transitions_per_episode
    state_dim = 4
    action_dim = 2

    episode_starts = np.arange(0, total_transitions, transitions_per_episode)
    episode_ends = episode_starts + transitions_per_episode

    return _data_loader.NumpyDataset(
        states=np.random.randn(total_transitions, state_dim).astype(np.float32),
        actions=np.random.randn(total_transitions, action_dim).astype(np.float32),
        next_states=np.random.randn(total_transitions, state_dim).astype(np.float32),
        next_actions=np.random.randn(total_transitions, action_dim).astype(np.float32),
        rewards=np.random.randn(total_transitions).astype(np.float32),
        mc_returns=np.random.randn(total_transitions).astype(np.float32),
        terminations=np.zeros(total_transitions, dtype=bool),
        truncations=np.zeros(total_transitions, dtype=bool),
        episode_starts=episode_starts,
        episode_ends=episode_ends,
    )


class TestUniformRandomSampler:
    def test_returns_correct_number_of_indices(self):
        dataset = make_test_dataset()
        config = samplers.UniformRandomSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert len(indices) == 8

    def test_indices_are_unique(self):
        dataset = make_test_dataset()
        config = samplers.UniformRandomSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert len(set(indices)) == 8

    def test_indices_are_valid(self):
        dataset = make_test_dataset()
        config = samplers.UniformRandomSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert all(0 <= idx < len(dataset) for idx in indices)


class TestTrajectoryUniformSampler:
    def test_returns_correct_number_of_indices(self):
        dataset = make_test_dataset()
        config = samplers.TrajectoryUniformSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert len(indices) == 8

    def test_indices_from_same_episode(self):
        dataset = make_test_dataset(transitions_per_episode=20)
        config = samplers.TrajectoryUniformSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()

        # Find which episode each index belongs to
        episode_ids = set()
        for idx in indices:
            for ep_idx in range(dataset.num_episodes):
                start = dataset.episode_starts[ep_idx]
                end = dataset.episode_ends[ep_idx]
                if start <= idx < end:
                    episode_ids.add(ep_idx)
                    break

        assert len(episode_ids) == 1

    def test_raises_when_episodes_too_short(self):
        dataset = make_test_dataset(transitions_per_episode=5)
        config = samplers.TrajectoryUniformSamplerConfig(num_transitions_per_sample=10)

        with pytest.raises(ValueError, match="No episodes have at least"):
            config.create(dataset, np.random.default_rng(42))


class TestTrajectoryOrderedSampler:
    def test_returns_correct_number_of_indices(self):
        dataset = make_test_dataset()
        config = samplers.TrajectoryOrderedSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert len(indices) == 8

    def test_indices_are_sorted(self):
        dataset = make_test_dataset()
        config = samplers.TrajectoryOrderedSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert list(indices) == sorted(indices)

    def test_indices_from_same_episode(self):
        dataset = make_test_dataset(transitions_per_episode=20)
        config = samplers.TrajectoryOrderedSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()

        # All indices should be from same episode
        episode_ids = set()
        for idx in indices:
            for ep_idx in range(dataset.num_episodes):
                start = dataset.episode_starts[ep_idx]
                end = dataset.episode_ends[ep_idx]
                if start <= idx < end:
                    episode_ids.add(ep_idx)
                    break

        assert len(episode_ids) == 1


class TestTrajectoryConsecutiveSampler:
    def test_returns_correct_number_of_indices(self):
        dataset = make_test_dataset()
        config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert len(indices) == 8

    def test_indices_are_consecutive(self):
        dataset = make_test_dataset()
        config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        for _ in range(10):
            indices = sampler.sample()
            # Consecutive means each index is at most 1 greater than previous
            diffs = np.diff(indices)
            assert np.all((diffs == 0) | (diffs == 1))

    def test_indices_from_same_episode(self):
        dataset = make_test_dataset(transitions_per_episode=20)
        config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        for _ in range(10):
            indices = sampler.sample()

            # All indices should be from same episode
            episode_ids = set()
            for idx in indices:
                for ep_idx in range(dataset.num_episodes):
                    start = dataset.episode_starts[ep_idx]
                    end = dataset.episode_ends[ep_idx]
                    if start <= idx < end:
                        episode_ids.add(ep_idx)
                        break

            assert len(episode_ids) == 1

    def test_pads_with_last_transition_at_episode_end(self):
        dataset = make_test_dataset(num_episodes=1, transitions_per_episode=10)
        config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=8)

        # Use a fixed seed and sample many times to hit end-of-episode cases
        rng = np.random.default_rng(42)
        sampler = config.create(dataset, rng)

        found_padded_sample = False
        for _ in range(100):
            indices = sampler.sample()
            # Check if sample extends to episode end with padding
            if indices[-1] == 9:  # Last valid index is 9 (end - 1)
                # Count how many times the last index appears
                last_idx_count = np.sum(indices == 9)
                if last_idx_count > 1:
                    found_padded_sample = True
                    # Verify padding structure: should be increasing then constant
                    break

        assert found_padded_sample, "Should have found a padded sample near episode end"

    def test_works_with_short_episodes(self):
        # Episodes shorter than num_transitions_per_sample should still work
        dataset = make_test_dataset(num_episodes=5, transitions_per_episode=5)
        config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=8)
        sampler = config.create(dataset, np.random.default_rng(42))

        indices = sampler.sample()
        assert len(indices) == 8
        # All indices should be valid (within dataset bounds)
        assert np.all(indices >= 0)
        assert np.all(indices < len(dataset))

    def test_all_transitions_can_be_first_in_sample(self):
        # Verify uniform coverage - all transitions can be sampled as first element
        dataset = make_test_dataset(num_episodes=1, transitions_per_episode=10)
        config = samplers.TrajectoryConsecutiveSamplerConfig(num_transitions_per_sample=4)
        sampler = config.create(dataset, np.random.default_rng(42))

        first_indices_seen = set()
        for _ in range(500):
            indices = sampler.sample()
            first_indices_seen.add(indices[0])

        # Should be able to start from any of the 10 positions
        assert first_indices_seen == set(range(10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
