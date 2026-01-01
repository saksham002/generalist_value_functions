import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_numpy_dataset_reward_transformation():
    """Test that reward_scale and reward_bias are applied correctly to NumpyDataset."""
    import numpy as np

    from openpi.shared import rl_utils

    # Create a NumpyDataset with known values
    states = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    actions = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    raw_rewards = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    terminations = np.array([False, False, False, True], dtype=bool)
    truncations = np.array([False, False, False, False], dtype=bool)

    # Apply transformation: r' = 0.5 * r - 0.25
    reward_scale = 0.5
    reward_bias = -0.25
    transformed_rewards = raw_rewards * reward_scale + reward_bias

    # Compute MC returns from transformed rewards
    discount = 0.99
    dones = np.logical_or(terminations, truncations)
    mc_returns = rl_utils.compute_mc_returns(transformed_rewards, dones, discount)

    # Create dataset with transformed values
    dataset = _data_loader.NumpyDataset(
        states=states,
        actions=actions,
        next_states=np.roll(states, -1, axis=0),
        next_actions=np.roll(actions, -1, axis=0),
        rewards=transformed_rewards,
        mc_returns=mc_returns,
        terminations=terminations,
        truncations=truncations,
        episode_starts=np.array([0], dtype=np.int64),
        episode_ends=np.array([4], dtype=np.int64),
    )

    # Verify rewards are transformed
    np.testing.assert_allclose(dataset[0]["reward"], 0.5 * 1.0 - 0.25)  # 0.25
    np.testing.assert_allclose(dataset[1]["reward"], 0.5 * 0.0 - 0.25)  # -0.25
    np.testing.assert_allclose(dataset[2]["reward"], 0.5 * 1.0 - 0.25)  # 0.25
    np.testing.assert_allclose(dataset[3]["reward"], 0.5 * 0.0 - 0.25)  # -0.25

    # Verify MC returns are computed from transformed rewards
    # At t=3 (terminal): mc_return = -0.25
    np.testing.assert_allclose(dataset[3]["mc_return"], -0.25, rtol=1e-5)
    # At t=2: mc_return = 0.25 + 0.99 * (-0.25) = 0.25 - 0.2475 = 0.0025
    np.testing.assert_allclose(dataset[2]["mc_return"], 0.25 + 0.99 * (-0.25), rtol=1e-5)


def test_data_config_reward_transformation_defaults():
    """Test that reward transformation defaults are identity (no change)."""
    config = _config.DataConfig()
    assert config.reward_scale == 1.0
    assert config.reward_bias == 0.0
