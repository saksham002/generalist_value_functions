import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import openpi.shared.legacy_d4rl_utils as legacy_d4rl_utils
from openpi.training import data_loader as _data_loader
from openpi.training import config as _config

def test_calc_return_to_go_sparse():
    # Test standard case
    rewards = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    masks = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    gamma = 0.9
    reward_neg = 0.0
    
    returns = legacy_d4rl_utils.calc_return_to_go_sparse(
        rewards, masks, gamma, reward_neg, is_sparse_reward=True
    )
    
    # t=2: 1.0 + 0.9 * 0 * 0 = 1.0
    # t=1: 0.0 + 0.9 * 1.0 * 1 = 0.9
    # t=0: 1.0 + 0.9 * 0.9 * 1 = 1.81
    expected = np.array([1.81, 0.9, 1.0], dtype=np.float32)
    np.testing.assert_allclose(returns, expected, rtol=1e-5)

    # Test entirely failed trajectory in sparse reward env
    rewards = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    masks = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    returns = legacy_d4rl_utils.calc_return_to_go_sparse(
        rewards, masks, gamma, reward_neg, is_sparse_reward=True
    )
    # Should be reward_neg / (1 - gamma) = 0.0
    np.testing.assert_allclose(returns, np.zeros(3), rtol=1e-5)
    
    # Test with negative reward_neg
    reward_neg = -1.0
    rewards = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    returns = legacy_d4rl_utils.calc_return_to_go_sparse(
        rewards, masks, gamma, reward_neg, is_sparse_reward=True
    )
    # Should be -1.0 / (1 - 0.9) = -10.0
    np.testing.assert_allclose(returns, np.full(3, -10.0), rtol=1e-5)

@patch("openpi.shared.legacy_d4rl_utils.d4rl")
@patch("openpi.shared.legacy_d4rl_utils.gym")
def test_load_legacy_d4rl_dataset(mock_gym, mock_d4rl):
    if mock_d4rl is None:
        pytest.skip("d4rl not available for mocking")
        
    # Mock gym and d4rl
    mock_env = MagicMock()
    mock_gym.make.return_value = mock_env
    
    mock_dataset = {
        "observations": np.zeros((5, 2)),
        "actions": np.zeros((5, 1)),
        "next_observations": np.zeros((5, 2)),
        "rewards": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        "terminals": np.array([0.0, 0.0, 1.0, 0.0, 1.0]),
    }
    mock_d4rl.qlearning_dataset.return_value = mock_dataset
    
    dataset = legacy_d4rl_utils.load_legacy_d4rl_dataset("test-env")
    
    assert "mc_returns" in dataset
    assert len(dataset["episode_starts"]) == 2
    assert dataset["episode_starts"][0] == 0
    assert dataset["episode_ends"][0] == 2
    assert dataset["episode_starts"][1] == 2
    assert dataset["episode_ends"][1] == 4

def test_create_numpy_dataset_from_legacy_d4rl():
    with patch("openpi.shared.legacy_d4rl_utils.load_legacy_d4rl_dataset") as mock_load:
        mock_load.return_value = {
            "observations": np.zeros((2, 2)),
            "actions": np.zeros((2, 1)),
            "next_observations": np.zeros((2, 2)),
            "next_actions": np.zeros((2, 1)),
            "rewards": np.zeros(2),
            "mc_returns": np.zeros(2),
            "terminals": np.zeros(2, dtype=bool),
            "episode_starts": np.array([0]),
            "episode_ends": np.array([2]),
        }
        
        ds = _data_loader.create_numpy_dataset_from_legacy_d4rl("test-env")
        assert isinstance(ds, _data_loader.NumpyDataset)
        assert len(ds) == 2
