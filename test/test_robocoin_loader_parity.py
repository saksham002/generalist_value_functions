import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


SEED = 86
SARSA_RLDS_CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_chunk_wise_rlds"
PI05_RLDS_CONFIG_NAME = "robocoin_bimanual_pi05_rlds"
CQL_RLDS_CONFIG_NAME = "robocoin_bimanual_paligemma_cql_rlds"


def _make_train_config(config_name: str) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    return dataclasses.replace(config, batch_size = 256, seed = SEED, num_workers = 0)


def _action_horizon(config: _config.TrainConfig) -> int:
    action_horizon = config.action_horizon
    if action_horizon is None:
        action_horizon = config.model.action_horizon
    if action_horizon is None:
        raise ValueError("Action horizon must be set on either TrainConfig or the model config.")
    return action_horizon


def _make_rlds_raw_loader(config_name: str):
    config = _make_train_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_rlds_dataset(
        data_config,
        _action_horizon(config),
        config.batch_size,
        split = "train",
        shuffle = False,
    )
    dataset = _data_loader.transform_iterable_dataset(dataset, data_config, is_batched = True)
    return _data_loader.RLDSDataLoader(dataset, num_batches = 1)


def _print_batch_structure(batch, path: str = "") -> None:
    if isinstance(batch, dict):
        for key in sorted(batch.keys()):
            _print_batch_structure(batch[key], path = f"{path}.{key}" if path else key)
        return
    arr = np.asarray(batch)
    print(f"  {path}: shape={arr.shape}, dtype={arr.dtype}")


def _assert_normalized_bounds(batch: dict, key: str) -> None:
    values = np.asarray(batch[key], dtype = np.float32)
    abs_values = np.abs(values)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    fraction_above_one = float(np.mean(abs_values > 1.0))

    print(
        f"  stats[{key}]: min={min_value:.4f}, max={max_value:.4f}, "
        f"frac_abs_gt_1={fraction_above_one:.4%}"
    )

    assert np.max(abs_values) < 1.30, f"{key} max absolute value out of range: {np.max(abs_values)}"
    assert fraction_above_one < 0.05, f"{key} has too many values with |x| > 1: {fraction_above_one:.4%}"


@pytest.mark.manual
@pytest.mark.parametrize("config_name", [PI05_RLDS_CONFIG_NAME, SARSA_RLDS_CONFIG_NAME, CQL_RLDS_CONFIG_NAME])
def test_robocoin_rlds_batch_structure(config_name: str):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    loader = _make_rlds_raw_loader(config_name)
    batch = next(iter(loader))

    print(f"\nRLDS batch keys and shapes (config={config_name}):")
    _print_batch_structure(batch)
    state = np.asarray(batch["state"], dtype = np.float32)
    print(f"\n  state shape: {state.shape}")
    print(f"  state mean (per-dim): {np.mean(state, axis = 0)}")
    print(f"  state min  (per-dim): {np.min(state, axis = 0)}")
    print(f"  state max  (per-dim): {np.max(state, axis = 0)}")

    _assert_normalized_bounds(batch, "state")
    _assert_normalized_bounds(batch, "actions")
