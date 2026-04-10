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
REAL_HANG_SUBTASK_ONLY_CONFIG_NAME = "real_hang_paligemma_q_sarsa_subtask_only"
REAL_HANG_ALL_SUBTASKS_CONFIG_NAME = "real_hang_paligemma_q_sarsa_all_subtasks"
REAL_HANG_ALL_SUBTASKS_PREDICT_CONFIG_NAME = "real_hang_paligemma_q_sarsa_all_subtasks_predict_current_subtask"
VARIABLE_HORIZON_CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_variable_horizon"


def _make_train_config(config_name: str, fine_tune: str | None = None) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    if fine_tune is not None:
        ft_config = _config.get_fine_tune_config(fine_tune)
        config = ft_config.apply_overrides(config, pretrained_step = None)
    return dataclasses.replace(config, batch_size = 256, seed = SEED, num_workers = 0)


def _action_horizon(config: _config.TrainConfig) -> int:
    action_horizon = config.action_horizon
    if action_horizon is None:
        action_horizon = config.model.action_horizon
    if action_horizon is None:
        raise ValueError("Action horizon must be set on either TrainConfig or the model config.")
    return action_horizon


def _make_rlds_raw_loader(config_name: str, fine_tune: str | None = None):
    config = _make_train_config(config_name, fine_tune = fine_tune)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.critic_mode:
        dataset = _data_loader.create_rlds_dataset(
            data_config,
            _action_horizon(config),
            config.batch_size,
            split = "train",
            shuffle = False,
        )
        dataset = _data_loader.transform_iterable_dataset(dataset, data_config, is_batched = True)
        return _data_loader.RLDSDataLoader(dataset, num_batches = 12000), data_config
    else:
        loader = _data_loader.create_data_loader(config, shuffle = False, num_batches = 12000)
        return loader, data_config


def _print_batch_structure(batch, path: str = "") -> None:
    if isinstance(batch, tuple):
        for i, item in enumerate(batch):
            _print_batch_structure(item, path = f"{path}[{i}]" if path else f"[{i}]")
        return
    if isinstance(batch, dict):
        for key in sorted(batch.keys()):
            _print_batch_structure(batch[key], path = f"{path}.{key}" if path else key)
        return
    if hasattr(batch, "__dataclass_fields__"):
        for field_name in batch.__dataclass_fields__:
            _print_batch_structure(getattr(batch, field_name), path = f"{path}.{field_name}" if path else field_name)
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
@pytest.mark.parametrize(
    "config_name",
    [
        PI05_RLDS_CONFIG_NAME,
        SARSA_RLDS_CONFIG_NAME,
        CQL_RLDS_CONFIG_NAME,
        REAL_HANG_SUBTASK_ONLY_CONFIG_NAME,
        REAL_HANG_ALL_SUBTASKS_CONFIG_NAME,
        REAL_HANG_ALL_SUBTASKS_PREDICT_CONFIG_NAME,
        VARIABLE_HORIZON_CONFIG_NAME,
    ],
)
def test_robocoin_rlds_batch_structure(config_name: str):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    loader, data_config = _make_rlds_raw_loader(config_name)
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print(f"\nRLDS batch keys and shapes (config={config_name}):")
            _print_batch_structure(batch)
            if data_config.critic_mode:
                state = np.asarray(batch["state"], dtype = np.float32)
            else:
                state = np.asarray(batch[0].state, dtype = np.float32)
            print(f"\n  state shape: {state.shape}")
            print(f"  state mean (per-dim): {np.mean(state, axis = 0)}")
            print(f"  state min  (per-dim): {np.min(state, axis = 0)}")
            print(f"  state max  (per-dim): {np.max(state, axis = 0)}")

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}")

        if data_config.critic_mode:
            # _assert_normalized_bounds(batch, "state")
            # _assert_normalized_bounds(batch, "actions")

            action_mask = np.asarray(batch["action_mask"], dtype = np.int32)
            next_action_mask = np.asarray(batch["next_action_mask"], dtype = np.int32)
            termination = np.asarray(batch["termination"], dtype = bool)
            assert np.all(action_mask >= next_action_mask), "Expected action_mask >= next_action_mask elementwise"
            assert np.all(termination | np.any(next_action_mask > 0, axis = -1)), (
                "Expected every non-terminal datapoint to have at least one valid next action"
            )

            if np.any(np.asarray(batch["steps_to_subtask_end"]) < 15):
                import ipdb; ipdb.set_trace()  # noqa: T100


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required = True)
    parser.add_argument("--fine-tune", default = None)
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    config_name = args.config_name
    loader, data_config = _make_rlds_raw_loader(config_name, fine_tune = args.fine_tune)
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print(f"\nRLDS batch keys and shapes (config={config_name}, fine_tune={args.fine_tune}):")
            _print_batch_structure(batch)
            if data_config.critic_mode:
                state = np.asarray(batch["state"], dtype = np.float32)
            else:
                state = np.asarray(batch[0].state, dtype = np.float32)
            print(f"\n  state shape: {state.shape}")
            print(f"  state mean (per-dim): {np.mean(state, axis = 0)}")
            print(f"  state min  (per-dim): {np.min(state, axis = 0)}")
            print(f"  state max  (per-dim): {np.max(state, axis = 0)}")

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}")

        if data_config.critic_mode:
            action_mask = np.asarray(batch["action_mask"], dtype = np.int32)
            next_action_mask = np.asarray(batch["next_action_mask"], dtype = np.int32)
            termination = np.asarray(batch["termination"], dtype = bool)
            assert np.all(action_mask >= next_action_mask), "Expected action_mask >= next_action_mask elementwise"
            assert np.all(termination | np.any(next_action_mask > 0, axis = -1)), (
                "Expected every non-terminal datapoint to have at least one valid next action"
            )

            if np.any(np.asarray(batch["steps_to_subtask_end"]) < 15):
                import ipdb; ipdb.set_trace()  # noqa: T100
