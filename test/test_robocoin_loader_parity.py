import dataclasses
import pathlib

import numpy as np
import pytest
import tensorflow as tf

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import ipdb


SEED = 86
NUM_BATCHES_TO_SCAN = 100
SARSA_RLDS_CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_chunk_wise_rlds"
PI05_RLDS_CONFIG_NAME = "robocoin_bimanual_pi05_rlds"
REAL_HANG_PI05_RLDS_CONFIG_NAME = "real_hang_pi05_filter_intervention"
CQL_RLDS_CONFIG_NAME = "robocoin_bimanual_paligemma_cql_rlds"
REAL_HANG_SUBTASK_ONLY_CONFIG_NAME = "real_hang_paligemma_q_sarsa_subtask_only"
REAL_HANG_ALL_SUBTASKS_CONFIG_NAME = "real_hang_paligemma_q_sarsa_all_subtasks"
REAL_HANG_ALL_SUBTASKS_PREDICT_CONFIG_NAME = "real_hang_paligemma_q_sarsa_all_subtasks_predict_current_subtask"
VARIABLE_HORIZON_CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_variable_horizon"
REAL_HANG_60_HZ_CONFIG_NAME = "real_hang_pi05_filter_intervention_60_Hz"
GEMMA4_Q_SARSA_CONFIG_NAME = "robocoin_bimanual_gemma4_q_sarsa"
DEBUG_DIR = pathlib.Path("test/debug")


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
        return _data_loader.RLDSDataLoader(dataset, num_batches = NUM_BATCHES_TO_SCAN), data_config
    else:
        loader = _data_loader.create_data_loader(config, shuffle = False, num_batches = NUM_BATCHES_TO_SCAN)
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


def _first_datapoint(tree):
    if isinstance(tree, tuple):
        return tuple(_first_datapoint(item) for item in tree)
    if isinstance(tree, dict):
        return {key: _first_datapoint(value) for key, value in tree.items()}
    if hasattr(tree, "__dataclass_fields__"):
        return {
            field_name: _first_datapoint(getattr(tree, field_name))
            for field_name in tree.__dataclass_fields__
        }
    arr = np.asarray(tree)
    if arr.ndim == 0:
        return arr
    return arr[0]


def _flatten_tree(tree, path: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, tuple):
        result = {}
        for idx, item in enumerate(tree):
            result.update(_flatten_tree(item, path = f"{path}.{idx}" if path else str(idx)))
        return result
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            result.update(_flatten_tree(value, path = f"{path}.{key}" if path else str(key)))
        return result
    return {path: np.asarray(tree)}


def _save_first_datapoint(batch, *, config_name: str, fine_tune: str | None) -> pathlib.Path:
    datapoint = _first_datapoint(batch)
    flattened = _flatten_tree(datapoint)
    DEBUG_DIR.mkdir(parents = True, exist_ok = True)
    suffix = f"_{fine_tune}" if fine_tune is not None else ""
    path = DEBUG_DIR / f"{config_name}{suffix}_first_datapoint.npz"
    np.savez(path, **flattened)
    return path


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


def _extract_state_and_actions(batch, critic_mode: bool) -> tuple[np.ndarray, np.ndarray]:
    if critic_mode:
        state = np.asarray(batch["state"], dtype = np.float32)
        actions = np.asarray(batch["actions"], dtype = np.float32)
    else:
        state = np.asarray(batch[0].state, dtype = np.float32)
        actions = np.asarray(batch[1], dtype = np.float32)
    return state, actions


def _make_out_of_range_stats(name: str, values: np.ndarray) -> dict[str, np.ndarray | str]:
    num_dims = values.shape[-1]
    return {
        "name": name,
        "count_above_one": np.zeros(num_dims, dtype = np.int64),
        "total_count": np.zeros(num_dims, dtype = np.int64),
        "max_abs": np.zeros(num_dims, dtype = np.float32),
        # Per-dim min/max/sum accumulated across all batches (not abs-valued).
        "min_per_dim": np.full(num_dims, np.inf, dtype = np.float32),
        "max_per_dim": np.full(num_dims, -np.inf, dtype = np.float32),
        "sum_per_dim": np.zeros(num_dims, dtype = np.float64),
    }


def _update_out_of_range_stats(stats: dict[str, np.ndarray | str], values: np.ndarray) -> None:
    values_f32 = np.asarray(values, dtype = np.float32)
    flattened = values_f32.reshape(-1, values_f32.shape[-1])
    abs_flattened = np.abs(flattened)
    stats["count_above_one"] += np.sum(abs_flattened > 1.0, axis = 0, dtype = np.int64)
    stats["total_count"] += flattened.shape[0]
    stats["max_abs"][:] = np.maximum(stats["max_abs"], np.max(abs_flattened, axis = 0))
    stats["min_per_dim"][:] = np.minimum(stats["min_per_dim"], np.min(flattened, axis = 0))
    stats["max_per_dim"][:] = np.maximum(stats["max_per_dim"], np.max(flattened, axis = 0))
    stats["sum_per_dim"] += np.sum(flattened.astype(np.float64), axis = 0)


def _print_out_of_range_stats(stats: dict[str, np.ndarray | str]) -> None:
    fractions = stats["count_above_one"] / np.maximum(stats["total_count"], 1)
    mean_per_dim = stats["sum_per_dim"] / np.maximum(stats["total_count"], 1)
    print(f"\n  {stats['name']} per-dim frac_abs_gt_1: {fractions}")
    print(f"  {stats['name']} per-dim max_abs: {stats['max_abs']}")
    print(f"  {stats['name']} per-dim min (across batches): {stats['min_per_dim']}")
    print(f"  {stats['name']} per-dim max (across batches): {stats['max_per_dim']}")
    print(f"  {stats['name']} per-dim mean (across batches): {mean_per_dim}")


@pytest.mark.manual
@pytest.mark.parametrize(
    "config_name",
    [
        PI05_RLDS_CONFIG_NAME,
        REAL_HANG_PI05_RLDS_CONFIG_NAME,
        SARSA_RLDS_CONFIG_NAME,
        CQL_RLDS_CONFIG_NAME,
        REAL_HANG_SUBTASK_ONLY_CONFIG_NAME,
        REAL_HANG_ALL_SUBTASKS_CONFIG_NAME,
        REAL_HANG_ALL_SUBTASKS_PREDICT_CONFIG_NAME,
        VARIABLE_HORIZON_CONFIG_NAME,
        REAL_HANG_60_HZ_CONFIG_NAME,
        GEMMA4_Q_SARSA_CONFIG_NAME,
    ],
)
def test_robocoin_rlds_batch_structure(config_name: str):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    loader, data_config = _make_rlds_raw_loader(config_name)
    state_out_of_range_stats = None
    action_out_of_range_stats = None
    for batch_idx, batch in enumerate(loader):
        state, actions = _extract_state_and_actions(batch, data_config.critic_mode)
        if state_out_of_range_stats is None:
            state_out_of_range_stats = _make_out_of_range_stats("state", state)
            action_out_of_range_stats = _make_out_of_range_stats("actions", actions)
        _update_out_of_range_stats(state_out_of_range_stats, state)
        _update_out_of_range_stats(action_out_of_range_stats, actions)

        if batch_idx == 0:
            print(f"\nRLDS batch keys and shapes (config={config_name}):")
            _print_batch_structure(batch)
            print(f"\n  state shape: {state.shape}")
            # ipdb.set_trace()  # noqa: T100

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
                ipdb.set_trace()  # noqa: T100

    if state_out_of_range_stats is not None and action_out_of_range_stats is not None:
        print(f"\nPer-dimension out-of-range summary (config={config_name}, batches={NUM_BATCHES_TO_SCAN}):")
        _print_out_of_range_stats(state_out_of_range_stats)
        _print_out_of_range_stats(action_out_of_range_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required = True)
    parser.add_argument("--fine-tune", default = None)
    parser.add_argument("--save_datapoint", action = "store_true")
    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    config_name = args.config_name
    loader, data_config = _make_rlds_raw_loader(config_name, fine_tune = args.fine_tune)
    state_out_of_range_stats = None
    action_out_of_range_stats = None
    for batch_idx, batch in enumerate(loader):
        state, actions = _extract_state_and_actions(batch, data_config.critic_mode)
        if state_out_of_range_stats is None:
            state_out_of_range_stats = _make_out_of_range_stats("state", state)
            action_out_of_range_stats = _make_out_of_range_stats("actions", actions)
        _update_out_of_range_stats(state_out_of_range_stats, state)
        _update_out_of_range_stats(action_out_of_range_stats, actions)

        if batch_idx == 0:
            print(f"\nRLDS batch keys and shapes (config={config_name}, fine_tune={args.fine_tune}):")
            _print_batch_structure(batch)
            print(f"\n  state shape: {state.shape}")
            if args.save_datapoint:
                path = _save_first_datapoint(batch, config_name = config_name, fine_tune = args.fine_tune)
                print(f"Saved first datapoint to {path}")
            ipdb.set_trace()  # noqa: T100

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
                ipdb.set_trace()  # noqa: T100

    if state_out_of_range_stats is not None and action_out_of_range_stats is not None:
        print(
            f"\nPer-dimension out-of-range summary "
            f"(config={config_name}, fine_tune={args.fine_tune}, batches={NUM_BATCHES_TO_SCAN}):"
        )
        _print_out_of_range_stats(state_out_of_range_stats)
        _print_out_of_range_stats(action_out_of_range_stats)
