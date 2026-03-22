import dataclasses
import os

import numpy as np
import pytest
import tensorflow as tf

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


os.environ["OPENPI_ROBOCOIN_PARITY_KEEP_REPO_INDEX"] = "1"


SEED = 86
NUM_BATCHES_TO_SCAN = 500
LEGACY_CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_chunk_wise"
RLDS_CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_chunk_wise_rlds"


def _make_train_config(config_name: str) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    return dataclasses.replace(config, batch_size = 4, seed = SEED, num_workers = 0)


def _index_sample(tree, index: int):
    if isinstance(tree, dict):
        return {key: _index_sample(value, index) for key, value in tree.items()}
    return np.asarray(tree)[index]


def _sample_key(sample) -> tuple[int, int]:
    return int(np.asarray(sample["repo_index"]).item()), int(np.asarray(sample["index"]).item())


def _collect_samples(config_name: str) -> dict[tuple[int, int], dict]:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    loader = _data_loader.create_data_loader(
        _make_train_config(config_name),
        shuffle = False,
        num_batches = NUM_BATCHES_TO_SCAN,
    )

    samples: dict[tuple[int, int], dict] = {}
    for batch in loader:
        batch_size = next(np.asarray(value).shape[0] for value in batch.values() if hasattr(value, "shape"))
        for sample_index in range(batch_size):
            sample = _index_sample(batch, sample_index)
            key = _sample_key(sample)
            if key in samples:
                raise ValueError(f"Duplicate sample for key={key}")
            samples[key] = sample
    return samples


def _assert_batch_equal(lhs, rhs, path: str = "batch") -> None:
    if isinstance(lhs, dict):
        assert isinstance(rhs, dict), f"{path}: expected dict, got {type(rhs)}"
        lhs_keys = set(lhs.keys()) - {"_traj_index"}
        rhs_keys = set(rhs.keys()) - {"_traj_index"}
        assert lhs_keys == rhs_keys, f"{path}: key mismatch {sorted(lhs_keys)} != {sorted(rhs_keys)}"
        for key in sorted(lhs_keys):
            _assert_batch_equal(lhs[key], rhs[key], path = f"{path}.{key}")
        return

    lhs_array = np.asarray(lhs)
    rhs_array = np.asarray(rhs)

    assert lhs_array.shape == rhs_array.shape, f"{path}: shape mismatch {lhs_array.shape} != {rhs_array.shape}"
    assert lhs_array.dtype == rhs_array.dtype, f"{path}: dtype mismatch {lhs_array.dtype} != {rhs_array.dtype}"
    np.testing.assert_array_equal(lhs_array, rhs_array, err_msg = path)


def _count_steps_buckets(samples: list[dict]) -> tuple[int, int]:
    less_than_30 = 0
    at_least_30_less_than_60 = 0
    for sample in samples:
        steps_to_subtask_end = int(np.asarray(sample["steps_to_subtask_end"]).item())
        if steps_to_subtask_end < 30:
            less_than_30 += 1
        elif steps_to_subtask_end < 60:
            at_least_30_less_than_60 += 1
    return less_than_30, at_least_30_less_than_60


@pytest.mark.manual
def test_robocoin_legacy_and_rlds_train_batches_match_exactly():
    legacy_samples = _collect_samples(LEGACY_CONFIG_NAME)
    rlds_samples = _collect_samples(RLDS_CONFIG_NAME)

    matched_keys = sorted(set(legacy_samples) & set(rlds_samples))
    assert matched_keys, "No overlapping (repo_index, index) pairs found across the scanned batches."

    compared_samples = [legacy_samples[key] for key in matched_keys]
    less_than_30_count, at_least_30_less_than_60_count = _count_steps_buckets(compared_samples)
    sample_keys = sorted(set(next(iter(compared_samples)).keys()) - {"_traj_index"})

    print(f"legacy collected samples: {len(legacy_samples)}")
    print(f"rlds collected samples: {len(rlds_samples)}")
    print(f"matched data points compared exactly: {len(matched_keys)}")
    print(f"compared sample keys: {sample_keys}")
    print(f"matched data points with steps_to_subtask_end < 30: {less_than_30_count}")
    print(f"matched data points with 30 <= steps_to_subtask_end < 60: {at_least_30_less_than_60_count}")

    for key in matched_keys:
        _assert_batch_equal(legacy_samples[key], rlds_samples[key], path = f"sample[{key}]")
