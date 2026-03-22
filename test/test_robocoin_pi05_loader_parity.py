import dataclasses
import os

import numpy as np
import pytest
import tensorflow as tf

from openpi.models.tokenizer import create_tokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.robocoin_data_loader import RoboCOINDataLoader


os.environ["OPENPI_ROBOCOIN_PARITY_KEEP_REPO_INDEX"] = "1"


SEED = 86
NUM_BATCHES_TO_SCAN = 500
LEGACY_CONFIG_NAME = "robocoin_bimanual_pi05"
RLDS_CONFIG_NAME = "robocoin_bimanual_pi05_rlds"


def _make_train_config(config_name: str) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    return dataclasses.replace(config, batch_size = 4, seed = SEED, num_workers = 0)


def _action_horizon(config: _config.TrainConfig) -> int:
    action_horizon = config.action_horizon
    if action_horizon is None:
        action_horizon = config.model.action_horizon
    if action_horizon is None:
        raise ValueError("Action horizon must be set on either TrainConfig or the model config.")
    return action_horizon


def _make_legacy_raw_loader(config: _config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    robocoin_config = dataclasses.replace(
        data_config.robocoin_data_config,
        action_horizon = _action_horizon(config),
        batch_size = config.batch_size,
        shuffle = False,
        seed = config.seed,
        state_norm_stats = data_config.norm_stats,
        use_quantile_norm = data_config.use_quantile_norm,
    )
    num_images = robocoin_config.max_cameras if config.backbone_variant == "gemma3" else 0
    tokenizer = create_tokenizer(config.backbone_variant, robocoin_config.max_token_len, num_images = num_images)
    model_transforms = list(data_config.model_transforms.inputs) if data_config.model_transforms.inputs else None
    return RoboCOINDataLoader(
        robocoin_config,
        tokenizer = tokenizer,
        num_batches = NUM_BATCHES_TO_SCAN,
        model_transforms = model_transforms,
    )


def _make_rlds_raw_loader(config: _config.TrainConfig):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_rlds_dataset(
        data_config,
        _action_horizon(config),
        config.batch_size,
        split = "train",
        shuffle = False,
    )
    dataset = _data_loader.transform_iterable_dataset(dataset, data_config, is_batched = True)
    return _data_loader.RLDSDataLoader(dataset, num_batches = NUM_BATCHES_TO_SCAN)


def _index_sample(tree, index: int):
    if isinstance(tree, dict):
        return {key: _index_sample(value, index) for key, value in tree.items()}
    return np.asarray(tree)[index]


def _sample_key(sample) -> tuple[int, int]:
    return int(np.asarray(sample["repo_index"]).item()), int(np.asarray(sample["index"]).item())


def _collect_samples(loader) -> dict[tuple[int, int], dict]:
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


@pytest.mark.manual
def test_robocoin_pi05_legacy_and_rlds_train_batches_match_on_overlapping_samples():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    legacy_loader = _make_legacy_raw_loader(_make_train_config(LEGACY_CONFIG_NAME))

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    rlds_loader = _make_rlds_raw_loader(_make_train_config(RLDS_CONFIG_NAME))

    legacy_samples = _collect_samples(legacy_loader)
    rlds_samples = _collect_samples(rlds_loader)

    matched_keys = sorted(set(legacy_samples) & set(rlds_samples))
    assert matched_keys, "No overlapping (repo_index, index) pairs found across the scanned batches."

    sample_keys = sorted(set(next(iter(legacy_samples.values())).keys()) - {"_traj_index"})
    print(f"legacy collected samples: {len(legacy_samples)}")
    print(f"rlds collected samples: {len(rlds_samples)}")
    print(f"matched data points compared exactly: {len(matched_keys)}")
    print(f"compared sample keys: {sample_keys}")

    for key in matched_keys:
        _assert_batch_equal(legacy_samples[key], rlds_samples[key], path = f"sample[{key}]")
