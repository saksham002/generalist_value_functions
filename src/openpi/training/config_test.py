import pathlib

import numpy as np

from openpi.training import config as _config
from openpi.training import rlds_dataset
from openpi.value_functions import heads as _heads
from openpi.value_functions import value_function as _value_function
from openpi.value_functions.networks import paligemma as _paligemma_network


def _make_norm_stats():
    zeros = np.zeros(14, dtype = np.float32)
    ones = np.ones(14, dtype = np.float32)
    stats = _config._transforms.NormStats(mean = zeros, std = ones, q01 = zeros, q99 = ones)
    return {
        "Split_aloha": {
            "state": stats,
            "actions": stats,
            "next_state": stats,
            "next_actions": stats,
        }
    }


def test_robocoin_rlds_data_config_chunk_wise_create(monkeypatch):
    monkeypatch.setattr(_config.RoboCoinRldsDataConfig, "_load_robocoin_norm_stats", lambda self: _make_norm_stats())

    data_config_factory = _config.RoboCoinRldsDataConfig(
        rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
        datasets = (rlds_dataset.RLDSDataset(name = "robocoin_bimanual", version = "1.0.0", weight = 1.0),),
        use_eef = True,
        td_n = 50,
        use_chunk_wise_delta = True,
        use_quantile_norm = True,
    )
    model_config = _value_function.SARSAValueFunctionConfig(
        network_config = _paligemma_network.PaliGemmaNetworkConfig(
            state_dim = 14,
            num_cameras = 3,
            image_size = (224, 224),
            max_token_len = 48,
            action_dim = 14,
            dtype = "float32",
        ),
        head_config = _heads.RegressionHeadConfig(),
    )

    data_config = data_config_factory.create(pathlib.Path("."), model_config)

    assert data_config.rlds_dataset_class == "robocoin"
    assert data_config.rlds_data_dir == "gs://saksham-euw4/robocoin_bimanual"
    assert data_config.datasets[0].name == "robocoin_bimanual"
    assert data_config.critic_mode
    assert data_config.robocoin_use_eef
    assert data_config.use_quantile_norm
    assert data_config.val_split == "val"
    assert data_config.clip_normalized_bounds == {
        "state": (-1.25, 1.25),
        "actions": (-1.25, 1.25),
        "next_state": (-1.25, 1.25),
        "next_actions": (-1.25, 1.25),
    }
    assert data_config.rlds_kwargs == {
        "td_n": 50,
        "filter_n": None,
        "mask_50fps": False,
        "use_chunk_wise_delta": True,
        "shuffle_buffer_size": 250_000,
        "num_parallel_reads": 8,
        "num_parallel_calls": 8,
    }
    assert data_config.data_transforms.inputs == []
    assert isinstance(data_config.model_transforms.inputs[0], _config._transforms.ReplaceMaskedActions)
    assert isinstance(data_config.model_transforms.inputs[1], _config._transforms.ResizeImages)
    assert isinstance(data_config.model_transforms.inputs[2], _config.DecodeRoboCoinPromptBytes)
    assert isinstance(data_config.model_transforms.inputs[3], _config._transforms.TokenizePrompt)


def test_real_hang_pi05_filter_intervention_only_pads_actions(monkeypatch):
    monkeypatch.setattr(_config.RoboCoinRldsDataConfig, "_load_norm_stats", lambda self, *_: _make_norm_stats())

    cfg = _config.get_config("real_hang_pi05_filter_intervention")

    assert cfg.model.pad_state_to_action_dim is False

    _, action_spec = cfg.model.inputs_spec()
    assert action_spec.shape == (1, 50, 32)

    data_config = cfg.data.create(pathlib.Path("."), cfg.model)
    pad_transform = data_config.model_transforms.inputs[-1]
    assert isinstance(pad_transform, _config._transforms.PadStatesAndActions)
    assert pad_transform.model_action_dim == 32
    assert pad_transform.action_dim_offset == 14
    assert pad_transform.pad_state is False
