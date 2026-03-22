"""Checkpoint loading for eval-time policy and value bundles."""

from __future__ import annotations

import copy
import pathlib

import jax
import jax.numpy as jnp

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import download as _download
from openpi.training import checkpoints as _checkpoints

from eval.helpers import runtime_types


def _load_model_from_checkpoint(train_config, checkpoint_dir: pathlib.Path):
    weight_path = checkpoint_dir / "model.safetensors"
    if weight_path.exists():
        raise NotImplementedError("PyTorch eval bundles are not supported by the real-world best-of-n path yet.")
    return train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype = jnp.bfloat16))


def load_policy_bundle(
    train_config,
    checkpoint_dir: pathlib.Path | str,
    *,
    default_prompt: str | None = None,
    sample_kwargs: dict | None = None,
) -> runtime_types.PolicyBundle:
    checkpoint_dir = pathlib.Path(_download.maybe_download(str(checkpoint_dir)))
    model = _load_model_from_checkpoint(train_config, checkpoint_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Policy bundle requires data_config.asset_id to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    repack_transforms = _transforms.Group()
    input_transform = _transforms.compose(
        [
            *repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ]
    )
    return runtime_types.PolicyBundle(
        model = model,
        input_transform = input_transform,
        output_transform = output_transform,
        sample_kwargs = sample_kwargs or {},
        metadata = copy.deepcopy(train_config.policy_metadata),
        rng = jax.random.key(0),
        train_config = train_config,
    )


def load_value_bundle(
    train_config,
    checkpoint_dir: pathlib.Path | str,
) -> runtime_types.ValueBundle:
    checkpoint_dir = pathlib.Path(_download.maybe_download(str(checkpoint_dir)))
    model = _load_model_from_checkpoint(train_config, checkpoint_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Value bundle requires data_config.asset_id to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    input_transform = _transforms.compose(
        [
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    return runtime_types.ValueBundle(
        model = model,
        input_transform = input_transform,
        train_config = train_config,
    )
