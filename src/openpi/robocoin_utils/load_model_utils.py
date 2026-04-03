"""Utilities for loading policy and value function models from checkpoints."""

import dataclasses
import importlib
import logging
import os

import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as _sharding

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))


def _load_script_module(script_name: str):
    """Dynamically import a script from scripts/ as a module."""
    path = os.path.join(_SCRIPTS_DIR, script_name)
    module_name = script_name.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_module():
    """Dynamically import train_value_function.py as a module."""
    return _load_script_module("train_value_function.py")


def restore_state_with_shardings(checkpoint_manager, state_shape, state_sharding):
    """Restore checkpoint with explicit target shardings for cross-device loading.

    Use this instead of checkpoints.restore_state when the checkpoint was saved
    on a different device topology (e.g. loading a TPU pod checkpoint on a
    single GPU).
    """
    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state_shape)
        train_state_sharding, params_sharding = _checkpoints._split_params(state_sharding)

        def _to_restore_args(sharding_tree):
            return jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding = s), sharding_tree)

        restored = checkpoint_manager.restore(
            step = None,
            args = ocp.args.Composite(
                train_state = ocp.args.PyTreeRestore(
                    item = train_state,
                    restore_args = _to_restore_args(train_state_sharding),
                ),
                params = ocp.args.PyTreeRestore(
                    item = {"params": params},
                    restore_args = {"params": _to_restore_args(params_sharding)},
                ),
            ),
        )
    return _checkpoints._merge_params(restored["train_state"], restored["params"])


def restore_params_with_shardings(checkpoint_manager, state_shape, state_sharding, *, step: int | None = None):
    """Restore only inference params with explicit target shardings.

    This avoids materializing optimizer state when loading a checkpoint for
    evaluation, which substantially reduces memory pressure compared to
    restoring the full training state.
    """
    with at.disable_typechecking():
        _, params = _checkpoints._split_params(state_shape)
        _, params_sharding = _checkpoints._split_params(state_sharding)

        def _to_restore_args(sharding_tree):
            return jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding = s), sharding_tree)

        restored = checkpoint_manager.restore(
            step = step,
            args = ocp.args.Composite(
                params = ocp.args.PyTreeRestore(
                    item = {"params": params},
                    restore_args = {"params": _to_restore_args(params_sharding)},
                ),
            ),
        )
    return restored["params"]


# =============================================================================
# Policy loading
# =============================================================================


@dataclasses.dataclass(frozen = True)
class LoadPolicyConfig:
    """Configuration for loading a policy checkpoint."""

    config_name: str
    checkpoint_path: str
    fine_tune: str | None = None
    step: int | None = None


def load_policy(load_config: LoadPolicyConfig):
    """Load a pi0 policy model from a checkpoint.

    Returns:
        model: The loaded policy model (BaseModel).
        config: The resolved TrainConfig.
    """
    config = _config.get_config(load_config.config_name)
    if load_config.fine_tune is not None:
        ft_config = _config.get_fine_tune_config(load_config.fine_tune)
        config = ft_config.apply_overrides(config, pretrained_step = None)

    config = dataclasses.replace(config, fsdp_devices = jax.local_device_count())

    train_module = _load_script_module("train.py")
    rng = jax.random.PRNGKey(86)
    mesh = _sharding.make_mesh(config.fsdp_devices)
    train_state_shape, state_sharding = train_module.init_train_state(config, rng, mesh, resume = True)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        load_config.checkpoint_path,
        keep_period = None,
        overwrite = False,
        resume = True,
    )
    restored_params = restore_params_with_shardings(checkpoint_manager, train_state_shape, state_sharding, step = load_config.step)
    params = restored_params["params"]
    model = nnx.merge(train_state_shape.model_def, params)
    logger.info(f"Loaded policy from {load_config.checkpoint_path}")

    return model, config
