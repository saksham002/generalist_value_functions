"""Utilities for loading value function models from checkpoints."""

import importlib
import os

import jax
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints


def load_train_module():
    """Dynamically import train_value_function.py as a module."""
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts")
    path = os.path.normpath(os.path.join(scripts_dir, "train_value_function.py"))
    spec = importlib.util.spec_from_file_location("train_value_function", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def restore_params_with_shardings(checkpoint_manager, state_shape, state_sharding):
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
            step = None,
            args = ocp.args.Composite(
                params = ocp.args.PyTreeRestore(
                    item = {"params": params},
                    restore_args = {"params": _to_restore_args(params_sharding)},
                ),
            ),
        )
    return restored["params"]
