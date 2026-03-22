"""Compare a trained checkpoint against freshly initialized PaliGemma weights.

Computes the L2 norm of the parameter difference using the same include/exclude
filter as param_norm in train_value_function.py (nnx.Param with ndim > 1,
excluding bias/scale/pos_embedding/input_embedding and target_network/head).

Usage:
    python test/compare_checkpoint_to_init.py \
        --checkpoint-path gs://saksham-euw4/checkpoints/robocoin/value_functions/Q/robocoin_bimanual_paligemma_q_sarsa_chunk_wise/robocoin_bimanual_paligemma_q_sarsa_chunk_wise
"""

import argparse
import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.robocoin_utils.load_model_utils import load_train_module, restore_state_with_shardings
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders


CONFIG_NAME = "robocoin_bimanual_paligemma_q_sarsa_chunk_wise"

KERNEL_EXCLUDE_REGEX = ".*/(bias|scale|pos_embedding|input_embedding)"
TARGET_REGEX = ".*target_(network|head)/.*"


def create_fresh_model(config: _config.TrainConfig, rng: jax.Array) -> nnx.State:
    """Create a fresh model with PaliGemma pretrained weights loaded."""
    import openpi.value_functions.base_value_functions as _value_fn
    import openpi.value_functions.value_function as _value_fn_impl

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(f"Expected BaseValueFunctionConfig, got {type(config.model)}")

    if isinstance(config.model, _value_fn_impl.ValueFunctionConfig):
        model_config = dataclasses.replace(config.model, action_horizon = config.action_horizon)
    else:
        model_config = config.model

    rng, model_rng = jax.random.split(rng)
    model = model_config.create(model_rng)

    # Load PaliGemma pretrained weights
    params_shape = nnx.state(model).to_pure_dict()
    params_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params_shape)
    partial_params = _load_weights_and_validate(config.weight_loader, params_shape)

    # Merge loaded weights into model
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(partial_params)
    model = nnx.merge(graphdef, state)

    # Apply dtype casting (same as init_train_state)
    params = nnx.state(model)
    params = nnx_utils.state_map(
        params,
        config.freeze_filter,
        lambda p: p.replace(p.value.astype(jnp.bfloat16)),
    )
    weight_dtype = jnp.dtype(model_config.weight_dtype) if hasattr(model_config, "weight_dtype") else jnp.float32
    params = nnx_utils.state_map(
        params,
        config.trainable_filter,
        lambda p: p.replace(p.value.astype(weight_dtype)),
    )

    return params, model


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected = params_shape, got = loaded_params, check_shapes = True, check_dtypes = True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def load_checkpoint_params(config: _config.TrainConfig, checkpoint_path: str) -> nnx.State:
    """Load params from a checkpoint using CheckpointManager (handles FSDP shards).

    Overrides fsdp_devices to match local device count and passes explicit
    target shardings so checkpoints saved on multi-device TPU pods can be
    restored on a single GPU.
    """
    train_module = load_train_module()

    local_devices = jax.local_device_count()
    local_config = dataclasses.replace(config, fsdp_devices = local_devices)

    init_rng = jax.random.PRNGKey(86)
    mesh = sharding.make_mesh(local_devices)

    train_state_shape, state_sharding = train_module.init_train_state(local_config, init_rng, mesh, resume = True)

    mngr, _ = _checkpoints.initialize_checkpoint_dir(
        checkpoint_path, keep_period = None, overwrite = False, resume = True
    )
    train_state = restore_state_with_shardings(mngr, train_state_shape, state_sharding)

    critic_state = train_state.critic
    return critic_state.params, critic_state.model_def


def filter_kernel_params(params: nnx.State, model: nnx.Module, include_target: bool = False) -> nnx.State:
    """Filter params using the same logic as kernel_params in train_value_function.py."""
    filters = [
        nnx.Param,
        nnx.Not(nnx_utils.PathRegex(KERNEL_EXCLUDE_REGEX)),
        lambda _, x: x.value.ndim > 1,
    ]
    if not include_target:
        filters.append(nnx.Not(nnx_utils.PathRegex(TARGET_REGEX)))

    return nnx.state(model, nnx.All(*filters))


def compute_l2_diff(params_a: nnx.State, params_b: nnx.State) -> jnp.ndarray:
    """Compute L2 norm of difference between two param sets, all in bfloat16."""
    flat_a = traverse_util.flatten_dict(params_a.to_pure_dict())
    flat_b = traverse_util.flatten_dict(params_b.to_pure_dict())

    assert set(flat_a.keys()) == set(flat_b.keys()), (
        f"Key mismatch: {set(flat_a.keys()) - set(flat_b.keys())} vs {set(flat_b.keys()) - set(flat_a.keys())}"
    )

    sum_sq = jnp.zeros((), dtype = jnp.bfloat16)
    per_param_norms = {}

    for key in sorted(flat_a.keys()):
        a = flat_a[key].astype(jnp.bfloat16)
        b = flat_b[key].astype(jnp.bfloat16)
        diff_sq = jnp.sum(jnp.square(a - b))
        sum_sq = sum_sq + diff_sq
        per_param_norms["/".join(key)] = jnp.sqrt(diff_sq)

    return jnp.sqrt(sum_sq), per_param_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type = str,
        default = "gs://saksham-euw4/checkpoints/robocoin/value_functions/Q/"
        "robocoin_bimanual_paligemma_q_sarsa_chunk_wise/"
        "robocoin_bimanual_paligemma_q_sarsa_chunk_wise",
    )
    args = parser.parse_args()

    config = _config.get_config(CONFIG_NAME)
    rng = jax.random.key(86)

    print("Creating fresh model with PaliGemma pretrained weights...")
    fresh_params, model = create_fresh_model(config, rng)
    graphdef, _ = nnx.split(model)

    print(f"Loading checkpoint from {args.checkpoint_path}...")
    ckpt_params, ckpt_model_def = load_checkpoint_params(config, args.checkpoint_path)

    print("\nFiltering to kernel params (same as param_norm filter)...")

    fresh_model = nnx.merge(graphdef, fresh_params)
    fresh_kernel = filter_kernel_params(fresh_params, fresh_model)

    ckpt_model = nnx.merge(ckpt_model_def, ckpt_params)
    ckpt_kernel = filter_kernel_params(ckpt_params, ckpt_model)

    print(f"Number of kernel param arrays: {len(traverse_util.flatten_dict(fresh_kernel.to_pure_dict()))}")

    # Compute L2 norm of difference
    print("\nComputing L2 norm of parameter difference (bfloat16)...")
    total_l2, per_param = compute_l2_diff(fresh_kernel, ckpt_kernel)

    print(f"\nTotal L2 norm of difference: {float(total_l2):.6f}")
    print(f"\nFresh model param_norm: {float(jnp.sqrt(sum(jnp.sum(jnp.square(v.astype(jnp.bfloat16))) for v in jax.tree.leaves(fresh_kernel)))):.4f}")
    print(f"Checkpoint param_norm: {float(jnp.sqrt(sum(jnp.sum(jnp.square(v.astype(jnp.bfloat16))) for v in jax.tree.leaves(ckpt_kernel)))):.4f}")

    print("\n--- Per-parameter L2 norms (top 20) ---")
    sorted_norms = sorted(per_param.items(), key = lambda x: float(x[1]), reverse = True)
    for name, norm in sorted_norms[:20]:
        print(f"  {name}: {float(norm):.6f}")

    if len(sorted_norms) > 20:
        print(f"  ... ({len(sorted_norms) - 20} more)")

    # Also check: how many params are exactly identical?
    flat_fresh = traverse_util.flatten_dict(fresh_kernel.to_pure_dict())
    flat_ckpt = traverse_util.flatten_dict(ckpt_kernel.to_pure_dict())
    num_identical = sum(
        1 for k in flat_fresh if jnp.allclose(flat_fresh[k].astype(jnp.bfloat16), flat_ckpt[k].astype(jnp.bfloat16))
    )
    print(f"\nParams with zero difference: {num_identical}/{len(flat_fresh)}")


if __name__ == "__main__":
    main()
