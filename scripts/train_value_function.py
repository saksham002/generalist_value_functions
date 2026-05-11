"""Training script for value functions with optional policy training.

Supports:
- Critic-only training (default)
- Joint critic + policy training with configurable update ratio
- Policy extraction with frozen critic (critic_steps_per_policy_step=0)
"""

import dataclasses
import functools
import gc
import logging
import platform as _platform
import os
import pickle
import threading
from typing import Any
import pdb

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.stats
import tqdm_loggable.auto as tqdm
import wandb
from jax.experimental import multihost_utils

from openpi.models import model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.evaluation as _evaluation
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.robocoin_utils.load_model_utils as _load_model_utils
from openpi.training.time_utils import Timer
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
import openpi.value_functions.base_value_functions as _value_fn
import openpi.value_functions.value_function as _value_fn_impl
from openpi.robocoin_utils.utils import (
    cache_val_episodes,
    count_subtask_segments,
    decode_episode_images,
    get_obs_and_action,
    predict_values,
    stack_frames,
    stack_images,
)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


# =============================================================================
# Memory Debugging for TensorFlow + TPU
# =============================================================================

def get_memory_stats() -> dict[str, float]:
    """Get comprehensive memory statistics for debugging TF+TPU memory leaks.
    
    Returns dict with memory stats in MB. Logged to wandb for tracking.
    """
    import gc
    import psutil
    import sys
    import tensorflow as tf
    
    stats = {}
    
    # 1. Process RSS (Resident Set Size) - actual physical memory used
    process = psutil.Process()
    stats["process_rss_mb"] = process.memory_info().rss / 1024 / 1024
    stats["process_vms_mb"] = process.memory_info().vms / 1024 / 1024
    
    # 2. Python garbage collector stats
    gc_stats = gc.get_stats()
    stats["gc_gen0_collections"] = gc_stats[0]["collections"]
    stats["gc_gen1_collections"] = gc_stats[1]["collections"]
    stats["gc_gen2_collections"] = gc_stats[2]["collections"]
    
    # 3. Count of Python objects by type (top memory consumers)
    type_counts = {}
    for obj in gc.get_objects():
        try:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        except Exception:
            pass
    
    # Track key types that might leak
    stats["py_dict_count"] = type_counts.get("dict", 0)
    stats["py_list_count"] = type_counts.get("list", 0)
    stats["py_ndarray_count"] = type_counts.get("ndarray", 0)
    stats["py_EagerTensor_count"] = type_counts.get("EagerTensor", 0)
    stats["py_ResourceVariable_count"] = type_counts.get("ResourceVariable", 0)
    
    # 4. JAX compilation cache stats - try multiple approaches
    try:
        import jax
        # JAX doesn't expose cache stats directly, but we can count live compilations
        # by looking at the number of JIT-compiled functions in memory
        jit_count = 0
        pjit_count = 0
        for obj in gc.get_objects():
            try:
                obj_type = type(obj).__name__
                if 'JittedFunction' in obj_type or 'jit' in obj_type.lower():
                    jit_count += 1
                if 'pjit' in obj_type.lower():
                    pjit_count += 1
            except Exception:
                pass
        stats["jax_jit_objects"] = jit_count
        stats["jax_pjit_objects"] = pjit_count
        
        # Try to get live executable count from JAX runtime
        try:
            from jax._src import xla_bridge
            backend = xla_bridge.get_backend()
            has_live_exec = hasattr(backend, 'live_executables')
            has_live_arrays = hasattr(backend, 'live_arrays')
            if has_live_exec:
                stats["jax_live_executables"] = len(backend.live_executables())
            else:
                logging.info(f"[MemoryDebug] backend has no 'live_executables' attr. Available: {[a for a in dir(backend) if not a.startswith('_')][:20]}")
            if has_live_arrays:
                stats["jax_live_arrays"] = len(backend.live_arrays())
            else:
                logging.info(f"[MemoryDebug] backend has no 'live_arrays' attr")
        except Exception as e:
            logging.info(f"[MemoryDebug] JAX backend error: {e}")
    except Exception as e:
        logging.info(f"[MemoryDebug] JAX cache error: {e}")
    
    # 5. TensorFlow/TPU memory stats
    try:
        # List all devices to understand what's available
        tpus = tf.config.list_physical_devices('TPU')
        stats["tf_num_tpus"] = len(tpus)
        logging.info(f"[MemoryDebug] Found {len(tpus)} TPUs")
        
        # Try to get TPU memory info (may not be available)
        for i, tpu in enumerate(tpus):
            try:
                mem_info = tf.config.experimental.get_memory_info(f'TPU:{i}')
                if mem_info:
                    stats[f"tf_tpu{i}_current_mb"] = mem_info.get("current", 0) / 1024 / 1024
                    stats[f"tf_tpu{i}_peak_mb"] = mem_info.get("peak", 0) / 1024 / 1024
                else:
                    logging.info(f"[MemoryDebug] TPU:{i} get_memory_info returned None/empty")
            except Exception as e:
                logging.info(f"[MemoryDebug] TPU:{i} get_memory_info failed: {e}")
        
        # Try libtpu memory stats via JAX backend
        try:
            from jax._src import xla_bridge
            backend = xla_bridge.get_backend()
            stats["jax_backend_platform"] = backend.platform
            has_mem_stats = hasattr(backend, 'memory_stats')
            if has_mem_stats:
                mem_stats = backend.memory_stats()
                if mem_stats:
                    for k, v in mem_stats.items():
                        stats[f"tpu_mem_{k}"] = v
                    logging.info(f"[MemoryDebug] Got memory_stats keys: {list(mem_stats.keys())}")
                else:
                    logging.info(f"[MemoryDebug] backend.memory_stats() returned None/empty")
            else:
                logging.info(f"[MemoryDebug] backend has no 'memory_stats' attr. Platform={backend.platform}")
        except Exception as e:
            logging.info(f"[MemoryDebug] TPU backend memory_stats error: {e}")
        # Modern JAX exposes HBM stats per-device. Sum across local devices for
        # an aggregate, plus log per-device peak (useful when devices diverge).
        try:
            local_devs = jax.local_devices()
            used_total = 0
            limit_total = 0
            peak_total = 0
            for di, dev in enumerate(local_devs):
                if not hasattr(dev, "memory_stats"):
                    continue
                ms = dev.memory_stats()
                if not ms:
                    continue
                used = ms.get("bytes_in_use", 0)
                peak = ms.get("peak_bytes_in_use", 0)
                limit = ms.get("bytes_limit", 0) or ms.get("bytes_reservable_limit", 0)
                used_total += used
                peak_total += peak
                limit_total += limit
                stats[f"hbm_dev{di}_used_gb"] = used / 1e9
                stats[f"hbm_dev{di}_peak_gb"] = peak / 1e9
                stats[f"hbm_dev{di}_limit_gb"] = limit / 1e9
            if limit_total > 0:
                stats["hbm_used_gb_total"] = used_total / 1e9
                stats["hbm_peak_gb_total"] = peak_total / 1e9
                stats["hbm_limit_gb_total"] = limit_total / 1e9
                logging.info(
                    f"[MemoryDebug] HBM (sum over {len(local_devs)} local devs): "
                    f"used={used_total / 1e9:.2f} GB, "
                    f"peak={peak_total / 1e9:.2f} GB, "
                    f"limit={limit_total / 1e9:.2f} GB"
                )
        except Exception as e:
            logging.info(f"[MemoryDebug] device.memory_stats error: {e}")
    except Exception as e:
        logging.info(f"[MemoryDebug] TF device listing error: {e}")
    
    # 6. Count TensorFlow objects in Python heap (fallback)
    tf_tensor_count = 0
    tf_variable_count = 0
    tf_dataset_count = 0
    for obj in gc.get_objects():
        try:
            obj_type = type(obj).__name__
            if 'Tensor' in obj_type and 'tensorflow' in str(type(obj).__module__):
                tf_tensor_count += 1
            if 'Variable' in obj_type and 'tensorflow' in str(type(obj).__module__):
                tf_variable_count += 1
            if 'Dataset' in obj_type:
                tf_dataset_count += 1
        except Exception:
            pass
    stats["tf_tensor_objects"] = tf_tensor_count
    stats["tf_variable_objects"] = tf_variable_count
    stats["tf_dataset_objects"] = tf_dataset_count
    
    # 7. Track TensorFlow graph function cache
    try:
        from tensorflow.python.eager import context
        ctx = context.context()
        if hasattr(ctx, '_function_cache'):
            stats["tf_function_cache_size"] = len(ctx._function_cache)
        # Also check for concrete functions
        if hasattr(ctx, '_functions'):
            stats["tf_functions_count"] = len(ctx._functions)
    except Exception as e:
        stats["tf_ctx_error"] = str(e)[:50]
    
    # 8. Try to get TF allocator stats (for debugging memory fragmentation)
    try:
        from tensorflow.python.framework import config as tf_config_module
        # Try to get BFC allocator stats
        if hasattr(tf_config_module, 'get_memory_growth'):
            stats["tf_memory_growth_enabled"] = 1
    except Exception:
        pass
    
    # 9. Track glibc malloc stats (if available) - helps detect fragmentation
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        # Try mallinfo2 (glibc 2.33+) or mallinfo
        if hasattr(libc, 'mallinfo2'):
            class MallInfo2(ctypes.Structure):
                _fields_ = [
                    ("arena", ctypes.c_size_t),      # Non-mmapped space allocated
                    ("ordblks", ctypes.c_size_t),   # Free chunks
                    ("smblks", ctypes.c_size_t),    # Free fastbin blocks
                    ("hblks", ctypes.c_size_t),     # Mmapped regions
                    ("hblkhd", ctypes.c_size_t),    # Space in mmapped regions
                    ("usmblks", ctypes.c_size_t),   # Always 0
                    ("fsmblks", ctypes.c_size_t),   # Space in freed fastbin blocks
                    ("uordblks", ctypes.c_size_t),  # Total allocated space
                    ("fordblks", ctypes.c_size_t),  # Total free space
                    ("keepcost", ctypes.c_size_t),  # Releasable space
                ]
            libc.mallinfo2.restype = MallInfo2
            mi = libc.mallinfo2()
            stats["malloc_arena_mb"] = mi.arena / 1024 / 1024
            stats["malloc_used_mb"] = mi.uordblks / 1024 / 1024
            stats["malloc_free_mb"] = mi.fordblks / 1024 / 1024
            stats["malloc_mmap_mb"] = mi.hblkhd / 1024 / 1024
            stats["malloc_releasable_mb"] = mi.keepcost / 1024 / 1024
    except Exception as e:
        pass  # malloc stats not available
    
    # 8. Approximate size of tracked objects (sample)
    sample_size = 0
    for obj in list(gc.get_objects())[:10000]:
        try:
            sample_size += sys.getsizeof(obj)
        except Exception:
            pass
    stats["py_sample_objects_mb"] = sample_size / 1024 / 1024
    
    return stats


def log_memory_debug(step: int, data_loader=None, force_gc: bool = False, log_to_wandb: bool = True):
    """Log detailed memory stats for debugging.
    
    Args:
        step: Current training step
        data_loader: Optional data loader to get buffer stats (e.g., AsyncBatchPrefetcher)
        force_gc: If True, force garbage collection and log before/after
        log_to_wandb: If True, log to wandb
    """
    import gc
    
    prefix = f"[Memory Step {step}]"
    
    if force_gc:
        stats_before = get_memory_stats()
        gc.collect()
        
        # Clear TensorFlow/Keras internal state
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            logging.info(f"{prefix} tf.keras.backend.clear_session() called")
        except Exception as e:
            logging.info(f"{prefix} clear_session failed: {e}")
        
        # Try to return freed memory to OS (helps with fragmentation)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            trimmed = libc.malloc_trim(0)
            logging.info(f"{prefix} malloc_trim returned {trimmed} (1=released memory, 0=nothing to release)")
        except Exception as e:
            logging.info(f"{prefix} malloc_trim failed: {e}")
        
        stats_after = get_memory_stats()
        
        freed_mb = stats_before["process_rss_mb"] - stats_after["process_rss_mb"]
        logging.info(f"{prefix} GC+malloc_trim freed: {freed_mb:.1f} MB")
        logging.info(f"{prefix} RSS: {stats_after['process_rss_mb']:.1f} MB (was {stats_before['process_rss_mb']:.1f})")
        stats = stats_after
    else:
        stats = get_memory_stats()
        logging.info(f"{prefix} RSS: {stats['process_rss_mb']:.1f} MB, VMS: {stats['process_vms_mb']:.1f} MB")
    
    # Log Python object counts that might indicate leaks
    logging.info(
        f"{prefix} PyObjects: dict={stats['py_dict_count']}, list={stats['py_list_count']}, "
        f"ndarray={stats['py_ndarray_count']}, EagerTensor={stats['py_EagerTensor_count']}"
    )
    
    # Log data loader buffer stats if available
    prefetcher = None
    if data_loader is not None:
        if hasattr(data_loader, 'buffer'):
            prefetcher = data_loader
        elif hasattr(data_loader, '_data_loader'):
            inner = data_loader._data_loader
            if hasattr(inner, '_iterator') and inner._iterator is not None:
                prefetcher = inner._iterator
    
    if prefetcher is not None and hasattr(prefetcher, 'buffer'):
        try:
            buffer_size = prefetcher.buffer.qsize()
            logging.info(f"{prefix} DataLoader buffer size: {buffer_size}")
            stats["dataloader_buffer_size"] = buffer_size
        except Exception:
            pass
    
    # Log to wandb with memory/ prefix
    if log_to_wandb:
        try:
            wandb_stats = {f"memory/{k}": v for k, v in stats.items()}
            wandb.log(wandb_stats, step=step)
        except Exception:
            pass

    return stats


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    ft_config: _config.FineTuneConfig | None = None,
    start_new: bool = False,
):
    # Only worker 0 should initialize wandb to avoid file conflicts and duplicate runs
    if not enabled or jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming and not start_new:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        base_name = config.exp_name if config.exp_name else config.name
        run_name = f"{base_name}/{ft_config.name}" if ft_config is not None else base_name
        wandb.init(
            name=run_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group=config.wandb_group,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.ActorCriticTrainState, Any]:
    """Initialize training state for a value function model (and optional policy)."""
    critic_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(f"Expected BaseValueFunctionConfig, got {type(config.model)}")
    if isinstance(config.model, _value_fn_impl.ValueFunctionConfig):
        model_config: _value_fn.BaseValueFunctionConfig = dataclasses.replace(
            config.model, action_horizon = config.action_horizon,
        )
    else:
        model_config = config.model

    def init_critic(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            nnx_utils.replace_state_from_pure_dict_numeric_key_compat(state, partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        weight_dtype = jnp.dtype(model_config.weight_dtype) if hasattr(model_config, "weight_dtype") else jnp.float32
        target_dtype = jnp.dtype(model_config.target_dtype) if hasattr(model_config, "target_dtype") else jnp.float32
        params = nnx_utils.state_map(
            params,
            config.trainable_filter,
            lambda p: p.replace(p.value.astype(weight_dtype)),
        )
        params = nnx_utils.state_map(
            params,
            nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*"),
            lambda p: p.replace(p.value.astype(target_dtype)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=critic_tx,
            opt_state=critic_tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else jax.tree.map(jnp.copy, params),
        )

    policy_tx = None
    if config.policy is not None:
        policy_schedule = config.policy_lr_schedule if config.policy_lr_schedule is not None else config.lr_schedule
        policy_tx = _optimizer.create_optimizer(config.optimizer, policy_schedule, weight_decay_mask=None)

    def init_policy(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        if config.policy is None or policy_tx is None:
            raise ValueError("Config does not specify a policy")
        policy_model = config.policy.create(rng)

        if partial_params is not None:
            graphdef, state = nnx.split(policy_model)
            nnx_utils.replace_state_from_pure_dict_numeric_key_compat(state, partial_params)
            policy_model = nnx.merge(graphdef, state)

        policy_params = nnx.state(policy_model)
        policy_params = nnx_utils.state_map(
            policy_params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        return training_utils.TrainState(
            step=0,
            params=policy_params,
            model_def=nnx.graphdef(policy_model),
            tx=policy_tx,
            opt_state=policy_tx.init(policy_params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    def init_actor_critic(
        rng: at.KeyArrayLike,
        critic_partial_params: at.Params | None = None,
        policy_partial_params: at.Params | None = None,
    ) -> training_utils.ActorCriticTrainState:
        rng, critic_rng, policy_rng = jax.random.split(rng, 3)
        critic_state = init_critic(critic_rng, critic_partial_params)

        policy_state = None
        if config.policy is not None:
            policy_state = init_policy(policy_rng, policy_partial_params)

        return training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)

    train_state_shape = jax.eval_shape(init_actor_critic, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    critic_partial_params = _load_weights_and_validate(
        config.weight_loader, train_state_shape.critic.params.to_pure_dict()
    )
    policy_partial_params = None
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Pre-shard partial_params to match the FSDP output sharding so each chip
    # only receives its shard (~x/N GB) instead of the full replicated copy
    # (~x GB). This avoids OOM during init on memory-constrained devices.
    critic_params_sharding = sharding.fsdp_sharding(critic_partial_params, mesh)
    policy_params_sharding = None

    # `jax.device_put(tree, sharding)` calls `multihost_utils.assert_equal`
    # under the hood, which `broadcast_one_to_all`s each leaf with
    # `out_shardings=PartitionSpec()` (REPLICATED). For very large param trees
    # like Gemma 4 e2b (~27 GB total, largest leaf ~9 GB), this replicated
    # output overflows TPU HBM (16 GiB per chip on v5litepod) — the chip can't
    # hold a full unsharded copy of the largest leaf. PaliGemma (224x224)
    # works because its largest leaf is small.
    #
    # Workaround: each host already loaded the *same* deterministic .npz, so
    # the assert_equal step is redundant. Construct each leaf's sharded
    # `jax.Array` directly from local numpy via `make_array_from_callback` —
    # each chip materialises only its own shard, no global broadcast.
    def _make_sharded_array(arr, sharding):
        if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            return arr
        shape = arr.shape
        def _cb(idx):
            return np.asarray(arr[idx])
        return jax.make_array_from_callback(shape, sharding, _cb)
    critic_partial_params = jax.tree.map(_make_sharded_array, critic_partial_params, critic_params_sharding)

    donate_argnums = (1,) if policy_partial_params is None else (1, 2)
    in_shardings = (
        replicated_sharding,
        critic_params_sharding,
        policy_params_sharding,
    )

    train_state = jax.jit(
        init_actor_critic,
        donate_argnums = donate_argnums,
        in_shardings = in_shardings,
        out_shardings = state_sharding,
    )(init_rng, critic_partial_params, policy_partial_params)

    return train_state, state_sharding


@at.typecheck
def value_function_train_step(
    config: _config.TrainConfig,
    lr_schedule: optax.Schedule,
    state: training_utils.TrainState,
    policy_state: training_utils.TrainState | None,
    batch: dict[str, Any],
    rng: at.KeyArrayLike,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step for value function.

    Args:
        config: Training configuration.
        state: Current training state.
        batch: Batch of data.
        rng: Random key for algorithms that need stochasticity (e.g., SAC).

    Returns:
        Tuple of (new_state, info_dict).
    """
    # pdb.set_trace()
    model = nnx.merge(state.model_def, state.params)
    policy = None if policy_state is None else nnx.merge(policy_state.model_def, policy_state.params)

    if isinstance(config.model, _value_fn.BaseMultiValueFunctionConfig):
        transition = _value_fn.MultiTransition.from_batch(batch)
    else:
        transition = _value_fn.Transition.from_batch(batch)

    def loss_fn(model: _value_fn.BaseValueFunction):
        # compute_loss returns (per_sample_loss, info_dict)
        per_sample_loss, value_info = model.compute_loss(transition, train=True, rng=rng, policy=policy)
        mean_loss = jnp.mean(per_sample_loss)
        return mean_loss, value_info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, value_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)

    # Call post_step_update for algorithms that need it (e.g., SAC target network update)
    model.post_step_update()

    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # The trailing-name regex uses `[/_]` so it also matches the stacked-layer-params
    # layout, where per-layer scales/biases live under names like
    # `g0__pre_attention_norm__scale` (separator `__` instead of `/`).
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*[/_](bias|scale|pos_embedding|input_embedding)")),
            nnx.Not(nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    target_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*target_(q_)?(network|head)/.*"),
            nnx.Not(nnx_utils.PathRegex(".*[/_](bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # Batch statistics
    # For RoboCasa configs the state and action are in bimanual-EEF layout: dims 0:7
    # are the right arm (real values), dims 7:14 are zero-filler for the absent left
    # arm. Restrict stats to the meaningful slice so the filler doesn't distort them.
    is_robocasa = isinstance(config.data, _config.RLDSRoboCasaDataConfig)
    obs_state = transition.observation.state[..., :7] if is_robocasa else transition.observation.state
    action = transition.action[..., :7] if is_robocasa else transition.action
    batch_stats = {
        # Observation stats
        "batch/obs_mean": jnp.mean(obs_state),
        "batch/obs_std": jnp.std(obs_state),
        "batch/obs_min": jnp.min(obs_state),
        "batch/obs_max": jnp.max(obs_state),
        "batch/obs_out_of_range_frac": jnp.mean((jnp.abs(obs_state) >= 1.001).astype(jnp.float32)),
        # Action stats
        "batch/action_mean": jnp.mean(action),
        "batch/action_std": jnp.std(action),
        "batch/action_min": jnp.min(action),
        "batch/action_max": jnp.max(action),
        "batch/action_out_of_range_frac": jnp.mean((jnp.abs(action) >= 1.001).astype(jnp.float32)),
        # Reward stats
        "batch/reward_mean": jnp.mean(transition.reward),
        "batch/reward_std": jnp.std(transition.reward),
        "batch/reward_min": jnp.min(transition.reward),
        "batch/reward_max": jnp.max(transition.reward),
        # MC return stats
        "batch/mc_return_mean": jnp.mean(transition.mc_return),
        "batch/mc_return_std": jnp.std(transition.mc_return),
        "batch/mc_return_min": jnp.min(transition.mc_return),
        "batch/mc_return_max": jnp.max(transition.mc_return),
        # Termination stats
        "batch/termination_mean": jnp.mean(transition.termination.astype(jnp.float32)),
        "batch/termination_std": jnp.std(transition.termination.astype(jnp.float32)),
        "batch/termination_min": jnp.min(transition.termination.astype(jnp.float32)),
        "batch/termination_max": jnp.max(transition.termination.astype(jnp.float32)),
        # Truncation stats
        "batch/truncation_mean": jnp.mean(transition.truncation.astype(jnp.float32)),
        "batch/truncation_std": jnp.std(transition.truncation.astype(jnp.float32)),
        "batch/truncation_min": jnp.min(transition.truncation.astype(jnp.float32)),
        "batch/truncation_max": jnp.max(transition.truncation.astype(jnp.float32)),
    }
    if transition.counterfactual_next_actions is not None:
        batch_stats["batch/counterfactual_next_actions_mean"] = jnp.mean(transition.counterfactual_next_actions)
        batch_stats["batch/counterfactual_next_actions_std"] = jnp.std(transition.counterfactual_next_actions)
        batch_stats["batch/counterfactual_next_actions_min"] = jnp.min(transition.counterfactual_next_actions)
        batch_stats["batch/counterfactual_next_actions_max"] = jnp.max(transition.counterfactual_next_actions)
        batch_stats["batch/counterfactual_next_actions_out_of_range_frac"] = jnp.mean(
            (jnp.abs(transition.counterfactual_next_actions) >= 1.001).astype(jnp.float32)
        )

    batch_size = transition.reward.shape[0]

    value_stats = {}
    non_terminal = ~transition.termination
    num_non_terminal = jnp.maximum(jnp.sum(non_terminal.astype(jnp.float32)), 1.0)

    for key in ("predicted_value", "target_value", "td_error"):
        if key in value_info:
            arr = value_info.pop(key)
            value_stats[f"{key}_mean"] = jnp.sum(arr * non_terminal.astype(arr.dtype)) / num_non_terminal
            value_stats[f"{key}_std"] = jnp.std(arr)

    if "next_value" in value_info:
        next_val = value_info.pop("next_value")
        value_stats["next_value_mean"] = jnp.sum(next_val * non_terminal.astype(next_val.dtype)) / num_non_terminal

    if "mc_loss" in value_info:
        mc_loss_arr = value_info.pop("mc_loss")
        value_stats["mc_loss"] = jnp.mean(mc_loss_arr)

    if "next_token_loss" in value_info:
        next_token_loss_arr = value_info.pop("next_token_loss")
        value_stats["next_token_loss"] = jnp.mean(next_token_loss_arr)

    grads_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), grads)
    kernel_params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), kernel_params)
    target_params_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), target_params)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads_f32),
        "param_norm": optax.global_norm(kernel_params_f32),
        "target_param_norm": optax.global_norm(target_params_f32),
        "learning_rate": lr_schedule(state.step),
        **value_info,
        **value_stats,
        **batch_stats,
    }

    if "steps_to_subtask_end" in batch:
        steps = jnp.asarray(batch["steps_to_subtask_end"]).astype(jnp.float32)
        info["batch/steps_to_subtask_end_mean"] = jnp.mean(steps)
        info["batch/steps_to_subtask_end_std"] = jnp.std(steps)
        info["batch/steps_to_subtask_end_min"] = jnp.min(steps)
        info["batch/steps_to_subtask_end_max"] = jnp.max(steps)
    return new_state, info


@at.typecheck
def policy_train_step(
    config: _config.TrainConfig,
    lr_schedule: optax.Schedule,
    critic_state: training_utils.TrainState,
    policy_state: training_utils.TrainState,
    batch: dict[str, Any],
    rng: at.KeyArrayLike,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step for policy.

    Args:
        config: Training configuration.
        lr_schedule: Learning rate schedule.
        critic_state: Critic (value function) train state (frozen during policy update).
        policy_state: Policy train state to update.
        batch: Batch of data. For single-transition: state [batch, state_dim], actions [batch, ah, ad].
            For multi-transition: state [batch, n, state_dim], actions [batch, n, ah, ad].
        rng: Random key.

    Returns:
        Tuple of (new_policy_state, info_dict).
    """
    policy = nnx.merge(policy_state.model_def, policy_state.params)
    critic = nnx.merge(critic_state.model_def, critic_state.params)

    state = jnp.asarray(batch["state"])
    actions = jnp.asarray(batch["actions"])

    observation = _model.Observation(
        images={},
        image_masks={},
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )

    def loss_fn(policy):
        loss, info = config.policy_extraction.compute_loss(
            policy=policy,
            observation=observation,
            rng=rng,
            data_action=actions,
            value_function=critic,
        )
        return jnp.mean(loss), info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, policy_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(policy)

    params = policy_state.params.filter(config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(grads, policy_state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(policy, new_params)
    new_params = nnx.state(policy)

    new_policy_state = dataclasses.replace(
        policy_state,
        step=policy_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )

    grads_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), grads)

    info = {
        "policy/loss": loss,
        "policy/grad_norm": optax.global_norm(grads_f32),
        "policy/learning_rate": lr_schedule(policy_state.step),
        **{f"policy/{k}": v for k, v in policy_info.items()},
    }
    return new_policy_state, info


def get_trajectory_frames(dataset, episode_idx: int) -> list[dict]:
    """Extract all frames from a specific episode in the dataset.

    Supports both NumpyDataset and LeRobotDataset.
    """
    # Check if this is a NumpyDataset (has get_episode_frames method)
    if hasattr(dataset, "get_episode_frames"):
        return dataset.get_episode_frames(episode_idx)

    # LeRobotDataset path: use episode_data_index
    episode_data_index = dataset.episode_data_index
    start_idx = episode_data_index["from"][episode_idx].item()
    end_idx = episode_data_index["to"][episode_idx].item()

    frames = []
    for idx in range(start_idx, end_idx):
        frame = dataset[idx]
        frames.append(frame)
    return frames


def generate_validation_plots(
    model: _value_fn.BaseValueFunction,
    dataset,
    val_episode_indices: list[int],
    step: int,
    *,
    action_conditioned: bool,
    data_config: _config.DataConfig,
) -> dict:
    """Generate validation plots comparing predicted values vs MC returns.

    For multi-transition models, generates two types of plots:
    - random_trajectory_transitions: n-1 random frames + current frame, use only current output
    - continuous_chunks: process n consecutive frames, use all n outputs

    If the environment is PointMaze, also plots the ground-truth Q-values from the oracle
    and computes oracle ranking metrics.

    Args:
        model: The value function model
        dataset: The dataset
        val_episode_indices: List of episode indices to plot
        step: Current training step
        action_conditioned: Whether the model is action-conditioned (Q vs V)
        data_config: Data configuration containing norm_stats for normalization

    Returns:
        Dictionary of wandb images keyed by trajectory index, plus oracle ranking metrics
    """

    is_pointmaze = data_config.minari_dataset_id and "pointmaze" in data_config.minari_dataset_id.lower()
    oracle = None
    if is_pointmaze:
        from openpi.point_maze_utils.point_maze_oracle import PointMazeOracle

        oracle = PointMazeOracle()

    # Detect multi-transition model - check network or q_network (for IQL models)
    primary_network = getattr(model, "network", None) or getattr(model, "q_network", None)
    assert primary_network is not None, f"Model {type(model).__name__} has no network or q_network attribute"
    is_multi_transition = hasattr(primary_network, "num_transitions_per_sample")
    num_transitions = None
    if is_multi_transition:
        num_transitions = primary_network.num_transitions_per_sample

    # Create normalization transform
    normalize = _transforms.Normalize(
        data_config.norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        strict=False,
    )

    images = {}
    all_ranking_metrics: list[dict[str, float]] = []

    for ep_idx in val_episode_indices:
        try:
            frames = get_trajectory_frames(dataset, ep_idx)
        except Exception as e:
            logging.warning(f"Could not load episode {ep_idx}: {e}")
            continue

        if len(frames) == 0:
            continue

        # Skip the first frame for PointMaze to avoid artifacts from goal changes
        # (first frame may have a different goal than the rest of the trajectory)
        if is_pointmaze:
            frames = frames[1:]

        # Collect mc_returns for ground truth
        mc_returns = []
        for frame in frames:
            mc_return = frame.get("mc_return")
            if mc_return is None:
                continue
            if hasattr(mc_return, "numpy"):
                mc_return = mc_return.numpy()
            mc_return = np.asarray(mc_return).item() if np.asarray(mc_return).size == 1 else np.asarray(mc_return)
            mc_returns.append(float(mc_return))

        if len(mc_returns) == 0:
            continue

        # Compute oracle values if available
        oracle_values = None
        fixed_goal = None
        if oracle:
            oracle_values = []
            # Fixed goal for PointMaze consistency.
            sample_state = np.array(frames[-1]["state"])
            fixed_goal = sample_state[2:4]

            for frame in frames:
                if frame.get("mc_return") is None:
                    continue
                # Extract PointMaze state/goal/action from flattened observation
                # Config: [achieved_goal(2), desired_goal(2), observation(4)]
                # observation is (x, y, vx, vy)
                state_vec = np.array(frame["state"])
                action_vec = np.array(frame["actions"])

                if state_vec.shape[-1] != 8:
                    raise ValueError(f"Oracle expects PointMaze observation with 8 dimensions, got {state_vec.shape}")

                goal = fixed_goal
                state = state_vec[4:8]
                q = oracle.compute_dense_distance(state, goal, action_vec)
                oracle_values.append(q)

                # Check constraint: Oracle (Optimal) >= MC (Suboptimal)
                # Allow a small margin (0.5) for discrete/continuous approx errors
                if frame.get("mc_return") is not None:
                    mc = float(frame["mc_return"])
                    if q < mc - 0.5:
                        logging.warning(
                            f"Oracle Value Constraint Violated! Episode {ep_idx}, Step {step}, "
                            f"Oracle={q:.4f}, MC={mc:.4f}, Diff={q - mc:.4f}. "
                            f"State={state}, Goal={goal}. "
                            "Oracle should be >= MC (Optimal >= Policy)."
                        )

            # Compute oracle ranking metrics for this episode
            ep_ranking_metrics = _compute_oracle_ranking_metrics(
                model,
                oracle,
                frames,
                num_transitions,
                normalize,
                fixed_goal,
                action_conditioned=action_conditioned,
            )
            if ep_ranking_metrics:
                all_ranking_metrics.append(ep_ranking_metrics)

        if is_multi_transition:
            # Generate plots for multi-transition models
            images.update(
                _generate_multi_transition_plots(
                    model,
                    frames,
                    mc_returns,
                    ep_idx,
                    step,
                    num_transitions,
                    normalize,
                    action_conditioned,
                    _model,
                    oracle_values,
                )
            )
        else:
            # Standard single-transition plot
            predicted_values = _compute_single_transition_values(model, frames, normalize, action_conditioned, _model)
            images[f"val/episode_{ep_idx}"] = _create_value_plot(
                mc_returns, predicted_values, ep_idx, step, "", oracle_values
            )

    # Aggregate ranking metrics across episodes
    if all_ranking_metrics:
        for key in all_ranking_metrics[0]:
            values = [m[key] for m in all_ranking_metrics if key in m]
            if values:
                images[key] = float(np.mean(values))

    return images


def _normalize_frame(frame: dict, normalize, _model) -> tuple:
    """Extract and normalize state/action from a frame."""
    state = frame.get("state")
    if hasattr(state, "numpy"):
        state = state.numpy()
    state = np.asarray(state, dtype=np.float32)
    normalized_data = normalize({"state": state})
    normalized_state = normalized_data["state"]

    action = frame.get("actions", frame.get("action"))
    if action is not None:
        action = np.asarray(action, dtype=np.float32)
        normalized_act_data = normalize({"actions": action})
        action = normalized_act_data["actions"]

    return normalized_state, action


def _compute_single_transition_values(model, frames, normalize, action_conditioned, _model) -> list[float]:
    """Compute predicted values for single-transition models."""
    predicted_values = []
    for frame in frames:
        if frame.get("mc_return") is None:
            continue
        normalized_state, action = _normalize_frame(frame, normalize, _model)
        obs = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(normalized_state[None, ...]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        act = jnp.asarray(action[None, ...]) if action_conditioned and action is not None else None
        pred_value = model.compute_value(obs, act, take_min_over_ensemble=True)
        predicted_values.append(float(jax.device_get(pred_value[0])))
    return predicted_values


def _generate_multi_transition_plots(
    model,
    frames,
    mc_returns,
    ep_idx,
    step,
    num_transitions,
    normalize,
    action_conditioned,
    _model,
    oracle_values=None,
) -> dict:
    """Generate plots for multi-transition models.

    Returns two plots:
    1. random_trajectory_transitions: For each frame, sample n-1 random other frames
       from the trajectory, predict jointly, use only the current frame's output.
    2. continuous_chunks: Process consecutive chunks of n frames, use all outputs.
    """
    images = {}
    rng = np.random.default_rng(42)

    # Precompute normalized states and actions
    normalized_frames = []
    for frame in frames:
        if frame.get("mc_return") is None:
            continue
        state, action = _normalize_frame(frame, normalize, _model)
        normalized_frames.append((state, action))

    if len(normalized_frames) < num_transitions:
        return {}

    # Strategy 1: random_trajectory_transitions
    # For each frame, sample n-1 random frames + current frame, predict, use current output
    pred_random = []
    for i in range(len(normalized_frames)):
        # Sample n-1 other indices (can repeat for short trajectories)
        other_indices = [j for j in range(len(normalized_frames)) if j != i]
        if len(other_indices) < num_transitions - 1:
            sampled = rng.choice(other_indices, size=num_transitions - 1, replace=True)
        else:
            sampled = rng.choice(other_indices, size=num_transitions - 1, replace=False)

        # Current frame goes last so its output is at index n-1
        all_indices = [*sampled, i]
        current_output_idx = num_transitions - 1

        # Build multi-transition input
        states = np.stack([normalized_frames[j][0] for j in all_indices], axis=0)
        obs = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(states[None, ...]),  # [1, n, state_dim]
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        action = None
        if action_conditioned:
            actions = np.stack([normalized_frames[j][1] for j in all_indices], axis=0)
            action = jnp.asarray(actions[None, ...])  # [1, n, action_dim]

        pred_values = model.compute_value(obs, action, take_min_over_ensemble=True)  # [1, n]
        pred_random.append(float(jax.device_get(pred_values[0, current_output_idx])))

    images[f"val/episode_{ep_idx}_random"] = _create_value_plot(
        mc_returns, pred_random, ep_idx, step, " (random context)", oracle_values
    )

    # Strategy 2: continuous_chunks
    # Process n consecutive frames at a time, use all outputs
    pred_chunks = [None] * len(normalized_frames)
    for start in range(0, len(normalized_frames) - num_transitions + 1, num_transitions):
        chunk_indices = list(range(start, start + num_transitions))
        states = np.stack([normalized_frames[j][0] for j in chunk_indices], axis=0)
        obs = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(states[None, ...]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        action = None
        if action_conditioned:
            actions = np.stack([normalized_frames[j][1] for j in chunk_indices], axis=0)
            action = jnp.asarray(actions[None, ...])

        pred_values = model.compute_value(obs, action, take_min_over_ensemble=True)  # [1, n]
        for local_idx, global_idx in enumerate(chunk_indices):
            pred_chunks[global_idx] = float(jax.device_get(pred_values[0, local_idx]))

    # Handle remaining frames if trajectory length not divisible by n
    remaining = [i for i, v in enumerate(pred_chunks) if v is None]
    if remaining:
        # Fill with last n frames
        start = len(normalized_frames) - num_transitions
        chunk_indices = list(range(start, start + num_transitions))
        states = np.stack([normalized_frames[j][0] for j in chunk_indices], axis=0)
        obs = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(states[None, ...]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        action = None
        if action_conditioned:
            actions = np.stack([normalized_frames[j][1] for j in chunk_indices], axis=0)
            action = jnp.asarray(actions[None, ...])

        pred_values = model.compute_value(obs, action, take_min_over_ensemble=True)
        for local_idx, global_idx in enumerate(chunk_indices):
            if pred_chunks[global_idx] is None:
                pred_chunks[global_idx] = float(jax.device_get(pred_values[0, local_idx]))

    images[f"val/episode_{ep_idx}_chunks"] = _create_value_plot(
        mc_returns, pred_chunks, ep_idx, step, " (continuous chunks)", oracle_values
    )

    return images


def _create_value_video(
    mc_returns: list,
    predicted_values: list,
    ep_idx: int,
    step: int,
    suffix: str,
    oracle_values: list | None,
    frame_images: list[np.ndarray],
    fps: int,
    subtask_texts: list[str] | None = None,
    output_dir: str | None = None,
    plot_key: str = "",
) -> "wandb.Video | str":
    """Create a 2x2 layout video: left wrist (top-left), right wrist (bottom-left),
    value plot (top-right), base camera (bottom-right). Subtask list shown below the plot.

    When output_dir is set, saves an MP4 to disk via imageio and returns the file path.
    When output_dir is None, returns a wandb.Video (GIF)."""
    T = len(mc_returns)
    timesteps = np.arange(T)
    video_frames = []

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_left_wrist, ax_val = axes[0, 0], axes[0, 1]
    ax_right_wrist, ax_base = axes[1, 0], axes[1, 1]

    # Pre-compute subtask caption and adjust layout once
    subtask_caption = None
    if subtask_texts:
        numbered_subtasks = [f"{i+1}. {text}" for i, text in enumerate(subtask_texts)]
        lines = []
        for i in range(0, len(numbered_subtasks), 3):
            lines.append("  ".join(numbered_subtasks[i:i+3]))
        subtask_caption = "Subtasks:\n" + "\n".join(lines)
        num_lines = len(lines) + 1
        bottom_margin = 0.10 + 0.03 * num_lines
        fig.subplots_adjust(bottom = bottom_margin)

    for t in range(T):
        for ax in axes.flat:
            ax.cla()

        ax_left_wrist.imshow(frame_images[t][0])
        ax_left_wrist.axis("off")
        ax_left_wrist.set_title("Left Wrist", fontsize=12)

        ax_right_wrist.imshow(frame_images[t][1])
        ax_right_wrist.axis("off")
        ax_right_wrist.set_title("Right Wrist", fontsize=12)

        ax_base.imshow(frame_images[t][2])
        ax_base.axis("off")
        ax_base.set_title(f"Timestep {t}", fontsize=12)

        ax_val.plot(timesteps, mc_returns, label="MC Returns", color="blue", linewidth=2)
        ax_val.plot(timesteps, predicted_values, label="Predicted Value", color="orange", linewidth=2, linestyle="--")
        if oracle_values is not None:
            ax_val.plot(timesteps, oracle_values, label="Oracle Q-Value", color="green", linewidth=2, linestyle="-.", alpha=0.7)
        ax_val.axvline(x = t, color = "red", linewidth = 2, alpha = 0.8)
        ax_val.set_xlabel("Timestep", fontsize=12)
        ax_val.set_ylabel("Value", fontsize=12)
        ax_val.set_title(f"Episode {ep_idx} - Step {step}{suffix}", fontsize=14)
        ax_val.legend(fontsize=11)
        ax_val.grid(visible=True, alpha=0.3)

        if subtask_caption is None:
            plt.tight_layout()
        else:
            for txt in fig.texts:
                txt.remove()
            fig.text(0.5, 0.01, subtask_caption, ha='center', va='bottom', fontsize=9,
                     family='monospace', linespacing=1.5)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        video_frames.append(buf.copy())

    plt.close(fig)

    if output_dir is not None:
        import imageio

        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_") if plot_key else f"ep{ep_idx}_step{step}"
        out_path = os.path.join(output_dir, f"{sanitized_key}.mp4")
        # Ensure dimensions are divisible by 16 (imageio macro_block_size)
        h, w = video_frames[0].shape[:2]
        h_crop = h - (h % 16)
        w_crop = w - (w % 16)
        cropped_frames = [frame[:h_crop, :w_crop] for frame in video_frames]
        imageio.mimsave(out_path, cropped_frames, format = "mp4", fps = fps, codec = "libx264", quality = 8)
        logging.info(f"Saved video to {out_path}")
        return out_path

    video_array = np.stack(video_frames).transpose(0, 3, 1, 2)
    # GIF is used because wandb renders it inline. mp4 shows as "File type unknown" in the wandb UI.
    # GIF's 256-color palette quantization causes a visible quality drop (color jitter appearance),
    # but the video remains interpretable.
    return wandb.Video(video_array, fps = fps, format = "gif")


def _create_value_plot(
    mc_returns,
    predicted_values,
    ep_idx,
    step,
    suffix,
    oracle_values=None,
    subtask_texts: list[str] | None = None,
    plot_video: bool = False,
    frame_images: list[np.ndarray] | None = None,
    fps: int = 10,
    output_dir: str | None = None,
    plot_key: str = "",
) -> "wandb.Image | wandb.Video | str":
    """Create a matplotlib plot comparing MC returns vs predicted values.

    Args:
        mc_returns: List of MC return values.
        predicted_values: List of predicted values.
        ep_idx: Episode index (from dataset's episode_index, used as title).
        step: Training step.
        suffix: Suffix for the title.
        oracle_values: Optional list of oracle Q-values.
        subtask_texts: Optional ordered list of subtask segment texts for caption.
        plot_video: If True and frame_images are provided, returns a wandb.Video instead of wandb.Image.
        frame_images: Optional list of (3, 224, 224, 3) uint8 arrays (left wrist, right wrist, base_0) per timestep.
        fps: Frame rate of the episode, used when encoding the output video.
        output_dir: If set, save to disk instead of returning a wandb object.
        plot_key: Key used to derive the filename when saving to disk.
    """
    if plot_video and frame_images is not None and len(frame_images) == len(mc_returns):
        return _create_value_video(
            mc_returns, predicted_values, ep_idx, step, suffix, oracle_values, frame_images, fps, subtask_texts,
            output_dir = output_dir, plot_key = plot_key,
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    timesteps = np.arange(len(mc_returns))

    ax.plot(timesteps, mc_returns, label="MC Returns", color="blue", linewidth=2)
    ax.plot(
        timesteps,
        predicted_values,
        label="Predicted Value",
        color="orange",
        linewidth=2,
        linestyle="--",
    )

    if oracle_values is not None:
        ax.plot(
            timesteps,
            oracle_values,
            label="Oracle Q-Value",
            color="green",
            linewidth=2,
            linestyle="-.",
            alpha=0.7,
        )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Episode {ep_idx} - Step {step}{suffix}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(visible=True, alpha=0.3)
    
    # Add numbered subtask list below x-axis label if provided
    if subtask_texts:
        # Number the subtask segments and group 3 per line
        numbered_subtasks = [f"{i+1}. {text}" for i, text in enumerate(subtask_texts)]
        lines = []
        for i in range(0, len(numbered_subtasks), 3):
            line = "  ".join(numbered_subtasks[i:i+3])
            lines.append(line)
        caption = "Subtasks:\n" + "\n".join(lines)
        
        # Calculate bottom margin based on number of lines
        num_lines = len(lines) + 1  # +1 for "Subtasks:" header
        bottom_margin = 0.10 + 0.03 * num_lines
        plt.subplots_adjust(bottom=bottom_margin)
        
        # Place text below the x-axis label
        fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9, 
                 family='monospace', linespacing=1.5)
    else:
        plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_") if plot_key else f"ep{ep_idx}_step{step}"
        out_path = os.path.join(output_dir, f"{sanitized_key}.png")
        fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
        plt.close(fig)
        logging.info(f"Saved plot to {out_path}")
        return out_path

    img = wandb.Image(fig)
    plt.close(fig)
    return img


# Fixed action grid for PointMaze (2D action space [-1, 1] x [-1, 1])
FIXED_ACTION_GRID = np.array(
    [
        [-1.0, -1.0],
        [-1.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)


def _compute_oracle_ranking_metrics(
    model,
    oracle,
    frames,
    num_transitions: int | None,
    normalize,
    fixed_goal: np.ndarray,
    *,
    action_conditioned: bool,
) -> dict[str, float]:
    """Compute ranking correlation between predicted and oracle Q-values.

    Returns two metrics using Kendall's tau:
    - trajectory_patch_rank: Avg tau for continuous patches of n transitions
    - fixed_action_rank: Avg tau for fixed action set across all states
    """
    metrics = {}
    n = num_transitions if num_transitions is not None else 1

    # Precompute normalized frames with raw state/action for oracle
    normalized_frames = []
    for frame in frames:
        if frame.get("mc_return") is None:
            continue
        norm_state, norm_action = _normalize_frame(frame, normalize, _model)
        raw_state = np.array(frame["state"])
        raw_action = np.array(frame.get("actions", frame.get("action")))
        mc_return = frame["mc_return"]
        if hasattr(mc_return, "numpy"):
            mc_return = mc_return.numpy()
        mc_return = float(np.asarray(mc_return).item() if np.asarray(mc_return).size == 1 else np.asarray(mc_return))
        normalized_frames.append(
            {
                "norm_state": norm_state,
                "norm_action": norm_action,
                "raw_state": raw_state,
                "raw_action": raw_action,
                "mc_return": mc_return,
            }
        )

    if len(normalized_frames) < n:
        return {}

    # =============================================================================
    # Metric 1: Trajectory Patch Ranking
    # For continuous chunks of n transitions, compare predicted vs oracle/MC ordering
    # =============================================================================
    is_multi_transition = n > 1
    # Use consistent chunk size for fair comparison between model types
    chunk_size = 8
    oracle_patch_taus = []
    mc_patch_taus = []
    for start in range(0, len(normalized_frames) - chunk_size + 1, chunk_size):
        chunk = normalized_frames[start : start + chunk_size]

        # Compute predicted values
        if is_multi_transition:
            # Multi-transition: batch all n transitions together
            states = np.stack([f["norm_state"] for f in chunk], axis=0)
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jnp.asarray(states[None, ...]),  # [1, n, state_dim]
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )
            action = None
            if action_conditioned:
                actions = np.stack([f["norm_action"] for f in chunk], axis=0)
                action = jnp.asarray(actions[None, ...])  # [1, n, action_dim]

            pred_values = jax.device_get(model.compute_value(obs, action, take_min_over_ensemble=True))
            if pred_values.ndim == 2:
                pred_values = pred_values[0]  # [n]
        else:
            # Single-transition: call Q-function separately for each state/action
            pred_values = []
            for f in chunk:
                obs = _model.Observation(
                    images={},
                    image_masks={},
                    state=jnp.asarray(f["norm_state"][None, ...]),  # [1, state_dim]
                    tokenized_prompt=None,
                    tokenized_prompt_mask=None,
                )
                action = None
                if action_conditioned:
                    action = jnp.asarray(f["norm_action"][None, ...])  # [1, action_dim]
                pred_value = jax.device_get(model.compute_value(obs, action, take_min_over_ensemble=True))
                pred_values.append(float(pred_value[0]))
            pred_values = np.array(pred_values)

        oracle_values = []
        for f in chunk:
            raw_state = f["raw_state"]
            assert raw_state.shape[-1] == 8, f"Expected state dim 8, got {raw_state.shape[-1]}"
            state = raw_state[4:8]
            q = oracle.compute_dense_distance(state, fixed_goal, f["raw_action"])
            oracle_values.append(q)

        # Compute MC return ranking
        mc_values = np.array([f["mc_return"] for f in chunk])

        assert len(oracle_values) == chunk_size, f"Expected {chunk_size} oracle values, got {len(oracle_values)}"
        assert len(pred_values) == chunk_size, f"Expected {chunk_size} predictions, got {len(pred_values)}"
        assert len(mc_values) == chunk_size, f"Expected {chunk_size} MC values, got {len(mc_values)}"

        tau, _ = scipy.stats.kendalltau(pred_values, oracle_values)
        if not np.isnan(tau):
            oracle_patch_taus.append(tau)

        tau, _ = scipy.stats.kendalltau(pred_values, mc_values)
        if not np.isnan(tau):
            mc_patch_taus.append(tau)

    if oracle_patch_taus:
        metrics["val/oracle_rank_trajectory_patches"] = float(np.mean(oracle_patch_taus))
    if mc_patch_taus:
        metrics["val/mc_rank_trajectory_patches"] = float(np.mean(mc_patch_taus))

    # =============================================================================
    # Metric 2: Fixed Action Set Ranking
    # For each state, compare ordering of Q(s, a) across fixed actions
    # =============================================================================
    if not action_conditioned:
        return metrics

    action_taus = []
    for frame_data in normalized_frames:
        raw_state = frame_data["raw_state"]
        assert raw_state.shape[-1] == 8, f"Expected state dim 8, got {raw_state.shape[-1]}"
        state_4d = raw_state[4:8]
        norm_state = frame_data["norm_state"]

        # Compute oracle Q-values for all fixed actions
        oracle_qs = []
        for a in FIXED_ACTION_GRID:
            q = oracle.compute_dense_distance(state_4d, fixed_goal, a)
            oracle_qs.append(q)
        oracle_qs = np.array(oracle_qs)

        # Compute predicted Q-values for all fixed actions
        # Normalize actions using the same transform
        norm_actions = []
        for a in FIXED_ACTION_GRID:
            norm_a_data = normalize({"actions": a})
            norm_actions.append(norm_a_data["actions"])
        norm_actions = np.stack(norm_actions, axis=0)

        if is_multi_transition:
            # Multi-transition: use only first n actions from grid
            actions_to_use = min(n, len(FIXED_ACTION_GRID))
            batch_states = np.tile(norm_state[None, ...], (actions_to_use, 1))
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jnp.asarray(batch_states[None, ...]),  # [1, n, state_dim]
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )
            action = jnp.asarray(norm_actions[:actions_to_use][None, ...])  # [1, n, action_dim]
            pred_qs = jax.device_get(model.compute_value(obs, action, take_min_over_ensemble=True))
            if pred_qs.ndim == 2:
                pred_qs = pred_qs[0]  # [n]
            oracle_qs = oracle_qs[:actions_to_use]
        else:
            # Single-transition: call Q-function separately for each action
            pred_qs = []
            for norm_a in norm_actions:
                obs = _model.Observation(
                    images={},
                    image_masks={},
                    state=jnp.asarray(norm_state[None, ...]),  # [1, state_dim]
                    tokenized_prompt=None,
                    tokenized_prompt_mask=None,
                )
                action = jnp.asarray(norm_a[None, ...])  # [1, action_dim]
                pred_q = jax.device_get(model.compute_value(obs, action, take_min_over_ensemble=True))
                pred_qs.append(float(pred_q[0]))
            pred_qs = np.array(pred_qs)

        tau, _ = scipy.stats.kendalltau(pred_qs, oracle_qs)
        if not np.isnan(tau):
            action_taus.append(tau)

    if action_taus:
        metrics["val/oracle_rank_fixed_actions"] = float(np.mean(action_taus))

    return metrics


_render_thread: threading.Thread | None = None


def _create_attn_plot(
    attn_scores: list[np.ndarray],
    ep_idx: int,
    step: int,
    repo_id: str,
    action_conditioned: bool,
    output_dir: str | None = None,
    plot_key: str = "",
) -> "wandb.Image | str":
    """Create a line plot of per-modality CLS attention scores over time."""
    scores = np.stack(attn_scores, axis=0)  # [T, n_modalities]
    timesteps = np.arange(len(scores))
    labels = [f"img{i+1}" for i in range(3)] + ["text"]
    if scores.shape[1] > 4 + int(action_conditioned):
        labels.append("state")
    if action_conditioned:
        labels.append("action")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        ax.plot(timesteps, scores[:, i], label=label)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Mean Attention Scores", fontsize=12)
    ax.set_title(f"Episode {ep_idx} - Step {step} - CLS Attention ({repo_id})", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_") if plot_key else f"attn_ep{ep_idx}_step{step}"
        out_path = os.path.join(output_dir, f"{sanitized_key}.png")
        fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
        plt.close(fig)
        logging.info(f"Saved attention plot to {out_path}")
        return out_path

    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _render_and_log_plots(
    all_predictions: dict,
    all_predictions_neg: dict,
    all_predictions_random: dict,
    all_predictions_shuffled: dict,
    all_attn_scores: dict,
    ep_mc_returns: dict,
    ep_frame_images: dict,
    ep_fps: dict,
    ep_include_masks: dict,
    ep_subtasks: dict,
    ep_negative_subtasks: dict,
    traj_to_repo_ep: dict,
    action_conditioned: bool,
    step: int,
    *,
    all_predictions_counterfactual: dict | None = None,
    output_dir: str | None = None,
) -> None:
    images = {}
    for traj_idx in ep_mc_returns.keys():
        repo_id, ep_idx, part_suffix = traj_to_repo_ep[traj_idx]
        plot_key = f"val/{repo_id.removeprefix('RoboCOIN/')}_episode_{ep_idx}{part_suffix}"
        mc_returns = ep_mc_returns[traj_idx]
        include_masks = ep_include_masks[traj_idx]
        predicted_values = all_predictions[traj_idx]

        if len(predicted_values) != len(mc_returns):
            logging.warning(f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}): mismatch between predictions ({len(predicted_values)}) and mc_returns ({len(mc_returns)})")
            continue

        filtered_mc_returns = []
        filtered_predictions = []
        for mc, pred, mask in zip(mc_returns, predicted_values, include_masks):
            if mask:
                filtered_mc_returns.append(mc)
                filtered_predictions.append(pred)

        if len(filtered_mc_returns) == 0:
            logging.warning(f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}): no frames with include_subtask=True")
            continue

        subtasks = ep_subtasks.get(traj_idx, [])
        logging.info(f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}): {len(filtered_predictions)}/{len(predicted_values)} frames after include_subtask filter, subtasks={subtasks}")

        ep_frame_images[traj_idx] = [img for img, mask in zip(ep_frame_images[traj_idx], include_masks) if mask]

        images[plot_key] = _create_value_plot(
            filtered_mc_returns, filtered_predictions, ep_idx, step, " (RoboCOIN)",
            oracle_values = None, subtask_texts = subtasks,
            plot_video = True, frame_images = ep_frame_images[traj_idx],
            fps = ep_fps[traj_idx],
            output_dir = output_dir, plot_key = plot_key,
        )
        logging.info(f"Repo {repo_id}, episode {ep_idx} plot created")

        attn_scores = all_attn_scores.get(traj_idx, [])
        filtered_attn = [s for s, mask in zip(attn_scores, include_masks) if mask]
        if len(filtered_attn) == len(filtered_mc_returns) and len(filtered_attn) > 0:
            images[f"{plot_key}_attn"] = _create_attn_plot(
                filtered_attn, ep_idx, step, repo_id.removeprefix("RoboCOIN/"), action_conditioned,
                output_dir = output_dir, plot_key = f"{plot_key}_attn",
            )

        negative_subtasks = ep_negative_subtasks.get(traj_idx)
        predicted_values_neg = all_predictions_neg.get(traj_idx, [])
        if negative_subtasks and len(predicted_values_neg) == len(predicted_values):
            filtered_predictions_neg = [pred for pred, mask in zip(predicted_values_neg, include_masks) if mask]
            if len(filtered_predictions_neg) == len(filtered_mc_returns) and len(filtered_predictions_neg) > 0:
                images[f"{plot_key}_counterfactual_text"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_neg, ep_idx, step, " (Counterfactual Text)",
                    oracle_values = None, subtask_texts = negative_subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_counterfactual_text",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} counterfactual text plot created")

        predicted_values_random = all_predictions_random.get(traj_idx, [])
        if len(predicted_values_random) == len(predicted_values):
            filtered_predictions_random = [pred for pred, mask in zip(predicted_values_random, include_masks) if mask]
            if len(filtered_predictions_random) == len(filtered_mc_returns) and len(filtered_predictions_random) > 0:
                images[f"{plot_key}_random_actions"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_random, ep_idx, step, " (Random Actions)",
                    oracle_values = None, subtask_texts = subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_random_actions",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} random action plot created")

        predicted_values_shuffled = all_predictions_shuffled.get(traj_idx, [])
        if len(predicted_values_shuffled) == len(predicted_values):
            filtered_predictions_shuffled = [pred for pred, mask in zip(predicted_values_shuffled, include_masks) if mask]
            if len(filtered_predictions_shuffled) == len(filtered_mc_returns) and len(filtered_predictions_shuffled) > 0:
                images[f"{plot_key}_shuffled_actions"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_shuffled, ep_idx, step, " (Shuffled Actions)",
                    oracle_values = None, subtask_texts = subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_shuffled_actions",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} shuffled action plot created")

        predicted_values_counterfactual = all_predictions_counterfactual.get(traj_idx, []) if all_predictions_counterfactual is not None else []
        if len(predicted_values_counterfactual) == len(predicted_values):
            filtered_predictions_counterfactual = [
                pred for pred, mask in zip(predicted_values_counterfactual, include_masks) if mask
            ]
            if len(filtered_predictions_counterfactual) == len(filtered_mc_returns) and len(filtered_predictions_counterfactual) > 0:
                images[f"{plot_key}_counterfactual_actions"] = _create_value_plot(
                    filtered_mc_returns, filtered_predictions_counterfactual, ep_idx, step, " (Counterfactual Actions)",
                    oracle_values = None, subtask_texts = subtasks,
                    output_dir = output_dir, plot_key = f"{plot_key}_counterfactual_actions",
                )
                logging.info(f"Repo {repo_id}, episode {ep_idx} counterfactual action plot created")

    if output_dir is None and images:
        # Log without an explicit step, avoiding the "step must be monotonically
        # increasing" warning that occurs because this thread may run after further
        # training steps have been logged.
        wandb.log(images)
    logging.info(f"Render thread finished: {'saved' if output_dir else 'logged'} {len(images)} plots for step {step}")
    del images, ep_frame_images, all_predictions, all_predictions_neg, all_predictions_random, all_predictions_shuffled, all_predictions_counterfactual, all_attn_scores


def generate_validation_plots_dlimp(
    model: _value_fn.BaseValueFunction,
    val_episode_indices: list[int],
    step: int,
    *,
    action_conditioned: bool,
    data_config: _config.DataConfig,
    cache_dir: str,
    output_dir: str | None = None,
    batch_size: int = 64,
) -> dict:
    """Generate validation plots for RoboCOIN.

    Loads pre-cached validation episodes from ``cache_dir`` (populated at
    init time via ``cache_val_episodes``), runs value-function inference,
    and logs plots to W&B (async) or saves to ``output_dir`` on disk.

    Args:
        model: The value function model.
        val_episode_indices: List of episode indices to plot (used for count only).
        step: Current training step.
        action_conditioned: Whether the model is action-conditioned (Q vs V).
        data_config: Data configuration.
        cache_dir: Directory containing cached validation episodes.
        output_dir: If set, save plots/videos to this directory instead of logging to wandb.

    Returns:
        Empty dict (plots are logged/saved asynchronously by a background thread).
    """
    # Per-episode load → decode + resize → predict → free pipeline so peak host
    # RAM is bounded to one episode's decoded frames, not the entire val cache.
    # RoboCOIN cache pkls hold raw image bytes (decode_images=False at the dataset
    # level) and `decode_episode_images` materializes them lazily; the RoboCasa
    # branch caches already-decoded uint8 arrays (the data_transforms / model_transforms
    # pipeline requires decoded images), so `decode_episode_images` is a no-op there.
    image_size = data_config.rlds_kwargs.get("image_size")
    assert image_size is not None, (
        "image_size missing from data_config.rlds_kwargs — required to decode/resize "
        "compressed val cache images at predict time."
    )

    cache_files: list[tuple[str, str]] = []
    for filename in sorted(os.listdir(cache_dir)):
        if filename.endswith(".pkl"):
            traj_idx = filename[: -len(".pkl")]
            cache_files.append((traj_idx, os.path.join(cache_dir, filename)))
    logging.info(f"Found {len(cache_files)} cached validation episodes in {cache_dir}")

    MAX_SUBTASK_SEGMENTS = 16

    # Accumulators populated one episode at a time. Heavy per-episode arrays
    # (decoded images, frame dicts) are dropped after predict; we keep only
    # what the renderer needs.
    traj_to_repo_ep: dict[str, tuple[str, int, str]] = {}
    ep_subtasks: dict[str, list[str]] = {}
    ep_mc_returns: dict[str, list] = {}
    ep_frame_images: dict[str, list[np.ndarray]] = {}
    ep_fps: dict[str, int] = {}
    ep_include_masks: dict[str, list[bool]] = {}
    ep_negative_subtasks: dict[str, list[str]] = {}
    all_predictions: dict[str, list[float]] = {}
    all_predictions_neg: dict[str, list[float]] = {}
    all_predictions_random: dict[str, list[float]] = {}
    all_predictions_counterfactual: dict[str, list[float]] = {}
    all_predictions_shuffled: dict[str, list[float]] = {}
    all_attn_scores: dict[str, list[np.ndarray]] = {}

    for traj_idx, cache_file in cache_files:
        with open(cache_file, "rb") as f:
            frames = pickle.load(f)
        if not frames:
            logging.warning(f"Traj {traj_idx} cache empty, skipping")
            continue

        decode_episode_images(frames, image_size)

        repo_id_raw = frames[0]["repo_id"]
        if isinstance(repo_id_raw, np.ndarray):
            repo_id_raw = repo_id_raw.item()
        if isinstance(repo_id_raw, bytes):
            repo_id_raw = repo_id_raw.decode("utf-8")
        # RoboCOIN frames carry "episode_index"; RoboCasa frames carry "_traj_index"
        # (forwarded by the dataset and kept in _CACHE_KEYS).
        ep_idx_raw = frames[0].get("episode_index")
        if ep_idx_raw is None:
            ep_idx_raw = frames[0]["_traj_index"]
        if isinstance(ep_idx_raw, np.ndarray):
            ep_idx_raw = ep_idx_raw.item()
        ep_idx = int(ep_idx_raw)

        n_segments, split_frame_idx, segments = count_subtask_segments(frames)
        if n_segments > MAX_SUBTASK_SEGMENTS:
            split_seg_idx = n_segments // 2
            segment_specs = [
                (f"{traj_idx}_p0", "_part0", frames[:split_frame_idx], segments[:split_seg_idx]),
                (f"{traj_idx}_p1", "_part1", frames[split_frame_idx:], segments[split_seg_idx:]),
            ]
            logging.info(
                f"Traj {traj_idx} (repo {repo_id_raw}, episode {ep_idx}): "
                f"{n_segments} subtask segments > {MAX_SUBTASK_SEGMENTS}, splitting at "
                f"segment {split_seg_idx}, frame {split_frame_idx} "
                f"({split_frame_idx} + {len(frames) - split_frame_idx} frames)"
            )
        else:
            segment_specs = [(str(traj_idx), "", frames, segments)]

        for seg_key, part_suffix, seg_frames, seg_subtasks in segment_specs:
            if len(seg_frames) == 0:
                continue

            # Inject within-trajectory action permutation for the shuffled-actions plot.
            # Deterministic per-segment seed so plots are reproducible across runs.
            if action_conditioned and len(seg_frames) > 1 and "actions" in seg_frames[0]:
                rng_perm = np.random.default_rng(seed = 86)
                perm = rng_perm.permutation(len(seg_frames))
                shuffled_arrays = [seg_frames[p]["actions"] for p in perm]
                for i, frame in enumerate(seg_frames):
                    frame["shuffled_actions"] = shuffled_arrays[i]

            traj_to_repo_ep[seg_key] = (repo_id_raw, ep_idx, part_suffix)
            ep_subtasks[seg_key] = seg_subtasks
            ep_mc_returns[seg_key] = [f["mc_return"] for f in seg_frames]
            ep_frame_images[seg_key] = [
                np.stack([
                    np.asarray(f["image"]["left_wrist_0_rgb"]),
                    np.asarray(f["image"]["right_wrist_0_rgb"]),
                    np.asarray(f["image"]["base_0_rgb"]),
                ])
                for f in seg_frames
            ]
            ep_fps[seg_key] = int(seg_frames[0]["fps"])
            ep_include_masks[seg_key] = [bool(f.get("include_subtask", True)) for f in seg_frames]
            # Negative-text variant is RoboCOIN-only; skip when the cached frames
            # don't carry "negative_subtask_1_text".
            if seg_frames and "negative_subtask_1_text" in seg_frames[0]:
                _, _, negative_segments = count_subtask_segments(seg_frames, prefix = "negative_")
                ep_negative_subtasks[seg_key] = negative_segments
            else:
                ep_negative_subtasks[seg_key] = []

            seg_all_frames = [(seg_key, idx, frame) for idx, frame in enumerate(seg_frames)]
            seg_ep_mc_returns = {seg_key: ep_mc_returns[seg_key]}
            logging.info(
                f"Traj {seg_key} (repo {repo_id_raw}, episode {ep_idx}{part_suffix}): "
                f"running predictions on {len(seg_all_frames)} frames"
            )
            preds, preds_neg, preds_random, preds_cf, preds_shuffled, attn = predict_values(
                model, seg_all_frames, seg_ep_mc_returns, action_conditioned,
                batch_size = batch_size,
            )
            all_predictions[seg_key] = preds[seg_key]
            all_predictions_neg[seg_key] = preds_neg[seg_key]
            all_predictions_random[seg_key] = preds_random[seg_key]
            all_predictions_counterfactual[seg_key] = preds_cf[seg_key]
            all_predictions_shuffled[seg_key] = preds_shuffled[seg_key]
            all_attn_scores[seg_key] = attn[seg_key]

            del seg_all_frames

        # Drop the decoded frames for this trajectory before loading the next.
        del frames
        gc.collect()

    if not all_predictions:
        logging.warning("No valid frames found across all episodes")
        return {}

    logging.info(
        f"Computed predictions for {len(all_predictions)} trajectory segments"
    )

    if jax.process_index() == 0:
        global _render_thread
        if _render_thread is not None:
            if _render_thread.is_alive():
                logging.warning("Previous render thread still running, waiting for it to finish...")
            _render_thread.join()
            _render_thread = None
        _render_thread = threading.Thread(
            target = _render_and_log_plots,
            args = (
                all_predictions, all_predictions_neg, all_predictions_random,
                all_predictions_shuffled,
                all_attn_scores,
                ep_mc_returns, ep_frame_images, ep_fps, ep_include_masks,
                ep_subtasks, ep_negative_subtasks,
                traj_to_repo_ep, action_conditioned, step,
            ),
            kwargs = dict(
                all_predictions_counterfactual = all_predictions_counterfactual,
                output_dir = output_dir,
            ),
            daemon = True,
        )
        _render_thread.start()

    return {}


def main(config: _config.TrainConfig):
    """Train a value function."""
    init_logging()
    logger = logging.getLogger(__name__)

    # Initialize distributed training for TPU pods
    # Set PLATFORM=tpu environment variable to enable
    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        logger.info("Calling jax.distributed.initialize()")
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    logger.info(f"Running on: {_platform.node()}, platform: {platform}")

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(
            f"train_value_function.py requires a value function config, got {type(config.model).__name__}. "
            "Use configs like 'antmaze_large_diverse_v1_q_regression' or 'antmaze_large_diverse_v1_q_hl_gauss'."
        )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # Validate policy training settings
    policy_training_enabled = config.policy is not None and config.policy_extraction is not None
    if policy_training_enabled:
        logging.info(f"Policy training enabled with {type(config.policy_extraction).__name__}")
        logging.info(f"critic_steps_per_policy_step = {config.critic_steps_per_policy_step}")

    # Frozen critic mode (critic_steps_per_policy_step=0) requires a checkpoint
    if config.critic_steps_per_policy_step == 0:
        if not policy_training_enabled:
            raise ValueError(
                "critic_steps_per_policy_step=0 only makes sense with policy training. "
                "Set config.policy and config.policy_extraction."
            )
        if isinstance(config.weight_loader, _weight_loaders.NoOpWeightLoader):
            raise ValueError(
                "critic_steps_per_policy_step=0 (frozen critic) requires loading a critic checkpoint. "
                "Set weight_loader in config to load a pre-trained critic."
            )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Resolve FineTuneConfig and apply dataset overrides before creating the data loader
    ft_config = _config.get_fine_tune_config(config.fine_tune) if config.fine_tune is not None else None

    if ft_config is not None:
        ft_config = dataclasses.replace(ft_config, overwrite = config.overwrite, resume = config.resume)
        config = ft_config.apply_overrides(config)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    wandb_resuming = resuming and ft_config is None
    init_wandb(
        config,
        resuming = wandb_resuming,
        enabled = config.wandb_enabled,
        ft_config = ft_config,
        start_new = config.wandb_new,
    )
    logging.info(f"Initialized checkpoint manager with resuming={resuming}, config.resume={config.resume}")

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)

    raw_batch = next(data_iter)
    if isinstance(raw_batch, tuple):
        obs, _ = raw_batch
        batch = {"state": obs.state}
    else:
        batch = raw_batch

    logging.info(f"Initialized data loader. Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'tuple'}")

    # Create validation data_config (same overridden config.data)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Initialize variables for all branches
    num_episodes = None
    val_dataloader = None
    val_episode_indices: list[int] = []
    val_episodes_cache_dir = None

    if data_config.minari_dataset_id is not None:
        val_dataset = _data_loader.create_numpy_dataset_from_minari(
            data_config.minari_dataset_id,
            discount=data_config.discount,
            reward_scale=data_config.reward_scale,
            reward_bias=data_config.reward_bias,
        )
        val_dataloader = None
    elif data_config.legacy_d4rl_env_name is not None:
        val_dataset = _data_loader.create_numpy_dataset_from_legacy_d4rl(
            data_config.legacy_d4rl_env_name,
            discount=data_config.discount,
            reward_scale=data_config.reward_scale,
            reward_bias=data_config.reward_bias,
        )
        val_dataloader = None
    elif data_config.rlds_dataset_class == "robocasa":
        # Mirrors the RoboCOIN branch: cache transformed val trajectories to
        # `validation_cache_dir` so generate_validation_plots_dlimp can render
        # base + random + shuffled plots. Skips RoboCOIN-specific bits
        # (val_dataset_dir override, AddValidationVariants(include_negative=True)).
        action_horizon = config.action_horizon or config.model.action_horizon
        val_tokenizer = config.data._get_critic_tokenizer(config.model)
        assert val_tokenizer is not None, "RoboCasa validation requires a critic tokenizer."
        import dataclasses as _dc
        # RoboCasa data_transforms / model_transforms expect decoded uint8 arrays, so
        # decode upstream rather than carrying raw JPEG bytes through.
        _val_rlds_kwargs = {**data_config.rlds_kwargs, "decode_images": True}
        _val_data_config = _dc.replace(data_config, rlds_kwargs = _val_rlds_kwargs)
        val_trajectory_dataset = _data_loader.create_rlds_dataset(
            _val_data_config,
            action_horizon,
            config.batch_size,
            split = data_config.val_split,
            shuffle = False,
            return_trajectories = True,
        )
        val_input_transform = _transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
            *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
            *data_config.model_transforms.inputs,
            _config.AddValidationVariants(
                val_tokenizer,
                use_quantile_norm = data_config.use_quantile_norm,
                include_negative = False,
            ),
        ])
        val_episode_indices = list(range(config.num_val_trajectories))
        val_episodes_cache_dir = config.validation_cache_dir
        val_dataset = None

        allow_duplicate_repos = ft_config is not None
        cache_val_episodes(
            val_trajectory_dataset, config.num_val_trajectories, val_episodes_cache_dir,
            include_repos = config.include_repos, save_only = True,
            input_transform = val_input_transform,
            allow_duplicate_repos = allow_duplicate_repos,
        )
        if jax.process_count() > 1:
            jax.experimental.multihost_utils.sync_global_devices("val_cache_write")
        del val_trajectory_dataset, val_input_transform, _val_data_config, _val_rlds_kwargs
    elif data_config.rlds_dataset_class == "robocoin":
        action_horizon = config.action_horizon or config.model.action_horizon
        val_tokenizer = config.data._get_critic_tokenizer(config.model)
        assert val_tokenizer is not None, "RoboCOIN validation variants require a critic tokenizer."
        # Build a val-only data_config that (a) optionally points at val_dataset_dir
        # (a smaller variant of the same dataset) and (b) leaves images compressed.
        # `decode_images=False` keeps cam_X as the raw JPEG/PNG bytes coming out
        # of the TFRecord rather than decoded uint8 arrays, so the trajectory
        # iterator does not blow up host RAM during caching. The downstream
        # consumer (utils.cache_val_episodes → utils.load_episode_for_predict)
        # decodes + resizes lazily, one episode at a time, at prediction time.
        # These overrides only affect val trajectory caching — the training
        # data_config and loader are untouched.
        import dataclasses as _dc
        _val_rlds_kwargs = {**data_config.rlds_kwargs, "decode_images": False}
        _val_data_config = _dc.replace(
            data_config,
            rlds_data_dir = data_config.val_dataset_dir or data_config.rlds_data_dir,
            rlds_kwargs = _val_rlds_kwargs,
        )
        val_trajectory_dataset = _data_loader.create_rlds_dataset(
            _val_data_config,
            action_horizon,
            config.batch_size,
            split = data_config.val_split,
            shuffle = False,
            return_trajectories = True,
        )
        val_input_transform = _transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
            *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
            *data_config.model_transforms.inputs,
            _config.AddValidationVariants(
                val_tokenizer,
                use_quantile_norm = data_config.use_quantile_norm,
            ),
        ])
        val_episode_indices = list(range(config.num_val_trajectories))
        val_episodes_cache_dir = config.validation_cache_dir
        val_dataset = None

        # All workers participate in caching: worker 0 covers all repos up to
        # num_val_trajectories; non-zero workers only cache include_repos. Each
        # worker checks file existence before claiming/writing, so concurrent
        # writes to NFS are safe.
        allow_duplicate_repos = ft_config is not None
        cache_val_episodes(
            val_trajectory_dataset, config.num_val_trajectories, val_episodes_cache_dir,
            include_repos = config.include_repos, save_only = True,
            input_transform = val_input_transform,
            allow_duplicate_repos = allow_duplicate_repos,
        )
        if jax.process_count() > 1:
            jax.experimental.multihost_utils.sync_global_devices("val_cache_write")
        del val_trajectory_dataset, val_input_transform, _val_data_config, _val_rlds_kwargs
    else:
        # Non-RoboCOIN: use LeRobot dataset
        from lerobot.common.datasets import lerobot_dataset
        
        val_dataset = lerobot_dataset.LeRobotDataset(config.data.repo_id)
        
    if val_dataset is not None:
        num_episodes = val_dataset.num_episodes
        val_rng = np.random.default_rng(config.seed)
        
        # Filter to episodes with at least 10 frames for meaningful validation plots
        min_episode_length = 10
        if hasattr(val_dataset, "episode_starts") and hasattr(val_dataset, "episode_ends"):
            episode_lengths = val_dataset.episode_ends - val_dataset.episode_starts
            valid_episode_indices = np.where(episode_lengths >= min_episode_length)[0]
            if len(valid_episode_indices) < config.num_val_trajectories:
                logging.warning(
                    f"Only {len(valid_episode_indices)} episodes with >= {min_episode_length} frames, "
                    f"using all of them for validation"
                )
                val_episode_indices = valid_episode_indices.tolist()
            else:
                val_episode_indices = val_rng.choice(
                    valid_episode_indices, size=config.num_val_trajectories, replace=False
                ).tolist()
        else:
            val_episode_indices = val_rng.choice(
                num_episodes, size=min(config.num_val_trajectories, num_episodes), replace=False
            ).tolist()
        logging.info(f"Selected validation episodes: {val_episode_indices}")
        
        val_episodes_cache_dir = None  # No caching for non-RoboCOIN datasets
    
    action_conditioned = config.action_horizon is not None
    logging.info(f"Validation plots: action_conditioned={action_conditioned}")

    # Set up evaluation environment if enabled
    eval_env = None
    eval_enabled = config.eval_interval > 0 and config.eval_env is not None and policy_training_enabled
    if eval_enabled:
        num_eval_envs = min(config.eval_env.num_eval_episodes, 8)
        eval_env = _evaluation.create_vector_eval_env(
            config.eval_env,
            data_config,
            num_envs=num_eval_envs,
            render_mode="rgb_array" if config.eval_env.record_video else None,
        )
        logging.info(
            f"Evaluation enabled: {config.eval_env.num_eval_episodes} episodes every {config.eval_interval} steps "
            f"({num_eval_envs} parallel envs)"
        )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if resuming:
        logging.info("Resuming training from checkpoint")
        # train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        train_state = _load_model_utils.restore_state_with_shardings(
            checkpoint_manager, train_state, train_state_sharding,
        )

    # Unpack state and sharding
    if not isinstance(train_state, training_utils.ActorCriticTrainState):
        raise TypeError(f"Expected ActorCriticTrainState, got {type(train_state)}")

    critic_state = train_state.critic
    policy_state = train_state.policy
    critic_sharding = train_state_sharding.critic
    policy_sharding = train_state_sharding.policy
    logging.info(f"Initialized combined state:\nCritic: {training_utils.array_tree_to_info(critic_state.params)}")
    if policy_state:
        logging.info(f"Policy: {training_utils.array_tree_to_info(policy_state.params)}")

    jax.block_until_ready(critic_state)

    # === val_only mode: run one validation pass and exit ===
    if ft_config is not None and ft_config.val_only:
        logging.info("val_only mode: running validation plotting")
        step = int(critic_state.step)
        model = nnx.merge(critic_state.model_def, critic_state.params)

        if data_config.rlds_dataset_class in ("robocoin", "robocasa"):
            generate_validation_plots_dlimp(
                model = model,
                val_episode_indices = val_episode_indices,
                step = step,
                action_conditioned = action_conditioned,
                data_config = data_config,
                cache_dir = val_episodes_cache_dir,
                batch_size = 32,
            )
        else:
            plot_images = generate_validation_plots(
                model=model,
                dataset=val_dataset,
                val_episode_indices=val_episode_indices,
                step=step,
                action_conditioned=action_conditioned,
                data_config=data_config,
            )
            if jax.process_index() == 0 and plot_images:
                logging.info(f"Generated {len(plot_images)} validation plot items")
                wandb.log(plot_images)

        # Wait for background render thread if any
        if jax.process_index() == 0 and _render_thread is not None and _render_thread.is_alive():
            logging.info("Waiting for render thread to finish")
            _render_thread.join()

        if jax.process_count() > 1:
            multihost_utils.sync_global_devices("val_only_done")

        logging.info("val_only mode complete")
        return

    # Determine effective training parameters based on fine-tune config
    pretrained_step = int(critic_state.step)
    is_fine_tuning = ft_config is not None and not ft_config.val_only

    if is_fine_tuning:
        config, critic_state, critic_sharding, checkpoint_manager = ft_config.initialize(
            config, pretrained_step, critic_state, mesh,
        )

    lr_schedule = config.lr_schedule.create()

    if policy_state is not None:
        ptrain_step = jax.jit(
            functools.partial(value_function_train_step, config, lr_schedule),
            in_shardings=(critic_sharding, policy_sharding, data_sharding, replicated_sharding),
            out_shardings=(critic_sharding, replicated_sharding),
            donate_argnums=(0,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(value_function_train_step, config, lr_schedule),
            in_shardings=(critic_sharding, None, data_sharding, replicated_sharding),
            out_shardings=(critic_sharding, replicated_sharding),
            donate_argnums=(0,),
        )

    ppolicy_step = None
    if policy_state is not None:
        ppolicy_step = jax.jit(
            functools.partial(policy_train_step, config, lr_schedule),
            in_shardings=(
                critic_sharding,
                policy_sharding,
                data_sharding,
                replicated_sharding,
            ),
            out_shardings=(policy_sharding, replicated_sharding),
        )

    start_step = int(critic_state.step)
    # Fine-tune mode loads pretrained weights from a different dataset, so
    # advancing the new data iterator by ``start_step`` batches has no resume
    # semantics — skip the fast-forward and iterate only the fine-tune range.
    skip_fast_forward = config.fine_tune is not None
    if start_step > 0 and not skip_fast_forward:
        logging.info(f"Resuming with data-loader fast-forward through step {start_step}")
    loop_range = (
        range(start_step, config.num_train_steps)
        if skip_fast_forward
        else range(config.num_train_steps)
    )
    pbar = tqdm.tqdm(
        loop_range,
        total=config.num_train_steps,
        initial=start_step if skip_fast_forward else 0,
        dynamic_ncols=True,
    )

    timer = Timer()

    # Log initial timing info
    logging.info(f"Starting training with batch_size={config.batch_size}, num_workers={config.num_workers}")

    for step in pbar:
        # Split rng for this step
        rng, step_rng = jax.random.split(rng)
        if step < start_step:
            with timer.context("data_fetch"):
                raw_batch = next(data_iter)

            with timer.context("data_postprocess"):
                if isinstance(raw_batch, tuple):
                    obs, _ = raw_batch
                    batch = {"state": obs.state}
                else:
                    batch = raw_batch
            continue

        with timer.context("train_step_compute"), sharding.set_mesh(mesh):
            critic_state, info = ptrain_step(critic_state, policy_state, batch, step_rng)

        with timer.context("train_step_sync"):
            jax.block_until_ready(critic_state)
            jax.block_until_ready(info)

        # Policy training step (if enabled)
        if policy_training_enabled:
            ratio = config.critic_steps_per_policy_step
            should_update_policy = ratio == 0 or (step + 1) % ratio == 0
            if should_update_policy:
                rng, policy_rng = jax.random.split(rng)
                with timer.context("policy_step_compute"), sharding.set_mesh(mesh):
                    policy_state, policy_info = ppolicy_step(critic_state, policy_state, batch, policy_rng)
                with timer.context("policy_step_sync"):
                    jax.block_until_ready(policy_state)
                    jax.block_until_ready(policy_info)
                info.update(policy_info)

        if step % config.log_interval == 0:
            info = jax.device_get(info)
            # Add timing info to logged metrics (average and total)
            total_times = timer.get_total_times(reset=False)
            avg_times = timer.get_average_times(reset=True)
            timing_info = {f"average_times/{k}": v for k, v in avg_times.items()}
            timing_info.update({f"total_times/{k}": v for k, v in total_times.items()})
            info.update(timing_info)

            info = {k: float(v) for k, v in info.items()}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(info, step=step)
            
            # Memory debugging: log every 100 steps, force GC every 500 steps
            # force_gc = (step % 500 == 0)
            force_gc = False
            log_memory_debug(step, data_loader=data_loader, force_gc=force_gc, log_to_wandb=jax.process_index() == 0)

        # Break down data loading into components
        with timer.context("data_fetch"):
            raw_batch = next(data_iter)

        with timer.context("data_postprocess"):
            if isinstance(raw_batch, tuple):
                obs, _ = raw_batch
                batch = {"state": obs.state}
            else:
                batch = raw_batch

        if (step + 1) % config.save_interval == 0 or step + 1 == config.num_train_steps:
            with timer.context("checkpoint_save"):
                state_to_save = training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)
                _checkpoints.save_state(checkpoint_manager, state_to_save, data_loader, step + 1)

        # Generate validation plots (all workers participate for FSDP, only worker 0 creates plots/logs)
        # The third disjunct fires once at the resumed step so we can inspect the
        # restored model state before further training shifts it.
        if (
            (step + 1) % config.plot_interval == 0
            or step + 1 == config.num_train_steps
            # or (step % config.plot_interval == 0 and step == start_step and start_step > 0)
        ):
            with timer.context("validation_plot"):
                model = nnx.merge(critic_state.model_def, critic_state.params)

                if data_config.rlds_dataset_class in ("robocoin", "robocasa"):
                    generate_validation_plots_dlimp(
                        model = model,
                        val_episode_indices = val_episode_indices,
                        step = step,
                        action_conditioned = action_conditioned,
                        data_config = data_config,
                        cache_dir = val_episodes_cache_dir,
                        batch_size = 16,
                    )
                else:
                    plot_images = generate_validation_plots(
                        model=model,
                        dataset=val_dataset,
                        val_episode_indices=val_episode_indices,
                        step=step,
                        action_conditioned=action_conditioned,
                        data_config=data_config,
                    )
                    if jax.process_index() == 0:
                        if plot_images:
                            logging.info(f"Generated {len(plot_images)} validation plot items at step {step}")
                            wandb.log(plot_images, step=step)
                        else:
                            logging.warning(f"No validation plots generated at step {step}")
                del model

        # Policy evaluation (only on worker 0)
        if eval_enabled and ((step + 1) % config.eval_interval == 0 or step + 1 == config.num_train_steps):
            with timer.context("policy_eval"):
                policy_model = nnx.merge(policy_state.model_def, policy_state.params)

                # Create normalization and unnormalization transforms
                normalize = _transforms.Normalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                    strict=False,
                )
                # Only create unnormalize if actions were normalized
                should_unnormalize_actions = "actions" not in data_config.skip_normalize_keys
                if should_unnormalize_actions:
                    actions_norm_stats = {"actions": data_config.norm_stats["actions"]}
                    unnormalize = _transforms.Unnormalize(
                        actions_norm_stats,
                        use_quantiles=data_config.use_quantile_norm,
                    )
                else:
                    unnormalize = None

                should_normalize_state = "state" not in data_config.skip_normalize_keys
                eval_rng = jax.random.key(step)

                def policy_fn(
                    obs_batch: np.ndarray,
                    normalize=normalize,
                    unnormalize=unnormalize,
                    policy_model=policy_model,
                    step=step,
                    should_normalize_state=should_normalize_state,
                ) -> np.ndarray:
                    nonlocal eval_rng
                    step_rng, eval_rng = jax.random.split(eval_rng)

                    # Conditionally normalize batch of observations: [num_envs, obs_dim]
                    if should_normalize_state:
                        processed_obs = np.stack(
                            [normalize({"state": obs})["state"] for obs in obs_batch],
                            axis=0,
                        )
                    else:
                        processed_obs = np.asarray(obs_batch, dtype=np.float32)
                    assert not np.any(np.isnan(processed_obs)), f"NaN in processed_obs: {processed_obs}"

                    model_obs = _model.Observation(
                        images={},
                        image_masks={},
                        state=jnp.asarray(processed_obs),  # [num_envs, obs_dim]
                        tokenized_prompt=None,
                        tokenized_prompt_mask=None,
                    )

                    # Deterministic evaluation: use the mode of the action distribution.
                    transition = _model.wrap_observation_as_transition(model_obs)
                    actions = policy_model.sample_actions(step_rng, transition, deterministic=True)
                    # Take the first action in the horizon
                    # Shape: [num_envs, action_horizon, action_dim] -> [num_envs, action_dim]
                    actions = np.asarray(jax.device_get(actions[:, 0, :]))
                    assert not np.any(np.isnan(actions)), f"NaN in actions: {actions}"

                    if unnormalize is not None:
                        actions = np.stack(
                            [unnormalize({"actions": a})["actions"] for a in actions],
                            axis=0,
                        )
                        assert not np.any(np.isnan(actions)), f"NaN in unnormalized_actions: {actions}"
                    return actions

                eval_results = _evaluation.evaluate_policy_vectorized(
                    policy_fn=policy_fn,
                    vec_env=eval_env,
                    num_episodes=config.eval_env.num_eval_episodes,
                    seed=config.eval_env.seed + step,
                    record_video=config.eval_env.record_video,
                )
                eval_metrics = eval_results.to_dict()

                # Log video (1 episode per evaluation round)
                if eval_results.video_frames is not None:
                    # wandb.Video expects shape [T, C, H, W] for numpy arrays
                    video_frames = np.transpose(eval_results.video_frames, (0, 3, 1, 2))
                    eval_metrics["eval/video"] = wandb.Video(video_frames, fps=30, format="mp4")

                wandb.log(eval_metrics, step=step)
                logging.info(
                    f"Step {step} eval: mean_return={eval_results.mean_return:.2f}, "
                    f"std_return={eval_results.std_return:.2f}"
                )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    # Keep all hosts alive until rank 0 has fully completed async rendering/logging.

    if jax.process_index() == 0 and _render_thread is not None and _render_thread.is_alive():
        logging.info("Waiting for render thread to finish")
        _render_thread.join()

    if jax.process_count() > 1:
        logging.info("Waiting at post-render multihost barrier")
        multihost_utils.sync_global_devices("train_value_function_post_render_join")


if __name__ == "__main__":
    main(_config.cli())
