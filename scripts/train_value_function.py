"""Training script for value functions."""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.value_functions.base as _value_fn


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

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


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
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
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """Initialize training state for a value function model."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(f"Expected BaseValueFunctionConfig, got {type(config.model)}")
    model_config: _value_fn.BaseValueFunctionConfig = config.model

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: dict[str, Any],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step for value function."""
    model = nnx.merge(state.model_def, state.params)

    # Create transition once for both loss computation and batch stats
    transition = _value_fn.Transition.from_batch(batch)

    def loss_fn(model: _value_fn.BaseValueFunction):
        # compute_loss returns (per_sample_loss, info_dict)
        per_sample_loss, value_info = model.compute_loss(transition, train=True)
        return jnp.mean(per_sample_loss), value_info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, value_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    # Batch statistics
    obs_state = transition.observation.state
    batch_stats = {
        # Observation stats
        "batch/obs_mean": jnp.mean(obs_state),
        "batch/obs_std": jnp.std(obs_state),
        "batch/obs_min": jnp.min(obs_state),
        "batch/obs_max": jnp.max(obs_state),
        # Action stats
        "batch/action_mean": jnp.mean(transition.action),
        "batch/action_std": jnp.std(transition.action),
        "batch/action_min": jnp.min(transition.action),
        "batch/action_max": jnp.max(transition.action),
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

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **value_info,
        **batch_stats,
    }
    return new_state, info


def get_trajectory_frames(dataset, episode_idx: int) -> list[dict]:
    """Extract all frames from a specific episode in the dataset."""
    # Get episode boundary indices
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
) -> dict:
    """Generate validation plots comparing predicted values vs MC returns.

    Args:
        model: The value function model
        dataset: The LeRobot dataset
        val_episode_indices: List of episode indices to plot
        step: Current training step
        action_conditioned: Whether the model is action-conditioned (Q vs V)

    Returns:
        Dictionary of wandb images keyed by trajectory index
    """
    from openpi.models import model as _model

    images = {}

    for ep_idx in val_episode_indices:
        try:
            frames = get_trajectory_frames(dataset, ep_idx)
        except Exception as e:
            logging.warning(f"Could not load episode {ep_idx}: {e}")
            continue

        if len(frames) == 0:
            continue

        # Collect mc_returns and predicted values
        mc_returns = []
        predicted_values = []

        for frame in frames:
            # Extract MC return
            mc_return = frame.get("mc_return")
            if mc_return is None:
                continue
            if hasattr(mc_return, "numpy"):
                mc_return = mc_return.numpy()
            mc_return = np.asarray(mc_return).item() if np.asarray(mc_return).size == 1 else np.asarray(mc_return)
            mc_returns.append(float(mc_return))

            # Build observation for prediction
            state = frame.get("state")
            if hasattr(state, "numpy"):
                state = state.numpy()
            state = np.asarray(state, dtype=np.float32)

            obs = _model.Observation(
                images={},
                image_masks={},
                state=jnp.asarray(state[None, ...]),  # Add batch dim
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )

            # Get action if needed
            action = None
            if action_conditioned:
                act = frame.get("actions", frame.get("action"))
                if hasattr(act, "numpy"):
                    act = act.numpy()
                action = jnp.asarray(np.asarray(act, dtype=np.float32)[None, ...])

            # Compute predicted value
            pred_value = model.compute_value(obs, action)
            predicted_values.append(float(jax.device_get(pred_value[0])))

        if len(mc_returns) == 0:
            continue

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        timesteps = np.arange(len(mc_returns))

        ax.plot(timesteps, mc_returns, label="MC Returns", color="blue", linewidth=2)
        ax.plot(timesteps, predicted_values, label="Predicted Value", color="orange", linewidth=2, linestyle="--")

        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(f"Episode {ep_idx} - Step {step}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        images[f"val/episode_{ep_idx}"] = wandb.Image(fig)
        plt.close(fig)

    return images


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if not isinstance(config.model, _value_fn.BaseValueFunctionConfig):
        raise TypeError(
            f"train_value_function.py requires a value function config, got {type(config.model).__name__}. "
            "Use configs like 'antmaze_large_diverse_v1_q_regression' or 'antmaze_large_diverse_v1_q_hl_gauss'."
        )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

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

    # Select fixed validation trajectories for plotting
    from lerobot.common.datasets import lerobot_dataset

    val_dataset = lerobot_dataset.LeRobotDataset(config.data.repo_id)
    num_episodes = val_dataset.num_episodes
    val_rng = np.random.default_rng(config.seed)
    val_episode_indices = val_rng.choice(
        num_episodes, size=min(config.num_val_trajectories, num_episodes), replace=False
    ).tolist()
    logging.info(f"Selected validation episodes: {val_episode_indices}")
    action_conditioned = getattr(config.model, "action_conditioned", False)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        raw_batch = next(data_iter)
        if isinstance(raw_batch, tuple):
            obs, _ = raw_batch
            batch = {"state": obs.state}
        else:
            batch = raw_batch

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

        # Generate validation plots
        if step % config.plot_interval == 0 and step > 0:
            model = nnx.merge(train_state.model_def, train_state.params)
            plot_images = generate_validation_plots(
                model=model,
                dataset=val_dataset,
                val_episode_indices=val_episode_indices,
                step=step,
                action_conditioned=action_conditioned,
            )
            if plot_images:
                wandb.log(plot_images, step=step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
