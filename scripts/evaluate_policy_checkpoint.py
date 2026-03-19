"""Load a policy checkpoint (same as train.py resume), load a data batch, and run a forward pass."""

import argparse
import dataclasses
import logging
import os

from etils import epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def init_logging():
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy checkpoint on a single batch")
    parser.add_argument("config_name", type=str, help="Config name (e.g. robocoin_bimanual_pi05)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load (default: latest)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--fsdp_devices", type=int, default=None, help="Override FSDP devices")
    args = parser.parse_args()

    init_logging()

    config = _config.get_config(args.config_name)
    if args.checkpoint_dir is not None:
        config = dataclasses.replace(config, checkpoint_base_dir=args.checkpoint_dir)
    if args.fsdp_devices is not None:
        config = dataclasses.replace(config, fsdp_devices=args.fsdp_devices)
    config = dataclasses.replace(config, batch_size=args.batch_size, resume=True)

    logging.info(f"Config: {config.name}")
    logging.info(f"Checkpoint dir: {config.checkpoint_dir}")

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Load data
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False, num_batches=1)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    observation, actions = batch
    logging.info(f"Loaded batch:\n{training_utils.array_tree_to_info(batch)}")

    # Ensure GCS checkpoints have commit_success.txt markers (orbax requires them).
    # The async save callback may have failed (e.g. norm_stats serialization) leaving
    # valid checkpoint data without the finalization marker.
    ckpt_dir = epath.Path(config.checkpoint_dir)
    if str(ckpt_dir).startswith("gs://") and ckpt_dir.exists():
        for child in ckpt_dir.iterdir():
            if child.is_dir() and not (child / "commit_success.txt").exists():
                logging.info(f"Adding missing commit_success.txt to {child}")
                (child / "commit_success.txt").write_text("")

    # Initialize model shape and checkpoint manager
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise RuntimeError(f"No checkpoint found at {config.checkpoint_dir}")

    available_steps = sorted(checkpoint_manager.all_steps())
    step = args.step if args.step is not None else available_steps[-1]
    logging.info(f"Available steps: {available_steps}, loading step {step}")

    # Init train state shape (same as train.py resume path)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
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
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    # Restore checkpoint
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, data_loader, step=step)
    logging.info(f"Restored checkpoint at step {int(train_state.step)}")

    # Forward pass
    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()

    eval_rng = jax.random.key(86)
    with at.disable_typechecking():
        chunked_loss = model.compute_loss(eval_rng, observation, actions, train=False)

    chunked_loss = jax.device_get(chunked_loss)
    logging.info(f"Chunked loss shape: {chunked_loss.shape}")
    logging.info(f"Mean loss: {np.mean(chunked_loss):.6f}")
    logging.info(f"Per-timestep mean loss: {np.mean(chunked_loss, axis=0)}")

    # Apply masks if present
    mask = np.ones_like(chunked_loss)
    if observation.action_mask is not None:
        mask = mask * np.asarray(observation.action_mask)
    if observation.loss_mask is not None:
        mask = mask * np.asarray(observation.loss_mask)[:, None]
    masked_loss = np.sum(chunked_loss * mask) / np.maximum(np.sum(mask), 1.0)
    logging.info(f"Masked loss (same as training): {masked_loss:.6f}")

    # Per-sample losses
    sample_losses = np.sum(chunked_loss * mask, axis=-1) / np.maximum(np.sum(mask, axis=-1), 1.0)
    logging.info(f"Per-sample losses (first 8): {sample_losses[:8]}")
    logging.info(f"Loss std across samples: {np.std(sample_losses):.6f}")

    # Action statistics
    actions_np = np.asarray(actions)
    logging.info(f"Actions shape: {actions_np.shape}, range: [{actions_np.min():.4f}, {actions_np.max():.4f}]")

    state_np = np.asarray(observation.state)
    logging.info(f"State shape: {state_np.shape}, range: [{state_np.min():.4f}, {state_np.max():.4f}]")


if __name__ == "__main__":
    main()
