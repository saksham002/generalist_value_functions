"""Evaluate a value function model on specific RoboCOIN episodes.

Caches episodes to disk (one pickle per trajectory), then loads the model,
computes predicted values, and logs plots to wandb.

Episodes are specified as (repo_index, episode_index) pairs. The script first
iterates through the val split to find matching episodes; any remaining
episodes are searched in the train split.

Usage:
    python scripts/evaluate_value_function.py \
        --model robocoin_bimanual_paligemma_q_sarsa:gs://bucket/checkpoints/.../exp \
        --episodes 3,270 0,15 \
        --cache-dir /path/to/cache \
        [--project-name robocoin_value_eval]
"""

import argparse
import dataclasses as dc
import logging
import os
import pickle

import flax.nnx as nnx
import jax
import numpy as np
import optax
import wandb

from openpi.models.tokenizer import create_tokenizer
from openpi.robocoin_utils.load_model_utils import load_train_module, restore_state_with_shardings
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
logger = logging.getLogger()
logger.warning("evaluate_value_function.py: This script has not been ported to the RLDS pipeline and will not work.")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Evaluate value function on specific RoboCOIN episodes.")
    parser.add_argument(
        "--model",
        action = "append",
        required = True,
        metavar = "CONFIG_NAME:CHECKPOINT_PATH",
        help = "Model specification as config_name:checkpoint_path. Can be repeated.",
    )
    parser.add_argument(
        "--episodes",
        nargs = "+",
        required = True,
        metavar = "REPO_IDX,EP_IDX",
        help = "Episode specifications as repo_index,episode_index pairs.",
    )
    parser.add_argument("--cache-dir", required = True, help = "Directory for cached episode pickles.")
    parser.add_argument("--project-name", default = "robocoin_value_eval", help = "Wandb project name.")
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, str]:
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Model spec must be config_name:checkpoint_path, got: {spec}")
    return parts[0], parts[1]


def parse_episode_spec(spec: str) -> tuple[int, int]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"Episode spec must be repo_index,episode_index, got: {spec}")
    return int(parts[0]), int(parts[1])


def get_model_display_name(config_name: str, checkpoint_path: str) -> str:
    tail = checkpoint_path.rstrip("/").rsplit("/", 1)[-1]
    if tail.isdigit() or tail == config_name:
        return config_name
    return f"{config_name}/{tail}"


def compute_param_norm(model: nnx.Module) -> float:
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            nnx.Not(nnx_utils.PathRegex(".*target_(network|head)/.*")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return float(jax.device_get(optax.global_norm(kernel_params)))


def cache_episodes_by_identity(
    robocoin_config,
    backbone_variant: str,
    action_horizon: int,
    requested: list[tuple[int, int]],
    cache_dir: str,
):
    """Cache specific episodes identified by (repo_index, episode_index).

    First iterates the val split, then the train split for any remaining episodes.
    """
    os.makedirs(cache_dir, exist_ok = True)

    # Check which episodes are already cached
    already_cached: set[tuple[int, int]] = set()
    for filename in os.listdir(cache_dir):
        if filename.startswith("traj_") and filename.endswith(".pkl"):
            traj_idx = int(filename.replace("traj_", "").replace(".pkl", ""))
            cache_file = os.path.join(cache_dir, filename)
            with open(cache_file, "rb") as f:
                frames = pickle.load(f)
            if frames:
                repo_idx = int(frames[0]["_traj_index"])
                ep_idx = int(frames[0]["episode_index"])
                already_cached.add((repo_idx, ep_idx))

    remaining = set(requested) - already_cached
    if not remaining:
        logger.info(f"All {len(requested)} episodes already cached in {cache_dir}")
        return

    logger.info(f"{len(already_cached)} episodes already cached, {len(remaining)} to collect")

    for split in ("val", "train"):
        if not remaining:
            break

        logger.info(f"Searching {split} split for {len(remaining)} episodes...")
        num_images = robocoin_config.max_cameras if backbone_variant == "gemma3" else 0
        tokenizer = create_tokenizer(backbone_variant, robocoin_config.max_token_len, num_images = num_images)

        loader_config = dc.replace(
            robocoin_config,
            split = split,
            batch_size = 64,
            prefetch_buffer_size = 2,
            shuffle = False,
            repeat = False,
            action_horizon = action_horizon,
            state_norm_stats = robocoin_config.state_norm_stats,
            use_quantile_norm = robocoin_config.use_quantile_norm,
        )
        dataloader = create_robocoin_data_loader(loader_config, tokenizer = tokenizer)

        active_trajs: dict[int, tuple[int, int]] = {}
        traj_frames: dict[int, list[dict]] = {}
        found_count = 0

        for batch in dataloader:
            traj_indices = batch.get("_traj_index", None)
            if traj_indices is None:
                continue
            if hasattr(traj_indices, "device"):
                traj_indices = np.asarray(traj_indices)

            unique_batch_trajs = set(int(t) for t in traj_indices)

            completed = set(active_trajs.keys()) - unique_batch_trajs
            for traj_idx in completed:
                if traj_idx in traj_frames:
                    frames = traj_frames[traj_idx]
                    if frames:
                        frames.sort(key = lambda f: f["_frame_index"])
                    cache_file = os.path.join(cache_dir, f"traj_{traj_idx}.pkl")
                    with open(cache_file, "wb") as f:
                        pickle.dump(frames, f)
                    repo_idx, ep_idx = active_trajs[traj_idx]
                    logger.info(f"Saved traj {traj_idx} (repo {repo_idx}, episode {ep_idx}, {len(frames)} frames) to {cache_file}")
                    remaining.discard((repo_idx, ep_idx))
                    found_count += 1
                    del traj_frames[traj_idx]
                del active_trajs[traj_idx]

            if not remaining and not active_trajs:
                break

            batch_size = traj_indices.shape[0]
            for i in range(batch_size):
                traj_idx = int(traj_indices[i])

                if traj_idx not in traj_frames and traj_idx not in active_trajs:
                    ep_idx = int(batch["episode_index"][i])
                    repo_id = batch["repo_id"][i]
                    if isinstance(repo_id, bytes):
                        repo_id = repo_id.decode("utf-8")
                    repo_idx = traj_idx

                    if (repo_idx, ep_idx) not in remaining:
                        continue

                    traj_frames[traj_idx] = []
                    active_trajs[traj_idx] = (repo_idx, ep_idx)

                if traj_idx not in active_trajs:
                    continue

                frame = {}
                for key, value in batch.items():
                    if isinstance(value, dict):
                        frame[key] = {sub_key: np.asarray(sub_value[i]) for sub_key, sub_value in value.items()}
                    else:
                        frame[key] = np.asarray(value[i])
                traj_frames[traj_idx].append(frame)

        for traj_idx in list(active_trajs.keys()):
            if traj_idx in traj_frames:
                frames = traj_frames[traj_idx]
                if frames:
                    frames.sort(key = lambda f: f["_frame_index"])
                cache_file = os.path.join(cache_dir, f"traj_{traj_idx}.pkl")
                with open(cache_file, "wb") as f:
                    pickle.dump(frames, f)
                repo_idx, ep_idx = active_trajs[traj_idx]
                logger.info(f"Saved remaining traj {traj_idx} (repo {repo_idx}, episode {ep_idx}, {len(frames)} frames) to {cache_file}")
                remaining.discard((repo_idx, ep_idx))
                found_count += 1

        del dataloader
        logger.info(f"Found {found_count} episodes in {split} split, {len(remaining)} still remaining")

    if remaining:
        logger.warning(f"Could not find {len(remaining)} episodes: {remaining}")


def load_model(train_module, config_name: str, checkpoint_path: str):
    """Load model, auto-detecting local device count for cross-device restore."""
    config = _config.get_config(config_name)

    local_devices = jax.local_device_count()
    local_config = dc.replace(config, fsdp_devices = local_devices)

    init_rng = jax.random.PRNGKey(86)
    mesh = sharding.make_mesh(local_devices)

    train_state_shape, state_sharding = train_module.init_train_state(local_config, init_rng, mesh, resume = True)

    mngr, _ = _checkpoints.initialize_checkpoint_dir(
        checkpoint_path, keep_period = None, overwrite = False, resume = True
    )
    train_state = restore_state_with_shardings(mngr, train_state_shape, state_sharding)

    critic_state = train_state.critic
    model = nnx.merge(critic_state.model_def, critic_state.params)

    param_norm = compute_param_norm(model)
    logger.info(f"param_norm for {config_name} ({checkpoint_path}): {param_norm:.4f}")

    action_conditioned = model.network.action_conditioned

    return model, action_conditioned, config


def main():
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)

    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    args = parse_args()
    requested_episodes = [parse_episode_spec(s) for s in args.episodes]
    logger.info(f"Requested episodes (repo_idx, ep_idx): {requested_episodes}")

    model_specs = [parse_model_spec(s) for s in args.model]
    logger.info(f"Models: {[s[0] for s in model_specs]}")

    train_module = load_train_module()

    # Use the first model's config for dataloader setup
    first_config = _config.get_config(model_specs[0][0])
    data_config = first_config.data.create()
    robocoin_config = data_config.robocoin_data_config

    # Cache episodes (worker 0 only)
    if jax.process_index() == 0:
        cache_episodes_by_identity(
            robocoin_config,
            first_config.backbone_variant,
            first_config.action_horizon or 5,
            requested_episodes,
            args.cache_dir,
        )

    # Count cached episodes for val_episode_indices placeholder
    num_cached = sum(
        1 for f in os.listdir(args.cache_dir)
        if f.startswith("traj_") and f.endswith(".pkl")
    ) if os.path.isdir(args.cache_dir) else 0

    # Evaluate each model
    for model_idx, (config_name, checkpoint_path) in enumerate(model_specs):
        display_name = get_model_display_name(config_name, checkpoint_path)
        logger.info(f"Loading model: {display_name}")

        model, action_conditioned, config = load_model(train_module, config_name, checkpoint_path)

        if jax.process_index() == 0:
            wandb.init(
                project = args.project_name,
                name = f"eval_{display_name}",
                config = {"config_name": config_name, "checkpoint_path": checkpoint_path},
            )

        train_module.generate_validation_plots_dlimp(
            model = model,
            val_dataloader = None,
            val_episode_indices = list(range(num_cached)),
            step = 0,
            action_conditioned = action_conditioned,
            data_config = data_config,
            cache_dir = args.cache_dir,
            save_only = False,
        )

        # Wait for the background render thread to finish before closing wandb
        if hasattr(train_module, "_render_thread") and train_module._render_thread is not None:
            train_module._render_thread.join()
            train_module._render_thread = None

        if jax.process_index() == 0:
            wandb.finish()

        del model
        jax.clear_caches()
        logger.info(f"Done with {display_name}")

    logger.info("All evaluations complete")


if __name__ == "__main__":
    main()
