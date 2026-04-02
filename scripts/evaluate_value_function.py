"""Evaluate a value function checkpoint on RoboCOIN cached trajectories."""

import dataclasses
import logging
import os
import pickle
import tempfile

import flax.nnx as nnx
import jax
import numpy as np

from openpi.models.best_of_n import BestOfNWrapper
import openpi.models.model as _model
from openpi.robocoin_utils.load_model_utils import load_train_module, restore_params_with_shardings
from openpi.robocoin_utils.utils import cache_val_episodes, count_subtask_segments, get_obs_and_action
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.transforms as _transforms
import openpi.value_functions.base_value_functions as _base_vf

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen = True)
class EvalConfig:
    """Configuration for value function evaluation."""

    config_name: str
    checkpoint_path: str
    split: str = "val"
    fine_tune: str | None = None
    include_repos: tuple[str, ...] = ()
    num_trajectories: int = 10
    cache_dir: str | None = None
    output_dir: str | None = None
    project_name: str = "robocoin_value_eval"
    counterfactual_best_of_n: bool = False
    counterfactual_action_store_dir: str | None = None


def _resolve_eval_cache_dir(eval_config: EvalConfig) -> str:
    if eval_config.cache_dir is not None:
        return eval_config.cache_dir
    if eval_config.checkpoint_path.startswith("gs://"):
        return os.path.join(
            tempfile.gettempdir(),
            "openpi_eval_cache",
            eval_config.config_name,
            eval_config.split,
        )
    return os.path.join(eval_config.checkpoint_path, "eval_cache", eval_config.split)


def _load_cached_trajectories(cache_dir: str) -> dict[int, list[dict]]:
    traj_frames: dict[int, list[dict]] = {}
    for filename in os.listdir(cache_dir):
        if filename.startswith("traj_") and filename.endswith(".pkl"):
            traj_idx = int(filename.replace("traj_", "").replace(".pkl", ""))
            with open(os.path.join(cache_dir, filename), "rb") as f:
                traj_frames[traj_idx] = pickle.load(f)
    return traj_frames


def _split_trajectory_frames(
    traj_frames: dict[int, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, tuple[str, int, str]], dict[str, list[str]]]:
    max_subtask_segments = 16
    split_traj_frames: dict[str, list[dict]] = {}
    traj_to_repo_ep: dict[str, tuple[str, int, str]] = {}
    ep_subtasks: dict[str, list[str]] = {}

    for traj_idx, frames in traj_frames.items():
        if not frames:
            continue

        episode_index = int(frames[0]["episode_index"])
        repo_id = frames[0]["repo_id"]
        if isinstance(repo_id, np.ndarray):
            repo_id = repo_id.item()
        if isinstance(repo_id, bytes):
            repo_id = repo_id.decode("utf-8")

        num_segments, split_frame_idx, segments = count_subtask_segments(frames)
        if num_segments > max_subtask_segments:
            split_segment_idx = num_segments // 2
            key_part0 = f"{traj_idx}_p0"
            key_part1 = f"{traj_idx}_p1"
            split_traj_frames[key_part0] = frames[:split_frame_idx]
            split_traj_frames[key_part1] = frames[split_frame_idx:]
            traj_to_repo_ep[key_part0] = (repo_id, episode_index, "_part0")
            traj_to_repo_ep[key_part1] = (repo_id, episode_index, "_part1")
            ep_subtasks[key_part0] = segments[:split_segment_idx]
            ep_subtasks[key_part1] = segments[split_segment_idx:]
        else:
            key = str(traj_idx)
            split_traj_frames[key] = frames
            traj_to_repo_ep[key] = (repo_id, episode_index, "")
            ep_subtasks[key] = segments

    return split_traj_frames, traj_to_repo_ep, ep_subtasks


def main(eval_config: EvalConfig):
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)

    if eval_config.split not in {"train", "val"}:
        raise ValueError(f"--split must be 'train' or 'val', got {eval_config.split!r}.")
    if eval_config.counterfactual_best_of_n and eval_config.counterfactual_action_store_dir is None:
        raise ValueError("--counterfactual-action-store-dir is required with --counterfactual-best-of-n.")

    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    config = _config.get_config(eval_config.config_name)
    if eval_config.fine_tune is not None:
        ft_config = _config.get_fine_tune_config(eval_config.fine_tune)
        config = ft_config.apply_overrides(config, pretrained_step = None)

    if eval_config.include_repos:
        config = dataclasses.replace(config, include_repos = eval_config.include_repos)
    if eval_config.counterfactual_action_store_dir is not None:
        config = dataclasses.replace(
            config,
            data = dataclasses.replace(config.data, counterfactual_action_store_dir = eval_config.counterfactual_action_store_dir),
        )
    config = dataclasses.replace(
        config,
        num_val_trajectories = eval_config.num_trajectories,
        fsdp_devices = jax.local_device_count(),
    )

    train_module = load_train_module()
    rng = jax.random.PRNGKey(86)
    mesh = sharding.make_mesh(config.fsdp_devices)
    train_state_shape, state_sharding = train_module.init_train_state(config, rng, mesh, resume = True)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        eval_config.checkpoint_path,
        keep_period = None,
        overwrite = False,
        resume = True,
    )
    restored_params = restore_params_with_shardings(checkpoint_manager, train_state_shape, state_sharding)
    critic_state_shape = train_state_shape.critic
    critic_params = restored_params["params"]["critic"]["params"]
    model = nnx.merge(critic_state_shape.model_def, critic_params)
    action_conditioned = model.network.action_conditioned
    logger.info(f"Loaded model from {eval_config.checkpoint_path}, action_conditioned={action_conditioned}")

    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.action_horizon or config.model.action_horizon
    val_tokenizer = config.data._get_critic_tokenizer(config.model)
    assert val_tokenizer is not None, "RoboCOIN evaluation requires a critic tokenizer."

    val_input_transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
        *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
        *data_config.model_transforms.inputs,
        _config.AddRoboCoinValidationVariants(
            val_tokenizer,
            use_quantile_norm = data_config.use_quantile_norm,
        ),
    ])

    cache_dir = _resolve_eval_cache_dir(eval_config)
    split = data_config.val_split if eval_config.split == "val" else eval_config.split

    if jax.process_index() == 0:
        val_trajectory_dataset = _data_loader.create_rlds_dataset(
            data_config,
            action_horizon,
            config.batch_size,
            split = split,
            shuffle = False,
            return_trajectories = True,
        )
        cache_val_episodes(
            val_trajectory_dataset,
            eval_config.num_trajectories,
            cache_dir,
            include_repos = config.include_repos,
            save_only = True,
            input_transform = val_input_transform,
        )
        del val_trajectory_dataset
    if jax.process_count() > 1:
        jax.experimental.multihost_utils.sync_global_devices("eval_cache_write")

    if eval_config.output_dir is None and jax.process_index() == 0:
        import wandb

        wandb.init(
            project = eval_config.project_name,
            name = f"eval_{eval_config.config_name}",
            config = dataclasses.asdict(eval_config),
        )

    train_module.generate_validation_plots_dlimp(
        model = model,
        val_episode_indices = list(range(eval_config.num_trajectories)),
        step = 0,
        action_conditioned = action_conditioned,
        data_config = data_config,
        cache_dir = cache_dir,
        output_dir = eval_config.output_dir,
    )

    if eval_config.counterfactual_best_of_n and action_conditioned:
        logger.info("Running BestOfN counterfactual evaluation...")
        import jax.numpy as jnp

        traj_frames = _load_cached_trajectories(cache_dir)
        split_traj_frames, traj_to_repo_ep, ep_subtasks = _split_trajectory_frames(traj_frames)

        first_frames = next(iter(split_traj_frames.values()))
        num_samples = first_frames[0]["counterfactual_actions"].shape[0]
        network_config = config.model.network_config

        bon_model = BestOfNWrapper(
            action_dim = network_config.action_dim,
            action_horizon = action_horizon,
            max_token_len = network_config.max_token_len,
            base_model = None,
            num_samples = num_samples,
            take_min_over_ensemble = True,
            use_target_value = False,
            selection_mode = "argmax",
            softmax_temperature = 1.0,
        )

        @nnx.jit
        def _jitted_bon_eval(bon, vf, rng, obs, cf_actions):
            transition = _base_vf.Transition(
                observation = obs,
                counterfactual_actions = cf_actions,
            )
            best_action = bon.sample_actions(rng, transition, compute_next_action = False, value_function = vf)
            result = vf.compute_value(obs, best_action, take_min_over_ensemble = True)
            q_value = result[0] if isinstance(result, tuple) else result
            return q_value

        BATCH_SIZE = 8
        bon_rng = jax.random.PRNGKey(86)
        ep_mc_returns = {}
        ep_frame_images = {}
        ep_fps = {}
        ep_include_masks = {}
        bon_predictions: dict[str, list[float]] = {}

        for traj_idx, frames in split_traj_frames.items():
            ep_mc_returns[traj_idx] = [f["mc_return"] for f in frames]
            ep_frame_images[traj_idx] = [
                np.stack([
                    np.asarray(f["image"]["left_wrist_0_rgb"]),
                    np.asarray(f["image"]["right_wrist_0_rgb"]),
                    np.asarray(f["image"]["base_0_rgb"]),
                ])
                for f in frames
            ]
            ep_fps[traj_idx] = int(frames[0]["fps"])
            ep_include_masks[traj_idx] = [bool(f.get("include_subtask", True)) for f in frames]
            bon_predictions[traj_idx] = []

            for batch_start in range(0, len(frames), BATCH_SIZE):
                batch_frames = frames[batch_start : batch_start + BATCH_SIZE]
                obs, _ = get_obs_and_action(batch_frames, prefix = "", action_conditioned = True)
                cf_actions = jnp.asarray(np.stack([f["counterfactual_actions"] for f in batch_frames], axis = 0))
                bon_rng, step_rng = jax.random.split(bon_rng)
                q_values = jax.device_get(_jitted_bon_eval(bon_model, model, step_rng, obs, cf_actions))
                bon_predictions[traj_idx].extend(q_values.tolist())

        total = sum(len(preds) for preds in bon_predictions.values())
        logger.info(f"Computed {total} BestOfN predictions")

        if jax.process_index() == 0:
            best_of_n_images = {}
            for traj_idx in ep_mc_returns:
                repo_id, ep_idx, part_suffix = traj_to_repo_ep[traj_idx]
                plot_key = f"val/{repo_id.removeprefix('RoboCOIN/')}_episode_{ep_idx}{part_suffix}_best_of_n"
                mc_returns = ep_mc_returns[traj_idx]
                include_masks = ep_include_masks[traj_idx]
                predicted_values = bon_predictions[traj_idx]

                if len(predicted_values) != len(mc_returns):
                    raise ValueError(
                        f"BestOfN prediction length mismatch for {traj_idx}: "
                        f"{len(predicted_values)} predictions vs {len(mc_returns)} returns."
                    )

                filtered_mc = [mc for mc, include in zip(mc_returns, include_masks, strict = True) if include]
                filtered_pred = [pred for pred, include in zip(predicted_values, include_masks, strict = True) if include]
                filtered_images = [img for img, include in zip(ep_frame_images[traj_idx], include_masks, strict = True) if include]

                if not filtered_mc:
                    continue

                best_of_n_images[plot_key] = train_module._create_value_plot(
                    filtered_mc,
                    filtered_pred,
                    ep_idx,
                    0,
                    " (BestOfN Q)",
                    oracle_values = None,
                    subtask_texts = ep_subtasks[traj_idx],
                    plot_video = True,
                    frame_images = filtered_images,
                    fps = ep_fps[traj_idx],
                    output_dir = eval_config.output_dir,
                    plot_key = plot_key,
                )

            if eval_config.output_dir is None and best_of_n_images:
                import wandb

                wandb.log(best_of_n_images)

    if hasattr(train_module, "_render_thread") and train_module._render_thread is not None:
        train_module._render_thread.join()
        train_module._render_thread = None

    if eval_config.output_dir is None and jax.process_index() == 0:
        import wandb

        wandb.finish()

    logger.info("Evaluation complete")


if __name__ == "__main__":
    import tyro

    eval_config = tyro.cli(EvalConfig)
    main(eval_config)
