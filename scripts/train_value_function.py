"""Training script for value functions with optional policy training.

Supports:
- Critic-only training (default)
- Joint critic + policy training with configurable update ratio
- Policy extraction with frozen critic (critic_steps_per_policy_step=0)
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

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

from openpi.models import model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.evaluation as _evaluation
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
from openpi.training.time_utils import Timer
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
import openpi.value_functions.base as _value_fn
from openpi.value_functions.value_function import MultiValueFunctionConfig


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


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
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
        run_name = config.exp_name if config.exp_name else config.name
        wandb.init(
            name=run_name,
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
    model_config: _value_fn.BaseValueFunctionConfig = config.model

    def init_critic(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=critic_tx,
            opt_state=critic_tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    policy_tx = None
    if config.policy is not None:
        policy_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init_policy(rng: at.KeyArrayLike) -> training_utils.TrainState:
        if config.policy is None or policy_tx is None:
            raise ValueError("Config does not specify a policy")
        policy_model = config.policy.create(rng)
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
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.ActorCriticTrainState:
        rng, critic_rng, policy_rng = jax.random.split(rng, 3)
        critic_state = init_critic(critic_rng, partial_params)

        policy_state = None
        if config.policy is not None:
            policy_state = init_policy(policy_rng)

        return training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)

    train_state_shape = jax.eval_shape(init_actor_critic, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.critic.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init_actor_critic,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def value_function_train_step(
    config: _config.TrainConfig,
    lr_schedule: optax.Schedule,
    state: training_utils.TrainState,
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
    model = nnx.merge(state.model_def, state.params)

    if isinstance(config.model, MultiValueFunctionConfig):
        transition = _value_fn.MultiTransition.from_batch(batch)
    else:
        transition = _value_fn.Transition.from_batch(batch)

    def loss_fn(model: _value_fn.BaseValueFunction):
        # compute_loss returns (per_sample_loss, info_dict)
        per_sample_loss, value_info = model.compute_loss(transition, train=True, rng=rng)
        return jnp.mean(per_sample_loss), value_info

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
        "learning_rate": lr_schedule(state.step),
        **value_info,
        **batch_stats,
    }
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
        batch: Batch of data.
        rng: Random key.

    Returns:
        Tuple of (new_policy_state, info_dict).
    """
    policy = nnx.merge(policy_state.model_def, policy_state.params)
    critic = nnx.merge(critic_state.model_def, critic_state.params)

    # Build observation from batch - single transition only
    observation = _model.Observation(
        images={},
        image_masks={},
        state=jnp.asarray(batch["state"]),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )

    # Extract data action for BC-based objectives
    data_action = jnp.asarray(batch["actions"])

    def loss_fn(policy):
        loss, info = config.policy_extraction.compute_loss(
            policy=policy,
            observation=observation,
            rng=rng,
            data_action=data_action,
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

    info = {
        "policy/loss": loss,
        "policy/grad_norm": optax.global_norm(grads),
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
        pred_value = model.compute_value(obs, act)
        predicted_values.append(float(jax.device_get(pred_value[0])))
    return predicted_values


def _generate_multi_transition_plots(
    model, frames, mc_returns, ep_idx, step, num_transitions, normalize, action_conditioned, _model, oracle_values=None
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

        pred_values = model.compute_value(obs, action)  # [1, n]
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

        pred_values = model.compute_value(obs, action)  # [1, n]
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

        pred_values = model.compute_value(obs, action)
        for local_idx, global_idx in enumerate(chunk_indices):
            if pred_chunks[global_idx] is None:
                pred_chunks[global_idx] = float(jax.device_get(pred_values[0, local_idx]))

    images[f"val/episode_{ep_idx}_chunks"] = _create_value_plot(
        mc_returns, pred_chunks, ep_idx, step, " (continuous chunks)", oracle_values
    )

    return images


def _create_value_plot(mc_returns, predicted_values, ep_idx, step, suffix, oracle_values=None) -> "wandb.Image":
    """Create a matplotlib plot comparing MC returns vs predicted values."""
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

    plt.tight_layout()
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

            pred_values = jax.device_get(model.compute_value(obs, action))
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
                pred_value = jax.device_get(model.compute_value(obs, action))
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
            pred_qs = jax.device_get(model.compute_value(obs, action))
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
                pred_q = jax.device_get(model.compute_value(obs, action))
                pred_qs.append(float(pred_q[0]))
            pred_qs = np.array(pred_qs)

        tau, _ = scipy.stats.kendalltau(pred_qs, oracle_qs)
        if not np.isnan(tau):
            action_taus.append(tau)

    if action_taus:
        metrics["val/oracle_rank_fixed_actions"] = float(np.mean(action_taus))

    return metrics


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
    # Create validation dataset - use NumpyDataset for minari, LeRobotDataset otherwise
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.minari_dataset_id is not None:
        val_dataset = _data_loader.create_numpy_dataset_from_minari(
            data_config.minari_dataset_id,
            discount=data_config.discount,
            reward_scale=data_config.reward_scale,
            reward_bias=data_config.reward_bias,
        )
    else:
        from lerobot.common.datasets import lerobot_dataset

        val_dataset = lerobot_dataset.LeRobotDataset(config.data.repo_id)

    num_episodes = val_dataset.num_episodes
    val_rng = np.random.default_rng(config.seed)
    val_episode_indices = val_rng.choice(
        num_episodes, size=min(config.num_val_trajectories, num_episodes), replace=False
    ).tolist()
    logging.info(f"Selected validation episodes: {val_episode_indices}")
    network_config = getattr(config.model, "network_config", None)
    action_conditioned = getattr(network_config, "action_conditioned", False)

    # Set up evaluation environment if enabled
    eval_env = None
    eval_enabled = config.eval_interval > 0 and config.eval_env is not None and policy_training_enabled
    if eval_enabled:
        # Use vectorized environments for parallel evaluation
        num_eval_envs = min(config.eval_env.num_eval_episodes, 8)
        eval_env = _evaluation.create_vector_eval_env(config.eval_env, data_config, num_envs=num_eval_envs)
        logging.info(
            f"Evaluation enabled: {config.eval_env.num_eval_episodes} episodes every {config.eval_interval} steps "
            f"({num_eval_envs} parallel envs)"
        )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

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

    lr_schedule = config.lr_schedule.create()
    ptrain_step = jax.jit(
        functools.partial(value_function_train_step, config, lr_schedule),
        in_shardings=(critic_sharding, data_sharding, replicated_sharding),
        out_shardings=(critic_sharding, replicated_sharding),
        donate_argnums=(0,),
    )

    ppolicy_step = None
    if policy_state is not None:
        ppolicy_step = jax.jit(
            functools.partial(policy_train_step, config, lr_schedule),
            in_shardings=(critic_sharding, policy_sharding, data_sharding, replicated_sharding),
            out_shardings=(policy_sharding, replicated_sharding),
        )

    start_step = int(critic_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    timer = Timer()

    # Log initial timing info
    logging.info(f"Starting training with batch_size={config.batch_size}, num_workers={config.num_workers}")

    for step in pbar:
        # Split rng for this step
        rng, step_rng = jax.random.split(rng)

        with timer.context("train_step_compute"), sharding.set_mesh(mesh):
            critic_state, info = ptrain_step(critic_state, batch, step_rng)

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
                policy_info = jax.device_get(policy_info)
                info.update(policy_info)

        if step % config.log_interval == 0:
            info = jax.device_get(info)
            # Add timing info to logged metrics (average and total)
            total_times = timer.get_total_times(reset=False)
            avg_times = timer.get_average_times(reset=True)
            timing_info = {f"average_times/{k}": v for k, v in avg_times.items()}
            timing_info.update({f"total_times/{k}": v for k, v in total_times.items()})
            info.update(timing_info)

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(info, step=step)

        # Break down data loading into components
        with timer.context("data_fetch"):
            raw_batch = next(data_iter)

        with timer.context("data_postprocess"):
            if isinstance(raw_batch, tuple):
                obs, _ = raw_batch
                batch = {"state": obs.state}
            else:
                batch = raw_batch

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            with timer.context("checkpoint_save"):
                state_to_save = training_utils.ActorCriticTrainState(critic=critic_state, policy=policy_state)
                _checkpoints.save_state(checkpoint_manager, state_to_save, data_loader, step)

        # Generate validation plots
        if step % config.plot_interval == 0 and step > 0:
            with timer.context("validation_plot"):
                model = nnx.merge(critic_state.model_def, critic_state.params)

                plot_images = generate_validation_plots(
                    model=model,
                    dataset=val_dataset,
                    val_episode_indices=val_episode_indices,
                    step=step,
                    action_conditioned=action_conditioned,
                    data_config=data_config,
                )
                if plot_images:
                    wandb.log(plot_images, step=step)

        # Policy evaluation
        if eval_enabled and step % config.eval_interval == 0 and step > 0:
            with timer.context("policy_eval"):
                policy_model = nnx.merge(policy_state.model_def, policy_state.params)

                # Create normalization transform
                normalize = _transforms.Normalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                    strict=False,
                )

                def policy_fn(
                    obs_batch: np.ndarray,
                    normalize=normalize,
                    policy_model=policy_model,
                    step=step,
                ) -> np.ndarray:
                    # Normalize batch of observations: [num_envs, obs_dim]
                    normalized_obs = np.stack(
                        [normalize({"state": obs})["state"] for obs in obs_batch],
                        axis=0,
                    )

                    # Build model observation for batch
                    model_obs = _model.Observation(
                        images={},
                        image_masks={},
                        state=jnp.asarray(normalized_obs),  # [num_envs, obs_dim]
                        tokenized_prompt=None,
                        tokenized_prompt_mask=None,
                    )

                    # Sample actions from policy for all envs
                    actions = policy_model.sample_actions(rng=jax.random.key(step), observation=model_obs)
                    # Take the first action in the horizon and convert to numpy
                    # Shape: [num_envs, action_horizon, action_dim] -> [num_envs, action_dim]
                    return np.asarray(jax.device_get(actions[:, 0, :]))

                eval_results = _evaluation.evaluate_policy_vectorized(
                    policy_fn=policy_fn,
                    vec_env=eval_env,
                    num_episodes=config.eval_env.num_eval_episodes,
                    seed=config.eval_env.seed + step,
                )
                eval_metrics = eval_results.to_dict()
                wandb.log(eval_metrics, step=step)
                logging.info(
                    f"Step {step} eval: mean_return={eval_results.mean_return:.2f}, "
                    f"std_return={eval_results.std_return:.2f}"
                )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
