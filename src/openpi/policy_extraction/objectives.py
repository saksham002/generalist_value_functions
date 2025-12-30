"""Functional policy extraction objectives.

This module provides pure functions for computing policy training objectives.
These can be combined using weighted_sum_objective to create composite
training losses for policy extraction.

Each objective returns (per_sample_loss, info_dict) where:
- per_sample_loss has shape (batch,)
- info_dict contains metrics for logging
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions.base import BaseValueFunction


# Type alias for objective functions
PolicyObjective = Callable[..., tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]]


def ddpg_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    q_function: BaseValueFunction,
    *,
    aggregation: Literal["min", "mean"] = "min",
) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
    """DDPG objective: maximize Q(s, a) where a ~ π(·|s).

    This is the core policy improvement objective in actor-critic methods.
    Returns -Q(s, a) so that minimizing this loss maximizes Q.

    Args:
        policy: Policy model with action_distribution method.
        observation: Batch of observations.
        rng: Random key for sampling actions.
        q_function: Q-function (may be an ensemble).
        aggregation: How to aggregate over ensemble:
            - "min": Use minimum Q-value (pessimistic, SAC/TD3 style)
            - "mean": Use mean Q-value

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    dist = policy.action_distribution(rng, observation)
    actions_flat = dist.sample(seed=rng)

    batch_size = observation.state.shape[0]
    actions = actions_flat.reshape(batch_size, policy.action_horizon, policy.action_dim)

    # Compute Q-value with appropriate aggregation
    if hasattr(q_function, "compute_min_value") and aggregation == "min":
        q_value = q_function.compute_min_value(observation, actions)
    elif hasattr(q_function, "compute_mean_value") and aggregation == "mean":
        q_value = q_function.compute_mean_value(observation, actions)
    else:
        # Fallback for non-ensemble Q-functions
        q_value = q_function.compute_value(observation, actions)

    loss = -q_value  # Minimize negative Q = maximize Q
    assert loss.shape == (batch_size,), f"Expected loss shape ({batch_size},), got {loss.shape}"

    info = {
        "q_value_mean": jnp.mean(q_value),
        "q_value_std": jnp.std(q_value),
        "q_value_min": jnp.min(q_value),
        "q_value_max": jnp.max(q_value),
    }
    return loss, info


def bc_regularization_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    data_action: _model.Actions,
) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
    """Behavior cloning regularization: penalize deviation from data actions.

    This objective encourages the policy to stay close to the behavior
    demonstrated in the dataset. Useful for offline RL to prevent
    distribution shift.

    Returns ||π(s) - a_data||^2 (mean squared error).

    Args:
        policy: Policy model.
        observation: Batch of observations.
        rng: Random key for sampling actions.
        data_action: Ground truth actions from dataset,
            shape (batch, action_horizon, action_dim).

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    dist = policy.action_distribution(rng, observation)
    actions_flat = dist.sample(seed=rng)

    batch_size = observation.state.shape[0]
    policy_action = actions_flat.reshape(batch_size, policy.action_horizon, policy.action_dim)

    # MSE loss (mean over action_horizon and action_dim)
    squared_error = jnp.square(policy_action - data_action)
    mse = jnp.mean(squared_error, axis=(-2, -1))
    assert mse.shape == (batch_size,), f"Expected mse shape ({batch_size},), got {mse.shape}"

    info = {
        "bc_mse": jnp.mean(mse),
        "bc_mse_std": jnp.std(mse),
        "action_diff_mean": jnp.mean(jnp.abs(policy_action - data_action)),
    }
    return mse, info


def entropy_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
    """Entropy objective: log π(a|s).

    This objective encourages exploration by maximizing policy entropy.
    Returns log_prob so that minimizing this loss (with a negative weight
    or via temperature scaling) maximizes entropy.

    In SAC, this is combined with the DDPG objective:
        L = -Q(s, a) + α * log π(a|s)

    Args:
        policy: Policy model with action_distribution method.
        observation: Batch of observations.
        rng: Random key for sampling actions.

    Returns:
        Tuple of (per_sample_loss, info_dict).
        The loss is log_prob, so minimizing -loss maximizes entropy.
    """
    dist = policy.action_distribution(rng, observation)
    actions = dist.sample(seed=rng)
    log_prob = dist.log_prob(actions)
    batch_size = observation.state.shape[0]
    assert log_prob.shape == (batch_size,), f"Expected log_prob shape ({batch_size},), got {log_prob.shape}"

    info = {
        "entropy": jnp.mean(-log_prob),
        "entropy_std": jnp.std(-log_prob),
        "log_prob_mean": jnp.mean(log_prob),
        "log_prob_std": jnp.std(log_prob),
    }
    return log_prob, info


def weighted_sum_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    objectives_and_weights: Sequence[tuple[PolicyObjective, float, dict]],
) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
    """Compute weighted sum of policy objectives.

    This function combines multiple policy objectives into a single loss.
    Each objective is called with its specific kwargs and weighted.

    Args:
        policy: The policy model.
        observation: Batch of observations.
        rng: Random key.
        objectives_and_weights: List of (objective_fn, weight, kwargs) tuples.
            Each objective_fn is called as:
                objective_fn(policy, observation, rng, **kwargs)

    Returns:
        Tuple of (per_sample_loss, info_dict) where loss is sum of
        weighted objectives.

    Example:
        ```python
        objectives_and_weights = [
            (ddpg_objective, 1.0, {"q_function": q_fn, "aggregation": "min"}),
            (entropy_objective, temperature, {}),
            (bc_regularization_objective, 0.1, {"data_action": actions}),
        ]
        loss, info = weighted_sum_objective(policy, obs, rng, objectives_and_weights)
        ```
    """
    batch_size = observation.state.shape[0]
    total_loss = jnp.zeros(batch_size)
    all_info: dict[str, at.Array] = {}

    for objective_fn, weight, kwargs in objectives_and_weights:
        # Split rng for each objective to ensure independence
        rng, obj_rng = jax.random.split(rng)

        obj_loss, obj_info = objective_fn(policy, observation, obj_rng, **kwargs)

        total_loss = total_loss + weight * obj_loss

        # Prefix info keys with objective name for clarity
        obj_name = objective_fn.__name__.replace("_objective", "")
        all_info.update({f"{obj_name}/{k}": v for k, v in obj_info.items()})
        all_info[f"{obj_name}/weight"] = jnp.array(weight)
        all_info[f"{obj_name}/weighted_loss"] = jnp.mean(weight * obj_loss)

    assert total_loss.shape == (batch_size,), f"Expected total_loss shape ({batch_size},), got {total_loss.shape}"
    all_info["total_loss"] = jnp.mean(total_loss)

    return total_loss, all_info
