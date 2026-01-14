"""Functional policy extraction objectives and config classes.

This module provides pure functions for computing policy training objectives.
These can be combined using weighted_sum_objective to create composite
training losses for policy extraction.

Each objective returns (per_sample_loss, info_dict) where:
- per_sample_loss has shape (batch,)
- info_dict contains metrics for logging

Config classes (BasePolicyExtractionConfig, NoopPolicyConfig, DDPGPolicyConfig,
AWRPolicyConfig) wrap these objectives with hyperparameters.
"""

import abc
from collections.abc import Callable, Sequence
import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions.base_value_functions import BaseValueFunction

# Type alias for objective functions
PolicyObjective = Callable[..., tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]]


def ddpg_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    q_function: BaseValueFunction,
    *,
    aggregation: Literal["min", "mean"] = "min",
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
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
    take_min = aggregation == "min"
    q_value = q_function.compute_value(observation, actions, take_min_over_ensemble=take_min)

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
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
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
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """Entropy objective: log π(a|s).

    This objective encourages exploration by maximizing policy entropy.
    Returns log_prob so that minimizing this loss (with a negative weight
    or via temperature scaling) maximizes entropy.

    In SAC, this is combined with the DDPG objective:
        L = -Q(s, a) + alpha * log pi(a|s)

    Args:
        policy: Policy model with action_distribution method.
        observation: Batch of observations.
        rng: Random key for sampling actions.

    Returns:
        Tuple of (per_sample_loss, info_dict).
        The loss is log_prob, so minimizing -loss maximizes entropy.
    """
    dist = policy.action_distribution(rng, observation)
    actions, log_prob = dist.sample_and_log_prob(seed=rng)
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
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
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


def noop_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """No-op objective that returns zero loss.

    Use this for critic-only training where no policy updates are needed.

    Args:
        policy: Policy model (unused, but required by interface).
        observation: Batch of observations.
        rng: Random key (unused).

    Returns:
        Tuple of (zero loss, empty info dict).
    """
    del policy, rng
    batch_size = observation.state.shape[0]
    return jnp.zeros(batch_size), {}


def awr_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    data_action: _model.Actions,
    q_function: BaseValueFunction,
    v_function: BaseValueFunction,
    *,
    temperature: float = 1.0,
    clip_exp: float = 100.0,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """AWR objective: exp(A/β) * -log π(a|s).

    Advantage Weighted Regression (AWR) trains the policy to maximize
    likelihood of data actions, weighted by exponentiated advantages.

    The loss is: exp(A/β) * -log π(a|s), where β is temperature.

    Args:
        policy: Policy model with action_distribution method.
        observation: Batch of observations.
        rng: Random key for action distribution.
        data_action: Ground truth actions from dataset,
            shape (batch, action_horizon, action_dim).
        q_function: Q-function Q(s, a) for advantage computation.
        v_function: V-function V(s) for advantage computation.
        temperature: Temperature β for advantage weighting. Lower = more greedy.
        clip_exp: Maximum value for exp(A/β) to prevent explosion.

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    batch_size = observation.state.shape[0]

    q_value = q_function.compute_value(observation, data_action, take_min_over_ensemble=True)
    v_value = v_function.compute_value(observation)
    advantage = q_value - v_value

    weights = jnp.exp(advantage * temperature)
    weights = jnp.minimum(weights, clip_exp)
    weights = jax.lax.stop_gradient(weights)

    dist = policy.action_distribution(rng, observation)
    actions_flat = data_action.reshape(batch_size, -1)
    log_prob = dist.log_prob(actions_flat)

    loss = -weights * log_prob
    assert loss.shape == (batch_size,), f"Expected loss shape ({batch_size},), got {loss.shape}"

    info = {
        "advantage_mean": jnp.mean(advantage),
        "advantage_std": jnp.std(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_max": jnp.max(advantage),
        "weight_mean": jnp.mean(weights),
        "weight_std": jnp.std(weights),
        "weight_max": jnp.max(weights),
        "log_prob_mean": jnp.mean(log_prob),
        "log_prob_std": jnp.std(log_prob),
        "q_value_mean": jnp.mean(q_value),
        "v_value_mean": jnp.mean(v_value),
    }
    return loss, info


# =============================================================================
# Policy Extraction Config Classes
# =============================================================================


@dataclasses.dataclass(frozen=True)
class BasePolicyExtractionConfig(abc.ABC):
    """Base config for policy extraction objectives.

    Subclasses define specific objective types (noop, DDPG, AWR, etc.)
    with their own hyperparameters.
    """

    @abc.abstractmethod
    def compute_loss(
        self,
        policy: _model.BaseModel,
        observation: _model.Observation,
        rng: at.KeyArrayLike,
        data_action: _model.Actions | None = None,
        value_function: BaseValueFunction | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute policy extraction loss.

        Args:
            policy: The policy to train.
            observation: Batch of observations.
            rng: Random key.
            data_action: Actions from the dataset (for BC-based methods).
            value_function: Value function (Q and/or V) for advantage computation.

        Returns:
            Tuple of (per_sample_loss, info_dict).
        """


@dataclasses.dataclass(frozen=True)
class NoopPolicyConfig(BasePolicyExtractionConfig):
    """No policy training (critic-only mode).

    Use this when training only the value function without policy updates.
    Returns zero loss for all samples.
    """

    def compute_loss(
        self,
        policy: _model.BaseModel,
        observation: _model.Observation,
        rng: at.KeyArrayLike,
        data_action: _model.Actions | None = None,
        value_function: BaseValueFunction | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del data_action, value_function
        return noop_objective(policy, observation, rng)


@dataclasses.dataclass(frozen=True)
class DDPGPolicyConfig(BasePolicyExtractionConfig):
    """DDPG objective: maximize Q(s, π(s)).

    The policy is trained to output actions that maximize the Q-function.
    Optionally includes BC regularization and entropy bonus.
    """

    bc_weight: float = 0.0
    entropy_weight: float = 0.0
    q_aggregation: Literal["min", "mean"] = "min"

    def compute_loss(
        self,
        policy: _model.BaseModel,
        observation: _model.Observation,
        rng: at.KeyArrayLike,
        data_action: _model.Actions | None = None,
        value_function: BaseValueFunction | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        if value_function is None:
            raise ValueError("DDPGPolicyConfig requires a Q-function.")

        objectives_and_weights: list[tuple] = [
            (
                ddpg_objective,
                1.0,
                {"q_function": value_function, "aggregation": self.q_aggregation},
            ),
        ]

        if self.entropy_weight > 0:
            objectives_and_weights.append((entropy_objective, self.entropy_weight, {}))

        if self.bc_weight > 0 and data_action is not None:
            objectives_and_weights.append((bc_regularization_objective, self.bc_weight, {"data_action": data_action}))

        return weighted_sum_objective(policy, observation, rng, objectives_and_weights)


@dataclasses.dataclass(frozen=True)
class AWRPolicyConfig(BasePolicyExtractionConfig):
    """AWR objective: exp(A/β) weighted BC. Requires IQL (Q + V).

    Advantage Weighted Regression trains the policy by weighting the
    log-likelihood of data actions by exponentiated advantages.

    A = Q(s, a) - V(s)
    Loss = exp(A/β) * -log π(a|s)

    This is the standard policy extraction method for IQL.
    """

    temperature: float = 1.0
    clip_exp: float = 100.0
    bc_weight: float = 0.0

    def compute_loss(
        self,
        policy: _model.BaseModel,
        observation: _model.Observation,
        rng: at.KeyArrayLike,
        data_action: _model.Actions | None = None,
        value_function: BaseValueFunction | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        if data_action is None:
            raise ValueError("AWRPolicyConfig requires data_action.")
        if value_function is None:
            raise ValueError("AWRPolicyConfig requires a value function with Q and V.")

        # For IQL, we need both Q(s,a) and V(s). IQLValueFunction exposes both:
        # - compute_value(obs, action) -> Q(s, a)
        # - compute_value(obs, None) -> V(s)
        # We treat value_function as both Q and V since IQL exposes both.

        loss, info = awr_objective(
            policy,
            observation,
            rng,
            data_action,
            q_function=value_function,
            v_function=value_function,
            temperature=self.temperature,
            clip_exp=self.clip_exp,
        )

        if self.bc_weight > 0:
            bc_loss, bc_info = bc_regularization_objective(policy, observation, rng, data_action)
            loss = loss + self.bc_weight * bc_loss
            info.update({f"bc/{k}": v for k, v in bc_info.items()})
            info["bc/weight"] = self.bc_weight

        return loss, info


def awr_multi_objective(
    policy: _model.BaseModel,
    observation: _model.Observation,
    rng: at.KeyArrayLike,
    data_action: _model.Actions,
    q_function: BaseValueFunction,
    v_function: BaseValueFunction,
    *,
    temperature: float = 1.0,
    clip_exp: float = 100.0,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """AWR objective for multi-transition value functions.

    Flattens [batch, n, ...] to [batch*n, ...] so each transition independently
    contributes to policy training with its own advantage.

    Args:
        policy: Policy model with action_distribution method.
        observation: Multi-transition observations with state shape [batch, n, state_dim].
        rng: Random key for action distribution.
        data_action: Actions from dataset, shape [batch, n, action_horizon, action_dim].
        q_function: Multi-transition Q-function returning [batch, n].
        v_function: Multi-transition V-function returning [batch, n].
        temperature: Temperature β for advantage weighting. Lower = more greedy.
        clip_exp: Maximum value for exp(A/β) to prevent explosion.

    Returns:
        Tuple of (per_sample_loss with shape [batch*n], info_dict).
    """
    batch_size, n = observation.state.shape[:2]
    state_dim = observation.state.shape[2]
    flat_size = batch_size * n

    q_value = q_function.compute_value(observation, data_action, take_min_over_ensemble=True)
    v_value = v_function.compute_value(observation)
    advantage = q_value - v_value  # [batch, n]

    flat_advantage = advantage.reshape(flat_size)
    weights = jnp.exp(flat_advantage * temperature)
    weights = jnp.minimum(weights, clip_exp)
    weights = jax.lax.stop_gradient(weights)

    flat_state = observation.state.reshape(flat_size, state_dim)
    flat_obs = _model.Observation(
        images={},
        image_masks={},
        state=flat_state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )

    flat_action = data_action.reshape(flat_size, policy.action_horizon, policy.action_dim)

    dist = policy.action_distribution(rng, flat_obs)
    actions_flat = flat_action.reshape(flat_size, -1)
    log_prob = dist.log_prob(actions_flat)

    loss = -weights * log_prob

    info = {
        "advantage_mean": jnp.mean(advantage),
        "advantage_std": jnp.std(advantage),
        "advantage_min": jnp.min(advantage),
        "advantage_max": jnp.max(advantage),
        "weight_mean": jnp.mean(weights),
        "weight_std": jnp.std(weights),
        "weight_max": jnp.max(weights),
        "log_prob_mean": jnp.mean(log_prob),
        "log_prob_std": jnp.std(log_prob),
        "q_value_mean": jnp.mean(q_value),
        "v_value_mean": jnp.mean(v_value),
    }
    return loss, info


@dataclasses.dataclass(frozen=True)
class MultiAWRPolicyConfig(BasePolicyExtractionConfig):
    """AWR objective for multi-transition value functions.

    Flattens [batch, n, ...] to [batch*n, ...] so each transition independently
    contributes to policy training. Use with MultiIQLValueFunction.
    """

    temperature: float = 1.0
    clip_exp: float = 100.0
    bc_weight: float = 0.0

    def compute_loss(
        self,
        policy: _model.BaseModel,
        observation: _model.Observation,
        rng: at.KeyArrayLike,
        data_action: _model.Actions | None = None,
        value_function: BaseValueFunction | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        if data_action is None:
            raise ValueError("MultiAWRPolicyConfig requires data_action.")
        if value_function is None:
            raise ValueError("MultiAWRPolicyConfig requires a value function with Q and V.")

        loss, info = awr_multi_objective(
            policy,
            observation,
            rng,
            data_action,
            q_function=value_function,
            v_function=value_function,
            temperature=self.temperature,
            clip_exp=self.clip_exp,
        )

        if self.bc_weight > 0:
            batch_size, n = observation.state.shape[:2]
            flat_size = batch_size * n
            state_dim = observation.state.shape[2]
            flat_obs = _model.Observation(
                images={},
                image_masks={},
                state=observation.state.reshape(flat_size, state_dim),
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )
            flat_action = data_action.reshape(flat_size, policy.action_horizon, policy.action_dim)
            bc_loss, bc_info = bc_regularization_objective(policy, flat_obs, rng, flat_action)
            loss = loss + self.bc_weight * bc_loss
            info.update({f"bc/{k}": v for k, v in bc_info.items()})
            info["bc/weight"] = self.bc_weight

        return loss, info
