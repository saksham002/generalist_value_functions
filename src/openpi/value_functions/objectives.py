"""Functional value function training objectives.

All objectives are pure functions that compute loss given network, head, and transition.
They support both regression and categorical heads via _compute_value_loss helper.
"""

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.policy_extraction.temperature import Temperature
from openpi.shared import array_typing as at
from openpi.value_functions import hl_gauss as _hl_gauss
from openpi.value_functions.base import Transition
from openpi.value_functions.heads import CategoricalHead
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base import BaseValueNetwork


def _compute_value_loss(head: ValueHead, features: at.Array, target: at.Array) -> at.Array:
    """Unified loss computation for any head type.

    Args:
        head: Value head (RegressionHead or CategoricalHead).
        features: Network features of shape (batch, feature_dim).
        target: Target values of shape (batch,).

    Returns:
        Per-sample loss of shape (batch,).
    """
    if isinstance(head, CategoricalHead):
        logits = head.compute_logits(features)
        return _hl_gauss.hl_gauss_loss(logits, target, head.v_min, head.v_max, head.sigma)
    return jnp.square(head(features) - target)


def mc_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """Monte-Carlo regression: target = mc_return.

    Args:
        network: Value network for feature extraction.
        head: Value head for value prediction.
        transition: Transition containing observation, action, mc_return.

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    action = transition.action if network.action_conditioned else None
    features = network.compute_features(transition.observation, action)
    loss = _compute_value_loss(head, features, transition.mc_return)

    pred = head(features)
    td_error = pred - transition.mc_return

    info = {
        "predicted_value_mean": jnp.mean(pred),
        "predicted_value_std": jnp.std(pred),
        "target_value_mean": jnp.mean(transition.mc_return),
        "target_value_std": jnp.std(transition.mc_return),
        "td_error_mean": jnp.mean(td_error),
        "td_error_std": jnp.std(td_error),
    }
    return loss, info


def sarsa_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    *,
    discount: float = 0.99,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """SARSA: target = r + gamma * Q(s', a').

    Args:
        network: Online value network.
        head: Online value head.
        transition: Transition containing all SARSA components.
        target_network: Target network for stable target computation.
        target_head: Target head.
        discount: Discount factor gamma.

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    target_features = target_network.compute_features(transition.next_observation, transition.next_action)
    target_value = target_head(target_features)

    not_done = 1.0 - transition.termination.astype(jnp.float32)
    target = transition.reward + discount * not_done * target_value
    target = jax.lax.stop_gradient(target)

    action = transition.action if network.action_conditioned else None
    features = network.compute_features(transition.observation, action)
    loss = _compute_value_loss(head, features, target)

    pred = head(features)
    td_error = pred - target

    info = {
        "predicted_value_mean": jnp.mean(pred),
        "predicted_value_std": jnp.std(pred),
        "target_value_mean": jnp.mean(target),
        "target_value_std": jnp.std(target),
        "td_error_mean": jnp.mean(td_error),
        "td_error_std": jnp.std(td_error),
        "next_value_mean": jnp.mean(target_value),
    }
    return loss, info


def iql_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition,
    *,
    expectile: float = 0.7,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """IQL expectile regression.

    Uses asymmetric L2 loss (expectile loss) to avoid querying OOD actions.

    Args:
        network: Value network.
        head: Value head.
        transition: Transition with mc_return as the target.
        expectile: Asymmetry parameter (tau). Values > 0.5 weight positive errors more.

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    action = transition.action if network.action_conditioned else None
    features = network.compute_features(transition.observation, action)
    pred = head(features)

    target = transition.mc_return
    diff = target - pred
    weight = jnp.where(diff > 0, expectile, 1 - expectile)
    loss = weight * jnp.square(diff)

    info = {
        "predicted_value_mean": jnp.mean(pred),
        "predicted_value_std": jnp.std(pred),
        "target_value_mean": jnp.mean(target),
        "target_value_std": jnp.std(target),
        "td_error_mean": jnp.mean(diff),
        "td_error_std": jnp.std(diff),
        "positive_error_frac": jnp.mean((diff > 0).astype(jnp.float32)),
    }
    return loss, info


def sac_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    policy: _model.BaseModel,
    temperature: Temperature,
    *,
    discount: float = 0.99,
    rng: at.KeyArrayLike,
) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
    """SAC soft Bellman: target = r + gamma * (Q(s', a') - alpha * log pi(a'|s')).

    Args:
        network: Online Q-network.
        head: Online Q-head.
        transition: Transition with observation, action, reward, next_observation.
        target_network: Target Q-network.
        target_head: Target Q-head.
        policy: Policy network for sampling next actions.
        temperature: Temperature (alpha) for entropy regularization.
        discount: Discount factor gamma.
        rng: Random key for sampling next actions.

    Returns:
        Tuple of (per_sample_loss, info_dict).
    """
    rng, sample_rng = jax.random.split(rng)

    next_dist = policy.action_distribution(sample_rng, transition.next_observation)
    next_actions_flat = next_dist.sample(seed=sample_rng)
    next_log_prob = next_dist.log_prob(next_actions_flat)

    batch_size = transition.next_observation.state.shape[0]
    next_actions = next_actions_flat.reshape(batch_size, policy.action_horizon, policy.action_dim)

    target_features = target_network.compute_features(transition.next_observation, next_actions)
    target_value = target_head(target_features)

    not_done = 1.0 - transition.termination.astype(jnp.float32)
    target = transition.reward + discount * not_done * (target_value - temperature.value * next_log_prob)
    target = jax.lax.stop_gradient(target)

    features = network.compute_features(transition.observation, transition.action)
    loss = _compute_value_loss(head, features, target)

    pred = head(features)
    td_error = pred - target

    info = {
        "q_mean": jnp.mean(pred),
        "q_std": jnp.std(pred),
        "target_mean": jnp.mean(target),
        "target_std": jnp.std(target),
        "td_error_mean": jnp.mean(td_error),
        "td_error_std": jnp.std(td_error),
        "temperature": temperature.value,
        "next_entropy": jnp.mean(-next_log_prob),
        "next_log_prob_mean": jnp.mean(next_log_prob),
    }
    return loss, info
