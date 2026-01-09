"""Unified value function training objectives.

All objectives are pure functions that work with both single-transition (Transition)
and multi-transition (MultiTransition) inputs. The operations are element-wise and
handle arbitrary batch dimensions [batch] or [batch, n].
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.policy_extraction.temperature import Temperature
from openpi.shared import array_typing as at
from openpi.value_functions import hl_gauss as _hl_gauss
from openpi.value_functions.base import MultiTransition
from openpi.value_functions.base import Transition
from openpi.value_functions.heads import CategoricalHead
from openpi.value_functions.heads import EnsembleHead
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base import BaseValueNetwork


def _compute_value_loss(head: ValueHead, features: at.Array, target: at.Array) -> at.Array:
    """Unified loss computation for any head type.

    Works for both single-transition [batch, feature_dim] and
    multi-transition [batch, n, feature_dim] inputs.
    Also supports EnsembleHead with features [ensemble, batch, feature_dim].

    Args:
        head: Value head (RegressionHead, CategoricalHead, or EnsembleHead).
        features: Network features.
        target: Target values.

    Returns:
        Per-sample loss.
    """
    if isinstance(head, EnsembleHead):
        # For ensemble, compute loss for each member and mean over ensemble

        @nnx.vmap(in_axes=(0, 0, None), out_axes=0)
        def compute_single_loss(h: ValueHead, feats: at.Array, tgt: at.Array) -> at.Array:
            if isinstance(h, CategoricalHead):
                logits = h.compute_logits(feats)
                return _hl_gauss.hl_gauss_loss(logits, tgt, h.v_min, h.v_max, h.sigma)
            return jnp.square(h(feats) - tgt)

        all_losses = compute_single_loss(head.vectorized_head, features, target)
        return jnp.mean(all_losses, axis=0)

    if isinstance(head, CategoricalHead):
        logits = head.compute_logits(features)
        return _hl_gauss.hl_gauss_loss(logits, target, head.v_min, head.v_max, head.sigma)
    return jnp.square(head(features) - target)


def mc_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition | MultiTransition,
) -> tuple[at.Array, dict[str, at.Array]]:
    """Monte-Carlo regression: target = mc_return.

    Works for both Transition and MultiTransition inputs.

    Args:
        network: Value network for feature extraction.
        head: Value head for value prediction.
        transition: Transition or MultiTransition.

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
    transition: Transition | MultiTransition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    *,
    discount: float = 0.99,
) -> tuple[at.Array, dict[str, at.Array]]:
    """SARSA: target = r + gamma * Q(s', a').

    Works for both Transition and MultiTransition inputs.

    Args:
        network: Online value network.
        head: Online value head.
        transition: Transition or MultiTransition.
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
    q_network: BaseValueNetwork,
    q_head: ValueHead,
    target_q_network: BaseValueNetwork,
    target_q_head: ValueHead,
    v_network: BaseValueNetwork,
    v_head: ValueHead,
    target_v_network: BaseValueNetwork,
    target_v_head: ValueHead,
    transition: Transition | MultiTransition,
    *,
    expectile: float = 0.7,
    discount: float = 0.99,
) -> tuple[at.Array, at.Array, dict[str, at.Array]]:
    """Correct IQL objective with separate Q and V networks.

    Q(s, a) is trained with Bellman backup using V(s') as the target.
    V(s) is trained with expectile regression to min(target_Q(s, a)).

    Supports Q ensemble via EnsembleNetwork + EnsembleHead.
    When Q is ensemble, V target uses min(target_Q) over ensemble members.

    Works for both Transition and MultiTransition inputs.

    Args:
        q_network: Q-network (can be EnsembleNetwork).
        q_head: Q-function head (can be EnsembleHead).
        target_q_network: Target Q-network for V loss.
        target_q_head: Target Q-function head.
        v_network: V-network (state-only, not action-conditioned).
        v_head: V-function head.
        target_v_network: Target V-network for stable Q targets.
        target_v_head: Target V-function head.
        transition: Transition or MultiTransition.
        expectile: Asymmetry parameter (tau). Values > 0.5 weight positive errors more.
        discount: Discount factor gamma.

    Returns:
        Tuple of (q_loss, v_loss, info_dict).
    """
    # Q(s, a) - can be ensemble
    q_features = q_network.compute_features(transition.observation, transition.action)
    q_values = q_head(q_features)

    # For ensemble Q, q_values has shape [ensemble, batch] or [ensemble, batch, n]
    # We need scalar Q for logging - use mean over ensemble
    q_values_for_logging = jnp.mean(q_values, axis=0) if isinstance(q_head, EnsembleHead) else q_values

    # Target Q(s, a) for V loss - use min over ensemble for pessimism
    target_q_features = target_q_network.compute_features(transition.observation, transition.action)
    if isinstance(target_q_head, EnsembleHead):
        target_q_values = target_q_head.compute_min(target_q_features)
    else:
        target_q_values = target_q_head(target_q_features)
    target_q_values = jax.lax.stop_gradient(target_q_values)

    # V(s)
    v_features = v_network.compute_features(transition.observation, None)
    v_values = v_head(v_features)

    # Target V(s') for Bellman backup
    target_v_features = target_v_network.compute_features(transition.next_observation, None)
    target_v_values = target_v_head(target_v_features)
    target_v_values = jax.lax.stop_gradient(target_v_values)

    # Q loss: Bellman backup with V(s') target
    not_done = 1.0 - transition.termination.astype(jnp.float32)
    q_target = transition.reward + discount * not_done * target_v_values
    q_target = jax.lax.stop_gradient(q_target)

    # Q loss: _compute_value_loss handles both regular and ensemble heads
    q_loss = _compute_value_loss(q_head, q_features, q_target)

    # V loss: Expectile regression to min(target_Q(s, a))
    diff = target_q_values - v_values
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    v_loss = weight * jnp.square(diff)

    info = {
        "q_mean": jnp.mean(q_values_for_logging),
        "q_std": jnp.std(q_values_for_logging),
        "v_mean": jnp.mean(v_values),
        "v_std": jnp.std(v_values),
        "q_target_mean": jnp.mean(q_target),
        "q_target_std": jnp.std(q_target),
        "target_v_mean": jnp.mean(target_v_values),
        "target_q_mean": jnp.mean(target_q_values),
        "q_loss_mean": jnp.mean(q_loss),
        "v_loss_mean": jnp.mean(v_loss),
        "advantage_mean": jnp.mean(target_q_values - v_values),
        "positive_advantage_frac": jnp.mean((diff > 0).astype(jnp.float32)),
    }
    return q_loss, v_loss, info


def sac_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: Transition | MultiTransition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    policy: _model.BaseModel,
    temperature: Temperature,
    *,
    discount: float = 0.99,
    rng: at.KeyArrayLike,
) -> tuple[at.Array, dict[str, at.Array]]:
    """SAC soft Bellman: target = r + gamma * (Q(s', a') - alpha * log pi(a'|s')).

    Works for both Transition and MultiTransition inputs.

    Args:
        network: Online Q-network.
        head: Online Q-head.
        transition: Transition or MultiTransition.
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

    # Handle both single-transition and multi-transition cases
    state = transition.next_observation.state
    is_multi = state.ndim == 3  # [batch, n, state_dim]

    if is_multi:
        batch_size = state.shape[0]
        num_transitions = state.shape[1]

        # Flatten batch and time dimensions: [b, t, ...] -> [b*t, ...]
        flat_next_obs = jax.tree_map(
            lambda x: x.reshape((batch_size * num_transitions,) + x.shape[2:]) if x is not None else None,
            transition.next_observation,
        )
    else:
        batch_size = state.shape[0]
        flat_next_obs = transition.next_observation

    next_dist = policy.action_distribution(sample_rng, flat_next_obs)
    next_actions_flat = next_dist.sample(seed=sample_rng)
    next_log_prob = next_dist.log_prob(next_actions_flat)

    if is_multi:
        action_horizon = next_actions_flat.shape[1]
        action_dim = next_actions_flat.shape[2]
        next_actions = next_actions_flat.reshape(batch_size, num_transitions, action_horizon, action_dim)
        next_log_prob = next_log_prob.reshape(batch_size, num_transitions)
        target_next_obs = transition.next_observation
    else:
        next_actions = next_actions_flat.reshape(batch_size, policy.action_horizon, policy.action_dim)
        target_next_obs = transition.next_observation

    target_features = target_network.compute_features(target_next_obs, next_actions)
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
