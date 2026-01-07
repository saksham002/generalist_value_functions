"""Multi-transition objective functions for batch value learning.

These objectives apply the standard value function objectives element-wise
to multiple transitions, enabling joint value prediction for sequences.
"""

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.policy_extraction.temperature import Temperature
from openpi.shared import array_typing as at
from openpi.value_functions import hl_gauss as _hl_gauss
from openpi.value_functions.base import MultiTransition
from openpi.value_functions.heads import CategoricalHead
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base import BaseValueNetwork


def _compute_value_loss(head: ValueHead, features: at.Array, target: at.Array) -> at.Array:
    """Unified loss computation for any head type.

    Works for both single-transition [batch, feature_dim] and
    multi-transition [batch, n, feature_dim] inputs.
    """
    if isinstance(head, CategoricalHead):
        logits = head.compute_logits(features)
        return _hl_gauss.hl_gauss_loss(logits, target, head.v_min, head.v_max, head.sigma)
    return jnp.square(head(features) - target)


def _assert_multi_transition_shapes(transition: MultiTransition) -> None:
    """Assert that MultiTransition has correct shapes [batch, n, ...]."""
    state = transition.observation.state
    assert state.ndim == 3, f"Expected state shape [batch, n, state_dim], got {state.shape}"

    batch_size = state.shape[0]
    num_transitions = state.shape[1]

    assert transition.action.shape[:2] == (batch_size, num_transitions), (
        f"action shape {transition.action.shape} doesn't match state shape {state.shape}"
    )
    assert transition.reward.shape == (batch_size, num_transitions), (
        f"reward shape {transition.reward.shape} doesn't match (batch={batch_size}, n={num_transitions})"
    )
    assert transition.mc_return.shape == (batch_size, num_transitions), (
        f"mc_return shape {transition.mc_return.shape} doesn't match (batch={batch_size}, n={num_transitions})"
    )
    assert transition.termination.shape == (batch_size, num_transitions), (
        f"termination shape {transition.termination.shape} doesn't match (batch={batch_size}, n={num_transitions})"
    )


def mc_multi_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: MultiTransition,
) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
    """Monte-Carlo objective applied element-wise to n transitions.

    Args:
        network: Value network for feature extraction.
        head: Value head for value prediction.
        transition: MultiTransition with shape [batch, n, ...].

    Returns:
        Tuple of (per_sample_loss [batch, n], info_dict).
    """
    _assert_multi_transition_shapes(transition)

    action = transition.action if network.action_conditioned else None
    features = network.compute_features(transition.observation, action)
    loss = _compute_value_loss(head, features, transition.mc_return)

    batch_size = transition.observation.state.shape[0]
    num_transitions = transition.observation.state.shape[1]
    assert loss.shape == (batch_size, num_transitions), (
        f"Expected loss shape ({batch_size}, {num_transitions}), got {loss.shape}"
    )

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


def sarsa_multi_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: MultiTransition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    *,
    discount: float = 0.99,
) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
    """SARSA objective applied element-wise to n transitions.

    Args:
        network: Online value network.
        head: Online value head.
        transition: MultiTransition with shape [batch, n, ...].
        target_network: Target network for stable target computation.
        target_head: Target head.
        discount: Discount factor gamma.

    Returns:
        Tuple of (per_sample_loss [batch, n], info_dict).
    """
    _assert_multi_transition_shapes(transition)

    target_features = target_network.compute_features(transition.next_observation, transition.next_action)
    target_value = target_head(target_features)

    not_done = 1.0 - transition.termination.astype(jnp.float32)
    target = transition.reward + discount * not_done * target_value
    target = jax.lax.stop_gradient(target)

    action = transition.action if network.action_conditioned else None
    features = network.compute_features(transition.observation, action)
    loss = _compute_value_loss(head, features, target)

    batch_size = transition.observation.state.shape[0]
    num_transitions = transition.observation.state.shape[1]
    assert loss.shape == (batch_size, num_transitions), (
        f"Expected loss shape ({batch_size}, {num_transitions}), got {loss.shape}"
    )

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


def iql_multi_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: MultiTransition,
    *,
    expectile: float = 0.7,
) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
    """IQL expectile regression applied element-wise to n transitions.

    Args:
        network: Value network.
        head: Value head.
        transition: MultiTransition with shape [batch, n, ...].
        expectile: Asymmetry parameter (tau).

    Returns:
        Tuple of (per_sample_loss [batch, n], info_dict).
    """
    _assert_multi_transition_shapes(transition)

    action = transition.action if network.action_conditioned else None
    features = network.compute_features(transition.observation, action)
    pred = head(features)

    target = transition.mc_return
    diff = target - pred
    weight = jnp.where(diff > 0, expectile, 1 - expectile)
    loss = weight * jnp.square(diff)

    batch_size = transition.observation.state.shape[0]
    num_transitions = transition.observation.state.shape[1]
    assert loss.shape == (batch_size, num_transitions), (
        f"Expected loss shape ({batch_size}, {num_transitions}), got {loss.shape}"
    )

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


def sac_multi_objective(
    network: BaseValueNetwork,
    head: ValueHead,
    transition: MultiTransition,
    target_network: BaseValueNetwork,
    target_head: ValueHead,
    policy: _model.BaseModel,
    temperature: Temperature,
    *,
    discount: float = 0.99,
    rng: at.KeyArrayLike,
) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
    """SAC soft Bellman objective applied element-wise to n transitions.

    Args:
        network: Online Q-network.
        head: Online Q-head.
        transition: MultiTransition with shape [batch, n, ...].
        target_network: Target Q-network.
        target_head: Target Q-head.
        policy: Policy network for sampling next actions.
        temperature: Temperature (alpha) for entropy regularization.
        discount: Discount factor gamma.
        rng: Random key for sampling next actions.

    Returns:
        Tuple of (per_sample_loss [batch, n], info_dict).
    """
    _assert_multi_transition_shapes(transition)

    rng, sample_rng = jax.random.split(rng)

    batch_size = transition.next_observation.state.shape[0]
    num_transitions = transition.next_observation.state.shape[1]
    state_dim = transition.next_observation.state.shape[2]

    flat_state = transition.next_observation.state.reshape(batch_size * num_transitions, state_dim)
    flat_next_obs = _model.Observation(
        images={},
        image_masks={},
        state=flat_state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )

    next_dist = policy.action_distribution(sample_rng, flat_next_obs)
    next_actions_flat = next_dist.sample(seed=sample_rng)
    next_log_prob = next_dist.log_prob(next_actions_flat)

    action_horizon = next_actions_flat.shape[1]
    action_dim = next_actions_flat.shape[2]
    next_actions = next_actions_flat.reshape(batch_size, num_transitions, action_horizon, action_dim)
    next_log_prob = next_log_prob.reshape(batch_size, num_transitions)

    target_next_obs = _model.Observation(
        images={},
        image_masks={},
        state=transition.next_observation.state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )

    target_features = target_network.compute_features(target_next_obs, next_actions)
    target_value = target_head(target_features)

    not_done = 1.0 - transition.termination.astype(jnp.float32)
    target = transition.reward + discount * not_done * (target_value - temperature.value * next_log_prob)
    target = jax.lax.stop_gradient(target)

    features = network.compute_features(transition.observation, transition.action)
    loss = _compute_value_loss(head, features, target)

    assert loss.shape == (batch_size, num_transitions), (
        f"Expected loss shape ({batch_size}, {num_transitions}), got {loss.shape}"
    )

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
