"""SAC (Soft Actor-Critic) value function training.

This module provides the SAC critic (Q-function ensemble) with soft Bellman
backup. It supports both regression (MSE) and categorical (HL-Gauss) losses.

The SAC critic maintains:
- An ensemble of Q-functions (typically 2) for the online network
- A target Q-function ensemble for stable target computation
- A policy network for sampling next actions
- A temperature parameter for entropy regularization
"""

import dataclasses
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.tanh_gaussian import TanhGaussianConfig
from openpi.policy_extraction.temperature import Temperature
from openpi.policy_extraction.temperature import TemperatureConfig
from openpi.shared import array_typing as at
from openpi.value_functions import hl_gauss as _hl_gauss
from openpi.value_functions.base import BaseValueFunction
from openpi.value_functions.base import BaseValueFunctionConfig
from openpi.value_functions.base import Transition
from openpi.value_functions.ensemble import EnsembleValueFunction
from openpi.value_functions.ensemble import EnsembleValueFunctionConfig


@dataclasses.dataclass(frozen=True)
class SACValueFunctionConfig(BaseValueFunctionConfig):
    """Configuration for SAC-style Q-function training.

    Supports both regression (MSE) and categorical (HL-Gauss) loss types.
    The loss_type should match the base Q-function config in q_ensemble_config.
    """

    # Q-function ensemble configuration
    q_ensemble_config: EnsembleValueFunctionConfig

    # Policy configuration (for sampling next actions in target computation)
    policy_config: TanhGaussianConfig

    # Discount factor
    discount: float = 0.99

    # Target network update rate (Polyak averaging)
    # target = tau * online + (1 - tau) * target
    tau: float = 0.005

    # Temperature configuration
    temperature_config: TemperatureConfig = dataclasses.field(
        default_factory=TemperatureConfig
    )

    # Loss type: "regression" (MSE) or "categorical" (HL-Gauss)
    loss_type: Literal["regression", "categorical"] = "regression"

    # HL-Gauss parameters (only used if loss_type == "categorical")
    v_min: float = -100.0
    v_max: float = 100.0
    num_bins: int = 51
    sigma: float = 0.75

    @override
    def create(self, rng: at.KeyArrayLike) -> "SACValueFunction":
        """Create SAC value function with Q-ensemble, target, policy, and temperature."""
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        rng, q_rng, target_rng, policy_rng = jax.random.split(rng, 4)

        q_ensemble = self.q_ensemble_config.create(q_rng)
        target_q_ensemble = self.q_ensemble_config.create(target_rng)
        policy = self.policy_config.create(policy_rng)
        temperature = self.temperature_config.create(self.policy_config.action_dim)

        return SACValueFunction(
            q_ensemble=q_ensemble,
            target_q_ensemble=target_q_ensemble,
            policy=policy,
            temperature=temperature,
            discount=self.discount,
            tau=self.tau,
            loss_type=self.loss_type,
            v_min=self.v_min,
            v_max=self.v_max,
            num_bins=self.num_bins,
            sigma=self.sigma,
        )

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        return self.q_ensemble_config.inputs_spec(batch_size=batch_size)


class SACValueFunction(BaseValueFunction):
    """SAC Q-function ensemble with soft Bellman backup.

    This implements the SAC critic training objective:
    1. Sample next actions from policy: a' ~ π(·|s')
    2. Compute soft target: y = r + γ * (1 - done) * (min Q_target(s', a') - α * log π(a'|s'))
    3. Update Q-functions to minimize loss between Q(s, a) and y

    Supports both regression (MSE) and categorical (HL-Gauss) losses.
    """

    q_ensemble: EnsembleValueFunction
    target_q_ensemble: EnsembleValueFunction
    policy: _model.BaseModel  # TanhGaussian
    temperature: Temperature
    discount: float
    tau: float
    loss_type: Literal["regression", "categorical"]
    v_min: float
    v_max: float
    num_bins: int
    sigma: float

    def __init__(
        self,
        q_ensemble: EnsembleValueFunction,
        target_q_ensemble: EnsembleValueFunction,
        policy: _model.BaseModel,
        temperature: Temperature,
        discount: float,
        tau: float,
        loss_type: Literal["regression", "categorical"],
        v_min: float,
        v_max: float,
        num_bins: int,
        sigma: float,
    ):
        super().__init__()
        self.q_ensemble = q_ensemble
        self.target_q_ensemble = target_q_ensemble
        self.policy = policy
        self.temperature = temperature
        self.discount = discount
        self.tau = tau
        self.loss_type = loss_type
        self.v_min = v_min
        self.v_max = v_max
        self.num_bins = num_bins
        self.sigma = sigma

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "batch"]:
        """Compute min Q-value across ensemble (pessimistic estimate)."""
        return self.q_ensemble.compute_min_value(observation, action)

    def _compute_critic_loss_regression(
        self,
        q_values: at.Float[at.Array, "num_ensemble batch"],
        target: at.Float[at.Array, "batch"],
    ) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
        """Compute MSE loss for regression Q-functions.

        Args:
            q_values: Q-values from all ensemble members, shape (num_ensemble, batch).
            target: Target values, shape (batch,).

        Returns:
            Tuple of (mean_loss, info_dict).
        """
        td_errors = q_values - target[None, :]  # (num_ensemble, batch)
        all_losses = jnp.square(td_errors)
        mean_loss = jnp.mean(all_losses, axis=0)

        info = {
            "td_error_mean": jnp.mean(td_errors),
            "td_error_std": jnp.std(td_errors),
        }
        return mean_loss, info

    def _compute_critic_loss_categorical(
        self,
        observation: _model.Observation,
        action: _model.Actions,
        target: at.Float[at.Array, "batch"],
    ) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
        """Compute HL-Gauss loss for categorical Q-functions.

        Args:
            observation: Observation containing state.
            action: Actions.
            target: Target values, shape (batch,).

        Returns:
            Tuple of (mean_loss, info_dict).
        """
        # Get logits from ensemble: (num_ensemble, batch, num_bins)
        all_logits = self.q_ensemble.compute_logits(observation, action)

        # Compute HL-Gauss loss for each ensemble member
        def single_member_loss(logits: at.Float[at.Array, "batch num_bins"]) -> at.Float[at.Array, "batch"]:
            return _hl_gauss.hl_gauss_loss(logits, target, self.v_min, self.v_max, self.sigma)

        all_losses = jax.vmap(single_member_loss)(all_logits)  # (num_ensemble, batch)
        mean_loss = jnp.mean(all_losses, axis=0)

        # Compute predicted values for logging
        def logits_to_value(logits: at.Float[at.Array, "batch num_bins"]) -> at.Float[at.Array, "batch"]:
            return _hl_gauss.logits_to_expected_value(logits, self.v_min, self.v_max)

        all_values = jax.vmap(logits_to_value)(all_logits)  # (num_ensemble, batch)
        td_errors = all_values - target[None, :]

        info = {
            "td_error_mean": jnp.mean(td_errors),
            "td_error_std": jnp.std(td_errors),
            "hl_gauss_loss_mean": jnp.mean(all_losses),
        }
        return mean_loss, info

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
        """Compute SAC critic loss with soft Bellman backup.

        The soft Bellman target is:
            y = r + γ * (1 - done) * (min Q_target(s', a') - α * log π(a'|s'))

        where a' ~ π(·|s') is sampled from the current policy.

        Args:
            transition: SARSA transition with (s, a, r, s', a', mc_return, termination, truncation).
            train: Whether in training mode.
            rng: Random key for sampling next actions from policy.

        Returns:
            Tuple of (per_sample_loss, info_dict).
        """
        if rng is None:
            raise ValueError("SACValueFunction.compute_loss requires rng for sampling next actions")

        rng, sample_rng = jax.random.split(rng)

        # Get next action distribution and sample
        next_dist = self.policy.action_distribution(sample_rng, transition.next_observation)
        next_actions_flat = next_dist.sample(seed=sample_rng)
        next_log_prob = next_dist.log_prob(next_actions_flat)

        # Reshape actions for Q-function
        batch_size = transition.next_observation.state.shape[0]
        next_actions = next_actions_flat.reshape(
            batch_size, self.policy.action_horizon, self.policy.action_dim
        )

        # Compute target Q-values (min over target ensemble)
        target_q = self.target_q_ensemble.compute_min_value(
            transition.next_observation, next_actions
        )

        # Soft Bellman target: r + γ * (1 - done) * (Q_target - α * log π)
        not_done = 1.0 - transition.termination.astype(jnp.float32)
        target = transition.reward + self.discount * not_done * (
            target_q - self.temperature.value * next_log_prob
        )
        target = jax.lax.stop_gradient(target)

        # Compute loss based on loss type
        if self.loss_type == "regression":
            # Get Q-values for current state-action
            all_q_values = self.q_ensemble.compute_value(transition.observation, transition.action)
            loss, loss_info = self._compute_critic_loss_regression(all_q_values, target)
            q_mean = jnp.mean(all_q_values)
            q_std = jnp.std(all_q_values)
        else:  # categorical
            loss, loss_info = self._compute_critic_loss_categorical(
                transition.observation, transition.action, target
            )
            # For logging, compute expected values
            all_q_values = self.q_ensemble.compute_value(transition.observation, transition.action)
            q_mean = jnp.mean(all_q_values)
            q_std = jnp.std(all_q_values)

        info = {
            "q_mean": q_mean,
            "q_std": q_std,
            "target_mean": jnp.mean(target),
            "target_std": jnp.std(target),
            "temperature": self.temperature.value,
            "next_entropy": jnp.mean(-next_log_prob),
            "next_log_prob_mean": jnp.mean(next_log_prob),
            **loss_info,
        }

        return loss, info

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network.

        Updates target network parameters:
            target = tau * online + (1 - tau) * target
        """
        target_state = nnx.state(self.target_q_ensemble)
        online_state = nnx.state(self.q_ensemble)

        new_target_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_state,
            online_state,
        )
        nnx.update(self.target_q_ensemble, new_target_state)

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        deterministic: bool = False,
    ) -> _model.Actions:
        """Sample actions from the policy.

        Args:
            rng: Random key for sampling.
            observation: Observation containing state.
            deterministic: If True, return mean action instead of sampling.

        Returns:
            Actions of shape (batch, action_horizon, action_dim).
        """
        return self.policy.sample_actions(rng, observation, deterministic=deterministic)
