"""Tests for policy extraction objectives.

This module tests the functional objectives in objectives.py using fake critics
and policies with known, deterministic outputs to verify correctness.
"""

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.tanh_gaussian import TanhGaussianConfig
from openpi.policy_extraction import objectives
from openpi.shared import array_typing as at
from openpi.value_functions.base import BaseValueFunction


def make_observation(state: jnp.ndarray) -> _model.Observation:
    """Create a minimal Observation from a state tensor."""
    return _model.Observation(
        images={},
        image_masks={},
        state=state,
    )


class FakeQFunction(BaseValueFunction):
    """Fake Q-function that returns a known, deterministic value.

    Q(s, a) = sum(s) + sum(a) * action_weight

    This allows testing that objectives correctly use the Q-function output.
    """

    def __init__(self, action_weight: float = 1.0):
        self.action_weight = action_weight

    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Return sum(state) + sum(action) * weight."""
        state_sum = jnp.sum(observation.state, axis=-1)
        if action is not None:
            action_sum = jnp.sum(action, axis=(-2, -1))
            return state_sum + self.action_weight * action_sum
        return state_sum

    def compute_loss(self, transition, *, train=False, rng=None):
        raise NotImplementedError("FakeQFunction doesn't support training")


class FakeEnsembleQFunction(FakeQFunction):
    """Fake ensemble Q-function with min/mean aggregation methods."""

    def __init__(self, action_weight: float = 1.0, ensemble_spread: float = 1.0):
        super().__init__(action_weight)
        self.ensemble_spread = ensemble_spread

    def compute_min_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Return base value - spread (simulating min over ensemble)."""
        base = self.compute_value(observation, action)
        return base - self.ensemble_spread

    def compute_mean_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Return base value (simulating mean over ensemble)."""
        return self.compute_value(observation, action)


class FakePolicy(_model.BaseModel):
    """Fake policy that returns deterministic actions.

    Actions are computed as: state * action_scale (broadcast to action shape).
    This allows controlled, predictable outputs for testing.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        action_scale: float = 0.0,
    ):
        super().__init__(action_dim, action_horizon, max_token_len=0)
        self._action_scale = action_scale

    def action_distribution(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> distrax.Distribution:
        """Return deterministic distribution centered at scaled state mean."""
        batch_size = observation.state.shape[0]
        # Action = state_mean * scale, broadcast to action shape
        state_mean = jnp.mean(observation.state, axis=-1, keepdims=True)
        action_value = state_mean * self._action_scale

        # Broadcast to [batch, action_horizon * action_dim]
        flat_size = self.action_horizon * self.action_dim
        actions_flat = jnp.broadcast_to(action_value, (batch_size, flat_size))

        return distrax.Deterministic(loc=actions_flat)

    def sample_actions(self, rng, observation, **kwargs):
        dist = self.action_distribution(rng, observation)
        batch_size = observation.state.shape[0]
        return dist.sample(seed=rng).reshape(batch_size, self.action_horizon, self.action_dim)

    def compute_loss(self, rng, observation, actions, *, train=False):
        raise NotImplementedError("FakePolicy doesn't support training")


class TestDDPGObjective:
    """Tests for ddpg_objective."""

    def test_basic_loss_shape(self):
        """Test that DDPG objective returns correct shape."""
        batch_size = 4
        state_dim = 10
        action_dim = 4
        action_horizon = 1

        policy = FakePolicy(action_dim, action_horizon)
        q_fn = FakeQFunction()
        obs = make_observation(jnp.ones((batch_size, state_dim)))
        rng = jax.random.key(0)

        loss, info = objectives.ddpg_objective(policy, obs, rng, q_fn)

        assert loss.shape == (batch_size,)
        assert "q_value_mean" in info
        assert "q_value_std" in info

    def test_loss_equals_negative_q(self):
        """Test that loss = -Q(s, a)."""
        action_dim = 3
        action_horizon = 1

        # Policy outputs zeros (action_scale=0), so Q = sum(state)
        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        q_fn = FakeQFunction(action_weight=1.0)

        # States with known sums
        state = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
        obs = make_observation(state)
        rng = jax.random.key(42)

        loss, info = objectives.ddpg_objective(policy, obs, rng, q_fn)

        # Q = sum(state) + 0 (since actions are zeros)
        expected_q = jnp.array([15.0, 2.5])
        expected_loss = -expected_q

        np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)
        np.testing.assert_allclose(info["q_value_mean"], jnp.mean(expected_q), rtol=1e-5)

    def test_min_aggregation_uses_min_value(self):
        """Test that aggregation='min' uses compute_min_value."""
        action_dim = 3
        action_horizon = 1

        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        q_fn = FakeEnsembleQFunction(action_weight=1.0, ensemble_spread=2.0)

        state = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
        obs = make_observation(state)
        rng = jax.random.key(42)

        loss, _ = objectives.ddpg_objective(policy, obs, rng, q_fn, aggregation="min")

        # Q_min = sum(state) - spread = sum(state) - 2.0
        expected_q = jnp.array([15.0 - 2.0, 2.5 - 2.0])
        expected_loss = -expected_q

        np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)

    def test_mean_aggregation_uses_mean_value(self):
        """Test that aggregation='mean' uses compute_mean_value."""
        action_dim = 3
        action_horizon = 1

        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        q_fn = FakeEnsembleQFunction(action_weight=1.0, ensemble_spread=2.0)

        state = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
        obs = make_observation(state)
        rng = jax.random.key(42)

        loss, _ = objectives.ddpg_objective(policy, obs, rng, q_fn, aggregation="mean")

        # Q_mean = sum(state) (no spread subtraction)
        expected_q = jnp.array([15.0, 2.5])
        expected_loss = -expected_q

        np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


class TestBCRegularizationObjective:
    """Tests for bc_regularization_objective."""

    def test_basic_loss_shape(self):
        """Test that BC regularization returns correct shape."""
        batch_size = 4
        state_dim = 10
        action_dim = 4
        action_horizon = 2

        policy = FakePolicy(action_dim, action_horizon)
        obs = make_observation(jnp.ones((batch_size, state_dim)))
        data_action = jnp.ones((batch_size, action_horizon, action_dim))
        rng = jax.random.key(0)

        loss, info = objectives.bc_regularization_objective(policy, obs, rng, data_action)

        assert loss.shape == (batch_size,)
        assert "bc_mse" in info
        assert "action_diff_mean" in info

    def test_zero_loss_when_matching(self):
        """Test that loss is zero when policy matches data action."""
        batch_size = 2
        state_dim = 5
        action_dim = 3
        action_horizon = 1

        # Policy outputs zeros, data actions are also zeros
        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        obs = make_observation(jnp.ones((batch_size, state_dim)))
        data_action = jnp.zeros((batch_size, action_horizon, action_dim))
        rng = jax.random.key(0)

        loss, info = objectives.bc_regularization_objective(policy, obs, rng, data_action)

        np.testing.assert_allclose(loss, jnp.zeros(batch_size), atol=1e-6)
        np.testing.assert_allclose(info["bc_mse"], 0.0, atol=1e-6)

    def test_mse_computation(self):
        """Test MSE is correctly computed."""
        batch_size = 2
        state_dim = 5
        action_dim = 2
        action_horizon = 1

        # Policy outputs zeros
        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        obs = make_observation(jnp.ones((batch_size, state_dim)))

        # Data actions: [[1, 2], [3, 4]]
        data_action = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        rng = jax.random.key(0)

        loss, info = objectives.bc_regularization_objective(policy, obs, rng, data_action)

        # MSE for sample 0: (1^2 + 2^2) / 2 = 2.5
        # MSE for sample 1: (3^2 + 4^2) / 2 = 12.5
        expected_loss = jnp.array([2.5, 12.5])

        np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


class TestEntropyObjective:
    """Tests for entropy_objective."""

    def test_basic_loss_shape(self):
        """Test that entropy objective returns correct shape."""
        batch_size = 4
        state_dim = 10
        action_dim = 4
        action_horizon = 1

        config = TanhGaussianConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            std_parameterization="fixed",
            fixed_std=0.1,
        )
        rng = jax.random.key(0)
        policy = config.create(rng)

        obs = make_observation(jnp.ones((batch_size, state_dim)))
        rng = jax.random.key(1)

        loss, info = objectives.entropy_objective(policy, obs, rng)

        assert loss.shape == (batch_size,)
        assert "entropy" in info
        assert "log_prob_mean" in info

    def test_entropy_info_is_negative_log_prob(self):
        """Test that info['entropy'] = -log_prob."""
        batch_size = 2
        state_dim = 5
        action_dim = 2
        action_horizon = 1

        config = TanhGaussianConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            std_parameterization="fixed",
            fixed_std=0.1,
        )
        rng = jax.random.key(0)
        policy = config.create(rng)

        obs = make_observation(jnp.ones((batch_size, state_dim)))
        rng = jax.random.key(1)

        loss, info = objectives.entropy_objective(policy, obs, rng)

        # entropy = mean(-log_prob), loss = log_prob
        np.testing.assert_allclose(info["entropy"], jnp.mean(-loss), rtol=1e-5)


class TestWeightedSumObjective:
    """Tests for weighted_sum_objective."""

    def test_basic_shape(self):
        """Test that weighted sum returns correct shape."""
        batch_size = 4
        state_dim = 10
        action_dim = 4
        action_horizon = 1

        policy = FakePolicy(action_dim, action_horizon)
        q_fn = FakeQFunction()
        obs = make_observation(jnp.ones((batch_size, state_dim)))
        data_action = jnp.ones((batch_size, action_horizon, action_dim))
        rng = jax.random.key(0)

        objectives_and_weights = [
            (objectives.ddpg_objective, 1.0, {"q_function": q_fn}),
            (objectives.bc_regularization_objective, 0.5, {"data_action": data_action}),
        ]

        loss, info = objectives.weighted_sum_objective(policy, obs, rng, objectives_and_weights)

        assert loss.shape == (batch_size,)
        assert "ddpg/q_value_mean" in info
        assert "bc_regularization/bc_mse" in info
        assert "total_loss" in info

    def test_weighted_combination(self):
        """Test that weighted sum correctly combines objectives."""
        batch_size = 2
        action_dim = 3
        action_horizon = 1

        # Policy outputs zeros
        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        q_fn = FakeQFunction(action_weight=1.0)

        state = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]])
        obs = make_observation(state)

        # Data action = 1, policy outputs 0, so MSE = 1.0
        data_action = jnp.ones((batch_size, action_horizon, action_dim))
        rng = jax.random.key(0)

        ddpg_weight = 2.0
        bc_weight = 0.5

        objectives_and_weights = [
            (objectives.ddpg_objective, ddpg_weight, {"q_function": q_fn}),
            (objectives.bc_regularization_objective, bc_weight, {"data_action": data_action}),
        ]

        loss, info = objectives.weighted_sum_objective(policy, obs, rng, objectives_and_weights)

        # DDPG loss = -Q = -sum(state) = [-5, -10]
        # BC loss = MSE = 1.0 for all samples
        expected_ddpg_loss = jnp.array([-5.0, -10.0])
        expected_bc_loss = jnp.ones(batch_size)
        expected_total = ddpg_weight * expected_ddpg_loss + bc_weight * expected_bc_loss

        np.testing.assert_allclose(loss, expected_total, rtol=1e-5)

    def test_single_objective(self):
        """Test that single objective works correctly."""
        action_dim = 3
        action_horizon = 1

        policy = FakePolicy(action_dim, action_horizon, action_scale=0.0)
        q_fn = FakeQFunction()

        state = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
        obs = make_observation(state)
        rng = jax.random.key(0)

        objectives_and_weights = [
            (objectives.ddpg_objective, 1.0, {"q_function": q_fn}),
        ]

        weighted_loss, _ = objectives.weighted_sum_objective(policy, obs, rng, objectives_and_weights)

        # Compare with direct call
        direct_loss, _ = objectives.ddpg_objective(policy, obs, rng, q_fn)

        np.testing.assert_allclose(weighted_loss, direct_loss, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
