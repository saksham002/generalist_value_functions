"""Tests for value function implementations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.model import Observation
from openpi.value_functions.base import Transition
from openpi.value_functions.hl_gauss import compute_bin_centers
from openpi.value_functions.hl_gauss import compute_hl_gauss_targets
from openpi.value_functions.hl_gauss import hl_gauss_loss
from openpi.value_functions.hl_gauss import logits_to_expected_value
from openpi.value_functions.value_mlp import ValueMLPConfig


def make_observation(state: jnp.ndarray) -> Observation:
    """Create a minimal Observation from a state tensor."""
    return Observation(
        images={},
        image_masks={},
        state=state,
    )


def make_transition(
    batch_size: int,
    state_dim: int,
    action_dim: int = 4,
    mc_return: float = 0.0,
) -> Transition:
    """Create a minimal Transition for testing."""
    observation = Observation(
        images={},
        image_masks={},
        state=jnp.ones((batch_size, state_dim)),
    )
    next_observation = Observation(
        images={},
        image_masks={},
        state=jnp.ones((batch_size, state_dim)),
    )
    return Transition(
        observation=observation,
        action=jnp.ones((batch_size, action_dim)),
        reward=jnp.zeros(batch_size),
        next_observation=next_observation,
        next_action=jnp.ones((batch_size, action_dim)),
        mc_return=jnp.full(batch_size, mc_return),
        termination=jnp.zeros(batch_size, dtype=bool),
        truncation=jnp.zeros(batch_size, dtype=bool),
    )


class TestHLGauss:
    """Tests for HL-Gauss utilities."""

    def test_compute_bin_centers(self):
        """Test bin center computation."""
        centers = compute_bin_centers(-10.0, 10.0, 5)
        expected = jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        np.testing.assert_allclose(centers, expected, rtol=1e-5)

    def test_compute_hl_gauss_targets_shape(self):
        """Test that HL-Gauss targets have correct shape."""
        target_values = jnp.array([0.0, 5.0, -5.0])
        targets = compute_hl_gauss_targets(target_values, -10.0, 10.0, 51, 0.75)
        assert targets.shape == (3, 51)

    def test_compute_hl_gauss_targets_sum_to_one(self):
        """Test that HL-Gauss targets sum to 1."""
        target_values = jnp.array([0.0, 5.0, -5.0])
        targets = compute_hl_gauss_targets(target_values, -10.0, 10.0, 51, 0.75)
        np.testing.assert_allclose(jnp.sum(targets, axis=-1), jnp.ones(3), rtol=1e-5)

    def test_hl_gauss_loss_shape(self):
        """Test that HL-Gauss loss has correct shape."""
        logits = jnp.zeros((4, 51))
        target_values = jnp.array([0.0, 5.0, -5.0, 2.0])
        loss = hl_gauss_loss(logits, target_values, -10.0, 10.0, 0.75)
        assert loss.shape == (4,)

    def test_logits_to_expected_value(self):
        """Test conversion of logits to expected value."""
        # Uniform logits should give midpoint
        logits = jnp.zeros((2, 51))
        expected = logits_to_expected_value(logits, -10.0, 10.0)
        assert expected.shape == (2,)
        # Expected value of uniform over [-10, 10] is 0
        np.testing.assert_allclose(expected, jnp.zeros(2), atol=1e-5)


class TestRegressionValueMLP:
    """Tests for regression value function (V(s))."""

    def test_create_and_compute_value(self):
        """Test creating model and computing values."""
        config = ValueMLPConfig(state_dim=10, hidden_dims=(64, 64))
        rng = jax.random.key(0)
        model = config.create(rng)

        state = jnp.ones((4, 10))
        obs = make_observation(state)
        values = model.compute_value(obs)
        assert values.shape == (4,)

    def test_compute_loss(self):
        """Test loss computation."""
        config = ValueMLPConfig(state_dim=10, hidden_dims=(64, 64))
        rng = jax.random.key(0)
        model = config.create(rng)

        transition = make_transition(batch_size=4, state_dim=10, action_dim=4, mc_return=0.0)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)
        assert jnp.all(loss >= 0)
        assert isinstance(info, dict)
        assert "predicted_value_mean" in info

    def test_inputs_spec(self):
        """Test input specification for V(s)."""
        config = ValueMLPConfig(state_dim=10)
        spec = config.inputs_spec(batch_size=8)
        assert len(spec) == 2  # (obs, target) for V(s)
        obs_spec, target_spec = spec
        assert obs_spec.state.shape == (8, 10)
        assert target_spec.shape == (8,)


class TestRegressionQMLP:
    """Tests for regression Q-function (Q(s,a)) using action_conditioned=True."""

    def test_create_and_compute_value(self):
        """Test creating model and computing Q-values."""
        config = ValueMLPConfig(
            state_dim=10,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(64, 64),
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        state = jnp.ones((4, 10))
        obs = make_observation(state)
        action = jnp.ones((4, 1, 4))  # [batch, action_horizon, action_dim]
        values = model.compute_value(obs, action)
        assert values.shape == (4,)

    def test_compute_loss(self):
        """Test loss computation."""
        config = ValueMLPConfig(
            state_dim=10,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(64, 64),
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        transition = make_transition(batch_size=4, state_dim=10, action_dim=4, mc_return=0.0)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)
        assert jnp.all(loss >= 0)
        assert isinstance(info, dict)

    def test_inputs_spec(self):
        """Test input specification for Q(s,a)."""
        config = ValueMLPConfig(state_dim=10, action_conditioned=True, action_dim=4, action_horizon=1)
        spec = config.inputs_spec(batch_size=8)
        assert len(spec) == 3  # (obs, actions, target) for Q(s,a)
        obs_spec, action_spec, target_spec = spec
        assert obs_spec.state.shape == (8, 10)
        assert action_spec.shape == (8, 1, 4)
        assert target_spec.shape == (8,)


class TestCategoricalValueMLP:
    """Tests for categorical value function (V(s))."""

    def test_create_and_compute_value(self):
        """Test creating model and computing values."""
        config = ValueMLPConfig(
            state_dim=10, hidden_dims=(64, 64), use_hl_gauss=True, v_min=-10.0, v_max=10.0, num_bins=51
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        state = jnp.ones((4, 10))
        obs = make_observation(state)
        values = model.compute_value(obs)
        assert values.shape == (4,)

    def test_compute_logits(self):
        """Test logits computation."""
        config = ValueMLPConfig(
            state_dim=10, hidden_dims=(64, 64), use_hl_gauss=True, v_min=-10.0, v_max=10.0, num_bins=51
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        state = jnp.ones((4, 10))
        obs = make_observation(state)
        logits = model.compute_logits(obs)
        assert logits.shape == (4, 51)

    def test_compute_loss(self):
        """Test loss computation."""
        config = ValueMLPConfig(state_dim=10, hidden_dims=(64, 64), use_hl_gauss=True, v_min=-10.0, v_max=10.0)
        rng = jax.random.key(0)
        model = config.create(rng)

        transition = make_transition(batch_size=4, state_dim=10, action_dim=4, mc_return=0.0)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)
        assert jnp.all(loss >= 0)
        assert isinstance(info, dict)


class TestCategoricalQMLP:
    """Tests for categorical Q-function (Q(s,a)) using action_conditioned=True."""

    def test_create_and_compute_value(self):
        """Test creating model and computing Q-values."""
        config = ValueMLPConfig(
            state_dim=10,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(64, 64),
            use_hl_gauss=True,
            v_min=-10.0,
            v_max=10.0,
            num_bins=51,
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        state = jnp.ones((4, 10))
        obs = make_observation(state)
        action = jnp.ones((4, 1, 4))
        values = model.compute_value(obs, action)
        assert values.shape == (4,)

    def test_compute_logits(self):
        """Test logits computation."""
        config = ValueMLPConfig(
            state_dim=10,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(64, 64),
            use_hl_gauss=True,
            v_min=-10.0,
            v_max=10.0,
            num_bins=51,
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        state = jnp.ones((4, 10))
        obs = make_observation(state)
        action = jnp.ones((4, 1, 4))
        logits = model.compute_logits(obs, action)
        assert logits.shape == (4, 51)

    def test_compute_loss(self):
        """Test loss computation."""
        config = ValueMLPConfig(
            state_dim=10,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(64, 64),
            use_hl_gauss=True,
            v_min=-10.0,
            v_max=10.0,
        )
        rng = jax.random.key(0)
        model = config.create(rng)

        transition = make_transition(batch_size=4, state_dim=10, action_dim=4, mc_return=0.0)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)
        assert jnp.all(loss >= 0)
        assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
