"""Tests for new value function architecture."""

import jax
import jax.numpy as jnp
import pytest

from openpi.models.model import Observation
from openpi.value_functions.base import Transition
from openpi.value_functions.heads import CategoricalHeadConfig
from openpi.value_functions.heads import RegressionHeadConfig
from openpi.value_functions.networks.mlp import MLPNetworkConfig
from openpi.value_functions.value_function import IQLValueFunctionConfig
from openpi.value_functions.value_function import MCValueFunctionConfig
from openpi.value_functions.value_function import SARSAValueFunctionConfig


def make_observation(state: jnp.ndarray) -> Observation:
    return Observation(images={}, image_masks={}, state=state)


def make_transition(batch_size: int, state_dim: int, action_dim: int = 4) -> Transition:
    observation = make_observation(jnp.ones((batch_size, state_dim)))
    next_observation = make_observation(jnp.ones((batch_size, state_dim)))
    return Transition(
        observation=observation,
        action=jnp.ones((batch_size, action_dim)),
        reward=jnp.zeros(batch_size),
        next_observation=next_observation,
        next_action=jnp.ones((batch_size, action_dim)),
        mc_return=jnp.ones(batch_size) * 0.5,
        termination=jnp.zeros(batch_size, dtype=bool),
        truncation=jnp.zeros(batch_size, dtype=bool),
    )


class TestMLPNetwork:
    def test_create_and_forward(self):
        config = MLPNetworkConfig(state_dim=10, hidden_dims=(64, 32))
        network = config.create(jax.random.key(0))

        obs = make_observation(jnp.ones((4, 10)))
        features = network.compute_features(obs)

        assert features.shape == (4, 32)
        assert network.feature_dim == 32

    def test_action_conditioned(self):
        config = MLPNetworkConfig(state_dim=10, action_conditioned=True, action_dim=4, action_horizon=1)
        network = config.create(jax.random.key(0))

        obs = make_observation(jnp.ones((4, 10)))
        action = jnp.ones((4, 1, 4))
        features = network.compute_features(obs, action)

        assert features.shape == (4, 256)


class TestHeads:
    def test_regression_head(self):
        config = RegressionHeadConfig()
        head = config.create(64, jax.random.key(0))

        features = jnp.ones((4, 64))
        values = head(features)

        assert values.shape == (4,)

    def test_categorical_head(self):
        config = CategoricalHeadConfig(v_min=-10.0, v_max=10.0, num_bins=51)
        head = config.create(64, jax.random.key(0))

        features = jnp.ones((4, 64))
        values = head(features)
        logits = head.compute_logits(features)

        assert values.shape == (4,)
        assert logits.shape == (4, 51)


class TestMCValueFunction:
    def test_regression_head(self):
        config = MCValueFunctionConfig(
            network_config=MLPNetworkConfig(state_dim=10, hidden_dims=(32,)),
            head_config=RegressionHeadConfig(),
        )
        model = config.create(jax.random.key(0))

        obs = make_observation(jnp.ones((4, 10)))
        values = model.compute_value(obs)
        assert values.shape == (4,)

        transition = make_transition(4, 10)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)
        assert "predicted_value_mean" in info

    def test_categorical_head(self):
        config = MCValueFunctionConfig(
            network_config=MLPNetworkConfig(state_dim=10, hidden_dims=(32,)),
            head_config=CategoricalHeadConfig(v_min=0.0, v_max=1.0, num_bins=51),
        )
        model = config.create(jax.random.key(0))

        obs = make_observation(jnp.ones((4, 10)))
        values = model.compute_value(obs)
        assert values.shape == (4,)

        transition = make_transition(4, 10)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (4,)


class TestSARSAValueFunction:
    def test_with_target_network(self):
        config = SARSAValueFunctionConfig(
            network_config=MLPNetworkConfig(state_dim=10, action_conditioned=True, action_dim=4, hidden_dims=(32,)),
            head_config=RegressionHeadConfig(),
            discount=0.99,
            tau=0.005,
        )
        model = config.create(jax.random.key(0))

        transition = make_transition(4, 10, action_dim=4)
        loss, info = model.compute_loss(transition)

        assert loss.shape == (4,)
        assert "next_value_mean" in info

        # Test target network update
        model.post_step_update()


class TestIQLValueFunction:
    def test_expectile_regression(self):
        config = IQLValueFunctionConfig(
            network_config=MLPNetworkConfig(state_dim=10, hidden_dims=(32,)),
            head_config=RegressionHeadConfig(),
            expectile=0.7,
        )
        model = config.create(jax.random.key(0))

        transition = make_transition(4, 10)
        loss, info = model.compute_loss(transition)

        assert loss.shape == (4,)
        assert "positive_error_frac" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
