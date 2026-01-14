"""Tests for new value function architecture."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.model import Observation
from openpi.value_functions.base_value_functions import MultiTransition
from openpi.value_functions.base_value_functions import Transition
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
            q_network_config=MLPNetworkConfig(
                state_dim=10, action_conditioned=True, action_dim=4, action_horizon=1, hidden_dims=(32,)
            ),
            v_network_config=MLPNetworkConfig(state_dim=10, action_conditioned=False, hidden_dims=(32,)),
            q_head_config=RegressionHeadConfig(),
            v_head_config=RegressionHeadConfig(),
            expectile=0.7,
            discount=0.99,
            tau=0.005,
        )
        model = config.create(jax.random.key(0))

        transition = make_transition(4, 10, action_dim=4)
        loss, info = model.compute_loss(transition)

        assert loss.shape == (4,)
        assert "positive_advantage_frac" in info
        assert "q_mean" in info
        assert "v_mean" in info

        # Test target network update
        model.post_step_update()


class TestEnsembleNetwork:
    def test_ensemble_has_different_weights(self):
        """Verify each ensemble member has different weights."""

        from openpi.value_functions.networks.ensemble import EnsembleNetworkConfig

        config = EnsembleNetworkConfig(
            base_config=MLPNetworkConfig(
                state_dim=10, action_conditioned=True, action_dim=4, action_horizon=1, hidden_dims=(32,)
            ),
            ensemble_size=3,
        )
        network = config.create(jax.random.key(0))

        # Get all parameters from vectorized network
        state = nnx.state(network.vectorized_network)

        # Check that parameters have ensemble dimension and are different
        for _, param in jax.tree_util.tree_leaves_with_path(state):
            if hasattr(param, "value") and param.value.ndim >= 2:
                # First dimension should be ensemble_size
                assert param.value.shape[0] == 3, f"Expected ensemble dim 3, got {param.value.shape[0]}"
                # Each member should have different weights
                member_0 = param.value[0]
                member_1 = param.value[1]
                member_2 = param.value[2]
                assert not jnp.allclose(member_0, member_1), "Members 0 and 1 should have different weights"
                assert not jnp.allclose(member_1, member_2), "Members 1 and 2 should have different weights"

    def test_ensemble_network_output_shape(self):
        """Verify ensemble network output has correct shape [ensemble, batch, feature_dim]."""
        from openpi.value_functions.networks.ensemble import EnsembleNetworkConfig

        config = EnsembleNetworkConfig(
            base_config=MLPNetworkConfig(
                state_dim=10, action_conditioned=True, action_dim=4, action_horizon=1, hidden_dims=(32,)
            ),
            ensemble_size=2,
        )
        network = config.create(jax.random.key(0))

        obs = make_observation(jnp.ones((8, 10)))
        action = jnp.ones((8, 1, 4))
        features = network.compute_features(obs, action)

        # Shape should be [ensemble_size, batch_size, feature_dim]
        assert features.shape == (2, 8, 32), f"Expected (2, 8, 32), got {features.shape}"
        assert network.feature_dim == 32

    def test_ensemble_head_output_shape(self):
        """Verify ensemble head output has correct shape [ensemble, batch]."""
        from openpi.value_functions.heads import EnsembleHeadConfig

        head_config = EnsembleHeadConfig(base_config=RegressionHeadConfig(), ensemble_size=2)
        head = head_config.create(feature_dim=32, rng=jax.random.key(0))

        # Ensemble features: [ensemble, batch, feature_dim]
        features = jnp.ones((2, 8, 32))
        values = head(features)

        assert values.shape == (2, 8), f"Expected (2, 8), got {values.shape}"

    def test_ensemble_head_compute_min(self):
        """Verify compute_min returns minimum across ensemble."""
        from openpi.value_functions.heads import EnsembleHeadConfig

        head_config = EnsembleHeadConfig(base_config=RegressionHeadConfig(), ensemble_size=3)
        head = head_config.create(feature_dim=32, rng=jax.random.key(0))

        # Different features for each ensemble member to get different values
        features = jnp.stack(
            [
                jnp.ones((4, 32)) * 1.0,
                jnp.ones((4, 32)) * 2.0,
                jnp.ones((4, 32)) * 3.0,
            ]
        )  # [3, 4, 32]

        all_values = head(features)  # [3, 4]
        min_values = head.compute_min(features)  # [4]

        assert min_values.shape == (4,), f"Expected (4,), got {min_values.shape}"
        # Min should be less than or equal to all members
        assert jnp.all(min_values <= all_values[0])
        assert jnp.all(min_values <= all_values[1])
        assert jnp.all(min_values <= all_values[2])

    def test_ensemble_head_has_different_weights(self):
        """Verify each ensemble head member has different weights."""

        from openpi.value_functions.heads import EnsembleHeadConfig

        head_config = EnsembleHeadConfig(base_config=RegressionHeadConfig(), ensemble_size=3)
        head = head_config.create(feature_dim=32, rng=jax.random.key(0))

        state = nnx.state(head.vectorized_head)

        for _, param in jax.tree_util.tree_leaves_with_path(state):
            if hasattr(param, "value") and param.value.ndim >= 2:
                assert param.value.shape[0] == 3, f"Expected ensemble dim 3, got {param.value.shape[0]}"
                member_0 = param.value[0]
                member_1 = param.value[1]
                assert not jnp.allclose(member_0, member_1), "Members should have different weights"

    def test_ensemble_hl_gauss_loss(self):
        """Verify HL-Gauss loss computation for ensemble."""
        from openpi.value_functions.heads import EnsembleHeadConfig
        from openpi.value_functions.value_function_objectives import _compute_value_loss

        # Create ensemble of CategoricalHeads
        head_config = EnsembleHeadConfig(
            base_config=CategoricalHeadConfig(v_min=-1.0, v_max=1.0, num_bins=10, sigma=0.1),
            ensemble_size=2,
        )
        head = head_config.create(feature_dim=32, rng=jax.random.key(0))

        # Features: [ensemble, batch, feature_dim]
        # Target: [batch]
        features = jnp.zeros((2, 4, 32))
        target = jnp.zeros((4,))

        loss = _compute_value_loss(head, features, target)

        assert loss.shape == (4,)
        assert jnp.all(loss > 0)
        assert not jnp.isnan(loss).any()


# =============================================================================
# Multi-Transition Value Function Tests
# =============================================================================


def make_multi_observation(state: jnp.ndarray) -> Observation:
    """Create observation for multi-transition inputs."""
    return Observation(images={}, image_masks={}, state=state)


def make_multi_transition(
    batch_size: int, num_transitions_per_sample: int, state_dim: int, action_dim: int = 4, action_horizon: int = 1
) -> MultiTransition:
    """Create a MultiTransition with shape [batch, n, ...]."""
    observation = make_multi_observation(jnp.ones((batch_size, num_transitions_per_sample, state_dim)))
    next_observation = make_multi_observation(jnp.ones((batch_size, num_transitions_per_sample, state_dim)))
    return MultiTransition(
        observation=observation,
        action=jnp.ones((batch_size, num_transitions_per_sample, action_horizon, action_dim)),
        reward=jnp.zeros((batch_size, num_transitions_per_sample)),
        next_observation=next_observation,
        next_action=jnp.ones((batch_size, num_transitions_per_sample, action_horizon, action_dim)),
        mc_return=jnp.ones((batch_size, num_transitions_per_sample)) * 0.5,
        termination=jnp.zeros((batch_size, num_transitions_per_sample), dtype=bool),
        truncation=jnp.zeros((batch_size, num_transitions_per_sample), dtype=bool),
    )


class TestMultiMLPNetwork:
    def test_create_and_forward(self):
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig

        config = MultiMLPNetworkConfig(
            state_dim=10,
            num_transitions_per_sample=4,
            hidden_dims=(64, 32),
        )
        network = config.create(jax.random.key(0))

        obs = make_multi_observation(jnp.ones((2, 4, 10)))  # [batch=2, n=4, state_dim=10]
        features = network.compute_features(obs)

        assert features.shape == (2, 4, 32)  # [batch, n, feature_dim]
        assert network.feature_dim == 32

    def test_action_conditioned(self):
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig

        config = MultiMLPNetworkConfig(
            state_dim=10,
            num_transitions_per_sample=4,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
        )
        network = config.create(jax.random.key(0))

        obs = make_multi_observation(jnp.ones((2, 4, 10)))  # [batch=2, n=4, state_dim=10]
        action = jnp.ones((2, 4, 1, 4))  # [batch, n, action_horizon, action_dim]
        features = network.compute_features(obs, action)

        assert features.shape == (2, 4, 256)  # [batch, n, feature_dim]

    def test_batch_permutation_invariance(self):
        """Verify that reordering samples in batch dimension doesn't change outputs."""
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig

        config = MultiMLPNetworkConfig(
            state_dim=10,
            num_transitions_per_sample=4,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(32,),
        )
        network = config.create(jax.random.key(0))
        head = RegressionHeadConfig().create(network.feature_dim, jax.random.key(1))

        # Create distinct inputs for each batch element
        rng = jax.random.key(42)
        batch_size = 3
        num_transitions = 4
        state = jax.random.normal(rng, (batch_size, num_transitions, 10))
        action = jax.random.normal(rng, (batch_size, num_transitions, 1, 4))
        obs = make_multi_observation(state)

        features_original = network.compute_features(obs, action)
        assert features_original.shape == (batch_size, num_transitions, 32)

        values_original = head(features_original)
        assert values_original.shape == (batch_size, num_transitions)

        # Permute batch dimension: [0, 1, 2] -> [2, 0, 1]
        perm = jnp.array([2, 0, 1])
        obs_permuted = make_multi_observation(state[perm])
        action_permuted = action[perm]

        features_permuted = network.compute_features(obs_permuted, action_permuted)
        assert features_permuted.shape == (batch_size, num_transitions, 32)

        values_permuted = head(features_permuted)
        assert values_permuted.shape == (batch_size, num_transitions)

        # Features should be the same, just reordered by batch
        np.testing.assert_allclose(features_original[0], features_permuted[1], rtol=1e-5)
        np.testing.assert_allclose(features_original[1], features_permuted[2], rtol=1e-5)
        np.testing.assert_allclose(features_original[2], features_permuted[0], rtol=1e-5)

        # Values should be the same, just reordered by batch
        np.testing.assert_allclose(values_original[0], values_permuted[1], rtol=1e-5)
        np.testing.assert_allclose(values_original[1], values_permuted[2], rtol=1e-5)
        np.testing.assert_allclose(values_original[2], values_permuted[0], rtol=1e-5)

    def test_transition_order_changes_output(self):
        """Verify that reordering transitions within a sample DOES change outputs.

        This is the key property of multi-state value functions: the network
        sees all n transitions together, so their order matters.
        """
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig

        config = MultiMLPNetworkConfig(
            state_dim=10,
            num_transitions_per_sample=4,
            action_conditioned=True,
            action_dim=4,
            action_horizon=1,
            hidden_dims=(32,),
        )
        network = config.create(jax.random.key(0))
        head = RegressionHeadConfig().create(network.feature_dim, jax.random.key(1))

        # Create distinct inputs for each transition
        rng = jax.random.key(42)
        batch_size = 2
        num_transitions = 4
        state = jax.random.normal(rng, (batch_size, num_transitions, 10))
        action = jax.random.normal(rng, (batch_size, num_transitions, 1, 4))
        obs = make_multi_observation(state)

        features_original = network.compute_features(obs, action)
        assert features_original.shape == (batch_size, num_transitions, 32)

        values_original = head(features_original)
        assert values_original.shape == (batch_size, num_transitions)

        # Permute n dimension: [0, 1, 2, 3] -> [3, 2, 1, 0]
        n_perm = jnp.array([3, 2, 1, 0])
        obs_permuted = make_multi_observation(state[:, n_perm])
        action_permuted = action[:, n_perm]

        features_permuted = network.compute_features(obs_permuted, action_permuted)
        assert features_permuted.shape == (batch_size, num_transitions, 32)

        values_permuted = head(features_permuted)
        assert values_permuted.shape == (batch_size, num_transitions)

        # Outputs should be DIFFERENT (not just reordered) because all transitions
        # are concatenated and processed together through the MLP
        features_original_reordered = features_original[:, n_perm]
        assert not jnp.allclose(features_permuted, features_original_reordered, rtol=1e-5), (
            "Transition permutation should change feature values, not just reorder them"
        )

        values_original_reordered = values_original[:, n_perm]
        assert not jnp.allclose(values_permuted, values_original_reordered, rtol=1e-5), (
            "Transition permutation should change value predictions, not just reorder them"
        )

    def test_loss_batch_permutation_invariance(self):
        """Verify that loss is invariant to batch permutations."""
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig
        from openpi.value_functions.value_function import MultiMCValueFunctionConfig

        config = MultiMCValueFunctionConfig(
            network_config=MultiMLPNetworkConfig(
                state_dim=10,
                num_transitions_per_sample=4,
                hidden_dims=(32,),
            ),
            head_config=RegressionHeadConfig(),
        )
        model = config.create(jax.random.key(0))

        # Create random multi-transition batch
        rng = jax.random.key(42)
        batch_size = 3
        num_transitions = 4
        state = jax.random.normal(rng, (batch_size, num_transitions, 10))
        next_state = jax.random.normal(jax.random.key(43), (batch_size, num_transitions, 10))
        action = jax.random.normal(jax.random.key(44), (batch_size, num_transitions, 1, 4))
        mc_return = jax.random.uniform(jax.random.key(45), (batch_size, num_transitions))

        transition = MultiTransition(
            observation=make_multi_observation(state),
            action=action,
            reward=jnp.zeros((batch_size, num_transitions)),
            next_observation=make_multi_observation(next_state),
            next_action=action,
            mc_return=mc_return,
            termination=jnp.zeros((batch_size, num_transitions), dtype=bool),
            truncation=jnp.zeros((batch_size, num_transitions), dtype=bool),
        )

        loss_original, _ = model.compute_loss(transition)
        assert loss_original.shape == (batch_size, num_transitions)

        # Permute batch dimension
        perm = jnp.array([2, 0, 1])
        transition_permuted = MultiTransition(
            observation=make_multi_observation(state[perm]),
            action=action[perm],
            reward=jnp.zeros((batch_size, num_transitions)),
            next_observation=make_multi_observation(next_state[perm]),
            next_action=action[perm],
            mc_return=mc_return[perm],
            termination=jnp.zeros((batch_size, num_transitions), dtype=bool),
            truncation=jnp.zeros((batch_size, num_transitions), dtype=bool),
        )

        loss_permuted, _ = model.compute_loss(transition_permuted)
        assert loss_permuted.shape == (batch_size, num_transitions)

        # Losses should be the same, just reordered by batch
        np.testing.assert_allclose(loss_original[0], loss_permuted[1], rtol=1e-5)
        np.testing.assert_allclose(loss_original[1], loss_permuted[2], rtol=1e-5)
        np.testing.assert_allclose(loss_original[2], loss_permuted[0], rtol=1e-5)

    def test_loss_transition_order_changes_output(self):
        """Verify that loss changes when transitions are reordered within a sample."""
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig
        from openpi.value_functions.value_function import MultiMCValueFunctionConfig

        config = MultiMCValueFunctionConfig(
            network_config=MultiMLPNetworkConfig(
                state_dim=10,
                num_transitions_per_sample=4,
                hidden_dims=(32,),
            ),
            head_config=RegressionHeadConfig(),
        )
        model = config.create(jax.random.key(0))

        # Create random multi-transition batch
        rng = jax.random.key(42)
        batch_size = 2
        num_transitions = 4
        state = jax.random.normal(rng, (batch_size, num_transitions, 10))
        next_state = jax.random.normal(jax.random.key(43), (batch_size, num_transitions, 10))
        action = jax.random.normal(jax.random.key(44), (batch_size, num_transitions, 1, 4))
        mc_return = jax.random.uniform(jax.random.key(45), (batch_size, num_transitions))

        transition = MultiTransition(
            observation=make_multi_observation(state),
            action=action,
            reward=jnp.zeros((batch_size, num_transitions)),
            next_observation=make_multi_observation(next_state),
            next_action=action,
            mc_return=mc_return,
            termination=jnp.zeros((batch_size, num_transitions), dtype=bool),
            truncation=jnp.zeros((batch_size, num_transitions), dtype=bool),
        )

        loss_original, _ = model.compute_loss(transition)
        assert loss_original.shape == (batch_size, num_transitions)

        # Permute n dimension
        n_perm = jnp.array([3, 2, 1, 0])
        transition_permuted = MultiTransition(
            observation=make_multi_observation(state[:, n_perm]),
            action=action[:, n_perm],
            reward=jnp.zeros((batch_size, num_transitions)),
            next_observation=make_multi_observation(next_state[:, n_perm]),
            next_action=action[:, n_perm],
            mc_return=mc_return[:, n_perm],
            termination=jnp.zeros((batch_size, num_transitions), dtype=bool),
            truncation=jnp.zeros((batch_size, num_transitions), dtype=bool),
        )

        loss_permuted, _ = model.compute_loss(transition_permuted)
        assert loss_permuted.shape == (batch_size, num_transitions)

        # Losses should be DIFFERENT (not just reordered)
        loss_original_reordered = loss_original[:, n_perm]
        assert not jnp.allclose(loss_permuted, loss_original_reordered, rtol=1e-5), (
            "Transition permutation should change loss values, not just reorder them"
        )


class TestMultiMCValueFunction:
    def test_regression_head(self):
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig
        from openpi.value_functions.value_function import MultiMCValueFunctionConfig

        config = MultiMCValueFunctionConfig(
            network_config=MultiMLPNetworkConfig(
                state_dim=10,
                num_transitions_per_sample=4,
                hidden_dims=(32,),
            ),
            head_config=RegressionHeadConfig(),
        )
        model = config.create(jax.random.key(0))

        obs = make_multi_observation(jnp.ones((2, 4, 10)))
        values = model.compute_value(obs)
        assert values.shape == (2, 4)

        transition = make_multi_transition(2, 4, 10)
        loss, info = model.compute_loss(transition)
        assert loss.shape == (2, 4)
        assert "predicted_value_mean" in info


class TestMultiSARSAValueFunction:
    def test_with_target_network(self):
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig
        from openpi.value_functions.value_function import MultiSARSAValueFunctionConfig

        config = MultiSARSAValueFunctionConfig(
            network_config=MultiMLPNetworkConfig(
                state_dim=10,
                num_transitions_per_sample=4,
                action_conditioned=True,
                action_dim=4,
                hidden_dims=(32,),
            ),
            head_config=RegressionHeadConfig(),
            discount=0.99,
            tau=0.005,
        )
        model = config.create(jax.random.key(0))

        transition = make_multi_transition(2, 4, 10, action_dim=4)
        loss, info = model.compute_loss(transition)

        assert loss.shape == (2, 4)
        assert "next_value_mean" in info

        # Test target network update
        model.post_step_update()


class TestMultiIQLValueFunction:
    def test_expectile_regression(self):
        from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig
        from openpi.value_functions.value_function import MultiIQLValueFunctionConfig

        config = MultiIQLValueFunctionConfig(
            q_network_config=MultiMLPNetworkConfig(
                state_dim=10,
                num_transitions_per_sample=4,
                action_conditioned=True,
                action_dim=4,
                hidden_dims=(32,),
            ),
            v_network_config=MultiMLPNetworkConfig(
                state_dim=10,
                num_transitions_per_sample=4,
                action_conditioned=False,
                hidden_dims=(32,),
            ),
            q_head_config=RegressionHeadConfig(),
            v_head_config=RegressionHeadConfig(),
            expectile=0.7,
            discount=0.99,
            tau=0.005,
        )
        model = config.create(jax.random.key(0))

        transition = make_multi_transition(2, 4, 10, action_dim=4)
        loss, info = model.compute_loss(transition)

        assert loss.shape == (2, 4)
        assert "positive_advantage_frac" in info
        assert "q_mean" in info
        assert "v_mean" in info

        # Test target network update
        model.post_step_update()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
