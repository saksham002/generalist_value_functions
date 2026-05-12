import dataclasses

import equinox as eqx
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing_extensions import override

import openpi.transforms as _transforms
from openpi.models import best_of_n
from openpi.models import model as _model
from openpi.models import tanh_gaussian
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
from openpi.value_functions import base_value_functions as _base_vf

Transition = _base_vf.Transition


def _make_transition(obs: _model.Observation) -> Transition:
    return Transition(observation=obs)


@dataclasses.dataclass
class _MockValueFunction(_base_vf.BaseValueFunction):
    """Value function that returns the sum of actions as Q-value (for testing selection)."""

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        assert action is not None
        return action.sum(axis=(-2, -1))

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        assert action is not None
        # Return negative sum to verify use_target_value selects differently.
        return -action.sum(axis=(-2, -1))

    @override
    def compute_loss(self, transition, *, train=False, rng=None, policy=None):
        batch_size = transition.observation.state.shape[0]
        return jnp.zeros((batch_size,)), {}


@dataclasses.dataclass
class _MockPrefixCacheValueFunction(_base_vf.BaseValueFunction):
    prefix_cache_calls: int = 0
    last_prefix_cache: tuple[jax.Array, jax.Array] | None = None
    last_use_target: bool | None = None

    def compute_prefix_cache(
        self,
        observation: _model.Observation,
        use_target: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        self.prefix_cache_calls += 1
        self.last_use_target = use_target
        batch_size = observation.state.shape[0]
        kv_cache = {"k": jnp.arange(batch_size * 2, dtype = jnp.float32).reshape(1, batch_size, 2)}
        prefix_mask = jnp.ones((batch_size, 3), dtype = jnp.bool_)
        return kv_cache, prefix_mask

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
        prefix_cache: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b _n"]]:
        del observation, take_min_over_ensemble
        assert action is not None
        self.last_prefix_cache = prefix_cache
        values = action.sum(axis = (-2, -1))
        attn_scores = jnp.zeros((values.shape[0], 1), dtype = values.dtype)
        return values, attn_scores

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
        prefix_cache: tuple[jax.Array, jax.Array] | None = None,
    ) -> at.Float[at.Array, "*b"]:
        del observation, take_min_over_ensemble
        assert action is not None
        self.last_prefix_cache = prefix_cache
        return -action.sum(axis = (-2, -1))

    @override
    def compute_loss(self, transition, *, train=False, rng=None, policy=None):
        del transition, train, rng, policy
        return jnp.zeros((1,)), {}


def _make_config(num_samples=5, **kwargs):
    base = tanh_gaussian.TanhGaussianConfig(
        state_dim=4,
        action_dim=2,
        action_horizon=1,
    )
    return best_of_n.BestOfNWrapperConfig(
        base_model_config=base,
        num_samples=num_samples,
        **kwargs,
    )


def _make_obs(batch_size=3):
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.ones((batch_size, 4)),
    )


def test_sample_actions_shape():
    rng = jax.random.key(0)
    config = _make_config(num_samples=5)
    model = config.create(rng)
    vf = _MockValueFunction()

    obs = _make_obs(batch_size=3)
    transition = _make_transition(obs)
    actions = model.sample_actions(rng, transition, value_function=vf)
    assert actions.shape == (3, 1, 2)


def test_argmax_selects_best():
    """With argmax and Q = sum(action), the action with the largest sum should be selected."""
    rng = jax.random.key(42)
    config = _make_config(num_samples=50, selection_mode="argmax")
    model = config.create(rng)
    vf = _MockValueFunction()

    obs = _make_obs(batch_size=2)
    transition = _make_transition(obs)

    for i in range(3):
        rng_i = jax.random.fold_in(rng, i)
        rng_sample, _ = jax.random.split(rng_i)

        selected = model.sample_actions(rng_sample, transition, value_function=vf)

        # Reproduce the same candidates using eqx.filter_vmap (same as the implementation).
        sample_rngs = jax.random.split(jax.random.split(rng_sample)[0], 50)

        @eqx.filter_vmap(in_axes=(0, None, None))
        def sample_with_rng(rng_j, base_model, trans):
            return base_model.sample_actions(rng_j, trans, compute_next_action=False)

        all_actions = sample_with_rng(sample_rngs, model.base_model, transition)
        all_actions = jnp.moveaxis(all_actions, 0, 1)

        q_values = all_actions.sum(axis=(-2, -1))
        best_indices = jnp.argmax(q_values, axis=1)
        expected = all_actions[jnp.arange(2), best_indices]

        assert jnp.allclose(selected, expected), f"Iteration {i}: selected != expected"


def test_softmax_selection_shape():
    rng = jax.random.key(0)
    config = _make_config(num_samples=10, selection_mode="softmax", softmax_temperature=0.1)
    model = config.create(rng)
    vf = _MockValueFunction()

    obs = _make_obs(batch_size=4)
    transition = _make_transition(obs)
    actions = model.sample_actions(rng, transition, value_function=vf)
    assert actions.shape == (4, 1, 2)


def test_use_target_value():
    """With use_target_value=True, the mock returns negative Q-values,
    so argmax should select the action with the smallest sum."""
    rng = jax.random.key(0)
    config_normal = _make_config(num_samples=50, selection_mode="argmax", use_target_value=False)
    config_target = _make_config(num_samples=50, selection_mode="argmax", use_target_value=True)

    model_normal = config_normal.create(rng)
    model_target = config_target.create(rng)
    vf = _MockValueFunction()

    obs = _make_obs(batch_size=1)
    transition = _make_transition(obs)
    actions_normal = model_normal.sample_actions(rng, transition, value_function=vf)
    actions_target = model_target.sample_actions(rng, transition, value_function=vf)

    q_normal = actions_normal.sum()
    q_target = actions_target.sum()
    assert q_target <= q_normal + 1e-5


def test_compute_loss_delegates():
    rng = jax.random.key(0)
    config = _make_config()
    model = config.create(rng)

    obs = _make_obs(batch_size=2)
    actions = jnp.zeros((2, 1, 2))

    loss = model.compute_loss(rng, obs, actions)
    assert loss.shape == (2, 1)


def test_expand_observation():
    batch_size = 3
    num_samples = 4
    state_dim = 5

    obs = _model.Observation(
        images={"cam": jnp.arange(batch_size * 2 * 2 * 3, dtype=jnp.float32).reshape(batch_size, 2, 2, 3)},
        image_masks={"cam": jnp.ones((batch_size,), dtype=jnp.bool_)},
        state=jnp.arange(batch_size * state_dim, dtype=jnp.float32).reshape(batch_size, state_dim),
    )

    expanded = best_of_n.expand_observation(obs, num_samples)

    assert expanded.state.shape == (batch_size * num_samples, state_dim)
    assert expanded.images["cam"].shape == (batch_size * num_samples, 2, 2, 3)
    assert expanded.image_masks["cam"].shape == (batch_size * num_samples,)

    for i in range(batch_size):
        for j in range(num_samples):
            assert jnp.allclose(expanded.state[i * num_samples + j], obs.state[i])
            assert jnp.allclose(expanded.images["cam"][i * num_samples + j], obs.images["cam"][i])


def test_expand_observation_none_fields():
    obs = _model.Observation(
        images={},
        image_masks={},
        state=jnp.ones((2, 3)),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
    )
    expanded = best_of_n.expand_observation(obs, 5)
    assert expanded.state.shape == (10, 3)
    assert expanded.tokenized_prompt is None
    assert expanded.tokenized_prompt_mask is None


def test_config_delegates_properties():
    base = tanh_gaussian.TanhGaussianConfig(
        state_dim=4,
        action_dim=8,
        action_horizon=3,
        max_token_len=10,
    )
    config = best_of_n.BestOfNWrapperConfig(base_model_config=base)
    assert config.action_dim == 8
    assert config.action_horizon == 3
    assert config.max_token_len == 10
    assert config.model_type == _model.ModelType.BEST_OF_N


def test_inputs_spec_delegates():
    base = tanh_gaussian.TanhGaussianConfig(
        state_dim=4,
        action_dim=2,
        action_horizon=1,
    )
    config = best_of_n.BestOfNWrapperConfig(base_model_config=base)
    obs_spec, act_spec = config.inputs_spec(batch_size=8)
    assert obs_spec.state.shape == (8, 4)
    assert act_spec.shape == (8, 1, 2)


def test_best_of_n_with_in_context_mmdit():
    """Test BestOfN with InContextMMDiT base model (equinox-based).

    This tests the fix for the vmap issue where vmapping over equinox model
    parameters caused shape mismatches.
    """
    in_context_mmdit = pytest.importorskip("openpi.models.in_context_mmdit")

    rng = jax.random.key(0)
    base_config = in_context_mmdit.InContextMMDiTConfig(
        state_dim=8,
        action_dim=2,
        action_horizon=1,
        depth=2,
        obs_hidden_dim=32,
        action_hidden_dim=32,
        dim_head=16,
        num_heads=2,
        ff_mult=2,
        num_time_tokens=1,
        time_embed_dim=32,
        num_inference_steps=2,
        num_state_tokens=1,
    )
    config = best_of_n.BestOfNWrapperConfig(
        base_model_config=base_config,
        num_samples=4,
        selection_mode="argmax",
    )
    model = config.create(rng)
    vf = _MockValueFunction()

    obs = _model.Observation(
        images={},
        image_masks={},
        state=jnp.ones((2, 8)),
    )
    transition = _make_transition(obs)

    # This should not raise an error about shape mismatches
    actions = model.sample_actions(rng, transition, value_function=vf)
    assert actions.shape == (2, 1, 2)


def test_cached_best_of_n_uses_counterfactual_next_actions():
    rng = jax.random.key(0)
    config = best_of_n.BestOfNWrapperConfig(
        action_dim=2,
        action_horizon=1,
        num_samples=3,
        base_model_config=None,
        selection_mode="argmax",
    )
    model = config.create(rng)
    vf = _MockValueFunction()

    obs = _make_obs(batch_size=2)
    transition = Transition(
        observation=obs,
        action=jnp.zeros((2, 1, 2)),
        reward=jnp.zeros((2,)),
        next_observation=obs,
        next_action=jnp.zeros((2, 1, 2)),
        mc_return=jnp.zeros((2,)),
        termination=jnp.zeros((2,), dtype=bool),
        truncation=jnp.zeros((2,), dtype=bool),
        td_discount=None,
        counterfactual_next_actions=jnp.array(
            [
                [[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]],
                [[[3.0, 0.0]], [[0.5, 0.5]], [[1.0, 4.0]]],
            ]
        ),
    )

    actions = model.sample_actions(rng, transition, compute_next_action=True, value_function=vf)
    expected = jnp.array(
        [
            [[2.0, 2.0]],
            [[1.0, 4.0]],
        ]
    )
    assert jnp.allclose(actions, expected)


def test_cached_best_of_n_requires_enough_counterfactual_actions():
    rng = jax.random.key(0)
    config = best_of_n.BestOfNWrapperConfig(
        action_dim=2,
        action_horizon=1,
        num_samples=4,
        base_model_config=None,
    )
    model = config.create(rng)
    vf = _MockValueFunction()

    obs = _make_obs(batch_size=1)
    transition = Transition(
        observation=obs,
        action=jnp.zeros((1, 1, 2)),
        reward=jnp.zeros((1,)),
        next_observation=obs,
        next_action=jnp.zeros((1, 1, 2)),
        mc_return=jnp.zeros((1,)),
        termination=jnp.zeros((1,), dtype=bool),
        truncation=jnp.zeros((1,), dtype=bool),
        td_discount=None,
        counterfactual_next_actions=jnp.ones((1, 3, 1, 2)),
    )

    with pytest.raises(ValueError, match="expected at least 4, got 3"):
        model.sample_actions(rng, transition, compute_next_action=True, value_function=vf)


def test_best_of_n_reuses_prefix_cache_for_value_evaluation():
    rng = jax.random.key(0)
    config = _make_config(num_samples = 4, selection_mode = "argmax")
    model = config.create(rng)
    vf = _MockPrefixCacheValueFunction()

    obs = _make_obs(batch_size = 2)
    transition = _make_transition(obs)

    actions = model.sample_actions(rng, transition, value_function = vf)

    assert actions.shape == (2, 1, 2)
    assert vf.prefix_cache_calls == 1
    assert vf.last_prefix_cache is not None
    kv_cache, prefix_mask = vf.last_prefix_cache
    assert kv_cache["k"].shape == (1, 8, 2)
    assert prefix_mask.shape == (8, 3)


def test_best_of_n_handles_tuple_return_from_compute_value():
    rng = jax.random.key(0)
    config = best_of_n.BestOfNWrapperConfig(
        action_dim = 2,
        action_horizon = 1,
        num_samples = 3,
        base_model_config = None,
        selection_mode = "argmax",
    )
    model = config.create(rng)
    vf = _MockPrefixCacheValueFunction()

    obs = _make_obs(batch_size = 1)
    transition = Transition(
        observation = obs,
        action = jnp.zeros((1, 1, 2)),
        reward = jnp.zeros((1,)),
        next_observation = obs,
        next_action = jnp.zeros((1, 1, 2)),
        mc_return = jnp.zeros((1,)),
        termination = jnp.zeros((1,), dtype = bool),
        truncation = jnp.zeros((1,), dtype = bool),
        td_discount = None,
        counterfactual_next_actions = jnp.array([[[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]]]),
    )

    actions = model.sample_actions(rng, transition, compute_next_action = True, value_function = vf)

    assert jnp.allclose(actions, jnp.array([[[2.0, 2.0]]]))


def test_best_of_n_uses_target_prefix_cache_when_configured():
    rng = jax.random.key(0)
    config = _make_config(num_samples = 4, selection_mode = "argmax", use_target_value = True)
    model = config.create(rng)
    vf = _MockPrefixCacheValueFunction()

    obs = _make_obs(batch_size = 2)
    transition = _make_transition(obs)

    model.sample_actions(rng, transition, value_function = vf)

    assert vf.last_use_target is True


def _make_renormalize_wrapper(
    *,
    batch_size: int,
    num_samples: int,
    action_horizon: int,
    action_dim: int,
) -> best_of_n.BestOfNWrapper:
    """Build a cached-only wrapper exercising _renormalize_actions with identity norm stats.

    Identity norm stats (mean=0, std=1) make Unnormalize/Normalize no-ops so any difference
    between the eager and jit-traced outputs reflects the AbsoluteActions / pure_callback path.
    """
    identity_stats = {
        "actions": NormStats(
            mean = np.zeros((action_dim,), dtype = np.float32),
            std = np.ones((action_dim,), dtype = np.float32),
        )
    }
    # 14-d EEF layout used by the production data pipeline: positions and rpy are made
    # absolute, gripper dims are left untouched.
    delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
    absolute_actions = _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
    return best_of_n.BestOfNWrapper(
        action_dim = action_dim,
        action_horizon = action_horizon,
        max_token_len = 0,
        base_model = None,
        num_samples = num_samples,
        take_min_over_ensemble = True,
        use_target_value = False,
        convert_to_global = True,
        selection_mode = "argmax",
        softmax_temperature = 1.0,
        policy_norm_stats = identity_stats,
        critic_norm_stats = identity_stats,
        critic_action_dim_offset = None,
        critic_action_horizon = None,
        absolute_actions = absolute_actions,
    )


def test_renormalize_actions_matches_absolute_actions_eager():
    """The wrapper's renorm path with identity stats must equal a direct AbsoluteActions call."""
    batch_size, num_samples, action_horizon, action_dim = 2, 3, 4, 14
    wrapper = _make_renormalize_wrapper(
        batch_size = batch_size,
        num_samples = num_samples,
        action_horizon = action_horizon,
        action_dim = action_dim,
    )

    rng = np.random.default_rng(seed = 86)
    actions = jnp.asarray(
        rng.standard_normal((batch_size, num_samples, action_horizon, action_dim)).astype(np.float32)
    )
    initial_pose = jnp.asarray(
        rng.standard_normal((batch_size, 1, action_dim)).astype(np.float32)
    )

    out = wrapper._renormalize_actions(actions, initial_pose = initial_pose)

    delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
    abs_transform = _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
    state_flat = np.broadcast_to(
        np.asarray(initial_pose), (batch_size, num_samples, action_dim)
    ).reshape(batch_size * num_samples, action_dim)
    actions_flat = np.asarray(actions).reshape(batch_size * num_samples, action_horizon, action_dim)
    expected_flat = abs_transform({"state": state_flat, "actions": actions_flat})["actions"]
    expected = expected_flat.reshape(batch_size, num_samples, action_horizon, action_dim)

    assert jnp.allclose(out, jnp.asarray(expected), atol = 1e-5)


def test_renormalize_actions_runs_inside_nnx_jit():
    """jax.pure_callback must let _renormalize_actions execute under nnx.jit and match eager."""
    batch_size, num_samples, action_horizon, action_dim = 2, 3, 4, 14
    wrapper = _make_renormalize_wrapper(
        batch_size = batch_size,
        num_samples = num_samples,
        action_horizon = action_horizon,
        action_dim = action_dim,
    )

    rng = np.random.default_rng(seed = 86)
    actions = jnp.asarray(
        rng.standard_normal((batch_size, num_samples, action_horizon, action_dim)).astype(np.float32)
    )
    initial_pose = jnp.asarray(
        rng.standard_normal((batch_size, 1, action_dim)).astype(np.float32)
    )

    eager = wrapper._renormalize_actions(actions, initial_pose = initial_pose)

    @nnx.jit
    def _renorm_jit(model, acts, pose):
        return model._renormalize_actions(acts, initial_pose = pose)

    jitted = _renorm_jit(wrapper, actions, initial_pose)

    assert jitted.shape == eager.shape
    assert jnp.allclose(jitted, eager, atol = 1e-5)
