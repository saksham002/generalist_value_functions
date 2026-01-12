import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import tanh_gaussian


def test_tanh_gaussian_sampling():
    rng = jax.random.PRNGKey(0)
    config = tanh_gaussian.TanhGaussianConfig(
        state_dim=4,
        action_dim=2,
        action_horizon=1,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )
    model = config.create(rng)

    obs = _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((1, 4)),
    )

    # Stochastic sampling
    actions_stochastic = model.sample_actions(rng, obs, deterministic=False)
    assert actions_stochastic.shape == (1, 1, 2)

    # Deterministic sampling
    actions_deterministic = model.sample_actions(rng, obs, deterministic=True)
    assert actions_deterministic.shape == (1, 1, 2)

    # Check that deterministic actions are within bounds
    assert jnp.all(actions_deterministic >= -1.0)
    assert jnp.all(actions_deterministic <= 1.0)


def test_tanh_gaussian_bounds_none():
    rng = jax.random.PRNGKey(0)
    config = tanh_gaussian.TanhGaussianConfig(
        state_dim=4,
        action_dim=2,
        action_horizon=1,
        action_low=None,
        action_high=None,
    )
    model = config.create(rng)

    obs = _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((1, 4)),
    )

    actions_deterministic = model.sample_actions(rng, obs, deterministic=True)
    assert actions_deterministic.shape == (1, 1, 2)
    # Tanh outputs are in [-1, 1]
    assert jnp.all(actions_deterministic >= -1.0)
    assert jnp.all(actions_deterministic <= 1.0)
