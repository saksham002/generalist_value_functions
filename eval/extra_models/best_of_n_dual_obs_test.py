import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.value_functions import base_value_functions as _base_vf

from eval.extra_models.best_of_n_dual_obs import DualObservationBestOfNWrapper


class _MockPolicy(_model.BaseModel):
    def __init__(self):
        super().__init__(action_dim = 2, action_horizon = 1, max_token_len = 0)

    @override
    def compute_loss(self, rng, observation, actions, *, train = False):
        del rng, observation, actions, train
        return jnp.zeros((1, 1))

    @override
    def sample_actions(self, rng, transition, *, compute_next_action = False, **kwargs):
        del compute_next_action, kwargs
        batch_size = transition.observation.state.shape[0]
        sample_id = jax.random.randint(rng, (), 0, 100)
        values = jnp.broadcast_to(sample_id.astype(jnp.float32), (batch_size, 1, 2))
        return values


class _Critic(_base_vf.BaseValueFunction):
    @override
    def compute_value(self, observation, action = None, *, take_min_over_ensemble = False):
        del take_min_over_ensemble
        assert action is not None
        return observation.state[:, 0] + action[:, 0, 0]

    @override
    def compute_target_value(self, observation, action = None, *, take_min_over_ensemble = False):
        return self.compute_value(observation, action, take_min_over_ensemble = take_min_over_ensemble)

    @override
    def compute_loss(self, transition, *, train = False, rng = None, policy = None):
        del transition, train, rng, policy
        return jnp.zeros((1,)), {}


def test_dual_observation_best_of_n_uses_critic_observation():
    policy_observation = _model.Observation(images = {}, image_masks = {}, state = jnp.zeros((1, 4), dtype = jnp.float32))
    critic_observation = _model.Observation(
        images = {},
        image_masks = {},
        state = jnp.array([[100.0, 0.0, 0.0, 0.0]], dtype = jnp.float32),
    )
    transition = _model.wrap_observation_as_transition(policy_observation)
    wrapper = DualObservationBestOfNWrapper(
        action_dim = 2,
        action_horizon = 1,
        max_token_len = 0,
        base_model = _MockPolicy(),
        num_samples = 4,
        take_min_over_ensemble = True,
        use_target_value = False,
        selection_mode = "argmax",
        softmax_temperature = 1.0,
    )

    def critic_action_transform(actions: jnp.ndarray) -> jnp.ndarray:
        return actions

    selected = wrapper.sample_actions(
        jax.random.key(0),
        transition,
        value_function = _Critic(),
        critic_observation = critic_observation,
        critic_action_transform = critic_action_transform,
    )

    assert selected.shape == (1, 1, 2)
    assert float(selected[0, 0, 0]) == float(selected[0, 0, 1])
