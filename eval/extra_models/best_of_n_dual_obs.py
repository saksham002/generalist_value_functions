"""Best-of-N wrapper with separate policy and critic observations."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import best_of_n as _best_of_n
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions import base_value_functions as _base_vf


class DualObservationBestOfNWrapper(_best_of_n.BestOfNWrapper):
    """Best-of-N wrapper that can score with a separate critic observation."""

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool = False,
        value_function: _base_vf.BaseValueFunction | _base_vf.BaseMultiValueFunction | None = None,
        critic_observation: _model.Observation | None = None,
        critic_action_transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        **kwargs,
    ) -> _model.Actions:
        if value_function is None:
            raise ValueError("value_function is required for DualObservationBestOfNWrapper")
        if critic_observation is None or critic_action_transform is None:
            return super().sample_actions(
                rng,
                transition,
                compute_next_action = compute_next_action,
                value_function = value_function,
                **kwargs,
            )

        observation = transition.next_observation if compute_next_action else transition.observation
        rng_sample, rng_select = jax.random.split(rng)
        batch_size = observation.state.shape[0]

        if self.base_model is None:
            all_actions = self._get_cached_actions(transition, compute_next_action = compute_next_action)
        else:
            sample_rngs = jax.random.split(rng_sample, self.num_samples)

            @eqx.filter_vmap(in_axes = (0, None, None, None))
            def sample_with_rng(rng_i, model, trans, next_action):
                return model.sample_actions(rng_i, trans, compute_next_action = next_action, **kwargs)

            all_actions = sample_with_rng(sample_rngs, self.base_model, transition, compute_next_action)
            all_actions = jnp.moveaxis(all_actions, 0, 1)

        critic_actions = critic_action_transform(all_actions)
        num_samples = all_actions.shape[1]
        expanded_critic_obs = _best_of_n.expand_observation(critic_observation, num_samples)
        flat_critic_actions = critic_actions.reshape(
            batch_size * num_samples,
            critic_actions.shape[2],
            critic_actions.shape[3],
        )

        if self.use_target_value:
            q_values = value_function.compute_target_value(
                expanded_critic_obs,
                flat_critic_actions,
                take_min_over_ensemble = self.take_min_over_ensemble,
            )
        else:
            q_values = value_function.compute_value(
                expanded_critic_obs,
                flat_critic_actions,
                take_min_over_ensemble = self.take_min_over_ensemble,
            )
        q_values = q_values.reshape(batch_size, num_samples)

        if self.selection_mode == "argmax":
            indices = jnp.argmax(q_values, axis = 1)
        elif self.selection_mode == "softmax":
            logits = q_values / self.softmax_temperature
            indices = jax.random.categorical(rng_select, logits, axis = 1)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        return all_actions[jnp.arange(batch_size), indices]
