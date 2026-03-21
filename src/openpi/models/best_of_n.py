"""Best-of-N sampling wrapper for value-guided action selection.

This module provides a wrapper around any BaseModel policy that samples N action
candidates and selects the best one according to a value function (Q-function)
passed at runtime. This enables value-guided policy improvement without modifying
the underlying policy architecture.
"""

import dataclasses
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions import base_value_functions as _base_vf


@dataclasses.dataclass(frozen=True)
class BestOfNWrapperConfig(_model.BaseModelConfig):
    """Config for Best-of-N wrapper. Value function is passed at runtime.

    If base_model_config is None, the wrapper uses cached counterfactual actions
    from the transition instead of sampling from a base model. In this case,
    action_dim and action_horizon must be provided explicitly.
    """

    action_dim: int = 0
    action_horizon: int = 0
    max_token_len: int = 0

    # Optional: if None, uses cached counterfactual actions from transition
    base_model_config: _model.BaseModelConfig | None = None

    num_samples: int = 10

    take_min_over_ensemble: bool = True

    use_target_value: bool = False

    # "argmax": pick action with highest Q-value.
    # "softmax": sample action with probability proportional to exp(Q / temperature).
    selection_mode: Literal["argmax", "softmax"] = "argmax"

    # Temperature for softmax selection (only used when selection_mode="softmax").
    softmax_temperature: float = 1.0

    def __post_init__(self):
        if self.base_model_config is not None:
            # Sync inherited fields from base_model_config (frozen dataclass requires object.__setattr__).
            object.__setattr__(self, "action_dim", self.base_model_config.action_dim)
            object.__setattr__(self, "action_horizon", self.base_model_config.action_horizon)
            object.__setattr__(self, "max_token_len", self.base_model_config.max_token_len)
        # Cached-only mode: action_dim and action_horizon must be provided
        elif self.action_dim == 0 or self.action_horizon == 0:
            raise ValueError("action_dim and action_horizon must be provided when base_model_config is None")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.BEST_OF_N

    @override
    def create(self, rng: at.KeyArrayLike) -> "BestOfNWrapper":
        base_model = self.base_model_config.create(rng) if self.base_model_config is not None else None
        return BestOfNWrapper(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
            base_model=base_model,
            num_samples=self.num_samples,
            take_min_over_ensemble=self.take_min_over_ensemble,
            use_target_value=self.use_target_value,
            selection_mode=self.selection_mode,
            softmax_temperature=self.softmax_temperature,
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        if self.base_model_config is not None:
            return self.base_model_config.inputs_spec(batch_size=batch_size)
        # Cached-only mode: return minimal spec
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, 1], jnp.float32),
            )
        actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return obs, actions


@dataclasses.dataclass
class BestOfNWrapper(_model.BaseModel):
    """Wrapper that samples N actions and selects best via Q-value.

    Value function is passed at runtime to sample_actions(), not stored.

    If base_model is None, uses cached counterfactual actions from the transition
    instead of sampling from a base model.
    """

    base_model: _model.BaseModel | None
    num_samples: int
    take_min_over_ensemble: bool
    use_target_value: bool
    selection_mode: Literal["argmax", "softmax"]
    softmax_temperature: float

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        max_token_len: int,
        *,
        base_model: _model.BaseModel | None,
        num_samples: int,
        take_min_over_ensemble: bool,
        use_target_value: bool,
        selection_mode: Literal["argmax", "softmax"],
        softmax_temperature: float,
    ):
        super().__init__(action_dim, action_horizon, max_token_len)
        self.base_model = base_model
        self.num_samples = num_samples
        self.take_min_over_ensemble = take_min_over_ensemble
        self.use_target_value = use_target_value
        self.selection_mode = selection_mode
        self.softmax_temperature = softmax_temperature

    def _get_cached_actions(
        self,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool,
    ) -> at.Array:
        """Get cached counterfactual actions from transition."""
        if compute_next_action:
            if transition.counterfactual_next_actions is None:
                raise ValueError(
                    "compute_next_action=True but transition.counterfactual_next_actions is None. "
                    "Ensure the data pipeline populates this field."
                )
            actions = transition.counterfactual_next_actions
            if actions.shape[1] < self.num_samples:
                raise ValueError(
                    "Cached counterfactual actions provide fewer samples than BestOfN requires: "
                    f"expected at least {self.num_samples}, got {actions.shape[1]} "
                    f"for shape {actions.shape}."
                )
            if actions.shape[1] > self.num_samples:
                actions = actions[:, : self.num_samples]
            # Slice to model's action_horizon if cached actions have longer horizon
            if actions.shape[2] > self.action_horizon:
                actions = actions[:, :, : self.action_horizon, :]
            return actions
        raise ValueError(
            "BestOfNWrapper without base_model only supports compute_next_action=True "
            "(cached actions are for next-action selection in TD backup)"
        )

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool = False,
        value_function: _base_vf.BaseValueFunction | _base_vf.BaseMultiValueFunction | None = None,
        **kwargs,
    ) -> _model.Actions:
        """Sample N actions and select best via Q-value.

        If base_model is provided, samples from it. Otherwise, uses cached
        counterfactual actions from the transition.

        Args:
            rng: Random key for sampling.
            transition: Transition containing observation and next_observation.
            compute_next_action: If True, use next_observation.
            value_function: Action-conditioned value function (required).
            **kwargs: Forwarded to base model's sample_actions (if base_model exists).

        Returns:
            Actions of shape [batch, action_horizon, action_dim].
        """
        if value_function is None:
            raise ValueError("value_function is required for BestOfNWrapper")

        observation = transition.next_observation if compute_next_action else transition.observation
        assert observation.state.ndim == 2, (
            f"Expected observation with a single batch dimension [B, state_dim], got shape {observation.state.shape}"
        )

        rng_sample, rng_select = jax.random.split(rng)
        batch_size = observation.state.shape[0]

        if self.base_model is not None:
            # Sample from base model
            n = self.num_samples
            sample_rngs = jax.random.split(rng_sample, n)

            @eqx.filter_vmap(in_axes=(0, None, None, None))
            def sample_with_rng(rng_i, model, trans, next_action):
                return model.sample_actions(rng_i, trans, compute_next_action=next_action, **kwargs)

            all_actions = sample_with_rng(sample_rngs, self.base_model, transition, compute_next_action)
            # all_actions shape: [N, B, ah, ad]
            all_actions = jnp.moveaxis(all_actions, 0, 1)  # [B, N, ah, ad]
        else:
            # Use cached counterfactual actions
            all_actions = self._get_cached_actions(
                transition, compute_next_action=compute_next_action
            )  # [B, N, ah, ad]

        n = all_actions.shape[1]
        action_horizon = all_actions.shape[2]
        action_dim_size = all_actions.shape[3]

        expanded_obs = expand_observation(observation, n)
        flat_actions = all_actions.reshape(batch_size * n, action_horizon, action_dim_size)

        if self.use_target_value:
            q_values = value_function.compute_target_value(
                expanded_obs,
                flat_actions,
                take_min_over_ensemble=self.take_min_over_ensemble,
            )
        else:
            q_values = value_function.compute_value(
                expanded_obs,
                flat_actions,
                take_min_over_ensemble=self.take_min_over_ensemble,
            )
        q_values = q_values.reshape(batch_size, n)

        if self.selection_mode == "argmax":
            indices = jnp.argmax(q_values, axis=1)
        elif self.selection_mode == "softmax":
            logits = q_values / self.softmax_temperature
            indices = jax.random.categorical(rng_select, logits, axis=1)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        return all_actions[jnp.arange(batch_size), indices]

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Delegate loss computation to base model (enables BC pre-training)."""
        if self.base_model is None:
            batch_size = actions.shape[0]
            return jnp.zeros((batch_size, self.action_horizon))
        return self.base_model.compute_loss(rng, observation, actions, train=train)

    @override
    def action_distribution(
        self,
        rng: at.KeyArrayLike,
        transition: _base_vf.Transition,
        *,
        compute_next_action: bool = False,
        value_function: _base_vf.BaseValueFunction | _base_vf.BaseMultiValueFunction | None = None,
        **kwargs,
    ):
        """Return action distribution.

        If base_model exists, delegates to it. Otherwise, returns a deterministic
        distribution at the argmax action from cached counterfactual actions.
        """
        if self.base_model is not None:
            return self.base_model.action_distribution(
                rng, transition, compute_next_action=compute_next_action, **kwargs
            )

        # No base model: return deterministic distribution at best cached action
        import distrax

        best_action = self.sample_actions(
            rng, transition, compute_next_action=compute_next_action, value_function=value_function, **kwargs
        )
        batch_size = best_action.shape[0]
        return distrax.Deterministic(loc=best_action.reshape(batch_size, -1))


def expand_observation(observation: _model.Observation, num_samples: int) -> _model.Observation:
    """Repeat each observation num_samples times along the batch dimension.

    Given an observation with batch dimension B, produces an observation with
    batch dimension B*num_samples, where each original observation is repeated
    num_samples times contiguously.

    E.g., for B=2, N=3: [obs0, obs0, obs0, obs1, obs1, obs1]
    """

    def _repeat(x):
        if x is None:
            return None
        expanded = jnp.repeat(x[:, None], num_samples, axis=1)
        return expanded.reshape(x.shape[0] * num_samples, *x.shape[1:])

    return _model.Observation(
        images={k: _repeat(v) for k, v in observation.images.items()},
        image_masks={k: _repeat(v) for k, v in observation.image_masks.items()},
        state=_repeat(observation.state),
        tokenized_prompt=_repeat(observation.tokenized_prompt),
        tokenized_prompt_mask=_repeat(observation.tokenized_prompt_mask),
        token_ar_mask=_repeat(observation.token_ar_mask),
        token_loss_mask=_repeat(observation.token_loss_mask),
    )
