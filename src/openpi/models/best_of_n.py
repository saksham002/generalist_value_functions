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

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
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

    # Norm stats for renormalizing actions before passing to the critic.
    # Two valid modes:
    #   - both None: no renormalization (actions are already in critic space)
    #   - both provided: unnormalize from policy space, then normalize into critic space
    # Providing one without the other is not valid.
    policy_norm_stats: dict[str, NormStats] | None = None
    critic_norm_stats: dict[str, NormStats] | None = None

    # When True, converts cached counterfactual actions from chunk-wise-delta
    # format (delta[i] = global[i] - global[0]) to global by adding initial_pose
    # before normalizing into critic space.
    convert_to_global: bool = False

    # When the policy and critic have different action dimensions, slice the policy
    # action down to the critic's action dim before applying critic norm stats. The
    # slice covers indices [critic_action_dim_offset : critic_action_dim_offset + critic_action_dim],
    # where critic_action_dim is read from critic_norm_stats. Used e.g. when the
    # policy outputs a 32-d padded action and the critic only consumes the 14-d
    # EEF subset at offset 14. Must be set explicitly when dims differ.
    critic_action_dim_offset: int | None = None

    # Critic's expected action horizon. Hard-coded for the 60 Hz policy + 50-step critic
    # configuration: action_horizon must be 60 and critic_action_horizon must be 50.
    # The wrapper subsamples the policy chunk along the time axis (`[:, :, 1::2, :]` →
    # 30 steps), pads with zeros up to 50, and passes an action_mask of shape (B*N, 50)
    # with the first 30 entries True so the critic only attends to the real subsampled
    # actions. When None, the policy's action_horizon is forwarded to the critic unchanged.
    critic_action_horizon: int | None = None

    # Transform that converts chunk-wise-delta actions to absolute actions.
    # Required when convert_to_global=True; ignored otherwise. The transform's
    # __call__ runs in numpy/scipy (uses scipy.Rotation for rpy composition),
    # so it is invoked via jax.pure_callback inside _renormalize_actions.
    absolute_actions: _transforms.DataTransformFn | None = None

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
        if (self.policy_norm_stats is None) != (self.critic_norm_stats is None):
            raise ValueError(
                "policy_norm_stats and critic_norm_stats must both be provided or both be None. "
                f"Got policy_norm_stats={'set' if self.policy_norm_stats is not None else 'None'}, "
                f"critic_norm_stats={'set' if self.critic_norm_stats is not None else 'None'}."
            )

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
            convert_to_global=self.convert_to_global,
            selection_mode=self.selection_mode,
            softmax_temperature=self.softmax_temperature,
            policy_norm_stats=self.policy_norm_stats,
            critic_norm_stats=self.critic_norm_stats,
            critic_action_dim_offset=self.critic_action_dim_offset,
            critic_action_horizon=self.critic_action_horizon,
            absolute_actions=self.absolute_actions,
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
    convert_to_global: bool
    selection_mode: Literal["argmax", "softmax"]
    softmax_temperature: float
    policy_norm_stats: dict[str, NormStats] | None
    critic_norm_stats: dict[str, NormStats] | None
    critic_action_dim_offset: int | None
    critic_action_horizon: int | None
    absolute_actions: _transforms.DataTransformFn | None

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
        convert_to_global: bool = False,
        selection_mode: Literal["argmax", "softmax"],
        softmax_temperature: float,
        policy_norm_stats: dict[str, NormStats] | None = None,
        critic_norm_stats: dict[str, NormStats] | None = None,
        critic_action_dim_offset: int | None = None,
        critic_action_horizon: int | None = None,
        absolute_actions: _transforms.DataTransformFn | None = None,
    ):
        super().__init__(action_dim, action_horizon, max_token_len)
        self.base_model = base_model
        self.num_samples = num_samples
        self.take_min_over_ensemble = take_min_over_ensemble
        self.use_target_value = use_target_value
        self.convert_to_global = convert_to_global
        self.selection_mode = selection_mode
        self.softmax_temperature = softmax_temperature
        self.policy_norm_stats = policy_norm_stats
        self.critic_norm_stats = critic_norm_stats
        self.critic_action_dim_offset = critic_action_dim_offset
        if critic_action_horizon is not None and (action_horizon != 60 or critic_action_horizon != 50):
            raise ValueError(
                "critic_action_horizon is hard-coded to 50 with action_horizon hard-coded to 60. "
                f"Got critic_action_horizon={critic_action_horizon}, action_horizon={action_horizon}."
            )
        self.critic_action_horizon = critic_action_horizon
        if convert_to_global and absolute_actions is None:
            raise ValueError(
                "absolute_actions transform is required when convert_to_global=True."
            )
        self.absolute_actions = absolute_actions

    def _renormalize_actions(self, actions: at.Array, initial_pose: at.Array | None = None) -> at.Array:
        """Bring actions from policy normalized space into the critic's normalized action space.

        Steps:
        1. Optionally slice action dim to match critic (e.g. 32-d padded → 14-d EEF).
        2. Unnormalize from policy action space.
        3. Optionally convert delta actions to global using initial_pose.
        4. Normalize into critic action space.

        Both policy_norm_stats and critic_norm_stats must be set (enforced by config validation).
        """
        action_key = "actions"
        data = {action_key: actions}
        # Filter to just the "actions" key — Normalize/Unnormalize use strict=True and would
        # fail if norm_stats contains keys (e.g. "state") not present in data.
        policy_action_stats = {action_key: self.policy_norm_stats[action_key]}
        critic_action_stats = {action_key: self.critic_norm_stats[action_key]}
        # When the policy and critic have different action dims (e.g. 32-d padded
        # policy action vs 14-d EEF critic action), slice down to the critic's
        # action dim before applying any norm stats. policy_norm_stats for "actions"
        # are stored at the *unpadded* (critic) dim, so this slice must happen
        # before Unnormalize.
        critic_action_dim = critic_action_stats[action_key].mean.shape[-1]
        if data[action_key].shape[-1] != critic_action_dim:
            if self.critic_action_dim_offset is None:
                raise ValueError(
                    f"Action dim mismatch: input has {data[action_key].shape[-1]} but critic expects {critic_action_dim}. "
                    f"Set critic_action_dim_offset to specify which slice of the input to use."
                )
            start = self.critic_action_dim_offset
            data[action_key] = data[action_key][..., start : start + critic_action_dim]
        data = _transforms.Unnormalize(policy_action_stats)(data)
        if self.convert_to_global:
            assert initial_pose is not None, (
                "initial_pose is required when convert_to_global=True. Pass transition.action[:, :1, :] as initial_pose."
            )
            # AbsoluteActions runs in numpy/scipy (rpy composition uses scipy.Rotation),
            # so dispatch via pure_callback to keep the surrounding sample_actions JIT-able.
            actions_chunk = data[action_key]
            batch_size, num_samples, action_horizon, ad = actions_chunk.shape
            # Broadcast initial_pose [B, 1, ad] across the candidate axis and flatten the
            # leading [B, N] dims so AbsoluteActions sees a standard [batch, ah, ad] chunk.
            state_flat = jnp.broadcast_to(
                initial_pose[:, :, :], (batch_size, num_samples, ad)
            ).reshape(batch_size * num_samples, ad)
            actions_flat = actions_chunk.reshape(batch_size * num_samples, action_horizon, ad)

            def _absolute_callback(state_np, actions_np):
                return self.absolute_actions(
                    {"state": state_np, "actions": actions_np}
                )["actions"]

            absolute_flat = jax.pure_callback(
                _absolute_callback,
                jax.ShapeDtypeStruct(actions_flat.shape, actions_flat.dtype),
                state_flat,
                actions_flat,
            )
            data[action_key] = absolute_flat.reshape(batch_size, num_samples, action_horizon, ad)
        data = _transforms.Normalize(critic_action_stats)(data)
        return data[action_key]

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
        else:
            if transition.counterfactual_actions is None:
                raise ValueError(
                    "compute_next_action=False but transition.counterfactual_actions is None. "
                    "Ensure the data pipeline populates this field."
                )
            actions = transition.counterfactual_actions

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
            Tuple of:
              - selected actions of shape [batch, action_horizon, action_dim].
              - q_values of shape [batch, num_samples] (B always = 1 in the eval flow).
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

        # Renormalize actions into critic space if critic norm stats were provided.
        if self.critic_norm_stats is not None:
            initial_pose = transition.action[:, :1, :] if self.convert_to_global else None
            eval_actions = self._renormalize_actions(all_actions, initial_pose=initial_pose)
        else:
            eval_actions = all_actions

        # Subsample along the time axis and rebuild the action_mask when the critic
        # has its own action_horizon. The construction-time check enforces the only
        # supported pair: action_horizon=60, critic_action_horizon=50.
        # Subsample 1::2 → 30 real actions, pad with zeros up to 50, and feed an
        # action_mask of shape (B*N, 50) with the first 30 entries True so the
        # critic only attends to the real (non-padding) actions.
        critic_action_mask = None
        if self.critic_action_horizon is not None and self.critic_action_horizon != action_horizon:
            eval_actions = eval_actions[:, :, 1::2, :]
            real_len = eval_actions.shape[2]
            pad_len = self.critic_action_horizon - real_len
            padding = jnp.zeros(
                (batch_size, n, pad_len, eval_actions.shape[-1]),
                dtype = eval_actions.dtype,
            )
            eval_actions = jnp.concatenate([eval_actions, padding], axis = 2)
            action_horizon = self.critic_action_horizon
            mask_pattern = jnp.concatenate([
                jnp.ones(real_len, dtype = jnp.bool_),
                jnp.zeros(pad_len, dtype = jnp.bool_),
            ])
            critic_action_mask = jnp.broadcast_to(
                mask_pattern[None, :], (batch_size * n, action_horizon)
            )

        expanded_obs = expand_observation(observation, n)
        if critic_action_mask is not None:
            expanded_obs = dataclasses.replace(expanded_obs, action_mask = critic_action_mask)
        # eval_actions may have a different last dim than all_actions when the
        # critic consumes a sliced subset of the policy's action vector.
        flat_actions = eval_actions.reshape(batch_size * n, action_horizon, eval_actions.shape[-1])
        prefix_cache = None
        network = getattr(value_function, "q_network", getattr(value_function, "network", None))
        if network is not None and hasattr(network, "compute_prefix_cache"):
            raw_kv_cache, raw_prefix_mask = value_function.compute_prefix_cache(
                observation, use_target = self.use_target_value
            )
            repeated_kv_cache = jax.tree.map(
                lambda x: jnp.repeat(x, n, axis = 1), raw_kv_cache
            )
            repeated_prefix_mask = jnp.repeat(raw_prefix_mask, n, axis = 0)
            prefix_cache = (repeated_kv_cache, repeated_prefix_mask)

        value_kwargs = {
            "take_min_over_ensemble": self.take_min_over_ensemble,
        }
        if prefix_cache is not None:
            value_kwargs["prefix_cache"] = prefix_cache

        if self.use_target_value:
            result = value_function.compute_target_value(
                expanded_obs,
                flat_actions,
                **value_kwargs,
            )
        else:
            result = value_function.compute_value(
                expanded_obs,
                flat_actions,
                **value_kwargs,
            )
        q_values = result[0] if isinstance(result, tuple) else result
        q_values = q_values.reshape(batch_size, n)

        if self.selection_mode == "argmax":
            indices = jnp.argmax(q_values, axis=1)
        elif self.selection_mode == "softmax":
            logits = q_values / self.softmax_temperature
            indices = jax.random.categorical(rng_select, logits, axis=1)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        return all_actions[jnp.arange(batch_size), indices], q_values

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

        best_action, _ = self.sample_actions(
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
        action_mask=_repeat(observation.action_mask),
    )
