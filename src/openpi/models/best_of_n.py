"""Best-of-N sampling wrapper for value-guided action selection.

This module provides a wrapper around any BaseModel policy that samples N action
candidates and selects the best one according to a value function (Q-function)
passed at runtime. This enables value-guided policy improvement without modifying
the underlying policy architecture.
"""

import dataclasses
import os
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats

# Set OPENPI_DEBUG=1 in the environment to enable observation/action-stat logging
# during `BestOfNWrapper.sample_actions` and the policy-only path in
# `BestOfNPolicy._run_jit_inference`. Read once at module import.
_DEBUG: bool = os.environ.get("OPENPI_DEBUG", "0") == "1"
from openpi.value_functions import base_value_functions as _base_vf


def _log_model_inputs(
    tag: str,
    observation: _model.Observation,
    *,
    actions: jnp.ndarray | None = None,
) -> None:
    """Log key Observation fields (and optionally an action chunk) via jax.debug.print.

    Used to inspect what the policy and the critic actually see at sample time. Only the
    first element of the leading batch axis is shown to keep log lines compact. Fields
    that are None are skipped at trace time (not added to the JIT graph).
    """
    jax.debug.print(
        f"[debug] {tag} state shape={observation.state.shape} value={{v}}",
        v = observation.state[0],
    )
    if observation.action_mask is not None:
        jax.debug.print(
            f"[debug] {tag} action_mask shape={observation.action_mask.shape} value={{v}}",
            v = observation.action_mask[0],
        )
    if observation.tokenized_prompt is not None:
        jax.debug.print(
            f"[debug] {tag} tokenized_prompt shape={observation.tokenized_prompt.shape} value={{v}}",
            v = observation.tokenized_prompt[0],
        )
        if observation.tokenized_prompt_mask is not None:
            # Detokenize via host callback — PaligemmaTokenizer is numpy-only and
            # can't run under jit. Lazy-import to avoid pulling SentencePiece into
            # the import path of every BestOfN consumer.
            from openpi.robocoin_utils.utils import detokenize_prompt

            def _log_detokenized(token_ids, mask):
                print(
                    f"[debug] {tag} untokenized_prompt: "
                    f"{detokenize_prompt(token_ids, mask)!r}"
                )

            jax.debug.callback(
                _log_detokenized,
                observation.tokenized_prompt[0],
                observation.tokenized_prompt_mask[0],
            )
    if observation.tokenized_prompt_mask is not None:
        jax.debug.print(
            f"[debug] {tag} tokenized_prompt_mask shape={observation.tokenized_prompt_mask.shape} value={{v}}",
            v = observation.tokenized_prompt_mask[0],
        )
    for cam_name, img in observation.images.items():
        jax.debug.print(
            f"[debug] {tag} image[{cam_name}] shape={img.shape} min={{mn:.4f}} max={{mx:.4f}}",
            mn = jnp.min(img[0]), mx = jnp.max(img[0]),
        )
    for mask_name, mask in observation.image_masks.items():
        jax.debug.print(
            f"[debug] {tag} image_mask[{mask_name}] shape={mask.shape} value={{v}}",
            v = mask[0],
        )
    if actions is not None:
        # Print the full action chunk for the first batch element.
        chunk = actions[0]
        jax.debug.print(
            f"[debug] {tag} actions[0] shape={chunk.shape} value={{v}}",
            v = chunk,
        )


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

    # Normalization mode flags for each side. Mirror `data_config.use_quantile_norm`
    # for the policy and critic respectively, so that `_renormalize_actions` /
    # `_renormalize_state` invert and re-apply the correct transform on each end.
    policy_use_quantile_norm: bool = False
    critic_use_quantile_norm: bool = False

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
            policy_use_quantile_norm=self.policy_use_quantile_norm,
            critic_use_quantile_norm=self.critic_use_quantile_norm,
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
    policy_use_quantile_norm: bool
    critic_use_quantile_norm: bool
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
        policy_use_quantile_norm: bool = False,
        critic_use_quantile_norm: bool = False,
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
        self.policy_use_quantile_norm = policy_use_quantile_norm
        self.critic_use_quantile_norm = critic_use_quantile_norm
        self.critic_action_dim_offset = critic_action_dim_offset
        # Three supported (action_horizon, critic_action_horizon) regimes:
        #   - (60, 50): 60 Hz policy + 30 Hz critic. Stride-2 subsample
        #     (60 -> 30), then zero-pad to 50.
        #   - (30, 50): same-fps policy + critic. Skip subsample, zero-pad
        #     30 -> 50.
        #   - (50, 50): matched-horizon policy + critic. No subsample, no
        #     pad — the chunk is forwarded as-is to the critic. The 30 / 20
        #     valid / invalid split is carried entirely by `action_mask`.
        if critic_action_horizon is not None and not (
            critic_action_horizon == 50 and action_horizon in (30, 50, 60)
        ):
            raise ValueError(
                "Only (action_horizon=30, critic_action_horizon=50), "
                "(action_horizon=50, critic_action_horizon=50), "
                "and (action_horizon=60, critic_action_horizon=50) are supported. "
                f"Got action_horizon={action_horizon}, critic_action_horizon={critic_action_horizon}."
            )
        self.critic_action_horizon = critic_action_horizon
        if convert_to_global and absolute_actions is None:
            raise ValueError(
                "absolute_actions transform is required when convert_to_global=True."
            )
        self.absolute_actions = absolute_actions

    def _unnormalize_policy_actions(
        self, actions: at.Array, initial_pose: at.Array | None = None,
    ) -> at.Array:
        """Slice action_dim, undo policy normalization, optionally convert to global.

        Returns actions in raw policy-output space (e.g. metric eef_pos / radian
        eef_rot / [-1,1] gripper). The caller is then responsible for adapting
        the horizon (subsample / pad) before applying `_normalize_critic_actions`.

        Splitting this out from the original `_renormalize_actions` is required
        when the policy and critic use different action_horizons and per-position
        chunk-wise-delta norm-stats: critic stats have shape
        (critic_action_horizon, action_dim), so `Normalize(critic_stats)` cannot
        be applied to a chunk that's still at the policy's horizon. The horizon
        adapt has to happen between Unnormalize-with-policy-stats and
        Normalize-with-critic-stats.
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
        data = _transforms.Unnormalize(policy_action_stats, use_quantiles = self.policy_use_quantile_norm)(data)
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
        return data[action_key]

    def _normalize_critic_actions(self, actions: at.Array) -> at.Array:
        """Apply Normalize-with-critic-stats. Caller must have already adapted
        the horizon to `critic_action_horizon` (or skipped if horizons match)."""
        action_key = "actions"
        critic_action_stats = {action_key: self.critic_norm_stats[action_key]}
        data = _transforms.Normalize(
            critic_action_stats, use_quantiles = self.critic_use_quantile_norm,
        )({action_key: actions})
        return data[action_key]

    def _renormalize_state(self, state: at.Array) -> at.Array:
        """Bring state from policy normalized space into the critic's normalized state space."""
        state_key = "state"
        data = {state_key: state}
        policy_state_stats = {state_key: self.policy_norm_stats[state_key]}
        critic_state_stats = {state_key: self.critic_norm_stats[state_key]}
        data = _transforms.Unnormalize(policy_state_stats, use_quantiles = self.policy_use_quantile_norm)(data)
        data = _transforms.Normalize(critic_state_stats, use_quantiles = self.critic_use_quantile_norm)(data)
        return data[state_key]

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
        critic_tokenized_prompt: at.Array | None = None,
        critic_tokenized_prompt_mask: at.Array | None = None,
        sample_rngs: at.KeyArrayLike | None = None,
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
            critic_tokenized_prompt: Optional [B, T] int prompt tokens produced by the
                critic's own tokenizer. When provided, overrides expanded_obs.tokenized_prompt
                before the critic forward so the critic sees its in-distribution token IDs.
            critic_tokenized_prompt_mask: Matching [B, T] bool mask for critic_tokenized_prompt.
            sample_rngs: Optional pre-split rng array of shape (N, ...). If provided, used
                as the per-sample rngs directly (skipping the in-JIT `jax.random.split` of
                `rng`). Intended for the multi-host serve path: the caller hands in a
                DATA_AXIS-sharded rng array so each chip gets a unique rng (and therefore
                unique noise) without the cross-host all-gather collapsing per-host
                divergence into a "Frankenstein" tensor. Pass None for the legacy
                same-rng-per-call path (used by training / single-host eval).
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
            if _DEBUG:
                _log_model_inputs("policy", observation)
                # Probe a representative leaf of the underlying pi0/pi05 base model
                # so we can verify the restored params on the server match the
                # diagnostic's restored params bit-for-bit.
                if hasattr(self.base_model, "action_out_proj"):
                    aop_kernel = self.base_model.action_out_proj.kernel.value
                    aop_bias = self.base_model.action_out_proj.bias.value
                    jax.debug.print(
                        "[debug] base_model.action_out_proj/kernel shape={s} dtype={d} "
                        "mean={mn:.6e} std={st:.6e} min={mi:.6e} max={mx:.6e} sum={sm:.6e} "
                        "first8={f}",
                        s = aop_kernel.shape, d = aop_kernel.dtype,
                        mn = jnp.mean(aop_kernel.astype(jnp.float32)),
                        st = jnp.std(aop_kernel.astype(jnp.float32)),
                        mi = jnp.min(aop_kernel.astype(jnp.float32)),
                        mx = jnp.max(aop_kernel.astype(jnp.float32)),
                        sm = jnp.sum(aop_kernel.astype(jnp.float32)),
                        f = aop_kernel.reshape((-1,))[:8].astype(jnp.float32),
                    )
                    jax.debug.print(
                        "[debug] base_model.action_out_proj/bias shape={s} dtype={d} "
                        "mean={mn:.6e} std={st:.6e} sum={sm:.6e} value={v}",
                        s = aop_bias.shape, d = aop_bias.dtype,
                        mn = jnp.mean(aop_bias.astype(jnp.float32)),
                        st = jnp.std(aop_bias.astype(jnp.float32)),
                        sm = jnp.sum(aop_bias.astype(jnp.float32)),
                        v = aop_bias.astype(jnp.float32),
                    )
            # Two paths for picking the per-sample rngs that get fed to the
            # vmap'd policy.sample_actions call:
            #   - sample_rngs is None → legacy path: split `rng_sample` into
            #     ceil(num_samples / num_workers) keys per worker. Same rng on
            #     every host, so every host computes the same per-worker n
            #     candidates (no cross-worker diversity). Used by training /
            #     single-host eval.
            #   - sample_rngs is not None → caller hands in a sharded array of
            #     shape (N, ...) where N is leading-axis-sharded across the
            #     mesh's DATA_AXIS. Each chip gets a unique rng → unique noise.
            #     Used by the multi-host serve path to fan out diverse noise
            #     per chip without triggering the cross-host all-gather
            #     "Frankenstein" failure mode.
            if sample_rngs is None:
                num_workers = jax.process_count()
                n = max(1, -(-self.num_samples // num_workers))
                sample_rngs = jax.random.split(rng_sample, n)

            # Match main's vmap pattern: use `eqx.filter_vmap` with explicit
            # `in_axes=(0, None, None, None)`, passing `model`, `trans`, and
            # `next_action` as positional args (broadcast, not vmapped). This
            # avoids closing the NNX module / transition over `jax.vmap` —
            # which (under the previous `jax.vmap(sample_with_rng)(sample_rngs)`
            # closure pattern) was producing wrong outputs on the live server.
            @eqx.filter_vmap(in_axes = (0, None, None, None))
            def sample_with_rng(rng_i, model, trans, next_action):
                return model.sample_actions(
                    rng_i, trans, compute_next_action = next_action, **kwargs,
                )

            all_actions = sample_with_rng(sample_rngs, self.base_model, transition, compute_next_action)
            # all_actions shape: [N, B, ah, ad]
            all_actions = jnp.moveaxis(all_actions, 0, 1)  # [B, N, ah, ad]
            if _DEBUG:
                # Raw policy output in policy-normalized space, before the
                # critic-side renormalize / horizon-pad / action-dim slice.
                jax.debug.print(
                    f"[debug] policy raw_actions shape={all_actions.shape} sample0={{v}}",
                    v = all_actions[0, 0],
                )
        else:
            # Use cached counterfactual actions
            all_actions = self._get_cached_actions(
                transition, compute_next_action=compute_next_action
            )  # [B, N, ah, ad]

        n = all_actions.shape[1]
        action_horizon = all_actions.shape[2]

        # Bring actions into critic space. The order is deliberate:
        #   (1) Unnormalize policy stats — requires actions at policy_action_horizon
        #       since policy norm-stats are per-position, shape (policy_ah, ad).
        #   (2) Adapt horizon (subsample / pad) to critic_action_horizon.
        #   (3) Normalize critic stats — requires actions at critic_action_horizon
        #       since critic norm-stats are per-position, shape (critic_ah, ad).
        # Doing (3) before (2) was the prior bug (asserted on shape mismatch
        # between policy-horizon actions and critic-horizon stats).
        if self.critic_norm_stats is not None:
            initial_pose = transition.action[:, :1, :] if self.convert_to_global else None
            eval_actions = self._unnormalize_policy_actions(
                all_actions, initial_pose = initial_pose,
            )
        else:
            eval_actions = all_actions

        # Adapt the policy chunk to the critic's expected horizon. Two regimes are
        # supported (validated in __init__):
        #   - 60 Hz policy + 30 Hz critic (action_horizon=60, critic_action_horizon=50):
        #     stride-2 subsample (60 → 30 real actions), then zero-pad to 50,
        #     matching the fps_mask_30 mask the critic was trained on.
        #   - same-fps policy + critic (action_horizon=30, critic_action_horizon=50):
        #     skip the subsample, just zero-pad 30 → 50.
        # In both cases the action_mask is also subsampled (when applicable) and
        # zero-padded with False so the critic only attends to the real actions.
        critic_action_mask = None
        if self.critic_action_horizon is not None and self.critic_action_horizon != action_horizon:
            do_subsample = 6 * self.critic_action_horizon == 5 * action_horizon
            if do_subsample:
                subsampled_actions = eval_actions[:, :, 1::2, :]
            else:
                subsampled_actions = eval_actions
            subsampled_horizon = subsampled_actions.shape[2]
            pad_amount = self.critic_action_horizon - subsampled_horizon
            eval_actions = jnp.pad(
                subsampled_actions,
                ((0, 0), (0, 0), (0, pad_amount), (0, 0)),
            )

            if observation.action_mask is not None:
                subsampled_mask = (
                    observation.action_mask[:, 1::2] if do_subsample else observation.action_mask
                )
            else:
                subsampled_mask = jnp.ones((batch_size, subsampled_horizon), dtype = jnp.bool_)
            subsampled_mask_expanded = jnp.repeat(
                subsampled_mask[:, None, :], n, axis = 1,
            ).reshape(batch_size * n, subsampled_horizon)
            critic_action_mask = jnp.pad(
                subsampled_mask_expanded,
                ((0, 0), (0, pad_amount)),
                constant_values = False,
            )

            action_horizon = self.critic_action_horizon

        # (3) Apply Normalize-with-critic-stats now that the horizon matches.
        if self.critic_norm_stats is not None:
            eval_actions = self._normalize_critic_actions(eval_actions)

        # Renormalize state into critic space before expanding (one renorm vs. N redundant ones).
        # Policy already sampled above using `observation` in its own normalized space.
        critic_observation = observation
        if self.critic_norm_stats is not None:
            critic_state = self._renormalize_state(observation.state)
            critic_observation = dataclasses.replace(observation, state = critic_state)

        expanded_obs = expand_observation(critic_observation, n)
        if critic_action_mask is not None:
            expanded_obs = dataclasses.replace(expanded_obs, action_mask = critic_action_mask)
        if critic_tokenized_prompt is not None:
            # Broadcast critic-tokenized prompt over the candidate axis: [B, T] -> [B*N, T].
            critic_token_len = critic_tokenized_prompt.shape[-1]
            critic_prompt_expanded = jnp.broadcast_to(
                critic_tokenized_prompt[:, None, :], (batch_size, n, critic_token_len),
            ).reshape(batch_size * n, critic_token_len)
            critic_prompt_mask_expanded = jnp.broadcast_to(
                critic_tokenized_prompt_mask[:, None, :], (batch_size, n, critic_token_len),
            ).reshape(batch_size * n, critic_token_len)
            expanded_obs = dataclasses.replace(
                expanded_obs,
                tokenized_prompt = critic_prompt_expanded,
                tokenized_prompt_mask = critic_prompt_mask_expanded,
            )
        # actions to evaluate may have a different last dim than the policy actions when the
        # critic consumes a sliced subset of the policy's action vector.
        flat_actions = eval_actions.reshape(batch_size * n, action_horizon, eval_actions.shape[-1])

        if _DEBUG:
            _log_model_inputs("critic", expanded_obs, actions = flat_actions)

        # Critic prefix encodes images + prompt + state KVs. The critic was
        # trained on its OWN tokenized_prompt (different buffer length / vocab
        # than the policy's) and on critic-normalized state, so using `observation`
        # here would silently bake the policy's tokens + policy-normalized state
        # into the cached prefix and leave the critic off-distribution.
        prefix_observation = critic_observation
        if critic_tokenized_prompt is not None:
            prefix_observation = dataclasses.replace(
                prefix_observation,
                tokenized_prompt = critic_tokenized_prompt,
                tokenized_prompt_mask = critic_tokenized_prompt_mask,
            )

        prefix_cache = None
        network = getattr(value_function, "q_network", getattr(value_function, "network", None))
        # The sibling repo's value_function classes expose a top-level
        # `compute_prefix_cache` that delegates to the underlying network. The
        # in-repo SARSAValueFunction doesn't expose that wrapper, so checking
        # only the network's attribute and then calling on `value_function`
        # (which is what the sibling code does) raises AttributeError. Guard
        # both: only enter this branch when both the network supports the
        # KV-cache fast path AND the value_function exposes the wrapper call.
        if (
            network is not None
            and hasattr(network, "compute_prefix_cache")
            and hasattr(value_function, "compute_prefix_cache")
        ):
            raw_kv_cache, raw_prefix_mask = value_function.compute_prefix_cache(
                prefix_observation, use_target = self.use_target_value
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

        # Selection happens in the caller (host-side) so multi-host can
        # process_allgather + argmax across workers.
        return all_actions, q_values

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
