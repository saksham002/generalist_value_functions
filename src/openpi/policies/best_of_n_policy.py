"""Policy wrapper that loads a RoboCasa policy plus optional BestOfN critic.

Mirrors the construction + sampling flow of `LocalPolicy` from the sibling repo
(`batch_value_learning_eval/eval/xarm_scripts/eval_shirt_hang_remote.py`), but
exposes a `Policy`-compatible `infer(obs)` interface so that the existing
`WebsocketPolicyServer` / `examples/robocasa/main.py` client pair work
unchanged when the server is launched with `--critic.*` flags.

Two modes:

- **Policy-only**: only `policy_*` args provided. The wrapper still loads the
  policy through `openpi.robocoin_utils.load_model_utils.load_policy` and runs
  the same RoboCasa-style explicit slice + output-transform path the sibling
  uses, so action-dim padding (32-D model output → 14-D bimanual EEF slot)
  is unwrapped consistently with the BestOfN path.
- **Best-of-N**: `critic_*` args also provided. Builds a Wrapper around BestOfN
  around the loaded policy, JITs the BestOfN sample closure, and re-tokenizes
  the prompt with the critic's tokenizer at infer time. The policy and critic
  share norm stats + chunk-wise-delta format, so the policy's normalized
  output is scored by the critic directly (no renormalization).

The shape of the returned `infer(obs)` dict is `{"actions": np.ndarray,
"policy_timing": {"infer_ms": float}, "q_values": np.ndarray | None}`. The
RoboCasa client only reads `"actions"`, so `"q_values"` is purely additive.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Literal

import flax.nnx as nnx
import jax
from jax.experimental import multihost_utils as _multihost
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as _transforms
from openpi.models.best_of_n import _DEBUG
from openpi.models.best_of_n import BestOfNWrapper
from openpi.models.best_of_n import _log_model_inputs
from openpi.policies import policy as _policy_module
from openpi.policies import robocasa_policy as _robocasa_policy
from openpi.robocoin_utils.load_model_utils import LoadPolicyConfig
from openpi.robocoin_utils.load_model_utils import load_critic as _load_critic
from openpi.robocoin_utils.load_model_utils import load_policy as _load_policy
from openpi.shared.normalize import NormStats
from openpi.training import config as _config
from openpi.value_functions import base_value_functions as _base_vf

logger = logging.getLogger(__name__)

# Norm-stat keys the policy's Normalize / Unnormalize transforms reference at
# inference time. Mirrors LocalPolicy in the sibling repo.
_INFERENCE_NORM_KEYS = {"state", "actions", "next_state", "next_actions"}


# Keys whose dtype `broadcast_one_to_all` up-promotes during host transfer
# (e.g. uint8 -> uint32, bool -> int32) and that we have to cast back to keep
# Pi0 / BestOfN's jaxtyping annotations happy. Mapping is target-dtype.
_BROADCAST_RESTORE_DTYPES: dict[str, Any] = {
    # Observation fields
    "image": jnp.uint8,
    "image_mask": jnp.bool_,
    "tokenized_prompt_mask": jnp.bool_,
    "action_mask": jnp.bool_,
    "token_ar_mask": jnp.bool_,
    "token_loss_mask": jnp.bool_,
    # Critic-side extras
    "critic_token_mask": jnp.bool_,
}


def _restore_inference_dtypes(d: dict[str, Any]) -> dict[str, Any]:
    """Cast known fields back to their expected dtypes after a broadcast."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        target = _BROADCAST_RESTORE_DTYPES.get(k)
        if target is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {dk: dv.astype(target) if hasattr(dv, "astype") else dv for dk, dv in v.items()}
        elif hasattr(v, "astype"):
            out[k] = v.astype(target)
        else:
            out[k] = v
    return out


def _compute_acs(actions: np.ndarray) -> float:
    """Average cosine similarity across a BestOfN candidate action pool.

    `actions` is the (n, ah, ad) pool of candidate action chunks. For every
    unordered pair of the n candidates, the per-timestep cosine similarity
    between the two chunks is averaged across the ah horizon; those C(n, 2)
    pair scalars are then averaged into a single value. Returns NaN when n < 2.
    """
    n = actions.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(actions, axis = -1, keepdims = True)
    unit = actions / np.clip(norms, 1e-8, None)
    # per_timestep_cos[i, j, t] = <unit[i, t], unit[j, t]>
    per_timestep_cos = np.einsum("itd,jtd->ijt", unit, unit)
    cos_per_pair = per_timestep_cos.mean(axis = -1)  # (n, n): mean over horizon
    upper = np.triu_indices(n, k = 1)  # unique unordered pairs
    return float(cos_per_pair[upper].mean())


def _build_critic_kwargs(critic_config: Any) -> dict[str, Any]:
    """Pack the critic-side attributes BestOfNWrapper needs from the TrainConfig.

    Mirrors `eval_shirt_hang_remote.load_critic`'s critic_kwargs assembly so the
    BestOfNWrapper construction can stay close to the sibling's wiring. Uses
    `getattr` for `use_quantile_norm` because the RLDSRoboCasaDataConfig FT
    factory doesn't expose that field (it's hardcoded to False, matching the
    default), whereas RoboCoinRldsDataConfig used by the pretrain config does.
    """
    return {
        # FineTuneConfig overrides land on TrainConfig.action_horizon; fall back
        # to the model's action_horizon when the TrainConfig field is unset.
        "action_horizon": critic_config.action_horizon or critic_config.model.action_horizon,
        "use_chunk_wise_delta": critic_config.data.use_chunk_wise_delta,
        "subsample": getattr(critic_config.data, "subsample", False),
        "use_quantile_norm": getattr(critic_config.data, "use_quantile_norm", False),
        "tokenizer": critic_config.model.network_config.get_tokenizer(),
    }


class BestOfNPolicy(_base_policy.BasePolicy):
    """Loads a policy (and optional critic), serves an `infer(obs)` API.

    Construction matches LocalPolicy in
    `batch_value_learning_eval/eval/xarm_scripts/eval_shirt_hang_remote.py`,
    adapted to the websocket `infer(obs)` interface.
    """

    def __init__(
        self,
        policy_config_name: str,
        policy_checkpoint_dir: str,
        *,
        policy_step: int | None = None,
        policy_fine_tune_config: str | None = None,
        policy_task_description: str | None = None,
        critic_config_name: str | None = None,
        critic_checkpoint_dir: str | None = None,
        critic_step: int | None = None,
        critic_fine_tune_config: str | None = None,
        num_samples: int = 8,
        take_min_over_ensemble: bool = True,
        selection_mode: Literal["argmax", "softmax"] = "argmax",
        softmax_temperature: float = 1.0,
        critic_action_dim_offset: int | None = None,
        default_prompt: str | None = None,
        return_q_values: bool = True,
        prewarm: bool = True,
    ) -> None:
        # Both critic-config and critic-checkpoint must be provided together.
        critic_args_set = (critic_config_name is not None) or (critic_checkpoint_dir is not None)
        if critic_args_set and (critic_config_name is None or critic_checkpoint_dir is None):
            raise ValueError(
                "BestOfNPolicy: critic_config_name and critic_checkpoint_dir must be "
                "provided together (both or neither)."
            )
        self._return_q_values = return_q_values

        # ---- Load policy + build the per-RoboCasa Policy with full transforms.
        logger.info(f"Loading policy '{policy_config_name}' from {policy_checkpoint_dir} (step={policy_step})...")
        load_config = LoadPolicyConfig(
            config_name = policy_config_name,
            checkpoint_path = policy_checkpoint_dir,
            fine_tune = policy_fine_tune_config,
            step = policy_step,
        )
        model, config = _load_policy(load_config)
        logger.info("Policy checkpoint restored.")

        # The policy model lives on `config.policy` for IQL/SARSA-style
        # multi-component configs, otherwise on `config.model` directly.
        policy_model_config = config.policy if config.policy is not None else config.model

        data_config = config.data.create(config.assets_dirs, policy_model_config)
        asset_id = data_config.asset_id
        if policy_step is not None:
            step_dir = str(policy_step)
        else:
            step_dirs = sorted(
                (d for d in os.listdir(policy_checkpoint_dir) if d.isdigit()),
                key = int,
            )
            step_dir = step_dirs[-1]
            logger.info(f"No policy step specified, using latest step {step_dir}.")
        norm_stats_dir = os.path.join(policy_checkpoint_dir, step_dir, "assets", asset_id)
        logger.info(f"Loading policy norm stats from {norm_stats_dir}")
        all_norm_stats = _normalize.load(norm_stats_dir)
        norm_stats = {k: v for k, v in all_norm_stats.items() if k in _INFERENCE_NORM_KEYS}

        policy = _policy_module.Policy(
            model,
            transforms = [
                _transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
                *(
                    [_transforms.Clip(data_config.clip_normalized_bounds)]
                    if data_config.clip_normalized_bounds is not None
                    else []
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms = [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles = data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            metadata = config.policy_metadata,
        )

        # Borrow the prepared transform pipeline + model from the Policy
        # instance (mirrors LocalPolicy). This avoids reimplementing the
        # transform composition while still letting us slice the action_dim
        # padding before Unnormalize the way the sibling does.
        self._model = policy._model  # noqa: SLF001
        self._input_transform = policy._input_transform  # noqa: SLF001
        self._output_transform = policy._output_transform  # noqa: SLF001
        self._sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
        self._rng = policy._rng  # noqa: SLF001
        self._metadata = policy.metadata

        # Policy.__init__ skips this on the JAX path; force eval-mode here.
        self._model.eval()

        # Disable classifier-free guidance at eval time (matches LocalPolicy).
        if hasattr(self._model, "config") and hasattr(self._model.config, "guidance"):
            object.__setattr__(self._model.config, "guidance", 0.0)
            logger.info("Disabled classifier-free guidance (guidance=0.0).")

        # Wrap sample_actions so the debug print runs INSIDE the JIT graph.
        # Outside the JIT the actions tensor is multi-host sharded with empty
        # shards on some processes, and `jax.debug.print` fails on `device_put`
        # of a non-fully-addressable array. Inside the graph `jax.debug.print`
        # handles sharded arrays correctly (same as the BestOfN.debug path in
        # best_of_n.py). Mirrors the `_bestofn_sample` pattern: pass the model
        # as a positional argument to `@nnx.jit` rather than capture by closure
        # so the NNX state is sharded correctly.
        if _DEBUG:
            @nnx.jit
            def _sample_with_debug(model, rng_in, transition_in, noise = None):
                # ---- DIAGNOSTIC: fingerprints INSIDE the JIT graph for the
                # actual model inputs. Lets us detect any divergence between
                # V9 and V4 paths at the MODEL boundary even if the batched
                # dict was bit-identical at the host-side boundary.
                obs_in = transition_in.observation
                jax.debug.print(
                    "[mdl.debug] state sum={s:.6e}  rng={r}",
                    s = jnp.sum(obs_in.state.astype(jnp.float64)),
                    r = jax.random.key_data(rng_in),
                )
                # Image is (1, 224, 224, 3) → flatten size = 150528. Offsets
                # below all fall within [0, 150527]. Two images with same sum
                # but different content will diverge in these specific pixel
                # values.
                _img_offsets = jnp.array([0, 100, 1000, 10000, 50000, 100000, 140000, 150527])
                for cam in sorted(obs_in.images.keys()):
                    img_f64 = obs_in.images[cam].astype(jnp.float64)
                    img_flat = img_f64.reshape(-1)
                    jax.debug.print(
                        "[mdl.debug] image[{c}] sum={s:.6e}  shape={sh}",
                        c = cam,
                        s = jnp.sum(img_f64),
                        sh = obs_in.images[cam].shape,
                    )
                    jax.debug.print(
                        "[mdl.debug] image[{c}] pixels @ [0,100,1000,10000,50000,100000,140000,150527] = {v}",
                        c = cam,
                        v = img_flat[_img_offsets],
                    )
                if obs_in.tokenized_prompt is not None:
                    jax.debug.print(
                        "[mdl.debug] tokenized_prompt sum={s}  shape={sh}",
                        s = jnp.sum(obs_in.tokenized_prompt),
                        sh = obs_in.tokenized_prompt.shape,
                    )
                if obs_in.action_mask is not None:
                    jax.debug.print(
                        "[mdl.debug] action_mask sum={s}  shape={sh}",
                        s = jnp.sum(obs_in.action_mask.astype(jnp.int32)),
                        sh = obs_in.action_mask.shape,
                    )
                # action_out_proj kernel fingerprint inside JIT — proves whether
                # the loaded params' state differs between V9 and V4 calls.
                if hasattr(model, "action_out_proj"):
                    jax.debug.print(
                        "[mdl.debug] action_out_proj/kernel sum={s:.6e}  bias_sum={b:.6e}",
                        s = jnp.sum(model.action_out_proj.kernel.value.astype(jnp.float64)),
                        b = jnp.sum(model.action_out_proj.bias.value.astype(jnp.float64)),
                    )
                if hasattr(model, "action_in_proj"):
                    jax.debug.print(
                        "[mdl.debug] action_in_proj/kernel sum={s:.6e}  bias_sum={b:.6e}",
                        s = jnp.sum(model.action_in_proj.kernel.value.astype(jnp.float64)),
                        b = jnp.sum(model.action_in_proj.bias.value.astype(jnp.float64)),
                    )
                # Aggregate fingerprints over the entire NNX state — if these
                # differ between V9 and V4 calls despite the same model object,
                # that's the bug. Use abs-sum + sum-of-squares to catch sign-
                # cancelling differences. Only iterate float-leaves with
                # `.value`.
                _state_leaves = [
                    v.value for v in jax.tree.leaves(nnx.state(model))
                    if hasattr(v, "value") and hasattr(v.value, "shape")
                    and jnp.issubdtype(v.value.dtype, jnp.floating)
                ]
                jax.debug.print(
                    "[mdl.debug] full state: #leaves={n}  abs_sum={a:.6e}  sq_sum={s:.6e}",
                    n = len(_state_leaves),
                    a = sum(jnp.sum(jnp.abs(x.astype(jnp.float64))) for x in _state_leaves),
                    s = sum(jnp.sum(x.astype(jnp.float64) * x.astype(jnp.float64)) for x in _state_leaves),
                )

                # ALSO fingerprint the FULL transition (not just observation
                # fields we logged). Catches discrepancies in transition.action,
                # transition.next_observation, transition.next_action, etc.
                _trans_leaves = [
                    leaf for leaf in jax.tree.leaves(transition_in)
                    if hasattr(leaf, "shape") and jnp.issubdtype(leaf.dtype, jnp.floating)
                ]
                _trans_int_leaves = [
                    leaf for leaf in jax.tree.leaves(transition_in)
                    if hasattr(leaf, "shape") and (jnp.issubdtype(leaf.dtype, jnp.integer)
                                                   or jnp.issubdtype(leaf.dtype, jnp.bool_))
                ]
                jax.debug.print(
                    "[mdl.debug] transition: #float_leaves={nf}  abs_sum={a:.6e}  #int_leaves={ni}  int_sum={i}",
                    nf = len(_trans_leaves),
                    a = sum(jnp.sum(jnp.abs(x.astype(jnp.float64))) for x in _trans_leaves),
                    ni = len(_trans_int_leaves),
                    i = sum(jnp.sum(x.astype(jnp.int64)) for x in _trans_int_leaves),
                )
                actions = (
                    model.sample_actions(rng_in, transition_in, noise = noise)
                    if noise is not None
                    else model.sample_actions(rng_in, transition_in)
                )
                jax.debug.print(
                    f"[debug] policy raw_actions shape={actions.shape} sample0={{v}}",
                    v = actions[0],
                )
                return actions

            def _sample_jit_wrapper(rng_in, transition_in, **sample_kwargs):
                return _sample_with_debug(
                    self._model, rng_in, transition_in,
                    noise = sample_kwargs.get("noise"),
                )

            self._sample_actions_jit = _sample_jit_wrapper
        else:
            self._sample_actions_jit = nnx_utils.module_jit(self._model.sample_actions)

        # action_dim_offset + eef_action_dim define the slice from the model's
        # 32-D padded output back to the 14-D bimanual EEF block. norm_stats
        # for "actions" is stored at the unpadded (14) dim.
        self._action_dim_offset = policy_model_config.action_dim_offset
        self._eef_action_dim = norm_stats["actions"].mean.shape[-1]
        # State dim for the prewarm dummy (norm_stats["state"] is the
        # authoritative shape; fall back to the action dim if absent).
        self._state_dim = norm_stats["state"].mean.shape[-1]
        # Set in the critic branch below: True when the critic's data factory
        # is the HDF5 pipeline (sim_bimanual_assembly etc.), False for the
        # RoboCasa/RoboCoin pipeline. Drives the prewarm dummy obs schema.
        self._critic_is_hdf5 = False

        # JAX rank 0 binds the websocket; every other rank runs
        # participate_loop so the JIT'd inference doesn't deadlock waiting
        # for collectives. Rank → TPU worker IP mapping is deterministic
        # per pod; serve_policy.py logs it at startup.
        self._process_index = jax.process_index()
        self._num_processes = jax.process_count()
        self._is_multi_host = self._num_processes > 1
        self._inference_iter = 0  # bumped in lockstep on every host

        # Mesh used for the FSDP-sharded JIT call. Re-built from `config.fsdp_devices`
        # which load_policy pins to 16 — same mesh the policy params were restored
        # onto. Used below to declare the per-sample rngs as DATA_AXIS-sharded so
        # each chip gets a unique noise tensor.
        from openpi.training import sharding as _sharding_mod
        self._sharding_mod = _sharding_mod
        self._mesh = _sharding_mod.make_mesh(config.fsdp_devices)
        # Pad num_samples to a multiple of jax.device_count() so a single
        # PartitionSpec(DATA_AXIS, None) cleanly splits the leading axis across
        # all chips (1 unique sample per chip). User's num_samples is the lower
        # bound; the critic ends up scoring `num_samples_padded` candidates and
        # we argmax over all of them downstream.
        device_count = jax.device_count()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self._num_samples_padded = ((num_samples + device_count - 1) // device_count) * device_count
        if self._num_samples_padded != num_samples:
            logger.info(
                "BestOfNPolicy: padding num_samples %d -> %d (next multiple of device_count=%d) "
                "so the per-sample rngs cleanly shard along DATA_AXIS.",
                num_samples, self._num_samples_padded, device_count,
            )

        # Cache the obs schema (shapes / dtypes) so participating ranks can
        # construct dummies of the exact structure broadcast_one_to_all expects.
        # PaligemmaTokenizer used by RoboCasa configs writes max_token_len at
        # construction time; we just read it off the model.
        self._max_token_len = self._model.max_token_len
        self._policy_task_description = policy_task_description
        # Critic-side prompt override; set below after critic load when the
        # critic's data factory uses prompt_mode="task_description_predict_current_subtask"
        # (training prompt = constant task description; the subtask text is only
        # used for the auxiliary next-token loss, not for the input prompt).
        self._critic_task_description: str | None = None
        self._image_size = 224  # RoboCasa eval transforms always emit 224x224.
        self._image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        # Critic prompt length (only used when BestOfN + critic are active);
        # known after critic load below. Default to the policy len in the
        # absence of a critic so participate_loop's dummies still work.
        self._critic_max_token_len = self._max_token_len

        # ---- Optional: load critic + wrap in BestOfN.
        self._critic_model = None
        self._critic_tokenizer = None
        self._bestofn_sample = None
        self._bestofn = None
        self._policy_data_config = data_config

        if critic_args_set:
            logger.info(
                f"Loading critic '{critic_config_name}' from {critic_checkpoint_dir} "
                f"(step={critic_step}, fine_tune={critic_fine_tune_config})..."
            )
            critic_model, critic_norm_stats, critic_config, _ = _load_critic(
                critic_config_name,
                critic_checkpoint_dir,
                fine_tune = critic_fine_tune_config,
                step = critic_step,
            )
            critic_kwargs = _build_critic_kwargs(critic_config)
            self._critic_is_hdf5 = isinstance(critic_config.data, _config.Hdf5RldsDataConfig)
            self._critic_model = critic_model
            self._critic_model.eval()
            self._critic_tokenizer = critic_kwargs["tokenizer"]
            # Critic prompt token length (determines the dummy shape we
            # broadcast on participating workers).
            self._critic_max_token_len = critic_config.model.network_config.max_token_len

            # Auto-route the constant task description into critic tokenization
            # when the critic was trained with prompt_mode="task_description_predict_current_subtask".
            # In that training mode the input prompt was the task description (the
            # subtask text only appears in the next-token auxiliary loss labels),
            # so at eval time we must feed the same task description rather than
            # the client's dynamic subtask.
            critic_prompt_mode = getattr(critic_config.data, "prompt_mode", None)
            if critic_prompt_mode == "task_description_predict_current_subtask" and policy_task_description is not None:
                self._critic_task_description = policy_task_description
                logger.info(
                    f"Critic prompt_mode={critic_prompt_mode!r}: routing constant "
                    f"task_description into critic tokenization."
                )

            # `use_chunk_wise_delta` lives on the data FACTORY (RLDSRoboCasaDataConfig
            # / RoboCoinRldsDataConfig), not the runtime DataConfig instance, so
            # we read it from `config.data`. BestOfNWrapper asserts the policy
            # and critic values match (and that their norm stats are identical).
            policy_use_chunk_wise_delta = config.data.use_chunk_wise_delta
            critic_use_chunk_wise_delta = critic_kwargs["use_chunk_wise_delta"]
            critic_subsample = critic_kwargs["subsample"]

            resolved_offset = (
                critic_action_dim_offset if critic_action_dim_offset is not None else self._action_dim_offset
            )
            critic_action_dim = critic_norm_stats["actions"].mean.shape[-1]
            policy_action_horizon = self._model.action_horizon
            critic_action_horizon = critic_kwargs["action_horizon"]
            logger.info(
                f"Building BestOfNWrapper (num_samples={num_samples}, "
                f"action_dim_offset={resolved_offset}, "
                f"critic_action_dim={critic_action_dim} (policy action_dim={self._model.action_dim}), "
                f"policy_action_horizon={policy_action_horizon}, "
                f"critic_action_horizon={critic_action_horizon}, "
                f"policy_use_chunk_wise_delta={policy_use_chunk_wise_delta}, "
                f"critic_use_chunk_wise_delta={critic_use_chunk_wise_delta})"
            )
            self._bestofn = BestOfNWrapper(
                action_dim = self._model.action_dim,
                action_horizon = self._model.action_horizon,
                max_token_len = self._model.max_token_len,
                base_model = self._model,
                num_samples = num_samples,
                take_min_over_ensemble = take_min_over_ensemble,
                use_target_value = False,
                selection_mode = selection_mode,
                softmax_temperature = softmax_temperature,
                policy_norm_stats = norm_stats,
                critic_norm_stats = critic_norm_stats,
                policy_use_chunk_wise_delta = policy_use_chunk_wise_delta,
                critic_use_chunk_wise_delta = critic_use_chunk_wise_delta,
                critic_use_quantile_norm = critic_kwargs["use_quantile_norm"],
                critic_subsample = critic_subsample,
                critic_action_dim_offset = resolved_offset,
                critic_action_horizon = critic_action_horizon,
            )

            @nnx.jit
            def _bestofn_sample(bon, vf, rng, sample_rngs, transition, critic_prompt, critic_prompt_mask):
                # BestOfNWrapper.sample_actions returns (selected_action, q_values).
                # `rng` is replicated (used only for the softmax-selection rng);
                # `sample_rngs` is DATA_AXIS-sharded so each chip generates unique
                # initial noise. Forwarded as the optional `sample_rngs=` kwarg
                # so the wrapper skips its in-JIT split and uses these directly.
                return bon.sample_actions(
                    rng, transition, compute_next_action = False,
                    value_function = vf,
                    critic_tokenized_prompt = critic_prompt,
                    critic_tokenized_prompt_mask = critic_prompt_mask,
                    sample_rngs = sample_rngs,
                )

            self._bestofn_sample = _bestofn_sample
            logger.info("Critic loaded; BestOfN sample closure JIT-registered.")

        if prewarm:
            self._prewarm_jit()

        logger.info("BestOfNPolicy ready.")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Run a single inference step on rank 0 (or single-host).

        Mirrors LocalPolicy.predict but is callable from WebsocketPolicyServer
        (single-arg dict input, dict output).

        Multi-host: this method is called only on rank 0 (the websocket-bound
        host). It builds the inference inputs, broadcasts them to all other
        hosts via `jax.experimental.multihost_utils.broadcast_one_to_all`, and
        then triggers the JIT'd sampling closure. Ranks 1..N-1 are sitting in
        `participate_loop` and will reach the same broadcast in lockstep so
        the JAX collective ops complete.
        """
        # Make a copy since transforms mutate in place. Capture the prompt
        # string BEFORE _input_transform runs (TokenizePrompt pops it) so we
        # can re-tokenize for the critic.
        obs = jax.tree.map(lambda x: x, obs)
        batched, extras = self._prepare_inputs_rank0(obs, noise = noise)

        # NOTE: order matches participate_loop (broadcast_inputs → next_sample_rng)
        # so collective ops on multi-host hit in the same sequence on every rank.
        batched, extras = self._broadcast_inputs(batched, extras)
        sample_rng, sample_rngs = self._next_sample_rng()

        # ---- DIAGNOSTIC: dump fingerprints of batched + rng + extras keys
        # right before _run_jit_inference. Lets us compare against the SAME
        # path constructed manually in test scripts.
        if _DEBUG and self._process_index == 0:
            import hashlib as _hashlib
            def _fp(name, v):
                if hasattr(v, "shape"):
                    arr = np.asarray(jax.device_get(v))
                    h = _hashlib.sha256(arr.tobytes()).hexdigest()[:16]
                    logger.info(f"[infer.debug] {name} shape={arr.shape} dtype={arr.dtype} sha256[:16]={h} sum={float(arr.astype(np.float64).sum()):.6e}")
                else:
                    logger.info(f"[infer.debug] {name} = {v!r}")
            for k in sorted(batched.keys()):
                v = batched[k]
                if isinstance(v, dict):
                    for dk in sorted(v.keys()):
                        _fp(f"batched['{k}']['{dk}']", v[dk])
                else:
                    _fp(f"batched['{k}']", v)
            _fp("sample_rng", jax.random.key_data(sample_rng))
            # Don't fingerprint `sample_rngs` here — it's DATA_AXIS-sharded across
            # all hosts, so `jax.device_get` would try to fetch a non-process-local
            # array and raise. Per-host divergence is already verified by the
            # `[rng]` log inside `_next_sample_rng` (each host prints its
            # addressable shards).
            logger.info(f"[infer.debug] extras keys = {sorted(extras.keys())}")

        start_time = time.monotonic()
        # BestOfN path returns local [B, n_per_worker, ah, ad] + [B, n_per_worker];
        # policy-only path returns [B, ah, ad] + None.
        actions_out, q_values = self._run_jit_inference(batched, extras, sample_rng, sample_rngs)
        actions_out = jax.block_until_ready(actions_out)
        q_values_np: np.ndarray | None = None
        acs_value: float | None = None
        if q_values is not None:
            q_values = jax.block_until_ready(q_values)
            # actions_pool: (total, ah, ad), q_pool: (total,) — B=1 dropped.
            # `total == num_samples` because the BestOfNWrapper
            # trims its candidate pool to `num_samples` before the critic runs;
            # the padding to `num_samples_padded` happens only at the per-chip
            # rng layer to align with DATA_AXIS sharding, and the extra
            # candidates are dropped in the JIT graph itself.
            actions_pool, q_pool = self._gather_local_candidates(actions_out, q_values)
            n = int(self._bestofn.num_samples)
            actions_pool = actions_pool[:n]
            q_pool = q_pool[:n]
            best_idx = int(np.argmax(q_pool))
            actions_out = actions_pool[best_idx]
            q_values_np = q_pool
            # Average cosine similarity across the candidate pool, computed on
            # the 14-D EEF action slice (the non-EEF padded dims carry other
            # robot DOFs and would skew the similarity).
            eef_pool = actions_pool[
                ..., self._action_dim_offset : self._action_dim_offset + self._eef_action_dim
            ]
            acs_value = _compute_acs(np.asarray(eef_pool))
            logger.info(f"BestOfN q_values (n={n}): {q_values_np.tolist()}  acs={acs_value:.4f}")
        else:
            # Policy-only path: drop the always-1 leading batch dim so the
            # downstream slice + output_transform see (ah, ad) like the
            # BestOfN path does after _gather_local_candidates.
            actions_out = np.asarray(jax.device_get(actions_out))[0]
        infer_ms = (time.monotonic() - start_time) * 1000.0

        # Build a host-side observation snapshot for the output transform's
        # state passthrough (we already have the values on rank 0 in `obs`).
        observation = _model.Observation.from_dict(batched)

        # Slice the model's padded action vector down to the 14-D bimanual EEF
        # block before Unnormalize sees it (norm stats live at the unpadded
        # dim; without this slice Unnormalize would assert shape mismatch).
        # actions_out is now (ah, ad) — gather already dropped the B=1 dim.
        start = self._action_dim_offset
        actions_np = np.asarray(actions_out[..., start : start + self._eef_action_dim])

        decoded = self._output_transform({
            "state": np.asarray(observation.state[0]),
            "actions": actions_np,
            # `state`/`next_state` are passed through some transforms (e.g.
            # AbsoluteActions only touches actions, but mirroring the sibling
            # avoids any "missing key" failure for transforms that index
            # both).
            "next_state": np.asarray(observation.state[0]),
            "next_actions": actions_np,
        })

        result: dict[str, Any] = {
            "actions": np.asarray(decoded["actions"], dtype = np.float32),
            "policy_timing": {"infer_ms": infer_ms},
        }
        if self._return_q_values and q_values_np is not None:
            result["q_values"] = q_values_np
            if acs_value is not None:
                result["acs"] = acs_value
        return result

    # ---------------------------------------------------------------------
    # Multi-host coordination internals
    # ---------------------------------------------------------------------

    def _prepare_inputs_rank0(
        self, obs: dict, *, noise: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Rank-0 only: run the input transform pipeline + build the broadcast
        package (`batched` for the Observation, `extras` for everything else
        the JIT'd inference closure consumes).
        """
        prompt_str = obs.get("prompt")
        if self._policy_task_description is not None:
            obs = {**obs, "prompt": self._policy_task_description}

        transformed = self._input_transform(obs)

        batched: dict[str, Any] = {}
        for k, v in transformed.items():
            if isinstance(v, str):
                continue
            if isinstance(v, dict):
                batched[k] = {dk: jnp.asarray(dv)[None, ...] for dk, dv in v.items()}
            else:
                batched[k] = jnp.asarray(v)[None, ...]

        # Only fall back to all-True masks when the input transform did not
        # already supply them. RoboCasaBimanualEEFInputs sets `image_mask`
        # explicitly with `right_wrist_0_rgb=False` for the single-arm
        # RoboCasa setup (the right-wrist camera doesn't exist), and we
        # must preserve that or the model will attend to a zero-image.
        action_horizon = self._model.action_horizon
        if "action_mask" not in batched:
            # action_horizon=50 with RoboCasa target_fps=30 → only the first
            # 3*50//5 = 30 steps correspond to the 1-second model window;
            # last 20 are zero-padding the training-time fps_mask_30 masked
            # out (see RoboCasaRldsDataset._build_action_mask). Carry the
            # same 30 / 20 split here so policy attention + critic
            # observation match training. Other horizons (30, 60) default
            # to all-True; their adapt-time mask logic in
            # BestOfNWrapper.sample_actions handles subsample / pad.
            if action_horizon == 50:
                fps_mask = jnp.concatenate(
                    [jnp.ones(30, dtype = jnp.bool_), jnp.zeros(20, dtype = jnp.bool_)]
                )
                batched["action_mask"] = fps_mask[None, :]
            else:
                batched["action_mask"] = jnp.ones(action_horizon, dtype = jnp.bool_)[None, :]
        if "image_mask" not in batched:
            batched["image_mask"] = {k: jnp.array([True]) for k in batched.get("image", {})}

        extras: dict[str, Any] = {}
        if self._bestofn is not None:
            if self._critic_task_description is not None:
                # Eval drops the subtask suffix: tokenize the prefix only (no trailing "\n").
                critic_tokens, critic_token_mask, _, _ = _transforms._tokenize_robocoin_subtask_prompt(
                    self._critic_tokenizer, self._critic_task_description, "", append_newline = False,
                )
            else:
                critic_tokens, critic_token_mask = self._critic_tokenizer.tokenize(prompt_str, None)
            extras["critic_tokens"] = jnp.asarray(critic_tokens)[None, ...]
            extras["critic_token_mask"] = jnp.asarray(critic_token_mask)[None, ...]
        if noise is not None:
            noise_arr = jnp.asarray(noise)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr[None, ...]
            extras["noise"] = noise_arr
        return batched, extras

    def _make_dummy_inputs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Workers 1..N-1: construct broadcast-package dummies of the exact
        shapes/dtypes that rank 0 produces, so `broadcast_one_to_all` can
        merge them. Values are placeholders; rank 0's values overwrite them.
        """
        action_horizon = self._model.action_horizon
        batched: dict[str, Any] = {
            "state": jnp.zeros((1, self._eef_action_dim), dtype = jnp.float32),
            "image": {k: jnp.zeros((1, self._image_size, self._image_size, 3), dtype = jnp.uint8) for k in self._image_keys},
            "image_mask": {k: jnp.array([True], dtype = jnp.bool_) for k in self._image_keys},
            "tokenized_prompt": jnp.zeros((1, self._max_token_len), dtype = jnp.int32),
            "tokenized_prompt_mask": jnp.zeros((1, self._max_token_len), dtype = jnp.bool_),
            "action_mask": jnp.ones((1, action_horizon), dtype = jnp.bool_),
        }
        extras: dict[str, Any] = {}
        if self._bestofn is not None:
            extras["critic_tokens"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.int32)
            extras["critic_token_mask"] = jnp.zeros((1, self._critic_max_token_len), dtype = jnp.bool_)
        return batched, extras

    def _broadcast_inputs(
        self, batched: dict[str, Any], extras: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Broadcast the inference inputs from JAX rank 0 to all other hosts.
        No-op on a single-host setup. Source defaults to rank 0 (matches
        where the websocket binds in serve_policy.py).

        `broadcast_one_to_all` up-promotes narrow dtypes (bool → int32,
        uint8 → uint32); we cast back so Pi0.embed_prefix's jaxtyping
        annotation accepts the prefix mask.
        """
        if not self._is_multi_host:
            return batched, extras
        pkg = {"batched": batched, "extras": extras}
        pkg = _multihost.broadcast_one_to_all(pkg)
        batched = _restore_inference_dtypes(pkg["batched"])
        extras = _restore_inference_dtypes(pkg["extras"])
        return batched, extras

    def _next_sample_rng(self):
        """Build (rng_replicated, sample_rngs_sharded) for one inference call.

        `rng_replicated` is a single key, identical on every host — used by
        BestOfNWrapper for the post-sampling softmax-selection rng (and by
        downstream ops that should see the same rng on every host).

        `sample_rngs_sharded` is an `(num_samples_padded, ...)` key array
        sharded along `PartitionSpec(DATA_AXIS, ...)`. Every host computes the
        full split deterministically from `rng_replicated` (so the per-chip
        slices agree across hosts), then `device_put` distributes one unique
        rng per chip. The vmap'd policy.sample_actions inside the JIT then
        sees one different rng per chip → unique noise per chip → diverse
        candidates after `process_allgather`.

        Why two arrays: the caller's previous design folded `process_index`
        into a single replicated-typed rng, which under cross-host FSDP
        all-gather concatenated the disagreeing per-host slices into a
        "Frankenstein" tensor (the bug captured in project_status). Declaring
        the divergent rng as DATA_AXIS-sharded lets JAX route each chip's
        unique slice through SPMD without that collapse.
        """
        self._inference_iter += 1
        fold_value = self._num_processes * self._inference_iter
        base_rng = jax.random.fold_in(self._rng, fold_value)

        total_rngs = jax.random.split(base_rng, self._num_samples_padded)
        # Determine PartitionSpec based on key array layout: typed PRNG keys
        # have ndim=1, untyped uint32 keys have ndim=2 (trailing key-data axis).
        if total_rngs.ndim == 1:
            spec = jax.sharding.PartitionSpec(self._sharding_mod.DATA_AXIS)
        else:
            # ndim >= 2: shard leading axis only, replicate the trailing axes.
            spec = jax.sharding.PartitionSpec(
                self._sharding_mod.DATA_AXIS, *([None] * (total_rngs.ndim - 1))
            )
        sharding_spec = jax.sharding.NamedSharding(self._mesh, spec)
        # `jax.device_put` won't accept a multi-host sharding because each host
        # can only address its 4 local chips, not the global 32. `make_array_from_callback`
        # invokes `cb(index)` once per *addressable* shard on this host, asking
        # for the slice this shard owns. Since `total_rngs` is computed
        # deterministically from `base_rng` on every host, all hosts agree on
        # `total_rngs[index]` — so the resulting global array assembles the
        # right per-chip slices without any cross-host transfer.
        def cb(index):
            return total_rngs[index]
        sample_rngs = jax.make_array_from_callback(total_rngs.shape, sharding_spec, cb)

        # Diagnostic: dump THIS host's addressable slice of the sharded rng so we
        # can grep across worker logs and verify per-host divergence (the goal of
        # this fix). For 8 hosts × 4 chips, every host should see 4 unique key
        # values, and across hosts the 8×4 = 32 keys should all differ. If two
        # hosts log the same first-key, the FSDP all-gather is silently
        # collapsing per-host divergence again — same failure mode as the
        # original Frankenstein bug, just at the rng input.
        try:
            local_shards = sample_rngs.addressable_shards
            local_keys_repr = []
            for s in local_shards:
                arr = np.asarray(jax.random.key_data(s.data))
                local_keys_repr.append(arr.flatten().tolist())
            logger.info(
                "[rng] iter=%d rank=%d/%d local_n_shards=%d local_keys=%s",
                self._inference_iter, self._process_index, self._num_processes,
                len(local_shards), local_keys_repr,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"[rng] addressable_shards introspection failed: {exc!r}")
        return base_rng, sample_rngs

    def _gather_local_candidates(
        self, local_actions, local_q_values,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop the always-1 batch dim, gather across workers, return (actions[N, ah, ad], q_values[N]).

        With the new sharded-rng path, the JIT outputs are multi-host sharded
        along the candidate axis (axis 1 of (B=1, N, ah, ad)) — `jax.device_get`
        on such an array raises "Fetching value for `jax.Array` that spans
        non-addressable devices". Instead, walk the array's `addressable_shards`
        (each shard is a single-chip slice of size 1 along the candidate axis),
        materialize them with `np.asarray(s.data)`, and concat along axis 1 to
        get a host-local numpy view shaped (B=1, n_per_host, ah, ad). Then
        `process_allgather(tiled=True)` collects across hosts: the leading axis
        (= n_per_host after dropping B=1) is multiplied by num_processes, giving
        (num_samples_padded, ah, ad) on every host.
        """
        a_shards = [np.asarray(s.data) for s in local_actions.addressable_shards]
        q_shards = [np.asarray(s.data) for s in local_q_values.addressable_shards]
        # Each shard is one chip's slice along the sharded axis (axis 1). Concat axis=1
        # → host-local view (B=1, n_per_host, ...). For our setup n_per_host = 4.
        local_actions_np = np.concatenate(a_shards, axis = 1) if len(a_shards) > 1 else a_shards[0]
        local_q_np = np.concatenate(q_shards, axis = 1) if len(q_shards) > 1 else q_shards[0]
        # Drop B=1 leading dim.
        local_actions_np = local_actions_np[0]  # (n_per_host, ah, ad)
        local_q_np = local_q_np[0].astype(np.float32)  # (n_per_host,)
        if not self._is_multi_host:
            return local_actions_np, local_q_np
        # tiled=True concatenates host-local arrays along their leading axis,
        # giving (n_per_host * num_processes, ...) = (num_samples_padded, ...)
        # in process-index order. Order matches the rng layout from
        # _next_sample_rng (process p owns rng[p*n_per_host : (p+1)*n_per_host]).
        actions = np.asarray(_multihost.process_allgather(local_actions_np, tiled = True))
        q_values = np.asarray(_multihost.process_allgather(local_q_np, tiled = True), dtype = np.float32)
        return actions, q_values

    def _run_jit_inference(
        self, batched: dict[str, Any], extras: dict[str, Any], rng, sample_rngs,
    ) -> tuple[Any, Any | None]:
        """Build Observation/Transition and call the JIT'd sampling closure.
        Returns (actions_out, q_values_or_None). Workers 1..N-1 discard the
        outputs but still have to make the call so the collective ops complete.

        `rng` is a single replicated key; `sample_rngs` is the DATA_AXIS-sharded
        per-sample rng array (only consumed by the BestOfN-with-critic path).
        """
        observation = _model.Observation.from_dict(batched)
        if self._bestofn is not None:
            transition = _base_vf.Transition(observation = observation)
            actions_out, q_values = self._bestofn_sample(
                self._bestofn, self._critic_model, rng, sample_rngs, transition,
                extras["critic_tokens"], extras["critic_token_mask"],
            )
            return actions_out, q_values
        transition = _model.wrap_observation_as_transition(observation)
        sample_kwargs = dict(self._sample_kwargs)
        if "noise" in extras:
            sample_kwargs["noise"] = extras["noise"]
        if _DEBUG:
            _log_model_inputs("policy", observation)
        actions_out = self._sample_actions_jit(rng, transition, **sample_kwargs)
        # Note: the raw_actions debug print runs INSIDE the JIT graph (inside
        # `_sample_with_debug` wrapped by `_sample_actions_jit`), not here.
        return actions_out, None

    def participate_loop(self) -> None:
        """Workers 1..N-1: block on `broadcast_one_to_all` for each rank-0
        inference, run the JIT'd closure in lockstep, discard the outputs.

        Runs forever; killed when the tmux session is killed (or worker is
        preempted). No-op on single-host.
        """
        if not self._is_multi_host:
            return
        rank = self._process_index
        logger.info(
            f"Worker (JAX rank {rank}): participate_loop started; "
            f"waiting for rank-0 broadcasts..."
        )
        local_iter = 0
        while True:
            try:
                batched, extras = self._make_dummy_inputs()
                batched, extras = self._broadcast_inputs(batched, extras)
                sample_rng, sample_rngs = self._next_sample_rng()
                actions_out, q_values = self._run_jit_inference(
                    batched, extras, sample_rng, sample_rngs,
                )
                jax.block_until_ready(actions_out)
                if q_values is not None:
                    q_values = jax.block_until_ready(q_values)
                    # Participate in rank 0's process_allgather; discard result.
                    self._gather_local_candidates(actions_out, q_values)
                local_iter += 1
                if local_iter % 100 == 0:
                    logger.info(f"Worker (JAX rank {rank}): participated in {local_iter} inferences")
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"Worker (JAX rank {rank}) participate_loop iteration error: {exc!r}")
                # Don't break; keep participating so a transient failure
                # doesn't permanently desync the collective.

    def _prewarm_jit(self) -> None:
        """Trigger one JIT compile pass before serve_forever opens the port.

        On multi-host, every rank participates: rank 0 runs `infer` with a
        dummy obs (which broadcasts inputs to all hosts), and ranks 1..N-1
        block on the matching `broadcast_one_to_all` from `participate_loop`
        — running exactly one iteration here so the JIT graph compiles and
        peer collectives are warm before the websocket starts. After prewarm
        ranks 1..N-1 enter the long-lived `participate_loop` from
        scripts/serve_policy.py main().

        Without prewarm the first websocket request takes 30-60s while JAX
        compiles, which blows past the client's default read timeout.
        """
        try:
            if self._is_multi_host and self._process_index != 0:
                logger.info(
                    f"Worker (JAX rank {self._process_index}): prewarm = one participate iteration"
                )
                batched, extras = self._make_dummy_inputs()
                batched, extras = self._broadcast_inputs(batched, extras)
                sample_rng, sample_rngs = self._next_sample_rng()
                actions_out, q_values = self._run_jit_inference(
                    batched, extras, sample_rng, sample_rngs,
                )
                jax.block_until_ready(actions_out)
                if q_values is not None:
                    self._gather_local_candidates(actions_out, jax.block_until_ready(q_values))
                logger.info(f"Worker (JAX rank {self._process_index}): prewarm done")
                return

            logger.info("Prewarming BestOfNPolicy JIT (this may take 30-60s)...")
            zero_image = np.zeros((self._image_size, self._image_size, 3), dtype = np.uint8)
            if self._critic_is_hdf5:
                # HDF5 pipeline (sim_bimanual_assembly etc.): the input
                # transform expects an `image` dict + flat `state` + `prompt`,
                # matching what the client sends. State norm is quantile so a
                # zeros vector is fine (no quaternion conversion on this path).
                dummy_obs = {
                    "image": {k: zero_image for k in self._image_keys},
                    "state": np.zeros(self._state_dim, dtype = np.float32),
                    "prompt": "prewarm",
                }
            else:
                # RoboCasa/RoboCoin pipeline: flat observation/* keys with the
                # raw RoboCasa state. Identity quaternion (xyzw = [0,0,0,1]) at
                # the base- and eef-rotation slots; convert_raw_state_to_model_state
                # otherwise passes an all-zeros quaternion to scipy.Rotation
                # which raises.
                state = np.zeros(_robocasa_policy.ROBOCASA_RAW_STATE_DIM, dtype = np.float32)
                state[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32)   # base quat
                state[10:14] = np.array([0.0, 0.0, 0.0, 1.0], dtype = np.float32)  # eef quat
                dummy_obs = {
                    "observation/image": zero_image,
                    "observation/image_right": zero_image,
                    "observation/wrist_image": zero_image,
                    "observation/state": state,
                    "prompt": "prewarm",
                }
            _ = self.infer(dummy_obs)
            logger.info("Prewarm complete.")
        except Exception as exc:  # pylint: disable=broad-except
            # Prewarm is best-effort; if the dummy obs doesn't fit the loaded
            # config (e.g. a non-RoboCasa policy), fall back to lazy compile
            # at first real request.
            logger.warning(
                f"Prewarm failed ({exc!r}); will compile on first request instead."
            )


def create_bestofn_policy(
    *,
    policy_config_name: str,
    policy_checkpoint_dir: str,
    policy_step: int | None = None,
    policy_fine_tune_config: str | None = None,
    policy_task_description: str | None = None,
    critic_config_name: str | None = None,
    critic_checkpoint_dir: str | None = None,
    critic_step: int | None = None,
    critic_fine_tune_config: str | None = None,
    num_samples: int = 8,
    take_min_over_ensemble: bool = True,
    selection_mode: Literal["argmax", "softmax"] = "argmax",
    softmax_temperature: float = 1.0,
    critic_action_dim_offset: int | None = None,
    default_prompt: str | None = None,
    prewarm: bool = True,
) -> BestOfNPolicy:
    """Convenience factory; matches the kwargs the serve_policy CLI exposes."""
    return BestOfNPolicy(
        policy_config_name = policy_config_name,
        policy_checkpoint_dir = policy_checkpoint_dir,
        policy_step = policy_step,
        policy_fine_tune_config = policy_fine_tune_config,
        policy_task_description = policy_task_description,
        critic_config_name = critic_config_name,
        critic_checkpoint_dir = critic_checkpoint_dir,
        critic_step = critic_step,
        critic_fine_tune_config = critic_fine_tune_config,
        num_samples = num_samples,
        take_min_over_ensemble = take_min_over_ensemble,
        selection_mode = selection_mode,
        softmax_temperature = softmax_temperature,
        critic_action_dim_offset = critic_action_dim_offset,
        default_prompt = default_prompt,
        prewarm = prewarm,
    )
