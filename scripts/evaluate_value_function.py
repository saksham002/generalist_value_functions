"""Evaluate a value function checkpoint on RoboCOIN cached trajectories."""

import dataclasses
import logging
import os
import pickle

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.best_of_n import BestOfNWrapper
from openpi.models.best_of_n import BestOfNWrapperConfig
import openpi.models.model as _model
from openpi.policies.subtask_decoder import SubtaskDecoder
from openpi.robocoin_utils.load_model_utils import load_critic
from openpi.robocoin_utils.load_model_utils import load_train_module
from openpi.robocoin_utils.utils import cache_val_episodes
from openpi.robocoin_utils.utils import count_subtask_segments
from openpi.robocoin_utils.utils import decode_episode_images
from openpi.robocoin_utils.utils import get_obs_and_action
from openpi.robocoin_utils.utils import predict_values
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as _sharding
import openpi.transforms as _transforms
import openpi.value_functions.base_value_functions as _base_vf
from openpi.value_functions.networks.paligemma import NUM_PATCHES_PER_IMAGE
from openpi.value_functions.networks.paligemma import compute_rope_positions
from openpi.value_functions.networks.paligemma import make_attn_mask

logger = logging.getLogger(__name__)


# =============================================================================
# Named BestOfN policy configs for counterfactual evaluation.
# Pass the name via --policy-config on the CLI.
# =============================================================================
_POLICY_CONFIGS: dict[str, BestOfNWrapperConfig] = {
    "robocoin_bimanual_sarsa": BestOfNWrapperConfig(
        action_dim=14,
        action_horizon=50,
        base_model_config=None,
        num_samples=8,
        use_target_value=False,
    ),
    "robocoin_bimanual_cql": BestOfNWrapperConfig(
        action_dim=14,
        action_horizon=50,
        base_model_config=None,
        num_samples=8,
        use_target_value=True,
    ),
}


def get_policy_config(name: str) -> BestOfNWrapperConfig:
    if name not in _POLICY_CONFIGS:
        raise ValueError(
            f"Unknown policy config '{name}'. Available: {sorted(_POLICY_CONFIGS.keys())}"
        )
    return _POLICY_CONFIGS[name]


@dataclasses.dataclass(frozen = True)
class EvalConfig:
    """Configuration for value function evaluation."""

    config_name: str
    checkpoint_path: str
    split: str = "val"
    fine_tune: str | None = None
    include_repos: tuple[str, ...] = ()
    num_trajectories: int = 10
    cache_dir: str | None = None
    output_dir: str | None = None
    project_name: str = "robocoin_value_eval"
    counterfactual_best_of_n: bool = False
    counterfactual_action_store_dir: str | None = None
    policy_config: str | None = None
    policy_checkpoint_path: str | None = None
    # Critic checkpoint step to load. If None, uses the latest step.
    step: int | None = None
    override_task_prompt: str | None = None
    batch_size: int = 32
    # If True, populate the validation cache and exit before plotting.
    cache_only: bool = False
    # Action-gradient-norm mode (action-conditioned Q critics only). When set,
    # write one ``<demo>.npz`` per evaluated validation demo into this dir — each
    # holding the per-frame squared L2 norm of the action gradient,
    # ``||grad_a Q(s, a)||^2`` (summed over the whole 60x14 action chunk), the
    # predicted Q, and the MC return — then exit before the normal plotting /
    # subtask / BestOfN paths. Demos are drawn from the validation split,
    # filtered to is_partial=False and has_subtask_annotations=True.
    action_gradient_norm_dir: str | None = None
    # Number of validation demos to evaluate in action-gradient-norm mode.
    num_grad_demos: int = 5
    # Shared cache dir for action-gradient-norm mode. When set, demos are cached
    # here (and reused if already populated) instead of under
    # ``<action_gradient_norm_dir>/val_cache``. Point multiple runs (e.g. two
    # critics) at the same dir so they evaluate the identical cached demos; the
    # first run populates it and later runs reuse it. npz outputs still go to each
    # run's own ``action_gradient_norm_dir``.
    grad_cache_dir: str | None = None
    # Which action to evaluate Q(s, a) / grad_a Q at in action-gradient-norm mode:
    # "dataset" -> the behaviour action from the dataset (prefix="");
    # "cached"  -> the first cached counterfactual action, cached_action[0]
    #              (prefix="counterfactual_", i.e. counterfactual_actions[:, 0]).
    # "cached" requires the counterfactual action store to be joined, which only
    # happens on the train split (see rlds_dataset.py), so use --split train and
    # set --counterfactual-action-store-dir (or a config that already sets it).
    grad_action_source: str = "dataset"
    # Optional override of the data config's rlds_data_dir (e.g. point at a local
    # mirror of the TFDS data instead of the GCS default baked into the config).
    rlds_data_dir: str | None = None
    # Subtask critics only. When set, additionally write one ``<traj>.npz`` per
    # evaluated trajectory into this dir — holding every per-frame value
    # prediction, the ground-truth subtask boundaries, and the
    # autoregressively-decoded subtask predictions sampled at
    # ``subtask_decode_stride`` frame intervals. The per-trajectory video is
    # rendered either way.
    subtask_npz_dir: str | None = None
    # Frame stride between autoregressive subtask decodes. None decodes once per
    # second (i.e. fps frames); 1 decodes at every frame instead of forward-filling
    # between sparse decodes.
    subtask_decode_stride: int | None = None
    # Teacher-forced perplexity of the cached ground-truth subtask tokens, scored once
    # per decoded frame. Diagnostic only: it drives the video's perplexity curve and
    # never affects values or the decoded subtask. It
    # costs a second full prefix forward per frame plus a non-JIT'd decode walk, so
    # disable it when decoding at a small stride. False leaves the perplexities NaN,
    # which the plotting path already tolerates.
    score_gt_perplexity: bool = True
    # LANGUAGE axis (subtask critics only). Selects which subtask conditions the
    # per-frame value: False uses the ground-truth subtask cached in the .pkl,
    # True rebuilds each prompt as task_description + decoded_subtask + "\n" from
    # the SubtaskDecoder output. The subtask is always decoded either way (it
    # titles the video); this flag only decides whether it feeds the value.
    # Orthogonal to counterfactual_value_action.
    condition_on_decoded_subtask: bool = False
    # Force the standard validation plots even for a subtask critic, whose
    # prompt_mode would otherwise route to the subtask video and run the
    # autoregressive decode at every sampled frame. Use to score the cached
    # ground-truth-subtask prompts with no decoding at all.
    disable_subtask_decoding: bool = False
    # ACTION axis (action-conditioned critics only). Selects which action the
    # per-frame Q is evaluated at: False uses the dataset (behaviour) action, True
    # uses the highest-value cached counterfactual (policy-generated) action, which
    # requires the counterfactual action store to be joined for this split.
    # Orthogonal to condition_on_decoded_subtask.
    counterfactual_value_action: bool = False
    # Number of FSDP devices for the inference mesh. None → jax.device_count()
    # (prior default: pure FSDP, params sharded across devices, inputs
    # replicated). Set to 1 on a GPU node for pure data parallelism: params
    # replicated and each batched value forward split along the batch axis
    # across all devices (mesh = (device_count, 1)).
    fsdp_devices: int | None = None
    # Optional wandb run name (used when output_dir is None → wandb logging).
    # Defaults to f"eval_{config_name}" when unset.
    wandb_run_name: str | None = None


def _resolve_eval_cache_dir(eval_config: EvalConfig) -> str:
    if eval_config.cache_dir is not None:
        return eval_config.cache_dir
    return os.path.join(eval_config.checkpoint_path, "eval_cache", eval_config.split)


def _load_cached_trajectories(cache_dir: str) -> dict[str, list[dict]]:
    # Match generate_validation_plots_dlimp's loader: accept any .pkl, use the
    # bare filename (sans .pkl) as the trajectory key. The older `traj_<int>`
    # naming is no longer produced by cache_val_episodes, which writes
    # `<sanitized_repo_key>.pkl` (e.g. RoboCOIN__Split_aloha_pour_tea.pkl).
    # epath so a gs:// cache dir works (reads pickle bytes straight from GCS),
    # in addition to local/NFS paths.
    from etils import epath

    traj_frames: dict[str, list[dict]] = {}
    for path in sorted(epath.Path(cache_dir).iterdir()):
        if path.name.endswith(".pkl"):
            key = path.name[: -len(".pkl")]
            traj_frames[key] = pickle.loads(path.read_bytes())
    return traj_frames


def _split_trajectory_frames(
    traj_frames: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, tuple[str, int, str]], dict[str, list[str]]]:
    max_subtask_segments = 16
    split_traj_frames: dict[str, list[dict]] = {}
    traj_to_repo_ep: dict[str, tuple[str, int, str]] = {}
    ep_subtasks: dict[str, list[str]] = {}

    for traj_idx, frames in traj_frames.items():
        if not frames:
            continue

        episode_index = int(frames[0]["episode_index"])
        repo_id = frames[0]["repo_id"]
        if isinstance(repo_id, np.ndarray):
            repo_id = repo_id.item()
        if isinstance(repo_id, bytes):
            repo_id = repo_id.decode("utf-8")

        num_segments, split_frame_idx, segments = count_subtask_segments(frames)
        if num_segments > max_subtask_segments:
            split_segment_idx = num_segments // 2
            key_part0 = f"{traj_idx}_p0"
            key_part1 = f"{traj_idx}_p1"
            split_traj_frames[key_part0] = frames[:split_frame_idx]
            split_traj_frames[key_part1] = frames[split_frame_idx:]
            traj_to_repo_ep[key_part0] = (repo_id, episode_index, "_part0")
            traj_to_repo_ep[key_part1] = (repo_id, episode_index, "_part1")
            ep_subtasks[key_part0] = segments[:split_segment_idx]
            ep_subtasks[key_part1] = segments[split_segment_idx:]
        else:
            key = str(traj_idx)
            split_traj_frames[key] = frames
            traj_to_repo_ep[key] = (repo_id, episode_index, "")
            ep_subtasks[key] = segments

    return split_traj_frames, traj_to_repo_ep, ep_subtasks


# =============================================================================
# Subtask predictor for `task_description_predict_current_subtask` critics.
# Per-second autoregressive decode (max 16 tokens or until the emitted token list
# matches a subtask present in the trajectory) plus teacher-forced perplexity of
# the cached ground-truth subtask. KV-cached; gemma_2b uses bidirectional prefix
# + causal suffix, gemma4 stays causal throughout (the network builds it that
# way). Closure structure mirrors SubtaskPredictorPolicy._build_subtask_predictor_closures
# but inlined here so the policy file stays untouched.
# =============================================================================


def _build_subtask_decoder(critic_model) -> dict:
    """Return JIT'd {prefix_forward, decode_step, logits, is_gemma4} closures.

    Mirrors SubtaskPredictorPolicy._build_subtask_predictor_closures from
    src/openpi/policies/subtask_predictor_policy.py: the prefix includes the
    state token (via `_embed_prefix` / `_build_gemma4_prefix_cache_inputs`),
    matching the production serving path that's known to decode correctly.
    Earlier I dropped state based on the training-time cumsum analysis (suffix
    tokens never attend to state at training); empirically that change was
    inert (predictions stayed identical) and the real bug was the image dtype
    in `_build_critic_obs_for_frame`. Keep this aligned with production.
    """
    net = _get_critic_network(critic_model)
    is_gemma4 = "gemma4" in getattr(getattr(net, "config", None), "paligemma_variant", "")

    if not is_gemma4:
        @nnx.jit
        def _prefix_forward(model, observation):
            n = _get_critic_network(model)
            obs = _model.preprocess_observation(
                None, observation, train = False, image_resolution = n._image_size,
            )
            prefix_tokens_list, prefix_mask_list, prefix_ar_mask_list = n._embed_prefix(obs)
            prefix_tokens = jnp.concatenate(prefix_tokens_list, axis = 1)
            prefix_mask = jnp.concatenate(prefix_mask_list, axis = 1)
            prefix_ar_mask = jnp.array(prefix_ar_mask_list)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, suffix_mask = None)
            text_start = n._num_cameras * NUM_PATCHES_PER_IMAGE
            text_len = obs.tokenized_prompt.shape[1]
            positions = compute_rope_positions(
                prefix_mask,
                shift_start_index = text_start + text_len,
                subtask_start_index = None,
                subtask_end_index = None,
            )
            (hidden,), kv_cache = n.PaliGemma.llm(
                [prefix_tokens], mask = prefix_attn_mask, positions = positions,
            )
            last_text_pos = text_start + jnp.sum(
                obs.tokenized_prompt_mask.astype(jnp.int32), axis = 1,
            ) - 1
            last_hidden = jnp.take_along_axis(hidden, last_text_pos[:, None, None], axis = 1)
            return last_hidden, kv_cache, prefix_mask, last_text_pos

        def _decode_step(model, token_id, kv_cache, prefix_mask, last_text_pos, suffix_pos_so_far):
            # Not JIT'd: each step appends 1 to kv_cache, so shapes change every call.
            n = _get_critic_network(model)
            tok_arr = jnp.asarray(token_id, dtype = jnp.int32).reshape(1, 1)
            tok_embed = n.PaliGemma.llm(tok_arr, method = "embed")
            to_prefix = prefix_mask[:, None, :]
            to_suffix = jnp.ones((1, 1, suffix_pos_so_far + 1), dtype = jnp.bool_)
            mask = jnp.concatenate([to_prefix, to_suffix], axis = -1)
            position = (last_text_pos + suffix_pos_so_far + 1).reshape(1, 1)
            (hidden,), kv_cache = n.PaliGemma.llm(
                [tok_embed], mask = mask, positions = position, kv_cache = kv_cache,
            )
            return hidden, kv_cache
    else:
        @nnx.jit
        def _prefix_forward(model, observation):
            n = _get_critic_network(model)
            obs = _model.preprocess_observation(
                None, observation, train = False, image_resolution = n._image_size,
            )
            prefix_inputs = n._build_gemma4_prefix_cache_inputs(obs)
            (hidden,), kv_cache = n.PaliGemma.llm(
                [prefix_inputs["tokens"]],
                mask = prefix_inputs["attn_mask"],
                positions = prefix_inputs["positions"],
                kv_cache = prefix_inputs["empty_kv_cache"],
                adarms_cond = [None],
                per_layer_input = prefix_inputs["per_layer_input"],
            )
            prefix_mask = prefix_inputs["input_mask"]
            num_soft = n._num_soft_tokens_per_image
            tokens_per_block = num_soft + 4
            text_start = 1 + n._num_cameras * tokens_per_block
            last_text_pos = text_start + jnp.sum(
                obs.tokenized_prompt_mask.astype(jnp.int32), axis = 1,
            ) - 1
            last_hidden = jnp.take_along_axis(hidden, last_text_pos[:, None, None], axis = 1)
            return (
                last_hidden, kv_cache, prefix_mask, last_text_pos,
                jnp.asarray(prefix_inputs["prefix_len"]),
                jnp.asarray(prefix_inputs["cache_size"]),
            )

        @nnx.jit(static_argnames = ("suffix_pos_so_far", "prefix_len", "cache_size"))
        def _decode_step(
            model, token_id, kv_cache, prefix_mask, last_text_pos,
            prefix_len, cache_size, suffix_pos_so_far,
        ):
            n = _get_critic_network(model)
            tok_arr = jnp.asarray(token_id, dtype = jnp.int32).reshape(1, 1)
            tok_embed = n.PaliGemma.llm(tok_arr, method = "embed")
            per_layer_input = None
            if n._gemma4_per_layer_input_dim > 0:
                per_layer_input = n.PaliGemma.llm(
                    tok_embed, tok_arr, method = "encode_per_layer_input",
                )
            suffix_pad_len = cache_size - prefix_len
            prefix_portion = prefix_mask[:, None, :]
            suffix_arange = jnp.arange(suffix_pad_len)
            suffix_portion = (suffix_arange < (suffix_pos_so_far + 1))[None, None, :]
            attn_mask = jnp.concatenate([prefix_portion, suffix_portion], axis = -1)
            position = (last_text_pos + suffix_pos_so_far + 1).reshape(1, 1)
            (hidden,), kv_cache = n.PaliGemma.llm(
                [tok_embed],
                mask = attn_mask,
                positions = position,
                kv_cache = kv_cache,
                adarms_cond = [None],
                per_layer_input = per_layer_input,
            )
            return hidden, kv_cache

    @nnx.jit
    def _logits_from_hidden(model, hidden):
        return _get_critic_network(model).decode(hidden)

    return {
        "prefix_forward": _prefix_forward,
        "decode_step": _decode_step,
        "logits": _logits_from_hidden,
        "is_gemma4": is_gemma4,
    }


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_ids))
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return str(inner.decode(token_ids))
    return " ".join(str(t) for t in token_ids)


def _run_prefix_forward(closures, critic_model, obs):
    out = closures["prefix_forward"](critic_model, obs)
    if closures["is_gemma4"]:
        last_hidden, kv_cache, prefix_mask, last_text_pos, prefix_len, cache_size = out
        return {
            "last_hidden": last_hidden,
            "kv_cache": kv_cache,
            "prefix_mask": prefix_mask,
            "last_text_pos": last_text_pos,
            "prefix_len": int(np.asarray(prefix_len)),
            "cache_size": int(np.asarray(cache_size)),
        }
    last_hidden, kv_cache, prefix_mask, last_text_pos = out
    return {
        "last_hidden": last_hidden,
        "kv_cache": kv_cache,
        "prefix_mask": prefix_mask,
        "last_text_pos": last_text_pos,
    }


def _run_decode_step(closures, critic_model, prefix_state, token_id, suffix_pos):
    if closures["is_gemma4"]:
        return closures["decode_step"](
            critic_model, token_id, prefix_state["kv_cache"],
            prefix_state["prefix_mask"], prefix_state["last_text_pos"],
            prefix_state["prefix_len"], prefix_state["cache_size"], suffix_pos,
        )
    return closures["decode_step"](
        critic_model, token_id, prefix_state["kv_cache"],
        prefix_state["prefix_mask"], prefix_state["last_text_pos"], suffix_pos,
    )


def _score_gt_perplexity(
    closures,
    critic_model,
    critic_obs,
    gt_token_ids: list[int],
) -> float:
    """Teacher-forced perplexity of the cached ground-truth subtask tokens."""
    if not gt_token_ids:
        return float("nan")
    state = _run_prefix_forward(closures, critic_model, critic_obs)
    logits = closures["logits"](critic_model, state["last_hidden"])
    log_probs = jax.nn.log_softmax(logits[0, 0])
    total_neg_logp = -float(np.asarray(log_probs[gt_token_ids[0]]))
    kv_cache = state["kv_cache"]
    for k in range(1, len(gt_token_ids)):
        prev = gt_token_ids[k - 1]
        state["kv_cache"] = kv_cache
        hidden, kv_cache = _run_decode_step(closures, critic_model, state, prev, k - 1)
        logits = closures["logits"](critic_model, hidden)
        log_probs = jax.nn.log_softmax(logits[0, 0])
        total_neg_logp += -float(np.asarray(log_probs[gt_token_ids[k]]))
    return float(np.exp(total_neg_logp / len(gt_token_ids)))


def _extract_gt_subtask_tokens(frame: dict) -> list[int]:
    """Slice the cached ``tokenized_prompt`` between subtask_{start,end}_index."""
    tokens = np.asarray(frame["tokenized_prompt"]).tolist()
    start = int(np.asarray(frame["subtask_start_index"]))
    end = int(np.asarray(frame["subtask_end_index"]))
    if end < start:
        return []
    return [int(t) for t in tokens[start : end + 1]]


def _build_critic_obs_for_frame(
    frame: dict,
    prefix_tokens: np.ndarray,
    prefix_mask: np.ndarray,
    image_keys: tuple[str, ...],
) -> _model.Observation:
    """Single-frame Observation with prefix-only prompt and no subtask indices.

    Routes via `Observation.from_dict` so the uint8 -> float32 [-1, 1] image
    cast fires (the Observation constructor itself does NOT do this cast, and
    `preprocess_observation` only resizes / augments — it doesn't convert
    dtype). Skipping the cast feeds raw [0, 255] into the SigLIP encoder and
    completely OODs the model.
    """
    image_dict: dict = {}
    image_mask_dict: dict = {}
    for k in image_keys:
        img = np.asarray(frame["image"][k])
        image_dict[k] = jnp.asarray(img)[None, ...]
        image_mask_dict[k] = jnp.array([True], dtype = jnp.bool_)
    return _model.Observation.from_dict({
        "image": image_dict,
        "image_mask": image_mask_dict,
        "state": jnp.asarray(np.asarray(frame["state"]))[None, ...],
        "tokenized_prompt": jnp.asarray(prefix_tokens)[None, :],
        "tokenized_prompt_mask": jnp.asarray(prefix_mask)[None, :],
    })


def _render_subtask_video(
    mc_returns: list[float],
    predicted_values: list[float],
    perplexities: list[float | None],
    predicted_texts: list[str | None],
    gt_texts: list[str | None],
    frame_images: list[np.ndarray],
    fps: int,
    ep_idx: int,
    subtask_texts: list[str] | None,
    output_dir: str | None,
    plot_key: str,
    *,
    condition_on_decoded: bool = False,
):
    """5-subplot video laid out 2 rows x 3 cols (6th cell empty).

    Layout:
        Row 0: left wrist  | value plot  | perplexity plot (title = "Pred: ...\nGT: ...")
        Row 1: right wrist | base camera | (empty)

    Per-frame Predicted and Ground-Truth subtask strings are stacked as the
    perplexity subplot's two-line title so they sit directly above the
    perplexity curve, visible in every frame.
    """
    import matplotlib.pyplot as plt

    T = len(mc_returns)
    timesteps = np.arange(T)
    perplexity_arr = np.array(
        [p if p is not None and np.isfinite(p) else np.nan for p in perplexities],
        dtype = np.float64,
    )
    # Y-axis clip for the perplexity subplot: cap the displayed max at 5x the
    # min so a single huge spike (e.g. a model collapse) doesn't squash the
    # rest of the curve into a flat line. Underlying values are unchanged.
    finite_perp = perplexity_arr[np.isfinite(perplexity_arr)]
    perp_ylim: tuple[float, float] | None = None
    if finite_perp.size > 0:
        perp_min = float(np.min(finite_perp))
        perp_max_raw = float(np.max(finite_perp))
        perp_ylim = (perp_min, min(perp_max_raw, 5.0 * perp_min))

    fig, axes = plt.subplots(2, 3, figsize = (20, 10))
    ax_left_wrist = axes[0, 0]
    ax_val = axes[0, 1]
    ax_perp = axes[0, 2]
    ax_right_wrist = axes[1, 0]
    ax_base = axes[1, 1]
    ax_empty = axes[1, 2]

    subtask_caption = None
    if subtask_texts:
        numbered = [f"{i+1}. {t}" for i, t in enumerate(subtask_texts)]
        lines = ["  ".join(numbered[i : i + 3]) for i in range(0, len(numbered), 3)]
        subtask_caption = "Subtasks:\n" + "\n".join(lines)
        bottom_margin = 0.10 + 0.03 * (len(lines) + 1)
        fig.subplots_adjust(bottom = bottom_margin)

    video_frames = []
    for t in range(T):
        for ax in axes.flat:
            ax.cla()

        ax_left_wrist.imshow(frame_images[t][0])
        ax_left_wrist.axis("off")
        ax_left_wrist.set_title("Left Wrist", fontsize = 12)

        ax_right_wrist.imshow(frame_images[t][1])
        ax_right_wrist.axis("off")
        ax_right_wrist.set_title("Right Wrist", fontsize = 12)

        ax_base.imshow(frame_images[t][2])
        ax_base.axis("off")

        ax_val.plot(timesteps, mc_returns, label = "MC Returns", color = "blue", linewidth = 2)
        ax_val.plot(
            timesteps, predicted_values, label = "Predicted Value",
            color = "orange", linewidth = 2, linestyle = "--",
        )
        ax_val.axvline(x = t, color = "red", linewidth = 2, alpha = 0.8)
        ax_val.set_xlabel("Timestep", fontsize = 11)
        ax_val.set_ylabel("Value", fontsize = 11)
        if condition_on_decoded:
            decoded_title = predicted_texts[t] if predicted_texts[t] is not None else "(no decode yet)"
            ax_val.set_title(
                f"Episode {ep_idx} - Value | decoded subtask: {decoded_title}",
                fontsize = 12, wrap = True,
            )
        else:
            ax_val.set_title(f"Episode {ep_idx} - Value", fontsize = 12)
        ax_val.legend(fontsize = 10)
        ax_val.grid(visible = True, alpha = 0.3)

        pred_text = predicted_texts[t] if predicted_texts[t] is not None else "(no prediction this step)"
        gt_text = gt_texts[t] if gt_texts[t] is not None else "(no GT)"
        ax_perp.plot(timesteps, perplexity_arr, color = "purple", linewidth = 2, marker = "o", markersize = 3)
        ax_perp.axvline(x = t, color = "red", linewidth = 2, alpha = 0.8)
        ax_perp.set_xlabel("Timestep", fontsize = 32)
        ax_perp.set_ylabel("GT Subtask Perplexity", fontsize = 32)
        ax_perp.set_title(f"Pred: {pred_text}\nGT:   {gt_text}", fontsize = 28, wrap = True)
        ax_perp.grid(visible = True, alpha = 0.3)
        if perp_ylim is not None:
            ax_perp.set_ylim(perp_ylim[0], perp_ylim[1])

        ax_empty.axis("off")

        if subtask_caption is None:
            plt.tight_layout()
        else:
            for txt in fig.texts:
                txt.remove()
            fig.text(
                0.5, 0.01, subtask_caption,
                ha = "center", va = "bottom", fontsize = 9,
                family = "monospace", linespacing = 1.5,
            )

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype = np.uint8).reshape(h, w, 4)[:, :, :3]
        video_frames.append(buf.copy())

    plt.close(fig)

    if output_dir is not None:
        import imageio

        os.makedirs(output_dir, exist_ok = True)
        sanitized_key = plot_key.replace("/", "_")
        out_path = os.path.join(output_dir, f"{sanitized_key}.mp4")
        h, w = video_frames[0].shape[:2]
        # imageio requires both dims divisible by 16 (macro_block_size).
        h_crop = h - (h % 16)
        w_crop = w - (w % 16)
        cropped = [f[:h_crop, :w_crop] for f in video_frames]
        imageio.mimsave(out_path, cropped, format = "mp4", fps = fps, codec = "libx264", quality = 8)
        logger.info(f"Saved subtask video to {out_path}")
        return out_path

    import wandb

    video_array = np.stack(video_frames).transpose(0, 3, 1, 2)
    return wandb.Video(video_array, fps = fps, format = "gif")


def _subtask_boundary_indices(frames: list[dict]) -> list[int]:
    """Frame indices where the GT current-subtask token sequence changes."""
    boundaries: list[int] = []
    prev: tuple[int, ...] | None = None
    for i, f in enumerate(frames):
        if "subtask_start_index" not in f or "subtask_end_index" not in f:
            continue
        seq = tuple(_extract_gt_subtask_tokens(f))
        if prev is not None and seq != prev:
            boundaries.append(i)
        prev = seq
    return boundaries


def _run_subtask_prediction(
    model,
    val_tokenizer,
    cache_dir: str,
    image_size: tuple[int, int],
    eval_config: "EvalConfig",
    batch_size: int,
    action_conditioned: bool,
    mesh = None,
) -> None:
    """Render per-trajectory subtask-prediction videos.

    Only fires when the critic was trained with
    ``prompt_mode == "task_description_predict_current_subtask"``. The decoder
    closures (`prefix_forward` / `decode_step` / `logits`) are JIT'd and
    SPMD-sharded across every host, so EVERY rank must drive the same JIT call
    sequence in lockstep — otherwise non-rank-0 hosts sit at the downstream
    multihost barrier while rank 0 deadlocks waiting for cross-host collectives
    that never fire (mirrors SubtaskPredictorPolicy._run_subtask_predictor_lockstep).
    Per-frame text decoding, video rendering, and wandb.log are gated to rank 0.

    Stride between decoded frames is one per second (= ``fps`` cache frames).
    """
    is_rank0 = jax.process_index() == 0

    traj_frames = _load_cached_trajectories(cache_dir)
    split_traj_frames, traj_to_repo_ep, ep_subtasks = _split_trajectory_frames(traj_frames)
    if not split_traj_frames:
        if is_rank0:
            logger.warning("No cached trajectories found for subtask prediction.")
        return

    # The closures back the teacher-forced GT perplexity only; the subtask decode
    # itself goes through the shared SubtaskDecoder module below.
    if is_rank0:
        logger.info("Building GT-perplexity closures (gemma_2b/gemma4 KV-cache path).")
    closures = _build_subtask_decoder(model)
    image_keys = tuple(_get_critic_network(model).config.image_keys)

    # Single decode path: the shared SubtaskDecoder module, same code as serving /
    # BestOfN. `predict()` is deterministic + SPMD-lockstep, so calling it on every
    # rank yields identical tokens everywhere — which all ranks need when the value
    # prompt is rebuilt from the decode (predict_values is SPMD). Quiet its
    # per-decode INFO log on non-rank-0 to avoid 16x spam.
    decode_module = SubtaskDecoder(model, val_tokenizer, decode_every = 1, max_tokens = 16)
    if not is_rank0:
        logging.getLogger("openpi.policies.subtask_decoder").setLevel(logging.WARNING)
    else:
        logger.info(
            "Subtask prompt source: %s | value action source: %s",
            "decoded" if eval_config.condition_on_decoded_subtask else "ground-truth",
            "counterfactual" if eval_config.counterfactual_value_action else "dataset",
        )

    rendered: dict[str, object] = {}

    npz_mode = eval_config.subtask_npz_dir is not None

    for traj_idx, frames in split_traj_frames.items():
        decode_episode_images(frames, image_size)
        repo_id, ep_idx, part_suffix = traj_to_repo_ep[traj_idx]

        # Values are computed once per trajectory, after the decode loop, so that
        # both input axes are resolved first: the prompt (GT vs decoded subtask)
        # and the action (dataset vs counterfactual). Runs on every host (SPMD via
        # predict_values' _jitted_compute_value).
        seg_all_frames = [(traj_idx, i, f) for i, f in enumerate(frames)]
        seg_mc = {traj_idx: [f["mc_return"] for f in frames]}
        mc_returns = [float(np.asarray(v)) for v in seg_mc[traj_idx]]

        if is_rank0:
            logger.info(
                f"Traj {traj_idx} (repo {repo_id}, episode {ep_idx}{part_suffix}): {len(frames)} frames."
            )

        prefix_text_raw = frames[0]["subtask_1_text"]
        if isinstance(prefix_text_raw, bytes):
            prefix_text_raw = prefix_text_raw.decode("utf-8")
        prefix_text = str(prefix_text_raw)
        prefix_tokens, prefix_mask, _, _ = _transforms._tokenize_robocoin_subtask_prompt(
            val_tokenizer, prefix_text, "", append_newline = False,
        )

        fps = int(frames[0]["fps"])
        # Decode once per second unless an explicit stride is given; the last frame
        # is always sampled so the video's final title reflects a real decode.
        step = eval_config.subtask_decode_stride if eval_config.subtask_decode_stride is not None else max(1, fps)
        sample_indices = list(range(0, len(frames), step))
        if sample_indices and sample_indices[-1] != len(frames) - 1:
            sample_indices.append(len(frames) - 1)

        perplexities: list[float | None] = [None] * len(frames)
        predicted_texts: list[str | None] = [None] * len(frames)
        gt_texts: list[str | None] = [None] * len(frames)

        # The decoded text per sample index is needed on EVERY rank so the value
        # prompt can be rebuilt identically when conditioning on the decode.
        sampled_decoded_text: dict[int, str] = {}
        last_perp: float | None = None
        last_pred: str | None = None
        last_gt: str | None = None
        for sample_pos, t in enumerate(sample_indices):
            critic_obs = _build_critic_obs_for_frame(
                frames[t], prefix_tokens, prefix_mask, image_keys,
            )
            # Both calls below issue JIT'd SPMD collectives; every rank must
            # invoke them in lockstep. Non-rank-0 hosts discard the outputs but
            # still need to participate so cross-host attention/all-gather
            # collectives complete.
            # Module decode is deterministic + lockstep-safe → identical on every
            # rank; stored on all ranks for the value-prompt rebuild.
            decoded = decode_module.predict(critic_obs)
            predicted_tokens = decoded["predicted_subtask_tokens"]
            predicted_text = decoded["predicted_subtask"]
            sampled_decoded_text[t] = predicted_text
            gt_tokens = _extract_gt_subtask_tokens(frames[t])
            gt_perp = (
                _score_gt_perplexity(closures, model, critic_obs, gt_tokens)
                if (gt_tokens and eval_config.score_gt_perplexity) else float("nan")
            )
            if not is_rank0:
                continue
            gt_text = _decode_token_ids(val_tokenizer, gt_tokens) if gt_tokens else ""
            logger.info(
                f"  [t={t}] gt_pp={gt_perp:.4f} pred={predicted_text!r} pred_ids={predicted_tokens} "
                f"gt={gt_text!r} gt_ids={gt_tokens} (sample {sample_pos + 1}/{len(sample_indices)})"
            )
            last_perp = gt_perp
            last_pred = predicted_text
            last_gt = gt_text
            perplexities[t] = gt_perp
            predicted_texts[t] = predicted_text
            gt_texts[t] = gt_text

        # --- Input axis 1: the language prompt -------------------------------
        # Ground truth keeps the subtask cached in the .pkl. Decoded forward-fills
        # the decoded subtask across all frames (piecewise-constant between decode
        # samples) and rebuilds each prompt as `task_description + subtask + "\n"`.
        # Runs on EVERY rank (predict_values is SPMD; sampled_decoded_text is
        # identical across ranks).
        if eval_config.condition_on_decoded_subtask:
            filled_subtask = ""
            value_frames = []
            for i, f in enumerate(frames):
                if i in sampled_decoded_text:
                    filled_subtask = sampled_decoded_text[i] or ""
                tok, msk, s0, s1 = _transforms._tokenize_robocoin_subtask_prompt(
                    val_tokenizer, prefix_text, filled_subtask, append_newline = True,
                )
                nf = dict(f)
                nf["tokenized_prompt"] = np.asarray(tok)
                nf["tokenized_prompt_mask"] = np.asarray(msk)
                nf["subtask_start_index"] = np.int32(s0)
                nf["subtask_end_index"] = np.int32(s1)
                value_frames.append((traj_idx, i, nf))
        else:
            value_frames = seg_all_frames

        preds, _, _, preds_cf, _, _ = predict_values(
            model, value_frames, seg_mc, action_conditioned, batch_size = batch_size, mesh = mesh,
        )

        # --- Input axis 2: the action ----------------------------------------
        # Independent of the prompt axis above: dataset action vs the highest-value
        # cached counterfactual (policy-generated) action.
        if eval_config.counterfactual_value_action:
            if not preds_cf.get(traj_idx):
                raise ValueError(
                    f"--counterfactual-value-action set but no counterfactual predictions for {traj_idx}. "
                    "The cached action store must be joined for this split (it is joined on the split "
                    "that has a counterfactual_action_store-<split> shard; see the 'skipping join' warning)."
                )
            predicted_values = preds_cf[traj_idx]
        else:
            predicted_values = preds[traj_idx]

        if not is_rank0:
            continue

        # Persist all per-frame value predictions, GT subtask boundaries, and the
        # subtask predictions decoded at `step`-frame intervals. The video below
        # is rendered either way.
        if npz_mode:
            import io as _io
            from etils import epath

            sample_t = np.asarray(sample_indices, dtype = np.int32)
            predicted_subtasks = np.asarray([predicted_texts[t] or "" for t in sample_indices])
            gt_subtasks = np.asarray([gt_texts[t] or "" for t in sample_indices])
            npz_path = f"{eval_config.subtask_npz_dir.rstrip('/')}/{traj_idx}.npz"
            # Serialize to bytes, then write via epath so a gs:// subtask_npz_dir
            # works as well as local/NFS (np.savez can't write to gs:// directly).
            _buf = _io.BytesIO()
            np.savez(
                _buf,
                predicted_values = np.asarray(predicted_values, dtype = np.float64),
                subtask_boundaries = np.asarray(_subtask_boundary_indices(frames), dtype = np.int32),
                subtask_pred_t = sample_t,
                predicted_subtasks = predicted_subtasks,
                gt_subtasks = gt_subtasks,
                num_frames = np.int32(len(frames)),
            )
            _out = epath.Path(npz_path)
            _out.parent.mkdir(parents = True, exist_ok = True)
            _out.write_bytes(_buf.getvalue())
            logger.info(
                f"Saved subtask npz to {npz_path}: {len(predicted_values)} value preds, "
                f"{len(sample_t)} subtask decodes @ stride {step}."
            )

        # Forward-fill the per-frame display state so the video shows the most
        # recent decode + perplexity reading on every intermediate frame.
        for t in range(len(frames)):
            if perplexities[t] is None:
                perplexities[t] = last_perp if last_perp is not None else float("nan")
            if predicted_texts[t] is None:
                predicted_texts[t] = last_pred
            if gt_texts[t] is None:
                gt_texts[t] = last_gt
            # Track-the-leader for frames before the first sample (rare; happens only
            # if sample_indices is empty, which shouldn't occur).
            if perplexities[t] is not None and not np.isnan(perplexities[t]):
                last_perp = perplexities[t]
            if predicted_texts[t] is not None:
                last_pred = predicted_texts[t]
            if gt_texts[t] is not None:
                last_gt = gt_texts[t]

        frame_images = [
            np.stack([
                np.asarray(f["image"]["left_wrist_0_rgb"]),
                np.asarray(f["image"]["right_wrist_0_rgb"]),
                np.asarray(f["image"]["base_0_rgb"]),
            ])
            for f in frames
        ]

        plot_key = (
            f"val/{repo_id.removeprefix('RoboCOIN/')}_episode_{ep_idx}{part_suffix}_subtask"
        )
        rendered[plot_key] = _render_subtask_video(
            mc_returns = mc_returns,
            predicted_values = predicted_values,
            perplexities = perplexities,
            predicted_texts = predicted_texts,
            gt_texts = gt_texts,
            frame_images = frame_images,
            fps = fps,
            ep_idx = ep_idx,
            subtask_texts = ep_subtasks[traj_idx],
            output_dir = eval_config.output_dir,
            plot_key = plot_key,
            condition_on_decoded = eval_config.condition_on_decoded_subtask,
        )

    if is_rank0 and eval_config.output_dir is None and rendered:
        import wandb

        wandb.log(rendered)


def _get_critic_network(model):
    """Return the value network, tolerating CQL critics that expose ``q_network``.

    SARSA/MC value functions store the encoder as ``model.network``; CQL stores it
    as ``model.q_network``. The rest of this script assumes ``model.network``.
    """
    net = getattr(model, "network", None)
    if net is None:
        net = getattr(model, "q_network", None)
    if net is None:
        raise ValueError("Critic model exposes neither .network nor .q_network.")
    return net


@nnx.jit
def _jitted_action_grad_sq_norm(model, obs, act):
    """Per-sample ``||grad_a Q(s, a)||^2`` and Q for an action-conditioned critic.

    Summing Q over the batch and differentiating w.r.t. the batched action is a
    standard trick: each sample's Q depends only on its own action, so the grad
    of the sum w.r.t. ``act`` yields per-sample gradients ``[B, action_horizon,
    action_dim]``. The squared L2 norm is taken over the full action chunk.
    """
    def _q_sum(a):
        out = model.compute_value(obs, a, take_min_over_ensemble = True)
        q = out[0] if isinstance(out, tuple) else out
        return jnp.sum(q), q

    grads, q = jax.grad(_q_sum, has_aux = True)(act)
    grad_sq_norm = jnp.sum(grads ** 2, axis = tuple(range(1, grads.ndim)))
    return grad_sq_norm, q


def _run_action_gradient_norm(
    model,
    data_config,
    action_horizon: int,
    config,
    val_input_transform,
    split: str,
    eval_config: EvalConfig,
    action_conditioned: bool,
) -> None:
    """Compute and save per-frame ``||grad_a Q(s, a)||^2`` over validation demos.

    Caches validation demos (allow_duplicate_repos so single-task datasets like
    real_shirt_hang yield multiple trajectories), filters to the first
    ``num_grad_demos`` demos with is_partial=False and has_subtask_annotations=True,
    and writes one ``<demo>.npz`` per selected demo into ``action_gradient_norm_dir``.
    """
    if not action_conditioned:
        raise ValueError("action_gradient_norm requires an action-conditioned (Q) critic.")

    if eval_config.grad_action_source not in {"dataset", "cached"}:
        raise ValueError(
            f"--grad-action-source must be 'dataset' or 'cached', got {eval_config.grad_action_source!r}."
        )
    # "cached" evaluates Q / grad at cached_action[0] = counterfactual_actions[:, 0].
    # get_obs_and_action now returns all cached candidates, so the [:, 0] is taken at
    # the call site below (a single action is needed for the gradient).
    action_prefix = "counterfactual_" if eval_config.grad_action_source == "cached" else ""

    critic_network = _get_critic_network(model)
    image_size = tuple(critic_network.config.image_size)

    out_dir = eval_config.action_gradient_norm_dir
    # Shared cache (grad_cache_dir) lets multiple runs reuse the identical cached
    # demos; otherwise each run caches under its own output dir.
    cache_dir = eval_config.grad_cache_dir or os.path.join(out_dir, "val_cache")

    # Cache val demos. Only demos with is_partial=False and
    # has_subtask_annotations=True are cached: a filtering generator drops the
    # rest before cache_val_episodes sees them, so it keeps pulling trajectories
    # until num_grad_demos passing demos are written. allow_duplicate_repos=True
    # so a single-repo dataset (real_shirt_hang) yields distinct per-episode pkls
    # (matches train_value_function.py's FT path).
    val_trajectory_dataset = _data_loader.create_rlds_dataset(
        data_config,
        action_horizon,
        config.batch_size,
        split = split,
        shuffle = False,
        return_trajectories = True,
    )

    def _filter_passing_trajectories(dataset):
        for traj in dataset:
            if len(traj["repo_id"]) == 0:
                continue
            is_partial = bool(np.asarray(traj["is_partial"][0]))
            has_annotations = bool(np.asarray(traj["has_subtask_annotations"][0]))
            if (not is_partial) and has_annotations:
                yield traj

    cache_val_episodes(
        _filter_passing_trajectories(val_trajectory_dataset),
        eval_config.num_grad_demos,
        cache_dir,
        include_repos = (),
        save_only = True,
        input_transform = val_input_transform,
        allow_duplicate_repos = True,
    )
    del val_trajectory_dataset

    traj_frames = _load_cached_trajectories(cache_dir)
    selected = [(key, traj_frames[key]) for key in sorted(traj_frames.keys()) if traj_frames[key]]
    if len(selected) < eval_config.num_grad_demos:
        logger.warning(
            "Cached only %d/%d demos with is_partial=False and has_subtask_annotations=True; "
            "the validation split may not contain enough passing demos.",
            len(selected), eval_config.num_grad_demos,
        )

    if action_prefix == "counterfactual_" and selected and "counterfactual_actions" not in selected[0][1][0]:
        raise ValueError(
            "grad_action_source='cached' requires cached counterfactual_actions, but none are "
            "present in the cached frames. The counterfactual action store is only joined on the "
            "train split (see rlds_dataset.py); use --split train and a config / "
            "--counterfactual-action-store-dir that points at a store covering this split."
        )

    os.makedirs(out_dir, exist_ok = True)
    batch_size = eval_config.batch_size
    for key, frames in selected:
        decode_episode_images(frames, image_size)
        grad_sq_norms: list[float] = []
        q_values: list[float] = []
        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start : batch_start + batch_size]
            num_real = len(batch_frames)
            # Pad partial last batches to a fixed leading-axis size so the JIT'd
            # backward isn't recompiled per trailing-batch length.
            if num_real < batch_size:
                batch_frames = batch_frames + [batch_frames[-1]] * (batch_size - num_real)
            obs, act = get_obs_and_action(batch_frames, prefix = action_prefix, action_conditioned = True)
            if action_prefix == "counterfactual_" and act is not None:
                act = act[:, 0]
            grad_sq_norm_np, q_np = jax.device_get(_jitted_action_grad_sq_norm(model, obs, act))
            grad_sq_norms.extend(grad_sq_norm_np[:num_real].tolist())
            q_values.extend(q_np[:num_real].tolist())

        repo_id = frames[0]["repo_id"]
        if isinstance(repo_id, np.ndarray):
            repo_id = repo_id.item()
        if isinstance(repo_id, bytes):
            repo_id = repo_id.decode("utf-8")

        npz_path = os.path.join(out_dir, f"{key}.npz")
        np.savez(
            npz_path,
            grad_sq_norm = np.asarray(grad_sq_norms, dtype = np.float64),
            predicted_value = np.asarray(q_values, dtype = np.float64),
            mc_return = np.asarray([float(np.asarray(f["mc_return"])) for f in frames], dtype = np.float64),
            frame_index = np.asarray([int(np.asarray(f["_frame_index"])) for f in frames], dtype = np.int32),
            episode_index = np.int32(int(np.asarray(frames[0]["episode_index"]))),
            repo_id = str(repo_id),
            num_frames = np.int32(len(frames)),
            action_source = str(eval_config.grad_action_source),
        )
        logger.info(
            "Saved action-gradient-norm npz to %s (%d frames, mean ||grad_a Q||^2=%.4e).",
            npz_path, len(frames), float(np.mean(grad_sq_norms)),
        )


def main(eval_config: EvalConfig):
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)

    if eval_config.split not in {"train", "val"}:
        raise ValueError(f"--split must be 'train' or 'val', got {eval_config.split!r}.")
    if eval_config.counterfactual_best_of_n and eval_config.counterfactual_action_store_dir is None:
        raise ValueError("--counterfactual-action-store-dir is required with --counterfactual-best-of-n.")
    if eval_config.counterfactual_best_of_n and eval_config.policy_checkpoint_path is None:
        raise ValueError("--policy-checkpoint-path is required with --counterfactual-best-of-n.")

    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    train_module = load_train_module()
    def _config_override(config):
        config = dataclasses.replace(config, include_repos = eval_config.include_repos)
        if eval_config.counterfactual_action_store_dir is not None:
            config = dataclasses.replace(
                config,
                data = dataclasses.replace(config.data, counterfactual_action_store_dir = eval_config.counterfactual_action_store_dir),
            )
        if eval_config.rlds_data_dir is not None:
            config = dataclasses.replace(
                config,
                data = dataclasses.replace(config.data, rlds_data_dir = eval_config.rlds_data_dir),
            )
        return dataclasses.replace(
            config,
            num_val_trajectories = eval_config.num_trajectories,
        )

    resolved_fsdp_devices = eval_config.fsdp_devices or jax.device_count()
    model, critic_norm_stats, config, critic_step = load_critic(
        eval_config.config_name,
        eval_config.checkpoint_path,
        fine_tune = eval_config.fine_tune,
        step = eval_config.step,
        config_override = _config_override,
        fsdp_devices = resolved_fsdp_devices,
    )
    # Rebuild the same mesh load_critic used so we can shard inference inputs
    # consistently with the restored param sharding. With fsdp_devices=1 this is
    # (device_count, 1) → data-parallel: replicated params, batch-axis-split inputs.
    # Single-host only: on multi-host TPU, device_put of a host-local array across
    # the global mesh is invalid, so leave inputs unsharded there and rely on the
    # FSDP-sharded params (the original multi-host path).
    inference_mesh = _sharding.make_mesh(resolved_fsdp_devices) if jax.process_count() == 1 else None
    action_conditioned = _get_critic_network(model).action_conditioned
    logger.info(f"Loaded model from {eval_config.checkpoint_path}, action_conditioned={action_conditioned}")

    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.action_horizon or config.model.action_horizon
    val_tokenizer = config.data._get_critic_tokenizer(config.model)

    val_input_transform = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
        *([_transforms.Clip(data_config.clip_normalized_bounds)] if data_config.clip_normalized_bounds is not None else []),
        *data_config.model_transforms.inputs,
        _config.AddValidationVariants(
            val_tokenizer,
            use_quantile_norm = data_config.use_quantile_norm,
        ),
    ])

    cache_dir = _resolve_eval_cache_dir(eval_config)
    split = data_config.val_split if eval_config.split == "val" else eval_config.split

    if eval_config.action_gradient_norm_dir is not None:
        logger.info(
            "Action-gradient-norm mode: computing ||grad_a Q(s, a)||^2 over %d validation demos -> %s",
            eval_config.num_grad_demos, eval_config.action_gradient_norm_dir,
        )
        _run_action_gradient_norm(
            model = model,
            data_config = data_config,
            action_horizon = action_horizon,
            config = config,
            val_input_transform = val_input_transform,
            split = split,
            eval_config = eval_config,
            action_conditioned = action_conditioned,
        )
        logger.info("Action-gradient-norm computation complete.")
        return

    cache_complete = False
    # epath.iterdir handles gs:// cache dirs (virtual prefix) as well as local/NFS.
    from etils import epath

    existing_pkls: set[str] = set()
    try:
        existing_pkls = {p.name for p in epath.Path(cache_dir).iterdir() if p.name.endswith(".pkl")}
    except (FileNotFoundError, NotADirectoryError, OSError):
        existing_pkls = set()
    if existing_pkls:
        required_pkls = {f"{repo.replace('/', '__')}.pkl" for repo in config.include_repos}
        if required_pkls.issubset(existing_pkls) and len(existing_pkls) >= eval_config.num_trajectories:
            cache_complete = True
            logger.info(f"Cache complete at {cache_dir} ({len(existing_pkls)} pkls); skipping create_rlds_dataset")

    # cache_val_episodes is multi-worker by design: process 0 fills the
    # full-coverage slots, workers > 0 cache the include_repos. Run it on ALL
    # workers (matching train_value_function.py) — gating to process 0 drops
    # the per-worker include_repos coverage.
    if not cache_complete:
        val_trajectory_dataset = _data_loader.create_rlds_dataset(
            data_config,
            action_horizon,
            config.batch_size,
            split = split,
            shuffle = False,
            return_trajectories = True,
        )
        cache_val_episodes(
            val_trajectory_dataset,
            eval_config.num_trajectories,
            cache_dir,
            include_repos = config.include_repos,
            save_only = True,
            input_transform = val_input_transform,
            # Mirrors train_value_function.py's FT path: a fine-tune config targets a
            # single-repo dataset, so without this every val episode collapses onto one
            # repo_key and only one trajectory is ever cached.
            allow_duplicate_repos = eval_config.fine_tune is not None,
        )
        del val_trajectory_dataset

    if jax.process_count() > 1:
        jax.experimental.multihost_utils.sync_global_devices("eval_cache_write")

    if eval_config.cache_only:
        logger.info("--cache-only set: validation cache populated; exiting before plotting.")
        return

    override_prompt = None
    if eval_config.override_task_prompt is not None:
        critic_prompt_mode = getattr(config.data, "prompt_mode", None)
        if critic_prompt_mode == "task_description_predict_current_subtask":
            # Match the serving/eval path's critic tokenization (best_of_n_policy.py):
            # tokenize the task-description prefix only — empty subtask suffix, no
            # trailing newline. The subtask indices are discarded (cached val frames
            # carry none, so the critic forward runs with subtask_start_index=None).
            override_tokens, override_mask, _, _ = _transforms._tokenize_robocoin_subtask_prompt(
                val_tokenizer, eval_config.override_task_prompt, "", append_newline = False,
            )
            override_prompt = (override_tokens, override_mask)
        else:
            override_prompt = val_tokenizer.tokenize(eval_config.override_task_prompt)
        logger.info(
            f"Overriding tokenized_prompt with {eval_config.override_task_prompt!r} "
            f"(prompt_mode={critic_prompt_mode!r})"
        )

    if eval_config.output_dir is None and jax.process_index() == 0:
        import wandb

        wandb.init(
            project = eval_config.project_name,
            name = eval_config.wandb_run_name or f"eval_{eval_config.config_name}",
            config = dataclasses.asdict(eval_config),
        )

    # When the critic was trained with prompt_mode="task_description_predict_current_subtask",
    # the subtask video (rendered below) already carries the camera + value
    # subplots plus the per-second predicted/GT subtask titles and perplexity
    # curve. Skip the default 4-subplot video entirely — it's a duplicate, and
    # the value predictions are recomputed inside _run_subtask_prediction.
    critic_prompt_mode = (
        getattr(config.data, "prompt_mode", None)
        or getattr(config.data, "subtask_prompt_mode", None)
    )
    subtask_mode = (
        critic_prompt_mode == "task_description_predict_current_subtask"
        and not eval_config.disable_subtask_decoding
    )
    if not subtask_mode:
        train_module.generate_validation_plots_dlimp(
            model = model,
            val_episode_indices = list(range(eval_config.num_trajectories)),
            step = 0,
            action_conditioned = action_conditioned,
            data_config = data_config,
            cache_dir = cache_dir,
            output_dir = eval_config.output_dir,
            batch_size = eval_config.batch_size,
            override_prompt = override_prompt,
        )
    else:
        logger.info(
            "Subtask mode: skipping generate_validation_plots_dlimp; the subtask "
            "video below carries the camera + value + perplexity subplots."
        )
        logger.info(
            "Running subtask prediction (per-second autoregressive decode + GT perplexity)..."
        )
        image_size = data_config.rlds_kwargs.get("image_size", (224, 224))
        _run_subtask_prediction(
            model = model,
            val_tokenizer = val_tokenizer,
            cache_dir = cache_dir,
            image_size = tuple(image_size),
            eval_config = eval_config,
            batch_size = eval_config.batch_size,
            action_conditioned = action_conditioned,
            mesh = inference_mesh,
        )

    if eval_config.counterfactual_best_of_n and action_conditioned:
        logger.info("Running BestOfN counterfactual evaluation...")
        import jax.numpy as jnp

        if eval_config.policy_config is None:
            raise ValueError(
                "counterfactual_best_of_n requires --policy-config. "
                f"Available: {sorted(_POLICY_CONFIGS.keys())}"
            )
        policy_cfg = get_policy_config(eval_config.policy_config)

        traj_frames = _load_cached_trajectories(cache_dir)
        split_traj_frames, traj_to_repo_ep, ep_subtasks = _split_trajectory_frames(traj_frames)

        first_frames = next(iter(split_traj_frames.values()))
        num_samples = first_frames[0]["counterfactual_actions"].shape[0]
        network_config = config.model.network_config

        policy_norm_stats_dir = os.path.join(eval_config.policy_checkpoint_path, "assets", data_config.asset_id)
        policy_norm_stats = _normalize.load(policy_norm_stats_dir)
        logger.info(f"Loaded policy norm stats from {policy_norm_stats_dir}")

        bon_model = BestOfNWrapper(
            action_dim = network_config.action_dim,
            action_horizon = action_horizon,
            max_token_len = network_config.max_token_len,
            base_model = None,
            num_samples = num_samples,
            take_min_over_ensemble = policy_cfg.take_min_over_ensemble,
            use_target_value = policy_cfg.use_target_value,
            convert_to_global = policy_cfg.convert_to_global,
            selection_mode = policy_cfg.selection_mode,
            softmax_temperature = policy_cfg.softmax_temperature,
            policy_norm_stats = policy_norm_stats,
            critic_norm_stats = critic_norm_stats,
        )

        @nnx.jit
        def _jitted_bon_eval(bon, vf, rng, obs, action, cf_actions):
            transition = _base_vf.Transition(
                observation = obs,
                action = action,
                counterfactual_actions = cf_actions,
            )
            best_action = bon.sample_actions(rng, transition, compute_next_action = False, value_function = vf)
            result = vf.compute_value(obs, best_action, take_min_over_ensemble = True)
            q_value = result[0] if isinstance(result, tuple) else result
            return q_value

        BATCH_SIZE = 8
        bon_rng = jax.random.PRNGKey(86)
        ep_mc_returns = {}
        ep_frame_images = {}
        ep_fps = {}
        ep_include_masks = {}
        bon_predictions: dict[str, list[float]] = {}

        for traj_idx, frames in split_traj_frames.items():
            ep_mc_returns[traj_idx] = [f["mc_return"] for f in frames]
            ep_frame_images[traj_idx] = [
                np.stack([
                    np.asarray(f["image"]["left_wrist_0_rgb"]),
                    np.asarray(f["image"]["right_wrist_0_rgb"]),
                    np.asarray(f["image"]["base_0_rgb"]),
                ])
                for f in frames
            ]
            ep_fps[traj_idx] = int(frames[0]["fps"])
            ep_include_masks[traj_idx] = [bool(f.get("include_subtask", True)) for f in frames]
            bon_predictions[traj_idx] = []

            for batch_start in range(0, len(frames), BATCH_SIZE):
                batch_frames = frames[batch_start : batch_start + BATCH_SIZE]
                obs, action = get_obs_and_action(batch_frames, prefix = "", action_conditioned = True)
                cf_actions = jnp.asarray(np.stack([f["counterfactual_actions"] for f in batch_frames], axis = 0))
                bon_rng, step_rng = jax.random.split(bon_rng)
                q_values = jax.device_get(_jitted_bon_eval(bon_model, model, step_rng, obs, action, cf_actions))
                bon_predictions[traj_idx].extend(q_values.tolist())

        total = sum(len(preds) for preds in bon_predictions.values())
        logger.info(f"Computed {total} BestOfN predictions")

        if jax.process_index() == 0:
            best_of_n_images = {}
            for traj_idx in ep_mc_returns:
                repo_id, ep_idx, part_suffix = traj_to_repo_ep[traj_idx]
                plot_key = f"val/{repo_id.removeprefix('RoboCOIN/')}_episode_{ep_idx}{part_suffix}_best_of_n"
                mc_returns = ep_mc_returns[traj_idx]
                include_masks = ep_include_masks[traj_idx]
                predicted_values = bon_predictions[traj_idx]

                if len(predicted_values) != len(mc_returns):
                    raise ValueError(
                        f"BestOfN prediction length mismatch for {traj_idx}: "
                        f"{len(predicted_values)} predictions vs {len(mc_returns)} returns."
                    )

                filtered_mc = [mc for mc, include in zip(mc_returns, include_masks, strict = True) if include]
                filtered_pred = [pred for pred, include in zip(predicted_values, include_masks, strict = True) if include]
                filtered_images = [img for img, include in zip(ep_frame_images[traj_idx], include_masks, strict = True) if include]

                if not filtered_mc:
                    continue

                best_of_n_images[plot_key] = train_module._create_value_plot(
                    filtered_mc,
                    filtered_pred,
                    ep_idx,
                    0,
                    " (BestOfN Q)",
                    oracle_values = None,
                    subtask_texts = ep_subtasks[traj_idx],
                    plot_video = True,
                    frame_images = filtered_images,
                    fps = ep_fps[traj_idx],
                    output_dir = eval_config.output_dir,
                    plot_key = plot_key,
                )

            if eval_config.output_dir is None and best_of_n_images:
                import wandb

                wandb.log(best_of_n_images)

    # Keep all hosts alive until rank 0 has fully completed async rendering/logging.
    if (
        jax.process_index() == 0
        and train_module._render_thread is not None
        and train_module._render_thread.is_alive()
    ):
        logger.info("Waiting for render thread to finish")
        train_module._render_thread.join()
        train_module._render_thread = None

    if jax.process_count() > 1:
        logger.info("Waiting at post-render multihost barrier")
        jax.experimental.multihost_utils.sync_global_devices("evaluate_value_function_post_render_join")

    if eval_config.output_dir is None and jax.process_index() == 0:
        import wandb

        wandb.finish()

    logger.info("Evaluation complete")


if __name__ == "__main__":
    import tyro

    eval_config = tyro.cli(EvalConfig)
    main(eval_config)
