"""Reusable subtask-token decoder for PaliGemma value-function critics.

Extracted from `SubtaskPredictorPolicy` so any policy that already holds a
loaded PaliGemma critic (e.g. `BestOfNPolicy`) can optionally decode the current
subtask off the same observation, without re-forking the whole policy class.

Given a critic `Observation` (images + critic-tokenized prompt), the critic
autoregressively decodes up to `max_tokens` subtask tokens via a KV-cache fast
path that mirrors the training-time prefix attention pattern:

- gemma_2b: dynamic cache that grows by one per step (decode step is NOT JIT'd,
  so it does not recompile on every length).
- gemma4: fixed-size cache (decode step IS JIT'd, compiled once and reused).

The decode dominates per-call latency (~14s on L40S), so callers gate it with
`should_decode(force=...)`: it runs only once every `decode_every` calls, or
immediately when `force=True` (e.g. to decode a terminal frame on demand).
"""

from __future__ import annotations

import logging
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model

logger = logging.getLogger(__name__)

# The env's known subtask strings, used as early-exit targets during decoding.
DEFAULT_SUBTASK_TARGETS = (
    "Insert the white block into the pink one",
    "Insert the right end of this combination into the blue block",
    "Place the combination on the wooden platform",
)


def critic_value_network(critic_model: Any) -> Any:
    """Resolve the PaliGemma value network used for subtask decoding.

    SARSAValueFunction names the backbone `network`; CQLValueFunction exposes it
    as `q_network`. Accept either so the decoder works with both critic flavors.
    """
    net = getattr(critic_model, "network", None)
    if net is None:
        net = critic_model.q_network
    return net


class SubtaskDecoder:
    """Autoregressively decodes the current subtask from a loaded critic.

    Args:
        critic_model: A loaded value-function model whose backbone is a
            PaliGemma value network (resolved via `critic_value_network`).
        critic_tokenizer: The critic's tokenizer (used to decode token ids back
            to text and to build the early-exit target sequences).
        decode_every: Run the decode only once every N `should_decode()` calls.
            Use a large value to keep per-infer latency low; `force=True`
            overrides it for on-demand decodes.
        max_tokens: Max subtask tokens to decode per call.
        subtask_targets: Known subtask strings; decoding stops early once the
            emitted prefix exactly matches one of them.
    """

    def __init__(
        self,
        critic_model: Any,
        critic_tokenizer: Any,
        *,
        decode_every: int = 20,
        max_tokens: int = 16,
        subtask_targets: tuple[str, ...] = DEFAULT_SUBTASK_TARGETS,
    ) -> None:
        self._critic_model = critic_model
        self._critic_tokenizer = critic_tokenizer
        self.decode_every = int(decode_every)
        self.max_tokens = int(max_tokens)
        self._call_count = 0
        self._eos_id = self._resolve_eos_id()
        self._build_closures()
        self._build_targets(subtask_targets)

    # ------------------------------------------------------------------ API
    def should_decode(self, *, force: bool = False) -> bool:
        """Decide whether to decode on this call, advancing the cadence counter.

        Returns True when `force` is set or the per-call counter hits a multiple
        of `decode_every`. Call exactly once per infer so the cadence advances
        whether or not the decode actually runs.
        """
        decode = force or (self.decode_every > 0 and self._call_count % self.decode_every == 0)
        self._call_count += 1
        return decode

    def predict(self, critic_observation: _model.Observation) -> dict[str, Any]:
        """Decode the subtask; returns the fields to merge into an infer result."""
        tokens, text, perplexity = self._predict_tokens(critic_observation)
        logger.info(f"[subtask_predict] tokens={tokens} decoded={text!r} perplexity={perplexity:.4f}")
        return {
            "predicted_subtask": text,
            "predicted_subtask_tokens": list(map(int, tokens)),
            "subtask_perplexity": perplexity,
        }

    def run_lockstep(self, critic_observation: _model.Observation) -> None:
        """Drive the same JIT chain on a participating (non-rank-0) host.

        Each `predict` call is a sequence of cross-host JIT collectives; on
        multi-host setups every rank must run them in lockstep or the collectives
        desync. Outputs are discarded here. No-op effect on single-host.
        """
        self._predict_tokens(critic_observation)

    # ------------------------------------------------------------- internals
    def _resolve_eos_id(self) -> int | None:
        eos = getattr(self._critic_tokenizer, "eos_token_id", None)
        if eos is None:
            inner = getattr(self._critic_tokenizer, "tokenizer", None)
            eos = inner.eos_id() if inner is not None and hasattr(inner, "eos_id") else None
        return eos

    def _build_closures(self) -> None:
        from openpi.value_functions.networks.paligemma import NUM_PATCHES_PER_IMAGE
        from openpi.value_functions.networks.paligemma import compute_rope_positions
        from openpi.value_functions.networks.paligemma import make_attn_mask

        net = critic_value_network(self._critic_model)
        is_gemma4 = "gemma4" in getattr(getattr(net, "config", None), "paligemma_variant", "")
        self._is_gemma4 = is_gemma4

        if not is_gemma4:
            @nnx.jit
            def _prefix_forward(critic_model, observation):
                # gemma_2b path: replicates `compute_prefix_cache` with
                # suffix_mask=None (no subtask suffix in the prompt yet) so
                # images + task-description text get full bidirectional
                # attention exactly as at training time.
                n = critic_value_network(critic_model)
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
                last_hidden = jnp.take_along_axis(
                    hidden, last_text_pos[:, None, None], axis = 1,
                )
                return last_hidden, kv_cache, prefix_mask, last_text_pos

            def _decode_step(critic_model, token_id, kv_cache, prefix_mask, last_text_pos, suffix_pos_so_far):
                # gemma_2b path. NOT JIT'd: kv_cache extends by 1 every call,
                # so a JIT'd version would recompile 16 times.
                n = critic_value_network(critic_model)
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
            # gemma4 path: leverages `_build_gemma4_prefix_cache_inputs` from
            # the value-network for the prefix forward (fixed-size cache,
            # left-aligned write-into-index semantics). Each decode step
            # writes a single new K/V at end_index and advances by one — the
            # cache shape stays constant across calls, so JIT can apply.
            @nnx.jit
            def _prefix_forward(critic_model, observation):
                n = critic_value_network(critic_model)
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
                # Position of the last valid prompt token in the prefix; same
                # accounting as the gemma_2b path but using gemma4's text_end.
                num_soft = n._num_soft_tokens_per_image
                tokens_per_block = num_soft + 4
                text_start = 1 + n._num_cameras * tokens_per_block
                last_text_pos = text_start + jnp.sum(
                    obs.tokenized_prompt_mask.astype(jnp.int32), axis = 1,
                ) - 1
                last_hidden = jnp.take_along_axis(
                    hidden, last_text_pos[:, None, None], axis = 1,
                )
                # Return the cache_size + prefix_len so decode_step can build
                # the fixed-width [cache_size] mask.
                return (
                    last_hidden, kv_cache, prefix_mask, last_text_pos,
                    jnp.asarray(prefix_inputs["prefix_len"]),
                    jnp.asarray(prefix_inputs["cache_size"]),
                )

            @nnx.jit(static_argnames = ("suffix_pos_so_far", "prefix_len", "cache_size"))
            def _decode_step(critic_model, token_id, kv_cache, prefix_mask, last_text_pos,
                             prefix_len, cache_size, suffix_pos_so_far):
                # gemma4 cache is fixed-size; shapes never change between
                # iterations, so JIT compiles once and re-uses. prefix_len /
                # cache_size / suffix_pos_so_far are all static because they
                # determine the shape of the suffix region.
                n = critic_value_network(critic_model)
                tok_arr = jnp.asarray(token_id, dtype = jnp.int32).reshape(1, 1)
                tok_embed = n.PaliGemma.llm(tok_arr, method = "embed")
                per_layer_input = None
                if n._gemma4_per_layer_input_dim > 0:
                    per_layer_input = n.PaliGemma.llm(
                        tok_embed, tok_arr, method = "encode_per_layer_input",
                    )
                # Build the [B=1, 1, cache_size] attn_mask: bidirectional to
                # prefix-valid positions; causal within the suffix region —
                # this query token reads previously-emitted suffix slots
                # plus itself.
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
        def _logits_from_hidden(critic_model, hidden):
            return critic_value_network(critic_model).decode(hidden)

        self._prefix_forward = _prefix_forward
        self._decode_step = _decode_step
        self._logits = _logits_from_hidden

    def _build_targets(self, subtask_targets: tuple[str, ...]) -> None:
        # Pre-tokenize the known subtask strings the SAME way the autoregressive
        # decoder emits — bypass the public `tokenize()` (which adds a leading
        # BOS for paligemma and always appends "\n") and call the underlying
        # SentencePieceProcessor with add_bos=False directly so the resulting ids
        # align with the model's own token stream.
        inner_spp = getattr(self._critic_tokenizer, "_tokenizer", None)
        target_seqs: list[list[int]] = []
        for s in subtask_targets:
            if inner_spp is not None:
                tok_ids = [int(t) for t in inner_spp.encode(s, add_bos = False)]
            else:
                # Fallback: tokenize via public API, then strip BOS prefix and
                # trailing "\n" token (id 108 in gemma sentencepiece).
                toks, mask = self._critic_tokenizer.tokenize(s, None)
                n_valid = int(np.asarray(mask).sum())
                tok_ids = [int(t) for t in np.asarray(toks)[:n_valid].tolist()]
                if tok_ids and tok_ids[0] == 2:    # BOS
                    tok_ids = tok_ids[1:]
                if tok_ids and tok_ids[-1] == 108:  # "\n"
                    tok_ids = tok_ids[:-1]
            target_seqs.append(tok_ids)
        self._target_seqs = target_seqs
        logger.info(
            f"[subtask_predict] target token sequences (len each): "
            f"{[len(s) for s in target_seqs]}"
        )

    def _predict_tokens(self, critic_obs: _model.Observation) -> tuple[list[int], str, float]:
        out = self._prefix_forward(self._critic_model, critic_obs)
        # gemma_2b prefix returns (hidden, cache, prefix_mask, last_text_pos);
        # gemma4 also carries (prefix_len, cache_size) for the fixed-width mask.
        if self._is_gemma4:
            last_hidden, kv_cache, prefix_mask, last_text_pos, prefix_len, cache_size = out
            # Convert to Python ints OUTSIDE the JIT so `_decode_step` can take
            # them as static_argnames (they drive the suffix mask shape).
            prefix_len = int(np.asarray(prefix_len))
            cache_size = int(np.asarray(cache_size))
        else:
            last_hidden, kv_cache, prefix_mask, last_text_pos = out
        logits = self._logits(self._critic_model, last_hidden)
        # log-softmax + select chosen token's log-probability for the
        # autoregressive perplexity accumulator.
        log_probs = jax.nn.log_softmax(logits[0, 0])
        next_tok = int(np.asarray(jnp.argmax(logits[0, 0])))
        chosen_logp = float(np.asarray(log_probs[next_tok]))
        predicted: list[int] = [next_tok]
        total_neg_logp: float = -chosen_logp
        for k in range(1, self.max_tokens):
            if self._eos_id is not None and next_tok == self._eos_id:
                break
            # Early-exit once the emitted prefix exactly matches a known subtask.
            if any(predicted == seq for seq in self._target_seqs):
                break
            if self._is_gemma4:
                hidden, kv_cache = self._decode_step(
                    self._critic_model, next_tok, kv_cache, prefix_mask, last_text_pos,
                    prefix_len, cache_size, k - 1,
                )
            else:
                hidden, kv_cache = self._decode_step(
                    self._critic_model, next_tok, kv_cache, prefix_mask, last_text_pos, k - 1,
                )
            logits = self._logits(self._critic_model, hidden)
            log_probs = jax.nn.log_softmax(logits[0, 0])
            next_tok = int(np.asarray(jnp.argmax(logits[0, 0])))
            chosen_logp = float(np.asarray(log_probs[next_tok]))
            total_neg_logp += -chosen_logp
            predicted.append(next_tok)
        if hasattr(self._critic_tokenizer, "decode"):
            decoded = self._critic_tokenizer.decode(predicted)
        else:
            inner = getattr(self._critic_tokenizer, "tokenizer", None)
            decoded = inner.decode(predicted) if inner is not None else " ".join(map(str, predicted))
        # PP = exp(-(1/T) * sum_t log p(x_t | x_<t)); using the argmax token at
        # each step (greedy decode), so this is the perplexity of the model's
        # own most-confident path conditioned on the current observation.
        n_tokens = max(1, len(predicted))
        perplexity = float(np.exp(total_neg_logp / n_tokens))
        return predicted, str(decoded), perplexity
