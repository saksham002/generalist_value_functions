"""Gemma 4 LLM backbone for value function training.

Thin wrapper around the upstream Gemma 4 implementation at
the installed `gemma` package.

Imports `Embedder`, `Block`, `RMSNorm`, `AttentionType` from the fork and
exposes the same API surface as `openpi.models.gemma3.Module` so the existing
`PaliGemmaValueNetwork` can call it via the NNX bridge.

Two execution paths:

- Default (per-layer for-loop): each Block is individually `nn.remat`'d.
  Original behavior; supports KV-cache inference.
- ``stacked_layer_params=True``: per-layer params are stacked into one tensor
  per (FFW-group x attn-type); a single Block template is shared via
  `Module.apply` with sliced params in a manual layer for-loop. Cuts FSDP
  all-gathers from N to 4. Layer execution order preserved.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Literal

import flax.linen as nn
import flax.traverse_util as _traverse_util
import jax
import jax.numpy as jnp

from gemma.gm.nn.gemma4 import _config as _fork_config
from gemma.gm.nn.gemma4 import _layers as _fork_layers
from gemma.gm.nn.gemma4 import _modules as _fork_mod

import openpi.shared.array_typing as at

GEMMA4_VOCAB_SIZE = 262_144

# Keys for the KV cache layout we use: dict[str, dict[str, jax.Array]].
# Outer key is "layer_{i}"; inner dict follows the upstream LayerCache shape
# {"k", "v", "end_index", "positions"}.
KVCache = dict


# 0 -> LOCAL_SLIDING, 1 -> GLOBAL (matches gemma3.py `attn_pattern` convention).
_ATTN_TYPE_MAP = {
    0: _fork_mod.AttentionType.LOCAL_SLIDING,
    1: _fork_mod.AttentionType.GLOBAL,
}


@dataclasses.dataclass
class Config:
    """Configuration for the Gemma 4 transformer."""

    embed_dim: int
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: int
    # Per-layer attention pattern (0=LOCAL_SLIDING, 1=GLOBAL). Length equals num_layers.
    attn_pattern: tuple[int, ...]
    # Partial RoPE: fraction of head dims to rotate. Gemma 4 rotates only 25%
    # of the head dim for global layers.
    global_rope_proportion: float = 0.25
    local_rope_proportion: float = 1.0
    local_rope_base_freq: int = 10_000
    global_rope_base_freq: int = 1_000_000
    global_rope_scale_factor: float = 1.0
    local_rope_scale_factor: float = 1.0
    # Separate KV projection size for global layers (E2B: 512 vs head_dim=256).
    global_key_size: int | None = None
    # Number of KV heads for global layers (if different from local).
    num_global_kv_heads: int | None = None
    # Whether K == V for global layers (31B only).
    k_eq_v_global: bool = False
    # Per-layer input embedding dim (0 to disable).
    per_layer_input_dim: int = 0
    # Vision token projection dim for `Embedder.encode_vision` (should match
    # the vision encoder's `d_model`).
    vision_proj_dim: int | None = None
    qk_norm_with_scale: bool = True
    vocab_size: int = GEMMA4_VOCAB_SIZE
    final_logit_softcap: float | None = 30.0
    kv_cache_sharing_frac_shared_layers: float = 0.0
    kv_cache_sharing_share_global: bool = False
    kv_cache_sharing_share_local: bool = False
    override_kv_shared_ffw_hidden: int | None = None


Variant = Literal["gemma4_dummy", "gemma4_e2b", "gemma4_e4b", "gemma4_31b"]


def _repeat_pattern(pattern: tuple[int, ...], num_layers: int) -> tuple[int, ...]:
    out = pattern * (num_layers // len(pattern))
    if num_layers % len(pattern) != 0:
        out += pattern[: num_layers % len(pattern)]
    return out


def get_config(variant: Variant) -> Config:
    """Returns config for the specified Gemma 4 variant."""
    if variant == "gemma4_dummy":
        return Config(
            embed_dim = 64,
            num_layers = 6,
            hidden_dim = 128,
            num_heads = 4,
            num_kv_heads = 1,
            head_dim = 16,
            sliding_window_size = 32,
            attn_pattern = _repeat_pattern((0, 0, 0, 0, 1), num_layers = 6),
            global_key_size = None,
            per_layer_input_dim = 0,
            vocab_size = 512,
            vision_proj_dim = 64,
        )
    if variant == "gemma4_e2b":
        return Config(
            embed_dim = 1536,
            num_layers = 35,
            hidden_dim = 6144,  # embed_dim * 4
            num_heads = 8,
            num_kv_heads = 1,
            head_dim = 256,
            sliding_window_size = 512,
            attn_pattern = _repeat_pattern((0, 0, 0, 0, 1), num_layers = 35),
            global_rope_proportion = 0.25,
            local_rope_proportion = 1.0,
            global_key_size = 512,
            num_global_kv_heads = None,
            k_eq_v_global = False,
            per_layer_input_dim = 256,
            vision_proj_dim = 768,
            qk_norm_with_scale = True,
            kv_cache_sharing_frac_shared_layers = 20.0 / 35.0,
            kv_cache_sharing_share_global = True,
            kv_cache_sharing_share_local = True,
            override_kv_shared_ffw_hidden = int(1536 * 4 * 2),
        )
    if variant == "gemma4_e4b":
        return Config(
            embed_dim = 2560,
            num_layers = 42,
            hidden_dim = 10240,  # embed_dim * 4
            num_heads = 8,
            num_kv_heads = 2,
            head_dim = 256,
            sliding_window_size = 512,
            attn_pattern = _repeat_pattern((0, 0, 0, 0, 0, 1), num_layers = 42),
            global_rope_proportion = 0.25,
            local_rope_proportion = 1.0,
            global_key_size = 512,
            num_global_kv_heads = None,
            k_eq_v_global = False,
            per_layer_input_dim = 256,
            vision_proj_dim = 768,
            qk_norm_with_scale = True,
            kv_cache_sharing_frac_shared_layers = 18.0 / 42.0,
            kv_cache_sharing_share_global = True,
            kv_cache_sharing_share_local = True,
        )
    if variant == "gemma4_31b":
        raise NotImplementedError(f"Gemma 4 variant {variant!r} not yet implemented.")
    raise ValueError(f"Unknown variant: {variant!r}")


def _build_block_kwargs(config: Config, attn_type, hidden_dim_override: int | None = None):
    """Common kwargs for instantiating a fork ``Block`` from our ``Config``."""
    is_global = attn_type == _fork_mod.AttentionType.GLOBAL
    rope_base_frequency = (
        config.global_rope_base_freq if is_global else config.local_rope_base_freq
    )
    rope_scale_factor = (
        config.global_rope_scale_factor if is_global else config.local_rope_scale_factor
    )
    return dict(
        num_heads = config.num_heads,
        num_kv_heads = config.num_kv_heads,
        embed_dim = config.embed_dim,
        head_dim = config.head_dim,
        hidden_dim = hidden_dim_override if hidden_dim_override is not None else config.hidden_dim,
        use_post_attn_norm = True,
        use_post_ffw_norm = True,
        attn_type = attn_type,
        rope_base_frequency = rope_base_frequency,
        rope_scale_factor = rope_scale_factor,
        sliding_window_size = config.sliding_window_size,
        qk_norm_with_scale = config.qk_norm_with_scale,
        num_global_kv_heads = config.num_global_kv_heads,
        global_key_size = config.global_key_size,
        k_eq_v_global = config.k_eq_v_global,
        global_rope_proportion = config.global_rope_proportion,
        local_rope_proportion = config.local_rope_proportion,
        per_layer_input_dim = config.per_layer_input_dim,
    )



class Module(nn.Module):
    """Gemma 4 transformer module. Interface matches `gemma3.Module` for drop-in use."""

    configs: Sequence[Config]
    embed_dtype: str

    # Unused, kept for API parity with gemma3.Module.
    adarms: bool = False

    # Stack per-layer params per (FFW-group × attn-type) into single tensors;
    # index into the stacked tensor at each layer step. 4 FSDP all-gathers
    # (one per group) instead of one per layer.
    stacked_layer_params: bool = False

    def setup(self):
        config = self.configs[0]

        self.embedder = _fork_mod.Embedder(
            vocab_size = config.vocab_size,
            embed_dim = config.embed_dim,
            num_layers = config.num_layers,
            per_layer_input_dim = config.per_layer_input_dim,
            vision_proj_dim = config.vision_proj_dim,
            name = "embedder",
        )

        pattern = _repeat_pattern(config.attn_pattern, num_layers = config.num_layers)
        assert len(pattern) == config.num_layers, (
            f"attn_pattern length {len(pattern)} != num_layers {config.num_layers}"
        )
        attention_types = tuple(_ATTN_TYPE_MAP[int(flag)] for flag in pattern)
        sharing_config = None
        if config.kv_cache_sharing_frac_shared_layers > 0.0:
            sharing_config = _fork_config.KVCacheSharingConfig(
                frac_shared_layers = config.kv_cache_sharing_frac_shared_layers,
                share_global = config.kv_cache_sharing_share_global,
                share_local = config.kv_cache_sharing_share_local,
            )
        self.kv_cache_sharing_patterns = _fork_config.create_kv_cache_sharing_patterns(
            sharing_config,
            config.num_layers,
            attention_types,
        )

        if config.kv_cache_sharing_frac_shared_layers > 0.0:
            num_unshared = config.num_layers - int(
                config.kv_cache_sharing_frac_shared_layers * config.num_layers
            )
        else:
            num_unshared = config.num_layers
        self._num_unshared = num_unshared

        if self.stacked_layer_params:
            # Declare stacked params per (FFW-group × attn-type), apply the
            # Block forward via Module.apply with a sliced params dict in a
            # manual layer for-loop (preserves original layer order).
            self._setup_stacked_layer_params(config, attention_types, attn_pattern = pattern, num_unshared = num_unshared)
        else:
            block_cls = nn.remat(
                _fork_mod.Block,
                prevent_cse = False,
                policy = jax.checkpoint_policies.nothing_saveable,
            )

            blocks = []
            for layer_idx, attn_type in enumerate(attention_types):
                if (
                    self._kv_sharing_enabled(layer_number = layer_idx)
                    and config.override_kv_shared_ffw_hidden is not None
                ):
                    hidden_dim_override = config.override_kv_shared_ffw_hidden
                else:
                    hidden_dim_override = None
                kwargs = _build_block_kwargs(config, attn_type, hidden_dim_override)
                kwargs["name"] = f"layer_{layer_idx}"
                blocks.append(block_cls(**kwargs))
            self.blocks = blocks

        self.final_norm = _fork_layers.RMSNorm(name = "final_norm")

    # ---- stacked-per-group layer params ----

    def _setup_stacked_layer_params(self, config, attention_types, *, attn_pattern, num_unshared):
        """Discover Block param shapes per (FFW-group × attn-type) and declare
        a stacked self.param with leading axis = group size for each leaf.

        Group indexing: 0 = unshared LOCAL, 1 = unshared GLOBAL,
        2 = shared LOCAL, 3 = shared GLOBAL.
        """
        # Layer → (group_idx, within_idx). Build into local lists first; Flax
        # freezes attributes set during setup, so direct .append on a self.
        # attribute fails after the initial assignment.
        layer_to_group: list[tuple[int, int]] = []
        group_layers: list[list[int]] = [[], [], [], []]
        for i in range(config.num_layers):
            is_shared = i >= num_unshared
            attn_flag = int(attn_pattern[i])  # 0 LOCAL / 1 GLOBAL
            group_idx = (2 if is_shared else 0) + attn_flag
            within_idx = len(group_layers[group_idx])
            group_layers[group_idx].append(i)
            layer_to_group.append((group_idx, within_idx))
        self._layer_to_group = tuple(layer_to_group)
        self._group_layers = tuple(tuple(g) for g in group_layers)

        # Per-group Block templates (Python objects, NOT registered as
        # submodules — we don't want their unstacked params auto-created).
        block_cls = nn.remat(
            _fork_mod.Block,
            prevent_cse = False,
            policy = jax.checkpoint_policies.nothing_saveable,
        )
        templates: list[_fork_mod.Block | None] = []
        for group_idx in range(4):
            if not group_layers[group_idx]:
                templates.append(None)
                continue
            is_shared = group_idx >= 2
            attn_flag = group_idx % 2
            attn_type = _ATTN_TYPE_MAP[attn_flag]
            hidden_dim_override = (
                config.override_kv_shared_ffw_hidden if is_shared else None
            )
            kwargs = _build_block_kwargs(config, attn_type, hidden_dim_override)
            templates.append(block_cls(**kwargs))
        self._block_templates = tuple(templates)

        # Discover Block param tree shape via jax.eval_shape on init.
        # Use small dummy shapes (B=1, T=2) — only shapes matter.
        rng = jax.random.PRNGKey(0)
        b, t = 1, 2
        dummy_x = jnp.zeros((b, t, config.embed_dim), dtype = jnp.float32)
        dummy_pos = jnp.zeros((b, t), dtype = jnp.int32)
        dummy_mask = jnp.ones((b, t, t), dtype = jnp.bool_)
        dummy_pli = (
            jnp.zeros((b, t, config.per_layer_input_dim), dtype = jnp.float32)
            if config.per_layer_input_dim > 0
            else None
        )

        # Maps (group_idx, leaf_path_tuple) -> stacked array.
        stacked_dict: dict[tuple[int, tuple[str, ...]], jax.Array] = {}
        for group_idx, template in enumerate(self._block_templates):
            if template is None:
                continue
            n_group = len(self._group_layers[group_idx])
            param_tree = jax.eval_shape(
                template.init,
                rng,
                dummy_x,
                dummy_pos,
                None,         # cache
                dummy_mask,
                dummy_pli,
                None,         # kv_shared_cache
            )["params"]
            flat_specs = _traverse_util.flatten_dict(param_tree)
            for path_tuple, leaf_spec in flat_specs.items():
                # Flat name uses '__' separator (Flax param names disallow '/').
                name = f"g{group_idx}__" + "__".join(path_tuple)
                stacked = self.param(
                    name,
                    nn.initializers.zeros,
                    (n_group, *leaf_spec.shape),
                    leaf_spec.dtype,
                )
                stacked_dict[(group_idx, path_tuple)] = stacked
        self._stacked_params = stacked_dict

    def _get_stacked_layer_params(self, group_idx: int, within_idx: int) -> dict:
        """Slice the stacked params at within_idx, return a nested dict matching the Block's param tree."""
        flat: dict[tuple[str, ...], jax.Array] = {}
        for (g, path), stacked in self._stacked_params.items():
            if g == group_idx:
                flat[path] = stacked[within_idx]
        return _traverse_util.unflatten_dict(flat)

    # ---- Public embedding helpers ----

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def decode(self, x: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        logits = self.embedder.decode(x)
        config = self.configs[0]
        if config.final_logit_softcap is not None:
            logits /= config.final_logit_softcap
            logits = jnp.tanh(logits) * config.final_logit_softcap
        return logits

    def encode_vision(self, vision_tokens: jax.Array) -> jax.Array:
        """Project vision tokens from vision_proj_dim -> embed_dim."""
        return self.embedder.encode_vision(vision_tokens).astype(self.embed_dtype)

    def encode_per_layer_input(
        self, embeddings: jax.Array, tokens: jax.Array
    ) -> jax.Array:
        """Compute per-layer input embeddings of shape [B, T, num_layers, per_layer_input_dim]."""
        return self.embedder.encode_per_layer_input(embeddings, tokens)

    # ---- Main forward ----

    def __call__(
        self,
        embedded: Sequence[jax.Array | None],
        positions: jax.Array,
        mask: jax.Array,
        adarms_cond: Sequence[jax.Array | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,  # noqa: ARG002
        return_cls_attention_score_distribution: bool = False,
        per_layer_input: jax.Array | None = None,
    ):
        """Run the transformer stack.

        Args:
            embedded: Sequence with a single entry: input embeddings [B, T, D].
            positions: Absolute token positions [B, T].
            mask: Attention mask [B, T, S] where S is the key length (T if no cache).
            adarms_cond: Unused; kept for API parity with gemma3.Module.
            kv_cache: Optional per-layer KV cache dict (see KVCache typedef).
            deterministic: Unused; kept for API parity.
            return_cls_attention_score_distribution: If True, also return a
              stacked [B, num_layers, S] array of CLS attention rows.
            per_layer_input: Pre-computed per-layer input embeddings of shape
              [B, T, num_layers, per_layer_input_dim]. Required when the config
              has `per_layer_input_dim > 0`.

        Returns:
            ([output], new_kv_cache) or ([output], new_kv_cache, cls_attn)
        """
        del adarms_cond  # parity only

        config = self.configs[0]

        x = embedded[0]
        if x is None:
            raise ValueError("Gemma 4 expects a non-None embedded input at index 0.")
        x = x.astype(self.embed_dtype)
        assert x.dtype == jnp.dtype(self.embed_dtype)

        if config.per_layer_input_dim > 0 and per_layer_input is None:
            raise ValueError(
                "per_layer_input is required when per_layer_input_dim > 0."
            )

        new_cache: KVCache = {}

        if self.stacked_layer_params:
            # walk layers in original order, slicing into per-group
            # stacked params and applying the Block forward via Module.apply.
            cls_attn_layers: list[jax.Array] = []
            for layer_idx in range(config.num_layers):
                group_idx, within_idx = self._layer_to_group[layer_idx]
                template = self._block_templates[group_idx]
                layer_key = f"layer_{layer_idx}"
                layer_cache_in = kv_cache[layer_key] if kv_cache is not None else None
                if self._kv_sharing_enabled(layer_number = layer_idx):
                    shared_layer_key = f"layer_{self.kv_cache_sharing_patterns[layer_idx]}"
                    kv_shared_cache = new_cache.get(shared_layer_key)
                else:
                    kv_shared_cache = None
                if config.per_layer_input_dim > 0:
                    per_layer_input_slice = per_layer_input[:, :, layer_idx, :]
                else:
                    per_layer_input_slice = None

                sliced_params = self._get_stacked_layer_params(group_idx, within_idx)
                layer_cache_out, x, cls_attn_row = template.apply(
                    {"params": sliced_params},
                    x,
                    positions,
                    layer_cache_in,
                    mask,
                    per_layer_input_slice,
                    kv_shared_cache,
                )
                new_cache[layer_key] = layer_cache_out
                cls_attn_layers.append(cls_attn_row)
            stacked = jnp.stack(cls_attn_layers, axis = 0)
        else:
            cls_attn_layers: list[jax.Array] = []
            for layer_idx, block in enumerate(self.blocks):
                layer_key = f"layer_{layer_idx}"
                layer_cache_in = kv_cache[layer_key] if kv_cache is not None else None
                if self._kv_sharing_enabled(layer_number = layer_idx):
                    shared_layer_key = f"layer_{self.kv_cache_sharing_patterns[layer_idx]}"
                    kv_shared_cache = new_cache.get(shared_layer_key)
                else:
                    kv_shared_cache = None

                if config.per_layer_input_dim > 0:
                    per_layer_input_slice = per_layer_input[:, :, layer_idx, :]
                else:
                    per_layer_input_slice = None

                layer_cache_out, x, cls_attn_row = block(
                    x,
                    positions,
                    layer_cache_in,
                    mask,
                    per_layer_input_slice,
                    kv_shared_cache,
                )
                new_cache[layer_key] = layer_cache_out
                cls_attn_layers.append(cls_attn_row)
            stacked = jnp.stack(cls_attn_layers, axis = 0)

        output = self.final_norm(x)

        if return_cls_attention_score_distribution:
            # [num_layers, B, S] -> [B, num_layers, S]
            stacked = jnp.transpose(stacked, (1, 0, 2))
            return [output], new_cache, stacked
        return [output], new_cache

    def _kv_sharing_enabled(self, *, layer_number: int) -> bool:
        return layer_number != self.kv_cache_sharing_patterns[layer_number]

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method matching gemma3.Module.init for lazy_init."""
        del use_adarms  # parity only
        config = self.configs[0]
        dummy_tokens = jnp.zeros((1, 2), dtype = jnp.int32)
        embeddings = self.embed(dummy_tokens)
        positions = jnp.zeros((1, 2), dtype = jnp.int32)
        mask = jnp.ones((1, 2, 2), dtype = jnp.bool_)

        # Initialize the vision projection params (embedder.mm_input_projection,
        # embedder.mm_pre_projection_norm) when a vision encoder is wired up.
        if config.vision_proj_dim is not None:
            dummy_vision = jnp.zeros((1, 1, config.vision_proj_dim), dtype = jnp.float32)
            self.encode_vision(dummy_vision)

        per_layer_input = None
        if config.per_layer_input_dim > 0:
            per_layer_input = self.encode_per_layer_input(embeddings, dummy_tokens)

        self(
            [embeddings],
            positions,
            mask,
            None,
            per_layer_input = per_layer_input,
        )
