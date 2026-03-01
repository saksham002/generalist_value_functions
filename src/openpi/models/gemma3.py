"""Gemma 3 transformer implementation for value function backbone.

Standalone implementation of the Gemma 3 architecture (4B/27B variants) without
depending on the `gemma` PyPI package. Closely mirrors gemma.py structure
but with Gemma 3 architectural changes:
- QK-norm: RMSNorm on Q/K after projection, before RoPE
- Mixed local/global attention with different RoPE frequencies
- Sliding window mask for local attention layers
- Post-attention and post-FFW normalization layers
- Transposed gating einsum in FeedForward

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

GEMMA3_VOCAB_SIZE = 262_144


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window_size: int
    local_rope_base_freq: int = 10_000
    global_rope_base_freq: int = 1_000_000
    local_rope_scale_factor: float = 1.0
    global_rope_scale_factor: float = 8.0
    # 6-element pattern: 0=LOCAL_SLIDING, 1=GLOBAL; repeats to fill depth
    attn_pattern: tuple[int, ...] = (0, 0, 0, 0, 0, 1)


Variant = Literal["gemma3_dummy", "gemma3_4b", "gemma3_27b"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified Gemma 3 variant."""
    if variant == "gemma3_dummy":
        return Config(
            width = 64,
            depth = 6,
            mlp_dim = 128,
            num_heads = 8,
            num_kv_heads = 4,
            head_dim = 16,
            sliding_window_size = 32,
        )
    if variant == "gemma3_4b":
        return Config(
            width = 2560,
            depth = 34,
            mlp_dim = 10240,
            num_heads = 8,
            num_kv_heads = 4,
            head_dim = 256,
            sliding_window_size = 1024,
        )
    if variant == "gemma3_27b":
        return Config(
            width = 5376,
            depth = 62,
            mlp_dim = 21504,
            num_heads = 32,
            num_kv_heads = 16,
            head_dim = 128,
            sliding_window_size = 1024,
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis = -1, keepdims = True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs.astype(dtype), None


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


def _apply_rope(x, *, positions, base_frequency, scale_factor):
    """Applies RoPE positions [B, L] to x [B, L, H, D] with dynamic base frequency and scale factor."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype = jnp.float32)
    timescale = base_frequency ** freq_exponents
    timescale = timescale * scale_factor
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis = -1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis = -1)
    return res.astype(x.dtype)


def _create_sliding_mask(
    positions: at.Int[at.Array, "b t"],
    *,
    cache_positions: at.Int[at.Array, "b s"] | None = None,
    sliding_window_size: int,
) -> at.Bool[at.Array, "b t s"]:
    """Match the reference Gemma local-attention mask using absolute positions."""
    if cache_positions is None:
        cache_positions = positions

    cache_positions = cache_positions[..., None, :]
    positions = positions[..., :, None]
    sliding_mask = cache_positions > positions - sliding_window_size
    sliding_mask = sliding_mask & (cache_positions < positions + sliding_window_size)
    return sliding_mask


@at.typecheck
class Attention(nn.Module):
    """Attention module with QK-norm and mixed local/global attention."""

    config: Config

    @nn.compact
    def __call__(self, x, positions, attn_mask, kv_cache, attn_type_flag):
        config = self.config
        dtype = x.dtype

        q_einsum = self.param(
            "q_einsum",
            nn.initializers.lecun_normal(in_axis = -2, out_axis = -1, batch_axis = (0,)),
            (config.num_heads, config.width, config.head_dim),
        ).astype(dtype)
        q = jnp.einsum("BTD,NDH->BTNH", x, q_einsum)

        kv_einsum = self.param(
            "kv_einsum",
            nn.initializers.lecun_normal(in_axis = -2, out_axis = -1, batch_axis = (0, 1)),
            (2, config.num_kv_heads, config.width, config.head_dim),
        ).astype(dtype)
        k, v = jnp.einsum("BSD,XKDH->XBSKH", x, kv_einsum)

        # QK-norm: apply RMSNorm to Q and K before RoPE
        q, _ = RMSNorm(name = "query_norm")(q, None)
        k, _ = RMSNorm(name = "key_norm")(k, None)

        # Dynamic RoPE: select base_frequency and scale_factor based on attention type
        base_freq = jnp.where(
            attn_type_flag,
            jnp.float32(config.global_rope_base_freq),
            jnp.float32(config.local_rope_base_freq),
        )
        scale_factor = jnp.where(
            attn_type_flag,
            jnp.float32(config.global_rope_scale_factor),
            jnp.float32(config.local_rope_scale_factor),
        )

        q = _apply_rope(q, positions = positions, base_frequency = base_freq, scale_factor = scale_factor)
        q *= config.head_dim ** -0.5
        k = _apply_rope(k, positions = positions, base_frequency = base_freq, scale_factor = scale_factor)

        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v, cache_positions = kv_cache
            # cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis = 1)
            v = jnp.concatenate([cache_v, v], axis = 1)
            cache_positions = jnp.concatenate([cache_positions, positions], axis = 1)
            # cache_positions = jnp.concatenate([cache_positions, positions], axis = 1)
        else:
            cache_positions = positions
            # cache_positions = positions

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K = config.num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type = jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # Sliding window mask: restrict local attention using absolute token positions.
        sliding_mask = _create_sliding_mask(
            positions,
            cache_positions = cache_positions,
            sliding_window_size = config.sliding_window_size,
        )
        # seq_len_q = q.shape[1]
        # seq_len_k = k.shape[1]
        # q_positions = jnp.arange(seq_len_q)[None, :, None]
        # k_positions = jnp.arange(seq_len_k)[None, None, :]
        # sliding_mask = jnp.abs(q_positions - k_positions) < config.sliding_window_size
        sliding_mask = sliding_mask[:, None, :, :]  # [1, 1, T, S]

        # Apply sliding mask only for local attention (attn_type_flag == 0)
        effective_mask = jnp.where(attn_type_flag, attn_mask, attn_mask & sliding_mask)

        big_neg = -2.3819763e38
        masked_logits = jnp.where(effective_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis = -1).astype(dtype)
        # CLS is the last query position; average over K kv-heads and G query groups -> [B, S]
        cls_attn_row = probs[:, :, :, -1, :].mean(axis = (1, 2)).astype(jnp.float32)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        attn_vec_einsum = self.param(
            "attn_vec_einsum",
            nn.initializers.lecun_normal(in_axis = (-3, -2), out_axis = -1),
            (config.num_heads, config.head_dim, config.width),
        ).astype(dtype)
        output = jnp.einsum("BTNH,NHD->BTD", encoded, attn_vec_einsum)

        return (output, (k, v, cache_positions)), cls_attn_row
        # return (output, (k, v)), cls_attn_row


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module with transposed gating einsum (Gemma 3 style)."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        # Transposed gating einsum: shape (2, hidden_dim, features) instead of (2, features, hidden_dim)
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis = -1, out_axis = -2, batch_axis = (0,)),
            (2, self.hidden_dim, self.features),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0].T)
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1].T)
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis = -2, out_axis = -1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block with post-norms (Gemma 3 style)."""

    config: Config

    @nn.compact
    def __call__(self, x, kv_cache, positions, attn_mask, adarms_cond, attn_type_flag, deterministic = True):  # noqa: FBT002
        x = sharding.activation_sharding_constraint(x)

        # Pre-attention norm
        pre_attn, _ = RMSNorm(name = "pre_attention_norm")(x, None)
        pre_attn = sharding.activation_sharding_constraint(pre_attn)

        # Attention
        (post_attn, kv_cache), cls_attn_row = Attention(
            config = self.config, name = "attn"
        )(pre_attn, positions, attn_mask, kv_cache, attn_type_flag)
        post_attn = sharding.activation_sharding_constraint(post_attn)

        # Post-attention norm (Gemma 3 addition)
        post_attn, _ = RMSNorm(name = "post_attention_norm")(post_attn, None)

        # Residual
        x = x + post_attn
        x = sharding.activation_sharding_constraint(x)

        # Pre-FFW norm
        pre_ffw, _ = RMSNorm(name = "pre_ffw_norm")(x, None)

        # Feed-forward
        ffw_out = FeedForward(
            features = self.config.width,
            hidden_dim = self.config.mlp_dim,
            name = "mlp",
        )(pre_ffw)
        ffw_out = sharding.activation_sharding_constraint(ffw_out)

        # Post-FFW norm (Gemma 3 addition)
        ffw_out, _ = RMSNorm(name = "post_ffw_norm")(ffw_out, None)

        # Residual
        x = x + ffw_out
        x = sharding.activation_sharding_constraint(x)

        return x, (kv_cache, cls_attn_row)


KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"],
    at.Float[at.Array, "l b _t _v _h"],
    at.Int[at.Array, "l b _t"],
]
# KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Gemma 3 transformer module. Interface matches gemma.Module for drop-in use."""

    configs: Sequence[Config]
    embed_dtype: str

    adarms: bool = False

    def setup(self):
        config = self.configs[0]

        self.embedder = Embedder(
            vocab_size = GEMMA3_VOCAB_SIZE,
            embed_dim = config.width,
            name = "embedder",
        )

        # Build per-layer attention type flags from the repeating pattern
        pattern = config.attn_pattern
        attn_type_flags = [pattern[i % len(pattern)] for i in range(config.depth)]
        self._attn_type_flags = jnp.array(attn_type_flags, dtype = jnp.int32)

        block_cls = nn.remat(
            Block,
            prevent_cse = False,
            static_argnums = (6,),  # 0=self, 7=deterministic
            policy = jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes = {"params": 0},
            split_rngs = {"params": True, "dropout": True},
            in_axes = (
                0,       # kv_cache
                nn.broadcast,  # positions
                nn.broadcast,  # attn_mask
                nn.broadcast,  # adarms_cond
                0,       # attn_type_flag (per-layer)
                nn.broadcast,  # deterministic
            ),
            length = config.depth,
        )(
            config = config,
        )
        self.final_norm = RMSNorm(name = "final_norm")

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def decode(self, x: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        return self.embedder.decode(x)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
        return_cls_attention_score_distribution: bool = False,
    ) -> (
        tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]
        | tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache, at.Float[at.Array, "b _l _s"]]
    ):
        # Gemma 3 uses a single config (no multi-expert), but we accept the same interface
        x = embedded[0]
        x = x.astype(self.embed_dtype)
        mask = jnp.asarray(mask)[:, None, :, :]

        x, (kv_cache, all_cls_attn) = self.layers(
            x, kv_cache, positions, mask, None, self._attn_type_flags, deterministic
        )

        assert x.dtype == jnp.dtype(self.embed_dtype)

        output, _ = self.final_norm(x, None)

        if return_cls_attention_score_distribution:
            # Transpose [depth, B, S] -> [B, depth, S] = [B, L, S]
            return [output], kv_cache, jnp.transpose(all_cls_attn, (1, 0, 2))
        return [output], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters."""
        config = self.configs[0]
        self.embed(jnp.zeros((1, 1), dtype = jnp.int32))
        self(
            [jnp.zeros((1, 1, config.width))],
            jnp.zeros((1, 1), dtype = jnp.int32),
            jnp.zeros((1, 1, 1), dtype = bool),
            adarms_cond = [None],
        )
