"""Gemma 4 vision encoder wrapper.

Thin Flax Linen wrapper around the Gemma 4 `VisionEncoder` from the installed
`gemma` package. This file only:

1. Rescales images from openpi's `[-1, 1]` convention to `[0, 1]` (the fork's
   `VisionEntry` applies `2 * (x - 0.5)` internally, so it expects `[0, 1]`).
2. Patchifies the (fixed-size square) images.
3. Runs the encoder.
4. Returns `(soft_tokens, None)` to match the `(tokens, aux)` API that
   `PaliGemmaValueNetwork` expects from `self.PaliGemma.img(...)`.

For variable-aspect-ratio use, see the fork's `patchify_and_pad` helper. This
wrapper always uses fixed square images.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from gemma.gm.nn.gemma4.vision import _encoder as _fork_encoder
from gemma.gm.nn.gemma4.vision import _images as _fork_images


@dataclasses.dataclass
class Config:
    """Configuration for the Gemma 4 vision encoder."""

    d_model: int = 768
    num_layers: int = 16
    num_heads: int = 12
    ffw_hidden: int = 3072
    patch_size: int = 16
    # Output soft-token length. For fixed 336x336 with pool=3: 49.
    output_length: int = 49
    pos_emb_shape_yx: tuple[int, int] = (10240, 2)
    pooling_kernel_size: int = 3
    use_clipped_linears: bool = False
    standardize_embeddings: bool = False
    # Compute dtype for the vision transformer activations. Params remain f32;
    # forward/backward run in this dtype. Threaded from the parent network
    # config so vision matches the LLM's `embed_dtype` (bf16 halves activation
    # HBM vs f32).
    dtype: str = "float32"


Variant = Literal["gemma4_vision_dummy", "gemma4_vision_e2b", "gemma4_vision_e4b"]


def get_config(
    variant: Variant,
    *,
    image_size: tuple[int, int] = (336, 336),
    dtype: str = "float32",
) -> Config:
    """Returns vision config for the specified variant.

    `output_length` is derived from image_size, patch_size, and pooling_kernel_size.
    `dtype` is the compute dtype for the vision transformer activations.
    """
    if variant == "gemma4_vision_dummy":
        # Small config for tests: 48x48 image, 3x3 patches (pool=2) → 4 tokens.
        return Config(
            d_model = 64,
            num_layers = 2,
            num_heads = 4,
            ffw_hidden = 128,
            patch_size = 16,
            output_length = _num_soft_tokens_for_resolution(image_size[0], image_size[1], 16, 2),
            pos_emb_shape_yx = (64, 2),
            pooling_kernel_size = 2,
            dtype = dtype,
        )
    if variant == "gemma4_vision_e2b":
        patch_size = 16
        pooling_kernel_size = 3
        return Config(
            d_model = 768,
            num_layers = 16,
            num_heads = 12,
            ffw_hidden = 3072,
            patch_size = patch_size,
            output_length = _num_soft_tokens_for_resolution(
                image_size[0], image_size[1], patch_size, pooling_kernel_size
            ),
            pos_emb_shape_yx = (10240, 2),
            pooling_kernel_size = pooling_kernel_size,
            use_clipped_linears = True,
            dtype = dtype,
        )
    if variant == "gemma4_vision_e4b":
        patch_size = 16
        pooling_kernel_size = 3
        return Config(
            d_model = 768,
            num_layers = 16,
            num_heads = 12,
            ffw_hidden = 3072,
            patch_size = patch_size,
            output_length = _num_soft_tokens_for_resolution(
                image_size[0], image_size[1], patch_size, pooling_kernel_size
            ),
            pos_emb_shape_yx = (10240, 2),
            pooling_kernel_size = pooling_kernel_size,
            use_clipped_linears = True,
            dtype = dtype,
        )
    raise ValueError(f"Unknown vision variant: {variant!r}")


def _num_soft_tokens_for_resolution(
    height: int, width: int, patch_size: int, pooling_kernel_size: int
) -> int:
    """Compute pooled soft-token count for a fixed square image."""
    h_patches = height // patch_size
    w_patches = width // patch_size
    assert h_patches % pooling_kernel_size == 0, (
        f"height_patches={h_patches} must be divisible by pool={pooling_kernel_size}"
    )
    assert w_patches % pooling_kernel_size == 0, (
        f"width_patches={w_patches} must be divisible by pool={pooling_kernel_size}"
    )
    return (h_patches // pooling_kernel_size) * (w_patches // pooling_kernel_size)


def num_soft_tokens_for_resolution(
    height: int, width: int, *, patch_size: int = 16, pooling_kernel_size: int = 3
) -> int:
    """Public alias for the soft-token count helper."""
    return _num_soft_tokens_for_resolution(height, width, patch_size, pooling_kernel_size)


class Module(nn.Module):
    """Gemma 4 vision module. Interface matches `siglip.Module(...).__call__`."""

    config: Config

    def setup(self):
        self.encoder = _fork_encoder.VisionEncoder(
            d_model = self.config.d_model,
            num_layers = self.config.num_layers,
            num_heads = self.config.num_heads,
            ffw_hidden = self.config.ffw_hidden,
            patch_size = self.config.patch_size,
            output_length = self.config.output_length,
            pos_emb_shape_yx = self.config.pos_emb_shape_yx,
            pooling_kernel_size = self.config.pooling_kernel_size,
            use_clipped_linears = self.config.use_clipped_linears,
            standardize_embeddings = self.config.standardize_embeddings,
            name = "encoder",
        )

    def __call__(self, image: jax.Array, *, train: bool = False) -> tuple[jax.Array, None]:
        del train
        # openpi convention: images are in [-1, 1]. Fork's VisionEntry applies
        # `2 * (x - 0.5)` internally, so rescale to [0, 1] first.
        image_01 = (image.astype(jnp.float32) + 1.0) / 2.0
        patches, positions_xy = _fork_images.patchify(image_01, self.config.patch_size)
        # Cast patches to the vision compute dtype so the entire encoder runs in
        # bf16 (vs default f32). Halves activation HBM (~3 GB → ~1.5 GB at
        # B=256, 240x240) and matches the LLM's `embed_dtype` for downstream concat.
        patches = patches.astype(jnp.dtype(self.config.dtype))
        (soft_tokens, _mask), = self.encoder(patches, positions_xy)
        return soft_tokens, None
