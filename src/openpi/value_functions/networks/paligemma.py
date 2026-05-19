"""PaliGemma network for V(s, l) value functions with image+state+text inputs.

Uses a pre-trained PaliGemma backbone (ViT + Gemma LLM) as a feature encoder
for training subtask-conditioned state value functions V(s, l) where l is a subtask.

Sequence structure (Gemma 3):
    [BOS] [img1_block(260)] [img2_block(260)] [img3_block(260)] [text_tokens...] [state_embed] [CLS]

Where:
- img*_patches: 256 patches per image from ViT (14x14 patches from 224x224)
- text_tokens: Tokenized subtask prompt (randomly sampled at data loading time)
- state_embed: Projected proprioceptive state (single token)
- CLS: Global learnable token for value extraction - attends to all but not attended by others

Attention Mask Design:
- Image patches and text tokens can attend to each other (bidirectional)
- State token can attend to images and text (but images/text cannot attend to state)
- CLS token (last position) can attend to ALL tokens but CANNOT be attended by others

This design allows:
1. CLS to gather global context from all modalities for value prediction
2. State to condition on visual-language features
3. Visual-language stream to remain clean (not "polluted" by state)
4. CLS at the end enables simple causal masking for its attention pattern

V(s, l) Training Design:
- At data loading time, one non-null subtask is randomly sampled per datapoint
- The sampled subtask's text is used as tokenized_prompt
- MC target = gamma ** steps_to_subtask_end[sampled_idx]

Position Embeddings:
- Uses cumulative position indices based on valid token masks
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.model import IMAGE_KEYS
import openpi.models.gemma as _gemma
import openpi.models.gemma3 as _gemma3
import openpi.models.gemma4 as _gemma4
import openpi.models.gemma4_vision as _gemma4_vision
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.value_functions.networks.base_networks import BaseValueNetwork

logger = logging.getLogger(__name__)


# Number of patches per image (224/14 = 16, 16*16 = 256)
NUM_PATCHES_PER_IMAGE = 256

# Gemma 3 image block: [\n\n, <SOI>, 256 patches, <EOI>, \n\n]
GEMMA3_TOKENS_PER_IMAGE_BLOCK = NUM_PATCHES_PER_IMAGE + 4
MAX_SUBTASK_TOKENS_FOR_LOSS = 48


def make_attn_mask(
    input_mask: jax.Array,
    mask_ar: jax.Array,
    *,
    suffix_mask: jax.Array | None = None,
) -> jax.Array:
    """Create attention mask for value network with state and CLS tokens at the end.

    Sequence structure: [images] [prefix_text] [suffix_text] [state] [CLS]

    The cumsum of mask_ar groups tokens: positions sharing the same cumsum value
    attend bidirectionally; a position with higher cumsum can attend to lower but
    not vice-versa.  Suffix text tokens have mask_ar=1 so they are causal among
    themselves and invisible to prefix tokens (lower cumsum).

    When suffix_mask is provided, an additional constraint prevents non-suffix
    queries (state, actions, CLS) from attending to suffix keys, keeping the
    value-relevant outputs independent of the subtask text.

    Args:
        input_mask: bool[B, N] true if part of the input, false if padding.
        mask_ar: bool[B, N] or bool[N]. 0=bidirectional, 1=causal.
        suffix_mask: Optional bool[B, N]. True for suffix (subtask) text positions.
            Non-suffix queries are blocked from attending to suffix keys.

    Returns:
        Attention mask [B, N, N] where True means "can attend".
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)

    cumsum = jnp.cumsum(mask_ar, axis=1)
    mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]

    valid = input_mask[:, None, :] * input_mask[:, :, None]
    mask = jnp.logical_and(mask, valid)

    if suffix_mask is not None:
        suffix_mask = jnp.broadcast_to(suffix_mask, input_mask.shape)
        non_suffix_seeing_suffix = (~suffix_mask[:, :, None]) & suffix_mask[:, None, :]
        mask = mask & ~non_suffix_seeing_suffix

    return mask


def make_gemma3_attn_mask(
    input_mask: jax.Array,
    num_cameras: int,
) -> jax.Array:
    """Create attention mask for Gemma 3 value network with bidirectional image-patch blocks.

    Sequence layout:
        [BOS] [img_block_1 (260)] [img_block_2 (260)] ... [text] [state] [(actions)] [CLS]

    Each 260-token image block: [\\n\\n, <SOI>, 256 patches, <EOI>, \\n\\n]

    Attention rules:
    - Causal base: every token can attend to itself and all earlier tokens (if both valid).
    - Bidirectional overlay: within each image block, patch positions (offsets 2..257) attend
      to each other bidirectionally. Demarcation tokens (\\n\\n, <SOI>, <EOI>) stay causal.

    Args:
        input_mask: bool[B, S] — True for valid positions, False for padding.
        num_cameras: number of image blocks in the sequence (after BOS).

    Returns:
        Attention mask [B, S, S] where True means "can attend".
    """
    batch_size, seq_len = input_mask.shape

    # Causal base mask
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))  # [S, S]
    causal = jnp.broadcast_to(causal[None], (batch_size, seq_len, seq_len))

    # Build bidirectional overlay for image patch positions (offset by 1 for BOS)
    bidirectional = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)
    for cam_idx in range(num_cameras):
        block_start = 1 + cam_idx * GEMMA3_TOKENS_PER_IMAGE_BLOCK
        patch_start = block_start + 2   # skip \n\n, <SOI>
        patch_end = patch_start + NUM_PATCHES_PER_IMAGE  # 256 patches
        # Patches within same block can attend bidirectionally
        patch_range = jnp.arange(seq_len)
        in_block = (patch_range >= patch_start) & (patch_range < patch_end)
        bidirectional = bidirectional | (in_block[None, :] & in_block[:, None])

    mask = causal | jnp.broadcast_to(bidirectional[None], (batch_size, seq_len, seq_len))

    # Apply valid positions mask (both query and key must be valid)
    valid = input_mask[:, None, :] & input_mask[:, :, None]
    mask = mask & valid

    return mask


def make_gemma4_attn_mask(
    input_mask: jax.Array,
    *,
    suffix_mask: jax.Array | None = None,
    action_start: int | None = None,
    action_length: int | None = None,
) -> jax.Array:
    """Create attention mask for Gemma 4 value network.

    Sequence layout (value-network default):
        [BOS] [img_block_1] ... [img_block_N] [text] [state] [(actions)] [CLS]

    Attention is causal throughout the visual/text/state/action stream by
    default. When ``action_start`` and ``action_length`` are provided, the
    action block is made bidirectional within itself: each action token can
    attend to every other action token in both directions, while tokens
    outside the block remain governed by the causal rule.

    Args:
        input_mask: bool[B, S] — True for valid positions, False for padding.
        suffix_mask: Optional bool[B, S]. True for subtask target text positions.
            Non-suffix queries are blocked from attending to suffix keys.
        action_start: absolute column index of the first action token, or None
            if no bidirectional action block is requested.
        action_length: number of action tokens in the action block, or None.

    Returns:
        Attention mask [B, S, S] where True means "can attend".
    """
    batch_size, seq_len = input_mask.shape

    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype = jnp.bool_))
    mask = jnp.broadcast_to(causal[None], (batch_size, seq_len, seq_len))

    if action_start is not None and action_length is not None and action_length > 0:
        positions = jnp.arange(seq_len)
        in_action = (positions >= action_start) & (positions < action_start + action_length)
        action_block = in_action[None, :] & in_action[:, None]  # [S, S]
        mask = mask | action_block[None]

    valid = input_mask[:, None, :] & input_mask[:, :, None]
    mask = mask & valid

    if suffix_mask is not None:
        suffix_mask = jnp.broadcast_to(suffix_mask, input_mask.shape)
        non_suffix_seeing_suffix = (~suffix_mask[:, :, None]) & suffix_mask[:, None, :]
        mask = mask & ~non_suffix_seeing_suffix

    return mask


@dataclasses.dataclass(frozen=True)
class PaliGemmaNetworkConfig:
    """Configuration for PaliGemma-based value network.

    This network uses a pre-trained PaliGemma backbone to encode images,
    text, and proprioceptive state for V(s) or Q(s,a) estimation.

    Sequence structure for V(s):
        [img1_patches] [img2_patches] [img3_patches] [text_tokens] [state_embed] [CLS]

    Sequence structure for Q(s,a) (when action_conditioned=True):
        [img1_patches] [img2_patches] [img3_patches] [text_tokens] [state_embed] [action_tokens] [CLS]

    Where:
    - action_tokens: action_horizon embedded action vectors (one token per action)

    The network outputs the CLS token embedding which is
    passed directly to the value head for final prediction.
    """

    # Proprioceptive state dimension
    state_dim: int

    # Number of camera images (default 3 for RoboCOIN)
    num_cameras: int = 3

    # Input image resolution (should match PaliGemma training: 224x224)
    image_size: tuple[int, int] = (224, 224)

    # Maximum token length for text prompts
    max_token_len: int = 48

    # PaliGemma variant (matches pi0 config)
    paligemma_variant: str = "gemma_2b"

    # Dtype for computations
    dtype: str = "bfloat16"

    # Action dimension (required when action_horizon is provided)
    action_dim: int = 14

    # Whether to mask out the state token in the attention mask (for ablation studies)
    no_state: bool = False

    # Fix order in which to iterate through keys
    image_keys: tuple[str, str, str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # Apply ``flax.nnx.LayerNorm`` (with learnable scale + bias) to CLS features
    # before the value head. Keeps the head's ``lecun_normal`` init stable
    # across LLM backbones with different final-norm scale magnitudes (gemma-4
    # ckpt scales have RMS≈14 vs gemma-3 ≈1).
    use_layernorm: bool = False

    # Gemma 4 only: when True, action tokens attend bidirectionally within the
    # action block (every action token attends to every other action token).
    # Tokens outside the action block remain causal.
    action_block_bidirectional: bool = False

    def get_tokenizer(self, max_len: int | None = None):
        """Return the appropriate text tokenizer for this variant."""
        from openpi.models.tokenizer import Gemma3Tokenizer, Gemma4Tokenizer, PaligemmaTokenizer

        if max_len is None:
            max_len = self.max_token_len
        if self.paligemma_variant.startswith("gemma4_"):
            return Gemma4Tokenizer(max_len = max_len, num_images = self.num_cameras, use_bos = True)
        if self.paligemma_variant.startswith("gemma3_"):
            return Gemma3Tokenizer(max_len = max_len, num_images = self.num_cameras)
        return PaligemmaTokenizer(max_len = max_len)

    def create(self, rng: at.KeyArrayLike, action_horizon: int | None = None) -> PaliGemmaValueNetwork:
        """Create a new PaliGemma value network with initialized parameters."""
        return PaliGemmaValueNetwork(self, rngs = nnx.Rngs(rng), action_horizon = action_horizon)


class PaliGemmaValueNetwork(BaseValueNetwork):
    """PaliGemma-based value network for V(s, l) or Q(s, a, l) estimation.

    Architecture (following pi0.py initialization pattern):
    1. SigLIP ViT for image encoding -> 256 patches per image
    2. Gemma for text embedding + LLM processing
    3. Linear projection for state -> 1 embedding
    4. (If action_conditioned) Linear projection for each action -> action_horizon embeddings
    5. Learnable CLS token appended to sequence
    6. Forward through Gemma LLM with custom attention mask
    7. Extract CLS token output (last position) -> value head

    Token sequence for V(s) (example with 3 images):
        [256 patches img1] [256 patches img2] [256 patches img3] [L text tokens] [1 state token] [CLS]

    Token sequence for Q(s,a) (example with 3 images, action_horizon=H):
        [256 patches img1] [256 patches img2] [256 patches img3] [L text tokens] [1 state token] [H action tokens] [CLS]

    Attention Design for Q(s,a):
    - Image and text tokens: bidirectional with each other
    - State token: can attend to images/text, but images/text cannot attend to it
    - Action tokens: can attend to images/text/state, but state cannot attend to actions
    - CLS token (last): can attend to all, but others cannot attend to it
    """

    def __init__(self, config: PaliGemmaNetworkConfig, rngs: nnx.Rngs, action_horizon: int | None = None):
        super().__init__()

        self.config = config
        self._action_conditioned = action_horizon is not None
        self._is_gemma3 = config.paligemma_variant.startswith("gemma3_")
        self._is_gemma4 = config.paligemma_variant.startswith("gemma4_")
        self._num_cameras = config.num_cameras
        self._max_token_len = config.max_token_len
        self._action_horizon = action_horizon if action_horizon is not None else 0
        self._action_dim = config.action_dim
        self._no_state = config.no_state
        self._image_keys = config.image_keys
        self._image_size = config.image_size

        logger.info(
            "PaliGemmaValueNetwork: variant=%s, is_gemma3=%s, is_gemma4=%s, action_conditioned=%s, "
            "action_horizon=%s, no_state=%s, action_block_bidirectional=%s, use_layernorm=%s",
            config.paligemma_variant, self._is_gemma3, self._is_gemma4, self._action_conditioned,
            self._action_horizon, self._no_state, config.action_block_bidirectional, config.use_layernorm,
        )

        # Get config and module class based on variant
        if self._is_gemma4:
            paligemma_config = _gemma4.get_config(config.paligemma_variant)
            gemma_module_cls = _gemma4.Module
            embed_dim = paligemma_config.embed_dim
        elif self._is_gemma3:
            paligemma_config = _gemma3.get_config(config.paligemma_variant)
            gemma_module_cls = _gemma3.Module
            embed_dim = paligemma_config.width
        else:
            paligemma_config = _gemma.get_config(config.paligemma_variant)
            gemma_module_cls = _gemma.Module
            embed_dim = paligemma_config.width

        # Gemma 4 per-layer-input (E2B has 256, dummy has 0)
        self._gemma4_per_layer_input_dim = (
            paligemma_config.per_layer_input_dim if self._is_gemma4 else 0
        )

        # Initialize Gemma LLM (single config, no action expert). For Gemma 4,
        # use ``stacked_layer_params=True`` which cuts FSDP all-gathers from N
        # to 4 (one per FFW-group × attn-type) at the cost of a manual layer
        # for-loop in place of nn.scan.
        gemma_kwargs = dict(
            configs = [paligemma_config],
            embed_dtype = config.dtype,
            adarms = False,
        )
        if self._is_gemma4:
            gemma_kwargs["stacked_layer_params"] = True
        llm = nnx_bridge.ToNNX(gemma_module_cls(**gemma_kwargs))
        llm.lazy_init(rngs = rngs, method = "init", use_adarms = [False])

        # Initialize image encoder.
        # - Gemma 4: use the Gemma-4 vision encoder (imported from the fork). Raw soft tokens
        #   are later projected to the LLM embed_dim via `llm.encode_vision`.
        # - Gemma 3: SigLIP w/ mm_proj_dim, which applies the Gemma-3 multimodal projector
        #   (RMSNorm + Linear + sqrt(embed_dim) scale) inside SigLIP.
        # - Original PaliGemma: SigLIP with num_classes=embed_dim applying the projection in
        #   the SigLIP head Dense layer.
        if self._is_gemma4:
            vision_variant = config.paligemma_variant.replace("gemma4_", "gemma4_vision_", 1)
            vision_config = _gemma4_vision.get_config(
                vision_variant, image_size = config.image_size, dtype = config.dtype
            )
            self._num_soft_tokens_per_image = vision_config.output_length
            img = nnx_bridge.ToNNX(_gemma4_vision.Module(config = vision_config))
        else:
            if self._is_gemma3:
                siglip_kwargs = {
                    "variant": "So400m/14",
                    "pool_type": "none",
                    "scan": True,
                    "dtype_mm": config.dtype,
                    "output_tokens": NUM_PATCHES_PER_IMAGE,
                    "mm_proj_dim": embed_dim,
                }
            else:
                siglip_kwargs = {
                    "num_classes": paligemma_config.width,
                    "variant": "So400m/14",
                    "pool_type": "none",
                    "scan": True,
                    "dtype_mm": config.dtype,
                }
            img = nnx_bridge.ToNNX(_siglip.Module(**siglip_kwargs))
            self._num_soft_tokens_per_image = NUM_PATCHES_PER_IMAGE
        # Initialize with a fake image
        fake_image = jnp.zeros((1, config.image_size[0], config.image_size[1], 3), dtype=jnp.float32)
        img.lazy_init(fake_image, train=False, rngs=rngs)

        # Store as PaliGemma dict (matches weight loader key structure)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # CLS token - learnable embedding for value extraction
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 0.02)

        # State projection: state_dim -> embed_dim (single token)
        self.state_proj: nnx.Linear | None = None
        if not self._no_state:
            self.state_proj = nnx.Linear(config.state_dim, embed_dim, rngs=rngs)

        # Action projection: action_dim -> embed_dim (one token per action in chunk)
        self.action_proj: nnx.Linear | None = None
        if self._action_conditioned:
            self.action_proj = nnx.Linear(config.action_dim, embed_dim, rngs=rngs)

        # Feature dimension is the Gemma embedding dimension
        self._feature_dim = embed_dim
        self._embed_dim = embed_dim

        self._use_layernorm = config.use_layernorm
        self.cls_layer_norm = nnx.LayerNorm(embed_dim, rngs = rngs) if config.use_layernorm else None

        self._gemma4_action_block_bidir = config.action_block_bidirectional

    def _get_special_embeddings(self) -> jax.Array:
        """Return [BOS, \\n\\n, <SOI>, <EOI>] embeddings [1, 4, D]."""
        from openpi.models.tokenizer import Gemma3Tokenizer
        special_ids = jnp.array([[Gemma3Tokenizer.BOS_ID,
                                  Gemma3Tokenizer.NEWLINE_NEWLINE_ID,
                                  Gemma3Tokenizer.START_OF_IMAGE_ID,
                                  Gemma3Tokenizer.END_OF_IMAGE_ID]])
        return self.PaliGemma.llm(special_ids, method = "embed")

    def _get_gemma4_special_embeddings(self) -> jax.Array:
        """Return [BOS, \\n\\n, <SOI>, <EOI>] embeddings [1, 4, D] using the Gemma 4 vocab."""
        from openpi.models.tokenizer import Gemma4Tokenizer
        special_ids = jnp.array([[Gemma4Tokenizer.BOS_ID,
                                  Gemma4Tokenizer.NEWLINE_NEWLINE_ID,
                                  Gemma4Tokenizer.START_OF_IMAGE_ID,
                                  Gemma4Tokenizer.END_OF_IMAGE_ID]])
        return self.PaliGemma.llm(special_ids, method = "embed")

    @property
    def action_conditioned(self) -> bool:
        """V(s) is not action-conditioned."""
        return self._action_conditioned

    @property
    @override
    def feature_dim(self) -> int:
        """Return the output feature dimension (Gemma embed_dim)."""
        return self._feature_dim

    def _embed_sequence(
        self,
        observation: _model.Observation,
        action: jax.Array | None = None,
        action_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        """Embed full sequence: images + text + state + [actions] + CLS.

        Args:
            observation: Observation with images, state, and tokenized_prompt.
            action: Optional action chunk [B, action_horizon, action_dim] for Q(s,a).
            action_mask: Optional mask [B, action_horizon] for valid actions in chunk.

        Returns:
            Tuple of (tokens, input_mask, ar_mask, suffix_mask)
            - tokens: [B, seq_len, embed_dim]
            - input_mask: [B, seq_len] bool
            - ar_mask: [seq_len] or [B, seq_len] bool where 0=bidirectional
              (images, prefix text), 1=causal (subtask text, state, actions, CLS)
            - suffix_mask: [B, seq_len] bool or None — True for subtask text positions
        """
        tokens, input_mask, ar_mask = self._embed_prefix(observation)
        batch_size = observation.state.shape[0]

        # 4. Add action tokens if action_conditioned
        if self._action_conditioned:
            action_tokens = self.action_proj(action)  # [B, action_horizon, embed_dim]
            tokens.append(action_tokens)
            input_mask.append(action_mask)

            ar_mask.append(True)
            ar_mask += [False] * (self._action_horizon - 1)

        # 5. CLS token (last in sequence)
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, self._embed_dim))
        tokens.append(cls_tokens)
        input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))
        ar_mask.append(True)

        tokens = jnp.concatenate(tokens, axis = 1)
        input_mask = jnp.concatenate(input_mask, axis = 1)
        ar_mask = jnp.array(ar_mask)

        # Mark subtask text tokens as causal in ar_mask. The cumsum logic in
        # make_attn_mask then ensures suffix tokens are causal among themselves
        # and invisible to prefix tokens.
        suffix_mask = None
        if observation.subtask_start_index is not None and observation.tokenized_prompt is not None:
            text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
            text_len = observation.tokenized_prompt.shape[1]
            seq_positions = jnp.arange(ar_mask.shape[0])
            global_subtask_start = text_start + observation.subtask_start_index  # [B]
            suffix_mask = (seq_positions[None, :] >= global_subtask_start[:, None]) & (
                seq_positions[None, :] < text_start + text_len
            )
            ar_mask = ar_mask[None, :] | suffix_mask  # [B, N]

        return tokens, input_mask, ar_mask, suffix_mask

    def _embed_prefix(
        self,
        observation: _model.Observation,
    ) -> tuple[list[jax.Array], list[jax.Array], list[bool]]:
        """Embed the shared observation prefix: images + text + state.

        Returns all three outputs as plain Python lists (not concatenated) so callers
        can keep appending before a single final concatenation / jnp.array conversion.
        """
        batch_size = observation.state.shape[0]

        input_mask: list[jax.Array] = []
        ar_mask: list[bool] = []
        tokens: list[jax.Array] = []

        # 1. Embed images (following pi0.py pattern)
        for name in self._image_keys:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train = False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens: bidirectional
            ar_mask += [False] * image_tokens.shape[1]

        # 2. Add language (tokenized inputs)
        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method = "embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # 3. Add state token (skip entirely when no_state is True)
        if not self._no_state:
            state_token = self.state_proj(observation.state)[:, None, :]  # [B, 1, embed_dim]
            tokens.append(state_token)
            input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))
            # State uses ar_mask=True (causal - can attend to images/text but not be attended)
            ar_mask.append(True)

        return tokens, input_mask, ar_mask

    def _embed_sequence_gemma3(
        self,
        observation: _model.Observation,
        action: jax.Array | None = None,
        action_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Embed full sequence for Gemma 3 with image demarcation tokens.

        Sequence layout:
            [BOS] [img_block_1 (260)] ... [img_block_N (260)] [text] [state] [(actions)] [CLS]

        Gemma 3 wraps each image's patch tokens with demarcation tokens:
            [\\n\\n, <SOI>, 256 patches, <EOI>, \\n\\n]  (260 tokens per image)

        BOS is placed first (position 0) so that image blocks start at position 1,
        matching the reference gemma library which requires BOS before <start_of_image>.

        Returns:
            Tuple of (tokens, input_mask, attn_mask)
            - tokens: [B, seq_len, embed_dim]
            - input_mask: [B, seq_len] bool
            - attn_mask: [B, seq_len, seq_len] bool — causal + bidirectional-patch-block mask
        """
        batch_size = observation.state.shape[0]

        special_emb = self._get_special_embeddings()  # [1, 4, D]
        bos_emb = special_emb[:, 0 : 1, :]     # [1, 1, D]
        newline_emb = special_emb[:, 1 : 2, :]  # [1, 1, D]
        soi_emb = special_emb[:, 2 : 3, :]
        eoi_emb = special_emb[:, 3 : 4, :]

        tokens = []
        input_mask = []

        # BOS token at position 0
        tokens.append(jnp.broadcast_to(bos_emb, (batch_size, 1, self._embed_dim)))
        input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))

        # Build image blocks: [\n\n, <SOI>, patches(256), <EOI>, \n\n] per camera
        for name in self._image_keys:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train = False)
            cam_mask = observation.image_masks[name]  # [B, 1] or [B]

            nn_b = jnp.broadcast_to(newline_emb, (batch_size, 1, self._embed_dim))
            soi_b = jnp.broadcast_to(soi_emb, (batch_size, 1, self._embed_dim))
            eoi_b = jnp.broadcast_to(eoi_emb, (batch_size, 1, self._embed_dim))

            block = jnp.concatenate([nn_b, soi_b, image_tokens, eoi_b, nn_b], axis = 1)  # [B, 260, D]
            tokens.append(block)

            block_mask = einops.repeat(cam_mask, "b -> b s", s = GEMMA3_TOKENS_PER_IMAGE_BLOCK)
            input_mask.append(block_mask)

        # Embed text tokens, stripping BOS and <SOI> markers inserted by the tokenizer.
        # tokenized_prompt = [BOS, <SOI>_1, <SOI>_2, ..., <SOI>_N, text_1, ..., \n, pad...]
        # Strip BOS (already placed above) and SOI markers (handled by image blocks).
        num_cameras = self._num_cameras
        text_only = observation.tokenized_prompt[:, 1 + num_cameras :]
        text_mask = observation.tokenized_prompt_mask[:, 1 + num_cameras :]
        text_emb = self.PaliGemma.llm(text_only, method = "embed")
        tokens.append(text_emb)
        input_mask.append(text_mask)

        # State token
        if not self._no_state:
            state_token = self.state_proj(observation.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))

        # Action tokens
        if self._action_conditioned:
            action_tokens = self.action_proj(action)
            tokens.append(action_tokens)
            input_mask.append(action_mask)

        # CLS token
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, self._embed_dim))
        tokens.append(cls_tokens)
        input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))

        tokens = jnp.concatenate(tokens, axis = 1)
        input_mask = jnp.concatenate(input_mask, axis = 1)

        attn_mask = make_gemma3_attn_mask(input_mask, num_cameras)

        return tokens, input_mask, attn_mask

    def _embed_sequence_gemma4(
        self,
        observation: _model.Observation,
        action: jax.Array | None = None,
        action_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Embed full sequence for Gemma 4 with image demarcation tokens.

        Sequence layout:
            [BOS] [img_block_1] ... [img_block_N] [text] [state] [(actions)] [CLS]

        Each image block wraps `num_soft_tokens_per_image` soft tokens with:
            [\\n\\n, <SOI>, <soft tokens>, <EOI>, \\n\\n]

        Raw vision tokens returned by the Gemma-4 vision encoder live in
        `vision_proj_dim` and are projected to `embed_dim` via the LLM embedder's
        `encode_vision` before concatenation.

        Returns:
            (tokens, input_mask, attn_mask, token_ids)
            - tokens: [B, seq_len, embed_dim]
            - input_mask: [B, seq_len] bool
            - attn_mask: [B, seq_len, seq_len] bool — causal, with bidirectional actions
            - token_ids: [B, seq_len] int32 — vocabulary IDs for per-layer-input lookup.
              Text positions carry their true ids; all non-text positions (images, state,
              actions, CLS) carry the PAD id (0).
        """
        from openpi.models.tokenizer import Gemma4Tokenizer

        batch_size = observation.state.shape[0]
        num_soft = self._num_soft_tokens_per_image
        tokens_per_block = num_soft + 4  # \n\n, SOI, soft..., EOI, \n\n

        special_emb = self._get_gemma4_special_embeddings()  # [1, 4, D]
        newline_emb = special_emb[:, 1 : 2, :]
        soi_emb = special_emb[:, 2 : 3, :]
        eoi_emb = special_emb[:, 3 : 4, :]

        tokens: list[jax.Array] = []
        input_mask: list[jax.Array] = []
        token_ids: list[jax.Array] = []

        # BOS comes from tokenized_prompt, matching the original PaliGemma path.
        bos_ids = observation.tokenized_prompt[:, 0 : 1]
        tokens.append(self.PaliGemma.llm(bos_ids, method = "embed"))
        input_mask.append(observation.tokenized_prompt_mask[:, 0 : 1])
        token_ids.append(bos_ids.astype(jnp.int32))

        # Build image blocks: [\n\n, <SOI>, soft(num_soft), <EOI>, \n\n] per camera.
        block_ids_template = jnp.array(
            [Gemma4Tokenizer.NEWLINE_NEWLINE_ID, Gemma4Tokenizer.START_OF_IMAGE_ID]
            + [0] * num_soft
            + [Gemma4Tokenizer.END_OF_IMAGE_ID, Gemma4Tokenizer.NEWLINE_NEWLINE_ID],
            dtype = jnp.int32,
        )  # [tokens_per_block]

        for name in self._image_keys:
            raw_image_tokens, _ = self.PaliGemma.img(observation.images[name], train = False)
            # Project raw vision tokens (vision_proj_dim) -> embed_dim via LLM embedder.
            image_tokens = self.PaliGemma.llm(raw_image_tokens, method = "encode_vision")
            cam_mask = observation.image_masks[name]

            nn_b = jnp.broadcast_to(newline_emb, (batch_size, 1, self._embed_dim))
            soi_b = jnp.broadcast_to(soi_emb, (batch_size, 1, self._embed_dim))
            eoi_b = jnp.broadcast_to(eoi_emb, (batch_size, 1, self._embed_dim))

            block = jnp.concatenate([nn_b, soi_b, image_tokens, eoi_b, nn_b], axis = 1)  # [B, tokens_per_block, D]
            tokens.append(block)

            block_mask = einops.repeat(cam_mask, "b -> b s", s = tokens_per_block)
            input_mask.append(block_mask)
            token_ids.append(jnp.broadcast_to(block_ids_template[None, :], (batch_size, tokens_per_block)))

        # tokenized_prompt = [BOS, <SOI>_1, ..., <SOI>_N, text, pad...]
        # BOS is embedded above; image markers are replaced by full image blocks.
        num_cameras = self._num_cameras
        text_only = observation.tokenized_prompt[:, 1 + num_cameras :]
        text_mask = observation.tokenized_prompt_mask[:, 1 + num_cameras :]
        text_emb = self.PaliGemma.llm(text_only, method = "embed")
        tokens.append(text_emb)
        input_mask.append(text_mask)
        token_ids.append(text_only.astype(jnp.int32))
        text_start = 1 + self._num_cameras * tokens_per_block
        text_end = text_start + text_only.shape[1]

        # State token (PAD id — no natural text token for continuous state).
        if not self._no_state:
            state_token = self.state_proj(observation.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))
            token_ids.append(jnp.zeros((batch_size, 1), dtype = jnp.int32))

        # Action tokens (PAD id — no natural text token for continuous actions).
        if self._action_conditioned:
            action_tokens = self.action_proj(action)
            tokens.append(action_tokens)
            input_mask.append(action_mask)
            token_ids.append(jnp.zeros((batch_size, self._action_horizon), dtype = jnp.int32))

        # CLS token
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, self._embed_dim))
        tokens.append(cls_tokens)
        input_mask.append(jnp.ones((batch_size, 1), dtype = jnp.bool_))
        token_ids.append(jnp.zeros((batch_size, 1), dtype = jnp.int32))

        tokens = jnp.concatenate(tokens, axis = 1)
        input_mask = jnp.concatenate(input_mask, axis = 1)
        token_ids = jnp.concatenate(token_ids, axis = 1)

        suffix_mask = None
        if observation.subtask_start_index is not None:
            prompt_offset = 1 + self._num_cameras
            subtask_start_index = observation.subtask_start_index - prompt_offset
            seq_positions = jnp.arange(tokens.shape[1])
            global_subtask_start = text_start + subtask_start_index
            suffix_mask = (seq_positions[None, :] >= global_subtask_start[:, None]) & (
                seq_positions[None, :] < text_end
            )

        action_block_start = None
        action_block_length = None
        if self._action_conditioned:
            action_block_start = text_end + (0 if self._no_state else 1)
            action_block_length = self._action_horizon

        attn_mask = make_gemma4_attn_mask(
            input_mask,
            suffix_mask = suffix_mask,
            action_start = action_block_start if self._gemma4_action_block_bidir else None,
            action_length = action_block_length if self._gemma4_action_block_bidir else None,
        )

        return tokens, input_mask, attn_mask, token_ids

    def decode(self, x: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        return self.PaliGemma.llm(x, method = "decode")

    def _next_token_outputs(
        self,
        output: jax.Array,
        observation: _model.Observation,
    ) -> dict[str, jax.Array]:
        if observation.subtask_start_index is None:
            raise ValueError("subtask_start_index is required for next-token outputs.")
        if observation.subtask_end_index is None:
            raise ValueError("subtask_end_index is required for next-token outputs.")
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("tokenized_prompt and tokenized_prompt_mask are required for next-token outputs.")

        if self._is_gemma4:
            num_soft = self._num_soft_tokens_per_image
            tokens_per_block = num_soft + 4
            text_start = 1 + self._num_cameras * tokens_per_block
            prompt_offset = 1 + self._num_cameras
            prompt_tokens = observation.tokenized_prompt[:, prompt_offset:]
            prompt_mask = observation.tokenized_prompt_mask[:, prompt_offset:]
            subtask_start_index = observation.subtask_start_index - prompt_offset
            subtask_end_index = observation.subtask_end_index - prompt_offset
        else:
            text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
            prompt_tokens = observation.tokenized_prompt
            prompt_mask = observation.tokenized_prompt_mask
            subtask_start_index = observation.subtask_start_index
            subtask_end_index = observation.subtask_end_index

        text_len = prompt_tokens.shape[1]
        text_hidden = output[:, text_start : text_start + text_len, :]

        candidate_embeddings = text_hidden[:, :-1, :]
        candidate_targets = prompt_tokens[:, 1:]
        candidate_mask = prompt_mask[:, :-1] & prompt_mask[:, 1:]

        first_predict_pos = jnp.maximum(subtask_start_index - 1, 0)
        last_predict_pos = subtask_end_index - 1
        window_offsets = jnp.arange(MAX_SUBTASK_TOKENS_FOR_LOSS, dtype = first_predict_pos.dtype)[None, :]
        predict_positions = first_predict_pos[:, None] + window_offsets
        max_candidate_pos = candidate_targets.shape[1] - 1
        gather_positions = jnp.clip(predict_positions, 0, max_candidate_pos)
        valid_positions = (
            (predict_positions <= last_predict_pos[:, None])
            & (predict_positions <= max_candidate_pos)
            & jnp.take_along_axis(candidate_mask, gather_positions, axis = 1)
        )
        return {
            "next_token_embeddings": jnp.take_along_axis(candidate_embeddings, gather_positions[:, :, None], axis = 1),
            "next_token_targets": jnp.take_along_axis(candidate_targets, gather_positions, axis = 1),
            "next_token_mask": valid_positions,
        }

    @override
    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> (
        at.Float[at.Array, "*b feature_dim"]
        | tuple[at.Float[at.Array, "*b feature_dim"], dict[str, at.Array]]
        | tuple[at.Float[at.Array, "*b feature_dim"], at.Float[at.Array, "*b _n"]]
    ):
        """Compute features from observation (and optionally action) for value prediction.

        Args:
            observation: Observation containing images, image_masks, state,
                        tokenized_prompt, and tokenized_prompt_mask.
                        Images should already be in [-1, 1] range.
            action: Optional _model.Actions with:
                    - actions: [B, action_horizon, action_dim] action chunk
                    If action_conditioned=True, this should be provided.
            rng: Optional random key for image augmentation. If provided,
                 enables augmentation during training.

        Returns:
            When rng is not None (training): features of shape [batch, embed_dim].
            When rng is None (inference): (features, attn_scores) where attn_scores is
                [batch, n_modalities] = mean over Gemma layers of CLS attention, grouped
                by modality (img1..imgN, text, state, [actions if Q]).
        """
        if prefix_cache is not None and (self._is_gemma3 or self._is_gemma4):
            raise ValueError("prefix_cache is only supported for original PaliGemma value networks.")

        # Preprocess observation (handles resizing, default masks, augmentation)
        # train=True enables augmentation when rng is provided
        train = rng is not None
        if prefix_cache is None:
            observation = _model.preprocess_observation(rng, observation, train = train, image_resolution = self._image_size)

        # Extract action array and mask if action_conditioned
        action_array = None
        action_mask_array = None
        if self._action_conditioned:
            action_array = action  # [B, action_horizon, action_dim]
            assert action_array.shape[1] == self._action_horizon
            action_mask_array = observation.action_mask  # [B, action_horizon] or None

        if prefix_cache is not None:
            if train:
                raise ValueError("prefix_cache is only supported for inference.")
            if not self._action_conditioned:
                raise ValueError("prefix_cache is only supported for action-conditioned networks.")

            kv_cache, prefix_mask = prefix_cache

            suffix_tokens = []
            suffix_mask = []
            suffix_ar_mask = []

            action_tokens = self.action_proj(action_array)
            suffix_tokens.append(action_tokens)
            suffix_mask.append(action_mask_array)
            suffix_ar_mask.append(True)
            suffix_ar_mask += [False] * (self._action_horizon - 1)

            cls_tokens = jnp.broadcast_to(self.cls_token.value, (action_array.shape[0], 1, self._embed_dim))
            suffix_tokens.append(cls_tokens)
            suffix_mask.append(jnp.ones((action_array.shape[0], 1), dtype = jnp.bool_))
            suffix_ar_mask.append(True)

            suffix_tokens = jnp.concatenate(suffix_tokens, axis = 1)
            suffix_mask = jnp.concatenate(suffix_mask, axis = 1)
            suffix_ar_mask = jnp.array(suffix_ar_mask)

            suffix_to_suffix = make_attn_mask(suffix_mask, suffix_ar_mask)
            suffix_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s = suffix_tokens.shape[1])
            attn_mask = jnp.concatenate([suffix_to_prefix, suffix_to_suffix], axis = -1)
            positions = jnp.sum(prefix_mask.astype(jnp.int32), axis = -1)[:, None] + jnp.cumsum(suffix_mask.astype(jnp.int32), axis = -1) - 1

            (output,), _ = self.PaliGemma.llm(
                [suffix_tokens],
                mask = attn_mask,
                positions = positions,
                kv_cache = kv_cache,
                adarms_cond = [None],
            )
            return output[:, -1, :]

        # Build embeddings and attention mask
        gemma4_token_ids = None
        if self._is_gemma4:
            (
                tokens,
                input_mask,
                attn_mask,
                gemma4_token_ids,
            ) = self._embed_sequence_gemma4(
                observation, action = action_array, action_mask = action_mask_array
            )
        elif self._is_gemma3:
            tokens, input_mask, attn_mask = self._embed_sequence_gemma3(
                observation, action = action_array, action_mask = action_mask_array
            )
        else:
            tokens, input_mask, ar_mask, suffix_mask = self._embed_sequence(
                observation, action = action_array, action_mask = action_mask_array
            )
            attn_mask = make_attn_mask(input_mask, ar_mask, suffix_mask = suffix_mask)

        # Compute positions: cumsum of valid positions, starting from 0
        positions = jnp.cumsum(input_mask.astype(jnp.int32), axis = 1) - 1

        # Gemma 4 per-layer-input (computed once from embeddings + token ids).
        gemma4_per_layer_input = None
        if self._is_gemma4 and self._gemma4_per_layer_input_dim > 0:
            gemma4_per_layer_input = self.PaliGemma.llm(
                tokens, gemma4_token_ids, method = "encode_per_layer_input",
            )

        llm_extra_kwargs: dict[str, jax.Array] = {}
        if self._is_gemma4:
            llm_extra_kwargs["per_layer_input"] = gemma4_per_layer_input

        if train:
            # Forward through LLM (single expert, no adarms)
            (output,), _ = self.PaliGemma.llm(
                [tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None],
                **llm_extra_kwargs,
            )
            # Extract CLS token output (last position) for value prediction
            cls_features = output[:, -1, :]  # [B, embed_dim]
            if self._use_layernorm:
                cls_features = self.cls_layer_norm(cls_features)
            if observation.subtask_start_index is not None:
                if self._is_gemma3:
                    raise ValueError(
                        "subtask_start_index is not supported with the Gemma3 PaliGemma value network."
                    )
                return cls_features, self._next_token_outputs(output, observation)
            return cls_features

        # Inference: also return per-modality CLS attention scores
        (output,), _, all_cls_attn = self.PaliGemma.llm(
            [tokens],
            mask = attn_mask,
            positions = positions,
            adarms_cond = [None],
            return_cls_attention_score_distribution = True,
            **llm_extra_kwargs,
        )
        cls_features = output[:, -1, :]  # [B, embed_dim]
        if self._use_layernorm:
            cls_features = self.cls_layer_norm(cls_features)

        # all_cls_attn: [B, L, S]; mean over L -> [B, S] -> group by modality -> [B, n_modalities]
        cls_attn_mean = all_cls_attn.mean(axis = 1)
        attn_scores = self._group_attn_scores(cls_attn_mean, input_mask)

        return cls_features, attn_scores

    def compute_prefix_cache(
        self, observation: _model.Observation
    ) -> tuple[at.Array, at.Array]:
        """Compute and return the prefix KV cache for the given observation."""
        if self._is_gemma3 or self._is_gemma4:
            raise ValueError("compute_prefix_cache is only supported for original PaliGemma value networks.")

        observation = _model.preprocess_observation(None, observation, train = False, image_resolution = self._image_size)
        prefix_tokens_list, prefix_mask_list, prefix_ar_mask_list = self._embed_prefix(observation)
        prefix_tokens = jnp.concatenate(prefix_tokens_list, axis = 1)
        prefix_mask = jnp.concatenate(prefix_mask_list, axis = 1)
        prefix_ar_mask = jnp.array(prefix_ar_mask_list)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask.astype(jnp.int32), axis = 1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens], mask = prefix_attn_mask, positions = positions)
        return kv_cache, prefix_mask

    def _group_attn_scores(self, cls_attn_mean: jax.Array, input_mask: jax.Array | None = None) -> jax.Array:
        """Group per-position CLS attention [B, S] into per-modality scores [B, n_modalities].

        Modality order: [img1, img2, ..., imgN, text, (state if not no_state), (actions if Q)]
        Gemma 4 uses a masked sum per group (skips padding) and excludes image
        demarcation tokens. Other variants preserve the historical summed
        grouping. The CLS self-attention at the last sequence position is
        excluded from all groups, so the per-modality scores sum to ~1 across
        groups (the small remainder is the CLS-self-attention probability).
        """
        attn_parts = []

        if self._is_gemma4:
            if input_mask is None:
                raise ValueError("input_mask is required for Gemma 4 attention-score grouping.")

            def _masked_sum(start: int, end: int) -> jax.Array:
                group_mask = input_mask[:, start : end]
                group_scores = cls_attn_mean[:, start : end]
                return jnp.sum(jnp.where(group_mask, group_scores, 0.0), axis = -1)

            num_soft = self._num_soft_tokens_per_image
            tokens_per_block = num_soft + 4
            for i in range(self._num_cameras):
                block_start = 1 + i * tokens_per_block
                patch_start = block_start + 2
                patch_end = patch_start + num_soft
                attn_parts.append(_masked_sum(patch_start, patch_end))

            text_start = 1 + self._num_cameras * tokens_per_block
            text_end = text_start + self._max_token_len - 1
            attn_parts.append(_masked_sum(text_start, text_end))
            if not self._no_state:
                attn_parts.append(_masked_sum(text_end, text_end + 1))
                action_start = text_end + 1
            else:
                action_start = text_end
            if self._action_conditioned:
                attn_parts.append(_masked_sum(action_start, action_start + self._action_horizon))

            return jnp.stack(attn_parts, axis = -1)

        if self._is_gemma3:
            # Layout:
            #   [BOS] [img_block_1] ... [img_block_N] [text] [state] [(actions)] [CLS]
            # Each image block: [\n\n, <SOI>, <soft tokens>, <EOI>, \n\n].
            # We only sum the soft-token positions (offsets 2..2+N).
            num_soft = NUM_PATCHES_PER_IMAGE
            tokens_per_block = num_soft + 4
            for i in range(self._num_cameras):
                block_start = 1 + i * tokens_per_block
                patch_start = block_start + 2
                patch_end = patch_start + num_soft
                attn_parts.append(cls_attn_mean[:, patch_start : patch_end].sum(axis = -1))

            # Text starts after BOS + all image blocks.
            # The tokenizer emits [BOS, SOI x num_cameras, text, pad] of length
            # max_token_len + num_cameras; _embed_sequence_gemma4 strips [:, 1 + num_cameras:],
            # leaving (max_token_len - 1) text positions inserted here.
            text_start = 1 + self._num_cameras * tokens_per_block
            text_end = text_start + self._max_token_len - 1
        else:
            for i in range(self._num_cameras):
                start = i * NUM_PATCHES_PER_IMAGE
                end = start + NUM_PATCHES_PER_IMAGE
                attn_parts.append(cls_attn_mean[:, start : end].sum(axis = -1))

            text_start = self._num_cameras * NUM_PATCHES_PER_IMAGE
            text_end = text_start + self._max_token_len

        attn_parts.append(cls_attn_mean[:, text_start : text_end].sum(axis = -1))  # text
        if not self._no_state:
            attn_parts.append(cls_attn_mean[:, text_end])  # state (single token)
            action_start = text_end + 1
        else:
            action_start = text_end
        if self._action_conditioned:
            attn_parts.append(cls_attn_mean[:, action_start : action_start + self._action_horizon].sum(axis = -1))

        return jnp.stack(attn_parts, axis = -1)


# =============================================================================
# Inline self-test: python src/openpi/value_functions/networks/paligemma.py
# =============================================================================
if __name__ == "__main__":

    def _check(condition: bool, msg: str) -> None:
        if not condition:
            raise AssertionError(f"FAIL: {msg}")
        print(f"  PASS: {msg}")

    # Sequence layout used by both cases:
    #   [img (3)] [text (4)] [state (1)] [CLS (1)]   total = 9
    N_IMG, N_TEXT = 3, 4
    SEQ_LEN = N_IMG + N_TEXT + 2  # +state +CLS
    IMG = slice(0, N_IMG)
    TEXT = slice(N_IMG, N_IMG + N_TEXT)
    STATE = N_IMG + N_TEXT
    CLS = SEQ_LEN - 1

    # =========================================================================
    # Case 1 — no subtask (subtask_start_index is None)
    # =========================================================================
    print("Case 1: No subtask text")

    ar = jnp.array([False] * N_IMG + [False] * N_TEXT + [True, True])
    inp = jnp.ones((1, SEQ_LEN), dtype=jnp.bool_)
    m = make_attn_mask(inp, ar)[0]

    _check(m[IMG, IMG].all().item(), "img <-> img bidirectional")
    _check(m[IMG, TEXT].all().item(), "img -> text")
    _check(m[TEXT, IMG].all().item(), "text -> img")
    _check(m[TEXT, TEXT].all().item(), "text <-> text bidirectional")

    _check(m[STATE, IMG].all().item(), "state -> img")
    _check(m[STATE, TEXT].all().item(), "state -> text")
    _check(not m[IMG, STATE].any().item(), "img -/-> state")
    _check(not m[TEXT, STATE].any().item(), "text -/-> state")

    _check(m[CLS, :CLS].all().item(), "CLS -> all prior")
    _check(not m[:CLS, CLS].any().item(), "others -/-> CLS")

    # =========================================================================
    # Case 2 — with subtask text (batch=2, different subtask starts)
    # =========================================================================
    print("\nCase 2: With subtask text (batch=2, subtask starts at text offset 2 and 3)")

    N_PREFIX_A, N_SUFFIX_A = 2, 2  # batch element 0
    N_PREFIX_B, N_SUFFIX_B = 3, 1  # batch element 1

    inp2 = jnp.ones((2, SEQ_LEN), dtype=jnp.bool_)

    # Build per-element ar_mask [B, N]: suffix text positions + state + CLS are True.
    base_ar = jnp.array([False] * N_IMG + [False] * N_TEXT + [True, True])  # [N]
    subtask_start_index = jnp.array([N_PREFIX_A, N_PREFIX_B])  # within text region
    text_start = N_IMG
    seq_positions = jnp.arange(SEQ_LEN)
    global_subtask_start = text_start + subtask_start_index
    suffix = (seq_positions[None, :] >= global_subtask_start[:, None]) & (
        seq_positions[None, :] < text_start + N_TEXT
    )
    ar2 = base_ar[None, :] | suffix  # [2, N]

    m2 = make_attn_mask(inp2, ar2, suffix_mask=suffix)

    for b, (n_pre, n_suf) in enumerate([(N_PREFIX_A, N_SUFFIX_A), (N_PREFIX_B, N_SUFFIX_B)]):
        tag = f"[b={b}, prefix={n_pre}, suffix={n_suf}]"
        mb = m2[b]

        prefix_text = slice(N_IMG, N_IMG + n_pre)
        suffix_text = slice(N_IMG + n_pre, N_IMG + n_pre + n_suf)

        _check(mb[IMG, IMG].all().item(), f"{tag} img <-> img bidirectional")
        _check(mb[IMG, prefix_text].all().item(), f"{tag} img -> prefix_text")
        _check(mb[prefix_text, IMG].all().item(), f"{tag} prefix_text -> img")
        _check(mb[prefix_text, prefix_text].all().item(), f"{tag} prefix_text <-> prefix_text bidirectional")

        _check(not mb[prefix_text, suffix_text].any().item(), f"{tag} prefix_text -/-> suffix_text")
        _check(not mb[IMG, suffix_text].any().item(), f"{tag} img -/-> suffix_text")

        _check(mb[suffix_text, prefix_text].all().item(), f"{tag} suffix_text -> prefix_text")
        _check(mb[suffix_text, IMG].all().item(), f"{tag} suffix_text -> img")

        # Suffix tokens are causal among themselves.
        for qi in range(n_suf):
            for ki in range(n_suf):
                q_abs = N_IMG + n_pre + qi
                k_abs = N_IMG + n_pre + ki
                if ki <= qi:
                    _check(mb[q_abs, k_abs].item(), f"{tag} suffix[{qi}] -> suffix[{ki}] (causal ok)")
                else:
                    _check(not mb[q_abs, k_abs].item(), f"{tag} suffix[{qi}] -/-> suffix[{ki}] (causal block)")

        _check(mb[STATE, IMG].all().item(), f"{tag} state -> img")
        _check(mb[STATE, prefix_text].all().item(), f"{tag} state -> prefix_text")
        _check(not mb[STATE, suffix_text].any().item(), f"{tag} state -/-> suffix_text")

        _check(mb[CLS, IMG].all().item(), f"{tag} CLS -> img")
        _check(mb[CLS, prefix_text].all().item(), f"{tag} CLS -> prefix_text")
        _check(mb[CLS, STATE].item(), f"{tag} CLS -> state")
        _check(not mb[CLS, suffix_text].any().item(), f"{tag} CLS -/-> suffix_text")

        _check(not mb[:CLS, CLS].any().item(), f"{tag} others -/-> CLS")

    print("\nAll tests passed!")
