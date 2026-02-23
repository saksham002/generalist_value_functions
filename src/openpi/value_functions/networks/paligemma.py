"""PaliGemma network for V(s, l) value functions with image+state+text inputs.

Uses a pre-trained PaliGemma backbone (ViT + Gemma LLM) as a feature encoder
for training subtask-conditioned state value functions V(s, l) where l is a subtask.

Sequence structure:
    [img1_patches...] [img2_patches...] [img3_patches...] [text_tokens...] [state_embed] [CLS]

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

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.model import IMAGE_KEYS
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.value_functions.networks.base_networks import BaseValueNetwork


# Number of patches per image (224/14 = 16, 16*16 = 256)
NUM_PATCHES_PER_IMAGE = 256


def make_attn_mask(
    input_mask: jax.Array,
    mask_ar: jax.Array,
) -> jax.Array:
    """Create attention mask for value network with state and CLS tokens at the end.

    Sequence structure: [images] [text] [state] [CLS]

    Desired attention pattern:
    - Images/Text: bidirectional with each other
    - State: can attend to images/text, cannot be attended by images/text
    - CLS (last position): can attend to ALL, cannot be attended by others

    The ar_mask format [0, 0...0, 1, 1] naturally achieves this via causal masking:
    - Positions with ar_mask=0: bidirectional (images/text)
    - Positions with ar_mask=1: causal (state, CLS) - can attend to earlier but not later

    Args:
        input_mask: bool[B, N] true if part of the input, false if padding.
        mask_ar: bool[B, N] or bool[N]. Format: [0, 0...0, 1, 1] where 0=bidirectional
            (images/text), 1=causal (state, CLS).

    Returns:
        Attention mask [B, N, N] where True means "can attend".
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)

    # Standard causal/bidirectional mask construction
    # cumsum groups tokens by their ar_mask value
    # Tokens with same cumsum value can attend to each other (bidirectional)
    # Tokens with higher cumsum can attend to lower (causal)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, N, N]

    # Apply valid positions mask (both positions must be valid)
    valid = input_mask[:, None, :] * input_mask[:, :, None]
    mask = jnp.logical_and(mask, valid)

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

    # Whether to freeze the PaliGemma backbone during training
    freeze_backbone: bool = True

    # Maximum token length for text prompts
    max_token_len: int = 48

    # PaliGemma variant (matches pi0 config)
    paligemma_variant: str = "gemma_2b"

    # Dtype for computations
    dtype: str = "bfloat16"

    # Whether this network is action-conditioned (Q(s,a) vs V(s))
    action_conditioned: bool = False

    # Action dimension (required when action_conditioned=True)
    action_dim: int = 14

    # Number of actions in chunk (required when action_conditioned=True)
    action_horizon: int = 1

    # Whether to mask out the state token in the attention mask (for ablation studies)
    mask_state: bool = False

    # Fix order in which to iterate through keys
    image_keys: tuple[str, str, str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def create(self, rng: at.KeyArrayLike) -> PaliGemmaValueNetwork:
        """Create a new PaliGemma value network with initialized parameters."""
        return PaliGemmaValueNetwork(self, rngs=nnx.Rngs(rng))


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

    def __init__(self, config: PaliGemmaNetworkConfig, rngs: nnx.Rngs):
        super().__init__()

        self.config = config
        self._action_conditioned = config.action_conditioned
        self._freeze_backbone = config.freeze_backbone
        self._num_cameras = config.num_cameras
        self._max_token_len = config.max_token_len
        self._action_horizon = config.action_horizon
        self._action_dim = config.action_dim
        self._mask_state = config.mask_state
        self._image_keys = config.image_keys

        # Get PaliGemma config (same as pi0.py)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        embed_dim = paligemma_config.width

        # Initialize Gemma LLM (single config, no action expert)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config],
                embed_dtype=config.dtype,
                adarms=False,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])

        # Initialize SigLIP image encoder
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # Initialize with a fake image
        fake_image = jnp.zeros((1, config.image_size[0], config.image_size[1], 3), dtype=jnp.float32)
        img.lazy_init(fake_image, train=False, rngs=rngs)

        # Store as PaliGemma dict (matches weight loader key structure)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # CLS token - learnable embedding for value extraction
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 0.02)

        # State projection: state_dim -> embed_dim (single token)
        self.state_proj = nnx.Linear(config.state_dim, embed_dim, rngs=rngs)

        # Action projection: action_dim -> embed_dim (one token per action in chunk)
        # Only created if action_conditioned is True
        self.action_proj: nnx.Linear | None = None
        if config.action_conditioned:
            self.action_proj = nnx.Linear(config.action_dim, embed_dim, rngs=rngs)

        # Feature dimension is the Gemma embedding dimension
        self._feature_dim = embed_dim
        self._embed_dim = embed_dim

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
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Embed full sequence: images + text + state + [actions] + CLS.

        Args:
            observation: Observation with images, state, and tokenized_prompt.
            action: Optional action chunk [B, action_horizon, action_dim] for Q(s,a).
            action_mask: Optional mask [B, action_horizon] for valid actions in chunk.

        Returns:
            Tuple of (tokens, input_mask, ar_mask)
            - tokens: [B, seq_len, embed_dim]
            - input_mask: [B, seq_len] bool
            - ar_mask: [seq_len] bool - format [0, 0...0, 1, 1] where:
              - 0s are for images/text (bidirectional)
              - 1s are for state, actions, and CLS (causal: can attend to earlier, not later)
        """
        batch_size = observation.state.shape[0]

        input_mask = []
        ar_mask = []
        tokens = []

        # 1. Embed images (following pi0.py pattern)
        for name in self._image_keys:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)

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
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            # Text tokens: bidirectional with images
            ar_mask += [False] * tokenized_inputs.shape[1]

        # 3. Add state token
        state_token = self.state_proj(observation.state)[:, None, :]  # [B, 1, embed_dim]
        tokens.append(state_token)
        if self._mask_state:
            input_mask.append(jnp.zeros((batch_size, 1), dtype=jnp.bool_))
        else:
            input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
        # State uses ar_mask=True (causal - can attend to images/text but not be attended)
        ar_mask.append(True)

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
        input_mask.append(jnp.ones((batch_size, 1), dtype=jnp.bool_))
        # CLS uses ar_mask=True (causal - can attend to all previous but not be attended)
        ar_mask.append(True)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @override
    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "*b feature_dim"] | tuple[at.Float[at.Array, "*b feature_dim"], at.Float[at.Array, "*b _n"]]:
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
        # Preprocess observation (handles resizing, default masks, augmentation)
        # train=True enables augmentation when rng is provided
        train = rng is not None
        observation = _model.preprocess_observation(rng, observation, train=train)

        # Extract action array and mask if action_conditioned
        action_array = None
        action_mask_array = None
        if self._action_conditioned:
            action_array = action  # [B, action_horizon, action_dim]
            assert action_array.shape[1] == self._action_horizon
            action_mask_array = observation.action_mask  # [B, action_horizon] or None

        # Build embeddings: [images] [text] [state] [actions (if Q(s,a))] [CLS]
        tokens, input_mask, ar_mask = self._embed_sequence(
            observation, action=action_array, action_mask=action_mask_array
        )

        # Compute positions: cumsum of valid positions, starting from 0
        positions = jnp.cumsum(input_mask.astype(jnp.int32), axis=1) - 1

        # Create attention mask
        # ar_mask is [0...0, 1, 1, ...] where state, actions, and CLS are causal
        attn_mask = make_attn_mask(input_mask, ar_mask)

        if train:
            # Forward through LLM (single expert, no adarms)
            (output,), _ = self.PaliGemma.llm(
                [tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None],
            )
            # Extract CLS token output (last position) for value prediction
            cls_features = output[:, -1, :]  # [B, embed_dim]
            return cls_features

        # Inference: also return per-modality CLS attention scores
        (output,), _, all_cls_attn = self.PaliGemma.llm(
            [tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None],
            return_cls_attention_score_distribution=True,
        )
        cls_features = output[:, -1, :]  # [B, embed_dim]

        # all_cls_attn: [B, L, S]; mean over L -> [B, S] -> group by modality -> [B, n_modalities]
        cls_attn_mean = all_cls_attn.mean(axis=1)
        attn_scores = self._group_attn_scores(cls_attn_mean)

        return cls_features, attn_scores

    def _group_attn_scores(self, cls_attn_mean: jax.Array) -> jax.Array:
        """Group per-position CLS attention [B, S] into per-modality scores [B, n_modalities].

        Modality order: [img1, img2, ..., imgN, text, state, (actions if Q)]
        Each image group sums over 256 patch positions; text/action groups sum over their tokens.
        The CLS self-attention at the last sequence position is excluded from all groups.
        """
        img_end = self._num_cameras * NUM_PATCHES_PER_IMAGE
        text_end = img_end + self._max_token_len

        attn_parts = []
        for i in range(self._num_cameras):
            start = i * NUM_PATCHES_PER_IMAGE
            end = start + NUM_PATCHES_PER_IMAGE
            attn_parts.append(cls_attn_mean[:, start:end].sum(axis=-1))
        attn_parts.append(cls_attn_mean[:, img_end:text_end].sum(axis=-1))  # text
        attn_parts.append(cls_attn_mean[:, text_end])  # state (single token)
        if self._action_conditioned:
            action_start = text_end + 1
            attn_parts.append(cls_attn_mean[:, action_start:action_start + self._action_horizon].sum(axis=-1))

        return jnp.stack(attn_parts, axis=-1)
