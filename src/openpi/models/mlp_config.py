"""Configuration for the MLP model."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.mlp import MLP


@dataclasses.dataclass(frozen=True)
class MLPConfig(_model.BaseModelConfig):
    """Configuration for a simple MLP policy model.

    This model is designed for state-only environments like D4RL (antmaze, etc.)
    where there are no image observations.
    """

    # State dimension (observation space size).
    state_dim: int = 29  # Default for antmaze

    # Action dimension.
    action_dim: int = 8  # Default for antmaze

    # Action horizon (number of actions to predict at once).
    # For simple MLP, we typically predict single actions.
    action_horizon: int = 1

    # Hidden layer dimensions.
    hidden_dims: tuple[int, ...] = (256, 256)

    # Data type for model parameters.
    dtype: str = "float32"

    # Action bounds for clipping (per dimension). If None, no clipping is applied.
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    # Maximum token length (not used by MLP, but required by base class).
    max_token_len: int = 1

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.MLP

    @override
    def create(self, rng: at.KeyArrayLike) -> "MLP":
        from openpi.models.mlp import MLP

        return MLP(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Returns the input specification for the model.

        MLP is designed for state-only environments (like D4RL), so we don't
        include any image specifications.
        """
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                # No images for state-only D4RL environments
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
                tokenized_prompt=None,
                tokenized_prompt_mask=None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec
