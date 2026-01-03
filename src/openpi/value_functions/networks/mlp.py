"""MLP network for value functions."""

import dataclasses
import functools

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions.networks.base import BaseValueNetwork


@functools.cache
def _get_orthogonal_init(scale: float):
    """Get a cached orthogonal initializer for the given scale.

    Caching ensures the same function object is returned for the same scale.
    This is required because jax.eval_shape and actual jit calls both invoke
    the model constructor, and we need the kernel_init function to have a
    consistent identity for pytree structure matching.
    """
    return nnx.initializers.orthogonal(scale=scale)


@dataclasses.dataclass(frozen=True)
class MLPNetworkConfig:
    """Configuration for MLP network."""

    state_dim: int
    action_conditioned: bool = False
    action_dim: int = 0
    action_horizon: int = 1
    hidden_dims: tuple[int, ...] = (256, 256)
    use_layer_norm: bool = False
    orthogonal_init_scale: float | None = None

    def create(self, rng: at.KeyArrayLike) -> "MLPNetwork":
        return MLPNetwork(self, rngs=nnx.Rngs(rng))


class MLPNetwork(BaseValueNetwork):
    """MLP network that computes features from observations (and optionally actions)."""

    def __init__(self, config: MLPNetworkConfig, rngs: nnx.Rngs):
        super().__init__()

        self.action_conditioned = config.action_conditioned
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self._hidden_dims = config.hidden_dims

        kernel_init = _get_orthogonal_init(config.orthogonal_init_scale) if config.orthogonal_init_scale else None

        layers: list[nnx.Linear] = []
        layer_norms: list[nnx.LayerNorm | None] = []

        if config.action_conditioned:
            in_dim = config.state_dim + config.action_horizon * config.action_dim
        else:
            in_dim = config.state_dim

        for hidden_dim in config.hidden_dims:
            if kernel_init is not None:
                layers.append(nnx.Linear(in_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs))
            else:
                layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            if config.use_layer_norm:
                layer_norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            else:
                layer_norms.append(None)
            in_dim = hidden_dim

        self.layers = layers
        self.layer_norms = layer_norms

    @property
    def feature_dim(self) -> int:
        return self._hidden_dims[-1]

    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b feature_dim"]:
        """Forward pass through MLP, returning final hidden layer output."""
        state = observation.state

        if self.action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned network")
            batch_size = action.shape[0]
            action_flat = action.reshape(batch_size, -1)
            x = jnp.concatenate([state, action_flat], axis=-1)
        else:
            x = state

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
            x = nnx.swish(x)

        return x
