"""MLP network for value functions."""

import dataclasses
import functools

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions.networks.base_networks import BaseValueNetwork


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
        """Forward pass through MLP, returning final hidden layer output.

        Handles actions of various shapes:
        - [batch, action_dim]: 2D actions (flattened)
        - [batch, action_horizon, action_dim]: 3D actions
        - [batch, n, action_horizon, action_dim]: 4D multi-transition actions
        """
        state = observation.state

        if self.action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned network")
            # Flatten trailing action dimensions
            num_leading = state.ndim - 1
            action_dims = action.shape[num_leading:]
            action_flat_dim = 1
            for d in action_dims:
                action_flat_dim *= d
            action_flat = action.reshape(*action.shape[:num_leading], action_flat_dim)
            x = jnp.concatenate([state, action_flat], axis=-1)
        else:
            x = state

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
            x = nnx.swish(x)

        return x


@dataclasses.dataclass(frozen=True)
class MultiMLPNetworkConfig:
    """Configuration for MLP network that processes multiple transitions jointly.

    For multi-state value functions: all n (state, action) pairs are concatenated
    into a single input, and n separate feature vectors are output.
    """

    state_dim: int
    num_transitions_per_sample: int
    action_conditioned: bool = False
    action_dim: int = 0
    action_horizon: int = 1
    hidden_dims: tuple[int, ...] = (256, 256)
    use_layer_norm: bool = False
    orthogonal_init_scale: float | None = None

    def create(self, rng: at.KeyArrayLike) -> "MultiMLPNetwork":
        return MultiMLPNetwork(self, rngs=nnx.Rngs(rng))


class MultiMLPNetwork(BaseValueNetwork):
    """MLP network for multi-state value functions.

    Takes n (state, action) pairs, concatenates all of them into a single input
    vector, and outputs n separate feature vectors for predicting n values.
    """

    def __init__(self, config: MultiMLPNetworkConfig, rngs: nnx.Rngs):
        super().__init__()

        self.action_conditioned = config.action_conditioned
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_transitions_per_sample = config.num_transitions_per_sample
        self._hidden_dims = config.hidden_dims

        kernel_init = _get_orthogonal_init(config.orthogonal_init_scale) if config.orthogonal_init_scale else None

        # Input dimension: all n (state, action) pairs concatenated
        single_input_dim = config.state_dim
        if config.action_conditioned:
            single_input_dim += config.action_horizon * config.action_dim
        total_input_dim = single_input_dim * config.num_transitions_per_sample

        # Build shared MLP layers
        layers: list[nnx.Linear] = []
        layer_norms: list[nnx.LayerNorm | None] = []
        in_dim = total_input_dim

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

        # Output projection: from hidden_dim to n * hidden_dim (one feature vector per transition)
        if kernel_init is not None:
            self.output_proj = nnx.Linear(
                config.hidden_dims[-1],
                config.num_transitions_per_sample * config.hidden_dims[-1],
                kernel_init=kernel_init,
                rngs=rngs,
            )
        else:
            self.output_proj = nnx.Linear(
                config.hidden_dims[-1],
                config.num_transitions_per_sample * config.hidden_dims[-1],
                rngs=rngs,
            )

    @property
    def feature_dim(self) -> int:
        return self._hidden_dims[-1]

    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b n feature_dim"]:
        """Forward pass: concatenate all transitions, process, output n features.

        Args:
            observation: With state shape [batch, n, state_dim].
            action: With shape [batch, n, action_horizon, action_dim] if action_conditioned.

        Returns:
            Features of shape [batch, n, feature_dim].
        """
        state = observation.state  # [batch, n, state_dim]
        batch_size = state.shape[0]

        if self.action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned network")
            # Handle both [batch, n, action_dim] and [batch, n, action_horizon, action_dim]
            num_transitions = action.shape[1]
            if action.ndim == 4:
                # [batch, n, action_horizon, action_dim] -> [batch, n, action_horizon * action_dim]
                action_horizon = action.shape[2]
                action_dim = action.shape[3]
                action_flat = action.reshape(batch_size, num_transitions, action_horizon * action_dim)
            else:
                # [batch, n, action_dim] - already flat
                action_flat = action
            # [batch, n, state_dim + flattened_action_dim]
            per_transition = jnp.concatenate([state, action_flat], axis=-1)
        else:
            per_transition = state

        if per_transition.ndim != 3:
            raise ValueError(
                f"MultiMLPNetwork expects 3D input (batch, n, dim), got {per_transition.ndim}D "
                f"with shape {per_transition.shape}"
            )

        # Flatten all n transitions into single input: [batch, n * single_dim]
        _, num_transitions, single_dim = per_transition.shape

        x = per_transition.reshape(batch_size, num_transitions * single_dim)

        # Process through MLP
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
            x = nnx.swish(x)

        # Project to n separate feature vectors
        x = self.output_proj(x)  # [batch, n * feature_dim]
        return x.reshape(batch_size, self.num_transitions_per_sample, self._hidden_dims[-1])
