"""MLP implementations for value functions.

This module provides a unified MLP-based value function that supports both
regression (MSE) and categorical (HL-Gauss) objectives. The value function can be
configured to be action-conditioned (Q-function) or not (V-function).
"""

import dataclasses
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.model import ModelType
from openpi.shared import array_typing as at
from openpi.value_functions import hl_gauss as _hl_gauss
from openpi.value_functions.base import BaseValueFunction
from openpi.value_functions.base import BaseValueFunctionConfig
from openpi.value_functions.base import Transition


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
class ValueMLPConfig(BaseValueFunctionConfig):
    """Configuration for unified MLP value function.

    Supports both regression (MSE) and categorical (HL-Gauss) objectives.
    Can be configured as V(s) or Q(s,a) based on action_conditioned parameter.
    """

    # State dimension.
    state_dim: int

    # Value function configs use the MLP_CRITIC model type.
    model_type: ModelType = ModelType.MLP_CRITIC

    # Whether to condition on actions (Q-function) or not (V-function).
    action_conditioned: bool = False

    # Action dimension (only used if action_conditioned=True).
    action_dim: int = 8

    # Action horizon (only used if action_conditioned=True).
    action_horizon: int = 1

    # Hidden layer dimensions.
    hidden_dims: tuple[int, ...] = (256, 256)

    # Data type for model parameters.
    dtype: str = "float32"

    # Use layer normalization after each hidden layer.
    use_layer_norm: bool = False

    # Scale for orthogonal initialization.
    orthogonal_init_scale: float = 1e-2

    # ========== HL-Gauss (Categorical) Configuration ==========
    # If True, use HL-Gauss categorical loss. If False, use MSE regression loss.
    use_hl_gauss: bool = False

    # These are only used when use_hl_gauss=True
    v_min: float = 0.0  # Minimum value of the support
    v_max: float = 1.0  # Maximum value of the support
    num_bins: int = 51  # Number of bins
    sigma: float = 0.75  # Gaussian smoothing standard deviation

    @override
    def create(self, rng: at.KeyArrayLike) -> "ValueMLP":
        return ValueMLP(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size], jnp.float32)
        if self.action_conditioned:
            actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
            return obs, actions, target
        return obs, target


class ValueMLP(BaseValueFunction):
    """Unified MLP value function with support for regression and HL-Gauss losses.

    Can be V(s) or Q(s,a) depending on config.
    """

    def __init__(self, config: ValueMLPConfig, rngs: nnx.Rngs):
        super().__init__()

        self.state_dim = config.state_dim
        self.action_conditioned = config.action_conditioned
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_dims = config.hidden_dims
        self.use_layer_norm = config.use_layer_norm
        self.use_hl_gauss = config.use_hl_gauss

        # HL-Gauss parameters (only used when use_hl_gauss=True)
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.num_bins = config.num_bins
        self.sigma = config.sigma

        # Kernel initializer - use cached function for consistent pytree identity
        kernel_init = _get_orthogonal_init(config.orthogonal_init_scale)

        # Build MLP layers
        layers: list[nnx.Linear] = []
        layer_norms: list[nnx.LayerNorm | None] = []
        if config.action_conditioned:
            in_dim = config.state_dim + config.action_horizon * config.action_dim
        else:
            in_dim = config.state_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs))
            if config.use_layer_norm:
                layer_norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            else:
                layer_norms.append(None)
            in_dim = hidden_dim

        # Output layer: num_bins for HL-Gauss, 1 for regression
        out_dim = config.num_bins if config.use_hl_gauss else 1
        layers.append(nnx.Linear(in_dim, out_dim, kernel_init=kernel_init, rngs=rngs))

        self.layers = layers
        self.layer_norms = layer_norms

    def _forward(
        self,
        state: at.Float[at.Array, "b s"],
        action: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> at.Float[at.Array, "b out"]:
        """Forward pass through the MLP."""
        if self.action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned value function")
            batch_size = action.shape[0]
            action_flat = action.reshape(batch_size, -1)
            x = jnp.concatenate([state, action_flat], axis=-1)
        else:
            x = state

        # Hidden layers with Swish activation
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_layer_norm and self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
            x = nnx.swish(x)

        # Output layer (no activation)
        return self.layers[-1](x)

    def compute_logits(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "b num_bins"]:
        """Compute raw logits over bins (only valid when use_hl_gauss=True)."""
        if not self.use_hl_gauss:
            raise ValueError("compute_logits is only available when use_hl_gauss=True")
        return self._forward(observation.state, action)

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        output = self._forward(observation.state, action)
        if self.use_hl_gauss:
            return _hl_gauss.logits_to_expected_value(output, self.v_min, self.v_max)
        return output.squeeze(-1)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del rng  # Unused for value function
        action = transition.action if self.action_conditioned else None
        output = self._forward(transition.observation.state, action)
        # MC learning: target is the Monte-Carlo return
        target = transition.mc_return

        if self.use_hl_gauss:
            per_sample_loss = _hl_gauss.hl_gauss_loss(output, target, self.v_min, self.v_max, self.sigma)
            predicted_value = _hl_gauss.logits_to_expected_value(output, self.v_min, self.v_max)
        else:
            predicted_value = output.squeeze(-1)
            td_error = predicted_value - target
            per_sample_loss = jnp.square(td_error)

        td_error = predicted_value - target

        info = {
            "predicted_value_mean": jnp.mean(predicted_value),
            "predicted_value_std": jnp.std(predicted_value),
            "target_value_mean": jnp.mean(target),
            "target_value_std": jnp.std(target),
            "td_error_mean": jnp.mean(td_error),
            "td_error_std": jnp.std(td_error),
        }
        return per_sample_loss, info
