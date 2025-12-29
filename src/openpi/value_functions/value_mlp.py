"""MLP implementations for value functions.

This module provides MLP-based implementations of value functions for both
regression and categorical (HL-Gauss) objectives. Value functions can be
configured to be action-conditioned (Q-function) or not (V-function).
"""

import dataclasses

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

# =============================================================================
# Regression Value Function (MSE loss)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RegressionValueMLPConfig(BaseValueFunctionConfig):
    """Configuration for regression-based MLP value function.

    Can be configured as V(s) or Q(s,a) based on action_conditioned parameter.
    """

    # Value function configs use the MLP_CRITIC model type.
    model_type: ModelType = ModelType.MLP_CRITIC

    # State dimension.
    state_dim: int = 29  # Default for antmaze

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

    @override
    def create(self, rng: at.KeyArrayLike) -> "RegressionValueMLP":
        return RegressionValueMLP(self, rngs=nnx.Rngs(rng))

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


class RegressionValueMLP(BaseValueFunction):
    """MLP value function trained with MSE loss.

    Outputs a single scalar value. Can be V(s) or Q(s,a) depending on config.
    """

    def __init__(self, config: RegressionValueMLPConfig, rngs: nnx.Rngs):
        super().__init__()

        self.state_dim = config.state_dim
        self.action_conditioned = config.action_conditioned
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_dims = config.hidden_dims

        # Build MLP layers
        layers = []
        if config.action_conditioned:
            in_dim = config.state_dim + config.action_horizon * config.action_dim
        else:
            in_dim = config.state_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        # Output layer: single scalar value
        layers.append(nnx.Linear(in_dim, 1, rngs=rngs))

        self.layers = layers

    def _forward(
        self,
        state: at.Float[at.Array, "b s"],
        action: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> at.Float[at.Array, "b 1"]:
        """Forward pass through the MLP."""
        if self.action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned value function")
            batch_size = action.shape[0]
            action_flat = action.reshape(batch_size, -1)
            x = jnp.concatenate([state, action_flat], axis=-1)
        else:
            x = state

        # Hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer (no activation)
        return self.layers[-1](x)

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        return self._forward(observation.state, action).squeeze(-1)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        action = transition.action if self.action_conditioned else None
        predicted_value = self.compute_value(transition.observation, action)
        # MC learning: target is the Monte-Carlo return
        target = transition.mc_return
        td_error = predicted_value - target
        per_sample_loss = jnp.square(td_error)

        info = {
            "predicted_value_mean": jnp.mean(predicted_value),
            "predicted_value_std": jnp.std(predicted_value),
            "target_value_mean": jnp.mean(target),
            "target_value_std": jnp.std(target),
            "td_error_mean": jnp.mean(td_error),
            "td_error_std": jnp.std(td_error),
        }
        return per_sample_loss, info


# =============================================================================
# Categorical Value Function (HL-Gauss loss)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CategoricalValueMLPConfig(BaseValueFunctionConfig):
    """Configuration for categorical (HL-Gauss) MLP value function.

    Can be configured as V(s) or Q(s,a) based on action_conditioned parameter.
    """

    # Required fields (no defaults) must come first
    v_min: float  # Minimum value of the support
    v_max: float  # Maximum value of the support

    # State dimension.
    state_dim: int

    # Fields with defaults
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

    num_bins: int = 51  # Number of bins
    sigma: float = 0.75  # Gaussian smoothing standard deviation

    @override
    def create(self, rng: at.KeyArrayLike) -> "CategoricalValueMLP":
        return CategoricalValueMLP(self, rngs=nnx.Rngs(rng))

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


class CategoricalValueMLP(BaseValueFunction):
    """MLP value function trained with HL-Gauss cross-entropy loss.

    Outputs logits over discrete bins. Can be V(s) or Q(s,a) depending on config.
    """

    def __init__(self, config: CategoricalValueMLPConfig, rngs: nnx.Rngs):
        super().__init__()

        self.state_dim = config.state_dim
        self.action_conditioned = config.action_conditioned
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hidden_dims = config.hidden_dims
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.num_bins = config.num_bins
        self.sigma = config.sigma

        # Build MLP layers
        layers = []
        if config.action_conditioned:
            in_dim = config.state_dim + config.action_horizon * config.action_dim
        else:
            in_dim = config.state_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        # Output layer: logits over bins
        layers.append(nnx.Linear(in_dim, config.num_bins, rngs=rngs))

        self.layers = layers

    def _forward(
        self,
        state: at.Float[at.Array, "b s"],
        action: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> at.Float[at.Array, "b num_bins"]:
        """Forward pass through the MLP, returning logits."""
        if self.action_conditioned:
            if action is None:
                raise ValueError("action required for action-conditioned value function")
            batch_size = action.shape[0]
            action_flat = action.reshape(batch_size, -1)
            x = jnp.concatenate([state, action_flat], axis=-1)
        else:
            x = state

        # Hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer (no activation - raw logits)
        return self.layers[-1](x)

    def compute_logits(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "b num_bins"]:
        """Compute raw logits over bins."""
        return self._forward(observation.state, action)

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        logits = self._forward(observation.state, action)
        return _hl_gauss.logits_to_expected_value(logits, self.v_min, self.v_max)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        action = transition.action if self.action_conditioned else None
        logits = self._forward(transition.observation.state, action)
        # MC learning: target is the Monte-Carlo return
        target = transition.mc_return
        per_sample_loss = _hl_gauss.hl_gauss_loss(logits, target, self.v_min, self.v_max, self.sigma)

        # Compute predicted value for logging
        predicted_value = _hl_gauss.logits_to_expected_value(logits, self.v_min, self.v_max)
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
