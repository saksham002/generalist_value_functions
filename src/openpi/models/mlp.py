"""Simple MLP model for state-only environments like D4RL."""

import flax.nnx as nnx
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import mlp_config as _mlp_config
from openpi.models import model as _model
from openpi.shared import array_typing as at


class MLP(_model.BaseModel):
    """Simple MLP policy for state-only environments.

    This model takes state observations and predicts actions using a feedforward
    neural network. It is trained with MSE loss on action prediction.

    Unlike diffusion-based models like Pi0, this model directly outputs actions.
    """

    def __init__(self, config: _mlp_config.MLPConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.state_dim = config.state_dim
        self.hidden_dims = config.hidden_dims

        # Action bounds for constraining outputs (stored as tuples, converted to arrays at runtime)
        self._action_low = config.action_low
        self._action_high = config.action_high

        # Build MLP layers
        layers = []
        in_dim = config.state_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        # Output layer: predict action_horizon * action_dim values
        output_dim = config.action_horizon * config.action_dim
        layers.append(nnx.Linear(in_dim, output_dim, rngs=rngs))

        self.layers = layers
        self.deterministic = True

    def forward(self, state: at.Float[at.Array, "b s"]) -> at.Float[at.Array, "b ah ad"]:
        """Forward pass through the MLP.

        Args:
            state: State observations of shape [batch, state_dim].

        Returns:
            Predicted actions of shape [batch, action_horizon, action_dim].
            Actions are constrained to [action_low, action_high] via tanh if bounds are set.
        """
        x = state

        # Hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.relu(x)

        # Output layer (no activation yet)
        x = self.layers[-1](x)

        # Reshape to [batch, action_horizon, action_dim]
        batch_size = state.shape[0]
        actions = x.reshape(batch_size, self.action_horizon, self.action_dim)

        # Apply tanh to constrain actions to valid range
        if self._action_low is not None and self._action_high is not None:
            # tanh outputs [-1, 1], scale to [action_low, action_high]
            action_low = jnp.array(self._action_low)
            action_high = jnp.array(self._action_high)
            actions = jnp.tanh(actions)
            actions = (actions + 1.0) / 2.0 * (action_high - action_low) + action_low

        return actions

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute MSE loss for action prediction.

        Args:
            rng: Random key (unused for MLP but required by interface).
            observation: Observation containing state.
            actions: Ground truth actions of shape [batch, action_horizon, action_dim].
            train: Whether in training mode (unused for MLP).

        Returns:
            Per-timestep MSE loss of shape [batch, action_horizon].
        """
        # Get state from observation
        state = observation.state

        # Forward pass
        predicted_actions = self.forward(state)

        # Compute MSE loss per timestep (average over action dimension)
        # actions shape: [batch, action_horizon, action_dim]
        # predicted_actions shape: [batch, action_horizon, action_dim]
        return jnp.mean(jnp.square(predicted_actions - actions), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        **kwargs,
    ) -> _model.Actions:
        """Sample actions by running a forward pass.

        For MLP, this is simply a deterministic forward pass (no sampling involved).
        Actions are constrained to the valid range via tanh if bounds are set.

        Args:
            rng: Random key (unused for deterministic MLP).
            observation: Observation containing state.
            **kwargs: Additional arguments (unused).

        Returns:
            Predicted actions of shape [batch, action_horizon, action_dim].
        """
        return self.forward(observation.state)
