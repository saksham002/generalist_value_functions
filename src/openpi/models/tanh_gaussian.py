"""TanhGaussian policy model for continuous control.

This module provides a tanh-squashed Gaussian policy suitable for SAC and
other maximum entropy RL algorithms. The policy outputs a diagonal Gaussian
distribution that is then squashed through tanh to bound actions.
"""

import dataclasses

import distrax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override
import tyro

from openpi.models import model as _model
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class TanhGaussianConfig(_model.BaseModelConfig):
    """Configuration for TanhGaussian policy.

    This policy outputs a tanh-squashed Gaussian distribution over actions,
    suitable for continuous control with bounded action spaces.
    """

    # State dimension (required)
    state_dim: int = tyro.MISSING

    # Action dimension (required) - override from parent
    action_dim: int = tyro.MISSING

    # Action horizon (number of timesteps to predict)
    action_horizon: int = 1

    # Not used for this model, but required by BaseModelConfig
    max_token_len: int = 0

    # Hidden layer dimensions
    hidden_dims: tuple[int, ...] = (256, 256)

    # Action bounds (for scaling after tanh from [-1,1] to [low, high])
    # If None, actions are in [-1, 1]
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    # Whether log_std is computed from state (True) or is a learned parameter (False).
    # When False, log_std is a state-independent learned parameter (matching IQL).
    # Ignored when std_parameterization="fixed".
    state_dependent_std: bool = True

    # Std parameterization: "exp", "softplus", or "fixed"
    #   - "exp"/"softplus": std computed via Dense layer (state_dependent_std controls state-dependence)
    #   - "fixed": constant std value (always state-independent, ignores state_dependent_std)
    std_parameterization: str = "exp"

    # Fixed std value (only used if std_parameterization == "fixed")
    fixed_std: float = 0.1

    # Bounds on log_std for numerical stability
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # Dropout rate for the policy network
    dropout_rate: float = 0.0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.TANH_GAUSSIAN

    @override
    def create(self, rng: at.KeyArrayLike) -> "TanhGaussian":
        return TanhGaussian(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
            )
        actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return obs, actions


class TanhGaussian(_model.BaseModel):
    """Tanh-squashed Gaussian policy for continuous control.

    This policy network outputs a diagonal Gaussian distribution that is
    squashed through tanh to produce bounded actions. The log probability
    computation accounts for the tanh transformation via the change of
    variables formula.

    The network architecture is a simple MLP with separate output heads
    for the mean and log standard deviation of the Gaussian.
    """

    def __init__(self, config: TanhGaussianConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.state_dim = config.state_dim
        self.hidden_dims = config.hidden_dims
        self.state_dependent_std = config.state_dependent_std
        self.std_parameterization = config.std_parameterization
        self.fixed_std = config.fixed_std
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        if config.action_low is not None:
            self._action_low = jnp.array(config.action_low)
        else:
            self._action_low = None

        if config.action_high is not None:
            self._action_high = jnp.array(config.action_high)
        else:
            self._action_high = None

        # Build MLP backbone
        layers = []
        in_dim = config.state_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.layers = layers

        # Output heads
        output_dim = config.action_horizon * config.action_dim
        self.mean_head = nnx.Linear(in_dim, output_dim, rngs=rngs)

        # Log std setup:
        #   - "fixed": no learnable parameters, constant std
        #   - state_dependent_std=True: Dense layer outputs log_std
        #   - state_dependent_std=False: learned parameter (matching IQL)
        if config.std_parameterization == "fixed":
            self.log_std_head = None
            self.log_std_param = None
        elif config.state_dependent_std:
            self.log_std_head = nnx.Linear(in_dim, output_dim, rngs=rngs)
            self.log_std_param = None
        else:
            self.log_std_head = None
            self.log_std_param = nnx.Param(jnp.zeros((output_dim,)))

        self.dropout_rate = config.dropout_rate
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def _forward(
        self,
        state: at.Float[at.Array, "b s"],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, "b ah ad"]]:
        """Forward pass returning (mean, log_std) before tanh.

        Args:
            state: State observations of shape [batch, state_dim].
            train: Whether to use dropout.

        Returns:
            Tuple of (mean, log_std), each of shape [batch, action_horizon, action_dim].
        """
        x = state
        for layer in self.layers:
            x = nnx.relu(layer(x))
            x = self.dropout(x, deterministic=not train)

        batch_size = state.shape[0]
        mean = self.mean_head(x).reshape(batch_size, self.action_horizon, self.action_dim)

        if self.std_parameterization == "fixed":
            # Constant std (state-independent, not learned)
            log_std = jnp.full_like(mean, jnp.log(self.fixed_std))
        elif self.state_dependent_std:
            # State-dependent log_std from Dense layer
            if self.std_parameterization == "exp":
                log_std = self.log_std_head(x).reshape(batch_size, self.action_horizon, self.action_dim)
                log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            elif self.std_parameterization == "softplus":
                raw = self.log_std_head(x).reshape(batch_size, self.action_horizon, self.action_dim)
                std = jax.nn.softplus(raw)
                log_std = jnp.log(std + 1e-8)
                log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            else:
                raise ValueError(f"Unknown std_parameterization: {self.std_parameterization}")
        else:
            # State-independent learned param
            log_std = self.log_std_param.value.reshape(self.action_horizon, self.action_dim)
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            log_std = jnp.broadcast_to(log_std, (batch_size, self.action_horizon, self.action_dim))

        return mean, log_std

    @override
    def action_distribution(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> distrax.Distribution:
        """Return the tanh-squashed Gaussian distribution over actions.

        Args:
            rng: Random key (unused, but required by interface).
            observation: Observation containing state.
            train: Whether to use dropout.

        Returns:
            A distrax.Transformed distribution with MultivariateNormalDiag base
            and Tanh (+ optional affine scaling) bijectors.
        """
        mean, log_std = self._forward(observation.state, train=train)
        std = jnp.exp(log_std)

        # Flatten action_horizon and action_dim for distribution
        batch_size = mean.shape[0]
        mean_flat = mean.reshape(batch_size, -1)
        std_flat = std.reshape(batch_size, -1)

        # Base distribution: diagonal Gaussian
        base_dist = distrax.MultivariateNormalDiag(loc=mean_flat, scale_diag=std_flat)

        # Build bijector chain: Tanh first, then optional affine scaling
        bijectors = []

        # Tanh squashes to [-1, 1]
        bijectors.append(distrax.Block(distrax.Tanh(), ndims=1))

        # Add affine scaling if action bounds specified
        if self._action_low is not None and self._action_high is not None:
            # Tanh outputs in [-1, 1], scale to [low, high]
            # y = (tanh(x) + 1) / 2 * (high - low) + low
            #   = tanh(x) * (high - low) / 2 + (high + low) / 2
            low_flat = jnp.tile(self._action_low, self.action_horizon)
            high_flat = jnp.tile(self._action_high, self.action_horizon)
            scale = (high_flat - low_flat) / 2.0
            shift = (high_flat + low_flat) / 2.0
            bijectors.append(distrax.Block(distrax.ScalarAffine(shift=shift, scale=scale), ndims=1))

        # Chain bijectors (applied in order: first Tanh, then ScalarAffine)
        bijector = distrax.Chain(bijectors[::-1]) if len(bijectors) > 1 else bijectors[0]

        return distrax.Transformed(base_dist, bijector)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        **kwargs,
    ) -> _model.Actions:
        """Sample actions from the policy.

        Args:
            rng: Random key for sampling.
            observation: Observation containing state.
            **kwargs: Additional arguments. Supports 'deterministic' (bool).

        Returns:
            Sampled actions of shape [batch, action_horizon, action_dim].
        """
        deterministic = kwargs.get("deterministic", False)
        dist = self.action_distribution(rng, observation)

        actions_flat = dist.bijector.forward(dist.distribution.loc) if deterministic else dist.sample(seed=rng)

        batch_size = observation.state.shape[0]
        return actions_flat.reshape(batch_size, self.action_horizon, self.action_dim)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute negative log likelihood loss (for BC training).

        Args:
            rng: Random key (unused for loss computation).
            observation: Observation containing state.
            actions: Ground truth actions of shape [batch, action_horizon, action_dim].
            train: Whether in training mode.

        Returns:
            Per-timestep NLL loss of shape [batch, action_horizon].
        """
        dist = self.action_distribution(rng, observation, train=train)

        batch_size = actions.shape[0]
        actions_flat = actions.reshape(batch_size, -1)

        # Compute log probability
        log_prob = dist.log_prob(actions_flat)

        # Return per-timestep loss (expand to match expected shape)
        # log_prob is scalar per sample, so we broadcast to action_horizon
        return jnp.broadcast_to(-log_prob[:, None], (batch_size, self.action_horizon))
