"""Ensemble network using vmap for efficient parallel computation.

Wraps a base network config to create an ensemble of networks with
vectorized parameters of shape (ensemble_size, ...).
"""

import dataclasses

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions.networks import base_networks as _base
from openpi.value_functions.networks import mlp as _mlp


@dataclasses.dataclass(frozen=True)
class EnsembleNetworkConfig:
    """Config for an ensemble of networks using vmap.

    Creates vectorized networks with parameters of shape (ensemble_size, ...).
    """

    base_config: _mlp.MLPNetworkConfig | _mlp.MultiMLPNetworkConfig
    ensemble_size: int = 2

    @property
    def action_conditioned(self) -> bool:
        return self.base_config.action_conditioned

    @property
    def state_dim(self) -> int:
        return self.base_config.state_dim

    @property
    def action_dim(self) -> int:
        return self.base_config.action_dim

    @property
    def action_horizon(self) -> int:
        return self.base_config.action_horizon

    def create(self, rng: at.KeyArrayLike) -> "EnsembleNetwork":
        rngs = nnx.Rngs(rng)

        @nnx.split_rngs(splits=self.ensemble_size)
        @nnx.vmap
        def create_member(rngs: nnx.Rngs) -> _base.BaseValueNetwork:
            return self.base_config.create(rngs.params())

        vectorized_network = create_member(rngs)
        return EnsembleNetwork(
            vectorized_network=vectorized_network,
            ensemble_size=self.ensemble_size,
        )


class EnsembleNetwork(_base.BaseValueNetwork):
    """Ensemble of networks with vectorized parameters.

    Uses nnx.vmap for efficient parallel computation over ensemble members.
    """

    vectorized_network: _base.BaseValueNetwork
    ensemble_size: int
    _action_conditioned: bool

    def __init__(
        self,
        vectorized_network: _base.BaseValueNetwork,
        ensemble_size: int,
    ):
        super().__init__()
        self.vectorized_network = vectorized_network
        self.ensemble_size = ensemble_size
        self._action_conditioned = vectorized_network.action_conditioned

    @property
    def action_conditioned(self) -> bool:
        return self._action_conditioned

    @property
    def num_transitions_per_sample(self) -> int | None:
        return getattr(self.vectorized_network, "num_transitions_per_sample", None)

    @property
    @override
    def feature_dim(self) -> int:
        return self.vectorized_network.feature_dim

    @override
    def compute_features(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "ensemble *b feature_dim"]:
        """Compute features for all ensemble members.

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).
            rng: Optional random key for stochastic operations.

        Returns:
            Features of shape [ensemble, batch, feature_dim] or
            [ensemble, batch, n, feature_dim] for multi-transition.
        """

        @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
        def compute_single(
            network: _base.BaseValueNetwork,
            obs: _model.Observation,
            act: _model.Actions | None,
            rng_key: at.KeyArrayLike | None,
        ) -> at.Array:
            return network.compute_features(obs, act, rng=rng_key)

        return compute_single(self.vectorized_network, observation, action, rng)

    def compute_features_single(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        member_idx: int = 0,
        rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "*b feature_dim"]:
        """Compute features for a single ensemble member.

        Useful for inference when you only need one member's output.

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).
            member_idx: Which ensemble member to use.
            rng: Optional random key for stochastic operations.

        Returns:
            Features of shape [batch, feature_dim].
        """

        @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
        def compute_single(
            network: _base.BaseValueNetwork,
            obs: _model.Observation,
            act: _model.Actions | None,
            rng_key: at.KeyArrayLike | None,
        ) -> at.Array:
            return network.compute_features(obs, act, rng=rng_key)

        all_features = compute_single(self.vectorized_network, observation, action, rng)
        return all_features[member_idx]


@dataclasses.dataclass(frozen=True)
class EnsembleMultiNetworkConfig:
    """Config for an ensemble of multi-transition networks using vmap."""

    base_config: _mlp.MultiMLPNetworkConfig
    ensemble_size: int = 2

    @property
    def action_conditioned(self) -> bool:
        return self.base_config.action_conditioned

    @property
    def state_dim(self) -> int:
        return self.base_config.state_dim

    @property
    def action_dim(self) -> int:
        return self.base_config.action_dim

    @property
    def action_horizon(self) -> int:
        return self.base_config.action_horizon

    @property
    def num_transitions_per_sample(self) -> int:
        return self.base_config.num_transitions_per_sample

    def create(self, rng: at.KeyArrayLike) -> EnsembleNetwork:
        rngs = nnx.Rngs(rng)

        @nnx.split_rngs(splits=self.ensemble_size)
        @nnx.vmap
        def create_member(rngs: nnx.Rngs) -> _base.BaseValueNetwork:
            return self.base_config.create(rngs.params())

        vectorized_network = create_member(rngs)
        return EnsembleNetwork(
            vectorized_network=vectorized_network,
            ensemble_size=self.ensemble_size,
        )
