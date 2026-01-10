"""Unified value function with config hierarchy.

Provides composable value functions using network + head + objective pattern.
Each objective-specific config inherits from the base and adds only the
parameters it needs.
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.policy_extraction.temperature import Temperature
from openpi.shared import array_typing as at
from openpi.value_functions import value_function_objectives as _objectives
from openpi.value_functions.base import BaseValueFunction
from openpi.value_functions.base import BaseValueFunctionConfig
from openpi.value_functions.base import MultiTransition
from openpi.value_functions.base import Transition
from openpi.value_functions.heads import EnsembleHeadConfig
from openpi.value_functions.heads import HeadConfig
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base import BaseValueNetwork
from openpi.value_functions.networks.ensemble import EnsembleMultiNetworkConfig
from openpi.value_functions.networks.ensemble import EnsembleNetworkConfig
from openpi.value_functions.networks.mlp import MLPNetworkConfig
from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig

# =============================================================================
# Single-Transition Value Functions
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ValueFunctionConfig(BaseValueFunctionConfig):
    """Base config for value functions.

    Composes a network config with a head config. Subclasses define
    the objective and any objective-specific parameters.
    """

    network_config: MLPNetworkConfig
    head_config: HeadConfig

    @override
    def create(self, rng: at.KeyArrayLike) -> ValueFunction:
        raise NotImplementedError("Use a specific config subclass (MCValueFunctionConfig, etc.)")

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        nc = self.network_config
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, nc.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size], jnp.float32)
        if nc.action_conditioned:
            actions = jax.ShapeDtypeStruct([batch_size, nc.action_horizon, nc.action_dim], jnp.float32)
            return obs, actions, target
        return obs, target


@dataclasses.dataclass(frozen=True)
class MCValueFunctionConfig(ValueFunctionConfig):
    """Monte-Carlo value function config.

    Uses mc_return from transitions as target.
    """

    @override
    def create(self, rng: at.KeyArrayLike) -> MCValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return MCValueFunction(network=network, head=head)


@dataclasses.dataclass(frozen=True)
class SARSAValueFunctionConfig(ValueFunctionConfig):
    """SARSA value function config with target network."""

    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> SARSAValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng, target_net_rng, target_head_rng = jax.random.split(rng, 4)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_network = self.network_config.create(target_net_rng)
        target_head = self.head_config.create(target_network.feature_dim, target_head_rng)

        return SARSAValueFunction(
            network=network,
            head=head,
            target_network=target_network,
            target_head=target_head,
            discount=self.discount,
            tau=self.tau,
        )


@dataclasses.dataclass(frozen=True)
class IQLValueFunctionConfig(BaseValueFunctionConfig):
    """IQL config with separate Q and V networks.

    Q(s, a): Action-conditioned, trained with Bellman backup using V(s')
    V(s): State-only, trained with expectile loss to min(target_Q(s, a))

    Supports optional Q ensemble via EnsembleNetworkConfig and EnsembleHeadConfig.
    When using ensemble, V target is computed as min over ensemble members.
    """

    q_network_config: MLPNetworkConfig | EnsembleNetworkConfig  # action_conditioned=True
    v_network_config: MLPNetworkConfig  # action_conditioned=False
    q_head_config: HeadConfig | EnsembleHeadConfig  # Can be ensemble for min over Q
    v_head_config: HeadConfig  # V is never ensemble

    expectile: float = 0.7
    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> IQLValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        keys = jax.random.split(rng, 8)

        q_network = self.q_network_config.create(keys[0])
        q_head = self.q_head_config.create(q_network.feature_dim, keys[1])
        target_q_network = self.q_network_config.create(keys[2])
        target_q_head = self.q_head_config.create(target_q_network.feature_dim, keys[3])

        v_network = self.v_network_config.create(keys[4])
        v_head = self.v_head_config.create(v_network.feature_dim, keys[5])
        target_v_network = self.v_network_config.create(keys[6])
        target_v_head = self.v_head_config.create(target_v_network.feature_dim, keys[7])

        return IQLValueFunction(
            q_network=q_network,
            q_head=q_head,
            target_q_network=target_q_network,
            target_q_head=target_q_head,
            v_network=v_network,
            v_head=v_head,
            target_v_network=target_v_network,
            target_v_head=target_v_head,
            expectile=self.expectile,
            discount=self.discount,
            tau=self.tau,
        )

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]:
        qc = self.q_network_config
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, qc.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size], jnp.float32)
        actions = jax.ShapeDtypeStruct([batch_size, qc.action_horizon, qc.action_dim], jnp.float32)
        return obs, actions, target


@dataclasses.dataclass(frozen=True)
class SACValueFunctionConfig(ValueFunctionConfig):
    """SAC Q-function config.

    Policy and temperature are NOT owned by the value function.
    They should be passed to compute_loss when needed.
    """

    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> SACValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng, target_net_rng, target_head_rng = jax.random.split(rng, 4)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_network = self.network_config.create(target_net_rng)
        target_head = self.head_config.create(target_network.feature_dim, target_head_rng)

        return SACValueFunction(
            network=network,
            head=head,
            target_network=target_network,
            target_head=target_head,
            discount=self.discount,
            tau=self.tau,
        )


class ValueFunction(BaseValueFunction):
    """Base value function class with network + head composition."""

    network: BaseValueNetwork
    head: ValueHead

    def __init__(self, network: BaseValueNetwork, head: ValueHead):
        super().__init__()
        self.network = network
        self.head = head

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        features = self.network.compute_features(observation, action)
        val = self.head(features)
        if take_min_over_ensemble and val.ndim > 1:
            val = jnp.min(val, axis=0)
        return val


class MCValueFunction(ValueFunction):
    """Monte-Carlo value function."""

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        return _objectives.mc_objective(self.network, self.head, transition)


class SARSAValueFunction(ValueFunction):
    """SARSA value function with target network."""

    target_network: BaseValueNetwork
    target_head: ValueHead
    discount: float
    tau: float

    def __init__(
        self,
        network: BaseValueNetwork,
        head: ValueHead,
        target_network: BaseValueNetwork,
        target_head: ValueHead,
        discount: float,
        tau: float,
    ):
        super().__init__(network, head)
        self.target_network = target_network
        self.target_head = target_head
        self.discount = discount
        self.tau = tau

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del train, rng
        return _objectives.sarsa_objective(
            self.network,
            self.head,
            transition,
            self.target_network,
            self.target_head,
            discount=self.discount,
        )

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_network, self.network, self.tau)
        _polyak_update(self.target_head, self.head, self.tau)


class IQLValueFunction(BaseValueFunction):
    """IQL with separate Q and V networks.

    Components:
    - q_network + q_head: Q(s, a) trained with Bellman backup using V(s')
    - target_q_network + target_q_head: Target Q for V loss (min over ensemble for pessimism)
    - v_network + v_head: V(s) trained with expectile loss to min(target_Q(s, a))
    - target_v_network + target_v_head: Target V for stable Q targets

    Supports Q ensemble via EnsembleNetwork + EnsembleHead.
    When Q is ensemble, V target uses min(target_Q) over ensemble members.
    """

    q_network: BaseValueNetwork
    q_head: ValueHead
    target_q_network: BaseValueNetwork
    target_q_head: ValueHead
    v_network: BaseValueNetwork
    v_head: ValueHead
    target_v_network: BaseValueNetwork
    target_v_head: ValueHead
    expectile: float
    discount: float
    tau: float

    def __init__(
        self,
        q_network: BaseValueNetwork,
        q_head: ValueHead,
        target_q_network: BaseValueNetwork,
        target_q_head: ValueHead,
        v_network: BaseValueNetwork,
        v_head: ValueHead,
        target_v_network: BaseValueNetwork,
        target_v_head: ValueHead,
        expectile: float,
        discount: float,
        tau: float,
    ):
        super().__init__()
        self.q_network = q_network
        self.q_head = q_head
        self.target_q_network = target_q_network
        self.target_q_head = target_q_head
        self.v_network = v_network
        self.v_head = v_head
        self.target_v_network = target_v_network
        self.target_v_head = target_v_head
        self.expectile = expectile
        self.discount = discount
        self.tau = tau

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute Q(s, a) if action provided, else V(s)."""
        if action is not None:
            features = self.q_network.compute_features(observation, action)
            val = self.q_head(features)
            if take_min_over_ensemble and val.ndim > 1:
                val = jnp.min(val, axis=0)
            return val
        features = self.v_network.compute_features(observation, None)
        return self.v_head(features)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute combined Q + V loss.

        Returns the sum of Q and V losses for gradient computation.
        """
        del train, rng
        q_loss, v_loss, info = _objectives.iql_objective(
            self.q_network,
            self.q_head,
            self.target_q_network,
            self.target_q_head,
            self.v_network,
            self.v_head,
            self.target_v_network,
            self.target_v_head,
            transition,
            expectile=self.expectile,
            discount=self.discount,
        )
        return q_loss + v_loss, info

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target Q and V networks."""
        _polyak_update(self.target_q_network, self.q_network, self.tau)
        _polyak_update(self.target_q_head, self.q_head, self.tau)
        _polyak_update(self.target_v_network, self.v_network, self.tau)
        _polyak_update(self.target_v_head, self.v_head, self.tau)


class SACValueFunction(ValueFunction):
    """SAC Q-function with target network.

    Policy and temperature are NOT owned by this class.
    Pass them to compute_sac_loss() instead.
    """

    target_network: BaseValueNetwork
    target_head: ValueHead
    discount: float
    tau: float

    def __init__(
        self,
        network: BaseValueNetwork,
        head: ValueHead,
        target_network: BaseValueNetwork,
        target_head: ValueHead,
        discount: float,
        tau: float,
    ):
        super().__init__(network, head)
        self.target_network = target_network
        self.target_head = target_head
        self.discount = discount
        self.tau = tau

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        raise NotImplementedError("SACValueFunction requires policy and temperature. Use compute_sac_loss() instead.")

    def compute_sac_loss(
        self,
        transition: Transition,
        policy: _model.BaseModel,
        temperature: Temperature,
        *,
        rng: at.KeyArrayLike,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute SAC loss with external policy and temperature."""
        return _objectives.sac_objective(
            self.network,
            self.head,
            transition,
            self.target_network,
            self.target_head,
            policy,
            temperature,
            discount=self.discount,
            rng=rng,
        )

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_network, self.network, self.tau)
        _polyak_update(self.target_head, self.head, self.tau)


# =============================================================================
# Multi-Transition Value Functions
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiValueFunctionConfig(BaseValueFunctionConfig):
    """Base config for multi-transition value functions.

    Uses MultiMLPNetworkConfig that processes n (state, action) pairs jointly.
    """

    network_config: MultiMLPNetworkConfig
    head_config: HeadConfig

    @override
    def create(self, rng: at.KeyArrayLike) -> MultiValueFunction:
        raise NotImplementedError("Use a specific config subclass (MultiMCValueFunctionConfig, etc.)")

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b n"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b n"]]
    ):
        nc = self.network_config
        n = nc.num_transitions_per_sample
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, n, nc.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size, n], jnp.float32)
        if nc.action_conditioned:
            actions = jax.ShapeDtypeStruct([batch_size, n, nc.action_horizon, nc.action_dim], jnp.float32)
            return obs, actions, target
        return obs, target


@dataclasses.dataclass(frozen=True)
class MultiMCValueFunctionConfig(MultiValueFunctionConfig):
    """Multi-transition Monte-Carlo value function config."""

    @override
    def create(self, rng: at.KeyArrayLike) -> MultiMCValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return MultiMCValueFunction(network=network, head=head)


@dataclasses.dataclass(frozen=True)
class MultiSARSAValueFunctionConfig(MultiValueFunctionConfig):
    """Multi-transition SARSA value function config with target network."""

    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> MultiSARSAValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng, target_net_rng, target_head_rng = jax.random.split(rng, 4)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_network = self.network_config.create(target_net_rng)
        target_head = self.head_config.create(target_network.feature_dim, target_head_rng)

        return MultiSARSAValueFunction(
            network=network,
            head=head,
            target_network=target_network,
            target_head=target_head,
            discount=self.discount,
            tau=self.tau,
        )


@dataclasses.dataclass(frozen=True)
class MultiIQLValueFunctionConfig(BaseValueFunctionConfig):
    """Multi-transition IQL config with separate Q and V networks.

    Uses joint multi-transition networks for cross-state information sharing.
    Q(s, a): Action-conditioned, trained with Bellman backup using V(s')
    V(s): State-only, trained with expectile loss to min(target_Q(s, a))

    Supports Q ensemble via EnsembleMultiNetworkConfig and EnsembleHeadConfig.
    """

    q_network_config: MultiMLPNetworkConfig | EnsembleMultiNetworkConfig
    v_network_config: MultiMLPNetworkConfig
    q_head_config: HeadConfig | EnsembleHeadConfig
    v_head_config: HeadConfig

    expectile: float = 0.7
    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> MultiIQLValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        keys = jax.random.split(rng, 8)

        q_network = self.q_network_config.create(keys[0])
        q_head = self.q_head_config.create(q_network.feature_dim, keys[1])
        target_q_network = self.q_network_config.create(keys[2])
        target_q_head = self.q_head_config.create(target_q_network.feature_dim, keys[3])

        v_network = self.v_network_config.create(keys[4])
        v_head = self.v_head_config.create(v_network.feature_dim, keys[5])
        target_v_network = self.v_network_config.create(keys[6])
        target_v_head = self.v_head_config.create(target_v_network.feature_dim, keys[7])

        return MultiIQLValueFunction(
            q_network=q_network,
            q_head=q_head,
            target_q_network=target_q_network,
            target_q_head=target_q_head,
            v_network=v_network,
            v_head=v_head,
            target_v_network=target_v_network,
            target_v_head=target_v_head,
            expectile=self.expectile,
            discount=self.discount,
            tau=self.tau,
        )

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b n"]]:
        qc = self.q_network_config
        n = qc.num_transitions_per_sample
        with at.disable_typechecking():
            obs = _model.Observation(
                images={},
                image_masks={},
                state=jax.ShapeDtypeStruct([batch_size, n, qc.state_dim], jnp.float32),
            )
        target = jax.ShapeDtypeStruct([batch_size, n], jnp.float32)
        actions = jax.ShapeDtypeStruct([batch_size, n, qc.action_horizon, qc.action_dim], jnp.float32)
        return obs, actions, target


class MultiValueFunction(BaseValueFunction):
    """Base multi-transition value function class."""

    network: BaseValueNetwork
    head: ValueHead

    def __init__(self, network: BaseValueNetwork, head: ValueHead):
        super().__init__()
        self.network = network
        self.head = head

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b n"]:
        """Compute values for all n transitions in the batch."""
        features = self.network.compute_features(observation, action)
        val = self.head(features)
        if take_min_over_ensemble and val.ndim > 2:
            # Expected shape [batch, n], ensemble shape [ensemble, batch, n]
            val = jnp.min(val, axis=0)
        return val


class MultiMCValueFunction(MultiValueFunction):
    """Multi-transition Monte-Carlo value function."""

    @override
    def compute_loss(
        self,
        transition: MultiTransition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        return _objectives.mc_objective(self.network, self.head, transition)


class MultiSARSAValueFunction(MultiValueFunction):
    """Multi-transition SARSA value function with target network."""

    target_network: BaseValueNetwork
    target_head: ValueHead
    discount: float
    tau: float

    def __init__(
        self,
        network: BaseValueNetwork,
        head: ValueHead,
        target_network: BaseValueNetwork,
        target_head: ValueHead,
        discount: float,
        tau: float,
    ):
        super().__init__(network, head)
        self.target_network = target_network
        self.target_head = target_head
        self.discount = discount
        self.tau = tau

    @override
    def compute_loss(
        self,
        transition: MultiTransition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        del train, rng
        return _objectives.sarsa_objective(
            self.network,
            self.head,
            transition,
            self.target_network,
            self.target_head,
            discount=self.discount,
        )

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_network, self.network, self.tau)
        _polyak_update(self.target_head, self.head, self.tau)


class MultiIQLValueFunction(BaseValueFunction):
    """Multi-transition IQL with separate Q and V networks.

    Uses joint multi-transition networks for cross-state information sharing.
    Supports Q ensemble via EnsembleNetwork + EnsembleHead.
    """

    q_network: BaseValueNetwork
    q_head: ValueHead
    target_q_network: BaseValueNetwork
    target_q_head: ValueHead
    v_network: BaseValueNetwork
    v_head: ValueHead
    target_v_network: BaseValueNetwork
    target_v_head: ValueHead
    expectile: float
    discount: float
    tau: float

    def __init__(
        self,
        q_network: BaseValueNetwork,
        q_head: ValueHead,
        target_q_network: BaseValueNetwork,
        target_q_head: ValueHead,
        v_network: BaseValueNetwork,
        v_head: ValueHead,
        target_v_network: BaseValueNetwork,
        target_v_head: ValueHead,
        expectile: float,
        discount: float,
        tau: float,
    ):
        super().__init__()
        self.q_network = q_network
        self.q_head = q_head
        self.target_q_network = target_q_network
        self.target_q_head = target_q_head
        self.v_network = v_network
        self.v_head = v_head
        self.target_v_network = target_v_network
        self.target_v_head = target_v_head
        self.expectile = expectile
        self.discount = discount
        self.tau = tau

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b n"]:
        """Compute Q(s, a) if action provided, else V(s)."""
        if action is not None:
            features = self.q_network.compute_features(observation, action)
            val = self.q_head(features)
            if take_min_over_ensemble and val.ndim > 2:
                # Check for ensemble dimension [ensemble, batch, n]
                val = jnp.min(val, axis=0)
            return val
        features = self.v_network.compute_features(observation, None)
        return self.v_head(features)

    @override
    def compute_loss(
        self,
        transition: MultiTransition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        """Compute combined Q + V loss."""
        del train, rng
        q_loss, v_loss, info = _objectives.iql_objective(
            self.q_network,
            self.q_head,
            self.target_q_network,
            self.target_q_head,
            self.v_network,
            self.v_head,
            self.target_v_network,
            self.target_v_head,
            transition,
            expectile=self.expectile,
            discount=self.discount,
        )
        return q_loss + v_loss, info

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target Q and V networks."""
        _polyak_update(self.target_q_network, self.q_network, self.tau)
        _polyak_update(self.target_q_head, self.q_head, self.tau)
        _polyak_update(self.target_v_network, self.v_network, self.tau)
        _polyak_update(self.target_v_head, self.v_head, self.tau)


# =============================================================================
# Utilities
# =============================================================================


def _polyak_update(target_module: nnx.Module, online_module: nnx.Module, tau: float) -> None:
    """Polyak averaging: target = tau * online + (1 - tau) * target."""
    target_state = nnx.state(target_module)
    online_state = nnx.state(online_module)
    new_target_state = jax.tree.map(
        lambda t, o: tau * o + (1.0 - tau) * t,
        target_state,
        online_state,
    )
    nnx.update(target_module, new_target_state)
