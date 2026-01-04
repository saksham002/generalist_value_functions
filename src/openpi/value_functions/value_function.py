"""Unified value function with config hierarchy.

Provides composable value functions using network + head + objective pattern.
Each objective-specific config inherits from the base and adds only the
parameters it needs.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.policy_extraction.temperature import Temperature
from openpi.shared import array_typing as at
from openpi.value_functions import multi_objectives as _multi_objectives
from openpi.value_functions import objectives as _objectives
from openpi.value_functions.base import BaseValueFunction
from openpi.value_functions.base import BaseValueFunctionConfig
from openpi.value_functions.base import MultiTransition
from openpi.value_functions.base import Transition
from openpi.value_functions.heads import HeadConfig
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base import BaseValueNetwork
from openpi.value_functions.networks.mlp import MLPNetworkConfig
from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig


@dataclasses.dataclass(frozen=True)
class ValueFunctionConfig(BaseValueFunctionConfig):
    """Base config for value functions.

    Composes a network config with a head config. Subclasses define
    the objective and any objective-specific parameters.
    """

    network_config: MLPNetworkConfig
    head_config: HeadConfig

    @override
    def create(self, rng: at.KeyArrayLike) -> "ValueFunction":
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
    def create(self, rng: at.KeyArrayLike) -> "MCValueFunction":
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
    def create(self, rng: at.KeyArrayLike) -> "SARSAValueFunction":
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
class IQLValueFunctionConfig(ValueFunctionConfig):
    """IQL value function config with expectile regression."""

    expectile: float = 0.7

    @override
    def create(self, rng: at.KeyArrayLike) -> "IQLValueFunction":
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return IQLValueFunction(network=network, head=head, expectile=self.expectile)


@dataclasses.dataclass(frozen=True)
class SACValueFunctionConfig(ValueFunctionConfig):
    """SAC Q-function config.

    Policy and temperature are NOT owned by the value function.
    They should be passed to compute_loss when needed.
    """

    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> "SACValueFunction":
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
    ) -> at.Float[at.Array, "*b"]:
        features = self.network.compute_features(observation, action)
        return self.head(features)


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
        target_net_state = nnx.state(self.target_network)
        online_net_state = nnx.state(self.network)
        new_target_net_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_net_state,
            online_net_state,
        )
        nnx.update(self.target_network, new_target_net_state)

        target_head_state = nnx.state(self.target_head)
        online_head_state = nnx.state(self.head)
        new_target_head_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_head_state,
            online_head_state,
        )
        nnx.update(self.target_head, new_target_head_state)


class IQLValueFunction(ValueFunction):
    """IQL value function with expectile regression."""

    expectile: float

    def __init__(self, network: BaseValueNetwork, head: ValueHead, expectile: float):
        super().__init__(network, head)
        self.expectile = expectile

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del train, rng
        return _objectives.iql_objective(self.network, self.head, transition, expectile=self.expectile)


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
        target_net_state = nnx.state(self.target_network)
        online_net_state = nnx.state(self.network)
        new_target_net_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_net_state,
            online_net_state,
        )
        nnx.update(self.target_network, new_target_net_state)

        target_head_state = nnx.state(self.target_head)
        online_head_state = nnx.state(self.head)
        new_target_head_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_head_state,
            online_head_state,
        )
        nnx.update(self.target_head, new_target_head_state)


# =============================================================================
# Multi-Transition Value Functions
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiValueFunctionConfig(BaseValueFunctionConfig):
    """Base config for multi-transition value functions.

    Uses MultiMLPNetworkConfig that processes n (state, action) pairs jointly.
    """

    network_config: "MultiMLPNetworkConfig"
    head_config: HeadConfig

    @override
    def create(self, rng: at.KeyArrayLike) -> "MultiValueFunction":
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
    def create(self, rng: at.KeyArrayLike) -> "MultiMCValueFunction":
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
    def create(self, rng: at.KeyArrayLike) -> "MultiSARSAValueFunction":
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
class MultiIQLValueFunctionConfig(MultiValueFunctionConfig):
    """Multi-transition IQL value function config with expectile regression."""

    expectile: float = 0.7

    @override
    def create(self, rng: at.KeyArrayLike) -> "MultiIQLValueFunction":
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return MultiIQLValueFunction(network=network, head=head, expectile=self.expectile)


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
    ) -> at.Float[at.Array, "*b n"]:
        """Compute values for all n transitions in the batch."""
        features = self.network.compute_features(observation, action)
        return self.head(features)


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
        return _multi_objectives.mc_multi_objective(self.network, self.head, transition)


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
        return _multi_objectives.sarsa_multi_objective(
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
        target_net_state = nnx.state(self.target_network)
        online_net_state = nnx.state(self.network)
        new_target_net_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_net_state,
            online_net_state,
        )
        nnx.update(self.target_network, new_target_net_state)

        target_head_state = nnx.state(self.target_head)
        online_head_state = nnx.state(self.head)
        new_target_head_state = jax.tree.map(
            lambda t, o: self.tau * o + (1.0 - self.tau) * t,
            target_head_state,
            online_head_state,
        )
        nnx.update(self.target_head, new_target_head_state)


class MultiIQLValueFunction(MultiValueFunction):
    """Multi-transition IQL value function with expectile regression."""

    expectile: float

    def __init__(self, network: BaseValueNetwork, head: ValueHead, expectile: float):
        super().__init__(network, head)
        self.expectile = expectile

    @override
    def compute_loss(
        self,
        transition: MultiTransition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        del train, rng
        return _multi_objectives.iql_multi_objective(self.network, self.head, transition, expectile=self.expectile)
