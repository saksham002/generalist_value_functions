"""Unified value function with config hierarchy.

Provides composable value functions using network + head + objective pattern.
Each objective-specific config inherits from the base and adds only the
parameters it needs.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import best_of_n as _best_of_n
from openpi.models import model as _model
from openpi.policy_extraction.temperature import Temperature
from openpi.shared import array_typing as at
from openpi.shared.action_bounds import ActionBounds
from openpi.value_functions import value_function_objectives as _objectives
from openpi.value_functions.base_value_functions import BaseMultiValueFunction
from openpi.value_functions.base_value_functions import BaseMultiValueFunctionConfig
from openpi.value_functions.base_value_functions import BaseValueFunction
from openpi.value_functions.base_value_functions import BaseValueFunctionConfig
from openpi.value_functions.base_value_functions import MultiTransition
from openpi.value_functions.base_value_functions import Transition
from openpi.value_functions.heads import EnsembleHeadConfig
from openpi.value_functions.heads import HeadConfig
from openpi.value_functions.heads import ValueHead
from openpi.value_functions.networks.base_networks import BaseValueNetwork
from openpi.value_functions.networks.ensemble import EnsembleMultiNetworkConfig
from openpi.value_functions.networks.ensemble import EnsembleNetworkConfig
from openpi.value_functions.networks.mlp import MLPNetworkConfig
from openpi.value_functions.networks.mlp import MultiMLPNetworkConfig
from openpi.value_functions.networks.paligemma import PaliGemmaNetworkConfig

# =============================================================================
# Single-Transition Value Functions
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ValueFunctionConfig(BaseValueFunctionConfig):
    """Base config for value functions.

    Composes a network config with a head config. Subclasses define
    the objective and any objective-specific parameters.
    """

    network_config: MLPNetworkConfig | PaliGemmaNetworkConfig
    head_config: HeadConfig

    # Number of actions in the action chunk. None means V(s), not Q(s,a).
    action_horizon: int | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> ValueFunction:
        raise NotImplementedError("Use a specific config subclass (MCValueFunctionConfig, etc.)")

    @property
    def weight_dtype(self) -> str:
        """Dtype for model weights, derived from the network config."""
        if isinstance(self.network_config, PaliGemmaNetworkConfig):
            return self.network_config.dtype
        return "float32"

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
        if self.action_horizon is not None:
            actions = jax.ShapeDtypeStruct([batch_size, self.action_horizon, nc.action_dim], jnp.float32)
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

        # PaliGemmaNetworkConfig receives action_horizon at creation time instead of
        # storing it as a config field. MLPNetworkConfig keeps it in its own config.
        if isinstance(self.network_config, PaliGemmaNetworkConfig):
            network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return MCValueFunction(network=network, head=head)


@dataclasses.dataclass(frozen=True)
class SARSAValueFunctionConfig(ValueFunctionConfig):
    """SARSA value function config with target network."""

    discount: float = 0.99
    tau: float = 0.005
    next_token_loss_weight: float = 0.0

    @override
    def create(self, rng: at.KeyArrayLike) -> SARSAValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng, 2)

        if isinstance(self.network_config, PaliGemmaNetworkConfig):
            network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
            target_network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            network = self.network_config.create(net_rng)
            target_network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_head = self.head_config.create(target_network.feature_dim, head_rng)

        return SARSAValueFunction(
            network=network,
            head=head,
            target_network=target_network,
            target_head=target_head,
            discount=self.discount,
            tau=self.tau,
            next_token_loss_weight=self.next_token_loss_weight,
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
        net_rng, head_rng = jax.random.split(rng, 2)

        if isinstance(self.network_config, PaliGemmaNetworkConfig):
            network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
            target_network = self.network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            network = self.network_config.create(net_rng)
            target_network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_head = self.head_config.create(target_network.feature_dim, head_rng)

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
    ) -> at.Float[at.Array, "*b"] | tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b _n"]]:
        out = self.network.compute_features(observation, action)
        if isinstance(out, tuple):
            features, attn_scores = out[0], out[1]
            val = self.head(features)
            if take_min_over_ensemble and val.ndim > 1:
                val = jnp.min(val, axis=0)
            return val, attn_scores
        val = self.head(out)
        if take_min_over_ensemble and val.ndim > 1:
            val = jnp.min(val, axis=0)
        return val

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """For value functions without target network, return the same as compute_value."""
        result = self.compute_value(observation, action, take_min_over_ensemble = take_min_over_ensemble)
        if isinstance(result, tuple):
            return result[0]
        return result


class MCValueFunction(ValueFunction):
    """Monte-Carlo value function."""

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del policy
        return _objectives.mc_objective(self.network, self.head, transition, rng=rng)


class SARSAValueFunction(ValueFunction):
    """SARSA value function with target network."""

    target_network: BaseValueNetwork
    target_head: ValueHead
    discount: float
    tau: float
    next_token_loss_weight: float

    def __init__(
        self,
        network: BaseValueNetwork,
        head: ValueHead,
        target_network: BaseValueNetwork,
        target_head: ValueHead,
        discount: float,
        tau: float,
        next_token_loss_weight: float,
    ):
        super().__init__(network, head)
        self.target_network = target_network
        self.target_head = target_head
        self.discount = discount
        self.tau = tau
        self.next_token_loss_weight = next_token_loss_weight

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute target value using target network."""
        target_out = self.target_network.compute_features(observation, action)
        target_features = target_out[0] if isinstance(target_out, tuple) else target_out
        val = self.target_head(target_features)
        if take_min_over_ensemble and val.ndim > 1:
            val = jnp.min(val, axis=0)
        return val

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        del train, policy
        return _objectives.sarsa_objective(
            self.network,
            self.head,
            transition,
            self.target_network,
            self.target_head,
            discount=self.discount,
            next_token_loss_weight=self.next_token_loss_weight,
            rng=rng,
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
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute target Q(s, a) if action provided, else target V(s)."""
        if action is not None:
            features = self.target_q_network.compute_features(observation, action)
            val = self.target_q_head(features)
            if take_min_over_ensemble and val.ndim > 1:
                val = jnp.min(val, axis=0)
            return val
        features = self.target_v_network.compute_features(observation, None)
        return self.target_v_head(features)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute combined Q + V loss.

        Returns the sum of Q and V losses for gradient computation.
        """
        del train, rng, policy
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
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute target value using target network."""
        features = self.target_network.compute_features(observation, action)
        val = self.target_head(features)
        if take_min_over_ensemble and val.ndim > 1:
            val = jnp.min(val, axis=0)
        return val

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
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


@dataclasses.dataclass(frozen=True)
class CQLValueFunctionConfig(BaseValueFunctionConfig):
    """CQL config with Q-network and target Q-network.

    Supports any network config implementing the create/feature_dim protocol.
    """

    q_network_config: MLPNetworkConfig | PaliGemmaNetworkConfig | EnsembleNetworkConfig
    q_head_config: HeadConfig | EnsembleHeadConfig

    # Number of actions in the action chunk. None means V(s), not Q(s,a).
    action_horizon: int | None = None

    discount: float = 0.99
    tau: float = 0.005

    action_bounds: ActionBounds = dataclasses.field(
        default_factory = lambda: ActionBounds.from_uniform(-1.0, 1.0, action_dim = 1, is_normalized = True)
    )
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    cql_n_actions: int = 4
    cql_action_sample_method: Literal["uniform", "normal"] = "uniform"
    cql_importance_sample: bool = True
    only_use_next_actions_for_cql: bool = False
    cql_max_target_backup: bool = False
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf
    use_calql: bool = False
    use_calql_on_random_actions: bool = True

    @property
    def weight_dtype(self) -> str:
        """Dtype for model weights, derived from the network config."""
        if isinstance(self.q_network_config, PaliGemmaNetworkConfig):
            return self.q_network_config.dtype
        return "float32"

    @override
    def create(self, rng: at.KeyArrayLike) -> CQLValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng, 2)

        if isinstance(self.q_network_config, PaliGemmaNetworkConfig):
            q_network = self.q_network_config.create(net_rng, action_horizon = self.action_horizon)
            target_q_network = self.q_network_config.create(net_rng, action_horizon = self.action_horizon)
        else:
            q_network = self.q_network_config.create(net_rng)
            target_q_network = self.q_network_config.create(net_rng)
        q_head = self.q_head_config.create(q_network.feature_dim, head_rng)
        target_q_head = self.q_head_config.create(target_q_network.feature_dim, head_rng)

        return CQLValueFunction(
            q_network=q_network,
            q_head=q_head,
            target_q_network=target_q_network,
            target_q_head=target_q_head,
            discount=self.discount,
            tau=self.tau,
            action_bounds=self.action_bounds,
            cql_alpha=self.cql_alpha,
            cql_temp=self.cql_temp,
            cql_n_actions=self.cql_n_actions,
            cql_action_sample_method=self.cql_action_sample_method,
            cql_importance_sample=self.cql_importance_sample,
            only_use_next_actions_for_cql=self.only_use_next_actions_for_cql,
            cql_max_target_backup=self.cql_max_target_backup,
            cql_clip_diff_min=self.cql_clip_diff_min,
            cql_clip_diff_max=self.cql_clip_diff_max,
            use_calql=self.use_calql,
            use_calql_on_random_actions=self.use_calql_on_random_actions,
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
        action_horizon = self.action_horizon if self.action_horizon is not None else 1
        actions = jax.ShapeDtypeStruct([batch_size, action_horizon, qc.action_dim], jnp.float32)
        return obs, actions, target


class CQLValueFunction(BaseValueFunction):
    """CQL Q-function with target network and conservative penalty.

    Policy is not owned by this class. Pass policy to compute_loss().
    """

    q_network: BaseValueNetwork
    q_head: ValueHead
    target_q_network: BaseValueNetwork
    target_q_head: ValueHead
    discount: float
    tau: float

    action_bounds: ActionBounds
    cql_alpha: float
    cql_temp: float
    cql_n_actions: int
    cql_action_sample_method: Literal["uniform", "normal"]
    cql_importance_sample: bool
    only_use_next_actions_for_cql: bool
    cql_max_target_backup: bool
    cql_clip_diff_min: float
    cql_clip_diff_max: float
    use_calql: bool
    use_calql_on_random_actions: bool

    def __init__(
        self,
        q_network: BaseValueNetwork,
        q_head: ValueHead,
        target_q_network: BaseValueNetwork,
        target_q_head: ValueHead,
        discount: float,
        tau: float,
        *,
        action_bounds: ActionBounds,
        cql_alpha: float,
        cql_temp: float,
        cql_n_actions: int,
        cql_action_sample_method: Literal["uniform", "normal"],
        cql_importance_sample: bool,
        only_use_next_actions_for_cql: bool,
        cql_max_target_backup: bool,
        cql_clip_diff_min: float,
        cql_clip_diff_max: float,
        use_calql: bool,
        use_calql_on_random_actions: bool,
    ):
        super().__init__()
        self.q_network = q_network
        self.q_head = q_head
        self.target_q_network = target_q_network
        self.target_q_head = target_q_head
        self.discount = discount
        self.tau = tau
        self.action_bounds = action_bounds
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.cql_n_actions = cql_n_actions
        self.cql_action_sample_method = cql_action_sample_method
        self.cql_importance_sample = cql_importance_sample
        self.only_use_next_actions_for_cql = only_use_next_actions_for_cql
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.use_calql = use_calql
        self.use_calql_on_random_actions = use_calql_on_random_actions

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"] | tuple[at.Float[at.Array, "*b"], at.Float[at.Array, "*b _n"]]:
        feature_kwargs = {"prefix_cache": prefix_cache} if prefix_cache is not None else {}
        out = self.q_network.compute_features(observation, action, **feature_kwargs)
        if isinstance(out, tuple):
            features, attn_scores = out[0], out[1]
            val = self.q_head(features)
            if take_min_over_ensemble and val.ndim > 1:
                val = jnp.min(val, axis = 0)
            return val, attn_scores
        val = self.q_head(out)
        if take_min_over_ensemble and val.ndim > 1:
            val = jnp.min(val, axis = 0)
        return val

    @override
    def compute_target_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
        *,
        take_min_over_ensemble: bool = False,
        prefix_cache: tuple[at.Array, at.Array] | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute target Q-value using target network."""
        feature_kwargs = {"prefix_cache": prefix_cache} if prefix_cache is not None else {}
        target_out = self.target_q_network.compute_features(observation, action, **feature_kwargs)
        features = target_out[0] if isinstance(target_out, tuple) else target_out
        val = self.target_q_head(features)
        if take_min_over_ensemble and val.ndim > 1:
            val = jnp.min(val, axis = 0)
        return val

    def compute_prefix_cache(
        self,
        observation: _model.Observation,
        use_target: bool = False,
    ) -> tuple[at.Array, at.Array]:
        network = self.target_q_network if use_target else self.q_network
        if not hasattr(network, "compute_prefix_cache"):
            raise AttributeError(
                f"{type(network).__name__} does not support prefix caching. "
                "Callers should check hasattr before invoking."
            )
        return network.compute_prefix_cache(observation)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        if policy is None:
            raise ValueError("CQLValueFunction requires a policy for action sampling.")
        if rng is None:
            raise ValueError("CQLValueFunction requires rng for action sampling.")
        del train
        only_use_next_actions_for_cql = self.only_use_next_actions_for_cql
        if isinstance(policy, _best_of_n.BestOfNWrapper) and policy.base_model is None:
            only_use_next_actions_for_cql = True
        q_loss, cql_loss, info = _objectives.cql_objective(
            self.q_network,
            self.q_head,
            self.target_q_network,
            self.target_q_head,
            transition,
            policy,
            rng=rng,
            discount=self.discount,
            action_bounds=self.action_bounds,
            cql_alpha=self.cql_alpha,
            cql_temp=self.cql_temp,
            cql_n_actions=self.cql_n_actions,
            cql_action_sample_method=self.cql_action_sample_method,
            cql_importance_sample=self.cql_importance_sample,
            only_use_next_actions_for_cql=only_use_next_actions_for_cql,
            cql_max_target_backup=self.cql_max_target_backup,
            cql_clip_diff_min=self.cql_clip_diff_min,
            cql_clip_diff_max=self.cql_clip_diff_max,
            use_calql=self.use_calql,
            use_calql_on_random_actions=self.use_calql_on_random_actions,
            value_function=self,
        )
        total_loss = q_loss + self.cql_alpha * cql_loss
        info["cql_alpha"] = jnp.array(self.cql_alpha)
        return total_loss, info

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_q_network, self.q_network, self.tau)
        _polyak_update(self.target_q_head, self.q_head, self.tau)


# =============================================================================
# Multi-Transition Value Functions
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MultiMCValueFunctionConfig(BaseMultiValueFunctionConfig):
    """Multi-transition Monte-Carlo value function config."""

    network_config: MultiMLPNetworkConfig
    head_config: HeadConfig

    @override
    def create(self, rng: at.KeyArrayLike) -> MultiMCValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)

        return MultiMCValueFunction(network=network, head=head)

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
class MultiSARSAValueFunctionConfig(BaseMultiValueFunctionConfig):
    """Multi-transition SARSA value function config with target network."""

    network_config: MultiMLPNetworkConfig
    head_config: HeadConfig
    discount: float = 0.99
    tau: float = 0.005

    @override
    def create(self, rng: at.KeyArrayLike) -> MultiSARSAValueFunction:
        rng = jax.random.key(rng) if isinstance(rng, int) else rng
        net_rng, head_rng = jax.random.split(rng, 2)

        network = self.network_config.create(net_rng)
        head = self.head_config.create(network.feature_dim, head_rng)
        target_network = self.network_config.create(net_rng)
        target_head = self.head_config.create(target_network.feature_dim, head_rng)

        return MultiSARSAValueFunction(
            network=network,
            head=head,
            target_network=target_network,
            target_head=target_head,
            discount=self.discount,
            tau=self.tau,
        )

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
class MultiIQLValueFunctionConfig(BaseMultiValueFunctionConfig):
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


class MultiValueFunction(BaseMultiValueFunction):
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
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        del policy
        return _objectives.mc_objective(self.network, self.head, transition, rng=rng)


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
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        del train, policy
        return _objectives.sarsa_objective(
            self.network,
            self.head,
            transition,
            self.target_network,
            self.target_head,
            discount=self.discount,
            rng=rng,
        )

    @override
    def post_step_update(self) -> None:
        """Polyak averaging for target network."""
        _polyak_update(self.target_network, self.network, self.tau)
        _polyak_update(self.target_head, self.head, self.tau)


class MultiIQLValueFunction(BaseMultiValueFunction):
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
        policy: _model.BaseModel | None = None,
    ) -> tuple[at.Float[at.Array, "*b n"], dict[str, at.Array]]:
        """Compute combined Q + V loss."""
        del train, rng, policy
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
