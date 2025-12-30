"""Ensemble of value functions using vmap for efficiency.

This module provides an ensemble wrapper for value functions that uses
JAX's vmap to efficiently compute over ensemble members with different
parameters. This is useful for SAC, TD3, and other algorithms that use
multiple Q-functions for stability.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.value_functions.base import BaseValueFunction
from openpi.value_functions.base import BaseValueFunctionConfig
from openpi.value_functions.base import Transition


@dataclasses.dataclass(frozen=True)
class EnsembleValueFunctionConfig(BaseValueFunctionConfig):
    """Configuration for an ensemble of value functions.

    Uses vmap to efficiently compute over ensemble members with different
    parameters. Each member is initialized independently with a different
    random seed.
    """

    # The base value function config to ensemble
    base_config: BaseValueFunctionConfig

    # Number of value functions in the ensemble
    num_ensemble: int = 2

    @override
    def create(self, rng: at.KeyArrayLike) -> "EnsembleValueFunction":
        """Create an ensemble of value functions using vmap.

        Each ensemble member has independent parameters initialized with
        different random keys.
        """
        # Create ensemble using vmap over initialization
        rngs = nnx.Rngs(rng)

        @nnx.split_rngs(splits=self.num_ensemble)
        @nnx.vmap
        def create_member(rngs: nnx.Rngs) -> BaseValueFunction:
            return self.base_config.create(rngs.params())

        # This creates a single module with vectorized parameters
        # Parameters have shape (num_ensemble, ...)
        vectorized_member = create_member(rngs)

        return EnsembleValueFunction(
            vectorized_member=vectorized_member,
            num_ensemble=self.num_ensemble,
            action_conditioned=getattr(self.base_config, "action_conditioned", True),
        )

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        return self.base_config.inputs_spec(batch_size=batch_size)


class EnsembleValueFunction(BaseValueFunction):
    """Ensemble of value functions using vmap for efficient computation.

    Parameters are stacked along axis 0, and computations are vectorized
    over the ensemble dimension using nnx.vmap.

    This is useful for:
    - SAC/TD3 (min over 2 Q-functions for pessimistic estimate)
    - Uncertainty estimation (variance across ensemble)
    - REDQ-style subsampling
    """

    vectorized_member: BaseValueFunction  # Has params with shape (num_ensemble, ...)
    num_ensemble: int
    action_conditioned: bool

    def __init__(
        self,
        vectorized_member: BaseValueFunction,
        num_ensemble: int,
        action_conditioned: bool,
    ):
        super().__init__()
        self.vectorized_member = vectorized_member
        self.num_ensemble = num_ensemble
        self.action_conditioned = action_conditioned

    @override
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "num_ensemble batch"]:
        """Compute values for all ensemble members.

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).

        Returns:
            Values of shape (num_ensemble, batch).
        """

        @nnx.vmap(in_axes=(0, None, None), out_axes=0)
        def compute_single(
            member: BaseValueFunction,
            obs: _model.Observation,
            act: _model.Actions | None,
        ):
            return member.compute_value(obs, act)

        return compute_single(self.vectorized_member, observation, action)

    def compute_min_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "batch"]:
        """Compute min value across ensemble (pessimistic estimate).

        This is used in SAC/TD3 to prevent overestimation bias.

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).

        Returns:
            Min values of shape (batch,).
        """
        all_values = self.compute_value(observation, action)
        return jnp.min(all_values, axis=0)

    def compute_mean_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "batch"]:
        """Compute mean value across ensemble.

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).

        Returns:
            Mean values of shape (batch,).
        """
        all_values = self.compute_value(observation, action)
        return jnp.mean(all_values, axis=0)

    def compute_std_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "batch"]:
        """Compute std of values across ensemble (uncertainty estimate).

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).

        Returns:
            Std of values of shape (batch,).
        """
        all_values = self.compute_value(observation, action)
        return jnp.std(all_values, axis=0)

    @override
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
        """Compute mean loss across ensemble members.

        Each member computes loss against the same target (from transition).

        Args:
            transition: SARSA transition with targets.
            train: Whether in training mode.
            rng: Random key (passed to member compute_loss).

        Returns:
            Tuple of (mean_loss, info_dict).
        """

        @nnx.vmap(in_axes=(0, None), out_axes=(0, 0))
        def compute_single_loss(
            member: BaseValueFunction, trans: Transition
        ) -> tuple[at.Float[at.Array, "batch"], dict[str, at.Array]]:
            return member.compute_loss(trans, train=train, rng=rng)

        # Returns (num_ensemble, batch) losses and dict of (num_ensemble, ...) infos
        all_losses, all_infos = compute_single_loss(self.vectorized_member, transition)

        # Mean loss across ensemble
        mean_loss = jnp.mean(all_losses, axis=0)

        # Aggregate info
        info = {
            "ensemble_loss_mean": jnp.mean(mean_loss),
            "ensemble_loss_std": jnp.std(all_losses),
        }

        # Add mean of each info field across ensemble
        for key, value in all_infos.items():
            info[f"ensemble_{key}"] = jnp.mean(value)

        return mean_loss, info

    def compute_logits(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "num_ensemble batch num_bins"]:
        """Compute logits for categorical ensemble members.

        This is only valid for ensembles of CategoricalValueMLP.

        Args:
            observation: Observation containing state.
            action: Actions (required if action_conditioned=True).

        Returns:
            Logits of shape (num_ensemble, batch, num_bins).
        """

        @nnx.vmap(in_axes=(0, None, None), out_axes=0)
        def compute_single_logits(
            member: BaseValueFunction,
            obs: _model.Observation,
            act: _model.Actions | None,
        ):
            # Assumes member has compute_logits method (CategoricalValueMLP)
            return member.compute_logits(obs, act)

        return compute_single_logits(self.vectorized_member, observation, action)
