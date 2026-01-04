"""Base classes for value functions.

This module defines the abstract base classes for value functions, following
the pattern established by BaseModel and BaseModelConfig in the models module.

Value functions can be configured to be action-conditioned (Q-function) or not
(V-function) using the same base class hierarchy.
"""

import abc
import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Transition:
    """A transition (o, a, r, o', a') plus MC return for value function training.

    Different algorithms compute targets differently:
    - MC: target = mc_return
    - TD(0): target = reward + gamma * V(next_observation) * (1 - termination)
    - SARSA: target = reward + gamma * Q(next_observation, next_action) * (1 - termination)

    Note: For bootstrapping, only mask with termination, not truncation.
    Truncated states should still bootstrap to avoid bias.
    """

    observation: _model.Observation
    action: at.Float[at.Array, "*b action_dim"]
    reward: at.Float[at.Array, "*b"]
    next_observation: _model.Observation
    next_action: at.Float[at.Array, "*b action_dim"]
    mc_return: at.Float[at.Array, "*b"]
    termination: at.Bool[at.Array, "*b"]
    truncation: at.Bool[at.Array, "*b"]

    @classmethod
    def from_batch(cls, batch: dict) -> "Transition":
        """Create a Transition from a batch dictionary.

        Expects batch to contain:
        - state, next_state: Low-dim state observations
        - action, next_action: Actions
        - reward, mc_return: Reward signals
        - termination, truncation: Episode boundary flags
        """
        # Build Observation from batch - supports state-only for now
        observation = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(batch["state"]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        next_observation = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(batch["next_state"]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        return cls(
            observation=observation,
            action=jnp.asarray(batch["actions"]),
            reward=jnp.asarray(batch["reward"]),
            next_observation=next_observation,
            next_action=jnp.asarray(batch["next_actions"]),
            mc_return=jnp.asarray(batch["mc_return"]),
            termination=jnp.asarray(batch["termination"]),
            truncation=jnp.asarray(batch["truncation"]),
        )


@dataclasses.dataclass(frozen=True)
class MultiTransition:
    """A sequence of transitions for multi-state value functions.

    All arrays have shape [batch, num_transitions_per_sample, ...].
    This enables value functions that jointly predict values for multiple states.
    """

    observation: _model.Observation
    action: at.Float[at.Array, "*b n action_dim"]
    reward: at.Float[at.Array, "*b n"]
    next_observation: _model.Observation
    next_action: at.Float[at.Array, "*b n action_dim"]
    mc_return: at.Float[at.Array, "*b n"]
    termination: at.Bool[at.Array, "*b n"]
    truncation: at.Bool[at.Array, "*b n"]

    @classmethod
    def from_batch(cls, batch: dict) -> "MultiTransition":
        """Create a MultiTransition from a batch dictionary.

        Expects batch arrays to have shape [batch, num_transitions_per_sample, ...].
        """
        observation = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(batch["state"]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        next_observation = _model.Observation(
            images={},
            image_masks={},
            state=jnp.asarray(batch["next_state"]),
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        return cls(
            observation=observation,
            action=jnp.asarray(batch["actions"]),
            reward=jnp.asarray(batch["reward"]),
            next_observation=next_observation,
            next_action=jnp.asarray(batch["next_actions"]),
            mc_return=jnp.asarray(batch["mc_return"]),
            termination=jnp.asarray(batch["termination"]),
            truncation=jnp.asarray(batch["truncation"]),
        )


@dataclasses.dataclass(frozen=True)
class BaseValueFunctionConfig(abc.ABC):
    """Configuration shared by all value functions.

    Specific value function implementations should inherit from this class
    and implement the `create` method to instantiate the corresponding model.

    Value functions can be action-conditioned (Q-function) or not (V-function)
    based on the implementation's configuration.
    """

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseValueFunction":
        """Create a new value function, initializing parameters."""

    @abc.abstractmethod
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> (
        tuple[_model.Observation, at.Float[at.Array, "*b"]]
        | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "*b"]]
    ):
        """Returns the input specification for the value function.

        For V(s): returns (observation_spec, target_spec)
        For Q(s,a): returns (observation_spec, action_spec, target_spec)
        """


@dataclasses.dataclass
class BaseValueFunction(nnx.Module, abc.ABC):
    """Base class for value function implementations.

    Value functions estimate expected returns. They can be:
    - V(s): State-only value functions
    - Q(s,a): Action-conditioned value functions (Q-functions)

    The action parameter is optional to support both cases with the same interface.
    """

    @abc.abstractmethod
    def compute_value(
        self,
        observation: _model.Observation,
        action: _model.Actions | None = None,
    ) -> at.Float[at.Array, "*b"]:
        """Compute the value for each observation (and optionally action) in the batch.

        Args:
            observation: Observation containing state (and optionally images, prompt).
            action: Actions of shape [batch, action_horizon, action_dim].
                    Required for action-conditioned value functions.

        Returns:
            Estimated values of shape [batch].
        """

    @abc.abstractmethod
    def compute_loss(
        self,
        transition: Transition,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute the loss for value function training.

        Each algorithm implementation determines how to compute targets:
        - MC: uses transition.mc_return directly
        - TD: computes target from reward + gamma * V(next_state)
        - SARSA: computes target from reward + gamma * Q(next_state, next_action)
        - SAC: computes soft Bellman target with entropy bonus

        Args:
            transition: Full SARSA transition with (s, a, r, s', a', mc_return).
            train: Whether in training mode.
            rng: Random key for algorithms that need stochasticity (e.g., SAC).

        Returns:
            Tuple of:
            - Per-sample loss of shape [batch]
            - Dict of additional info to log (e.g., predicted values, TD errors)
        """

    def post_step_update(self) -> None:
        """Called after each training step.

        Override this method for operations that should happen after each
        gradient update, such as target network updates (Polyak averaging).

        Default implementation does nothing.
        """
