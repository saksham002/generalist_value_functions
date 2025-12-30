"""Learnable temperature for SAC-style entropy regularization.

The temperature (alpha) controls the trade-off between exploration (entropy)
and exploitation (reward maximization) in SAC. It can be fixed or learned
automatically to maintain a target entropy level.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class TemperatureConfig:
    """Configuration for temperature (entropy coefficient).

    The temperature controls the entropy bonus in SAC:
    - Higher temperature → more exploration (higher entropy)
    - Lower temperature → more exploitation (lower entropy)

    When learnable=True, the temperature is automatically adjusted to
    maintain the target entropy level.
    """

    # Initial temperature value
    init_temperature: float = 1.0

    # Whether to learn temperature automatically
    learnable: bool = True

    # Target entropy for automatic tuning
    # If None, defaults to -action_dim (heuristic from SAC paper)
    target_entropy: float | None = None

    def create(self, action_dim: int) -> "Temperature":
        """Create a Temperature module.

        Args:
            action_dim: Action dimension (used for default target entropy).

        Returns:
            Initialized Temperature module.
        """
        target = self.target_entropy if self.target_entropy is not None else float(-action_dim)
        return Temperature(
            log_temperature=jnp.log(self.init_temperature),
            target_entropy=target,
            learnable=self.learnable,
        )


class Temperature(nnx.Module):
    """Learnable temperature for SAC-style entropy regularization.

    Maintains log_temperature as the learnable parameter to ensure
    temperature is always positive (via exp).

    The temperature loss encourages the policy to maintain a target
    entropy level:
    - If actual entropy < target: loss is positive, gradient increases alpha
    - If actual entropy > target: loss is negative, gradient decreases alpha
    """

    target_entropy: float
    learnable: bool

    def __init__(self, log_temperature: float, target_entropy: float, *, learnable: bool):
        """Initialize Temperature module.

        Args:
            log_temperature: Initial log temperature value.
            target_entropy: Target entropy for automatic tuning.
            learnable: Whether temperature is learnable.
        """
        self.target_entropy = target_entropy
        self.learnable = learnable

        if learnable:
            self.log_temperature = nnx.Param(jnp.array(log_temperature))
        else:
            self._log_temperature_value = jnp.array(log_temperature)

    @property
    def value(self) -> at.Float[at.Array, ""]:
        """Current temperature value (always positive via exp)."""
        if self.learnable:
            return jnp.exp(self.log_temperature.value)
        return jnp.exp(self._log_temperature_value)

    def compute_loss(
        self,
        log_prob: at.Float[at.Array, "*b"],
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        """Compute temperature loss for automatic tuning.

        The loss is:
            L(alpha) = -alpha * mean(log pi(a|s) + H_target)
                     = -alpha * mean(log pi(a|s) - (-H_target))
                     = alpha * mean(H_target - (-log pi(a|s)))
                     = alpha * mean(H_target - entropy)

        Gradient w.r.t. log(alpha):
            dL/dlog(alpha) = alpha * mean(H_target - entropy)

        This pushes alpha:
        - Up when entropy < target (need more exploration)
        - Down when entropy > target (need less exploration)

        Args:
            log_prob: Log probabilities of sampled actions, shape (batch,).

        Returns:
            Tuple of (loss, info_dict).
        """
        if not self.learnable:
            return jnp.array(0.0), {"temperature": self.value}

        # Detach log_prob (don't backprop through policy for temperature update)
        log_prob = jax.lax.stop_gradient(log_prob)

        # Current entropy estimate (negative log prob)
        current_entropy = jnp.mean(-log_prob)

        # Loss: -alpha * mean(log pi + H_target) = alpha * mean(H_target - entropy)
        loss = -self.value * jnp.mean(log_prob + self.target_entropy)

        info = {
            "temperature": self.value,
            "target_entropy": jnp.array(self.target_entropy),
            "current_entropy": current_entropy,
            "entropy_gap": current_entropy - self.target_entropy,
        }
        return loss, info
