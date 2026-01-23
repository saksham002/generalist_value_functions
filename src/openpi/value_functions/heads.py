"""Value heads for converting network features to value predictions."""

import dataclasses

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.shared import array_typing as at
from openpi.value_functions import hl_gauss as _hl_gauss


@dataclasses.dataclass(frozen=True)
class RegressionHeadConfig:
    """Configuration for regression output head."""

    orthogonal_init_scale: float | None = None

    def create(self, feature_dim: int, rng: at.KeyArrayLike) -> "RegressionHead":
        return RegressionHead(feature_dim, self.orthogonal_init_scale, rngs=nnx.Rngs(rng))


class RegressionHead(nnx.Module):
    """Linear projection from features to scalar value."""

    def __init__(
        self,
        feature_dim: int,
        orthogonal_init_scale: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        if orthogonal_init_scale is not None:
            kernel_init = nnx.initializers.orthogonal(scale=orthogonal_init_scale)
            self.linear = nnx.Linear(feature_dim, 1, kernel_init=kernel_init, rngs=rngs)
        else:
            self.linear = nnx.Linear(feature_dim, 1, rngs=rngs)

    def __call__(self, features: at.Float[at.Array, "*b feature_dim"]) -> at.Float[at.Array, "*b"]:
        """Compute scalar value from features."""
        return self.linear(features).squeeze(-1)


@dataclasses.dataclass(frozen=True)
class CategoricalHeadConfig:
    """Configuration for categorical (HL-Gauss) output head."""

    v_min: float = 0.0
    v_max: float = 1.0
    num_bins: int = 51
    sigma: float = 0.75
    orthogonal_init_scale: float | None = None

    def create(self, feature_dim: int, rng: at.KeyArrayLike) -> "CategoricalHead":
        return CategoricalHead(
            feature_dim,
            self.v_min,
            self.v_max,
            self.num_bins,
            self.sigma,
            self.orthogonal_init_scale,
            rngs=nnx.Rngs(rng),
        )


class CategoricalHead(nnx.Module):
    """Categorical (HL-Gauss) output head.

    Outputs a distribution over value bins and computes expected value.
    """

    v_min: float
    v_max: float
    num_bins: int
    sigma: float

    def __init__(
        self,
        feature_dim: int,
        v_min: float,
        v_max: float,
        num_bins: int,
        sigma: float,
        orthogonal_init_scale: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_bins = num_bins
        self.sigma = sigma

        if orthogonal_init_scale is not None:
            kernel_init = nnx.initializers.orthogonal(scale=orthogonal_init_scale)
            self.linear = nnx.Linear(feature_dim, num_bins, kernel_init=kernel_init, rngs=rngs)
        else:
            self.linear = nnx.Linear(feature_dim, num_bins, rngs=rngs)

    def compute_logits(self, features: at.Float[at.Array, "*b feature_dim"]) -> at.Float[at.Array, "*b num_bins"]:
        """Compute raw logits over bins."""
        return self.linear(features)

    def __call__(self, features: at.Float[at.Array, "*b feature_dim"]) -> at.Float[at.Array, "*b"]:
        """Compute expected value from features."""
        logits = self.compute_logits(features)
        return _hl_gauss.logits_to_expected_value(logits, self.v_min, self.v_max)


@dataclasses.dataclass(frozen=True)
class CrossEntropyHeadConfig:
    """Configuration for cross-entropy classification output head.
    
    Discretizes the target value range [v_min, v_max] into num_bins bins and 
    uses standard cross-entropy loss with hard bin assignments (no Gaussian smoothing).
    """

    v_min: float = 0.0
    v_max: float = 1.0
    num_bins: int = 51
    orthogonal_init_scale: float | None = None

    def create(self, feature_dim: int, rng: at.KeyArrayLike) -> "CrossEntropyHead":
        return CrossEntropyHead(
            feature_dim,
            self.v_min,
            self.v_max,
            self.num_bins,
            self.orthogonal_init_scale,
            rngs=nnx.Rngs(rng),
        )


class CrossEntropyHead(nnx.Module):
    """Cross-entropy classification head with hard bin discretization.

    Discretizes target values into bins and uses standard cross-entropy loss.
    Unlike CategoricalHead (HL-Gauss), this uses hard one-hot targets.
    """

    v_min: float
    v_max: float
    num_bins: int

    def __init__(
        self,
        feature_dim: int,
        v_min: float,
        v_max: float,
        num_bins: int,
        orthogonal_init_scale: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_bins = num_bins

        if orthogonal_init_scale is not None:
            kernel_init = nnx.initializers.orthogonal(scale=orthogonal_init_scale)
            self.linear = nnx.Linear(feature_dim, num_bins, kernel_init=kernel_init, rngs=rngs)
        else:
            self.linear = nnx.Linear(feature_dim, num_bins, rngs=rngs)

    def compute_logits(self, features: at.Float[at.Array, "*b feature_dim"]) -> at.Float[at.Array, "*b num_bins"]:
        """Compute raw logits over bins."""
        return self.linear(features)

    def __call__(self, features: at.Float[at.Array, "*b feature_dim"]) -> at.Float[at.Array, "*b"]:
        """Compute expected value from features (same as CategoricalHead)."""
        logits = self.compute_logits(features)
        return _hl_gauss.logits_to_expected_value(logits, self.v_min, self.v_max)


# Type alias for value heads
ValueHead = RegressionHead | CategoricalHead | CrossEntropyHead
HeadConfig = RegressionHeadConfig | CategoricalHeadConfig | CrossEntropyHeadConfig


# =============================================================================
# Ensemble Heads
# =============================================================================


@dataclasses.dataclass(frozen=True)
class EnsembleHeadConfig:
    """Config for an ensemble of heads using vmap.

    Creates vectorized heads with parameters of shape (ensemble_size, ...).
    """

    base_config: HeadConfig
    ensemble_size: int = 2

    def create(self, feature_dim: int, rng: at.KeyArrayLike) -> "EnsembleHead":
        rngs = nnx.Rngs(rng)

        @nnx.split_rngs(splits=self.ensemble_size)
        @nnx.vmap
        def create_member(rngs: nnx.Rngs) -> ValueHead:
            return self.base_config.create(feature_dim, rngs.params())

        vectorized_head = create_member(rngs)
        return EnsembleHead(
            vectorized_head=vectorized_head,
            ensemble_size=self.ensemble_size,
        )


class EnsembleHead(nnx.Module):
    """Ensemble of heads with vectorized parameters.

    Uses nnx.vmap for efficient parallel computation over ensemble members.
    Expects features from EnsembleNetwork with shape [ensemble, batch, feature_dim].
    """

    vectorized_head: ValueHead
    ensemble_size: int

    def __init__(self, vectorized_head: ValueHead, ensemble_size: int):
        super().__init__()
        self.vectorized_head = vectorized_head
        self.ensemble_size = ensemble_size

    def __call__(self, features: at.Array) -> at.Array:
        """Compute values for all ensemble members.

        Args:
            features: [ensemble, batch, feature_dim] from EnsembleNetwork.

        Returns:
            Values of shape [ensemble, batch] or [ensemble, batch, n].
        """

        @nnx.vmap(in_axes=(0, 0), out_axes=0)
        def compute_single(head: ValueHead, feats: at.Array) -> at.Array:
            return head(feats)

        return compute_single(self.vectorized_head, features)

    def compute_min(self, features: at.Array) -> at.Array:
        """Compute min value across ensemble (pessimistic estimate).

        Used in IQL/SAC/TD3 to prevent overestimation bias.

        Args:
            features: [ensemble, batch, feature_dim] from EnsembleNetwork.

        Returns:
            Min values of shape [batch] or [batch, n].
        """
        all_values = self(features)
        return jnp.min(all_values, axis=0)

    def compute_mean(self, features: at.Array) -> at.Array:
        """Compute mean value across ensemble.

        Args:
            features: [ensemble, batch, feature_dim] from EnsembleNetwork.

        Returns:
            Mean values of shape [batch] or [batch, n].
        """
        all_values = self(features)
        return jnp.mean(all_values, axis=0)


# Extended type aliases including ensemble types
AnyValueHead = ValueHead | EnsembleHead
AnyHeadConfig = HeadConfig | EnsembleHeadConfig
