"""Value heads for converting network features to value predictions."""

import dataclasses

import flax.nnx as nnx

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


# Type alias for value heads
ValueHead = RegressionHead | CategoricalHead
HeadConfig = RegressionHeadConfig | CategoricalHeadConfig
