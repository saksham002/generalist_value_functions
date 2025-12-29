"""HL-Gauss (Histogram Loss with Gaussian smoothing) utilities.

This module implements the HL-Gauss loss function for training categorical value
functions. Instead of predicting a scalar value directly, the model outputs
logits over discrete bins, and targets are converted to soft categorical labels
using Gaussian smoothing.

Reference:
    Imani & White, "Improving Regression Performance with Distributional Losses"
    https://arxiv.org/abs/1806.04613

    Farebrother et al., "Stop Regressing: Training Value Functions via
    Classification for Scalable Deep RL"
    https://arxiv.org/abs/2403.03950
"""

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


def compute_bin_centers(
    v_min: float,
    v_max: float,
    num_bins: int,
) -> at.Float[at.Array, "..."]:  # [num_bins]
    """Compute the center values for each bin.

    Args:
        v_min: Minimum value of the support.
        v_max: Maximum value of the support.
        num_bins: Number of bins.

    Returns:
        Bin centers of shape [num_bins].
    """
    return jnp.linspace(v_min, v_max, num_bins)


def compute_hl_gauss_targets(
    target_values: at.Float[at.Array, "*b"],
    v_min: float,
    v_max: float,
    num_bins: int,
    sigma: float,
) -> at.Float[at.Array, "b num_bins"]:
    """Convert scalar targets to soft categorical labels using Gaussian smoothing.

    The HL-Gauss method distributes probability mass across bins using a Gaussian
    centered at the target value. This leverages the ordinal structure of the
    regression task.

    Args:
        target_values: Target values of shape [batch].
        v_min: Minimum value of the support.
        v_max: Maximum value of the support.
        num_bins: Number of bins.
        sigma: Standard deviation of the Gaussian smoothing.

    Returns:
        Soft categorical targets of shape [batch, num_bins], normalized to sum to 1.
    """
    # Compute bin centers
    bin_centers = compute_bin_centers(v_min, v_max, num_bins)  # [num_bins]

    # Clip target values to the support range
    target_values = jnp.clip(target_values, v_min, v_max)

    # Compute Gaussian probabilities for each bin
    # target_values: [batch] -> [batch, 1]
    # bin_centers: [num_bins] -> [1, num_bins]
    target_values = target_values[:, None]  # [batch, 1]
    bin_centers = bin_centers[None, :]  # [1, num_bins]

    # Gaussian PDF (unnormalized)
    log_probs = -0.5 * jnp.square((bin_centers - target_values) / sigma)  # [batch, num_bins]

    # Normalize to get probabilities (softmax over bins)
    return jax.nn.softmax(log_probs, axis=-1)  # [batch, num_bins]


def hl_gauss_loss(
    logits: at.Float[at.Array, "b num_bins"],
    target_values: at.Float[at.Array, "*b"],
    v_min: float,
    v_max: float,
    sigma: float,
) -> at.Float[at.Array, "*b"]:
    """Compute HL-Gauss cross-entropy loss.

    Args:
        logits: Predicted logits of shape [batch, num_bins].
        target_values: Target values of shape [batch].
        v_min: Minimum value of the support.
        v_max: Maximum value of the support.
        sigma: Standard deviation of the Gaussian smoothing.

    Returns:
        Per-sample cross-entropy loss of shape [batch].
    """
    num_bins = logits.shape[-1]

    # Compute soft targets
    targets = compute_hl_gauss_targets(target_values, v_min, v_max, num_bins, sigma)

    # Compute cross-entropy loss: -sum(targets * log_softmax(logits))
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # [batch, num_bins]
    return -jnp.sum(targets * log_probs, axis=-1)  # [batch]


def logits_to_expected_value(
    logits: at.Float[at.Array, "b num_bins"],
    v_min: float,
    v_max: float,
) -> at.Float[at.Array, "*b"]:
    """Convert logits to expected value.

    Args:
        logits: Predicted logits of shape [batch, num_bins].
        v_min: Minimum value of the support.
        v_max: Maximum value of the support.

    Returns:
        Expected values of shape [batch].
    """
    num_bins = logits.shape[-1]
    bin_centers = compute_bin_centers(v_min, v_max, num_bins)  # [num_bins]

    # Softmax to get probabilities
    probs = jax.nn.softmax(logits, axis=-1)  # [batch, num_bins]

    # Expected value: sum(probs * bin_centers)
    return jnp.sum(probs * bin_centers[None, :], axis=-1)  # [batch]
