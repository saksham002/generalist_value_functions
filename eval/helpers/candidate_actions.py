"""Helpers for converting policy action chunks into critic action chunks."""

from __future__ import annotations

import numpy as np


def slice_real_actions(
    action_chunk: np.ndarray,
    *,
    real_action_start: int,
    real_action_dim: int,
) -> np.ndarray:
    """Extract the executable action dimensions from a padded policy chunk."""
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [T, A], got {action_chunk.shape}")
    required_dim = real_action_start + real_action_dim
    if action_chunk.shape[1] < required_dim:
        raise ValueError(
            f"Action dim {action_chunk.shape[1]} < required {required_dim} "
            f"(start = {real_action_start}, dim = {real_action_dim})"
        )
    return np.asarray(
        action_chunk[:, real_action_start : real_action_start + real_action_dim],
        dtype = np.float32,
    )


def subsample_even_actions(
    action_chunk: np.ndarray,
    *,
    target_length: int = 30,
) -> np.ndarray:
    """Subsample alternate actions from a 60 Hz chunk to the critic horizon."""
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [T, A], got {action_chunk.shape}")
    if action_chunk.shape[0] < target_length * 2:
        raise ValueError(
            f"Expected at least {target_length * 2} actions for even-index subsampling, got {action_chunk.shape[0]}"
        )
    return np.asarray(action_chunk[::2][:target_length], dtype = np.float32)


def batch_subsample_even_actions(
    action_chunks: np.ndarray,
    *,
    target_length: int = 30,
) -> np.ndarray:
    """Vectorized wrapper for subsampling a batch of candidate action chunks."""
    if action_chunks.ndim != 4:
        raise ValueError(f"Expected candidate actions with shape [B, N, T, A], got {action_chunks.shape}")
    batch_size, num_samples, _, action_dim = action_chunks.shape
    out = np.zeros((batch_size, num_samples, target_length, action_dim), dtype = np.float32)
    for batch_idx in range(batch_size):
        for sample_idx in range(num_samples):
            out[batch_idx, sample_idx] = subsample_even_actions(
                np.asarray(action_chunks[batch_idx, sample_idx]),
                target_length = target_length,
            )
    return out
