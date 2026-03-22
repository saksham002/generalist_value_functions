"""Runtime preprocessing for eval-time policy and critic models."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model

from eval.helpers import runtime_types


def prepare_policy_observation(
    bundle: runtime_types.PolicyBundle,
    raw_inputs: dict[str, Any],
) -> tuple[dict[str, Any], _model.Observation]:
    transformed = jax.tree.map(lambda x: x, raw_inputs)
    transformed = bundle.input_transform(transformed)
    batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
    return batched, _model.Observation.from_dict(batched)


def decode_policy_actions(
    bundle: runtime_types.PolicyBundle,
    transformed_inputs: dict[str, Any],
    normalized_actions: np.ndarray,
) -> np.ndarray:
    outputs = {
        "state": np.asarray(transformed_inputs["state"][0]),
        "actions": np.asarray(normalized_actions),
    }
    postprocessed = bundle.output_transform(outputs)
    return np.asarray(postprocessed["actions"], dtype = np.float32)


def prepare_value_inputs(
    bundle: runtime_types.ValueBundle,
    raw_inputs: dict[str, Any],
) -> tuple[_model.Observation, jax.Array]:
    transformed = jax.tree.map(lambda x: x, raw_inputs)
    transformed = bundle.input_transform(transformed)
    batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
    observation = _model.Observation.from_dict(batched)
    return observation, jnp.asarray(batched["actions"])
