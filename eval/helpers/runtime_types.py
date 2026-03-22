"""Typed runtime containers for evaluation model bundles."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax

from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class PolicyBundle:
    """Loaded policy model plus its runtime transform stack."""

    model: _model.BaseModel
    input_transform: Any
    output_transform: Any
    sample_kwargs: dict[str, Any]
    metadata: dict[str, Any]
    rng: jax.Array
    train_config: Any


@dataclasses.dataclass(frozen=True)
class ValueBundle:
    """Loaded value model plus its runtime transform stack."""

    model: Any
    input_transform: Any
    train_config: Any


@dataclasses.dataclass(frozen=True)
class EvalObservationPair:
    """Policy-facing and critic-facing raw observations for a single env state."""

    policy_inputs: dict[str, Any]
    critic_inputs: dict[str, Any]
    prompt: str
