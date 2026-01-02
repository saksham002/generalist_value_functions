"""Data transforms for value function training.

These transforms handle the preparation of SARSA tuples and Monte-Carlo returns
for training value functions with reinforcement learning.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.value_functions import hl_gauss as _hl_gauss


@dataclasses.dataclass(frozen=True)
class ValueFunctionInputs(transforms.DataTransformFn):
    """Transform inputs for value function training.

    Expects the dataset to contain the following RL fields:
    - state, next_state: State observations
    - actions, next_actions: Actions taken (singular action/next_action also supported)
    - reward, mc_return: Reward signals
    - termination: True if episode ended naturally (goal/failure)
    - truncation: True if episode was cut off (time limit)

    Note: For bootstrapping (TD learning), only use termination.
    Truncated states should still bootstrap to avoid bias.
    """

    def __call__(self, data: dict) -> dict:
        # Helper to extract scalar from potentially shaped tensor
        def to_scalar(x):
            if hasattr(x, "numpy"):
                x = x.numpy()
            x = np.asarray(x, dtype=np.float32)
            if x.ndim > 0 and x.size == 1:
                return x.item()
            return x

        def to_bool(x):
            if hasattr(x, "numpy"):
                x = x.numpy()
            x = np.asarray(x)
            if x.ndim > 0 and x.size == 1:
                return bool(x.item())
            return bool(x)

        mc_return = to_scalar(data["mc_return"])

        # Support both actions (LeRobot convention) and action (legacy)
        action = data.get("actions", data.get("action"))
        next_action = data.get("next_actions", data.get("next_action"))

        result = {
            "state": data["state"],
            "actions": action,  # Keep as 'actions' for normalization compatibility
            "reward": to_scalar(data["reward"]),
            "next_state": data["next_state"],
            "next_actions": next_action,  # Keep as 'next_actions' for consistency
            "mc_return": np.float32(mc_return),
            "termination": np.array(to_bool(data["termination"]), dtype=np.bool_),
            "truncation": np.array(to_bool(data["truncation"]), dtype=np.bool_),
        }

        # Pass through prompt if available
        if "prompt" in data:
            result["prompt"] = data["prompt"]

        return result


@dataclasses.dataclass(frozen=True)
class RegressionValueOutputs(transforms.DataTransformFn):
    """Output transform for regression critics.

    Simply returns the scalar value prediction.
    """

    def __call__(self, data: dict) -> dict:
        return {"value": np.asarray(data["value"])}


@dataclasses.dataclass(frozen=True)
class CategoricalValueOutputs(transforms.DataTransformFn):
    """Output transform for categorical critics.

    Converts logits over bins to an expected value.
    """

    v_min: float
    v_max: float
    num_bins: int = 51

    def __call__(self, data: dict) -> dict:
        logits = data["logits"]
        expected_value = _hl_gauss.logits_to_expected_value(logits, self.v_min, self.v_max)
        return {"value": np.asarray(expected_value)}


def make_value_function_example(
    obs_dim: int = 29,
    action_dim: int = 8,
) -> dict:
    """Creates a random input example for value function training."""
    return {
        "state": np.random.rand(obs_dim).astype(np.float32),
        "action": np.random.rand(action_dim).astype(np.float32),
        "reward": np.float32(np.random.rand()),
        "next_state": np.random.rand(obs_dim).astype(np.float32),
        "next_action": np.random.rand(action_dim).astype(np.float32),
        "mc_return": np.float32(np.random.rand() * 10.0),
        "termination": np.array(False, dtype=np.bool_),  # noqa: FBT003
        "truncation": np.array(False, dtype=np.bool_),  # noqa: FBT003
        "prompt": "value_function_training",
    }


# Backwards compatibility aliases
D4RLValueFunctionInputs = ValueFunctionInputs
D4RLRegressionValueOutputs = RegressionValueOutputs
D4RLCategoricalValueOutputs = CategoricalValueOutputs
make_d4rl_rl_example = make_value_function_example
