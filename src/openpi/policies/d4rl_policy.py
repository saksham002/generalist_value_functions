"""
D4RL policy transforms.

These transforms handle state-only D4RL data (no images).
"""

import dataclasses

import numpy as np

from openpi import transforms


def make_d4rl_example(obs_dim: int = 29, action_dim: int = 8) -> dict:
    """Creates a random input example for D4RL policies."""
    return {
        "state": np.random.rand(obs_dim).astype(np.float32),
        "prompt": "antmaze-umaze-v2",
    }


@dataclasses.dataclass(frozen=True)
class D4RLInputs(transforms.DataTransformFn):
    """
    Transform inputs for state-only D4RL data.

    D4RL environments don't have images, only state observations.
    This transform prepares the state for the model.
    """

    def __call__(self, data: dict) -> dict:
        inputs = {
            # After repack, state is directly under 'state' key
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (environment name used as task description).
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class D4RLOutputs(transforms.DataTransformFn):
    """
    Transform outputs from the model back to D4RL action format.

    Used during inference only.
    """

    action_dim: int = 8  # Default for antmaze

    def __call__(self, data: dict) -> dict:
        # Return actions truncated to the correct dimension.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
