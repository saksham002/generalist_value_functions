"""Policy extraction module for offline RL.

This module provides components for extracting policies from learned
value functions. It includes:

- Functional objectives (DDPG, BC regularization, entropy)
- Temperature module for learnable entropy coefficient
- Weighted sum objective for combining multiple objectives

Example usage for SAC-style policy extraction:

    from openpi.policy_extraction.objectives import (
        ddpg_objective,
        entropy_objective,
        weighted_sum_objective,
    )
    from openpi.policy_extraction.temperature import TemperatureConfig

    # Create objectives
    objectives = [
        (ddpg_objective, 1.0, {"q_function": q_fn, "aggregation": "min"}),
        (entropy_objective, temperature.value, {}),
    ]

    # Compute combined loss
    loss, info = weighted_sum_objective(policy, observation, rng, objectives)
"""

from openpi.policy_extraction.objectives import AWRPolicyConfig
from openpi.policy_extraction.objectives import BasePolicyExtractionConfig
from openpi.policy_extraction.objectives import DDPGPolicyConfig
from openpi.policy_extraction.objectives import MultiAWRPolicyConfig
from openpi.policy_extraction.objectives import NoopPolicyConfig
from openpi.policy_extraction.objectives import awr_multi_objective
from openpi.policy_extraction.objectives import awr_objective
from openpi.policy_extraction.objectives import bc_regularization_objective
from openpi.policy_extraction.objectives import ddpg_objective
from openpi.policy_extraction.objectives import entropy_objective
from openpi.policy_extraction.objectives import noop_objective
from openpi.policy_extraction.objectives import weighted_sum_objective
from openpi.policy_extraction.temperature import Temperature
from openpi.policy_extraction.temperature import TemperatureConfig

__all__ = [
    # Policy Extraction Configs
    "AWRPolicyConfig",
    "BasePolicyExtractionConfig",
    "DDPGPolicyConfig",
    "MultiAWRPolicyConfig",
    "NoopPolicyConfig",
    # Temperature
    "Temperature",
    "TemperatureConfig",
    # Objectives
    "awr_multi_objective",
    "awr_objective",
    "bc_regularization_objective",
    "ddpg_objective",
    "entropy_objective",
    "noop_objective",
    "weighted_sum_objective",
]
