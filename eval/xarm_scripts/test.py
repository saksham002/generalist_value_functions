"""Test loading a pi0 policy checkpoint."""

import logging

from openpi.robocoin_utils.load_model_utils import LoadPolicyConfig
from openpi.robocoin_utils.load_model_utils import load_policy

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)

load_config = LoadPolicyConfig(
    config_name = "robocoin_bimanual_pi05",
    checkpoint_path = "/data/group_data/rl/saksham3/checkpoints/robocoin/pi05_finetune/real_hang_pi05_finetune/",
    fine_tune = "real_hang_pi05_finetune",
    step = 275000,
)
model, config = load_policy(load_config)
print(f"Model type: {type(model).__name__}")
print(f"action_dim={model.action_dim}, action_horizon={model.action_horizon}")
print(f"Config model type: {config.model.model_type}")