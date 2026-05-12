"""Test loading a pi0 policy checkpoint."""

import logging

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)
logger = logging.getLogger(__name__)

logger.info("Importing openpi.robocoin_utils.load_model_utils.LoadPolicyConfig...")
from openpi.robocoin_utils.load_model_utils import LoadPolicyConfig

logger.info("Importing openpi.robocoin_utils.load_model_utils.load_policy...")
from openpi.robocoin_utils.load_model_utils import load_policy

logger.info("All imports complete.")

load_config = LoadPolicyConfig(
    config_name = "robocoin_bimanual_pi05",
    checkpoint_path = "/data/group_data/rl/saksham3/checkpoints/robocoin/pi05_finetune/robocoin_bimanual_pi05_rlds/real_hang_state_pi05_finetune/",
    fine_tune = "real_hang_pi05_finetune",
    step = 329999,
)
model, config = load_policy(load_config)
print(f"Model type: {type(model).__name__}")
print(f"action_dim={model.action_dim}, action_horizon={model.action_horizon}")
print(f"Config model type: {config.model.model_type}")