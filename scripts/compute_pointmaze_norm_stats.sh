#!/bin/bash
set -e

# List of all pointmaze configurations
CONFIGS=(
    "pointmaze_large_v2_q_regression"
    "pointmaze_large_v2_q_sarsa"
    "pointmaze_large_v2_q_multi_mc"
    "pointmaze_large_v2_q_multi_mc_consecutive"
    "pointmaze_large_v2_q_multi_sarsa_consecutive"
    "pointmaze_large_v2_q_multi_sarsa_hl_gauss_consecutive"
)

# Loop through each config and compute norm stats
for config in "${CONFIGS[@]}"; do
    echo "Computing norm stats for $config..."
    uv run scripts/compute_norm_stats.py --config-name "$config"
done
