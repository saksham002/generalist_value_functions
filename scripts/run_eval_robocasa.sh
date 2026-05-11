#!/bin/bash
# Sim eval client. Connects to a remote serve_policy.py and steps the local MuJoCo env.

set -e

# =============================================================================
# Configuration
# =============================================================================
# Server runs on the TPU worker with process_index=0 (re-grep ~/tpu_job_output.log if pod recreated).
POLICY_HOST="35.204.26.43"
POLICY_PORT=8000

# Must match the policy server's POLICY_CONFIG (used to derive action fps).
POLICY_CONFIG="robocasa_pi05_finetune"
TASK_SET="CloseBlenderLid"
SPLIT="target"   # "target" | "pretrain"
NUM_TRIALS=100
SEED=86
LOG_DIR="/data/user_data/saksham3/eval_logs/robocasa_bestofn64_step50k"
LOG_VIDEOS=true
REPLAN_STEPS=10
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Derive model/env action fps from the policy training config (no hardcoding).
read -r MODEL_ACTION_FPS ENV_ACTION_FPS < <(
    PYTHONPATH="${REPO_ROOT}/src" /data/user_data/saksham3/vla/bin/python -c "
from openpi.training.config import get_config
d = get_config('${POLICY_CONFIG}').data
print(d.interpolation_config.target_fps, d.native_fps)
"
)

# Activate the robocasa_sim conda env (MuJoCo + robocasa).
# shellcheck disable=SC1091
source /home/saksham3/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/saksham3/conda-envs/robocasa_sim

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

echo "================================================"
echo "RoboCasa sim eval"
echo "================================================"
echo "  Task:       ${TASK_SET}"
echo "  Split:      ${SPLIT}"
echo "  Trials:     ${NUM_TRIALS}"
echo "  Server:     ${POLICY_HOST}:${POLICY_PORT}"
echo "  Log dir:    ${LOG_DIR}"
echo "  Log videos: ${LOG_VIDEOS}"
echo "  Model fps:  ${MODEL_ACTION_FPS} (from ${POLICY_CONFIG})"
echo "  Env fps:    ${ENV_ACTION_FPS}"
echo ""

mkdir -p "${LOG_DIR}"

LOG_VIDEOS_FLAG=""
if [ "${LOG_VIDEOS}" = true ]; then
    LOG_VIDEOS_FLAG="--log-videos"
fi

python examples/robocasa/main.py \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --task-set "${TASK_SET}" \
    --split "${SPLIT}" \
    --num-trials "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --replan-steps "${REPLAN_STEPS}" \
    --log-dir "${LOG_DIR}" \
    --model-action-fps "${MODEL_ACTION_FPS}" \
    --env-action-fps "${ENV_ACTION_FPS}" \
    ${LOG_VIDEOS_FLAG}
