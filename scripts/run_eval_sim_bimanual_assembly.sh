#!/bin/bash
# Sim eval client for sim_bimanual_assembly. Connects to a remote serve_policy.py
# and steps the local MuJoCo env (DoubleInsertDualXarmsGymEnv at 60 Hz).

set -e

# =============================================================================
# Configuration
# =============================================================================
# Server runs on the TPU worker with process_index=0 (worker 6 of v5e-tpu-32-1).
POLICY_HOST="34.147.12.116"
POLICY_PORT=8000

POLICY_CONFIG="sim_bimanual_assembly_pi05"
TASK_SET="DoubleInsertDualXarms"
SPLIT="target"
NUM_TRIALS=100
SEED=86
# Pure base dir — NO task in it (the gym env names the task). Final tree:
# <LOG_DIR>/<env>/<METHOD>/<RUN_TAG>/{videos,stats.json,eval.log}.
LOG_DIR="/data/user_data/saksham3/eval_logs/sim_bimanual_assembly"
# Mirror these to serve_policy_sim_bimanual_assembly.sh for the current phase.
# METHOD (the <env>/<METHOD>/ path component) is DERIVED so the run dir encodes
# the policy + critic checkpoints (e.g. bc_p99999 / bestofn8_p99999_c240000).
POLICY_STEP=10000
CRITIC_ENABLE=false
NUM_SAMPLES=8
CRITIC_STEP=240000
if [ "${CRITIC_ENABLE}" = true ]; then
    METHOD="bestofn${NUM_SAMPLES}_p${POLICY_STEP}_c${CRITIC_STEP}"
else
    METHOD="bc_p${POLICY_STEP}"
fi
LOG_VIDEOS=true
REPLAN_STEPS=30
# Resume options. To resume an interrupted eval: set START_EPISODE_IDX to the
# 0-indexed episode to start from, and RESUME_FROM_DIR to the run-date subdir
# (…/evals/<split>/<env>/<YYYY-MM-DD-HH-MM>). Both default to off.
START_EPISODE_IDX=0
RESUME_FROM_DIR=""
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Activate the sim_bimanual_assembly conda env (MuJoCo + dual_xarms_sim).
# shellcheck disable=SC1091
source /home/saksham3/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/saksham3/conda-envs/sim_bimanual_assembly

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

echo "================================================"
echo "Sim bimanual assembly eval"
echo "================================================"
echo "  Task:       ${TASK_SET}"
echo "  Split:      ${SPLIT}"
echo "  Trials:     ${NUM_TRIALS}"
echo "  Server:     ${POLICY_HOST}:${POLICY_PORT}"
echo "  Log dir:    ${LOG_DIR}"
echo "  Log videos: ${LOG_VIDEOS}"
echo "  Replan:     ${REPLAN_STEPS}"
if [ -n "${RESUME_FROM_DIR}" ]; then
    echo "  Resume:     start_idx=${START_EPISODE_IDX} dir=${RESUME_FROM_DIR}"
fi
echo ""

# Date-time leaf, set here in the bash script; the client builds the full
# <LOG_DIR>/<env>/<METHOD>/<RUN_TAG> path (mkdir -p handled client-side).
RUN_TAG="$(date +%Y-%m-%d-%H-%M)"

LOG_VIDEOS_FLAG=""
if [ "${LOG_VIDEOS}" = true ]; then
    LOG_VIDEOS_FLAG="--log-videos"
fi

RESUME_FLAGS=""
if [ -n "${RESUME_FROM_DIR}" ]; then
    RESUME_FLAGS="--resume-from-dir ${RESUME_FROM_DIR}"
fi

python examples/sim_bimanual_assembly/main.py \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --task-set "${TASK_SET}" \
    --split "${SPLIT}" \
    --num-trials "${NUM_TRIALS}" \
    --seed "${SEED}" \
    --replan-steps "${REPLAN_STEPS}" \
    --log-dir "${LOG_DIR}" \
    --method "${METHOD}" \
    --run-tag "${RUN_TAG}" \
    --start-episode-idx "${START_EPISODE_IDX}" \
    ${RESUME_FLAGS} \
    ${LOG_VIDEOS_FLAG}
