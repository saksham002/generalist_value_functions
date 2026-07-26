#!/bin/bash
# Sim eval client for sim_xarm_packing. Connects to a remote serve_policy.py
# and steps the local MuJoCo env (PackingEnv, 60 Hz).

set -e

# =============================================================================
# Configuration
# =============================================================================
# Server runs on the TPU worker with jax process_index=0. Default: v4-64-0's
# rank-0 worker (gcloud worker 0) external IP — see project_status/tpu_guide.md.
# Re-probe (grep process_index=0 in ~/tpu_job_output.log across workers) and
# override after any pod change/recreation.
POLICY_HOST="${POLICY_HOST:-35.186.49.73}"
POLICY_PORT=8000

# Path to the vla_exploration packing sim: <checkout>/RAC/dual_xarms/dual_xarms_sim
# (the directory CONTAINING the dual_xarms_sim package). Prepended to PYTHONPATH
# so the packing version of dual_xarms_sim shadows any installed copy.
PACKING_RAC_PATH="${PACKING_RAC_PATH:-/data/group_data/rl/saksham3/vla_exploration/RAC/dual_xarms/dual_xarms_sim}"

POLICY_CONFIG="${POLICY_CONFIG:-sim_xarm_packing_pi05_subtask}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED=86
# Per-episode goal + instruction specs (see examples/sim_xarm_packing/main.py
# docstring for the JSON schema). Needs >= NUM_TRIALS entries.
EPISODES_JSON="${EPISODES_JSON:-examples/sim_xarm_packing/eval_episodes.json}"
# Pure base dir — NO task in it (the env names the task). Final tree:
#   policy-only: <LOG_DIR>/Packing/<POLICY_CONFIG>/bc[_noise<level>]/
#   best-of-N:   <LOG_DIR>/Packing/<POLICY_CONFIG>/<CRITIC_FT_CONFIG>_<CRITIC_STEP>/best_of_n_<N>[_noise<level>]/
# Each layout terminates in {videos/, stats.json, eval.log}. The eval client
# refuses to overwrite an existing populated dir (raises FileExistsError).
LOG_DIR="/data/group_data/rl/saksham3/eval_logs/sim_xarm_packing"
# Mirror these to serve_policy_sim_xarm_packing.sh for the current phase.
POLICY_STEP="${POLICY_STEP:-69999}"
CRITIC_ENABLE="${CRITIC_ENABLE:-true}"
# BC-only phase: critic enabled (subtask decoding + Q logging) but 1 sample.
NUM_SAMPLES="${NUM_SAMPLES:-1}"
# Subtask AR decode cadence — must mirror serve_policy_sim_xarm_packing.sh.
# Only affects the log-path discriminator below (server controls actual cadence).
SUBTASK_DECODE_EVERY="${SUBTASK_DECODE_EVERY:-4}"
CRITIC_STEP="${CRITIC_STEP:-250000}"
CRITIC_FT_CONFIG="${CRITIC_FT_CONFIG:-sim_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar}"
INJECT_NOISE="${INJECT_NOISE:-false}"
NOISE_LEVEL="${NOISE_LEVEL:-0.0}"
NOISE_SUFFIX=""
if [ "${INJECT_NOISE}" = true ]; then
    NOISE_SUFFIX="_${NOISE_LEVEL}"
fi
# A/B knob: env deltas against the measured TCP instead of the mocap target
# (dataset actions are next-step mocap targets, so false is the
# training-consistent default). Adds an _sdelta method-path discriminator.
STATE_RELATIVE_DELTAS="${STATE_RELATIVE_DELTAS:-false}"
DELTA_SUFFIX=""
STATE_RELATIVE_FLAG=""
if [ "${STATE_RELATIVE_DELTAS}" = true ]; then
    DELTA_SUFFIX="_sdelta"
    STATE_RELATIVE_FLAG="--state-relative-deltas"
fi
# Flow-matching integration steps — must mirror serve_policy_sim_xarm_packing.sh.
# Empty → model default (10); only affects the log-path discriminator below.
NUM_STEPS="${NUM_STEPS:-}"
STEPS_SUFFIX=""
if [ -n "${NUM_STEPS}" ]; then
    STEPS_SUFFIX="_steps${NUM_STEPS}"
fi
if [ "${CRITIC_ENABLE}" = true ]; then
    # "gtsub" = the policy conditions on the client-computed ground-truth
    # subtask (the critic still AR-decodes for its value prompt).
    if [ "${NUM_SAMPLES}" = 1 ]; then
        # Single sample == plain BC action selection (critic only decodes the
        # subtask + logs Q).
        METHOD="${POLICY_CONFIG}/${CRITIC_FT_CONFIG}_${CRITIC_STEP}/bc_gtsub${DELTA_SUFFIX}_de${SUBTASK_DECODE_EVERY}${NOISE_SUFFIX}${STEPS_SUFFIX}"
    else
        METHOD="${POLICY_CONFIG}/${CRITIC_FT_CONFIG}_${CRITIC_STEP}/best_of_n_${NUM_SAMPLES}_gtsub${DELTA_SUFFIX}_de${SUBTASK_DECODE_EVERY}${NOISE_SUFFIX}${STEPS_SUFFIX}"
    fi
else
    METHOD="${POLICY_CONFIG}/bc${NOISE_SUFFIX}${STEPS_SUFFIX}"
fi
LOG_VIDEOS=true
# Per-episode predictions JSON (per-infer-call Q values + decoded subtasks).
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"
# Scene variant — must match the visuals the training data was rendered with.
SCENE_VARIANT="${SCENE_VARIANT:-ycb_visual}"
TIME_LIMIT_S="${TIME_LIMIT_S:-400}"
# Env cadence (env_control_freq / replan_steps) is derived inside the python
# client from the server's policy.metadata (policy_subsample).
# Resume options. To resume an interrupted eval: set START_EPISODE_IDX to the
# 0-indexed episode to start from, and RESUME_FROM_DIR to the populated run dir.
START_EPISODE_IDX="${START_EPISODE_IDX:-0}"
RESUME_FROM_DIR="${RESUME_FROM_DIR:-}"
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "${PACKING_RAC_PATH}/dual_xarms_sim" ]; then
    echo "ERROR: PACKING_RAC_PATH=${PACKING_RAC_PATH} does not contain a dual_xarms_sim package." >&2
    echo "Point it at <vla_exploration checkout>/RAC/dual_xarms/dual_xarms_sim." >&2
    exit 1
fi

# Activate the sim conda env (MuJoCo + mink + gymnasium).
# shellcheck disable=SC1091
source /home/saksham3/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/saksham3/conda-envs/sim_bimanual_assembly

export PACKING_RAC_PATH
export PYTHONPATH="${PACKING_RAC_PATH}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

echo "================================================"
echo "Sim xArm packing eval"
echo "================================================"
echo "  Trials:     ${NUM_TRIALS}"
echo "  Episodes:   ${EPISODES_JSON}"
echo "  Server:     ${POLICY_HOST}:${POLICY_PORT}"
echo "  Log dir:    ${LOG_DIR}"
echo "  Method:     ${METHOD}"
echo "  Log videos: ${LOG_VIDEOS}"
echo "  Scene:      ${SCENE_VARIANT}"
echo "  Sim path:   ${PACKING_RAC_PATH}"
echo "  Cadence:    (client-side, derived from server metadata)"
if [ -n "${RESUME_FROM_DIR}" ]; then
    echo "  Resume:     start_idx=${START_EPISODE_IDX} dir=${RESUME_FROM_DIR}"
fi
echo ""

LOG_VIDEOS_FLAG=""
if [ "${LOG_VIDEOS}" = true ]; then
    LOG_VIDEOS_FLAG="--log-videos"
fi

SAVE_PREDICTIONS_FLAG=""
if [ "${SAVE_PREDICTIONS}" = true ]; then
    SAVE_PREDICTIONS_FLAG="--save-predictions"
fi

RESUME_FLAGS=""
if [ -n "${RESUME_FROM_DIR}" ]; then
    RESUME_FLAGS="--resume-from-dir ${RESUME_FROM_DIR}"
fi

python examples/sim_xarm_packing/main.py \
    --host "${POLICY_HOST}" \
    --port "${POLICY_PORT}" \
    --episodes-json "${EPISODES_JSON}" \
    --num-trials "${NUM_TRIALS}" \
    --time-limit-s "${TIME_LIMIT_S}" \
    --scene-variant "${SCENE_VARIANT}" \
    --seed "${SEED}" \
    --log-dir "${LOG_DIR}" \
    --method "${METHOD}" \
    --start-episode-idx "${START_EPISODE_IDX}" \
    ${RESUME_FLAGS} \
    ${LOG_VIDEOS_FLAG} \
    ${SAVE_PREDICTIONS_FLAG} \
    ${STATE_RELATIVE_FLAG}
