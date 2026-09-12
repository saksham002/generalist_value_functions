#!/bin/bash
# Lego eval client. Connects to a remote serve_policy.py (TPU) and the local YAM
# bimanual environment (gello ZMQ nodes must already be running: launch_nodes.sh),
# then runs episodes.
#
# Runs in this repo's uv env; yam_teleop (the ZMQ robot env) is installed into it with
#   uv pip install pyzmq && uv pip install --no-deps -e ~/gello-yam-teleop/yam_teleop

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
# External IP of the rank-0 TPU worker (from the "Creating server" line in the
# server startup logs). Re-check per launch — rank-0 placement can change.
POLICY_HOST="35.186.17.226"
POLICY_PORT=8000
ENV_CONFIG="${HOME}/gello-yam-teleop/yam_teleop/configs/env.yaml"

# 24 tasks in lego_tasks.json; task idx = episode idx.
NUM_EPISODES=24
START_EPISODE_IDX=0
QUERY_FREQ=30
MAX_STEPS=9000

# Per-episode task description + ordered subtask prompts (Enter advances to the next).
# Empty falls back to lego_tasks.json next to the client script.
TASKS_FILE=""
NO_RESET=false
CONTROL_FREQ=60
# Videos go to eval/yam_scripts/lego/videos/${VIDEO_SUBDIR}/episode_<n>.mp4.
LOG_VIDEOS=true
VIDEO_SUBDIR="BoN_2"
# Match the server: HAS_CRITIC=true only when CRITIC_ENABLE=true in serve_policy_lego.sh.
HAS_CRITIC=true
NUM_SAMPLES=8
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "================================================"
echo "Lego (YAM) WebSocket eval client"
echo "================================================"
echo "  Policy server: ${POLICY_HOST}:${POLICY_PORT}"
echo "  Env config:    ${ENV_CONFIG}"
echo "  Episodes:      ${NUM_EPISODES} (starting at ${START_EPISODE_IDX})"
echo "  Query freq:    ${QUERY_FREQ} steps"
echo "  Max steps:     ${MAX_STEPS}"
echo "  Tasks:         per-episode from ${TASKS_FILE:-lego_tasks.json} (task idx = episode idx)"
echo "  No reset:      ${NO_RESET}"
echo "  Has critic:    ${HAS_CRITIC}"
echo "  Log videos:    ${LOG_VIDEOS} (videos/${VIDEO_SUBDIR})"
echo ""

ARGS=""
[ "${NO_RESET}" = true ]     && ARGS="${ARGS} --args.no-reset"
[ "${HAS_CRITIC}" = true ]   && ARGS="${ARGS} --args.has-critic"
[ "${HAS_CRITIC}" = false ]  && ARGS="${ARGS} --args.no-has-critic"
[ "${LOG_VIDEOS}" = true ]   && ARGS="${ARGS} --args.log-videos"
[ "${LOG_VIDEOS}" = false ]  && ARGS="${ARGS} --args.no-log-videos"
[ -n "${TASKS_FILE}" ]   && ARGS="${ARGS} --args.tasks-file ${TASKS_FILE}"

cd "${REPO_ROOT}"
uv run eval/yam_scripts/lego/eval_lego_websocket.py \
    --args.policy-host "${POLICY_HOST}" \
    --args.policy-port "${POLICY_PORT}" \
    --args.env-config "${ENV_CONFIG}" \
    --args.num-episodes "${NUM_EPISODES}" \
    --args.start-episode-idx "${START_EPISODE_IDX}" \
    --args.query-freq "${QUERY_FREQ}" \
    --args.max-steps "${MAX_STEPS}" \
    --args.control-freq "${CONTROL_FREQ}" \
    --args.num-samples "${NUM_SAMPLES}" \
    --args.video-subdir "${VIDEO_SUBDIR}" \
    ${ARGS}
