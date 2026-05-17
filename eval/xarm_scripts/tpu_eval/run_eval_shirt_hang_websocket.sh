#!/bin/bash
# Shirt-hang eval client. Connects to a remote serve_policy.py (TPU) and the
# physical robot environment server, then runs episodes.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
# IP of TPU worker 0 (read from serve_policy_shirt_hang.sh startup logs).
POLICY_HOST="localhost"
POLICY_PORT=8005

ROBOT_HOST="xarmpc.pc.cs.cmu.edu"
ROBOT_PORT=8080

NUM_EPISODES=10
START_EPISODE_IDX=0
CONTROL_FREQ=60
QUERY_FREQ=30
MAX_STEPS=7200

HAS_CRITIC=true
NUM_SAMPLES=8

LOG_VIDEOS=false
DEBUG_VALUES=false
MANUAL=true
USE_TASK_DESCRIPTION=false
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================"
echo "Shirt-hang WebSocket eval client"
echo "================================================"
echo "  Policy server: ${POLICY_HOST}:${POLICY_PORT}"
echo "  Robot server:  ${ROBOT_HOST}:${ROBOT_PORT}"
echo "  Episodes:      ${NUM_EPISODES} (starting at ${START_EPISODE_IDX})"
echo "  Control freq:  ${CONTROL_FREQ} Hz"
echo "  Query freq:    ${QUERY_FREQ} steps"
echo "  Has critic:    ${HAS_CRITIC}"
echo "  Num samples:   ${NUM_SAMPLES}"
echo "  Manual:        ${MANUAL}"
echo "  Log videos:    ${LOG_VIDEOS}"
echo ""

ARGS=""
[ "${HAS_CRITIC}" = true ]          && ARGS="${ARGS} --args.has-critic"
[ "${HAS_CRITIC}" = false ]         && ARGS="${ARGS} --args.no-has-critic"
[ "${LOG_VIDEOS}" = true ]          && ARGS="${ARGS} --args.log-videos"
[ "${DEBUG_VALUES}" = true ]        && ARGS="${ARGS} --args.debug-values"
[ "${MANUAL}" = true ]              && ARGS="${ARGS} --args.manual"
[ "${USE_TASK_DESCRIPTION}" = true ] && ARGS="${ARGS} --args.use-task-description"

uv run eval/xarm_scripts/tpu_eval/eval_shirt_hang_websocket.py \
    --args.policy-host "${POLICY_HOST}" \
    --args.policy-port "${POLICY_PORT}" \
    --args.robot-host "${ROBOT_HOST}" \
    --args.robot-port "${ROBOT_PORT}" \
    --args.num-episodes "${NUM_EPISODES}" \
    --args.start-episode-idx "${START_EPISODE_IDX}" \
    --args.control-freq "${CONTROL_FREQ}" \
    --args.query-freq "${QUERY_FREQ}" \
    --args.max-steps "${MAX_STEPS}" \
    --args.num-samples "${NUM_SAMPLES}" \
    ${ARGS}
