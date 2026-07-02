#!/bin/bash
# Shirt-hang eval client. Connects to a remote serve_policy.py (TPU) and the
# physical robot environment server, then runs episodes.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
# External IP of the rank-0 TPU worker (from the "Creating server" line in the
# server startup logs). Re-check per launch — rank-0 placement can change.
POLICY_HOST="34.12.73.186"
POLICY_PORT=8005
ROBOT_HOST="xarmpc.pc.cs.cmu.edu"
ROBOT_PORT=8080

NUM_EPISODES=50
START_EPISODE_IDX=15
CONTROL_FREQ=60
QUERY_FREQ=30
MAX_STEPS=7200

HAS_CRITIC=true
NUM_SAMPLES=8
# true: critic conditioned on per-step subtask prompts; false: critic gets TASK_DESCRIPTION (same prompt as the policy).
# Requires a critic served with a prompt_mode that reads the client prompt (NOT task_description_predict_current_subtask).
USE_CRITIC_SUBTASKS=false
# true: POLICY conditioned on the per-step subtask prompts from packing_tasks.json ("subtasks"),
# advanced by pressing Enter; false: policy conditioned on the full TASK_DESCRIPTION.
# Requires a policy whose prompt_mode reads the client prompt (do NOT pass --task-description to
# serve_policy.py, or the server would pin the policy prompt and this would have no effect).
USE_POLICY_SUBTASKS=true
# Each episode loads its own task from packing_tasks.json, keyed on its episode index (0-7
# small+medium, 8-15 small+large, 16-23 medium+large). Set START_EPISODE_IDX to pick which task
# the run begins on; each subsequent episode advances to the next task.
# Fallback prompt used only if packing_tasks.json is missing; otherwise overridden per episode.
TASK_DESCRIPTION=""

LOG_VIDEOS=true
VIDEO_SUBDIR="redo realworld_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar, 250k steps"   # optional subdir under eval/xarm_scripts/tpu_eval/packing/videos/ (empty = save directly there)
DEBUG_VALUES=false
MANUAL=false
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================"
echo "Shirt-hang WebSocket eval client"
echo "================================================"
echo "  Policy server: ${POLICY_HOST}:${POLICY_PORT}"
echo "  Robot server:  ${ROBOT_HOST}:${ROBOT_PORT}"
echo "  Episodes:      ${NUM_EPISODES} (starting at ${START_EPISODE_IDX})"
echo "  Tasks:         per-episode from packing_tasks.json (task idx = episode idx)"
echo "  Control freq:  ${CONTROL_FREQ} Hz"
echo "  Query freq:    ${QUERY_FREQ} steps"
echo "  Has critic:    ${HAS_CRITIC}"
echo "  Num samples:   ${NUM_SAMPLES}"
echo "  Critic prompt: $([ "${USE_CRITIC_SUBTASKS}" = true ] && echo subtasks || echo "task description")"
echo "  Policy prompt: $([ "${USE_POLICY_SUBTASKS}" = true ] && echo "subtasks (Enter to advance)" || echo "task description")"
echo "  Manual:        ${MANUAL}"
echo "  Log videos:    ${LOG_VIDEOS}"
echo ""

ARGS=""
[ "${HAS_CRITIC}" = true ]          && ARGS="${ARGS} --args.has-critic"
[ "${HAS_CRITIC}" = false ]         && ARGS="${ARGS} --args.no-has-critic"
[ "${LOG_VIDEOS}" = true ]          && ARGS="${ARGS} --args.log-videos"
[ "${DEBUG_VALUES}" = true ]        && ARGS="${ARGS} --args.debug-values"
[ "${MANUAL}" = true ]              && ARGS="${ARGS} --args.manual"
[ "${USE_CRITIC_SUBTASKS}" = true ]  && ARGS="${ARGS} --args.use-critic-subtasks"
[ "${USE_CRITIC_SUBTASKS}" = false ] && ARGS="${ARGS} --args.no-use-critic-subtasks"
[ "${USE_POLICY_SUBTASKS}" = true ]  && ARGS="${ARGS} --args.use-policy-subtasks"
[ "${USE_POLICY_SUBTASKS}" = false ] && ARGS="${ARGS} --args.no-use-policy-subtasks"

uv run eval/xarm_scripts/tpu_eval/packing/eval_packing_websocket.py \
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
    --args.video-subdir "${VIDEO_SUBDIR}" \
    --args.task-description "${TASK_DESCRIPTION}" \
    ${ARGS}
