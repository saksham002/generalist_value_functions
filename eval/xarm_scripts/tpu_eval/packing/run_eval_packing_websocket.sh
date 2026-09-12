#!/bin/bash
# Shirt-hang eval client. Connects to a remote serve_policy.py (TPU) and the
# physical robot environment server, then runs episodes.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
# External IP of the rank-0 TPU worker (from the "Creating server" line in the
# server startup logs). Re-check per launch — rank-0 placement can change.
POLICY_HOST="35.186.17.145"  # v4-64-0 worker 5
POLICY_PORT=8000
ROBOT_HOST="xarmpc.pc.cs.cmu.edu"
ROBOT_PORT=8080

# The tasks file holds 24 tasks (8 per box pairing) and task idx = episode idx, so a full
# sweep is episodes 0-23. Going past 23 raises IndexError in load_packing_task.
NUM_EPISODES=6
START_EPISODE_IDX=18
CONTROL_FREQ=60
QUERY_FREQ=30
# Per-episode budget = SECONDS_PER_SUBTASK x (number of subtasks in the task) x CONTROL_FREQ.
# MAX_STEPS is only the fallback for episodes with no subtask list (i.e. no tasks file).
SECONDS_PER_SUBTASK=25
MAX_STEPS=7200

HAS_CRITIC=true
NUM_SAMPLES=32
# true: critic conditioned on per-step subtask prompts; false: critic gets TASK_DESCRIPTION (same prompt as the policy).
# Requires a critic served with a prompt_mode that reads the client prompt (NOT task_description_predict_current_subtask).
USE_CRITIC_SUBTASKS=true
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

# Task set loaded per-episode (task idx = episode idx). packing_tasks_harder.json = the harder
# 5-8 subtask set; packing_tasks.json = the original 3-5 subtask set. Resolved to an absolute
# path (relative to this script) below.
TASKS_FILE="packing_tasks_harder.json"

LOG_VIDEOS=true
VIDEO_SUBDIR="BestOfN N32 sarsa_subtask_ar critic 250k, realworld_xarm_packing_pi05_subtask 70k, harder tasks" # optional subdir under eval/xarm_scripts/tpu_eval/packing/videos/ (empty = save directly there)
DEBUG_VALUES=false
MANUAL=true
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TASKS_FILE_PATH="${SCRIPT_DIR}/${TASKS_FILE}"
cd "${REPO_ROOT}"

echo "================================================"
echo "Shirt-hang WebSocket eval client"
echo "================================================"
echo "  Policy server: ${POLICY_HOST}:${POLICY_PORT}"
echo "  Robot server:  ${ROBOT_HOST}:${ROBOT_PORT}"
echo "  Episodes:      ${NUM_EPISODES} (starting at ${START_EPISODE_IDX})"
echo "  Tasks:         per-episode from ${TASKS_FILE} (task idx = episode idx)"
echo "  Control freq:  ${CONTROL_FREQ} Hz"
echo "  Episode budget: ${SECONDS_PER_SUBTASK}s per subtask x CONTROL_FREQ (fallback ${MAX_STEPS} steps)"
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
    --args.seconds-per-subtask "${SECONDS_PER_SUBTASK}" \
    --args.num-samples "${NUM_SAMPLES}" \
    --args.video-subdir "${VIDEO_SUBDIR}" \
    --args.task-description "${TASK_DESCRIPTION}" \
    --args.tasks-file "${TASKS_FILE_PATH}" \
    ${ARGS}
