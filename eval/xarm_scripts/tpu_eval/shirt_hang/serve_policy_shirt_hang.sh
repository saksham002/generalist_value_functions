#!/bin/bash
# RoboCasa policy server launcher. Delegates to scripts/run_on_tpu.py.
# Worker 0 binds the websocket; workers 1..N-1 participate in the JIT collective.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
TPU_TYPE="v5e-64"
TPU_NAME="v5e-tpu-64-1"
PORT=8005

POLICY_CONFIG="real_shirt_hang_pi05"
POLICY_DIR="gs://saksham-euw4/checkpoints/robocoin/pi05_finetune/real_shirt_hang_pi05/real_shirt_hang_pi05"
POLICY_STEP=60000

# Set CRITIC_ENABLE=false to serve the policy without BestOfN.
CRITIC_ENABLE=false
CRITIC_CONFIG="robocoin_bimanual_paligemma_cql_rlds"
CRITIC_DIR="gs://saksham-euw4/checkpoints/robocoin/value_functions/Q/robocoin_bimanual_paligemma_cql_rlds/robocoin_bimanual_paligemma_cql_rlds/real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final"
CRITIC_STEP=240000
CRITIC_FT_CONFIG="real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final"
NUM_SAMPLES=8
# AR-subtask decode cadence: the critic AR-decodes the predicted subtask once every
# N critic calls (and conditions Q on it). Experiment with 4 and 10.
#SUBTASK_DECODE_EVERY=10
# When true, pass --critic.expect-critic-images: the critic consumes a separate obs["critic_image"]
# stream (the eval client must send it; needed when the critic's image size/pipeline differs from the
# policy, e.g. a gemma4 critic). When false, the critic reuses the policy's obs["image"].
EXPECT_CRITIC_IMAGES=false
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)" #Should resolve to batch_value_learning

# tyro requires each subcommand selector to be IMMEDIATELY followed by its own flags.
POLICY_BLOCK="policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${POLICY_DIR} --policy.step ${POLICY_STEP}"
if [ "${CRITIC_ENABLE}" = true ]; then
    PRE_POLICY_FLAGS=""
    if [ "${EXPECT_CRITIC_IMAGES}" = true ]; then
        CRITIC_IMAGES_FLAG="--critic.expect-critic-images"
    else
        CRITIC_IMAGES_FLAG=""
    fi
    # Omit --critic.fine-tune-config when empty (CriticArgs.fine_tune_config defaults
    # to None); passing the flag with no value is a tyro parse error.
    if [ -n "${CRITIC_FT_CONFIG}" ]; then
        CRITIC_FT_FLAG="--critic.fine-tune-config ${CRITIC_FT_CONFIG}"
    else
        CRITIC_FT_FLAG=""
    fi
    CRITIC_BLOCK="critic:critic-args --critic.config ${CRITIC_CONFIG} --critic.dir ${CRITIC_DIR} --critic.step ${CRITIC_STEP} ${CRITIC_FT_FLAG} --critic.num-samples ${NUM_SAMPLES} --critic.sample-parallel --critic.fsdp-devices 16 ${CRITIC_IMAGES_FLAG}"
else
    # --use-bestofn-loader routes through BestOfNPolicy so the action-dim slice runs before Unnormalize.
    PRE_POLICY_FLAGS="--use-bestofn-loader"
    CRITIC_BLOCK=""
fi

TASK_DESCRIPTION="Place the shirt on the hanger and hang it from the rod."

SERVE_INVOCATION="python scripts/serve_policy.py --port ${PORT} --task-description '${TASK_DESCRIPTION}' ${PRE_POLICY_FLAGS} ${POLICY_BLOCK} ${CRITIC_BLOCK}"
COMMAND="${SERVE_INVOCATION}"

echo "================================================"
echo "RoboCasa policy server (via run_on_tpu.py)"
echo "================================================"
echo "  TPU:         ${TPU_NAME} (${TPU_TYPE}, worker 0 only)"
echo "  Port:        ${PORT}"
echo "  Policy cfg:  ${POLICY_CONFIG}"
echo "  Policy dir:  ${POLICY_DIR}"
echo "  Policy step: ${POLICY_STEP}"
if [ "${CRITIC_ENABLE}" = true ]; then
    echo "  Critic cfg:  ${CRITIC_CONFIG}"
    echo "  Critic dir:  ${CRITIC_DIR}"
    echo "  Critic step: ${CRITIC_STEP}"
    echo "  Critic FT:   ${CRITIC_FT_CONFIG}"
    echo "  Num samples: ${NUM_SAMPLES}"
fi
echo ""
echo "Health check:    curl http://<tpu-worker-0-ip>:${PORT}/healthz"
echo "Reattach to job: gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=europe-west4-b --worker=0 --command=\"tmux attach -t job\""
echo "Local terminal:  ctrl-C to release; the remote server keeps running."
echo ""

cd "${REPO_ROOT}"
exec python scripts/run_on_tpu.py \
    --tpu-type "${TPU_TYPE}" \
    --tpu-name "${TPU_NAME}" \
    --nfs-user jeffyu \
    --command "${COMMAND}"
