#!/bin/bash
# RoboCasa policy server launcher. Delegates to scripts/run_on_tpu.py.
# Worker 0 binds the websocket; workers 1..N-1 participate in the JIT collective.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
TPU_TYPE="v5e-32"
TPU_NAME="v5e-tpu-32-0"
PORT=8000

POLICY_CONFIG="robocasa_pi05_finetune"
POLICY_DIR="gs://saksham-euw4/checkpoints/robocoin/pi05_finetune/robocasa_pi05_finetune/robocasa_pi05_finetune"
POLICY_STEP=50000

# Set CRITIC_ENABLE=false to serve the policy without BestOfN.
CRITIC_ENABLE=true
CRITIC_CONFIG="robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta"
CRITIC_DIR="gs://saksham-euw4/checkpoints/robocoin/value_functions/Q/robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta/robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta/robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta"
CRITIC_STEP=235000
CRITIC_FT_CONFIG="robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta"
NUM_SAMPLES=64
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# tyro requires each subcommand selector to be IMMEDIATELY followed by its own flags.
POLICY_BLOCK="policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${POLICY_DIR} --policy.step ${POLICY_STEP}"
if [ "${CRITIC_ENABLE}" = true ]; then
    PRE_POLICY_FLAGS=""
    CRITIC_BLOCK="critic:critic-args --critic.config ${CRITIC_CONFIG} --critic.dir ${CRITIC_DIR} --critic.step ${CRITIC_STEP} --critic.fine-tune-config ${CRITIC_FT_CONFIG} --critic.num-samples ${NUM_SAMPLES}"
else
    # --use-bestofn-loader routes through BestOfNPolicy so the action-dim slice runs before Unnormalize.
    PRE_POLICY_FLAGS="--use-bestofn-loader"
    CRITIC_BLOCK=""
fi

SERVE_INVOCATION="python scripts/serve_policy.py --port ${PORT} ${PRE_POLICY_FLAGS} ${POLICY_BLOCK} ${CRITIC_BLOCK}"
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
    --command "${COMMAND}"
