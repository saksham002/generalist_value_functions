#!/bin/bash
# Sim xArm packing policy server launcher. Delegates to scripts/run_on_tpu.py.
# Worker 0 binds the websocket; workers 1..N-1 participate in the JIT collective.
#
# Prompt routing: the policy (sim_xarm_packing_pi05_subtask, prompt_mode
# "subtask") is conditioned on the client-sent `prompt`, which the eval client
# fills with the critic's AR-decoded subtask. The critic (prompt_mode
# "task_description_predict_current_subtask") reads the per-call
# `task_description` the client sends, so no constant --task-description flag
# is passed here.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
TPU_TYPE="${TPU_TYPE:-v4-64}"
TPU_NAME="${TPU_NAME:-v4-64-0}"
PORT=8000
# Set EXPECT_CRITIC_IMAGES=true to have the server consume obs["critic_image"]
# from the client and feed those into the critic's Observation.
EXPECT_CRITIC_IMAGES="${EXPECT_CRITIC_IMAGES:-false}"

POLICY_CONFIG="${POLICY_CONFIG:-sim_xarm_packing_pi05_subtask}"
# usc2 mirror (v4-64-0 lives in us-central2-b; euw4 originals at the same
# relative path — route any re-mirroring through a local copy, never a direct
# cross-region bucket-to-bucket cp).
POLICY_DIR="${POLICY_DIR:-gs://saksham-usc2/checkpoints/robocoin/pi05_finetune/${POLICY_CONFIG}/${POLICY_CONFIG}}"
POLICY_STEP="${POLICY_STEP:-69999}"

# Set CRITIC_ENABLE=false to serve the policy without BestOfN.
CRITIC_ENABLE="${CRITIC_ENABLE:-true}"
CRITIC_CONFIG="${CRITIC_CONFIG:-robocoin_bimanual_paligemma_cql_rlds_subtask_ar}"
CRITIC_FT_CONFIG="${CRITIC_FT_CONFIG:-sim_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar}"
CRITIC_DIR="${CRITIC_DIR:-gs://saksham-usc2/checkpoints/robocoin/value_functions/Q/${CRITIC_CONFIG}/${CRITIC_CONFIG}/${CRITIC_FT_CONFIG}}"
CRITIC_STEP="${CRITIC_STEP:-250000}"
# BC-only phase: critic enabled (for subtask decoding + Q logging) but a single
# policy sample, so action selection is plain BC. Raise for BestOfN phases.
NUM_SAMPLES="${NUM_SAMPLES:-1}"
# Subtask AR decode cadence (inference calls between re-decodes). Only used by
# predict_subtask_ar critics; ignored by non-AR critics.
SUBTASK_DECODE_EVERY="${SUBTASK_DECODE_EVERY:-4}"
# Set SAMPLE_PARALLEL=true to enable BestOfN sample-parallel mode. FSDP_DEVICES
# pins the inference mesh's fsdp axis; keep it at the trained fsdp=16 so every
# host compiles the known topology (fsdp=64 hit cross-host XLA-compilation
# nondeterminism on v5e-64). num_samples not a multiple of BATCH_AXIS
# (= device_count/fsdp_devices = 4, e.g. the n=1 BC-only phase) is padded up
# server-side; the extra candidates are discarded host-side.
SAMPLE_PARALLEL="${SAMPLE_PARALLEL:-true}"
FSDP_DEVICES="${FSDP_DEVICES:-16}"
# Set INJECT_NOISE=true to switch BestOfN to "sample 1 action + N Gaussian
# perturbations", scored by the critic. NOISE_LEVEL is the eps stddev (in
# policy-normalized space). Defaults reproduce the standard BestOfN behavior.
INJECT_NOISE="${INJECT_NOISE:-false}"
NOISE_LEVEL="${NOISE_LEVEL:-0.0}"
# Flow-matching integration (Euler) steps for the policy. Empty → model default (10).
NUM_STEPS="${NUM_STEPS:-}"
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# tyro requires each subcommand selector to be IMMEDIATELY followed by its own flags.
POLICY_BLOCK="policy:checkpoint --policy.config ${POLICY_CONFIG} --policy.dir ${POLICY_DIR} --policy.step ${POLICY_STEP}"
if [ "${CRITIC_ENABLE}" = true ]; then
    PRE_POLICY_FLAGS=""
    EXPECT_CRITIC_IMAGES_FLAG=""
    if [ "${EXPECT_CRITIC_IMAGES}" = true ]; then
        EXPECT_CRITIC_IMAGES_FLAG="--critic.expect-critic-images"
    fi
    SAMPLE_PARALLEL_FLAG=""
    if [ "${SAMPLE_PARALLEL}" = true ]; then
        SAMPLE_PARALLEL_FLAG="--critic.sample-parallel"
    fi
    FSDP_DEVICES_FLAG=""
    if [ -n "${FSDP_DEVICES}" ]; then
        FSDP_DEVICES_FLAG="--critic.fsdp-devices ${FSDP_DEVICES}"
    fi
    INJECT_NOISE_FLAG=""
    if [ "${INJECT_NOISE}" = true ]; then
        INJECT_NOISE_FLAG="--critic.inject-noise --critic.noise-level ${NOISE_LEVEL}"
    fi
    CRITIC_BLOCK="critic:critic-args --critic.config ${CRITIC_CONFIG} --critic.dir ${CRITIC_DIR} --critic.step ${CRITIC_STEP} --critic.fine-tune-config ${CRITIC_FT_CONFIG} --critic.num-samples ${NUM_SAMPLES} --critic.subtask-decode-every ${SUBTASK_DECODE_EVERY} ${EXPECT_CRITIC_IMAGES_FLAG} ${SAMPLE_PARALLEL_FLAG} ${FSDP_DEVICES_FLAG} ${INJECT_NOISE_FLAG}"
else
    # --use-bestofn-loader routes through BestOfNPolicy so the action-dim slice runs before Unnormalize.
    BC_NOISE_FLAG=""
    if [ "${INJECT_NOISE}" = true ]; then
        BC_NOISE_FLAG="--inject-noise --noise-level ${NOISE_LEVEL}"
    fi
    PRE_POLICY_FLAGS="--use-bestofn-loader ${BC_NOISE_FLAG}"
    CRITIC_BLOCK=""
fi

NUM_STEPS_FLAG=""
if [ -n "${NUM_STEPS}" ]; then
    NUM_STEPS_FLAG="--num-steps ${NUM_STEPS}"
fi

SERVE_INVOCATION="python scripts/serve_policy.py --port ${PORT} ${NUM_STEPS_FLAG} ${PRE_POLICY_FLAGS} ${POLICY_BLOCK} ${CRITIC_BLOCK}"
COMMAND="${SERVE_INVOCATION}"

echo "================================================"
echo "Sim xArm packing policy server (via run_on_tpu.py)"
echo "================================================"
echo "  TPU:         ${TPU_NAME} (${TPU_TYPE}, worker 0 only)"
echo "  Port:        ${PORT}"
echo "  Policy cfg:  ${POLICY_CONFIG}"
echo "  Policy dir:  ${POLICY_DIR}"
echo "  Policy step: ${POLICY_STEP}"
echo "  Num steps:   ${NUM_STEPS:-default(10)}"
if [ "${CRITIC_ENABLE}" = true ]; then
    echo "  Critic cfg:  ${CRITIC_CONFIG}"
    echo "  Critic dir:  ${CRITIC_DIR}"
    echo "  Critic step: ${CRITIC_STEP}"
    echo "  Critic FT:   ${CRITIC_FT_CONFIG}"
    echo "  Num samples: ${NUM_SAMPLES}"
    echo "  Decode every:${SUBTASK_DECODE_EVERY}"
    echo "  Sample par.: ${SAMPLE_PARALLEL}"
    echo "  Inj noise:   ${INJECT_NOISE} (level=${NOISE_LEVEL})"
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
