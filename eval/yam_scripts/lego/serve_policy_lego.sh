#!/bin/bash
# Lego (YAM bimanual) policy server launcher. Delegates to scripts/run_on_tpu.py.
# Worker 0 binds the websocket; workers 1..N-1 participate in the JIT collective.
#
# Local prerequisites (see CLAUDE.md > "Running on TPU Pods"): a Python with `tyro` and
# `requests` (LAUNCHER_PYTHON, default ~/.venvs/tpu-launcher/bin/python) and the Gemma
# helper checkout next to the repo (~/gemma), which the launcher syncs to the pod.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================
TPU_TYPE="v4-64"
TPU_NAME="v4-64-0"
TPU_ZONE="us-central2-b"
PORT=8000

# lego_pi05_subtask is a joint-space pi-0.5 policy (14D: 2 x (6 joints + gripper)) with
# prompt_mode="subtask": the eval client sends the current subtask as obs["prompt"].
POLICY_CONFIG="lego_pi05_subtask"
POLICY_DIR="gs://saksham-usc2/checkpoints/robocoin/pi05_finetune/lego_pi05_subtask/lego_pi05_subtask"
POLICY_STEP=19999

# Set CRITIC_ENABLE=true to serve BestOfN with the EEF subtask_ar critic (steps 240000
# and 250000, mirrored from saksham-euw4). The td240 variant lives next to it under
# .../lego_paligemma_cql_rlds_finetune_subtask_ar_td240/250000.
CRITIC_ENABLE=true
CRITIC_CONFIG="robocoin_bimanual_paligemma_cql_rlds_subtask_ar"
CRITIC_DIR="gs://saksham-usc2/checkpoints/robocoin/value_functions/Q/robocoin_bimanual_paligemma_cql_rlds_subtask_ar/robocoin_bimanual_paligemma_cql_rlds_subtask_ar/lego_paligemma_cql_rlds_finetune_subtask_ar"
CRITIC_STEP=240000
CRITIC_FT_CONFIG="lego_paligemma_cql_rlds_finetune_subtask_ar"
NUM_SAMPLES=8
# AR-subtask decode cadence: the critic AR-decodes the predicted subtask once every
# N critic calls (and conditions Q on it).
SUBTASK_DECODE_EVERY=4
# When true, pass --critic.expect-critic-images: the critic consumes a separate obs["critic_image"]
# stream (the eval client must send it; needed when the critic's image size/pipeline differs from the
# policy, e.g. a gemma4 critic). When false, the critic reuses the policy's obs["image"].
EXPECT_CRITIC_IMAGES=false

# Task description for the critic's task_description_predict_current_subtask prompt. The
# policy ignores it (subtask prompt mode). Leave empty to require the eval client to send
# obs["task_description"] per call instead (a subtask-decoding critic fails loudly without one).
TASK_DESCRIPTION=""
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LAUNCHER_PYTHON="${LAUNCHER_PYTHON:-$HOME/.venvs/tpu-launcher/bin/python}"

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
    CRITIC_BLOCK="critic:critic-args --critic.config ${CRITIC_CONFIG} --critic.dir ${CRITIC_DIR} --critic.step ${CRITIC_STEP} ${CRITIC_FT_FLAG} --critic.num-samples ${NUM_SAMPLES} --critic.sample-parallel --critic.fsdp-devices 16 --critic.subtask-decode-every ${SUBTASK_DECODE_EVERY} ${CRITIC_IMAGES_FLAG}"
else
    # --use-bestofn-loader routes through BestOfNPolicy so the action-dim slice runs before Unnormalize.
    PRE_POLICY_FLAGS="--use-bestofn-loader"
    CRITIC_BLOCK=""
fi

SERVE_INVOCATION="python scripts/serve_policy.py --port ${PORT} --task-description '${TASK_DESCRIPTION}' ${PRE_POLICY_FLAGS} ${POLICY_BLOCK} ${CRITIC_BLOCK}"
COMMAND="${SERVE_INVOCATION}"

echo "================================================"
echo "Lego (YAM) policy server (via run_on_tpu.py)"
echo "================================================"
echo "  TPU:         ${TPU_NAME} (${TPU_TYPE}, ${TPU_ZONE}, worker 0 binds the port)"
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
else
    echo "  Critic:      disabled"
fi
echo ""
echo "Health check:    curl http://<tpu-worker-0-ip>:${PORT}/healthz"
echo "Reattach to job: gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${TPU_ZONE} --worker=0 --command=\"tmux attach -t job\""
echo "Local terminal:  ctrl-C to release; the remote server keeps running."
echo ""

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${LAUNCHER_PYTHON}" scripts/run_on_tpu.py \
    --tpu-type "${TPU_TYPE}" \
    --tpu-name "${TPU_NAME}" \
    --zone "${TPU_ZONE}" \
    --nfs-user jeffyu \
    --command "${COMMAND}"
