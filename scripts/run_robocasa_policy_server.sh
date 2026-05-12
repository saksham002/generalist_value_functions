#!/bin/bash
# Run the openpi WebSocket policy server for RoboCasa eval.
#
# Activates the existing uv venv (where openpi proper lives) and launches
# scripts/serve_policy.py against the chosen TrainConfig + checkpoint. The
# server binds 0.0.0.0:${PORT}; the matching eval client lives in a separate
# conda env and connects via ${HOST}:${PORT}.
#
# Configuration (env vars or CLI args):
#   CONFIG_NAME      Train config name (default: robocasa_pi05_finetune)
#   CHECKPOINT_DIR   Required. Path or gs:// URI to the checkpoint step dir.
#   PORT             Default 8000.
#
# Example:
#   CHECKPOINT_DIR=/data/user_data/saksham3/checkpoints/robocasa_pi05_finetune/.../50000 \
#       ./scripts/run_robocasa_policy_server.sh

set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-robocasa_pi05_finetune_gpu}"
PORT="${PORT:-8000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: CHECKPOINT_DIR is required."
    echo "Usage: CHECKPOINT_DIR=<path-or-gs-uri> $0"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

conda deactivate 2>/dev/null || true
source /data/user_data/saksham3/vla/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "Host        : $(hostname)"
echo "CUDA dev    : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config      : $CONFIG_NAME"
echo "Checkpoint  : $CHECKPOINT_DIR"
echo "Port        : $PORT"

exec python scripts/serve_policy.py policy:checkpoint \
    --policy.config="$CONFIG_NAME" \
    --policy.dir="$CHECKPOINT_DIR" \
    --port="$PORT"
