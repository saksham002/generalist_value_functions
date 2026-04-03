#!/bin/bash
# Run the shirt-hang eval script.
# Loads the policy locally and connects to the robot environment server.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================

CONFIG_NAME="robocoin_bimanual_pi05"
CHECKPOINT_DIR="/data/user_data/jeffyu/checkpoints/robocoin/pi05_finetune/real_hang_pi05_finetune"
FINE_TUNE_CONFIG="real_hang_pi05_finetune"  # Optional: FineTuneConfig name from config.py. Leave empty to skip.
STEP=275000

ROBOT_HOST="xarmpc.pc.cs.cmu.edu"
ROBOT_PORT=8080

NUM_EPISODES=1
DEBUG=false  # Set to true to skip policy loading and just save images / print state

# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Shirt-Hang Eval${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "  Config:     ${YELLOW}${CONFIG_NAME}${NC}"
echo -e "  Checkpoint: ${YELLOW}${CHECKPOINT_DIR}${NC}"
if [ -n "${FINE_TUNE_CONFIG}" ]; then
    echo -e "  Fine-tune:  ${YELLOW}${FINE_TUNE_CONFIG}${NC}"
fi
echo -e "  Robot:      ${YELLOW}${ROBOT_HOST}:${ROBOT_PORT}${NC}"
echo -e "  Episodes:   ${YELLOW}${NUM_EPISODES}${NC}"
echo -e "  Debug:      ${YELLOW}${DEBUG}${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}ERROR: pyproject.toml not found at ${REPO_ROOT}${NC}"
    exit 1
fi

FINE_TUNE_ARGS=""
if [ -n "${FINE_TUNE_CONFIG}" ]; then
    FINE_TUNE_ARGS="--args.fine-tune-config ${FINE_TUNE_CONFIG}"
fi

DEBUG_ARGS=""
if [ "${DEBUG}" = true ]; then
    DEBUG_ARGS="--args.debug"
fi

echo -e "${BLUE}Starting eval...${NC}"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run eval/xarm_scripts/eval_shirt_hang_remote.py \
    --args.config-name "${CONFIG_NAME}" \
    --args.checkpoint-dir "${CHECKPOINT_DIR}" \
    --args.robot-host "${ROBOT_HOST}" \
    --args.robot-port "${ROBOT_PORT}" \
    --args.num-episodes "${NUM_EPISODES}" \
    ${FINE_TUNE_ARGS} \
    ${DEBUG_ARGS}
