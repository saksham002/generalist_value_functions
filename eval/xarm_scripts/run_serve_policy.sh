#!/bin/bash
# Start the policy server on Babel (GPU machine).
# Run this in a terminal on Babel before starting the eval script.

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================

CONFIG_NAME="FILL IN HERE"
CHECKPOINT_DIR="/path/to/checkpoint"

PORT=8080
HOST="0.0.0.0"

# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Policy Server${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "  Config:     ${YELLOW}${CONFIG_NAME}${NC}"
echo -e "  Checkpoint: ${YELLOW}${CHECKPOINT_DIR}${NC}"
echo -e "  Listening:  ${YELLOW}${HOST}:${PORT}${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}ERROR: pyproject.toml not found at ${REPO_ROOT}${NC}"
    exit 1
fi

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo -e "${RED}ERROR: Checkpoint not found at ${CHECKPOINT_DIR}${NC}"
    exit 1
fi

echo -e "${BLUE}Starting policy server...${NC}"
echo -e "${YELLOW}Leave this running and start the eval script in a separate terminal.${NC}"
echo ""

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run eval/xarm_scripts/serve_policy.py \
    --config-name "${CONFIG_NAME}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
