#!/bin/bash
# Run the shirt-hang eval using the remote policy server.
# Requires:
#   TAB 1 (robot machine): robot_environment_server.py
#   TAB 2 (Babel):         run_serve_policy.sh
#   TAB 3 (here):          this script

set -e

# =============================================================================
# Configuration — edit these before running
# =============================================================================

POLICY_HOST="babel-gpu-node"    # hostname of machine running serve_policy.py
POLICY_PORT=8080

ROBOT_HOST="robot-machine"      # hostname of machine running robot_environment_server.py
ROBOT_PORT=8081

EMBODIMENT="dual_xarms"

NUM_EPISODES=1
QUERY_FREQ=30
MAX_STEPS=1800

REAL_ACTION_START=0
REAL_ACTION_DIM=14

# Timeout must be long enough to cover JAX JIT compilation on first call (~2-5 min)
POLICY_TIMEOUT=300

# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Shirt-Hang Eval — Policy Server Mode${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "  Policy server: ${YELLOW}${POLICY_HOST}:${POLICY_PORT}${NC}"
echo -e "  Robot server:  ${YELLOW}${ROBOT_HOST}:${ROBOT_PORT}${NC}"
echo -e "  Embodiment:    ${YELLOW}${EMBODIMENT}${NC} (prompt auto-detected from robot state)"
echo -e "  Episodes:      ${YELLOW}${NUM_EPISODES}${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}ERROR: pyproject.toml not found at ${REPO_ROOT}${NC}"
    exit 1
fi

# Check policy server is reachable before starting
echo -e "${BLUE}Checking policy server...${NC}"
if curl -s --max-time 5 "http://${POLICY_HOST}:${POLICY_PORT}/api/health" > /dev/null 2>&1; then
    echo -e "${GREEN}Policy server is up.${NC}"
else
    echo -e "${YELLOW}Policy server not yet responding — the eval script will wait for it.${NC}"
fi

# Check robot server is reachable
echo -e "${BLUE}Checking robot server...${NC}"
if curl -s --max-time 5 "http://${ROBOT_HOST}:${ROBOT_PORT}/api/health" > /dev/null 2>&1; then
    echo -e "${GREEN}Robot server is up.${NC}"
else
    echo -e "${RED}ERROR: Robot server not responding at ${ROBOT_HOST}:${ROBOT_PORT}${NC}"
    echo -e "${YELLOW}Please start robot_environment_server.py on the robot machine first.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Starting eval...${NC}"
echo ""

uv run eval/xarm_scripts/eval_shirt_hang_policy_server.py \
    --embodiment "${EMBODIMENT}" \
    --policy-host "${POLICY_HOST}" \
    --policy-port "${POLICY_PORT}" \
    --robot-host "${ROBOT_HOST}" \
    --robot-port "${ROBOT_PORT}" \
    --num-episodes "${NUM_EPISODES}" \
    --query-freq "${QUERY_FREQ}" \
    --max-steps "${MAX_STEPS}" \
    --real-action-start "${REAL_ACTION_START}" \
    --real-action-dim "${REAL_ACTION_DIM}" \
    --policy-timeout "${POLICY_TIMEOUT}"

echo ""
echo -e "${GREEN}Eval complete.${NC}"
