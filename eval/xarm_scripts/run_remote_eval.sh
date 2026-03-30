#!/bin/bash
# Run Policy Evaluation on Remote Robot
set -e

# Configuration
CONFIG="pi05_shirt_hang_rac_lora"
CHECKPOINT_ROOT="/home/huzheyuan/kshitiz2/openpi/checkpoints/"
CHECKPOINT_STEP=190000

# Robot server configuration (same machine, different conda env)
ROBOT_HOST="localhost"  # Server runs on localhost in different terminal
ROBOT_PORT=8080

# Evaluation settings
NUM_EPISODES=60
QUERY_FREQ=30
HORIZON=60
MAX_ENV_STEPS=14400 # Increased for manual intervention (30 min at 60Hz)

# Video settings
VIDEO_DIR="videos/pi05_shirt_hang_rac_190000_60ep"
VIDEO_FPS=60

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Remote Robot Policy Evaluation (TAB 2)${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${YELLOW}PREREQUISITE: Robot server must be running in TAB 1${NC}"
echo -e "${BLUE}If not started yet, open another tab and run:${NC}"
echo -e "  ${BLUE}cd ~/kshitiz2/flow_match/bc_flowmatch/scripts${NC}"
echo -e "  ${BLUE}conda activate flow_ksh${NC}"
echo -e "  ${BLUE}./run_robot_server.sh${NC}"
echo ""
echo -e "${RED}  CRITICAL: Robot server frame stacking config MUST match training config!${NC}"
echo -e "${YELLOW}OpenPI shirt_hang_rac was trained with obs_history_len=1 (NO frame stacking)${NC}"
echo -e "${YELLOW}Robot server in run_robot_server.sh MUST have: OBS_HISTORY_LEN=1${NC}"
echo ""
echo -e "${YELLOW}(Optional) Test connection first:${NC}"
echo -e "  ${BLUE}cd ~/kshitiz2/openpi/scripts && python3 test_robot_connection.py${NC}"
echo ""

# Activate conda environment
echo -e "${BLUE}Activating conda environment: openpi_mj336${NC}"
eval "$(conda shell.bash hook)"
conda activate openpi_mj336
echo ""

# Set RAC_PATH and find dual_xarms_sim package
# The actual location is /media/huzheyuan/data0/huzheyuan_folder_backup/dual_xarms/dual_xarms_sim
DUAL_XARMS_SIM_PATH=""
for path in "/media/huzheyuan/data0/huzheyuan_folder_backup/dual_xarms/dual_xarms_sim" "${HOME}/dual_xarms/dual_xarms_sim" "${HOME}/kshitiz2/RAC/dual_xarms/dual_xarms_sim" "/data/user_data/kshitiz2/RAC/dual_xarms/dual_xarms_sim"; do
    if [ -d "${path}" ] && [ -f "${path}/dual_xarms_sim/__init__.py" ]; then
        DUAL_XARMS_SIM_PATH="${path}"
        break
    fi
done

if [ -n "${DUAL_XARMS_SIM_PATH}" ]; then
    export PYTHONPATH="${DUAL_XARMS_SIM_PATH}:${PYTHONPATH}"
    # Set RAC_PATH for backward compatibility
    if [ -d "${HOME}/kshitiz2/RAC" ]; then
        export RAC_PATH="${HOME}/kshitiz2/RAC"
    elif [ -d "$(dirname $(dirname ${DUAL_XARMS_SIM_PATH}))/RAC" ]; then
        export RAC_PATH="$(dirname $(dirname ${DUAL_XARMS_SIM_PATH}))/RAC"
    else
        export RAC_PATH="$(dirname $(dirname ${DUAL_XARMS_SIM_PATH}))"
    fi
    echo -e "${GREEN} Found dual_xarms_sim at: ${DUAL_XARMS_SIM_PATH}${NC}"
    echo -e "${GREEN} RAC_PATH set to: ${RAC_PATH}${NC}"
else
    echo -e "${RED}ERROR: dual_xarms_sim not found in expected locations${NC}"
    echo -e "${YELLOW}Tried: ${HOME}/dual_xarms/dual_xarms_sim, ${HOME}/kshitiz2/RAC/dual_xarms/dual_xarms_sim${NC}"
    exit 1
fi

# Change to openpi directory (where pyproject.toml is located)
# Use absolute path to openpi root
OPENPI_ROOT="${HOME}/kshitiz2/openpi"
if [ ! -d "${OPENPI_ROOT}" ]; then
    # Fallback: try to find it relative to script location
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    OPENPI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${OPENPI_ROOT}"
echo -e "${BLUE}Changed to openpi directory: ${OPENPI_ROOT}${NC}"
echo -e "${BLUE}Current directory: $(pwd)${NC}"

# Verify we're in the right place
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}ERROR: pyproject.toml not found in ${OPENPI_ROOT}${NC}"
    exit 1
fi

if [ ! -f "scripts/xarm_scripts/eval_shirt_hang_rac_deploy.py" ]; then
    echo -e "${RED}ERROR: scripts/xarm_scripts/eval_shirt_hang_rac_deploy.py not found!${NC}"
    echo -e "${YELLOW}Looking in: ${OPENPI_ROOT}/scripts/xarm_scripts/eval_shirt_hang_rac_deploy.py${NC}"
    exit 1
fi

# Ensure uv dependencies are synced (especially dual-xarms-sim)
echo -e "${BLUE}Syncing uv dependencies...${NC}"
uv sync
echo ""

# Verify dual-xarms-sim can be found
echo -e "${BLUE}Verifying dual-xarms-sim installation...${NC}"
if uv run python -c "import dual_xarms_sim; print(' dual_xarms_sim imported successfully')" 2>/dev/null; then
    echo -e "${GREEN} dual_xarms_sim is available${NC}"
else
    echo -e "${YELLOW}Warning: dual_xarms_sim not found via uv, will rely on RAC_PATH fallback${NC}"
fi
echo ""

echo -e "Configuration: ${YELLOW}${CONFIG}${NC}"
echo -e "Checkpoint: ${YELLOW}${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}${NC}"
echo -e "Robot Server: ${YELLOW}${ROBOT_HOST}:${ROBOT_PORT}${NC}"
echo -e "Episodes: ${YELLOW}${NUM_EPISODES}${NC}"
echo -e "Video Directory: ${YELLOW}${VIDEO_DIR}${NC}"
echo -e "${BLUE}Note: Robot server videos saved to: flow_match/bc_flowmatch/scripts/robot_server_videos/${NC}"
echo ""

# Verify checkpoint exists
echo -e "${BLUE}Verifying checkpoint...${NC}"
if [ ! -d "${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}" ]; then
    echo -e "${RED}ERROR: Checkpoint not found at ${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}${NC}"
    exit 1
fi
echo -e "${GREEN}${NC} Checkpoint verified"

# Verify robot server is running
echo ""
echo -e "${BLUE}Checking robot server...${NC}"
if curl -s http://${ROBOT_HOST}:${ROBOT_PORT}/api/status > /dev/null 2>&1; then
    echo -e "${GREEN}${NC} Robot server is running"
else
    echo -e "${RED}ERROR: Robot server not responding at ${ROBOT_HOST}:${ROBOT_PORT}${NC}"
    echo -e "${YELLOW}Please start the robot server first:${NC}"
    echo -e "  ${BLUE}cd ~/kshitiz2/flow_match/bc_flowmatch/scripts${NC}"
    echo -e "  ${BLUE}conda activate flow_ksh${NC}"
    echo -e "  ${BLUE}./run_robot_server.sh${NC}"
    exit 1
fi

# Check robot server config
echo ""
echo -e "${BLUE}Checking robot server configuration...${NC}"
SERVER_INFO=$(curl -s http://${ROBOT_HOST}:${ROBOT_PORT}/api/metadata)
OBS_HIST=$(echo $SERVER_INFO | grep -o '"obs_history_len":[0-9]*' | cut -d: -f2)
HIST_GAP=$(echo $SERVER_INFO | grep -o '"history_gap":[0-9]*' | cut -d: -f2)

echo -e "  obs_history_len: ${YELLOW}${OBS_HIST}${NC} (should be 1)"
echo -e "  history_gap: ${YELLOW}${HIST_GAP}${NC} (should be 29)"

if [ "$OBS_HIST" != "1" ]; then
    echo -e "${RED}ERROR: Robot server has wrong obs_history_len=${OBS_HIST} (should be 1)${NC}"
    echo -e "${YELLOW}Please restart robot server with correct config!${NC}"
    exit 1
fi
echo -e "${GREEN}${NC} Robot server configuration is correct"

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${BLUE}Starting evaluation...${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# Run evaluation with remote robot flag (from openpi root directory)
# RAC_PATH is already exported above
# NOTE: Videos are saved in two places:
#   1. Robot server saves to: flow_match/bc_flowmatch/scripts/robot_server_videos/
#   2. Client saves to: openpi/videos/ (if not using --remote_robot)
uv run scripts/xarm_scripts/eval_shirt_hang_rac_deploy.py \
    "${CONFIG}" \
    --checkpoint_root "${CHECKPOINT_ROOT}" \
    --steps "${CHECKPOINT_STEP}" \
    --num_episodes "${NUM_EPISODES}" \
    --query_freq "${QUERY_FREQ}" \
    --horizon "${HORIZON}" \
    --max_env_steps "${MAX_ENV_STEPS}" \
    --video_dir "${VIDEO_DIR}" \
    --video_fps "${VIDEO_FPS}" \
    --remote_robot \
    --robot_host "${ROBOT_HOST}" \
    --robot_port "${ROBOT_PORT}" \
    --enable_intervention \
    --forced_replan_after_intervention \
    --manual_labeling \
    --log_wandb \
    --wandb_project "openpi_shirt_hang_rac_deploy" \
    --trace_replans \
    --dump_episode_jsonl \
    --save_data_dir "data/openpi_demo_190k" \
    --record

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}Evaluation complete!${NC}"
echo -e "${GREEN}================================================================${NC}"

