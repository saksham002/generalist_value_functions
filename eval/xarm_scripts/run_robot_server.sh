#!/bin/bash
# Start Robot Environment Server
# Run this in TAB 1 with flow_ksh conda environment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Robot Environment Server (TAB 1)${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check conda environment
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo -e "Current conda environment: ${YELLOW}${CURRENT_ENV}${NC}"

if [ "$CURRENT_ENV" != "flow_ksh" ] && [ "$CURRENT_ENV" != "flow_jeff" ]; then
    echo -e "${RED}ERROR: Not in flow_ksh or flow_jeff conda environment!${NC}"
    echo -e "${YELLOW}Please activate one first:${NC}"
    echo -e "  ${BLUE}conda activate flow_ksh${NC} or ${BLUE}conda activate flow_jeff${NC}"
    echo ""
    exit 1
fi

# Server configuration
PORT=8080
HOST="xarmpc.pc.cs.cmu.edu"
CONTROL_FREQ=60
TIME_LIMIT=1800  # 30 minutes (increased for manual intervention + policy eval)

# Environment settings
ENABLE_INTERVENTION="--enable_intervention"
OVERLAY_HEATMAP="/media/huzheyuan/data0/huzheyuan_folder_backup/dual_xarms/dual_xarms_sim/dual_xarms_sim/real_hang_r0.png"

# Frame stacking settings (must match training config)
# CRITICAL: OpenPI shirt_hang_rac was trained with obs_history_len=1 (NO frame stacking)
# BC FlowMatch uses obs_history_len=2, but OpenPI uses 1
OBS_HISTORY_LEN=1  # Changed from 2 to 1 to match OpenPI training config
HISTORY_GAP=29

# Video settings
VIDEO_DIR="./robot_server_videos"

echo ""
echo -e "${BLUE}Server Configuration:${NC}"
echo -e "  Port: ${YELLOW}${PORT}${NC}"
echo -e "  Host: ${YELLOW}${HOST}${NC} (accessible from localhost)"
echo -e "  Control freq: ${YELLOW}${CONTROL_FREQ} Hz${NC}"
echo -e "  Intervention: ${YELLOW}Enabled (Oculus)${NC}"
echo -e "  Frame stacking: ${YELLOW}${OBS_HISTORY_LEN} frames (gap=${HISTORY_GAP})${NC}"
echo -e "  Video dir: ${YELLOW}${VIDEO_DIR}${NC}"
echo ""
echo -e "${BLUE}Heatmap overlay: ${YELLOW}${OVERLAY_HEATMAP}${NC}"
echo ""
echo -e "${GREEN}Starting robot server...${NC}"
echo -e "${YELLOW}Open another terminal tab and run the policy client!${NC}"
echo ""

# Start server
python3 robot_environment_server.py \
    --port "${PORT}" \
    --host "${HOST}" \
    --control_freq "${CONTROL_FREQ}" \
    --time_limit "${TIME_LIMIT}" \
    --overlay_heatmap "${OVERLAY_HEATMAP}" \
    ${ENABLE_INTERVENTION} \
    --obs_history_len "${OBS_HISTORY_LEN}" \
    --history_gap "${HISTORY_GAP}" \
    --video_dir "${VIDEO_DIR}"

echo ""
echo -e "${GREEN}Server stopped${NC}"

