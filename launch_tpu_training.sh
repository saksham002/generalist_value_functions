#!/bin/bash

# Launch training on TPU pod
# This script can be run from a LOCAL machine or directly from the HPC cluster.
# It:
# 1. (local mode) Rsyncs code from HPC cluster to local
# 2. Rsyncs code from the current machine to TPU pod NFS
# 3. Launches tpc with pod_config.py
# 4. Connects to the pod via SSH

set -e

# Check if a TPU VM name is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <tpu-vm-name> <tpu-ip> [config_name] [extra_train_args] [sync_source]"
    echo ""
    echo "Arguments:"
    echo "  tpu-vm-name    Name of the TPU VM (e.g., v5e-tpu-64-0)"
    echo "  tpu-ip         IP address of the TPU VM for rsync"
    echo "  config_name    (optional) Training config name (default: robocoin_paligemma_v_mc)"
    echo "  train_args     (optional) Extra training args as comma-separated key=value pairs"
    echo "  sync_source    (optional) Where code is synced from: local|hpc (default: local)"
    echo ""
    echo "Example:"
    echo "  $0 v5e-tpu-64-0 34.90.176.237"
    echo "  $0 v5e-tpu-64-0 34.90.176.237 robocoin_paligemma_v_mc_ce"
    echo "  $0 v5e-tpu-64-0 34.90.176.237 robocoin_paligemma_v_mc num_train_steps=50000,lr=1e-5"
    echo "  $0 v5e-tpu-64-0 34.90.176.237 robocoin_paligemma_v_mc \"\" hpc"
    exit 1
fi

TPU_VM_NAME=$1
TPU_IP=$2
CONFIG_NAME=${3:-robocoin_paligemma_v_mc}
TRAIN_ARGS=${4:-}
SYNC_SOURCE=${5:-local}

if [[ "$SYNC_SOURCE" != "local" && "$SYNC_SOURCE" != "hpc" ]]; then
    echo "ERROR: sync_source must be 'local' or 'hpc' (got: '$SYNC_SOURCE')"
    exit 1
fi

if [[ "$SYNC_SOURCE" == "hpc" ]]; then
    CHOWN_USER="saksham3"
else
    CHOWN_USER="saksham"
fi

PROJECT="cmu-aidm-v2"
ZONE="europe-west4-b"

# Code directory is always batch_value_learning
CODE_DIR="batch_value_learning"

# Resolve repo root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine NFS user directory based on pod name
if [[ "$TPU_VM_NAME" =~ 64-1$ ]]; then
    NFS_USER="saksham3"
else
    NFS_USER="saksham"
fi

# HPC cluster source
HPC_HOST="saksham3@login.babel.cs.cmu.edu"
HPC_SRC_DIR="~/projects/AIRe/robocoin/batch_value_learning"

# Local staging directory (repo root)
LOCAL_DIR="${REPO_ROOT}/"

# TPU NFS destination
NFS_DIR="/nfs/aidm_nfs/${NFS_USER}/batch_value_learning"
DEST_DIR="${TPU_IP}:${NFS_DIR}"

TRAIN_SCRIPT="${NFS_DIR}/scripts/train_value_function.py"

# Cache file for TPU name/zone mapping
CACHE_FILE="$HOME/.cache/tpus"
mkdir -p "$(dirname "$CACHE_FILE")"

# Check if the TPU info is already cached
if [ -f "$CACHE_FILE" ]; then
    CACHED_INFO=$(grep "^$TPU_VM_NAME:" "$CACHE_FILE")
    if [ -n "$CACHED_INFO" ]; then
        CACHED_ZONE=$(echo "$CACHED_INFO" | cut -d':' -f2)
        NUM_WORKERS=$(echo "$CACHED_INFO" | cut -d':' -f3)
    fi
fi

if [ -n "$CACHED_ZONE" ]; then
    ZONE=$CACHED_ZONE
else
    # Get the TPU information
    echo "Getting TPU information for $TPU_VM_NAME"
    for MAYBE_ZONE in europe-west4-b; do
        TPU_INFO=$(gcloud compute tpus tpu-vm describe $TPU_VM_NAME --project=$PROJECT --zone=$MAYBE_ZONE --format=json 2>/dev/null)
        echo "TPU_INFO: $TPU_INFO"
        if [ $? -eq 0 ]; then
            # Cache the successful name/zone mapping and number of workers
            ZONE=$MAYBE_ZONE
            NUM_WORKERS=$(echo "$TPU_INFO" | jq '.networkEndpoints | length')
            echo "$TPU_VM_NAME:$ZONE:$NUM_WORKERS" >> "$CACHE_FILE"
            break
        fi
    done
fi

echo "============================================"
echo "TPU Training Launch Script"
echo "============================================"
echo "TPU_VM_NAME: $TPU_VM_NAME"
echo "TPU_IP: $TPU_IP"
echo "ZONE: $ZONE"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "TRAIN_ARGS: $TRAIN_ARGS"
echo "SYNC_SOURCE: $SYNC_SOURCE"
echo "CHOWN_USER: $CHOWN_USER"
echo "Number of workers: ${NUM_WORKERS:-unknown}"
echo "============================================"

if [[ "$SYNC_SOURCE" == "local" ]]; then
    # Step 1: Rsync from HPC cluster to local
    echo "[Step 1/3] Syncing from HPC cluster to local..."
    echo "rsync -avzL --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.pyc' --exclude wandb ${HPC_HOST}:${HPC_SRC_DIR}/ ${LOCAL_DIR}"
    rsync -avzL --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.pyc' --exclude wandb ${HPC_HOST}:${HPC_SRC_DIR}/ ${LOCAL_DIR}
else
    # Running on HPC: skip HPC->local sync and push the checked-out repo directly to the TPU NFS.
    echo "[Step 1/3] SYNC_SOURCE=hpc: skipping HPC->local sync (using ${LOCAL_DIR})"
fi

# Step 2a: Ensure NFS is mounted on all TPU workers (must happen before rsync to NFS)
echo "[Step 2a] Checking/mounting NFS on all TPU workers..."
export POD_NAME=$TPU_VM_NAME
MOUNT_CMD="if ! mount | grep -q aidm_nfs; then echo 'NFS not mounted, mounting...'; sudo apt -y update && sudo apt -y install nfs-common && sudo mkdir -p -m 777 /nfs/aidm_nfs && sudo mount -o rw,intr 10.155.154.42:/europe /nfs/aidm_nfs && echo 'NFS mounted successfully'; else echo 'NFS already mounted'; fi"
tpc run --project=$PROJECT --zone=$ZONE --name=$TPU_VM_NAME --command="$MOUNT_CMD"

# Step 2b: Sync from local to TPU pod NFS (now that NFS is mounted)
echo "[Step 2b] Syncing to TPU NFS..."
ssh ${TPU_IP} "mkdir -p ${NFS_DIR}"
rsync -avz --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.pyc' --exclude wandb --exclude .pytest_cache ${LOCAL_DIR} ${TPU_IP}:${NFS_DIR}/

# Step 3a: Copy .bashrc from worker 0 to local, then upload to all workers
echo "[Step 3a] Syncing .bashrc from worker 0 to all workers..."
BASHRC_LOCAL="/tmp/tpu_bashrc_${TPU_VM_NAME}"
gcloud compute tpus tpu-vm scp --project=$PROJECT --zone=$ZONE ${TPU_VM_NAME}:~/.bashrc ${BASHRC_LOCAL} --worker=0
tpc upload --project=$PROJECT --zone=$ZONE --name=$TPU_VM_NAME --upload_path="${BASHRC_LOCAL}:~/.bashrc"

# Step 3b: Launch the pod configuration
echo "[Step 3b] Launching training job..."
export CONFIG_NAME
export TRAIN_ARGS
export TRAIN_SCRIPT
export NFS_USER
export CHOWN_USER
POD_NAME=$TPU_VM_NAME tpc launch pod_config.py --project=$PROJECT

# Connect to the pod
echo "Connecting to TPU pod..."
bash ssh_pod.sh $TPU_VM_NAME $PROJECT
