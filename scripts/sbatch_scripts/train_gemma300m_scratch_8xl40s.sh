#!/bin/bash
#SBATCH --job-name=cql_300m_l40s
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/saksham3/projects/AIRe/robocoin/batch_value_learning/logs/cql_300m_l40s_%j.out
#SBATCH --error=/home/saksham3/projects/AIRe/robocoin/batch_value_learning/logs/cql_300m_l40s_%j.err

set -euo pipefail

REPO_ROOT="/home/saksham3/projects/AIRe/robocoin/batch_value_learning"
mkdir -p "$REPO_ROOT/logs"

conda deactivate 2>/dev/null || true
source /data/user_data/saksham3/vla/bin/activate

export OPENPI_DATA_HOME=/data/group_data/rl/saksham3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export NCCL_P2P_DISABLE=1
export JAX_TRACEBACK_FILTERING=off

cd "$REPO_ROOT"
echo "Host: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec python scripts/train_value_function.py real_shirt_hang_paligemma_cql_rlds_gemma300m_scratch --resume \
    --batch-size=128 \
    --fsdp-devices=1 \
    --checkpoint-base-dir=/data/user_data/saksham3/robocoin/value_functions/Q \
    --project-name=robocoin_value_learning \
    --log-interval=100 \
    --save-interval=25000 \
    --wandb-group='Value Functions'
