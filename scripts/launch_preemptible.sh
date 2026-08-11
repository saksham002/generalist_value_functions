#!/bin/bash
#SBATCH --job-name=launch_preemptible
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/saksham3/logs/launch_preemptible_%j.log
#
# SLURM entry point for a spot TPU run. Everything this used to do itself — creating and
# recreating the pod, polling for preemption, relaunching, guarding host RSS, deciding
# when the run is finished — now lives in scripts/run_on_tpu.py. This exists only to hold
# a SLURM allocation so the launcher survives the login session, and it must stay in the
# foreground for that to work.
#
# Zones are not passed: run_on_tpu.py reads live spot quota and races every eligible US/EU
# zone, so there is nothing here to keep in sync with a pod's actual location.
#
# Usage:
#   sbatch scripts/launch_preemptible.sh \
#       --tpu-type v5e-64 \
#       --done-marker gs://saksham-euw4/markers/<run-id> \
#       --command "python scripts/train_value_function.py <config> --resume --batch-size=128 ..."

set -u

TPU_TYPE=""
TRAIN_COMMAND=""
DONE_MARKER=""
EXTRA_ARGS=()

REPO_DIR="/home/saksham3/projects/AIRe/robocoin/batch_value_learning"
VENV="/data/user_data/saksham3/vla/bin/activate"

while [ $# -gt 0 ]; do
    case "$1" in
        --tpu-type)     TPU_TYPE="$2";      shift 2;;
        --command)      TRAIN_COMMAND="$2"; shift 2;;
        --done-marker)  DONE_MARKER="$2";   shift 2;;
        # Anything else is handed straight to run_on_tpu.py, so its flags do not have to
        # be mirrored here.
        *)              EXTRA_ARGS+=("$1"); shift;;
    esac
done

for required in TPU_TYPE TRAIN_COMMAND; do
    if [ -z "${!required}" ]; then
        echo "missing required arg: --$(echo "$required" | tr 'A-Z_' 'a-z-')" >&2
        exit 2
    fi
done

log() { echo "[$(date -Is)] $*"; }

log "spot launch: tpu_type=$TPU_TYPE"
log "command: $TRAIN_COMMAND"
# Logged because they decide where checkpoints come from and when the run is considered
# finished; without them the log cannot explain what the launcher actually did.
log "done-marker: ${DONE_MARKER:-none}"
log "extra args: ${EXTRA_ARGS[*]:-none}"

cd "$REPO_DIR" || exit 1
# shellcheck disable=SC1090
source "$VENV"

ARGS=(
    --tpu-type "$TPU_TYPE"
    --spot
    --retry-on-preemption
    --command "$TRAIN_COMMAND"
)
[ -n "$DONE_MARKER" ] && ARGS+=(--done-marker "$DONE_MARKER")
[ ${#EXTRA_ARGS[@]} -gt 0 ] && ARGS+=("${EXTRA_ARGS[@]}")

python scripts/run_on_tpu.py "${ARGS[@]}"
STATUS=$?
log "run_on_tpu.py exited with $STATUS"
exit $STATUS
