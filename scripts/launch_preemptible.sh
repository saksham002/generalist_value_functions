#!/bin/bash
#SBATCH --job-name=launch_preemptible
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/saksham3/logs/launch_preemptible_%j.log
#
# Keep a run alive on a PREEMPTIBLE TPU pod. Use this only for spot pods; on a
# reserved pod the creator-job step below is meaningless.
#
# Every --interval seconds, in order:
#   0. If --final-checkpoint is set and committed on GCS, the run is done -> exit 0.
#   1. Ensure a spot-pod creator job for this pod is queued/running; submit it if not.
#   2. Look at the pod:
#        - absent            -> nothing to do this tick (the creator will rebuild it)
#        - present, job up   -> nothing to do
#        - present, no job   -> kill unattended-upgrades, then launch the run
#
# Usage:
#   sbatch scripts/launch_preemptible.sh \
#       --pod v5e-saksham-spot-64-0 \
#       --zone europe-west4-a \
#       --tpu-type v5e-64 \
#       --creator-script random/create_spot_pod_until_ready.sbatch \
#       --creator-name spot_pod_creator \
#       --final-checkpoint gs://bucket/checkpoints/<cfg>/<exp>/70000 \
#       --command "python scripts/train.py lego_pi05_subtask --resume --batch-size=256 ..."

set -u

POD=""
ZONE=""
TPU_TYPE=""
CREATOR_SCRIPT=""
CREATOR_NAME=""
TRAIN_COMMAND=""
FINAL_CHECKPOINT=""
INTERVAL=300
# Matches the training entrypoints (train.py / train_value_function.py) so the
# script can tell "job already running on the pod" from "pod idle".
JOB_PATTERN="scripts/train"
REPO_DIR="/home/saksham3/projects/AIRe/robocoin/batch_value_learning"
VENV="/data/user_data/saksham3/vla/bin/activate"

while [ $# -gt 0 ]; do
    case "$1" in
        --pod) POD="$2"; shift 2;;
        --zone) ZONE="$2"; shift 2;;
        --tpu-type) TPU_TYPE="$2"; shift 2;;
        --creator-script) CREATOR_SCRIPT="$2"; shift 2;;
        --creator-name) CREATOR_NAME="$2"; shift 2;;
        --command) TRAIN_COMMAND="$2"; shift 2;;
        --final-checkpoint) FINAL_CHECKPOINT="$2"; shift 2;;
        --interval) INTERVAL="$2"; shift 2;;
        --job-pattern) JOB_PATTERN="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

for required in POD ZONE TPU_TYPE CREATOR_SCRIPT CREATOR_NAME TRAIN_COMMAND; do
    if [ -z "${!required}" ]; then
        echo "missing required arg: --$(echo $required | tr 'A-Z_' 'a-z-')" >&2
        exit 2
    fi
done

log() { echo "[$(date -Is)] $*"; }

log "watchdog start: pod=$POD zone=$ZONE type=$TPU_TYPE interval=${INTERVAL}s"
log "creator: $CREATOR_SCRIPT (job name '$CREATOR_NAME')"

cd "$REPO_DIR"
# shellcheck disable=SC1090
source "$VENV"

while true; do
    # --- 0. finished? ------------------------------------------------------
    # commit_success.txt is orbax's atomic marker: the directory alone can exist
    # mid-write, so checking it would exit on a partial checkpoint.
    if [ -n "$FINAL_CHECKPOINT" ] && \
       timeout 90 gcloud storage ls "${FINAL_CHECKPOINT%/}/commit_success.txt" >/dev/null 2>&1; then
        log "final checkpoint committed at $FINAL_CHECKPOINT — run complete, exiting"
        exit 0
    fi

    # --- 1. creator job ---------------------------------------------------
    # -h drops the header, %j is the job name; -x anchors so a longer name
    # (e.g. spot_pod_creator_b) does not satisfy the check for a shorter one.
    if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "$CREATOR_NAME"; then
        log "creator '$CREATOR_NAME' already queued/running"
    else
        log "creator '$CREATOR_NAME' absent -> submitting $CREATOR_SCRIPT"
        sbatch "$CREATOR_SCRIPT" 2>&1 | sed 's/^/    /'
    fi

    # --- 2. pod, then job on pod -------------------------------------------
    STATE=$(timeout 90 gcloud compute tpus tpu-vm describe "$POD" --zone="$ZONE" \
            --format="value(state)" 2>/dev/null | tr -d ' \t')
    HEALTH=$(timeout 90 gcloud compute tpus tpu-vm describe "$POD" --zone="$ZONE" \
            --format="value(health)" 2>/dev/null | tr -d ' \t')

    if [ -z "$STATE" ]; then
        log "pod absent — waiting for the creator to rebuild it"
    elif [ "$STATE" != "READY" ]; then
        log "pod state=$STATE (not READY)"
    elif [ "$HEALTH" != "HEALTHY" ]; then
        # READY alone is not enough: a pod under UNHEALTHY_MAINTENANCE still reports
        # READY while its workers are unreachable, so launching there just fails.
        log "pod state=READY but health=$HEALTH — not launchable"
    else
        # A launcher already in flight for this pod means setup/sync is still
        # running and the job has not appeared yet; do not start a second one.
        if pgrep -f "run_on_tpu.py.*$POD" >/dev/null 2>&1; then
            log "launcher already in flight for $POD"
        else
            RUNNING=$(timeout 120 gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --worker=0 \
                      --command="pgrep -c -f '$JOB_PATTERN'" 2>/dev/null \
                      | grep -Ev "^SSH:|SSH key|ssh batch|Warning" | tr -d ' \t' | head -1)
            RUNNING=${RUNNING:-0}

            if [ "$RUNNING" -gt 0 ] 2>/dev/null; then
                log "pod READY, job running (${RUNNING} procs)"
            else
                log "pod READY, no job — killing unattended-upgrades then launching"
                # The [u] bracket keeps the pattern from matching the ssh command
                # line that carries it, which would kill this very session.
                timeout 300 gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --worker=all \
                    --command="sudo pkill -9 -f '[u]nattended-upgr' 2>/dev/null; \
                               sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock; \
                               sudo dpkg --configure -a >/dev/null 2>&1; echo prepped" 2>&1 \
                    | grep -Ev "^SSH:|SSH key|ssh batch|Warning" | sed 's/^/    /'

                log "launching run on $POD"
                python scripts/run_on_tpu.py --tpu-type "$TPU_TYPE" --tpu-name "$POD" \
                    --command "$TRAIN_COMMAND" 2>&1 | sed 's/^/    /'
                # PIPESTATUS[0] is run_on_tpu's status; plain $? would be sed's.
                log "launcher returned ${PIPESTATUS[0]}"
            fi
        fi
    fi

    sleep "$INTERVAL"
done
