#!/bin/bash
#SBATCH --job-name=lego_baseline_watchdog
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/saksham3/logs/lego_baseline_watchdog_%j.log
#
# Watchdog for the lego_pi05_task_description_baseline run that is already training on a
# TPU pod without a launcher attached (its launcher died on a local gcloud reauth error,
# while the remote tmux job kept going).
#
# Behaviour:
#   - poll the pod; while training is alive, do nothing
#   - if the pod is preempted, hand off to run_on_tpu.py --spot --retry-on-preemption,
#     which acquires a new pod, replays the command and owns all further retries
#   - exit when the done-marker appears (the run finished on the TPU)
#
# Safeguard: a gcloud call that FAILS is "unknown", never "pod gone". An expired
# credential returns nothing exactly like an absent pod does, and treating that as a
# preemption would start a second concurrent run against the same checkpoint dir.
# Only an authoritative answer -- a successful describe reporting a non-READY state, or
# an explicit NOT_FOUND -- counts as preemption.

set -u

POD="v5e-saksham-spot-64-0"
ZONE="europe-west4-a"
TPU_TYPE="v5e-64"
MARKER="gs://saksham-euw4/markers/lego_pi05_task_description_baseline_refilter_v5e"
REPO="/home/saksham3/projects/AIRe/robocoin/batch_value_learning"
ERRFILE="/home/saksham3/logs/lego_baseline_watchdog_gcloud_err.txt"
POLL_SECONDS=180

TRAIN_COMMAND="python scripts/train.py lego_pi05_task_description_baseline --resume --batch-size=256 --checkpoint-base-dir=gs://saksham-euw4/checkpoints/robocoin/pi05_finetune --project-name=robocoin_value_learning --log-interval=100 --wandb-group='Policies'"

log() { echo "[$(date -u +%FT%TZ)] $*"; }
notify() { python ~/utils/slack.py "$*" >/dev/null 2>&1 || true; }

marker_present() {
    gcloud storage ls "$MARKER" >/dev/null 2>&1
}

# Echoes READY / PREEMPTED / UNKNOWN.
pod_status() {
    local state
    state=$(gcloud compute tpus tpu-vm describe "$POD" --zone="$ZONE" \
        --format='value(state)' 2>"$ERRFILE")
    if [ $? -eq 0 ] && [ -n "$state" ]; then
        if [ "$state" = "READY" ]; then echo "READY"; else echo "PREEMPTED"; fi
        return
    fi
    if grep -qiE "was not found|NOT_FOUND" "$ERRFILE" 2>/dev/null; then
        echo "PREEMPTED"
    else
        echo "UNKNOWN"
    fi
}

cd "$REPO" || exit 1
source /data/user_data/saksham3/vla/bin/activate

log "watchdog start: pod=$POD zone=$ZONE marker=$MARKER"

while true; do
    if marker_present; then
        log "done-marker present: run finished on the TPU"
        notify "lego_pi05_task_description_baseline: run COMPLETE (done-marker present). Watchdog exiting."
        exit 0
    fi

    status=$(pod_status)
    case "$status" in
        READY)
            log "pod READY"
            ;;
        UNKNOWN)
            # Almost always an expired credential or a network blip. Never relaunch here.
            log "pod state UNKNOWN (gcloud failed, NOT treated as preemption): $(head -2 "$ERRFILE" 2>/dev/null | tr '\n' ' ')"
            ;;
        PREEMPTED)
            log "pod $POD is gone/not READY -> taking over with run_on_tpu.py"
            notify "lego_pi05_task_description_baseline: pod $POD preempted. Watchdog handing off to run_on_tpu --spot --retry-on-preemption."
            python scripts/run_on_tpu.py \
                --tpu-type "$TPU_TYPE" \
                --spot \
                --retry-on-preemption \
                --done-marker "$MARKER" \
                --command "$TRAIN_COMMAND"
            rc=$?
            log "run_on_tpu.py exited with $rc"
            if [ $rc -eq 0 ] || marker_present; then
                notify "lego_pi05_task_description_baseline: run COMPLETE after relaunch. Watchdog exiting."
                exit 0
            fi
            notify "lego_pi05_task_description_baseline: relaunch via run_on_tpu exited $rc without completing. Watchdog exiting for a human look."
            exit 1
            ;;
    esac
    sleep "$POLL_SECONDS"
done
