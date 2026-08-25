#!/bin/bash
# Make a pod's validation-episode cache identical on every worker.
#
# Intended as a --post-launch-hook for run_on_tpu.py, which supplies TPU_NAME, TPU_ZONE,
# TPU_PROJECT, TPU_WORKER_COUNT and TPU_COMMAND.
#
# Why this is needed: validation plotting is a collective over the mesh — every host must
# enter the value computation together. Which episodes a host has is decided by whichever
# ones it cached, and only the host running JAX process 0 populates the cache. On a shared
# filesystem that is harmless, because the others read back what it wrote. On a per-worker
# path (anything under /home) they cannot, so they find nothing, take the "no valid frames"
# early return, and never enter the collective. The hosts that did enter then wait for
# peers that never arrive, and the TPU runtime halts the core with "an unexpected peer
# shows up in the launch group" — killing a healthy run at its first plot step.
#
# This is therefore a no-op unless the run's cache directory is a /home path.
#
# The copy is staged through a same-region GCS object rather than scp: a ~550 MB tarball
# pulled by every worker in parallel takes seconds, whereas scp through the launching host
# has repeatedly truncated at a few hundred MB.

set -u

POD="${TPU_NAME:-}"
ZONE="${TPU_ZONE:-}"
PROJECT="${TPU_PROJECT:-cmu-aidm-v2}"
COMMAND="${TPU_COMMAND:-}"
LOG="${VAL_CACHE_SYNC_LOG:-$HOME/logs/sync_val_cache_workers.log}"
POLL_SECONDS="${VAL_CACHE_POLL_SECONDS:-60}"
# Long enough to cover dataset iteration on a cold pod, short enough that a run which will
# never build a cache does not leave a poller alive for the rest of the job.
MAX_WAIT_SECONDS="${VAL_CACHE_MAX_WAIT_SECONDS:-5400}"

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -Is) [$POD] $*" >> "$LOG"; }

if [ -z "$POD" ] || [ -z "$ZONE" ]; then
    log "missing TPU_NAME/TPU_ZONE; nothing to do"
    exit 0
fi

# The fine-tune spelling wins when both appear: FineTuneConfig.apply_overrides copies its
# validation_cache_dir onto the TrainConfig, so the top-level flag is the one that loses.
CACHE=$(printf '%s' "$COMMAND" | grep -oE -- '--fine-tune\.validation-cache-dir[= ][^ ]+' | tail -1 | sed -E 's/^[^= ]+[= ]//')
if [ -z "$CACHE" ]; then
    CACHE=$(printf '%s' "$COMMAND" | grep -oE -- '--validation-cache-dir[= ][^ ]+' | tail -1 | sed -E 's/^[^= ]+[= ]//')
fi
CACHE=${CACHE%/}

# The launcher rewrites an NFS path to "$HOME/..." for a pod with no filer -- literally
# that string, expanded by the shell on the pod rather than here. "/home/..." is accepted
# too for a path written that way by hand. Anything else is on shared storage and needs
# nothing doing.
case "$CACHE" in
    '$HOME'/*|/home/*) ;;
    "")  log "no validation-cache-dir in command; nothing to do"; exit 0 ;;
    *)   log "cache dir $CACHE is shared, not per-worker; nothing to do"; exit 0 ;;
esac

# Same region as the pod, so the staging round trip stays local and free. Derived rather
# than listed: a hardcoded map silently gives up ("no staging bucket known") in exactly the
# zones a widened race newly reaches, which is when this hook matters most. Mirrors
# config.region_abbreviation: europe-west4 -> euw4, us-south1 -> uss1, us-central2 -> usc2.
REGION=${ZONE%-*}
LOCALITY=${REGION%%-*}
DIRECTION=${REGION#*-}
case "$LOCALITY" in europe*) CONTINENT=eu ;; *) CONTINENT=$LOCALITY ;; esac
BUCKET="saksham-${CONTINENT}$(printf '%s' "$DIRECTION" | cut -c1)$(printf '%s' "$DIRECTION" | tr -cd '0-9')"
if ! gcloud storage ls "gs://$BUCKET/" >/dev/null 2>&1; then
    log "staging bucket gs://$BUCKET for zone $ZONE does not exist; giving up"
    exit 0
fi

log "watching $CACHE (staging via gs://$BUCKET/tmp)"

worker_counts() {
    timeout 300 gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --project="$PROJECT" --worker=all \
        --command="echo \"PKL \$(hostname) \$(ls $CACHE/*.pkl 2>/dev/null | wc -l)\"" 2>/dev/null | grep -a '^PKL'
}

deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    # A pod that has gone away ends the watch: the retry gets its own hook invocation.
    state=$(timeout 120 gcloud compute tpus tpu-vm describe "$POD" --zone="$ZONE" --project="$PROJECT" \
        --format="value(state)" 2>/dev/null)
    case "$state" in
        READY|"") : ;;
        *) log "pod state=$state; ending watch"; exit 0 ;;
    esac

    counts=$(worker_counts)
    if [ -z "$counts" ]; then sleep "$POLL_SECONDS"; continue; fi
    src=$(echo "$counts" | awk '$3 > 0 {print $2}' | head -1)
    empty=$(echo "$counts" | awk '$3 == 0' | wc -l)

    if [ -n "$src" ] && [ "$empty" -eq 0 ]; then
        log "all workers already hold the cache; done"
        exit 0
    fi
    if [ -n "$src" ]; then
        idx=${src##*-w-}
        stage="gs://$BUCKET/tmp/valcache_$(basename "$CACHE")_$(date +%s).tgz"
        log "cache built on worker $idx; $empty worker(s) empty; staging via $stage"
        if ! timeout 900 gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --project="$PROJECT" --worker="$idx" \
            --command="cd \$(dirname $CACHE) && tar czf /tmp/valcache_sync.tgz \$(basename $CACHE) && gcloud storage cp /tmp/valcache_sync.tgz $stage" >> "$LOG" 2>&1; then
            log "staging failed; retrying next poll"
            sleep "$POLL_SECONDS"; continue
        fi
        timeout 900 gcloud compute tpus tpu-vm ssh "$POD" --zone="$ZONE" --project="$PROJECT" --worker=all \
            --command="mkdir -p \$(dirname $CACHE) && cd \$(dirname $CACHE) && gcloud storage cp $stage /tmp/valcache_pull.tgz >/dev/null 2>&1 && tar xzf /tmp/valcache_pull.tgz && echo \"SYNCED \$(hostname) \$(ls $CACHE/*.pkl 2>/dev/null | wc -l)\"" >> "$LOG" 2>&1
        after=$(worker_counts)
        still_empty=$(echo "$after" | awk '$3 == 0' | wc -l)
        if [ "$still_empty" -eq 0 ]; then
            log "fan-out complete: every worker holds $(echo "$after" | awk '{print $3}' | head -1) pkls"
            exit 0
        fi
        log "fan-out left $still_empty worker(s) empty; retrying next poll"
    fi
    sleep "$POLL_SECONDS"
done
log "gave up after ${MAX_WAIT_SECONDS}s without a complete cache"
