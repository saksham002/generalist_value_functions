# TPU Guide

Everything TPU-related lives here. `CLAUDE.md` points at this file rather than
duplicating it.

## Runtime version per family — get this wrong and the pod is unusable

A pod built with the wrong runtime image comes up but cannot run the job. The
version is **family-specific** and is NOT interchangeable:

| family | `--accelerator-type` | `--version` (runtime)   | host RAM | busy check   |
| ------ | -------------------- | ----------------------- | -------- | ------------ |
| v4     | `v4-64`              | `tpu-ubuntu2204-base`   | 400 GB   | `/dev/accel*`|
| v5e    | `v5litepod-64`       | `v2-alpha-tpuv5-lite`   | 188 GB   | `/dev/vfio/*`|
| v6e    | `v6e-32`             | `v2-alpha-tpuv6e`       | 708 GB   | `/dev/accel*`|

Note the v5e naming mismatch: the TPU *type* is `v5e-64` but the
*accelerator-type* passed to `gcloud` is `v5litepod-64`. v4 and v6e use the same
string for both.

These four values are the **only** per-family constants left in the repo
(`TPU_FAMILY_SPECS` in `src/openpi/tpu/config.py`, plus the quota metric ID in
`quota.py`). They cannot be discovered because they are inputs to *creating* a pod
that does not exist yet. Everything else — zone, NFS, bucket, candidate zones — is
read live; see "What is discovered".

## Spot allocation — no creator jobs

`random/create_spot_pod_until_ready.sbatch` and its v6e twin have been **deleted**.
Nothing maintains a named pod any more. `run_on_tpu.py --spot` does the whole job:

1. **Reuse.** Look for an idle READY pod of the right shape, in any zone, whose name
   contains the owner (`saksham`). Another user's pod is never a candidate.
2. **Reclaim.** If an owned pod is busy, read its process owners across all workers.
   Processes owned by `saksham3` — or ownership that cannot be determined — mean the
   pod is left alone. Anything else is stale or foreign: `pkill -9 python` plus
   `rm -f /tmp/libtpu_lockfile`, then reuse.
3. **Race.** Otherwise create one queued resource per zone returned by
   `quota.spot_quota_zones` (live Cloud Quotas, US/EU only — 29 zones for v5e-64),
   keep the first pod that is READY *and* answers `ssh --worker=all`, delete the losers.

Because these are queued resources, a stocked-out zone waits in the queue rather than
erroring — which is what structurally replaces the creator's create/sleep/retry loop.

Two failure classes, easy to confuse:

- `code 8` — `"There is no more capacity in the zone ..."`. A stockout: the queued
  resource waits, the zone stays in the race.
- `code 429` — `"Quota limit '...' has been exceeded"`. A ceiling; retrying never
  clears it, so the zone is dropped for the rest of that race.

`race_timeout` defaults to 6 h, not `wait_timeout`'s 600 s: a real stockout lasted 5+
hours, and timing out deletes every candidate and loses queue position.

## What is discovered

Nothing below appears in the repo. If a filer or a quota grant changes, the next
launch picks it up with no code change.

| Fact | Source | Function |
| ---- | ------ | -------- |
| candidate zones | Cloud Quotas API | `quota.spot_quota_zones` |
| where a pod is | Cloud Asset Inventory (one call, all zones) | `discovery.find_pod` |
| accelerator, runtime | `tpu-vm describe` | `discovery.describe_pod` |
| NFS server + mount | Filestore instances, region-scoped | `discovery.nfs_for_zone` |
| write bucket | region abbreviation, created on demand | `buckets.ensure_regional_bucket` |

**NFS is region-scoped, not zone-scoped.** europe-west4 holds two filers and the
euw4-*a* pod mounts the euw4-*b* one (`10.155.154.42:/europe`). A zone-exact match
would pick `10.201.77.226:/europe_west4_a` and the job would fail much later looking
unrelated, so ties are broken by reading a live pod's mount table. A region with no
filer (us-central1, us-east1) means local-disk mode.

`gcloud compute tpus tpu-vm list` requires `--zone`; Cloud Asset Inventory does not,
which is why pod lookup uses it. It lags for just-created pods, so the race carries
the zone it created in rather than looking it up.

## GCS path localization (spot only)

A raced pod can land in any US/EU zone, so `buckets.localize_command` rewrites the
command's bucket prefixes for where it actually is. Per URI:

| Case | Action |
| ---- | ------ |
| already in the destination bucket | untouched |
| training command's output dir (does not exist yet) | rewrite prefix, no copy |
| training command's read path, replica exists | rewrite prefix |
| training command's read path, no replica | **raises** — replicate it first |
| other command: single object, or dir with `commit_success.txt` | copy wholesale |
| other command: anything else | **raises** — could be 165 GB |

Same-region writes are free; inter-region is $0.02/GiB. At 43 GB per checkpoint and a
save every 2 500 steps that is ~$0.86 per save, ~$7 across a 20 k-step run.

## Pods

Work is almost entirely on **spot pods, raced across regions**, so there is no stable
pod list and no fixed IPs to record: a pod's name, zone and address are decided by the
race and change on every preemption. Reach a pod through `gcloud compute tpus tpu-vm
ssh <name> --zone=<zone>`, taking both from the launcher log, never from a note here.

**Families in use**: v4, v5e, v6e. Everything per-family lives in `TPU_FAMILY_SPECS`
(`src/openpi/tpu/config.py`), which is the single source of truth — this table is a
convenience copy and the code wins if they disagree.

| Family | Runtime version       | Host RAM | Device glob    | Chips/worker |
| ------ | --------------------- | -------- | -------------- | ------------ |
| v4     | `tpu-ubuntu2204-base` | 400 GB   | `/dev/accel*`  | 4            |
| v5e    | `v2-alpha-tpuv5-lite` | 188 GB   | `/dev/vfio/*`  | 4            |
| v6e    | `v2-alpha-tpuv6e`     | 708 GB   | `/dev/accel*`  | 4            |

**Chip layout**: the number in the type string is the total chip count, and every
worker has 4 chips — except v4, whose shape number counts TensorCores at 8 per worker:

- v5litepod-32 → 8 workers, v5litepod-64 → 16, v5litepod-128 → 32
- v6e-32 → 8 workers
- v4-64 → 8 workers

`jax.local_device_count()` returns 4 per worker; `jax.device_count()` returns the
total after `jax.distributed.initialize()`.

To find the `process_index==0` worker on any pod, run `jax.distributed.initialize()`
and print `jax.process_index()` on every worker; the one reporting 0 is rank 0.

New-pod setup (NFS mount, uv env, GCS/wandb credentials, code sync) is handled by the
`scripts/run_on_tpu.py` execution path — do not do it by hand.

## NFS

**IMPORTANT**: `/nfs/` paths are mounted on the TPU pods, NOT on the local dev
machine. To inspect them, SSH into a pod first.

**uid mappings differ per worker.** On one v4 pod the same user was uid 2004, 2005,
2006, 2008, 2009 and 2010 across eight workers. Consequences:

- `ls -l` ownership on `/nfs/` is not trustworthy — cross-check by mtime or content
  before concluding another user owns something.
- Any scheme depending on a single owner across the pod fails for most workers.
  `chmod` requires *ownership*, so `chmod 777` from one worker does not help the
  others; the directory must be world-writable already, or the file must live on
  each worker's own local disk.

**Python env**: the project venv is at `/nfs/aidm_nfs/saksham3/uv/vla/`. System
`/usr/bin/python` lacks jax/tensorflow. Activate it or call its python directly.

**CRITICAL**: Never modify or delete files on a TPU pod without explicit
confirmation.

### PaliGemma weights (`pt_224.npz`)

`PaliGemmaWeightLoader` fetches from the Google-owned
`gs://vertex-model-garden-paligemma-us`, which returns
`Anonymous caller does not have storage.objects.get access` — an IAM issue no auth
fix resolves.

Two approaches that do **not** work:

- **Symlink into the cache.** `download.py` does `local_path.resolve()` then
  `relative_to(cache_dir)`, so a symlink out to `/nfs/...` raises
  `ValueError: ... is not in the subpath of ...`.
- **NFS as `OPENPI_DATA_HOME`.** `get_cache_dir()` chmods the cache root on every
  startup, and with per-worker uids most workers get
  `PermissionError: Operation not permitted`. This applies to every asset routed
  through `maybe_download`, not just the weights.

What works: a **real file in each worker's own local cache**, which that worker owns:

```bash
gcloud compute tpus tpu-vm ssh <pod> --zone=<zone> --worker=all \
  --command="mkdir -p ~/.cache/openpi/vertex-model-garden-paligemma-us/paligemma && \
             cp -f <source>/pt_224.npz \
                   ~/.cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
```

Source copies: `/data/group_data/rl/saksham3/vertex-model-garden-paligemma-us/paligemma/pt_224.npz`
on babel (11,693,352,432 bytes), and `/nfs/aidm_nfs/saksham3/gemma/2b/pt_224.npz` on
the europe-west4 NFS. Verify the byte count after copying — a truncated 11 GB file
fails in a much more confusing way.

## Launching

### Spot pods — `scripts/launch_preemptible.sh`

Now a ~75-line SLURM wrapper. It holds an allocation so the launcher survives the
login session and execs `run_on_tpu.py --spot --retry-on-preemption`; everything else
(allocation, preemption retry, RSS guard, completion) lives in Python. No `--pod`, no
`--zone`, no creator wiring: the race decides where the pod goes.

```bash
sbatch scripts/launch_preemptible.sh \
    --tpu-type v5e-64 \
    --done-marker gs://saksham-euw4/markers/<run-id> \
    --command "python scripts/train_value_function.py <config> --resume --batch-size=128 ..."
```

Unrecognised flags pass straight through to `run_on_tpu.py`.

**Completion is a marker, not a checkpoint.** The launcher appends
`&& gcloud storage cp /dev/null <marker>` to your command, so only a successful exit
records completion. A fresh launcher seeing the marker returns 0 without touching a
pod, then deletes it — a one-shot handshake that cannot go stale. `run_on_tpu` knows
nothing about checkpoints, which keeps it usable for non-training jobs.

**`--retry-on-preemption` requires `--spot`** and now says so rather than failing
obscurely: a reserved launch has no way to re-acquire capacity once its pod is gone.

**The RSS guard never retries.** A daemon thread probes every worker every 300 s; over
`host_ram_gb x 0.95` it kills the run, notifies and returns 1. Relaunching would
reproduce the same memory profile.

### Non-spot pods — `scripts/run_on_tpu.py`

**CRITICAL — never redirect its output to a log file.** No `> some.log 2>&1`; use
the Bash tool's `run_in_background=true`, which captures output itself. Redirecting
ties the run to the local session, so the remote job dies when the session ends.

The `--command` must carry the training args explicitly; they are not injected:

```bash
python scripts/run_on_tpu.py --tpu-type v5e-64 --tpu-name <pod> \
  --command "python scripts/train_value_function.py <config> --resume \
    --batch-size=256 \
    --checkpoint-base-dir=gs://saksham-euw4/checkpoints/robocoin/value_functions/<Q|V> \
    --project-name=robocoin_value_learning \
    --log-interval=100 \
    --wandb-group='Value Functions'"
```

Batch size: 128 on v5e-32, 256 on v5e-64, unless stated otherwise.

W&B groups: `Policies` (policy models), `Value Functions` (VF pre-training),
`Fine-Tuned Value Functions` (when `--fine-tune` is set).

**Design rule**: keep the TPU launch pipeline generic. Do not hard-code
script-specific behaviour into `run_on_tpu.py` or `src/openpi/tpu/`; put it in the
launched script or its wrapper.

## Operations

### Job logs

`run_on_tpu.py` writes `~/tpu_job_output.log` per worker and the exit code to
`~/tpu_job_exit_code`. v4 jobs launched via `--tpu_run` log to
`/tmp/tpu_job_output.log` instead.

```bash
gcloud compute tpus tpu-vm ssh <pod> --zone=<zone> --worker=0 \
  --command="tail -100 ~/tpu_job_output.log"
```

Not every metric reaches stdout — `batch/*_out_of_range_frac`, `param_norm` and
`grad_norm` only go to wandb, so query the run rather than grepping the pod log.

### Checking if a pod is free

```bash
# v5e
gcloud compute tpus tpu-vm ssh <pod> --zone=<zone> --worker=all \
  --command="sudo lsof /dev/vfio/0 2>/dev/null || echo free"
# v4 — /dev/vfio does not exist here, the v5e check returns a false "free"
gcloud compute tpus tpu-vm ssh <pod> --zone=us-central2-b --worker=all \
  --command="sudo lsof /dev/accel* 2>/dev/null | grep -v COMMAND | awk '{print \$1,\$3}' | sort -u || echo free"
```

Ask before acting if the busy process belongs to another user.

### Killing processes

`sudo pkill -9 python` under `--worker=all` is sufficient; don't bother with
`pkill -f tpc_launch_script`.

### Stale lockfile

JAX takes `/tmp/libtpu_lockfile` on init; a crash leaves it behind and blocks the
next run. Confirm the pod is free, then
`sudo rm -f /tmp/libtpu_lockfile` on all workers.

### Wedged pods

`state=READY` with `health=UNHEALTHY_TENSORFLOW` / `UNHEALTHY_MAINTENANCE` /
`TIMEOUT` does not clear on its own and the watchdog logs `not launchable`
indefinitely. Delete the pod so the creator rebuilds it.

An **expired credential** looks identical to an absent pod when stderr is discarded:
`gcloud describe` returns nothing either way. This once parked a run for nine hours,
the watchdog logging `pod absent` while the pod was fine. If every pod suddenly reads
as absent, check `gcloud auth list` before believing it. Tooling that polls pod state
should distinguish a non-zero exit from an empty-but-successful result.

## Not yet implemented: local-disk (no-NFS) pods

`spot_quota_zones` returns US zones with no Filestore, and `nfs_for_zone` correctly
reports local-disk mode for them — but the setup path cannot yet use one:

- `setup.py` calls `mount_nfs` and `fix_val_cache_permissions` unconditionally; both
  dereference `nfs_mount_path`, which is `None` there.
- `code_sync` syncs to `worker=0` only, which is correct only because NFS is shared.
- `working_dir` defaults to `/nfs/aidm_nfs/<user>/batch_value_learning`.

So a launch that wins a us-central1 or us-east1 zone will fail during setup. Until
that lands, keep spot launches to families whose quota sits in europe-west4 or
us-central2. `random/gaps.md` records a working per-worker home-directory environment
built by hand on `v6e-saksham-spot-32-0`.
