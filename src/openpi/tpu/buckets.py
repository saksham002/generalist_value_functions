"""GCS primitives for keeping a run's data near the pod that is using it.

Storage egress drives zone economics: same-region is free, inter-region within a continent
is $0.02/GiB, and crossing continents is $0.05/GiB. A 43 GB checkpoint written across
regions therefore costs ~$0.86 every save.

This module is only the mechanism — bucket lookup, existence tests, committed-checkpoint
discovery and copying. The *policy* about which URI belongs where lives in
:mod:`openpi.tpu.launch`, where it can be decided from a parsed command rather than
re-derived from a string at each call site.
"""

import logging
import re
import subprocess

from openpi.tpu.config import canonical_bucket_for_region
from openpi.tpu.gcloud import run_gcloud

logger = logging.getLogger(__name__)

_GCS_TIMEOUT_SECONDS = 300
_COPY_TIMEOUT_SECONDS = 7200

# gcloud storage sizes its worker pool from the host's CPU count and buffers each shard in
# memory, which is more than either launcher host can absorb: a default-parallelism copy of a
# ~40 GiB checkpoint in ~1 GB shards took down the whole WSL2 VM on the laptop, and the GCE
# launcher is a 2 GB e2-small. Sliced downloads are disabled for the same reason -- they are
# what turns one object into several concurrent in-memory buffers. The carry happens once per
# launch and is not on any critical path, so throughput is the cheap thing to give up.
_COPY_ENV = {
    "CLOUDSDK_STORAGE_PROCESS_COUNT": "1",
    "CLOUDSDK_STORAGE_THREAD_COUNT": "1",
    "CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD": "0",
}

_GCS_URI = re.compile(r"^gs://(?P<bucket>[^/]+)/?(?P<path>.*)$")

# Inter-continent egress, used to put a number on a refusal rather than an adjective.
CROSS_CONTINENT_USD_PER_GIB = 0.05

# Written by train_value_function.py / train.py next to the checkpoint steps; read back on
# resume, so a carried checkpoint without it fails at init_wandb.
WANDB_ID_FILENAME = "wandb_id.txt"


def _run(args: list[str], *, timeout: int, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a `gcloud storage` command through the shared retry/auth layer.

    GCS calls flake for the same reasons the TPU control plane does, so they get the same
    treatment rather than a second, separate notion of what counts as a transient failure.
    """
    # args arrive as a full command line; run_gcloud supplies "gcloud" and --project itself.
    assert args[0] == "gcloud", f"expected a gcloud command, got {args[0]!r}"
    try:
        return run_gcloud(args[1:], check=False, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 1, "", f"timed out after {timeout}s")


def split_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/path`` into ``(bucket, path)``."""
    match = _GCS_URI.match(uri)
    if match is None:
        raise ValueError(f"Not a GCS URI: {uri!r}")
    return match["bucket"], match["path"]


def bucket_region(bucket: str) -> str | None:
    """Return a bucket's region in lowercase, or None if it does not exist."""
    result = _run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket}", "--format=value(location)"],
        timeout=_GCS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return None
    location = result.stdout.strip().lower()
    return location or None


def continent_of(region: str) -> str:
    """Coarse continent key; pods only ever run in the US or the EU."""
    return "eu" if region.startswith("europe") else region.partition("-")[0]


def ensure_regional_bucket(region: str, *, resource_owner: str) -> str:
    """Return the write bucket for ``region``, creating it if absent.

    Idempotent: an existing bucket in the right region is returned untouched. An existing
    bucket in the *wrong* region raises rather than silently writing across regions.
    """
    bucket = canonical_bucket_for_region(region, resource_owner=resource_owner)
    existing_region = bucket_region(bucket)
    if existing_region == region:
        return bucket
    if existing_region is not None:
        raise ValueError(
            f"Bucket gs://{bucket} already exists in {existing_region!r}, not {region!r}; "
            "refusing to write across regions."
        )

    logger.info("Creating write bucket gs://%s in %s", bucket, region)
    result = _run(
        [
            "gcloud",
            "storage",
            "buckets",
            "create",
            f"gs://{bucket}",
            f"--location={region}",
            "--uniform-bucket-level-access",
        ],
        timeout=_GCS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create gs://{bucket} in {region}: {result.stderr.strip()}")
    return bucket


def prefix_size_bytes(uri: str) -> int:
    """Total size in bytes of every object under a GCS prefix (0 for an empty prefix)."""
    result = _run(["gcloud", "storage", "du", "-s", uri.rstrip("/")], timeout=_GCS_TIMEOUT_SECONDS)
    if result.returncode != 0 or not result.stdout.strip():
        return 0
    return int(result.stdout.split()[0])


def uri_exists(uri: str) -> bool:
    return _run(["gcloud", "storage", "ls", uri], timeout=_GCS_TIMEOUT_SECONDS).returncode == 0


def is_checkpoint_dir(uri: str) -> bool:
    """A committed checkpoint directory, identified by its commit marker."""
    return uri_exists(f"{uri.rstrip('/')}/commit_success.txt")


def copy_prefix(source_uri: str, destination_uri: str, *, allow_cross_continent: bool = False) -> None:
    """Copy a GCS prefix.

    Bucket-to-bucket copies across continents are billed at the higher egress rate, so they
    are refused unless the caller opts in with ``allow_cross_continent``. Cost is the
    caller's decision, made explicitly, rather than inferred from size — an earlier
    threshold assumed a value-function checkpoint is a few hundred MB, but a PaliGemma
    critic with optimizer state is ~40 GiB, so it blocked exactly the carry it was meant to
    permit.
    """
    source_bucket, _ = split_uri(source_uri)
    destination_bucket, _ = split_uri(destination_uri)
    source_region = bucket_region(source_bucket)
    destination_region = bucket_region(destination_bucket)
    if source_region is None:
        raise ValueError(f"Source bucket gs://{source_bucket} does not exist")
    if destination_region is None:
        raise ValueError(f"Destination bucket gs://{destination_bucket} does not exist")

    if continent_of(source_region) != continent_of(destination_region):
        gibibytes = prefix_size_bytes(source_uri) / 2**30
        if not allow_cross_continent:
            raise PermissionError(
                f"Cross-continent copy {source_uri} -> {destination_uri} "
                f"({source_region} -> {destination_region}, {gibibytes:.1f} GiB) refused: "
                "pass --allow-cross-continent-checkpoint-transfer to the launcher to permit it "
                f"(~${gibibytes * CROSS_CONTINENT_USD_PER_GIB:.2f})."
            )
        logger.warning(
            "Cross-continent copy %s -> %s (%s -> %s): %.2f GiB (~$%.2f), explicitly allowed",
            source_uri,
            destination_uri,
            source_region,
            destination_region,
            gibibytes,
            gibibytes * CROSS_CONTINENT_USD_PER_GIB,
        )

    logger.info("Copying %s -> %s (%s -> %s)", source_uri, destination_uri, source_region, destination_region)
    result = _run(
        ["gcloud", "storage", "rsync", "-r", source_uri, destination_uri],
        timeout=_COPY_TIMEOUT_SECONDS,
        env=_COPY_ENV,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to copy {source_uri} -> {destination_uri}: {result.stderr.strip()}")


def latest_checkpoint(checkpoint_root: str) -> str | None:
    """Newest committed checkpoint directly under ``checkpoint_root``, or None.

    Only committed ones count: a directory without commit_success.txt is a torn write and
    resuming from it is worse than starting over. Steps are not necessarily round — the
    preemption path in train_value_function.py saves at whatever step SIGTERM landed on.
    """
    result = _run(["gcloud", "storage", "ls", f"{checkpoint_root.rstrip('/')}/"], timeout=_GCS_TIMEOUT_SECONDS)
    if result.returncode != 0:
        return None
    steps: list[int] = []
    for line in result.stdout.splitlines():
        tail = line.strip().rstrip("/").rsplit("/", 1)[-1]
        if tail.isdigit():
            steps.append(int(tail))
    for step in sorted(steps, reverse=True):
        candidate = f"{checkpoint_root.rstrip('/')}/{step}"
        if is_checkpoint_dir(candidate):
            return candidate
    return None


def carry_checkpoints(source_root: str, destination_root: str, *, allow_cross_continent: bool = False) -> str | None:
    """Copy the newest committed checkpoint from one region's bucket to another.

    Returns the destination path copied to, or None when there was nothing to carry or the
    destination already has equal or newer progress.
    """
    if source_root.rstrip("/") == destination_root.rstrip("/"):
        return None
    source = latest_checkpoint(source_root)
    if source is None:
        logger.info("Nothing to carry from %s", source_root)
        return None
    step = source.rsplit("/", 1)[-1]

    existing = latest_checkpoint(destination_root)
    if existing is not None and int(existing.rsplit("/", 1)[-1]) >= int(step):
        logger.info("%s already has step %s; not carrying %s", destination_root, existing.rsplit("/", 1)[-1], step)
        return None

    destination = f"{destination_root.rstrip('/')}/{step}"
    logger.info("Carrying checkpoint %s -> %s", source, destination)
    copy_prefix(source, destination, allow_cross_continent=allow_cross_continent)

    # The trainer resumes its wandb run from <root>/wandb_id.txt whenever a checkpoint
    # exists, so a carried step without the id file fails at init_wandb. Keep an existing
    # destination id: the run may already have been logged from this region.
    source_wandb_id = f"{source_root.rstrip('/')}/{WANDB_ID_FILENAME}"
    destination_wandb_id = f"{destination_root.rstrip('/')}/{WANDB_ID_FILENAME}"
    if uri_exists(source_wandb_id) and not uri_exists(destination_wandb_id):
        logger.info("Carrying %s -> %s", source_wandb_id, destination_wandb_id)
        result = _run(["gcloud", "storage", "cp", source_wandb_id, destination_wandb_id], timeout=_GCS_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to copy {source_wandb_id} -> {destination_wandb_id}: {result.stderr.strip()}")
    return destination


def marker_exists(marker: str) -> bool:
    return uri_exists(marker)


def remove_marker(marker: str) -> None:
    _run(["gcloud", "storage", "rm", marker], timeout=_GCS_TIMEOUT_SECONDS)
