"""Keep writes in the pod's own region, and reads on its own continent.

Storage egress drives zone economics: same-region is free, inter-region within a
continent is $0.02/GiB, and crossing continents is $0.05/GiB. A 43 GB checkpoint written
across regions therefore costs ~$0.86 every save.

So when a race places a pod somewhere new:

- **writes** are redirected into that region's bucket, created if it does not exist;
- **reads** are left alone when the source is on the same continent, and redirected to a
  same-continent replica when it is not.

A fine-tune's base checkpoint is a read that the run cannot start without, so it is
copied into the destination bucket and the path rewritten to match.
"""

import logging
import re
import subprocess

from openpi.tpu.config import TPUConfigWithType
from openpi.tpu.config import canonical_bucket_for_region
from openpi.tpu.config import region_from_zone
from openpi.tpu.gcloud import run_gcloud

logger = logging.getLogger(__name__)

_GCS_TIMEOUT_SECONDS = 300
_COPY_TIMEOUT_SECONDS = 7200

_GCS_URI = re.compile(r"^gs://(?P<bucket>[^/]+)/?(?P<path>.*)$")


def _run(args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    """Run a `gcloud storage` command through the shared retry/auth layer.

    GCS calls flake for the same reasons the TPU control plane does, so they get the same
    treatment rather than a second, separate notion of what counts as a transient failure.
    """
    # args arrive as a full command line; run_gcloud supplies "gcloud" and --project itself.
    assert args[0] == "gcloud", f"expected a gcloud command, got {args[0]!r}"
    try:
        return run_gcloud(args[1:], check=False, timeout=timeout)
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


def redirect_write_uri(uri: str, *, bucket: str) -> str:
    """Point a write destination at ``bucket``, preserving its path."""
    _, path = split_uri(uri)
    return f"gs://{bucket}/{path}" if path else f"gs://{bucket}"


def copy_prefix(source_uri: str, destination_uri: str) -> None:
    """Copy a GCS prefix, routing cross-continent transfers through local disk.

    Bucket-to-bucket copies across continents are billed at the higher egress rate and are
    explicitly avoided in this project, so those go source -> local -> destination.
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
        raise NotImplementedError(
            f"Cross-continent copy {source_uri} -> {destination_uri} "
            f"({source_region} -> {destination_region}) must be staged through local disk; "
            "do it explicitly rather than as a side effect of a launch."
        )

    logger.info("Copying %s -> %s (%s -> %s)", source_uri, destination_uri, source_region, destination_region)
    result = _run(["gcloud", "storage", "rsync", "-r", source_uri, destination_uri], timeout=_COPY_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to copy {source_uri} -> {destination_uri}: {result.stderr.strip()}")


_GS_URI_IN_COMMAND = re.compile(r"gs://[A-Za-z0-9._\-]+(?:/[^\s'\"]*)?")

# Commands whose GCS arguments are structured: the checkpoint dir is an output to redirect,
# and the data paths are reads that must already exist near the pod rather than be copied.
TRAIN_ENTRYPOINTS = ("train.py", "train_value_function.py")

# Flags whose value is a destination the run writes to. Identified by name rather than by
# whether the path exists: a checkpoint base dir is shared across runs, so it usually does
# exist, and testing existence would misread a write as a read.
WRITE_FLAGS = ("--checkpoint-base-dir", "--assets-base-dir", "--done-marker")


def write_uris(command: str) -> set[str]:
    """URIs that appear as the value of a known write flag."""
    found: set[str] = set()
    for flag in WRITE_FLAGS:
        for match in re.finditer(rf"{re.escape(flag)}[=\s]+(gs://[^\s'\"]+)", command):
            found.add(match.group(1).rstrip("/"))
    return found


def find_gs_uris(command: str) -> tuple[str, ...]:
    """Every distinct gs:// URI appearing in a command, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in _GS_URI_IN_COMMAND.finditer(command):
        seen.setdefault(match.group(0).rstrip("/"), None)
    return tuple(seen)


_STEP_IN_COMMAND = re.compile(r"--(?:[\w.-]+\.)?step[=\s]+(\d+)")


def steps_named_in(command: str) -> set[int]:
    """Checkpoint steps the command explicitly selects (``--step``, ``--critic.step``, ...).

    Used to bound what a localize copies: a checkpoint root can hold many steps, including
    odd-numbered preemption saves, and a launch only ever restores the ones it names.
    """
    return {int(m.group(1)) for m in _STEP_IN_COMMAND.finditer(command)}


def is_train_command(command: str) -> bool:
    return any(entrypoint in command for entrypoint in TRAIN_ENTRYPOINTS)


def uri_exists(uri: str) -> bool:
    return _run(["gcloud", "storage", "ls", uri], timeout=_GCS_TIMEOUT_SECONDS).returncode == 0


def is_checkpoint_dir(uri: str) -> bool:
    """A committed checkpoint directory, identified by its commit marker."""
    return uri_exists(f"{uri.rstrip('/')}/commit_success.txt")


def is_object(uri: str) -> bool:
    """A single object rather than a prefix: listing it returns exactly itself."""
    result = _run(["gcloud", "storage", "ls", uri], timeout=_GCS_TIMEOUT_SECONDS)
    if result.returncode != 0:
        return False
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines == [uri.rstrip("/")]


def latest_checkpoint(checkpoint_root: str) -> str | None:
    """Newest committed checkpoint directly under ``checkpoint_root``, or None.

    Only committed ones count: a directory without commit_success.txt is a torn write and
    resuming from it is worse than starting over.
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


def committed_steps(checkpoint_root: str) -> list[int]:
    """Every committed step directly under ``checkpoint_root``, ascending.

    Both the commit marker and a params/ subtree are required, so a root is never reported
    as offering a step that no restore could use.

    Steps are not necessarily round: the preemption path in train_value_function.py saves
    at whatever step SIGTERM landed on, which is why roots can carry entries like 15293
    alongside the usual save-interval multiples.
    """
    result = _run(["gcloud", "storage", "ls", f"{checkpoint_root.rstrip('/')}/"], timeout=_GCS_TIMEOUT_SECONDS)
    if result.returncode != 0:
        return []
    steps: list[int] = []
    for line in result.stdout.splitlines():
        tail = line.strip().rstrip("/").rsplit("/", 1)[-1]
        if not tail.isdigit():
            continue
        step_uri = f"{checkpoint_root.rstrip('/')}/{tail}"
        if is_checkpoint_dir(step_uri) and uri_exists(f"{step_uri}/params/"):
            steps.append(int(tail))
    return sorted(steps)


def carry_checkpoints(source_root: str, destination_root: str) -> str | None:
    """Copy the newest committed checkpoint from one region's bucket to another.

    A spot run that is preempted and re-raced can land in a different region, where
    localization points its writes at that region's bucket — an empty one. Without this
    the run silently restarts from step 0 while its progress sits in the old bucket.

    Returns the destination path copied to, or None when there was nothing to carry or the
    destination already has newer progress.
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
    copy_prefix(source, destination)
    return destination


def localize_command(command: str, tpu_config: TPUConfigWithType) -> tuple[str, dict[str, str]]:
    """Rewrite a command's GCS buckets for the region its pod actually landed in.

    Spot allocation can place a pod in any US or EU zone, so a command written against
    one bucket would otherwise read and write across regions for the whole run.

    What happens to each URI depends on what it is:

    - already in the destination bucket: untouched;
    - a training command's output directory: the bucket prefix is swapped, nothing copied,
      because the run creates that content itself;
    - a training command's read paths: swapped only if the same-region replica exists,
      otherwise an error naming what to replicate — silently copying a dataset as a side
      effect of a launch is exactly the cost this is meant to avoid;
    - any other command's URI: copied wholesale when it is a single object or a committed
      checkpoint directory, and rejected otherwise, so an arbitrary prefix is never dragged
      across regions by accident.

    Returns the rewritten command and a map of original URI to replacement.
    """
    region = region_from_zone(tpu_config.zone)
    destination = ensure_regional_bucket(region, resource_owner=tpu_config.resource_owner)
    training = is_train_command(command)
    declared_writes = write_uris(command)
    rewrites: dict[str, str] = {}

    for uri in find_gs_uris(command):
        source_bucket, path = split_uri(uri)
        if source_bucket == destination:
            continue
        target = f"gs://{destination}/{path}" if path else f"gs://{destination}"

        if training:
            # A declared write destination, or a path that does not exist yet: redirect it
            # and copy nothing, because the run produces that content itself.
            if uri in declared_writes or not uri_exists(uri):
                rewrites[uri] = target
                continue

            # An existing path is a read. Inter-region reads within a continent are cheap
            # relative to duplicating a dataset, so leave those alone; only a
            # cross-continent read is worth redirecting, and only to a replica that exists.
            source_region = bucket_region(source_bucket)
            if source_region is None:
                raise ValueError(f"Read source gs://{source_bucket} does not exist")
            if continent_of(source_region) == continent_of(region):
                logger.info("Leaving %s alone: same continent as %s", uri, region)
                continue
            if not uri_exists(target):
                raise ValueError(
                    f"{uri} is in {source_region}, the pod is in {region}, and there is no "
                    f"same-continent replica at {target}. Replicate it first; a launch will "
                    "not copy a dataset for you."
                )
            rewrites[uri] = target
            continue

        if is_object(uri) or is_checkpoint_dir(uri):
            copy_prefix(uri, target)
            rewrites[uri] = target
            continue

        # A checkpoint ROOT: the commit marker lives one level down, in each step. Serving
        # needs this shape — a critic is loaded by an orbax CheckpointManager rooted here
        # and selected by step, so the command cannot name a step directory instead. Copy
        # the committed steps individually rather than the prefix, which both skips torn
        # writes and keeps the step layout the manager expects.
        steps = committed_steps(uri)
        # A root can hold far more than the launch needs (10k + 20k + odd preemption
        # saves). Copy only the steps the command actually selects; fall back to all of
        # them when it names none, since then any of them may be restored.
        named = steps_named_in(command)
        if named:
            wanted = sorted(set(steps) & named)
            if wanted:
                steps = wanted
        if steps:
            logger.info("Localizing checkpoint root %s: copying committed steps %s", uri, steps)
            for step in steps:
                copy_prefix(f"{uri.rstrip('/')}/{step}", f"{target.rstrip('/')}/{step}")
            rewrites[uri] = target
            continue

        raise ValueError(
            f"Refusing to localize {uri}: it is neither a single object, a committed "
            "checkpoint directory, nor a root containing committed steps, so copying it "
            "could move an arbitrary amount of data. Replicate it explicitly if that is "
            "what you want."
        )

    localized = command
    for original, replacement in rewrites.items():
        localized = localized.replace(original, replacement)
    if rewrites:
        logger.info("Localized %d GCS path(s) for %s: %s", len(rewrites), region, rewrites)
    return localized, rewrites


def localize_paths(
    tpu_config: TPUConfigWithType,
    *,
    checkpoint_base_dir: str | None = None,
    read_uris: tuple[str, ...] = (),
    fine_tune_base_dir: str | None = None,
) -> dict[str, str]:
    """Rewrite a launch's GCS paths for the region a pod actually landed in.

    Returns a mapping of original URI to rewritten URI; unchanged paths are omitted.
    Reads on the pod's own continent are deliberately left alone — inter-region reads
    within a continent are cheap relative to the cost of duplicating a dataset.
    """
    region = region_from_zone(tpu_config.zone)
    bucket = ensure_regional_bucket(region, resource_owner=tpu_config.resource_owner)
    rewrites: dict[str, str] = {}

    if checkpoint_base_dir is not None:
        redirected = redirect_write_uri(checkpoint_base_dir, bucket=bucket)
        if redirected != checkpoint_base_dir:
            rewrites[checkpoint_base_dir] = redirected

    if fine_tune_base_dir is not None:
        redirected = redirect_write_uri(fine_tune_base_dir, bucket=bucket)
        if redirected != fine_tune_base_dir:
            copy_prefix(fine_tune_base_dir, redirected)
            rewrites[fine_tune_base_dir] = redirected

    for uri in read_uris:
        source_bucket, _ = split_uri(uri)
        source_region = bucket_region(source_bucket)
        if source_region is None:
            raise ValueError(f"Read source gs://{source_bucket} does not exist")
        if continent_of(source_region) == continent_of(region):
            continue
        redirected = redirect_write_uri(uri, bucket=bucket)
        copy_prefix(uri, redirected)
        rewrites[uri] = redirected

    return rewrites
