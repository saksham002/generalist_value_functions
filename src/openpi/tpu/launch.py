"""What a launch IS, and what it becomes once a pod is known.

The command used to be an opaque string that eight separate regexes took apart and a
final unanchored ``str.replace`` put back together. Here it is parsed exactly once into a
:class:`CommandPlan` — a list of located substrings, each tagged with what it means — and
every rewrite is a splice at a known offset. A URI that happens to be a string prefix of
another can no longer corrupt it, and the same parse answers every downstream question.

:class:`LaunchConfig` is what the operator asked for. :class:`ResolvedLaunch` is what that
becomes once a pod exists, and is where the four placement guarantees are enforced:

1. **reads stay on the pod's continent.** A read already on this continent is left alone —
   inter-region reads within a continent are cheap next to duplicating a dataset. A read
   on the other continent is redirected to its same-continent replica, and if there is no
   replica the launch fails rather than quietly paying inter-continent egress for the
   whole run.
2. **writes stay in the pod's own region**, in that region's bucket, created if absent.
3. **the carry** is the one sanctioned copy: the newest committed checkpoint under
   ``--carry-checkpoints-from`` (or the previous attempt's root) is copied into wherever
   this launch actually writes, and crossing continents to do it requires an explicit flag.
4. **NFS paths become local paths** when the pod has no filer, so a command written for a
   shared-filesystem pod means the same thing on a local-disk one.

The run id is derived here too, from the parts of the plan that identify the run and not
where it happens to be running — so the same run keeps its identity across a re-race into
another region, which is what makes the certificate in :mod:`openpi.tpu.certificate` a
usable resumption key.
"""

from collections.abc import Iterable, Mapping
import dataclasses
import enum
import functools
import hashlib
import logging
from pathlib import Path
import re

from openpi.tpu import buckets
from openpi.tpu.config import DEFAULT_PROJECT
from openpi.tpu.config import DEFAULT_TPU_USER
from openpi.tpu.config import PodConfig
from openpi.tpu.config import RemoteLayout

logger = logging.getLogger(__name__)

# Google's own bucket denies anonymous reads, so pods pull from a copy in this project.
# The path is bucket-relative: which bucket it comes from is decided per pod, so a US pod
# reads a US copy instead of pulling it across the Atlantic on every setup.
PALIGEMMA_WEIGHTS_PATH = "base_checkpoints/paligemma/pt_224.npz"

# Commands whose GCS arguments are structured: a checkpoint dir is an output, and the data
# paths are reads that must already exist near the pod.
TRAIN_ENTRYPOINTS = ("train.py", "train_value_function.py")

# Flags whose value is a destination the run writes to. Identified by name rather than by
# whether the path exists: a checkpoint base dir is shared across runs, so it usually does
# exist, and testing existence would misread a write as a read.
#
# --done-marker is deliberately absent. It is a handshake between the launcher and the job
# and both sides have to name the same object regardless of which region the pod landed
# in; it is one empty object, so leaving it un-localized costs nothing.
WRITE_FLAGS = ("--checkpoint-base-dir", "--assets-base-dir")

# One bucket per continent holds the datasets and counterfactual-action stores, rather than
# one per region: a dataset is read by pods in whichever zone the race wins, so replicating
# it per region would mean many copies of the same tens of GB.
HUB_REGION_BY_CONTINENT = {"us": "us-central2", "eu": "europe-west4"}

_GS_URI = re.compile(r"gs://[A-Za-z0-9._\-]+(?:/[^\s'\"]*)?")
# A key containing "data" or "store" and ending in "dir": --data.rlds-data-dir,
# --fine-tune.data-factory.counterfactual-action-store-dir, --data.assets.assets-dir.
_SHARED_DIR_FLAG = re.compile(r"--([\w.-]*(?:data|store)[\w.-]*dir)[=\s]+(gs://[^\s'\"]+)")
# Any flag whose value is an absolute POSIX path. Narrowed to NFS paths at classification
# time; matching broadly here keeps the pattern from having to know every flag name.
_PATH_FLAG = re.compile(r"--([\w.-]+)[=\s]+(/[^\s'\"]+)")
# Matches the fine-tune selector but not its override flags: `--fine-tune <name>` and
# `--fine-tune=<name>` are separated by whitespace or '=', whereas
# `--fine-tune.data-factory.rlds-data-dir` continues with a '.'.
_FINE_TUNE = re.compile(r"--fine-tune[=\s]+([\w-]+)")
_EXP_NAME = re.compile(r"--exp-name[=\s]+(\S+)")
_FLAG_VALUE = r"{flag}[=\s]+(\S+)"


class UriRole(enum.Enum):
    """What a GCS URI in a command is for, which decides where it may be moved."""

    WRITE = "write"
    """An output the run creates itself; redirected to the pod's region, never copied."""

    SHARED_READ = "shared_read"
    """A dataset or store; resolves to the continent's hub bucket."""

    READ = "read"
    """Any other input; must resolve to something on the pod's continent."""


@dataclasses.dataclass(frozen=True)
class Span:
    """A located substring of the command, so a rewrite is a splice, not a search."""

    start: int
    end: int
    text: str


@dataclasses.dataclass(frozen=True)
class UriRef:
    span: Span
    role: UriRole


@dataclasses.dataclass(frozen=True)
class PathRef:
    """An absolute filesystem path in the command, and the flag that carried it."""

    span: Span
    flag: str


@dataclasses.dataclass(frozen=True)
class CommandPlan:
    """A command taken apart once, with every interesting substring located."""

    command: str
    entrypoint: str | None
    config_name: str | None
    exp_name: str | None
    fine_tune: str | None
    checkpoint_base_dir: str | None
    uris: tuple[UriRef, ...]
    paths: tuple[PathRef, ...]

    @classmethod
    def parse(cls, command: str) -> "CommandPlan":
        tokens = command.split()
        entrypoint = next((token for token in tokens if token.endswith(".py")), None)
        config_name = None
        if entrypoint is not None:
            index = tokens.index(entrypoint)
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                config_name = tokens[index + 1]

        write_values = {
            match.group(1)
            for flag in WRITE_FLAGS
            for match in re.finditer(_FLAG_VALUE.format(flag=re.escape(flag)), command)
        }
        shared_values = {match.group(2).rstrip("/") for match in _SHARED_DIR_FLAG.finditer(command)}

        uris: list[UriRef] = []
        for match in _GS_URI.finditer(command):
            text = match.group(0)
            trimmed = text.rstrip("/")
            if trimmed in write_values or text in write_values:
                role = UriRole.WRITE
            elif trimmed in shared_values:
                role = UriRole.SHARED_READ
            else:
                role = UriRole.READ
            uris.append(UriRef(span=Span(match.start(), match.end(), text), role=role))

        paths = tuple(
            PathRef(span=Span(match.start(2), match.end(2), match.group(2)), flag=match.group(1))
            for match in _PATH_FLAG.finditer(command)
        )

        return cls(
            command=command,
            entrypoint=entrypoint,
            config_name=config_name,
            exp_name=(m.group(1) if (m := _EXP_NAME.search(command)) else None),
            fine_tune=(m.group(1) if (m := _FINE_TUNE.search(command)) else None),
            checkpoint_base_dir=next(iter(sorted(write_values & _checkpoint_values(command))), None),
            uris=tuple(uris),
            paths=paths,
        )

    @property
    def is_training(self) -> bool:
        return any(entrypoint in self.command for entrypoint in TRAIN_ENTRYPOINTS)

    @property
    def checkpoint_root(self) -> str | None:
        """The per-run checkpoint directory this command writes to.

        ``--checkpoint-base-dir`` is shared across runs; the run's own tree is
        ``<base>/<config>/<exp-name or config>``, matching ``TrainConfig.checkpoint_dir``.
        """
        if not self.is_training or self.checkpoint_base_dir is None or self.config_name is None:
            return None
        return f"{self.checkpoint_base_dir.rstrip('/')}/{self.config_name}/{self.exp_name or self.config_name}"

    def rewrite(self, replacements: Mapping[Span, str]) -> str:
        """Apply located replacements, right to left so earlier offsets stay valid."""
        rewritten = self.command
        for span in sorted(replacements, key=lambda s: s.start, reverse=True):
            rewritten = rewritten[: span.start] + replacements[span] + rewritten[span.end :]
        return rewritten

    def run_id(self, *, user: str) -> str:
        """A stable identity for this run, independent of where it is running.

        Deliberately excludes bucket names and zones: a re-raced run is localized into a
        different region's bucket, and an id that moved with it would stop matching the
        certificate left by its own earlier attempt — which is the one thing the id exists
        to make possible. The checkpoint base dir contributes its *path* only, for the same
        reason.
        """
        base_path = (
            buckets.split_uri(self.checkpoint_base_dir)[1]
            if _is_gcs(self.checkpoint_base_dir)
            else (self.checkpoint_base_dir or "")
        )
        identity = "|".join(
            [
                user,
                Path(self.entrypoint or "").name,
                self.config_name or "",
                self.exp_name or "",
                self.fine_tune or "",
                base_path.rstrip("/"),
            ]
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:8]
        label = self.exp_name or self.config_name or Path(self.entrypoint or "run").stem
        return f"{_slug(label)}-{digest}"


def _checkpoint_values(command: str) -> set[str]:
    return {match.group(1) for match in re.finditer(_FLAG_VALUE.format(flag="--checkpoint-base-dir"), command)}


def _is_gcs(uri: str | None) -> bool:
    return bool(uri) and uri.startswith("gs://")


def _slug(text: str, limit: int = 40) -> str:
    """A shell-safe, readable stem for a run id."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return (cleaned[:limit].rstrip("-._") or "run").lower()


@dataclasses.dataclass
class LaunchConfig:
    """Everything the operator asked for, before a pod exists."""

    command: str
    """Command to run on the TPU."""

    tpu_type: str | None = None
    """TPU type (e.g. 'v6e-8', 'v5e-128'). Optional for a spot launch: with none given, the
    race spans every shape in ``DEFAULT_SPOT_TPU_TYPES`` and keeps whichever lands first. A
    reserved launch still requires it, since a reserved pod is looked up by shape."""

    tpu_name: str | None = None
    """Specific TPU name to use. If not specified, finds or creates one."""

    user: str = DEFAULT_TPU_USER
    """Key in TPU_USERS; determines resource names and the regional write bucket."""

    project: str = DEFAULT_PROJECT
    """GCP project searched for pods, quota and Filestore instances."""

    spot: bool = False
    """Race every zone with live spot quota instead of reusing a reserved pod."""

    region: str | None = None
    """Restrict a launch to one region, e.g. 'europe-west4'. Reuse, the capacity race and
    preempted-resource cleanup all stay inside it, so the pod lands next to the job's data
    instead of winning a distant zone and reading across regions for the whole run."""

    only_my_pods: bool = False
    """Never reuse or reclaim a pod that does not carry this user's name. The idle probe is
    a heuristic, and a wrong answer on a colleague's pod starts a job on top of theirs."""

    done_marker: str | None = None
    """Empty GCS object the remote command creates on success. Its presence is how a fresh
    launcher learns the work already finished; it is deleted once detected, so the handshake
    is one-shot and cannot go stale. Never localized — both sides must name one object."""

    carry_checkpoints_from: str | None = None
    """Checkpoint root to seed this run from on its FIRST launch, e.g. a previous run's
    region bucket. The newest committed checkpoint under it is copied into wherever this
    launch actually writes. Across preemption retries the carry happens automatically."""

    allow_cross_continent_checkpoint_transfer: bool = False
    """Permit the checkpoint carry to copy across continents (US <-> EU), billed at
    $0.05/GiB. Off by default so a race that wins a far zone cannot silently move a 40 GiB
    checkpoint; a PaliGemma critic with optimizer state is that size."""

    post_launch_hook: str = ""
    """Command run locally right after the job starts, and again after every preemption
    retry. Backgrounded so monitoring is never blocked. Receives the pod's identity through
    TPU_NAME, TPU_ZONE, TPU_PROJECT, TPU_WORKER_COUNT and TPU_COMMAND."""

    nfs_user: str = ""
    """Directory name on the NFS mount. Defaults to the launching user's nfs_directory."""

    sync_code: bool = True
    """Whether to sync code before running."""

    install_deps: bool = False
    """Whether to force `uv sync` on the TPU even when the environment is already present."""

    local_code_dir: str = dataclasses.field(default_factory=lambda: str(Path(__file__).resolve().parents[3]))
    """Local code directory to sync from."""

    local_gemma_dir: str = dataclasses.field(
        default_factory=lambda: str(Path.home() / "projects/AIRe/robocoin/helper/gemma")
    )
    """Local Gemma helper checkout to sync from."""

    retry_on_preemption: bool = False
    """Whether to re-acquire capacity and replay the command if the TPU is preempted."""

    max_retries: int | None = None
    """Maximum number of retries. None means unlimited; 0 means none."""

    slack_webhook_url: str | None = None
    """Slack webhook URL for notifications."""

    poll_interval: int = 30
    """Seconds between job status checks."""

    race_timeout: int = 21600
    """Seconds a spot race may ride the queue. Stockouts last hours, so this is large:
    timing out deletes every candidate, losing queue position."""

    verbose: bool = False
    """Show all gcloud commands being executed."""

    local_tmux: bool = True
    """Create a local tmux session with one window per worker for viewing outputs."""

    endpoint_file: str | None = None
    """Path to write the allocated pod's name and zone to, as JSON, once it is resolved. A
    spot race picks both at runtime, so a client that has to reach the job has no other way
    to learn where it landed. Rewritten on every preemption retry."""

    @functools.cached_property
    def plan(self) -> CommandPlan:
        return CommandPlan.parse(self.command)

    @functools.cached_property
    def run_id(self) -> str:
        """This run's certificate contents; see :mod:`openpi.tpu.certificate`."""
        return self.plan.run_id(user=self.user)

    def resolve(self, pod: PodConfig) -> "ResolvedLaunch":
        """Bind this launch to a pod, enforcing the four placement guarantees."""
        return ResolvedLaunch.build(self, pod)


@dataclasses.dataclass(frozen=True)
class ResolvedLaunch:
    """A launch bound to one pod: concrete command, concrete paths, concrete carry."""

    config: LaunchConfig
    pod: PodConfig
    layout: RemoteLayout
    command: str
    rewrites: Mapping[str, str]
    checkpoint_root: str | None
    fine_tune_root: str | None

    @classmethod
    def build(cls, config: LaunchConfig, pod: PodConfig) -> "ResolvedLaunch":
        layout = pod.layout(config.nfs_user or None)
        plan = config.plan
        replacements: dict[Span, str] = {}
        rewrites: dict[str, str] = {}

        destination = buckets.ensure_regional_bucket(pod.region, resource_owner=pod.user.resource_owner)
        hub = _hub_bucket(pod)

        for ref in plan.uris:
            target = _placement_for(ref, pod=pod, destination=destination, hub=hub)
            if target is not None and target != ref.span.text:
                replacements[ref.span] = target
                rewrites[ref.span.text] = target

        for ref in plan.paths:
            localized = layout.localize_path(ref.span.text)
            if localized != ref.span.text:
                replacements[ref.span] = localized
                rewrites[ref.span.text] = localized
                logger.info("Localizing %s for a pod without NFS: %s -> %s", ref.flag, ref.span.text, localized)

        command = plan.rewrite(replacements)
        command = _adjust_mesh_flags(command, pod)
        if rewrites:
            logger.info("Localized %d path(s) for %s: %s", len(rewrites), pod.zone, rewrites)

        localized_plan = CommandPlan.parse(command)
        checkpoint_root = localized_plan.checkpoint_root
        fine_tune_root = (
            f"{checkpoint_root}/{localized_plan.fine_tune}" if checkpoint_root and localized_plan.fine_tune else None
        )
        return cls(
            config=config,
            pod=pod,
            layout=layout,
            command=command,
            rewrites=rewrites,
            checkpoint_root=checkpoint_root,
            fine_tune_root=fine_tune_root,
        )

    @property
    def paligemma_weights_uri(self) -> str:
        """This pod's own copy of the PaliGemma weights, in its continent's hub bucket."""
        bucket = _hub_bucket(self.pod) or self.pod.gcs_bucket
        return f"gs://{bucket}/{PALIGEMMA_WEIGHTS_PATH}"

    def carry_from(self, source_root: str | None) -> None:
        """Copy the newest committed checkpoint of a previous attempt into this one.

        A re-raced pod can land in a different region, where the localized writes point at
        that region's bucket — an empty one. Without this the run silently restarts from
        step 0 while its progress sits elsewhere. A fine-tune writes one level down, under
        ``<root>/<name>``, which the base carry never reaches, so it is carried separately.
        """
        if not source_root or not self.checkpoint_root:
            return
        allow = self.config.allow_cross_continent_checkpoint_transfer
        buckets.carry_checkpoints(source_root, self.checkpoint_root, allow_cross_continent=allow)
        if self.config.plan.fine_tune and self.fine_tune_root:
            buckets.carry_checkpoints(
                f"{source_root.rstrip('/')}/{self.config.plan.fine_tune}",
                self.fine_tune_root,
                allow_cross_continent=allow,
            )


def _hub_bucket(pod: PodConfig) -> str | None:
    """The continent-wide bucket a shared read should resolve to, or None if unmapped."""
    hub_region = HUB_REGION_BY_CONTINENT.get(buckets.continent_of(pod.region))
    if hub_region is None:
        return None
    return pod.user.gcs_buckets_by_region.get(hub_region) or buckets.canonical_bucket_for_region(
        hub_region, resource_owner=pod.user.resource_owner
    )


def _placement_for(ref: UriRef, *, pod: PodConfig, destination: str, hub: str | None) -> str | None:
    """Where one URI belongs for this pod, or None to leave it alone.

    Raises:
        ValueError: if a read is on another continent and has no same-continent replica.
    """
    source_bucket, path = buckets.split_uri(ref.span.text)
    wanted_bucket = hub if (ref.role is UriRole.SHARED_READ and hub is not None) else destination
    if source_bucket == wanted_bucket:
        return None
    target = f"gs://{wanted_bucket}/{path}" if path else f"gs://{wanted_bucket}"

    if ref.role is UriRole.WRITE:
        # The run creates this content itself, so there is nothing to copy: point it at the
        # pod's own region and let it be written there.
        return target

    source_region = buckets.bucket_region(source_bucket)
    if source_region is None:
        raise ValueError(f"Read source gs://{source_bucket} does not exist")
    if buckets.continent_of(source_region) == buckets.continent_of(pod.region):
        # Inter-region reads within a continent are cheap next to duplicating a dataset.
        logger.info("Leaving %s alone: same continent as %s", ref.span.text, pod.region)
        return None
    if not buckets.uri_exists(target):
        raise ValueError(
            f"{ref.span.text} is in {source_region}, the pod is in {pod.region}, and there is no "
            f"same-continent replica at {target}. Replicate it first: a launch will not copy a "
            "read across continents for you."
        )
    return target


def _adjust_mesh_flags(command: str, pod: PodConfig) -> str:
    """Make the FSDP axis expressible on the pod that was actually allocated.

    A pod's physical mesh is [4, hosts, 1] — four chips per host. An FSDP axis is only
    assignable if it is a product of a subset of those axis sizes, so a config written for
    a 16-host v5e-64 asks for 16 and fails on a v6e-32, whose mesh is [4, 8, 1]. The host
    count is always an axis, so it is the safe choice.

    Matches any namespaced spelling (``--fsdp-devices``, ``--critic.fsdp-devices``): a
    command that already pins the axis must not have a second, top-level flag appended,
    which its CLI would reject outright.
    """
    if pod.family != "v6e" or "fsdp-devices" in command:
        return command
    hosts = pod.worker_count
    logger.info("Setting --fsdp-devices=%d for %s (mesh [4, %d, 1])", hosts, pod.tpu_type, hosts)
    return f"{command} --fsdp-devices={hosts}"


def command_with_done_marker(command: str, marker: str | None) -> str:
    """Append marker creation so only a successful command records completion.

    Written by one worker only. The command runs on every worker, and GCS rate-limits
    mutation of a *single* object, so N workers creating the same marker at the same
    instant means one succeeds and the rest get HTTP 429 — which fails their half of the
    chain and reports a finished run as a failure. The worker index comes from the
    hostname's ``-w-<index>`` suffix; TPU_WORKER_ID is not set in the ssh environment. A
    hostname without that suffix is a single host, which therefore writes it too.

    The marker tail is one brace group: ``&&`` binds only to the first command of a
    ``;``-separated list, so an ungrouped tail runs even when the command fails — a crash
    then reports itself finished. ``-n "$h"`` guards the single-host branch for the same
    reason: if the failed chain skipped the hostname assignment, both ``$w`` and ``$h`` are
    empty and ``[ "$w" = "$h" ]`` alone would match.
    """
    if not marker:
        return command
    return (
        f"{command} && "
        f"{{ h=$(hostname); w=${{h##*-w-}}; "
        f'if [ -n "$h" ] && {{ [ "$w" = 0 ] || [ "$w" = "$h" ]; }}; then '
        f"gcloud storage cp /dev/null {marker}; fi; }}"
    )


def iter_uris(plan: CommandPlan, role: UriRole) -> Iterable[str]:
    """Every URI in a plan playing one role, for logging and tests."""
    return (ref.span.text for ref in plan.uris if ref.role is role)
