#!/usr/bin/env python3
"""TPU job orchestration CLI.

One attempt is a fixed sequence of phases — acquire, prepare, launch, watch — and each
phase returns a typed :class:`Outcome` rather than reaching for an exit code of its own.
The driver owns the retry budget, the notifications and the run certificate, so there is
one place where a launch can end and one place where a pod is released. That is what keeps
state from being assigned on one branch and read on another.

Example usage:

    # Race spot capacity for any of the default shapes and retry through preemption
    python scripts/run_on_tpu.py --spot --retry-on-preemption \\
      --command "python scripts/train_value_function.py my_config --resume"

    # Pin a specific pod
    python scripts/run_on_tpu.py --tpu-name v4-64-0 --tpu-type v4-64 \\
      --command "python scripts/serve_policy.py ..."
"""

import dataclasses
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

import tyro

from openpi.tpu import buckets
from openpi.tpu import certificate
from openpi.tpu import code_sync
from openpi.tpu import setup as tpu_setup
from openpi.tpu.certificate import PodState
from openpi.tpu.gcloud import ssh_command
from openpi.tpu.job import JobRunner
from openpi.tpu.job import JobStatus
from openpi.tpu.launch import LaunchConfig
from openpi.tpu.launch import ProgressReporter
from openpi.tpu.launch import ResolvedLaunch
from openpi.tpu.launch import command_with_done_marker
from openpi.tpu.manager import Acquisition
from openpi.tpu.manager import AllocationRequest
from openpi.tpu.manager import acquire
from openpi.tpu.manager import cleanup_preempted
from openpi.tpu.manager import is_tpu_preempted
from openpi.tpu.slack import SlackNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What a phase decided should happen next."""

    kind: Literal["advance", "retry", "finish"]
    exit_code: int = 0
    reason: str = ""
    holds_pod: bool = True
    """Whether this launch still owns the pod it was working on. False when the claim was
    lost to another run: the certificate on that pod is somebody else's and releasing it
    would hand their pod to the next launcher that walks past."""

    @classmethod
    def advance(cls) -> "Outcome":
        return cls(kind="advance")

    @classmethod
    def retry(cls, reason: str, *, holds_pod: bool = True) -> "Outcome":
        """Give this pod up and acquire another. Used only when the pod is gone."""
        return cls(kind="retry", reason=reason, holds_pod=holds_pod)

    @classmethod
    def finish(cls, exit_code: int, reason: str = "", *, holds_pod: bool = True) -> "Outcome":
        return cls(kind="finish", exit_code=exit_code, reason=reason, holds_pod=holds_pod)


def consume_done_marker(marker: str | None) -> bool:
    """Whether the work already completed, clearing the marker if so.

    Generic by construction: the launcher knows nothing about checkpoints, only that the
    command it was given reported success by creating this object. Deleting it makes the
    handshake one-shot, so it cannot go stale and turn a later legitimate launch into a
    no-op.
    """
    if not marker or not buckets.marker_exists(marker):
        return False
    logger.info("Completion marker %s present: work already finished", marker)
    buckets.remove_marker(marker)
    return True


def write_endpoint_file(path: str, acquisition: Acquisition) -> None:
    """Publish where the pod landed, for a client that has to reach the job.

    Written before setup rather than after the job starts: a client that blocks on this
    file should be free to begin probing while the pod is still being prepared, and setup
    on a fresh spot pod takes minutes.
    """
    endpoint = {
        "name": acquisition.name,
        "zone": acquisition.config.zone,
        "project": acquisition.config.project,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(endpoint))
    logger.info("Wrote endpoint %s to %s", endpoint, path)


def run_post_launch_hook(config: LaunchConfig, resolved: ResolvedLaunch, tpu_name: str) -> None:
    """Start the operator's post-launch hook for the pod the job was just started on.

    Runs per attempt rather than per launch: a preemption retry puts the job on a fresh pod
    whose local state starts empty, so whatever the hook sets up has to be redone there.
    Failure is logged and never propagated — a hook accompanies a run, it is not a
    precondition for it.
    """
    if not config.post_launch_hook:
        return
    env = {
        **os.environ,
        "TPU_NAME": tpu_name,
        "TPU_ZONE": resolved.pod.zone,
        "TPU_PROJECT": resolved.pod.project,
        "TPU_WORKER_COUNT": str(resolved.pod.worker_count),
        "TPU_COMMAND": resolved.command,
    }
    logger.info("Starting post-launch hook on %s: %s", tpu_name, config.post_launch_hook)
    try:
        # shell=True is deliberate: the hook is operator-supplied, exactly like the training
        # command this script already runs.
        subprocess.Popen(
            config.post_launch_hook,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        logger.warning("Could not start post-launch hook: %s", e)


class Launcher:
    """Drives one launch through however many pods it takes."""

    def __init__(self, config: LaunchConfig):
        self.config = config
        self.notifier = SlackNotifier(webhook_url=config.slack_webhook_url)
        self.run_id = config.run_id
        self.retry_count = 0
        self.progress = ProgressReporter(config, self.notifier, self.run_id)
        # Where the previous attempt wrote, so an attempt that lands elsewhere can carry it.
        self.previous_checkpoint_root: str | None = config.carry_checkpoints_from

    # -- driver ---------------------------------------------------------------------

    def run(self) -> int:
        logger.info("Launch %r: run certificate is %s", self.run_id, certificate.CERTIFICATE_FILE)
        while True:
            if consume_done_marker(self.config.done_marker):
                logger.info("Nothing to do; the work this launch describes has already finished")
                return 0

            try:
                acquisition = acquire(self._allocation_request())
            except Exception as e:
                logger.error("Failed to acquire a TPU: %s", e)
                return 1

            outcome = self._attempt(acquisition)
            logger.info("Attempt on %s ended: %s (%s)", acquisition.name, outcome.kind, outcome.reason or "no detail")

            if outcome.kind == "finish":
                # The job's own EXIT trap normally clears this the moment the command
                # returns; doing it again here covers an attempt that never got that far.
                # Skipped when the claim was lost, because the certificate then belongs to
                # the run that took the pod and is not ours to remove.
                if outcome.holds_pod:
                    certificate.release(acquisition.name, acquisition.config.zone, acquisition.config.project)
                return outcome.exit_code

            self.retry_count += 1

    def _allocation_request(self) -> AllocationRequest:
        return AllocationRequest(
            run_id=self.run_id,
            tpu_type=self.config.tpu_type,
            tpu_name=self.config.tpu_name_for_attempt(self.retry_count),
            user=self.config.user,
            project=self.config.project,
            spot=self.config.spot,
            region=self.config.region,
            only_my_pods=self.config.only_my_pods,
            zone=self.config.zone,
            continent=self.config.continent,
            race_timeout=self.config.race_timeout,
            max_race_zones=self.config.max_race_zones,
        )

    def _may_retry(self) -> bool:
        """Whether losing the pod right now may be answered by acquiring another.

        Only a spot launch can re-acquire: a reserved launch would go looking for an idle
        pod that the preemption just removed. One budget, one meaning — ``None`` is
        unlimited and ``0`` is none, on every path that consumes it.
        """
        if not (self.config.retry_on_preemption and self.config.spot):
            return False
        return self.config.max_retries is None or self.retry_count < self.config.max_retries

    # -- phases ---------------------------------------------------------------------

    def _attempt(self, acquisition: Acquisition) -> Outcome:
        """Prepare, launch and watch one pod."""
        pod = acquisition.config
        try:
            resolved = self.config.resolve(pod)
        except Exception as e:
            # A placement rule was violated — a read with no same-continent replica, most
            # likely. That is a launch-definition problem and retrying cannot fix it.
            logger.error("Cannot place this launch on %s (%s): %s", acquisition.name, pod.zone, e)
            return Outcome.finish(1, f"placement: {e}")

        outcome = self._hold(acquisition)
        if outcome.kind != "advance":
            return outcome

        if self.config.endpoint_file:
            write_endpoint_file(self.config.endpoint_file, acquisition)

        runner = JobRunner(acquisition.name, pod, resolved.layout)

        if acquisition.resumed:
            logger.info(
                "Attached to in-flight run %r on %s; skipping setup, code sync and job start",
                self.run_id,
                acquisition.name,
            )
        else:
            outcome = self._prepare_and_start(acquisition, resolved, runner)
            if outcome.kind != "advance":
                return outcome

        run_post_launch_hook(self.config, resolved, acquisition.name)
        return self._watch(acquisition, resolved, runner)

    def _hold(self, acquisition: Acquisition) -> Outcome:
        """Make sure this pod carries our certificate before anything is put on it.

        The reuse and named paths claim during acquisition, but a raced pod is brand new and
        nobody has written to it yet, so without this a freshly created pod runs its whole
        job uncertificated. Two things then go wrong: a restarted launcher cannot recognise
        its own work (``_find_our_pod`` reads a certificate that was never written) and
        races a second pod into the same checkpoint directory, and two launchers that
        collided on a zone and adopted the same queued resource both believe they won it.

        Claiming here rather than inside the race covers every acquisition path with one
        call, and re-claiming a pod we already hold is free: the probe returns CLAIMED for
        an idle pod of ours and OURS for one already running our job.
        """
        probe = certificate.probe_and_claim(
            acquisition.name, acquisition.config.zone, acquisition.config.project, self.run_id
        )
        if probe.state in (PodState.CLAIMED, PodState.OURS):
            return Outcome.advance()

        if probe.state is PodState.BUSY:
            held = f"run {probe.certificate!r}" if probe.certificate else "an unknown occupant"
            detail = f"{acquisition.name} is held by {held}"
        else:
            detail = f"{acquisition.name} stopped answering the certificate probe"
        # Someone took the pod between acquiring it and now. Acquiring another is the right
        # answer, but only within the budget that already bounds re-acquisition; otherwise
        # this would spin creating pods it never gets to keep.
        if self._may_retry():
            logger.warning("%s; giving it up and acquiring another", detail)
            return Outcome.retry("lost the pod before setup", holds_pod=False)
        logger.error("%s, and this launch cannot acquire another. Refusing to start a job on it.", detail)
        return Outcome.finish(1, "lost the pod before setup", holds_pod=False)

    def _prepare_and_start(self, acquisition: Acquisition, resolved: ResolvedLaunch, runner: JobRunner) -> Outcome:
        """Everything between holding a pod and having a job running on it."""
        name, pod, layout = acquisition.name, acquisition.config, resolved.layout

        try:
            self._carry_checkpoints(resolved)
        except PermissionError as e:
            # A cross-continent carry was refused. Correct, and not worth failing a launch
            # that can still start from whatever the destination already holds.
            logger.warning("Not carrying checkpoints onto %s: %s", name, e)
        except Exception as e:
            logger.error("Failed to carry checkpoints onto %s: %s", name, e)
            return Outcome.finish(1, f"checkpoint carry: {e}")

        try:
            if not tpu_setup.verify_setup(name, pod):
                logger.info("Setting up TPU %s...", name)
                tpu_setup.setup_tpu(name, pod, layout)
            # Idempotent and cheap when the file is already there, so it runs on every
            # launch: gating it behind the setup check meant a reused pod never got the
            # weights staged at all and died at weight load.
            code_sync.stage_paligemma_weights(name, pod, layout, resolved.paligemma_weights_uri)

            if self.config.sync_code:
                logger.info("Syncing code to TPU %s...", name)
                code_sync.sync_code(self.config.local_code_dir, name, pod, layout)
                # Dependencies first: the gemma helper installs into the venv, so it has to
                # exist by now.
                code_sync.ensure_uv_environment(name, pod, layout, force=self.config.install_deps)
                code_sync.sync_gemma_helper(self.config.local_gemma_dir, name, pod, layout)
                code_sync.install_gemma_helper(name, pod, layout)
                code_sync.sync_wandb_credentials(name, pod)
        except Exception as e:
            # A spot pod can vanish mid-setup, and every remaining ssh then hangs to its
            # timeout and surfaces as a setup error. Retrying is the whole point of
            # --retry-on-preemption, so ask the pod before giving up on the run.
            if self._may_retry() and is_tpu_preempted(name, pod.zone, pod.project):
                logger.info("%s was preempted while being prepared; re-acquiring", name)
                cleanup_preempted(pod.tpu_type, project=pod.project, region=self.config.region)
                return Outcome.retry("preempted during preparation")
            logger.error("Failed to prepare TPU %s: %s", name, e)
            if isinstance(e, subprocess.CalledProcessError):
                for stream, label in ((e.stdout, "stdout"), (e.stderr, "stderr")):
                    if stream:
                        logger.error("Command %s:\n%s", label, stream)
            return Outcome.finish(1, f"preparation: {e}")

        try:
            runner.start_job(command_with_done_marker(resolved.command, self.config.done_marker))
        except Exception as e:
            logger.error("Failed to start job on %s: %s", name, e)
            return Outcome.finish(1, f"job start: {e}")

        self.notifier.notify_started(name, pod.tpu_type, pod.zone, self.run_id, resolved.command)
        return Outcome.advance()

    def _carry_checkpoints(self, resolved: ResolvedLaunch) -> None:
        """Bring the previous attempt's progress into wherever this one writes."""
        resolved.carry_from(self.previous_checkpoint_root)
        # Recorded even when nothing was carried: it is where THIS attempt writes, and so
        # what the next attempt must carry from.
        self.previous_checkpoint_root = resolved.checkpoint_root or self.previous_checkpoint_root

    def _watch(self, acquisition: Acquisition, resolved: ResolvedLaunch, runner: JobRunner) -> Outcome:
        """Follow the job to its end and decide what that end means."""
        name, pod = acquisition.name, acquisition.config
        started_at = time.time()

        local_tmux = self.config.local_tmux and runner.create_local_tmux_session()
        if local_tmux:
            logger.info("Attach locally with: tmux attach-session -t %s", runner.local_session_name)
        try:
            try:
                on_poll = (
                    (lambda: self.progress.observe(runner.tail_bytes(), name)) if self.progress.enabled else None
                )
                status = runner.monitor_job(poll_interval=self.config.poll_interval, on_poll=on_poll)
            except Exception as monitor_error:
                # Monitoring throws when the pod stops answering — a timed-out ssh, a
                # vanished resource. That is overwhelmingly a preemption, and treating it as
                # a generic failure skips the retry loop that exists for exactly it.
                logger.warning("Monitoring %s failed (%s); checking the pod", name, monitor_error)
                if not is_tpu_preempted(name, pod.zone, pod.project):
                    raise
                status = JobStatus(state="preempted")
        finally:
            if local_tmux:
                runner.cleanup_local_tmux_session()

        elapsed = time.time() - started_at
        logger.info("Job on %s ended after %s: %s", name, _format_duration(elapsed), status.state)

        if status.state == "completed":
            # Reaching here means the run finished in *this* process, so clear the handshake:
            # a marker left behind would make the next legitimate launch a no-op.
            consume_done_marker(self.config.done_marker)
            self.notifier.notify_completion(name, self.run_id, _format_duration(elapsed), success=True)
            return Outcome.finish(0, "completed")

        if status.state == "preempted":
            cleanup_preempted(pod.tpu_type, project=pod.project, region=self.config.region)
            if not self._may_retry():
                if self.config.retry_on_preemption and not self.config.spot:
                    logger.error(
                        "--retry-on-preemption needs --spot: a reserved launch has no way to "
                        "re-acquire capacity after its pod is gone."
                    )
                self.notifier.notify_completion(
                    name,
                    self.run_id,
                    _format_duration(elapsed),
                    success=False,
                    output_tail="Preempted with no retry budget left.",
                )
                return Outcome.finish(1, "preempted, no retry available")
            self.notifier.notify_preemption(name, self.run_id, self.retry_count + 1, self.config.max_retries)
            return Outcome.retry("preempted")

        if status.output_tail and "libtpu_lockfile" in status.output_tail:
            # JAX leaves this behind when a job dies without cleanup, and it blocks the next
            # run on the same pod. The job's certificate was released when it exited, so the
            # pod is free to be re-acquired — usually the very same one.
            logger.warning("Detected a TPU lockfile error on %s; clearing it and retrying", name)
            try:
                ssh_command(
                    name, pod.zone, "sudo rm -f /tmp/libtpu_lockfile", project=pod.project, worker="all", check=False
                )
                if self.config.max_retries is None or self.retry_count < self.config.max_retries:
                    return Outcome.retry("libtpu lockfile cleared")
            except Exception as e:
                logger.error("Failed to clear the lockfile on %s: %s", name, e)

        if status.output_tail:
            logger.error("Last output:\n%s", status.output_tail)
        self.notifier.notify_completion(
            name, self.run_id, _format_duration(elapsed), success=False, output_tail=status.output_tail
        )
        return Outcome.finish(status.exit_code or 1, f"job failed (exit {status.exit_code})")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def main(config: LaunchConfig) -> int:
    if config.verbose:
        logging.getLogger("openpi.tpu.gcloud").setLevel(logging.DEBUG)
    return Launcher(config).run()


if __name__ == "__main__":
    sys.exit(main(tyro.cli(LaunchConfig)))
