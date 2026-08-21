"""Starting a job on a pod and watching it until it ends."""

from collections.abc import Callable
import dataclasses
import logging
import subprocess
import time
from typing import Literal

from openpi.tpu import certificate
from openpi.tpu.config import PodConfig
from openpi.tpu.config import RemoteLayout
from openpi.tpu.gcloud import _TERMINAL_TPU_STATES
from openpi.tpu.gcloud import get_tpu_state_and_health
from openpi.tpu.gcloud import ssh_command
from openpi.tpu.manager import is_tpu_preempted

logger = logging.getLogger(__name__)

JobState = Literal["running", "completed", "failed", "preempted"]

SESSION_NAME = "job"

# Liveness probes are one-line questions, and the answer is only useful while the round is
# still running. Inheriting the 900s default meant a single probe against a pod that had
# just been preempted burned 900s x 3 retries before the loop got back to its (cheap,
# control-plane) preemption check -- 45 minutes during which the run looked alive and no
# retry could start. Preemption is detected by `describe`, never by ssh; ssh only has to
# fail fast enough to let that check run.
PROBE_TIMEOUT_SECONDS = 60


@dataclasses.dataclass
class JobStatus:
    """Status of a running or completed job."""

    state: JobState
    exit_code: int | None = None
    output_tail: str = ""


class JobRunner:
    """Runs and monitors a job on a TPU via tmux."""

    def __init__(self, tpu_name: str, config: PodConfig, layout: RemoteLayout):
        self.tpu_name = tpu_name
        self.config = config
        self.layout = layout
        self._local_session_name = f"tpu-{tpu_name}"

    def _build_preamble(self) -> str:
        """Environment every worker sets up before the command runs."""
        return (
            "source ~/.bashrc && "
            f"source {self.layout.venv}/bin/activate && "
            f'export PATH="{self.layout.uv_root}/bin:$PATH" && '
            f'export UV_PROJECT_ENVIRONMENT="{self.layout.venv}" && '
            "sudo mkdir -p /tmp/tpu_logs && "
            "sudo chmod -R 777 /tmp/tpu_logs && "
            "export PLATFORM=tpu"
        )

    def start_job(self, command: str) -> None:
        """Start a job in a tmux session on every worker.

        The tmux session exits as soon as the command finishes, success or failure. The
        exit code persists in the layout's exit-code file and the full output in its log,
        so monitoring and post-mortem debugging work without a sleep loop holding the
        session open.

        The run certificate is released as soon as the command returns — explicitly, before
        the exit code is even recorded, and again from an EXIT trap so that a signalled or
        otherwise abnormally terminated shell still frees the pod. Both are ``rm -f``, so
        doing it twice costs nothing, and a pod whose job has ended is immediately available
        to the next launcher rather than waiting out a staleness timeout.
        """
        logger.info("Starting job on TPU %s: %s", self.tpu_name, command)

        ssh_command(
            self.tpu_name,
            self.config.zone,
            f"tmux kill-session -t {SESSION_NAME} 2>/dev/null || true; "
            f"rm -f {self.layout.exit_code_file} {self.layout.log_file}",
            project=self.config.project,
            worker="all",
            check=False,
        )

        full_command = f"{self._build_preamble()} && cd {self.layout.working_dir} && {self.build_job_body(command)}"

        escaped_command = full_command.replace("'", "'\\''")
        ssh_command(
            self.tpu_name,
            self.config.zone,
            f"tmux new-session -d -s {SESSION_NAME} '{escaped_command}'",
            project=self.config.project,
            worker="all",
        )
        logger.info("Job started in tmux session '%s'", SESSION_NAME)

    def build_job_body(self, command: str) -> str:
        """The half of the job line that survives the command, separated so it can be run.

        Every clause here is load-bearing and none of it is obvious:
        ``${PIPESTATUS[0]}`` takes the *command's* status rather than ``tee``'s, which is
        always 0; the ``trap`` covers a shell killed by a signal, where the explicit
        ``rm -f`` after the pipeline never runs; and the exit code is recorded after the
        certificate is dropped so a finished pod is free even if the write fails.

        Note this needs bash: ``PIPESTATUS`` is not POSIX and is empty under dash.
        """
        cert = certificate.CERTIFICATE_FILE
        return (
            f"trap 'rm -f {cert}' EXIT && "
            f"( {command} ) 2>&1 | tee {self.layout.log_file}; "
            f"job_exit_code=${{PIPESTATUS[0]}}; "
            f"rm -f {cert}; "
            f"echo ${{job_exit_code}} > {self.layout.exit_code_file}; "
            f"exit ${{job_exit_code}}"
        )

    def monitor_job(self, poll_interval: float = 30, on_poll: Callable[[], None] | None = None) -> JobStatus:
        """Watch a running job until it completes, fails, or its pod disappears.

        ``on_poll`` runs once per round while the job is still going, which is how progress
        reporting rides the existing cadence instead of opening a second polling loop. It is
        told nothing about the job and its failure never propagates: whatever it does is an
        accompaniment to the run, not a condition for it.
        """
        logger.info("Monitoring job in session '%s'", SESSION_NAME)

        while True:
            if is_tpu_preempted(self.tpu_name, self.config.zone, self.config.project):
                logger.warning("TPU %s was preempted", self.tpu_name)
                return JobStatus(state="preempted")

            if not self._is_session_running():
                exit_code = self._get_exit_code()
                output_tail = self.get_output()

                if exit_code == 0:
                    logger.info("Job completed successfully")
                    return JobStatus(state="completed", exit_code=exit_code, output_tail=output_tail)
                # No exit code means the pod stopped answering rather than the job reporting
                # a result. The session probe treats a terminal pod as "job ended", so a
                # preemption arrives here looking like a plain failure — and reporting it as
                # one skips the retry that exists for exactly this.
                if exit_code is None and is_tpu_preempted(self.tpu_name, self.config.zone, self.config.project):
                    logger.warning("TPU %s is gone and left no exit code; treating as preempted", self.tpu_name)
                    return JobStatus(state="preempted", output_tail=output_tail)
                logger.warning("Job failed with exit code %s", exit_code)
                return JobStatus(state="failed", exit_code=exit_code, output_tail=output_tail)

            if on_poll is not None:
                try:
                    on_poll()
                except Exception as e:
                    logger.debug("Poll hook failed: %s", e)

            time.sleep(poll_interval)

    def _is_session_running(self) -> bool:
        result = ssh_command(
            self.tpu_name,
            self.config.zone,
            f"tmux has-session -t {SESSION_NAME} 2>/dev/null && echo running || echo stopped",
            project=self.config.project,
            worker="0",
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        # An ssh that fails returns empty stdout, which would read as "stopped" and end the
        # run. Only an answer of "stopped" from a reachable pod counts as the job ending.
        if result.returncode != 0 or not result.stdout.strip():
            # Unless the pod is gone. "Assume still running" is right for a blip, but on a
            # preempted pod every probe fails forever, so the monitor never finishes and the
            # preemption retry it feeds never runs.
            state, _ = get_tpu_state_and_health(self.tpu_name, self.config.zone, self.config.project)
            if state in _TERMINAL_TPU_STATES:
                logger.info("Session probe on %s failed and the pod is %s; job ended", self.tpu_name, state)
                return False
            logger.info(
                "Session probe on %s inconclusive (rc=%s); assuming still running", self.tpu_name, result.returncode
            )
            return True
        return "running" in result.stdout

    def _get_exit_code(self) -> int | None:
        result = ssh_command(
            self.tpu_name,
            self.config.zone,
            f"cat {self.layout.exit_code_file} 2>/dev/null || echo ''",
            project=self.config.project,
            worker="0",
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        code_str = result.stdout.strip()
        try:
            return int(code_str) if code_str else None
        except ValueError:
            return None

    def tail_bytes(self, count: int = 2000) -> str:
        """The last ``count`` bytes of the job log.

        Bytes rather than lines, because a progress bar rewrites a single line with carriage
        returns and never emits a newline — ``tail -n`` on such a log returns either almost
        nothing or the entire file, whereas its last few hundred bytes always hold the most
        recent update.
        """
        result = ssh_command(
            self.tpu_name,
            self.config.zone,
            f"tail -c {count} {self.layout.log_file} 2>/dev/null || echo ''",
            project=self.config.project,
            worker="0",
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        return result.stdout

    def get_output(self, lines: int = 50) -> str:
        """Recent job output, from the persistent log or a tmux pane capture."""
        result = ssh_command(
            self.tpu_name,
            self.config.zone,
            f"tail -n {lines} {self.layout.log_file} 2>/dev/null || echo ''",
            project=self.config.project,
            worker="0",
            check=False,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
        if result.stdout.strip():
            return result.stdout

        result = ssh_command(
            self.tpu_name,
            self.config.zone,
            f"tmux capture-pane -t {SESSION_NAME} -p -S -{lines} 2>/dev/null || echo ''",
            project=self.config.project,
            worker="0",
            check=False,
        )
        return result.stdout

    def create_local_tmux_session(self) -> bool:
        """Create a local tmux session with one window per worker for viewing outputs."""
        self.cleanup_local_tmux_session()
        base = (
            f"gcloud compute tpus tpu-vm ssh {self.tpu_name} --zone={self.config.zone} --project={self.config.project}"
        )
        try:
            for worker in range(self.layout.worker_count):
                ssh_cmd = f"{base} --worker={worker} -- tmux attach-session -t {SESSION_NAME}"
                if worker == 0:
                    args = ["tmux", "new-session", "-d", "-s", self._local_session_name, "-n", f"worker-{worker}"]
                else:
                    args = ["tmux", "new-window", "-t", self._local_session_name, "-n", f"worker-{worker}"]
                subprocess.run([*args, ssh_cmd], check=True)
            logger.info(
                "Created local tmux session '%s' with %d worker windows",
                self._local_session_name,
                self.layout.worker_count,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to create local tmux session: %s", e)
            return False

    def cleanup_local_tmux_session(self) -> None:
        subprocess.run(["tmux", "kill-session", "-t", self._local_session_name], capture_output=True, check=False)

    @property
    def local_session_name(self) -> str:
        return self._local_session_name
