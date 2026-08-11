"""Job execution and monitoring for TPU jobs."""

import dataclasses
import logging
import subprocess
import time
from typing import Literal

from openpi.tpu.gcloud import _TERMINAL_TPU_STATES
from openpi.tpu.gcloud import get_tpu_state_and_health
from openpi.tpu.gcloud import ssh_command
from openpi.tpu.manager import is_tpu_preempted
from openpi.tpu.slack import SlackNotifier

logger = logging.getLogger(__name__)

JobState = Literal["pending", "running", "completed", "failed", "preempted"]


@dataclasses.dataclass
class JobStatus:
    """Status of a running or completed job."""

    state: JobState
    exit_code: int | None = None
    output_tail: str = ""


class JobRunner:
    """Runs and monitors jobs on TPU via tmux."""

    def __init__(
        self,
        tpu_name: str,
        zone: str,
        project: str,
        working_dir: str,
        notifier: SlackNotifier | None = None,
        num_workers: int = 1,
        nfs_mount_path: str = "/nfs/aidm_nfs",
        nfs_user: str = "saksham3",
    ):
        """Initialize the job runner.

        Args:
            tpu_name: TPU VM name
            zone: GCP zone
            project: GCP project ID
            working_dir: Working directory on TPU
            notifier: Optional Slack notifier
            num_workers: Number of TPU workers (hosts)
            nfs_mount_path: NFS mount path (e.g. /nfs/aidm_nfs)
        """
        self.tpu_name = tpu_name
        self.zone = zone
        self.project = project
        self.working_dir = working_dir
        self.notifier = notifier
        self.num_workers = num_workers
        self.nfs_mount_path = nfs_mount_path
        self.nfs_user = nfs_user
        self._exit_code_file = "~/tpu_job_exit_code"
        self._log_file = "~/tpu_job_output.log"
        self._local_session_name = f"tpu-{tpu_name}"

    def _build_job_preamble(self) -> str:
        """Build TPU job environment setup shared by all workers.

        A pod without a Filestore has no shared uv root, so the environment lives in each
        worker's own home. Interpolating nfs_mount_path unguarded renders the literal
        string "None" into the activate path and the job dies at start.
        """
        nfs = self.nfs_mount_path
        user = self.nfs_user
        uv_root = f"{nfs}/{user}/uv" if nfs else "$HOME/uv"
        paligemma_source = f"{nfs}/{user}/gemma/2b/pt_224.npz" if nfs else ""
        preamble = (
            "source ~/.bashrc && "
            f"source {uv_root}/vla/bin/activate && "
            f'export PATH="{uv_root}/bin:$PATH" && '
            f'export UV_PROJECT_ENVIRONMENT="{uv_root}/vla" && '
            "sudo mkdir -p /tmp/tpu_logs && "
            "sudo chmod -R 777 /tmp/tpu_logs && "
        )
        if paligemma_source:
            # Seed the local cache from NFS. Without a shared filesystem the weights are
            # staged per worker during setup instead, so there is nothing to copy here.
            preamble += (
                "PALIGEMMA_CACHE=$HOME/.cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz && "
                'if [ ! -f "$PALIGEMMA_CACHE" ]; then '
                'mkdir -p "$(dirname "$PALIGEMMA_CACHE")" && '
                f'cp {paligemma_source} "$PALIGEMMA_CACHE" || true; '
                "fi && "
            )
        return preamble + ("export PLATFORM=tpu")

    def start_job(self, command: str, session_name: str = "job") -> None:
        """Start a job in a tmux session.

        Args:
            command: Command to run
            session_name: tmux session name
        """
        logger.info("Starting job on TPU %s: %s", self.tpu_name, command)

        kill_cmd = f"tmux kill-session -t {session_name} 2>/dev/null || true"
        ssh_command(
            self.tpu_name,
            self.zone,
            kill_cmd,
            project=self.project,
            worker="all",
            check=False,
        )

        ssh_command(
            self.tpu_name,
            self.zone,
            f"rm -f {self._exit_code_file} {self._log_file}",
            project=self.project,
            worker="all",
            check=False,
        )

        preamble = self._build_job_preamble()
        # Tmux session exits as soon as the command finishes (success OR
        # failure). The exit code persists in {self._exit_code_file} and the
        # full output in {self._log_file}, so monitor_job + post-mortem
        # debugging keep working without a sleep loop holding the session.
        full_command = (
            f"{preamble} && "
            f"cd {self.working_dir} && "
            f"( {command} ) 2>&1 | tee {self._log_file}; "
            f"job_exit_code=${{PIPESTATUS[0]}}; "
            f"echo ${{job_exit_code}} > {self._exit_code_file}; "
            f"exit ${{job_exit_code}}"
        )

        escaped_command = full_command.replace("'", "'\\''")
        tmux_cmd = f"tmux new-session -d -s {session_name} '{escaped_command}'"

        ssh_command(
            self.tpu_name,
            self.zone,
            tmux_cmd,
            project=self.project,
            worker="all",
        )

        logger.info("Job started in tmux session '%s'", session_name)

    def monitor_job(
        self,
        session_name: str = "job",
        poll_interval: float = 30,
    ) -> JobStatus:
        """Monitor a running job until completion or preemption.

        Args:
            session_name: tmux session name
            poll_interval: Time between status checks in seconds

        Returns:
            Final job status
        """
        logger.info("Monitoring job in session '%s'", session_name)

        while True:
            if is_tpu_preempted(self.tpu_name, self.zone, self.project):
                logger.warning("TPU %s was preempted", self.tpu_name)
                return JobStatus(state="preempted")

            if not self._is_session_running(session_name):
                exit_code = self._get_exit_code()
                output_tail = self.get_output(session_name)

                if exit_code == 0:
                    logger.info("Job completed successfully")
                    return JobStatus(
                        state="completed",
                        exit_code=exit_code,
                        output_tail=output_tail,
                    )
                logger.warning("Job failed with exit code %s", exit_code)
                return JobStatus(
                    state="failed",
                    exit_code=exit_code,
                    output_tail=output_tail,
                )

            time.sleep(poll_interval)

    def _is_session_running(self, session_name: str) -> bool:
        """Check if a tmux session is still running.

        Args:
            session_name: tmux session name

        Returns:
            True if session exists
        """
        result = ssh_command(
            self.tpu_name,
            self.zone,
            f"tmux has-session -t {session_name} 2>/dev/null && echo running || echo stopped",
            project=self.project,
            worker="0",
            check=False,
        )
        # An ssh that fails returns empty stdout, which would read as "stopped" and end the
        # run. Only an answer of "stopped" from a reachable pod counts as the job ending;
        # anything else means keep polling.
        if result.returncode != 0 or not result.stdout.strip():
            # Unless the pod is gone. "Assume still running" is right for a blip, but on a
            # preempted pod every probe fails forever, so the monitor never finishes and the
            # preemption retry it feeds never runs.
            state, _ = get_tpu_state_and_health(self.tpu_name, self.zone, self.project)
            if state in _TERMINAL_TPU_STATES:
                logger.info(
                    "Session probe on %s failed and the pod is %s; treating the job as ended", self.tpu_name, state
                )
                return False
            logger.info(
                "Session probe on %s inconclusive (rc=%s); assuming still running", self.tpu_name, result.returncode
            )
            return True
        return "running" in result.stdout

    def _get_exit_code(self) -> int | None:
        """Get the exit code from the job.

        Returns:
            Exit code, or None if not available
        """
        result = ssh_command(
            self.tpu_name,
            self.zone,
            f"cat {self._exit_code_file} 2>/dev/null || echo ''",
            project=self.project,
            worker="0",
            check=False,
        )
        try:
            code_str = result.stdout.strip()
            if code_str:
                return int(code_str)
        except ValueError:
            pass
        return None

    def get_output(self, session_name: str = "job", lines: int = 50) -> str:
        """Get recent output from the job.

        Reads from the persistent log file, falling back to tmux pane capture.

        Args:
            session_name: tmux session name
            lines: Number of lines to capture

        Returns:
            Recent output as string
        """
        # Try to read from persistent log file first
        result = ssh_command(
            self.tpu_name,
            self.zone,
            f"tail -n {lines} {self._log_file} 2>/dev/null || echo ''",
            project=self.project,
            worker="0",
            check=False,
        )
        if result.stdout.strip():
            return result.stdout

        # Fall back to tmux capture if log file not available
        result = ssh_command(
            self.tpu_name,
            self.zone,
            f"tmux capture-pane -t {session_name} -p -S -{lines} 2>/dev/null || echo ''",
            project=self.project,
            worker="0",
            check=False,
        )
        return result.stdout

    def create_local_tmux_session(self, remote_session_name: str = "job") -> bool:
        """Create a local tmux session with one window per worker for viewing outputs.

        Each window SSHs into a TPU worker and attaches to the remote tmux session.

        Args:
            remote_session_name: Name of the remote tmux session to attach to

        Returns:
            True if session was created successfully
        """
        self.cleanup_local_tmux_session()

        gcloud_ssh_base = f"gcloud compute tpus tpu-vm ssh {self.tpu_name} --zone={self.zone} --project={self.project}"

        try:
            for worker in range(self.num_workers):
                ssh_cmd = f"{gcloud_ssh_base} --worker={worker} -- tmux attach-session -t {remote_session_name}"

                if worker == 0:
                    subprocess.run(
                        [
                            "tmux",
                            "new-session",
                            "-d",
                            "-s",
                            self._local_session_name,
                            "-n",
                            f"worker-{worker}",
                            ssh_cmd,
                        ],
                        check=True,
                    )
                else:
                    subprocess.run(
                        [
                            "tmux",
                            "new-window",
                            "-t",
                            self._local_session_name,
                            "-n",
                            f"worker-{worker}",
                            ssh_cmd,
                        ],
                        check=True,
                    )

            logger.info(
                "Created local tmux session '%s' with %d worker windows",
                self._local_session_name,
                self.num_workers,
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.warning("Failed to create local tmux session: %s", e)
            return False

    def cleanup_local_tmux_session(self) -> None:
        """Clean up the local tmux session if it exists."""
        subprocess.run(
            ["tmux", "kill-session", "-t", self._local_session_name],
            capture_output=True,
            check=False,
        )

    @property
    def local_session_name(self) -> str:
        """Name of the local tmux session."""
        return self._local_session_name
