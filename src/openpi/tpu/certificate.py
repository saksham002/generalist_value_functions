"""The run certificate: one file on a pod that says which run owns it.

``~/tpu_run_id`` holds the id of the run currently occupying a pod, and its *presence* —
not its age — is the whole signal. That gives one answer to three questions that used to
have three separate mechanisms:

- **is this pod free?** no certificate, and no foreign python running;
- **is this pod mine, still in flight?** a certificate whose contents match my run id, which
  is what lets a relaunched launcher (a requeued SLURM job, say) re-attach to its own work
  instead of racing for a second pod and starting a duplicate run;
- **may I take it?** the create is atomic, so two launchers sweeping in the same second
  cannot both win.

The lifecycle is deliberately narrow, and every step is somebody's explicit
responsibility:

1. the launcher **claims** it the moment a pod is assigned, before any setup — the window
   before that was the one gap the old setup-marker scheme never covered;
2. the job **releases** it from an EXIT trap when the command finishes, whether it
   succeeded or failed, so a finished pod is immediately free for the next launcher;
3. the launcher **releases** it on any terminal outcome it reaches itself, which covers the
   run never getting as far as starting a job.

Both releases are ``rm -f``, so doing it twice is not an error, and neither is doing it to
a pod that has already been torn down. What is deliberately absent is any notion of
staleness: a certificate that outlives its run is cleared by the next launcher that finds
the pod otherwise idle, rather than by a timeout racing a heartbeat.
"""

import dataclasses
import enum
import logging
import re

from openpi.tpu.gcloud import ssh_command

logger = logging.getLogger(__name__)

# Read and written on worker 0 only. Every worker shares it on an NFS pod, and on a
# local-disk pod worker 0 is the one every probe already talks to.
CERTIFICATE_FILE = "~/tpu_run_id"

# A run id is interpolated into shell on the pod, so it is restricted to characters that
# cannot mean anything to a shell rather than escaped at each use.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class PodState(enum.Enum):
    """What a pod is doing, as far as the certificate and process table can tell."""

    CLAIMED = "claimed"
    """It was free and this launcher now holds it."""

    OURS = "ours"
    """It already carries this run's certificate: our own work, still in flight."""

    BUSY = "busy"
    """Someone else's certificate, or foreign python processes, or both."""

    UNREACHABLE = "unreachable"
    """The probe did not answer. Never treated as free."""


@dataclasses.dataclass(frozen=True)
class PodProbe:
    """The result of a single combined certificate-and-process probe."""

    state: PodState
    certificate: str | None = None
    python_processes: int = 0

    @property
    def is_free(self) -> bool:
        return self.state is PodState.CLAIMED


def validate_run_id(run_id: str) -> str:
    """Reject a run id that could mean something to a shell."""
    if not _SAFE_RUN_ID.match(run_id):
        raise ValueError(
            f"Unsafe run id {run_id!r}: only letters, digits, dot, underscore and hyphen are allowed, "
            "because the id is written into a shell command on the pod."
        )
    return run_id


def probe_and_claim(tpu_name: str, zone: str, project: str, run_id: str, *, claim: bool = True) -> PodProbe:
    """Read the certificate, count python processes, and claim the pod if it is free.

    One ssh round trip does all three, which is what makes it safe: checking and then
    claiming in two calls leaves a window of seconds, long enough for two launchers
    sweeping at the same moment to both see a pod as free. The create uses ``noclobber``,
    so the kernel decides the winner.

    With ``claim=False`` the pod is only inspected, which is what the resume sweep wants:
    it is looking for its own certificate on pods of any shape and must not take a pod it
    merely walked past.
    """
    validate_run_id(run_id)
    write = (
        f'if ( set -o noclobber; echo "{run_id}" > {CERTIFICATE_FILE} ) 2>/dev/null; '
        f'then echo "CERT={run_id}"; else echo "CERT=$(cat {CERTIFICATE_FILE} 2>/dev/null)"; fi'
        if claim
        else f'echo "CERT=$(cat {CERTIFICATE_FILE} 2>/dev/null)"'
    )
    # The certificate is read first and the claim attempted only when nothing holds the pod,
    # so a pod running someone else's job is never written to.
    command = (
        f"held=$(cat {CERTIFICATE_FILE} 2>/dev/null); pys=$(pgrep -c python 2>/dev/null || echo 0); "
        f'echo "PYS=$pys"; '
        f'if [ -n "$held" ]; then echo "CERT=$held"; '
        f'elif [ "$pys" != 0 ]; then echo "CERT="; '
        f"else {write}; fi"
    )
    try:
        result = ssh_command(tpu_name, zone, command, project=project, worker="0", check=False)
    except Exception as e:
        logger.warning("Certificate probe on %s failed: %s", tpu_name, e)
        return PodProbe(state=PodState.UNREACHABLE)
    if result.returncode != 0 or "PYS=" not in result.stdout:
        logger.info("TPU %s did not answer the certificate probe (rc=%s)", tpu_name, result.returncode)
        return PodProbe(state=PodState.UNREACHABLE)

    fields = dict(
        line.strip().split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line and line.strip().startswith(("PYS=", "CERT="))
    )
    certificate = fields.get("CERT", "").strip() or None
    try:
        processes = int(fields.get("PYS", "0").strip() or 0)
    except ValueError:
        processes = 0

    if certificate == run_id:
        # Either we just won the claim, or this is our own job still running from a
        # previous incarnation of the launcher. The caller distinguishes the two by whether
        # it asked to claim.
        state = PodState.CLAIMED if claim and processes == 0 else PodState.OURS
        return PodProbe(state=state, certificate=certificate, python_processes=processes)
    if certificate is not None:
        logger.info("TPU %s is held by run %r", tpu_name, certificate)
        return PodProbe(state=PodState.BUSY, certificate=certificate, python_processes=processes)
    if processes > 0:
        logger.info("TPU %s has %d python process(es) and no certificate; treating it as busy", tpu_name, processes)
        return PodProbe(state=PodState.BUSY, python_processes=processes)
    logger.info("TPU %s did not accept the claim; another launcher got there first", tpu_name)
    return PodProbe(state=PodState.BUSY)


def read(tpu_name: str, zone: str, project: str) -> str | None:
    """The run id currently occupying a pod, or None if it is unheld or unreachable."""
    try:
        result = ssh_command(
            tpu_name, zone, f"cat {CERTIFICATE_FILE} 2>/dev/null", project=project, worker="0", check=False
        )
    except Exception as e:
        logger.info("Could not read the certificate on %s: %s", tpu_name, e)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def release(tpu_name: str, zone: str, project: str) -> None:
    """Drop this pod's certificate. Best effort, and safe to repeat."""
    try:
        ssh_command(tpu_name, zone, f"rm -f {CERTIFICATE_FILE}", project=project, worker="0", check=False, timeout=180)
        logger.info("Released the run certificate on %s", tpu_name)
    except Exception as e:
        # A pod that cannot be reached has either gone away, taking the certificate with
        # it, or will be cleared by the next launcher that finds it idle.
        logger.info("Could not release the certificate on %s: %s", tpu_name, e)
