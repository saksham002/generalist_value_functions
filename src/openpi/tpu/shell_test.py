"""Behavioural tests for the shell the launcher runs on pods.

Every command in this package is a string that some other shell executes minutes later on
a machine nobody is watching, and its failures are silent by construction: a wrong exit
code is recorded as the job's, a mis-parsed count is read as "busy", a permission error on
a timestamp aborts a launch. Reading the Python cannot catch any of that. These tests run
the real command strings under a real shell, stubbing only what must not touch a pod.

Two live bugs were found this way and are pinned below:
``test_an_idle_pod_is_claimable`` (in launch_test) and ``test_rsync_does_not_preserve_times``.
"""

import inspect
import shutil
import subprocess

import pytest

from openpi.tpu import code_sync
from openpi.tpu import job
from openpi.tpu.config import PodConfig
from openpi.tpu.config import RemoteLayout
from openpi.tpu.config import get_tpu_user
from openpi.tpu.job import JobRunner
from openpi.tpu.launch import command_with_done_marker

BASH = shutil.which("bash")
requires_bash = pytest.mark.skipif(BASH is None, reason="needs bash for PIPESTATUS")


def _layout(tmp_path) -> RemoteLayout:
    return RemoteLayout(uses_nfs=False, nfs_mount_path=None, nfs_user="saksham3", worker_count=1)


def _runner(tmp_path) -> JobRunner:
    pod = PodConfig(
        tpu_type="v5e-64",
        family="v5e",
        accelerator_type="v5litepod-64",
        zone="europe-west4-b",
        project="p",
        is_spot=True,
        runtime_version="v2-alpha-tpuv5-lite",
        nfs_server=None,
        nfs_mount_path=None,
        user_key="saksham",
        user=get_tpu_user("saksham"),
    )
    return JobRunner("pod-0", pod, _layout(tmp_path))


def _run(script: str, tmp_path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), **(env or {})},
        check=False,
    )


# --- the job line -----------------------------------------------------------------------


@requires_bash
def test_a_failing_command_records_its_own_exit_code(tmp_path) -> None:
    """`tee` always succeeds, so a naive $? would record every failed run as a success."""
    body = _runner(tmp_path).build_job_body("exit 42")
    result = _run(body, tmp_path)
    assert result.returncode == 42
    assert (tmp_path / "tpu_job_exit_code").read_text().strip() == "42"


@requires_bash
def test_a_successful_command_records_zero(tmp_path) -> None:
    body = _runner(tmp_path).build_job_body("true")
    result = _run(body, tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "tpu_job_exit_code").read_text().strip() == "0"


@requires_bash
def test_the_certificate_is_dropped_however_the_job_ends(tmp_path) -> None:
    for command in ("true", "exit 7"):
        (tmp_path / "tpu_run_id").write_text("some-run\n")
        _run(_runner(tmp_path).build_job_body(command), tmp_path)
        assert not (tmp_path / "tpu_run_id").exists(), f"certificate survived `{command}`"


@requires_bash
def test_the_certificate_is_dropped_when_the_shell_is_signalled(tmp_path) -> None:
    """The explicit rm never runs if the shell dies mid-pipeline; the EXIT trap is why."""
    (tmp_path / "tpu_run_id").write_text("some-run\n")
    body = _runner(tmp_path).build_job_body("kill -TERM $$")
    _run(body, tmp_path)
    assert not (tmp_path / "tpu_run_id").exists()


@requires_bash
def test_the_job_log_captures_stderr(tmp_path) -> None:
    """Monitoring reads this log for the failure tail, and tracebacks go to stderr."""
    _run(_runner(tmp_path).build_job_body("echo boom >&2; exit 1"), tmp_path)
    assert "boom" in (tmp_path / "tpu_job_output.log").read_text()


# --- the done-marker handshake ------------------------------------------------------------


@requires_bash
def test_the_marker_is_not_written_when_the_command_fails(tmp_path) -> None:
    """`&&` binds only to the first command of a `;` list, so the tail must be grouped."""
    script = command_with_done_marker("false", "MARKER").replace(
        "gcloud storage cp /dev/null MARKER", "touch marker"
    )
    _run(script, tmp_path, env={"HOSTNAME": "pod-w-0"})
    assert not (tmp_path / "marker").exists()


@requires_bash
def test_only_one_worker_writes_the_marker(tmp_path) -> None:
    """N workers mutating one GCS object at once means 429s, reported as a failed run."""
    script = command_with_done_marker("true", "MARKER").replace(
        "gcloud storage cp /dev/null MARKER", "touch marker"
    )
    written = []
    for worker in range(4):
        target = tmp_path / f"w{worker}"
        target.mkdir()
        # `hostname` is the only worker identifier available in the ssh environment.
        bin_dir = target / "bin"
        bin_dir.mkdir()
        (bin_dir / "hostname").write_text(f"#!/bin/sh\necho pod-w-{worker}\n")
        (bin_dir / "hostname").chmod(0o755)
        _run(script, target, env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
        written.append((target / "marker").exists())
    assert written == [True, False, False, False]


@requires_bash
def test_a_single_host_still_writes_the_marker(tmp_path) -> None:
    """A hostname with no -w- suffix is one host, which therefore must write it itself."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "hostname").write_text("#!/bin/sh\necho solo-host\n")
    (bin_dir / "hostname").chmod(0o755)
    script = command_with_done_marker("true", "MARKER").replace(
        "gcloud storage cp /dev/null MARKER", "touch marker"
    )
    _run(script, tmp_path, env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert (tmp_path / "marker").exists()


# --- process-owner parsing ----------------------------------------------------------------


@requires_bash
def test_process_owner_parsing_matches_real_python_names(tmp_path) -> None:
    """`ps` reports python3 / python3.11, not `python`, so an exact match finds nothing.

    Reading no owners is not harmless: the caller treats an empty set as "stale or
    foreign" and kills the processes.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ps").write_text(
        "#!/bin/sh\nprintf 'alice python3.11\\nbob python3\\nroot sshd\\nalice python\\n'\n"
    )
    (bin_dir / "ps").chmod(0o755)
    command = "ps -eo user=,comm= | awk '$2 ~ /^python/ {print $1}' | sort -u"
    result = _run(command, tmp_path, env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert sorted(result.stdout.split()) == ["alice", "bob"]


# --- rsync ---------------------------------------------------------------------------------


def test_rsync_does_not_preserve_times() -> None:
    """Preserving mtimes calls utime(), which needs ownership rather than write permission.

    Per-worker uid mappings on NFS mean a file synced by an earlier pod is owned by a
    different uid on this one, so `-t` made rsync exit 23 over timestamps alone and aborted
    the launch after the content had already copied.
    """
    for fn in (code_sync.sync_code_to_worker, code_sync.sync_gemma_helper):
        flags = [line for line in inspect.getsource(fn).splitlines() if '"-r' in line]
        assert flags, f"no rsync flag block found in {fn.__name__}"
        assert all("t" not in line.split('"')[1] for line in flags), (
            f"{fn.__name__} preserves times; utime() fails under per-worker uids"
        )


def test_monitor_probes_are_bounded(tmp_path, monkeypatch) -> None:
    """A probe against a preempted pod must not outlive the poll round.

    Preemption is detected by `describe` at the top of each round; an unbounded ssh probe
    later in the same round delayed that check by 900s x 3 retries, so a preempted run sat
    looking alive for 45 minutes before its retry could begin.

    Asserted on the timeout actually passed to ssh_command, rather than by reading the
    source: the probe strings also appear in docstrings, and matching those proves nothing.
    """
    seen: list[float | None] = []

    def fake_ssh(*args, **kwargs):
        seen.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="stopped\n", stderr="")

    monkeypatch.setattr(job, "ssh_command", fake_ssh)
    # Driven through the public loop, which is where the stall actually happened: one round
    # runs the session probe, the exit-code read and the output tail.
    monkeypatch.setattr(job, "is_tpu_preempted", lambda *a, **k: False)
    runner = _runner(tmp_path)
    status = runner.monitor_job(poll_interval=0)
    runner.tail_bytes()
    assert status.state in ("completed", "failed")

    assert seen, "no probe reached ssh_command"
    assert all(t is not None and t <= 120 for t in seen), f"unbounded probe timeouts: {seen}"
