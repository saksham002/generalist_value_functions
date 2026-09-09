"""Tests for command parsing, path placement and run identity.

None of these touch GCP: every external call in the package funnels through
``run_gcloud``, so the parts that decide *what* to do can be exercised directly and only
the parts that decide *how to ask GCP* need a live project.
"""

import dataclasses
import subprocess

import pytest

from openpi.tpu import certificate
from openpi.tpu import quota
from openpi.tpu.config import RemoteLayout
from openpi.tpu.config import get_tpu_name_prefix
from openpi.tpu.config import get_tpu_type_prefix
from openpi.tpu.filters import PodFilter
from openpi.tpu.launch import CommandPlan
from openpi.tpu.launch import LaunchConfig
from openpi.tpu.launch import ProgressReporter
from openpi.tpu.launch import UriRole
from openpi.tpu.launch import command_with_done_marker
from openpi.tpu.manager import AllocationRequest
from openpi.tpu.manager import may_reclaim

TRAIN_COMMAND = (
    "python scripts/train_value_function.py my_config --resume "
    "--fine-tune my_ft --exp-name run7 "
    "--checkpoint-base-dir gs://saksham-euw4/ckpt "
    "--data.rlds-data-dir gs://saksham-euw4/datasets/robocoin "
    "--validation-cache-dir /nfs/aidm_nfs/saksham3/robocoin/val_cache "
    "--step 20000"
)


def test_parse_identifies_the_run() -> None:
    plan = CommandPlan.parse(TRAIN_COMMAND)
    assert plan.entrypoint == "scripts/train_value_function.py"
    assert plan.config_name == "my_config"
    assert plan.exp_name == "run7"
    assert plan.fine_tune == "my_ft"
    assert plan.is_training
    assert plan.checkpoint_root == "gs://saksham-euw4/ckpt/my_config/run7"


def test_parse_assigns_uri_roles() -> None:
    plan = CommandPlan.parse(TRAIN_COMMAND)
    roles = {ref.span.text: ref.role for ref in plan.uris}
    assert roles["gs://saksham-euw4/ckpt"] is UriRole.WRITE
    assert roles["gs://saksham-euw4/datasets/robocoin"] is UriRole.SHARED_READ


def test_fine_tune_override_flags_are_not_the_fine_tune_name() -> None:
    plan = CommandPlan.parse(
        "python scripts/train_value_function.py cfg --fine-tune.data-factory.rlds-data-dir gs://b/x"
    )
    assert plan.fine_tune is None


def test_nfs_paths_are_located() -> None:
    plan = CommandPlan.parse(TRAIN_COMMAND)
    paths = {ref.flag: ref.span.text for ref in plan.paths}
    assert paths["validation-cache-dir"] == "/nfs/aidm_nfs/saksham3/robocoin/val_cache"


def test_rewrite_does_not_corrupt_a_uri_that_prefixes_another() -> None:
    """The bug a global str.replace had: gs://b/ckpt is a prefix of gs://b/ckpt2."""
    command = "python scripts/train.py cfg --checkpoint-base-dir gs://b/ckpt --data.rlds-data-dir gs://b/ckpt2"
    plan = CommandPlan.parse(command)
    short = next(ref for ref in plan.uris if ref.span.text == "gs://b/ckpt")
    rewritten = plan.rewrite({short.span: "gs://other/ckpt"})
    assert rewritten.endswith("gs://b/ckpt2")
    assert "gs://other/ckpt " in rewritten
    assert "gs://other/ckpt2" not in rewritten


def test_rewrite_applies_several_spans_without_shifting_offsets() -> None:
    command = "python scripts/train.py cfg --checkpoint-base-dir gs://a/x --data.rlds-data-dir gs://a/yy"
    plan = CommandPlan.parse(command)
    spans = {ref.span: f"gs://z/{i}" for i, ref in enumerate(plan.uris)}
    rewritten = plan.rewrite(spans)
    assert "gs://a/" not in rewritten
    assert rewritten.count("gs://z/") == 2


def test_run_id_is_stable_across_a_re_race_into_another_region() -> None:
    """The whole point of the certificate: the id must survive localization."""
    eu = LaunchConfig(command=TRAIN_COMMAND)
    us = LaunchConfig(command=TRAIN_COMMAND.replace("saksham-euw4", "saksham-usc2"))
    assert eu.run_id == us.run_id


def test_run_id_changes_when_the_run_changes() -> None:
    base = LaunchConfig(command=TRAIN_COMMAND)
    other_exp = LaunchConfig(command=TRAIN_COMMAND.replace("--exp-name run7", "--exp-name run8"))
    other_ft = LaunchConfig(command=TRAIN_COMMAND.replace("--fine-tune my_ft", "--fine-tune other_ft"))
    assert len({base.run_id, other_exp.run_id, other_ft.run_id}) == 3


def test_run_id_is_shell_safe() -> None:
    config = LaunchConfig(command=TRAIN_COMMAND)
    assert certificate.validate_run_id(config.run_id) == config.run_id
    assert "run7" in config.run_id


@pytest.mark.parametrize("unsafe", ["a b", "a;rm -rf /", "$(id)", "a'b"])
def test_unsafe_run_ids_are_rejected(unsafe: str) -> None:
    with pytest.raises(ValueError, match="Unsafe run id"):
        certificate.validate_run_id(unsafe)


def _layout(*, uses_nfs: bool) -> RemoteLayout:
    return RemoteLayout(
        uses_nfs=uses_nfs,
        nfs_mount_path="/nfs/aidm_nfs" if uses_nfs else None,
        nfs_user="saksham3",
        worker_count=8,
    )


def test_nfs_paths_survive_untouched_on_a_filer_pod() -> None:
    layout = _layout(uses_nfs=True)
    path = "/nfs/aidm_nfs/saksham3/robocoin/val_cache"
    assert layout.localize_path(path) == path
    assert layout.working_dir == "/nfs/aidm_nfs/saksham3/batch_value_learning"
    assert layout.sync_workers is None
    assert layout.shared_workers == "0"


def test_nfs_paths_become_home_paths_on_a_local_disk_pod() -> None:
    layout = _layout(uses_nfs=False)
    assert layout.localize_path("/nfs/aidm_nfs/saksham3/robocoin/val_cache") == "$HOME/robocoin/val_cache"
    assert layout.localize_path("/data/user_data/saksham3/cache") == "/data/user_data/saksham3/cache"
    assert layout.working_dir == "~/batch_value_learning"
    assert layout.sync_workers == list(range(8))
    assert layout.shared_workers == "all"


def test_done_marker_tail_only_runs_after_success() -> None:
    tail = command_with_done_marker("true", "gs://b/marker")
    assert tail.startswith("true && {")
    assert tail.count("gcloud storage cp /dev/null gs://b/marker") == 1
    assert command_with_done_marker("true", None) == "true"


def test_every_us_eu_zone_with_fitting_quota_is_raced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Corroboration orders candidates; it no longer excludes them.

    Excluding on it could never bootstrap a zone: being raced required occupancy and
    occupancy required being raced.
    """
    limits = {"us-central2-b": 64, "europe-west4-a": 2048, "us-east1-d": 1536, "asia-east1-c": 1536}
    monkeypatch.setattr(quota, "_fetch_quota_limits", lambda family, project: limits)
    monkeypatch.setattr(quota, "_fetch_quota_overrides", lambda family, project: frozenset({"us-central2-b"}))

    zones = quota.spot_quota_zones("v4", 64, project="p")
    # The overridden zone leads despite the smallest grant; off-continent is still excluded.
    assert zones == ("us-central2-b", "europe-west4-a", "us-east1-d")
    assert "asia-east1-c" not in zones


def test_a_zone_holding_a_pod_is_ordered_ahead_of_an_uncorroborated_one(monkeypatch: pytest.MonkeyPatch) -> None:
    limits = {"us-east1-d": 1536, "europe-west4-a": 1536}
    monkeypatch.setattr(quota, "_fetch_quota_limits", lambda family, project: limits)
    monkeypatch.setattr(quota, "_fetch_quota_overrides", lambda family, project: frozenset())

    occupied = frozenset({"us-east1-d"})
    assert quota.spot_quota_zones("v4", 64, project="p", occupied_zones=occupied)[0] == "us-east1-d"


def test_max_zones_caps_the_blast_radius(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each candidate holds quota in a shared project until a winner is picked."""
    limits = {f"us-west1-{c}": 1536 for c in "abc"} | {"europe-west4-a": 2048}
    monkeypatch.setattr(quota, "_fetch_quota_limits", lambda family, project: limits)
    monkeypatch.setattr(quota, "_fetch_quota_overrides", lambda family, project: frozenset({"europe-west4-a"}))

    assert quota.spot_quota_zones("v4", 64, project="p", max_zones=2) == ("europe-west4-a", "us-west1-a")


def test_a_shape_too_big_for_every_grant_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quota, "_fetch_quota_limits", lambda family, project: {"us-east1-d": 32})
    monkeypatch.setattr(quota, "_fetch_quota_overrides", lambda family, project: frozenset())
    with pytest.raises(ValueError, match="no zone there has spot quota"):
        quota.spot_quota_zones("v4", 64, project="p")


def test_region_filter_scopes_the_race(monkeypatch: pytest.MonkeyPatch) -> None:
    limits = {"us-central2-b": 64, "europe-west4-a": 64, "europe-west4-b": 64}
    monkeypatch.setattr(quota, "_fetch_quota_limits", lambda family, project: limits)
    monkeypatch.setattr(quota, "_fetch_quota_overrides", lambda family, project: frozenset(limits))

    assert quota.spot_quota_zones("v4", 64, project="p", region="europe-west4") == (
        "europe-west4-a",
        "europe-west4-b",
    )
    with pytest.raises(ValueError, match="No eligible zone"):
        quota.spot_quota_zones("v4", 64, project="p", region="us-east5")


# ---------------------------------------------------------------------------------------
# Placement: the four guarantees ResolvedLaunch enforces, with GCS stubbed out.
# ---------------------------------------------------------------------------------------

BUCKET_REGIONS = {
    "saksham-euw4": "europe-west4",
    "saksham-usc2": "us-central2",
}


def _pod(zone: str, *, uses_nfs: bool = True):
    from openpi.tpu.config import PodConfig
    from openpi.tpu.config import get_tpu_user

    return PodConfig(
        tpu_type="v4-64",
        family="v4",
        accelerator_type="v4-64",
        zone=zone,
        project="p",
        is_spot=True,
        runtime_version="tpu-ubuntu2204-base",
        nfs_server="10.0.0.1:/share" if uses_nfs else None,
        nfs_mount_path="/nfs/aidm_nfs" if uses_nfs else None,
        user_key="saksham",
        user=get_tpu_user("saksham"),
    )


@pytest.fixture
def stub_gcs(monkeypatch: pytest.MonkeyPatch):
    """Bucket regions and existence, without talking to GCS."""
    from openpi.tpu import buckets as buckets_module

    existing: set[str] = set()
    monkeypatch.setattr(buckets_module, "bucket_region", lambda b: BUCKET_REGIONS.get(b))
    monkeypatch.setattr(
        buckets_module,
        "ensure_regional_bucket",
        lambda region, *, resource_owner: {"europe-west4": "saksham-euw4", "us-central2": "saksham-usc2"}[region],
    )
    monkeypatch.setattr(buckets_module, "uri_exists", lambda uri: uri in existing)
    return existing


def test_writes_land_in_the_pods_own_region(stub_gcs) -> None:
    stub_gcs.add("gs://saksham-usc2/datasets/robocoin")
    resolved = LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("us-central2-b"))
    assert "--checkpoint-base-dir gs://saksham-usc2/ckpt " in resolved.command
    assert resolved.checkpoint_root == "gs://saksham-usc2/ckpt/my_config/run7"


def test_same_continent_reads_are_left_alone(stub_gcs) -> None:
    """europe-west4 -> europe-west4: nothing to do, and nothing copied."""
    resolved = LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("europe-west4-a"))
    assert "gs://saksham-euw4/datasets/robocoin" in resolved.command


def test_cross_continent_read_resolves_to_the_hub_replica(stub_gcs) -> None:
    stub_gcs.add("gs://saksham-usc2/datasets/robocoin")
    resolved = LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("us-central2-b"))
    assert "--data.rlds-data-dir gs://saksham-usc2/datasets/robocoin " in resolved.command


def test_cross_continent_read_without_a_replica_crashes(stub_gcs) -> None:
    """Guarantee 1: crash rather than quietly pay inter-continent egress all run."""
    with pytest.raises(ValueError, match="no same-continent replica"):
        LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("us-central2-b"))


def test_nfs_args_are_localized_only_on_a_local_disk_pod(stub_gcs) -> None:
    stub_gcs.add("gs://saksham-usc2/datasets/robocoin")
    on_filer = LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("europe-west4-a", uses_nfs=True))
    assert "--validation-cache-dir /nfs/aidm_nfs/saksham3/robocoin/val_cache" in on_filer.command

    local = LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("us-central2-b", uses_nfs=False))
    assert "--validation-cache-dir $HOME/robocoin/val_cache" in local.command
    assert "/nfs/aidm_nfs" not in local.command


def test_paligemma_comes_from_the_pods_own_continent(stub_gcs) -> None:
    stub_gcs.add("gs://saksham-usc2/datasets/robocoin")
    assert (
        LaunchConfig(command=TRAIN_COMMAND)
        .resolve(_pod("europe-west4-a"))
        .paligemma_weights_uri.startswith("gs://saksham-euw4/")
    )
    assert (
        LaunchConfig(command=TRAIN_COMMAND)
        .resolve(_pod("us-central2-b"))
        .paligemma_weights_uri.startswith("gs://saksham-usc2/")
    )


def test_carry_refuses_to_cross_continents_without_the_flag(stub_gcs, monkeypatch) -> None:
    """Guarantee 3: the carry is the one sanctioned copy, and it is gated."""
    from openpi.tpu import buckets as buckets_module

    calls: list[tuple] = []

    def fake_carry(source, destination, *, allow_cross_continent=False):
        calls.append((source, destination, allow_cross_continent))

    monkeypatch.setattr(buckets_module, "carry_checkpoints", fake_carry)
    stub_gcs.add("gs://saksham-usc2/datasets/robocoin")

    resolved = LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("us-central2-b"))
    resolved.carry_from("gs://saksham-euw4/ckpt/my_config/run7")
    # Base root and the fine-tune subtree, both refusing to cross by default.
    assert [c[2] for c in calls] == [False, False]
    assert calls[1][1].endswith("/my_ft")

    calls.clear()
    permitted = dataclasses.replace(
        LaunchConfig(command=TRAIN_COMMAND), allow_cross_continent_checkpoint_transfer=True
    ).resolve(_pod("us-central2-b"))
    permitted.carry_from("gs://saksham-euw4/ckpt/my_config/run7")
    assert [c[2] for c in calls] == [True, True]


def test_mesh_flag_is_added_only_for_v6e(stub_gcs) -> None:
    from openpi.tpu.config import PodConfig
    from openpi.tpu.config import get_tpu_user

    v6e = PodConfig(
        tpu_type="v6e-32",
        family="v6e",
        accelerator_type="v6e-32",
        zone="europe-west4-a",
        project="p",
        is_spot=True,
        runtime_version="v2-alpha-tpuv6e",
        nfs_server="s:/x",
        nfs_mount_path="/nfs/aidm_nfs",
        user_key="saksham",
        user=get_tpu_user("saksham"),
    )
    assert LaunchConfig(command=TRAIN_COMMAND).resolve(v6e).command.endswith("--fsdp-devices=8")
    assert "--fsdp-devices" not in LaunchConfig(command=TRAIN_COMMAND).resolve(_pod("europe-west4-a")).command


# --- Race naming ------------------------------------------------------------------------
# Two launches racing the same shape at the same moment must never name their candidates
# identically: candidates are torn down by name, so a shared name means one launch deletes
# the other's winner.


def _request(command: str) -> AllocationRequest:
    return AllocationRequest(run_id=LaunchConfig(command=command).run_id, tpu_type="v5e-64", spot=True)


def test_two_runs_racing_one_shape_get_different_name_prefixes() -> None:
    first = _request(TRAIN_COMMAND)
    second = _request(TRAIN_COMMAND.replace("--exp-name run7", "--exp-name run8"))
    prefixes = {
        get_tpu_name_prefix("v5e-64", resource_owner="saksham", is_spot=True, run_token=request.name_token)
        for request in (first, second)
    }
    assert len(prefixes) == 2


def test_the_name_token_survives_a_re_race_into_another_region() -> None:
    """A relaunched run must count indices in the namespace its earlier attempt used."""
    european = _request(TRAIN_COMMAND)
    american = _request(TRAIN_COMMAND.replace("saksham-euw4", "saksham-usc2"))
    assert european.name_token == american.name_token


def test_a_raced_name_stays_recognisable_to_reuse_and_cleanup() -> None:
    """The token goes before the index, so the checks that find a pod later still match."""
    prefix = get_tpu_name_prefix("v5e-64", resource_owner="saksham", is_spot=True, run_token="abc12345")
    name = f"{prefix}-0"
    # _find_idle_pod and cleanup_preempted both key on the family prefix.
    assert name.startswith(get_tpu_type_prefix("v5e-64"))
    # Ownership, and therefore the right to reclaim, is a substring test on the name.
    assert "saksham" in name
    assert name == "v5e-saksham-spot-64-abc12345-0"


def test_an_untokenised_prefix_is_unchanged() -> None:
    """Callers that do not scope by run keep the name they had."""
    assert get_tpu_name_prefix("v6e-32", resource_owner="saksham", is_spot=True) == "v6e-saksham-spot-32"


# --- Progress milestones -----------------------------------------------------------------


def _reporter(pattern: str, every: int = 20):
    return ProgressReporter(
        LaunchConfig(command=TRAIN_COMMAND, progress_pattern=pattern, progress_every=every),
        _RecordingNotifier(),
        "run-abc12345",
    )


class _RecordingNotifier:
    def __init__(self) -> None:
        self.milestones: list[int] = []

    def notify_progress(self, tpu_name: str, run_id: str, percent: int, detail: str = "") -> bool:
        self.milestones.append(percent)
        return True


def test_progress_is_disabled_without_a_pattern() -> None:
    assert not _reporter("").enabled
    assert _reporter(r"(\d+)%\|").enabled


def test_two_groups_read_as_current_over_total() -> None:
    assert _reporter(r"step (\d+)/(\d+)").percent_from("step 5000/20000") == pytest.approx(25.0)


def test_one_group_reads_as_a_percent() -> None:
    assert _reporter(r"(\d+)%\|").percent_from("  45%|####      | 9/20") == pytest.approx(45.0)


def test_the_last_match_in_the_tail_wins() -> None:
    """A log tail holds several updates; only the most recent says where the job is."""
    tail = "10%|# | 1/10\r 20%|## | 2/10\r 30%|### | 3/10"
    assert _reporter(r"(\d+)%\|").percent_from(tail) == pytest.approx(30.0)


def test_a_tail_with_no_progress_reports_nothing() -> None:
    assert _reporter(r"(\d+)%\|").percent_from("Traceback (most recent call last):") is None


def test_milestones_never_repeat_or_go_backwards() -> None:
    """A preemption retry resumes from a checkpoint; it must not replay what it passed."""
    reporter = _reporter(r"(\d+)%\|")
    for tail in ("21%|", "25%|", "44%|", "41%|", "9%|"):
        reporter.observe(tail, "pod-0")
    assert reporter.notifier.milestones == [20, 40]


def test_a_resumed_run_announces_only_where_it_lands() -> None:
    reporter = _reporter(r"(\d+)%\|")
    reporter.observe("83%|", "pod-0")
    assert reporter.notifier.milestones == [80]


def test_division_by_a_zero_total_is_not_fatal() -> None:
    assert _reporter(r"step (\d+)/(\d+)").percent_from("step 0/0") is None


def test_a_spot_launch_may_name_several_shapes() -> None:
    request = AllocationRequest(run_id="r-1", tpu_type="v5e-64,v6e-32", spot=True)
    assert request.shapes == ("v5e-64", "v6e-32")
    assert request.single_shape is None


def test_one_shape_still_reads_as_one() -> None:
    request = AllocationRequest(run_id="r-1", tpu_type="v5e-64", spot=True)
    assert request.shapes == ("v5e-64",)
    assert request.single_shape == "v5e-64"


# --- The certificate probe's shell ---------------------------------------------------
# Exercised against a stubbed pgrep rather than a pod: the failure this pins cost every
# reuse attempt in a launch session, and it is invisible to reading the Python.


def _run_probe(tmp_path, pgrep_stdout: str, pgrep_exit: int, run_id: str = "run-abc12345") -> dict[str, str]:
    """Run the real probe command under sh with pgrep stubbed, and parse what it printed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "pgrep"
    stub.write_text(f"#!/bin/sh\nprintf '{pgrep_stdout}'\nexit {pgrep_exit}\n")
    stub.chmod(0o755)
    result = subprocess.run(
        ["sh", "-c", certificate.probe_command(run_id)],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
        check=False,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if line.startswith(("PYS=", "CERT="))
    )


def test_an_idle_pod_is_claimable(tmp_path) -> None:
    """`pgrep -c` prints its count AND exits non-zero when that count is zero.

    An `|| echo 0` fallback therefore fired as well, making pys "0\\n0" -- which is not
    equal to "0", so the probe reported the pod busy. Every idle pod read as BUSY, reuse
    could never succeed, and a reclaim was aimed at pods doing nothing wrong.
    """
    fields = _run_probe(tmp_path, pgrep_stdout="0\\n", pgrep_exit=1)
    assert fields["PYS"].strip() == "0"
    assert fields["CERT"] == "run-abc12345"


def test_a_pod_running_python_is_not_claimed(tmp_path) -> None:
    fields = _run_probe(tmp_path, pgrep_stdout="3\\n", pgrep_exit=0)
    assert fields["PYS"].strip() == "3"
    assert fields["CERT"] == ""
    assert not (tmp_path / "tpu_run_id").exists()


def test_an_existing_certificate_is_never_overwritten(tmp_path) -> None:
    (tmp_path / "tpu_run_id").write_text("someone-elses-run\n")
    fields = _run_probe(tmp_path, pgrep_stdout="0\\n", pgrep_exit=1)
    assert fields["CERT"].strip() == "someone-elses-run"
    assert (tmp_path / "tpu_run_id").read_text().strip() == "someone-elses-run"


def test_a_certificate_holder_is_never_reclaimed() -> None:
    """Between a claim and its job starting, a pod runs no python for several minutes.

    Judging by processes alone read that as idle and took the pod from the run setting up
    on it, deleting its certificate and writing another.
    """
    held = certificate.PodProbe(state=certificate.PodState.BUSY, certificate="someone-elses-run")
    assert not may_reclaim(held, owned=True, reclaim_owned=True)


def test_stale_processes_with_no_certificate_may_be_reclaimed() -> None:
    stale = certificate.PodProbe(state=certificate.PodState.BUSY, python_processes=3)
    assert may_reclaim(stale, owned=True, reclaim_owned=True)
    # Ownership still gates it: never kill work on a pod that is not ours.
    assert not may_reclaim(stale, owned=False, reclaim_owned=True)
    assert not may_reclaim(stale, owned=True, reclaim_owned=False)


def test_an_unreachable_pod_is_never_reclaimed() -> None:
    unreachable = certificate.PodProbe(state=certificate.PodState.UNREACHABLE)
    assert not may_reclaim(unreachable, owned=True, reclaim_owned=True)


def test_a_startup_progress_bar_does_not_latch_the_reporter() -> None:
    """A job's own startup bars finish at 100%, and taking one silences every real milestone.

    Observed live: `Fetching 2 files: 100%|##########| 2/2` during weight download was read
    as the run being complete, after which the monotonic rule suppressed everything.
    """
    reporter = _reporter(r"(\d+)%\|")
    reporter.observe("Fetching 2 files: 100%|##########| 2/2 [00:02<00:00,  1.26s/it]", "pod-0")
    assert reporter.notifier.milestones == []
    # A real reading afterwards still works.
    reporter.observe("24%|##        |", "pod-0")
    assert reporter.notifier.milestones == [20]


def test_the_step_pattern_ignores_small_startup_bars() -> None:
    """The training bar's total has five or more digits; a startup bar's does not."""
    reporter = _reporter(r"(\d+)/(\d{5,}) \[")
    assert reporter.percent_from("Fetching 2 files: 100%|###| 2/2 [00:02<00:00]") is None
    assert reporter.percent_from(" 24%|##  | 56175/230000 [01:02<03:04]") == pytest.approx(24.42, abs=0.01)


def test_a_run_resuming_high_still_reports() -> None:
    """The guard must not suppress a legitimate late-stage resume."""
    reporter = _reporter(r"(\d+)/(\d{5,}) \[")
    reporter.observe(" 96%|####| 220000/230000 [01:02<03:04]", "pod-0")
    assert reporter.notifier.milestones == [80]


def test_a_launch_may_name_several_regions() -> None:
    """The safe zones for a launch are defined by pod properties -- a filer, an existing
    bucket -- and those span regions, so a single-region filter cannot express them."""
    request = AllocationRequest(run_id="r-1", region="europe-west4,us-central2", spot=True)
    assert request.regions == ("europe-west4", "us-central2")
    assert request.in_region("europe-west4-b")
    assert request.in_region("us-central2-b")
    # us-south1 has no Filestore and no bucket of ours: exactly what this excludes.
    assert not request.in_region("us-south1-a")


def test_no_region_means_unrestricted() -> None:
    request = AllocationRequest(run_id="r-1", spot=True)
    assert request.regions == ()
    assert request.in_region("us-west1-c")


def test_a_single_region_still_works() -> None:
    request = AllocationRequest(run_id="r-1", region="europe-west4", spot=True)
    assert request.regions == ("europe-west4",)
    assert request.in_region("europe-west4-a")
    assert not request.in_region("us-central2-b")


# --- PodFilter ----------------------------------------------------------------------------


def _filter(**kw):
    kw.setdefault("resource_owner", "saksham")
    return PodFilter.parse(**kw)


def test_the_filter_rejects_a_foreign_pod_of_the_wrong_shape() -> None:
    """Both halves of the failure that put a v5e/v6e launch on a colleague's v4."""
    f = _filter(tpu_type="v5e-64,v6e-32", only_my_pods=True)
    assert "not this user's pod" in f.rejects(name="v4-vansh-spot-64-noprop", zone="us-central2-b")
    # Even our own pod of an unasked-for shape is refused.
    assert "families" in f.rejects(name="v4-saksham-spot-64-0", zone="us-central2-b")
    assert f.rejects(name="v5e-saksham-spot-64-0", zone="europe-west4-b") is None


def test_a_reported_accelerator_beats_the_name() -> None:
    """A name is a guess; what the pod reports is authoritative."""
    f = _filter(tpu_type="v5e-64")
    assert f.rejects(name="v5e-saksham-spot-64-0", zone="europe-west4-b", accelerator_type="v5litepod-64") is None
    assert "accelerator" in f.rejects(
        name="v5e-saksham-spot-64-0", zone="europe-west4-b", accelerator_type="v5litepod-256"
    )


def test_zone_region_and_continent_filters() -> None:
    assert _filter(zone="europe-west4-a").rejects(name="p", zone="europe-west4-b") is not None
    assert _filter(zone="europe-west4-a").rejects(name="p", zone="europe-west4-a") is None
    assert _filter(region="europe-west4").rejects(name="p", zone="us-central2-b") is not None
    assert _filter(continent="eu").rejects(name="p", zone="us-central2-b") is not None
    assert _filter(continent="eu").rejects(name="p", zone="europe-west4-c") is None
    assert _filter(continent="us,eu").rejects(name="p", zone="us-west1-a") is None


def test_an_empty_filter_accepts_everything() -> None:
    assert _filter().rejects(name="anything", zone="us-west1-a") is None
    assert _filter().describe() == "unrestricted"


def test_the_resume_sweep_drops_only_the_shape() -> None:
    """A relaunched launcher does not know which shape won, but still must not stray."""
    f = _filter(tpu_type="v5e-64", region="europe-west4", only_my_pods=True).without_types()
    assert f.rejects(name="v6e-saksham-spot-32-0", zone="europe-west4-a") is None
    assert f.rejects(name="v6e-vansh-spot-32-0", zone="europe-west4-a") is not None
    assert f.rejects(name="v6e-saksham-spot-32-0", zone="us-west1-a") is not None


def test_a_reserved_launch_keeps_its_named_pod_across_retries() -> None:
    """A reserved pod does not disappear, so a retry must not go looking elsewhere.

    Dropping the name sent the retry to _find_idle_pod, which matched the family prefix
    and took a colleague's v4 for a launch pinned to one pod.
    """
    reserved = LaunchConfig(command="python train.py cfg", tpu_name="v4-64-0", tpu_type="v4-64")
    assert reserved.tpu_name_for_attempt(0) == "v4-64-0"
    assert reserved.tpu_name_for_attempt(3) == "v4-64-0"

    spot = LaunchConfig(command="python train.py cfg", tpu_name="v4-64-0", tpu_type="v4-64", spot=True)
    assert spot.tpu_name_for_attempt(0) == "v4-64-0"
    assert spot.tpu_name_for_attempt(1) is None


# ---------------------------------------------------------------------------------------
# The spot race honours the same placement restriction as the reuse pass.
# ---------------------------------------------------------------------------------------


def test_race_keeps_only_zones_the_launch_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--zone`` narrowed the reuse pass but not the race, which once put a launch pinned
    to europe-west4-a on a pod in us-west1-c with no filer."""
    from openpi.tpu import manager

    quota_zones = ("europe-west4-b", "us-east1-d", "europe-west4-a", "us-west1-c")
    monkeypatch.setattr(manager, "spot_race_configs", lambda *_, **__: tuple(_pod(z) for z in quota_zones))

    by_zone = AllocationRequest(run_id="r-1", tpu_type="v5e-64", spot=True, zone="europe-west4-a")
    assert [c.zone for c in manager.race_zone_configs("v5e-64", by_zone)] == ["europe-west4-a"]

    by_continent = AllocationRequest(run_id="r-1", tpu_type="v5e-64", spot=True, continent="eu")
    assert [c.zone for c in manager.race_zone_configs("v5e-64", by_continent)] == ["europe-west4-b", "europe-west4-a"]

    unrestricted = AllocationRequest(run_id="r-1", tpu_type="v5e-64", spot=True)
    assert [c.zone for c in manager.race_zone_configs("v5e-64", unrestricted)] == list(quota_zones)

    nowhere = AllocationRequest(run_id="r-1", tpu_type="v5e-64", spot=True, zone="us-central2-b")
    with pytest.raises(RuntimeError, match="placement restriction"):
        manager.race_zone_configs("v5e-64", nowhere)
