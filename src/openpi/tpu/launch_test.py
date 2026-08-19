"""Tests for command parsing, path placement and run identity.

None of these touch GCP: every external call in the package funnels through
``run_gcloud``, so the parts that decide *what* to do can be exercised directly and only
the parts that decide *how to ask GCP* need a live project.
"""

import dataclasses

import pytest

from openpi.tpu import certificate
from openpi.tpu import quota
from openpi.tpu.config import RemoteLayout
from openpi.tpu.launch import CommandPlan
from openpi.tpu.launch import LaunchConfig
from openpi.tpu.launch import UriRole
from openpi.tpu.launch import command_with_done_marker

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
    assert layout.localize_path("/nfs/aidm_nfs/saksham3/robocoin/val_cache") == "~/robocoin/val_cache"
    assert layout.localize_path("/data/user_data/saksham3/cache") == "/data/user_data/saksham3/cache"
    assert layout.working_dir == "~/batch_value_learning"
    assert layout.sync_workers == list(range(8))
    assert layout.shared_workers == "all"


def test_done_marker_tail_only_runs_after_success() -> None:
    tail = command_with_done_marker("true", "gs://b/marker")
    assert tail.startswith("true && {")
    assert tail.count("gcloud storage cp /dev/null gs://b/marker") == 1
    assert command_with_done_marker("true", None) == "true"


def test_a_zone_needs_corroborating_evidence_not_just_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """The project default grant covers zones that have never served capacity."""
    limits = {"us-central2-b": 64, "europe-west4-a": 64, "asia-east1-c": 64}
    monkeypatch.setattr(quota, "_fetch_quota_limits", lambda family, project: limits)
    monkeypatch.setattr(quota, "_fetch_quota_overrides", lambda family, project: frozenset({"us-central2-b"}))

    assert quota.spot_quota_zones("v4", 64, project="p") == ("us-central2-b",)
    # A zone that currently holds a pod is direct evidence it works.
    occupied = frozenset({"europe-west4-a"})
    assert quota.spot_quota_zones("v4", 64, project="p", occupied_zones=occupied) == (
        "europe-west4-a",
        "us-central2-b",
    )
    # asia-east1-c has quota and would be occupied, but is off-continent.
    assert "asia-east1-c" not in quota.spot_quota_zones(
        "v4", 64, project="p", occupied_zones=frozenset({"asia-east1-c", "europe-west4-a"})
    )


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
    assert "--validation-cache-dir ~/robocoin/val_cache" in local.command
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
