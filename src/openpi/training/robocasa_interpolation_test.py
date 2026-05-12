"""Manual-run visualization test for the RoboCasa 20 Hz → 30 Hz interpolation.

Loads ONE real RoboCasa episode, runs the dataset both at native 20 Hz and at
interpolated 30 Hz, and emits side-by-side videos of the right-side third-person
camera plus z / yaw / gripper time-series. Inspect the artifacts to verify the
resampling does what we expect:
  - 30 Hz video looks like a smooth 1.5× resampling of 20 Hz video.
  - z (linear) interpolates smoothly between source samples.
  - yaw (euler-xyz via quat-slerp) is smooth — no 2π jumps unless the trajectory
    actually crosses a gimbal-lock pose.
  - gripper (step / zero-order-hold) holds the source value until the next frame.

Marked `@pytest.mark.manual` because it requires real RLDS shards on disk and
is for diagnostic eyeballing rather than automated CI.

Run with:
    pytest --strict-markers -m manual src/openpi/training/robocasa_interpolation_test.py -s
"""

import pathlib

import imageio
import numpy as np
import pytest

import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.robocasa_rlds_dataset as robocasa_rlds_dataset
import openpi.training.state_action_spaces as state_action_spaces


_DEFAULT_OUT_DIR = pathlib.Path(__file__).resolve().parents[3] / "logs" / "robocasa_interp_test"


@pytest.mark.manual
def test_robocasa_20_to_30_hz_visualization(tmp_path = None):
    out_dir = pathlib.Path(tmp_path) if tmp_path is not None else _DEFAULT_OUT_DIR
    out_dir.mkdir(parents = True, exist_ok = True)

    common = dict(
        data_dir = "/data/group_data/rl/datasets/robocasa_rlds",
        batch_size = 1,
        datasets = (
            rlds_dataset.RLDSDataset(name = "target__atomic__turn_on_microwave", version = "1.0.0", weight = 1.0),
        ),
        critic_mode = False,
        return_trajectories = True,
        max_trajectories = 1,
        shuffle = False,
        action_chunk_size = 30,
    )
    raw_ds = robocasa_rlds_dataset.RoboCasaRldsDataset(**common)
    interp_ds = robocasa_rlds_dataset.RoboCasaRldsDataset(
        **common,
        interpolation_config = state_action_spaces.InterpolationConfig(
            target_fps = 30.0, action_horizon_seconds = 1.0,
        ),
        action_space_spec = state_action_spaces.ROBOCASA_NATIVE_ACTION_SPEC,
        state_space_spec = state_action_spaces.ROBOCASA_NATIVE_STATE_SPEC,
        native_fps = 20.0,
    )

    raw_traj = next(iter(raw_ds))
    interp_traj = next(iter(interp_ds))

    def _frames(traj, key):
        # cam_1 == robot0_agentview_right (after camera-key remap in trajectory_transforms).
        # trajectory mode emits already-decoded uint8 images.
        return [np.asarray(f) for f in np.asarray(traj["observation"][key])]

    imageio.mimsave(
        str(out_dir / "before_20hz.mp4"),
        _frames(raw_traj, "cam_1"),
        format = "mp4", fps = 20, codec = "libx264", quality = 8,
    )
    imageio.mimsave(
        str(out_dir / "after_30hz.mp4"),
        _frames(interp_traj, "cam_1"),
        format = "mp4", fps = 30, codec = "libx264", quality = 8,
    )

    # Converted state layout (13D, euler-xyz):
    #   base_pos[0:3], base_rot[3:6], eef_pos[6:9], eef_rot[9:12], gripper[12:13]
    state_raw = np.asarray(raw_traj["observation"]["state"])      # [T_raw, 13]
    state_int = np.asarray(interp_traj["observation"]["state"])   # [T_int, 13]
    z_raw, yaw_raw, grip_raw = state_raw[:, 8], state_raw[:, 11], state_raw[:, 12]
    z_int, yaw_int, grip_int = state_int[:, 8], state_int[:, 11], state_int[:, 12]

    t_raw = np.arange(len(z_raw)) / 20.0
    t_int = np.arange(len(z_int)) / 30.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize = (10, 8), sharex = True)
    series = [
        ("eef_z (m)", z_raw, z_int),
        ("eef_yaw (rad, euler-xyz z)", yaw_raw, yaw_int),
        ("gripper", grip_raw, grip_int),
    ]
    for ax, (name, raw, interp) in zip(axes, series):
        ax.plot(t_raw, raw, "o-", label = "20 Hz raw", markersize = 3)
        ax.plot(t_int, interp, "x-", label = "30 Hz interp", markersize = 3, alpha = 0.7)
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha = 0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(str(out_dir / "state_comparison.png"), dpi = 150)
    plt.close(fig)

    # Smoke assertions: durations roughly match (within one frame of the slower rate),
    # and the 30 Hz curve has ~1.5× more samples than 20 Hz.
    assert abs(t_raw[-1] - t_int[-1]) < 1.0 / 20.0
    ratio = len(z_int) / max(len(z_raw), 1)
    assert 1.4 < ratio < 1.6, f"unexpected resample ratio {ratio:.2f}"

    print(f"Wrote artifacts to {out_dir}/")
    print(f"  before_20hz.mp4   ({len(z_raw)} frames, {t_raw[-1]:.2f} s)")
    print(f"  after_30hz.mp4    ({len(z_int)} frames, {t_int[-1]:.2f} s)")
    print(f"  state_comparison.png")
