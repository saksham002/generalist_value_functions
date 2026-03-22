import numpy as np

from eval.helpers import candidate_actions
from eval.helpers.subtask_tracking import ShirtHangSubtaskDetector


def _make_obs(
    *,
    left_gripper: float = 850.0,
    right_gripper: float = 850.0,
    right_z: float = 0.35,
    right_speed: float = 0.1,
    left_speed: float = 0.1,
) -> dict:
    right_vel = np.array([right_speed, 0.0, 0.0, 0.0, 0.0, 0.0], dtype = np.float32)
    left_vel = np.array([left_speed, 0.0, 0.0, 0.0, 0.0, 0.0], dtype = np.float32)
    return {
        "state": {
            "left/gripper_pos": np.array([left_gripper], dtype = np.float32),
            "right/gripper_pos": np.array([right_gripper], dtype = np.float32),
            "right/tcp_pose": np.array([0.0, 0.0, right_z, 0.0, 0.0, 0.0, 1.0], dtype = np.float32),
            "right/tcp_vel": right_vel,
            "left/tcp_vel": left_vel,
        }
    }


def test_subsample_even_actions():
    actions = np.arange(60 * 14, dtype = np.float32).reshape(60, 14)
    subsampled = candidate_actions.subsample_even_actions(actions)
    assert subsampled.shape == (30, 14)
    assert np.allclose(subsampled[0], actions[0])
    assert np.allclose(subsampled[1], actions[2])


def test_shirt_hang_detector_advances_through_boundaries():
    detector = ShirtHangSubtaskDetector()

    prompt = detector.update(_make_obs(right_gripper = 300.0, right_z = 0.35), 0)
    assert prompt == "Grasp the hanger"

    prompt = detector.update(_make_obs(right_gripper = 300.0, right_z = 0.15), 2)
    assert prompt == "Lift the hanger off the rod"

    for sample_idx in range(10):
        prompt = detector.update(_make_obs(right_gripper = 300.0, right_z = 0.15, right_speed = 0.0), 4 + 2 * sample_idx)
    assert prompt == "Pass hanger from right to left arm"

    for sample_idx in range(40):
        prompt = detector.update(_make_obs(left_gripper = 300.0, right_gripper = 300.0, right_z = 0.15), 24 + 2 * sample_idx)
    assert prompt == "Hook one side of the shirt onto the hanger"
