"""Round-trip tests for the new rotation helpers in state_action_spaces."""

import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow as tf

import openpi.training.state_action_spaces as sas


class TestQuaternionToAxisAngleTF:
    def test_identity(self):
        # Identity quaternion (xyzw) → zero rotvec.
        q = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype = tf.float32)
        result = sas.quaternion_to_axis_angle_tf(q).numpy()
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, 0.0]], decimal = 6)

    def test_z_90deg(self):
        # 90° about +z (xyzw): [0, 0, sin(45°), cos(45°)] -> rotvec [0, 0, π/2].
        a = np.pi / 2
        q = tf.constant([[0.0, 0.0, np.sin(a / 2), np.cos(a / 2)]], dtype = tf.float32)
        result = sas.quaternion_to_axis_angle_tf(q).numpy()
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, a]], decimal = 6)


class TestAxisAngleToQuaternionTF:
    def test_identity(self):
        # Zero rotvec → identity quaternion (xyzw) = [0, 0, 0, 1].
        aa = tf.constant([[0.0, 0.0, 0.0]], dtype = tf.float32)
        q = sas.axis_angle_to_quaternion_tf(aa).numpy()
        np.testing.assert_array_almost_equal(q, [[0.0, 0.0, 0.0, 1.0]], decimal = 6)

    def test_round_trip(self):
        # Random rotvecs round-trip through (rotvec → quat → rotvec).
        rng = np.random.default_rng(86)
        rotvecs = rng.normal(size = (10, 3)).astype(np.float32) * 0.5  # small angles to stay < π
        q = sas.axis_angle_to_quaternion_tf(tf.constant(rotvecs)).numpy()
        recovered = sas.quaternion_to_axis_angle_tf(tf.constant(q)).numpy()
        np.testing.assert_array_almost_equal(recovered, rotvecs, decimal = 5)


class TestQuatMultiplyTF:
    def test_identity_left(self):
        identity = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype = tf.float32)
        q = tf.constant([[0.1, 0.2, 0.3, 0.9]], dtype = tf.float32)
        q = q / tf.norm(q, axis = -1, keepdims = True)
        product = sas.quat_multiply_tf(identity, q).numpy()
        np.testing.assert_array_almost_equal(product, q.numpy(), decimal = 5)

    def test_inverse(self):
        # q · q⁻¹ = identity (xyzw, conjugate negates xyz)
        q_np = np.array([[0.1, 0.2, 0.3, 0.9]], dtype = np.float32)
        q_np = q_np / np.linalg.norm(q_np, axis = -1, keepdims = True)
        q_inv_np = np.concatenate([-q_np[..., :3], q_np[..., 3:]], axis = -1)
        product = sas.quat_multiply_tf(tf.constant(q_np), tf.constant(q_inv_np)).numpy()
        np.testing.assert_array_almost_equal(product, [[0.0, 0.0, 0.0, 1.0]], decimal = 5)

    def test_matches_scipy(self):
        # Cross-check Hamilton product against scipy.Rotation composition.
        rng = np.random.default_rng(86)
        for _ in range(5):
            r1 = Rotation.from_rotvec(rng.normal(size = 3) * 0.3)
            r2 = Rotation.from_rotvec(rng.normal(size = 3) * 0.3)
            q1 = r1.as_quat().astype(np.float32)
            q2 = r2.as_quat().astype(np.float32)
            our = sas.quat_multiply_tf(
                tf.constant(q1[None]), tf.constant(q2[None])
            ).numpy()[0]
            scipy_result = (r1 * r2).as_quat().astype(np.float32)
            # Quaternions are equivalent up to sign — pick the matching sign.
            if np.dot(our, scipy_result) < 0:
                scipy_result = -scipy_result
            np.testing.assert_array_almost_equal(our, scipy_result, decimal = 5)


class TestQuaternionEulerXYZRoundTrip:
    def test_identity(self):
        q = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype = tf.float32)
        euler = sas.quaternion_to_euler_xyz_tf(q).numpy()
        np.testing.assert_array_almost_equal(euler, [[0.0, 0.0, 0.0]], decimal = 6)
        recovered = sas.euler_xyz_to_quaternion_tf(tf.constant(euler)).numpy()
        np.testing.assert_array_almost_equal(recovered, q.numpy(), decimal = 6)

    def test_round_trip_against_scipy(self):
        rng = np.random.default_rng(86)
        # Avoid pitch ≈ ±π/2 (gimbal lock); restrict euler magnitudes.
        eulers = rng.uniform(-1.0, 1.0, size = (10, 3)).astype(np.float32)
        q_ours = sas.euler_xyz_to_quaternion_tf(tf.constant(eulers)).numpy()
        q_scipy = Rotation.from_euler("xyz", eulers).as_quat().astype(np.float32)
        # Sign-align (quaternions q and -q represent the same rotation).
        signs = np.sign(np.sum(q_ours * q_scipy, axis = -1, keepdims = True))
        q_scipy = q_scipy * signs
        np.testing.assert_array_almost_equal(q_ours, q_scipy, decimal = 5)
        # And convert back through ours and check Euler matches.
        recovered = sas.quaternion_to_euler_xyz_tf(tf.constant(q_ours)).numpy()
        np.testing.assert_array_almost_equal(recovered, eulers, decimal = 4)


class TestInterpolateTrajectoryEulerXYZ:
    def test_euler_xyz_smooth(self):
        """Interpolating a smooth euler trajectory yields a smooth interpolated trajectory."""
        # 5 source frames at 20 Hz, 7 target frames at 30 Hz over the same duration.
        source_times = tf.constant([0.0, 0.05, 0.10, 0.15, 0.20], dtype = tf.float32)
        target_times = tf.constant([0.0, 0.0333, 0.0667, 0.10, 0.1333, 0.1667, 0.20], dtype = tf.float32)
        eulers = tf.constant(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
            dtype = tf.float32,
        )
        spec = sas.StateActionSpaceSpec(
            total_dim = 3,
            dimensions = (sas.DimensionSpec("rot", 0, 3, "euler_xyz", chunk_delta = False),),
        )
        result = sas.interpolate_trajectory_tf(eulers, source_times, target_times, spec).numpy()
        assert result.shape == (7, 3)
        # First and last should match endpoints.
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0, 0.0], decimal = 4)
        np.testing.assert_array_almost_equal(result[-1], [0.4, 0.0, 0.0], decimal = 4)
        # Roll should be monotonically non-decreasing (it's a smooth ramp).
        assert (np.diff(result[:, 0]) >= -1e-5).all()
