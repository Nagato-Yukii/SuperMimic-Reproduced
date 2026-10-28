import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.data.heading_velocity import heading_velocity


def quat(yaw):
    return Rotation.from_euler('z', np.asarray(yaw)[:, None]).as_quat()[:, [3,0,1,2]]


def test_heading_and_actual_endpoint_dt():
    t = np.arange(101)/50
    p = np.column_stack((0*t, 2*t, 0*t))
    v = heading_velocity(p, quat(np.full(101, np.pi/2)), 50)
    np.testing.assert_allclose(v[:, 0], 2, atol=1e-6)
    np.testing.assert_allclose(v[:, 1:], 0, atol=1e-6)
    np.testing.assert_allclose(v[[0,3,100], 0], [2,2,2], atol=1e-6)


def test_rotation_crossing_pi_and_quaternion_sign():
    t = np.arange(101)/50
    q = quat(3+0.6*t)
    q[::2] *= -1
    v = heading_velocity(np.zeros((101,3)), q, 50)
    np.testing.assert_allclose(v[:,2], .6, atol=1e-6)
    np.testing.assert_allclose(v[[0,100],2], .6, atol=1e-6)


def test_single_frame():
    np.testing.assert_array_equal(heading_velocity(np.zeros((1,3)), quat([0]), 50), [[0,0,0]])


def test_tilted_angular_velocity_is_not_euler_yaw_rate():
    t = np.arange(101)/50
    r = Rotation.from_rotvec(np.column_stack((.3*t, .4*t, .5*t)))
    v = heading_velocity(np.zeros((101,3)), r.as_quat()[:,[3,0,1,2]], 50)
    np.testing.assert_allclose(v[:,2], .5, atol=1e-6)


def test_adjacent_central_difference_50hz():
    t = np.arange(101)/50
    p = np.column_stack((t**3, 0*t, 0*t))
    v = heading_velocity(p, quat(np.zeros(101)), 50)
    np.testing.assert_allclose(v[1:-1,0], 3*t[1:-1]**2 + .02**2, rtol=1e-6)
    np.testing.assert_allclose(v[0,0], .02**2, rtol=1e-6)
    np.testing.assert_allclose(v[-1,0], (t[-1]**3-t[-2]**3)/.02, rtol=1e-6)
