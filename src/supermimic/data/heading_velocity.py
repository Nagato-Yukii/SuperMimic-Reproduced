"""Central pose differences in the centre frame's yaw-only heading axes."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def heading_velocity(position: np.ndarray, quaternion_wxyz: np.ndarray, fps: float, half_window: int = 1) -> np.ndarray:
    """Return (T,3) vx, vy [m/s], wz [rad/s], with clipped endpoint indices.

    Endpoints t-half_window and t+half_window are clipped independently; the
    denominator uses the actual endpoint time difference. Angular velocity
    uses the shortest SO(3) relative rotation in world axes, then yaw-only axes
    at t (so its z component is unchanged). It is not Euler yaw differentiation.
    """
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or q.shape != (len(p), 4):
        raise ValueError('Expected nonempty position (T,3) and quaternion (T,4)')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=1) < 1e-12):
        raise ValueError('Poses must be finite and quaternions nonzero')
    if not np.isfinite(fps) or fps <= 0 or not isinstance(half_window, int) or half_window < 1:
        raise ValueError('fps must be positive; half_window must be a positive integer')
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    t = np.arange(len(p))
    lo, hi = np.maximum(t-half_window, 0), np.minimum(t+half_window, len(p)-1)
    dt = np.maximum(hi - lo, 1) / fps
    linear_w = (p[hi] - p[lo]) / dt[:, None]
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    angular_w = (r[hi] * r[lo].inv()).as_rotvec() / dt[:, None]
    w, x, y, z = q.T
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack((c*linear_w[:, 0]+s*linear_w[:, 1],
                            -s*linear_w[:, 0]+c*linear_w[:, 1],
                            angular_w[:, 2])).astype(np.float32)
