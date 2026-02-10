from __future__ import annotations

import numpy as np


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate for wxyz."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz)."""
    vq = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    return quat_mul_wxyz(quat_mul_wxyz(q, vq), quat_conj_wxyz(q))[1:]


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse(q) (wxyz)."""
    return quat_rotate_wxyz(quat_conj_wxyz(q), v)
