import numpy as np


def euler_xyz_from_quat(quat: np.ndarray):
    """Convert a single (w, x, y, z) quaternion → Euler angles (roll, pitch, yaw) in radians.
    Input shape: (4,), returns a tuple of scalars.
    """
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quat must have shape (4,), got {q.shape}")
    q_w, q_x, q_y, q_z = q

    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.copysign(np.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else np.arcsin(sin_pitch)

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    # Wrap to [0, 2π)
    two_pi = 2.0 * np.pi
    return roll % two_pi, pitch % two_pi, yaw % two_pi


def get_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Return Euler (roll, pitch, yaw) in [-π, π] for a single quaternion.
    Input: (4,), Output: (3,)
    """
    r, p, y = euler_xyz_from_quat(quat)
    euler = np.array([r, p, y], dtype=float)
    euler[euler > np.pi] -= 2.0 * np.pi
    return euler
