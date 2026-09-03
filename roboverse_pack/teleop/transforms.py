from __future__ import annotations

"""Fixed frame transforms for teleoperation device profiles."""

import math
from dataclasses import dataclass
from typing import TypeAlias

Matrix3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

IDENTITY_ROT: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# Maintained openarm-teleop AVP frame profiles, ported from BiDexBench — a RoboVerse-internal
# predecessor project, not a third party. No upstream attribution is owed for this code.
AVP_R_VR_TO_ROBOT: Matrix3 = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)
AVP_R_LEFT_ARM: Matrix3 = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)
AVP_R_RIGHT_ARM: Matrix3 = (
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class TeleopTransformProfile:
    """Fixed transform profile applied before runtime calibration."""

    name: str
    description: str
    position_rot: Matrix3
    left_rot_post: Matrix3
    right_rot_post: Matrix3


PROFILES: dict[str, TeleopTransformProfile] = {
    "identity": TeleopTransformProfile(
        name="identity",
        description="No fixed transform. Only runtime calibration is applied downstream.",
        position_rot=IDENTITY_ROT,
        left_rot_post=IDENTITY_ROT,
        right_rot_post=IDENTITY_ROT,
    ),
    "avp_wrist_openarm_newton": TeleopTransformProfile(
        name="avp_wrist_openarm_newton",
        description=(
            "Vision Pro wrist-tracking profile for Newton/OpenArm collection. "
            "Newton uses sim-world XYZ directly, but the AVP lateral axis still needs "
            "a mirror so operator left/right matches the benchmark scene."
        ),
        position_rot=(
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        left_rot_post=AVP_R_LEFT_ARM,
        right_rot_post=AVP_R_RIGHT_ARM,
    ),
    "avp_wrist_openarm_mujoco": TeleopTransformProfile(
        name="avp_wrist_openarm_mujoco",
        description=(
            "Vision Pro wrist-tracking profile for MuJoCo/OpenArm collection. "
            "MuJoCo already matches the operator lateral axis, so it should not apply "
            "the Newton-specific X mirror."
        ),
        position_rot=IDENTITY_ROT,
        left_rot_post=AVP_R_LEFT_ARM,
        right_rot_post=AVP_R_RIGHT_ARM,
    ),
    "quest_xrt_controller_openarm": TeleopTransformProfile(
        name="quest_xrt_controller_openarm",
        description=(
            "Quest XRoboToolkit controller profile for OpenArm collection. "
            "It mirrors the lateral axis before downstream runtime calibration."
        ),
        position_rot=(
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        left_rot_post=IDENTITY_ROT,
        right_rot_post=IDENTITY_ROT,
    ),
    "avp_wrist_openarm_legacy": TeleopTransformProfile(
        name="avp_wrist_openarm_legacy",
        description="Legacy AVP wrist-tracking profile from the original real-robot openarm-teleop repo.",
        position_rot=AVP_R_VR_TO_ROBOT,
        left_rot_post=AVP_R_LEFT_ARM,
        right_rot_post=AVP_R_RIGHT_ARM,
    ),
}


def profile_choices() -> list[str]:
    """Return available transform profile names."""
    return sorted(PROFILES.keys())


def get_profile(name: str) -> TeleopTransformProfile:
    """Resolve a named teleoperation transform profile."""
    try:
        return PROFILES[str(name)]
    except KeyError as exc:
        choices = ", ".join(profile_choices())
        raise ValueError(f"unknown teleop transform profile '{name}', choose from: {choices}") from exc


def default_avp_transform_profile(simulator: str) -> str:
    """Return the default AVP wrist profile for a simulator backend."""
    if str(simulator).strip().lower() in {"mujoco", "isaaclab"}:
        return "avp_wrist_openarm_mujoco"
    return "avp_wrist_openarm_newton"


def _matvec3(rot: Matrix3, vec: list[float]) -> list[float]:
    return [
        rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
        rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
        rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2],
    ]


def _matmul3(a: Matrix3, b: Matrix3) -> Matrix3:
    rows: list[tuple[float, float, float]] = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
        rows.append((row[0], row[1], row[2]))
    return (rows[0], rows[1], rows[2])


def quat_xyzw_to_rotmat(quat: list[float]) -> Matrix3:
    """Convert an XYZW quaternion to a 3x3 rotation matrix."""
    x, y, z, w = [float(v) for v in quat]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def rotmat_to_quat_xyzw(rot: Matrix3) -> list[float]:
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-8:
        return [0.0, 0.0, 0.0, 1.0]
    inv = 1.0 / norm
    return [x * inv, y * inv, z * inv, w * inv]


def apply_profile_to_pose(
    profile: TeleopTransformProfile | str,
    hand_key: str,
    pose: list[float] | tuple[float, ...],
) -> list[float]:
    """Apply a fixed transform profile to a hand pose in XYZ+XYZW form."""
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    if hand_key not in ("left", "right"):
        raise ValueError(f"hand_key must be 'left' or 'right', got {hand_key!r}")
    pose_values = [float(value) for value in pose]
    if len(pose_values) != 7:
        raise ValueError(f"pose must have 7 values, got {len(pose_values)}")
    pos = _matvec3(resolved.position_rot, pose_values[:3])
    rot = quat_xyzw_to_rotmat(pose_values[3:7])
    rot = _matmul3(resolved.position_rot, rot)
    rot = _matmul3(rot, resolved.left_rot_post if hand_key == "left" else resolved.right_rot_post)
    quat = rotmat_to_quat_xyzw(rot)
    return pos + quat


__all__ = [
    "AVP_R_LEFT_ARM",
    "AVP_R_RIGHT_ARM",
    "AVP_R_VR_TO_ROBOT",
    "IDENTITY_ROT",
    "PROFILES",
    "TeleopTransformProfile",
    "apply_profile_to_pose",
    "default_avp_transform_profile",
    "get_profile",
    "profile_choices",
    "quat_xyzw_to_rotmat",
    "rotmat_to_quat_xyzw",
]
