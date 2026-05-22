"""RobotCfg for mjlab's Unitree G1 humanoid — exact mjlab actuator parity.

Gains are computed from Unitree motor specs (rotor inertias + gear ratios) the
same way mjlab does (`g1_constants.py`), not hand-tuned. This is required for
mjlab parity — flat hand-tuned gains do not reproduce mjlab training curves.

Source: mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py
"""

from __future__ import annotations

import math
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass

# ---------------------------------------------------------------------------
# Motor specs from Unitree (rotor inertia kg·m²; gear ratios dimensionless).
# Reflected inertia of a two-stage planetary gearbox: I = R0*(G1*G2)² + R1*G2² + R2.
# ---------------------------------------------------------------------------


def _reflected_inertia(rotor: tuple[float, float, float], gears: tuple[float, float, float]) -> float:
    """Two-stage planetary reflected inertia (mjlab.utils.actuator)."""
    assert gears[0] == 1
    r1 = rotor[0] * (gears[1] * gears[2]) ** 2
    r2 = rotor[1] * gears[2] ** 2
    r3 = rotor[2]
    return r1 + r2 + r3


# Motor 5020 (shoulder pitch/roll/yaw, elbow, wrist_roll)
_ROTOR_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
_GEARS_5020 = (1.0, 1 + (46 / 18), 1 + (56 / 16))  # (1, 3.556, 4.5)
_ARMATURE_5020 = _reflected_inertia(_ROTOR_5020, _GEARS_5020)

# Motor 7520-14 (hip_pitch, hip_yaw, waist_yaw)
_ROTOR_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
_GEARS_7520_14 = (1.0, 4.5, 1 + (48 / 22))
_ARMATURE_7520_14 = _reflected_inertia(_ROTOR_7520_14, _GEARS_7520_14)

# Motor 7520-22 (hip_roll, knee)
_ROTOR_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
_GEARS_7520_22 = (1.0, 4.5, 5.0)
_ARMATURE_7520_22 = _reflected_inertia(_ROTOR_7520_22, _GEARS_7520_22)

# Motor 4010 (wrist_pitch, wrist_yaw)
_ROTOR_4010 = (0.068e-4, 0.0, 0.0)
_GEARS_4010 = (1.0, 5.0, 5.0)
_ARMATURE_4010 = _reflected_inertia(_ROTOR_4010, _GEARS_4010)

# 2nd-order PD: stiffness = m·ω², damping = 2·ζ·m·ω with ω = 10·2π and ζ = 2.0 (over-damped)
_OMEGA = 10.0 * 2.0 * math.pi
_ZETA = 2.0
_K_FROM_M = lambda m: m * _OMEGA**2  # noqa: E731
_D_FROM_M = lambda m: 2.0 * _ZETA * m * _OMEGA  # noqa: E731

# Effort limits (Nm) and velocity limits (rad/s) per motor type — Unitree datasheet
_EFFORT_5020, _VEL_5020 = 25.0, 37.0
_EFFORT_7520_14, _VEL_7520_14 = 88.0, 32.0
_EFFORT_7520_22, _VEL_7520_22 = 139.0, 20.0
_EFFORT_4010, _VEL_4010 = 5.0, 22.0

# Computed effective stiffness/damping per joint group
K_5020, D_5020 = _K_FROM_M(_ARMATURE_5020), _D_FROM_M(_ARMATURE_5020)
K_7520_14, D_7520_14 = _K_FROM_M(_ARMATURE_7520_14), _D_FROM_M(_ARMATURE_7520_14)
K_7520_22, D_7520_22 = _K_FROM_M(_ARMATURE_7520_22), _D_FROM_M(_ARMATURE_7520_22)
K_4010, D_4010 = _K_FROM_M(_ARMATURE_4010), _D_FROM_M(_ARMATURE_4010)
# Waist + ankle 4-bar linkage — use 2× armature/effort (parallel dual 5020)
K_WAIST, D_WAIST = 2 * K_5020, 2 * D_5020
EFFORT_WAIST = 2 * _EFFORT_5020
K_ANKLE, D_ANKLE = 2 * K_5020, 2 * D_5020
EFFORT_ANKLE = 2 * _EFFORT_5020


_G1_JOINTS = (
    # left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _build_actuators() -> dict[str, BaseActuatorCfg]:
    """Map each joint to its exact mjlab motor spec."""
    cfgs: dict[str, BaseActuatorCfg] = {}
    for n in _G1_JOINTS:
        if "hip_roll" in n or "knee" in n:
            cfgs[n] = BaseActuatorCfg(
                stiffness=K_7520_22,
                damping=D_7520_22,
                effort_limit_sim=_EFFORT_7520_22,
                velocity_limit=_VEL_7520_22,
                armature=_ARMATURE_7520_22,
            )
        elif "hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint":
            cfgs[n] = BaseActuatorCfg(
                stiffness=K_7520_14,
                damping=D_7520_14,
                effort_limit_sim=_EFFORT_7520_14,
                velocity_limit=_VEL_7520_14,
                armature=_ARMATURE_7520_14,
            )
        elif "ankle" in n:
            cfgs[n] = BaseActuatorCfg(
                stiffness=K_ANKLE,
                damping=D_ANKLE,
                effort_limit_sim=EFFORT_ANKLE,
                velocity_limit=_VEL_5020,
                armature=2 * _ARMATURE_5020,
            )
        elif n in ("waist_pitch_joint", "waist_roll_joint"):
            cfgs[n] = BaseActuatorCfg(
                stiffness=K_WAIST,
                damping=D_WAIST,
                effort_limit_sim=EFFORT_WAIST,
                velocity_limit=_VEL_5020,
                armature=2 * _ARMATURE_5020,
            )
        elif "wrist_pitch" in n or "wrist_yaw" in n:
            cfgs[n] = BaseActuatorCfg(
                stiffness=K_4010,
                damping=D_4010,
                effort_limit_sim=_EFFORT_4010,
                velocity_limit=_VEL_4010,
                armature=_ARMATURE_4010,
            )
        else:  # shoulder_pitch/roll/yaw, elbow, wrist_roll → 5020
            cfgs[n] = BaseActuatorCfg(
                stiffness=K_5020,
                damping=D_5020,
                effort_limit_sim=_EFFORT_5020,
                velocity_limit=_VEL_5020,
                armature=_ARMATURE_5020,
            )
    return cfgs


# KNEES_BENT_KEYFRAME from mjlab g1_constants.py — this is the *training* init pose.
# HOME_KEYFRAME (hip_pitch=-0.1, knee=0.3) is the show-pose but is dynamically
# unstable under PD (pelvis pitches forward 39° in 20 zero-action steps).
_G1_DEFAULTS_KNEES_BENT = {
    # legs
    **{n: -0.312 for n in _G1_JOINTS if "hip_pitch" in n},
    **{n: 0.669 for n in _G1_JOINTS if "knee" in n},
    **{n: -0.363 for n in _G1_JOINTS if "ankle_pitch" in n},
    # everything else 0 except arms
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms — mjlab uses shoulder_pitch=0.2, shoulder_roll=±0.2, elbow=0.6
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


_G1_LIMITS = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "right_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "right_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}


# Action scale per joint = 0.25 * effort_limit / stiffness (mjlab convention; gives
# action range that maps to ~25% of motor capacity). Exported for use by task code.
G1_ACTION_SCALE: dict[str, float] = {}
for _n in _G1_JOINTS:
    if "hip_roll" in _n or "knee" in _n:
        G1_ACTION_SCALE[_n] = 0.25 * _EFFORT_7520_22 / K_7520_22
    elif "hip_pitch" in _n or "hip_yaw" in _n or _n == "waist_yaw_joint":
        G1_ACTION_SCALE[_n] = 0.25 * _EFFORT_7520_14 / K_7520_14
    elif "ankle" in _n:
        G1_ACTION_SCALE[_n] = 0.25 * EFFORT_ANKLE / K_ANKLE
    elif _n in ("waist_pitch_joint", "waist_roll_joint"):
        G1_ACTION_SCALE[_n] = 0.25 * EFFORT_WAIST / K_WAIST
    elif "wrist_pitch" in _n or "wrist_yaw" in _n:
        G1_ACTION_SCALE[_n] = 0.25 * _EFFORT_4010 / K_4010
    else:
        G1_ACTION_SCALE[_n] = 0.25 * _EFFORT_5020 / K_5020


@configclass
class MjlabG1Cfg(RobotCfg):
    """G1 humanoid — exact mjlab actuator + KNEES_BENT init pose. name='g1'."""

    name: str = "g1"
    mjcf_path: str = "roboverse_data/robots/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
    urdf_path: str = mjcf_path
    usd_path: str = mjcf_path

    num_joints: int = 29
    fix_base_link: bool = False
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    actuators: dict[str, BaseActuatorCfg] = _build_actuators()
    joint_limits: dict[str, tuple[float, float]] = _G1_LIMITS
    default_joint_positions: dict[str, float] = _G1_DEFAULTS_KNEES_BENT
    default_joint_velocities: dict[str, float] = {n: 0.0 for n in _G1_JOINTS}
    control_type: dict[str, Literal["position", "effort"]] = {n: "position" for n in _G1_JOINTS}
    # KNEES_BENT_KEYFRAME pelvis z = 0.76 (mjlab native)
    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.76)
    default_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
