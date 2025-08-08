"""
Go1-feet-only robot configuration for MuJoCo-Playground / MetaSim.

• 12 position actuators (3 per leg).
• Default PID gains and force limits are copied verbatim from the XML:
    - Global default:   kp = 35, kd = 0.5, force ±23.7 N·m
    - Knees override:   force ±35.55 N·m
"""

from __future__ import annotations

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class Go1FeetCfg(BaseRobotCfg):
    # ------------------------------------------------------------------
    # Basic meta-data
    # ------------------------------------------------------------------
    name: str = "go1_feet"
    num_joints: int = 12  # 4 legs × 3 DoF
    mjcf_path: str = "roboverse_data/robots/go1/go1_feet.xml"

    # ------------------------------------------------------------------
    # Simulation flags
    # ------------------------------------------------------------------
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    collapse_fixed_joints: bool = True

    # ------------------------------------------------------------------
    # Actuators
    # ------------------------------------------------------------------
    # Global XML defaults: kp=35, kd=0.5, force= ±23.7
    # Knee class overrides: force= ±35.55
    _HIP_KP = 35.0
    _HIP_KD = 0.5
    _HIP_FMAX = 23.7
    _KNEE_FMAX = 35.55

    actuators: dict[str, BaseActuatorCfg] = {
        # Front-Right (FR)
        "FR_hip": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "FR_thigh": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "FR_calf": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_KNEE_FMAX),
        # Front-Left (FL)
        "FL_hip": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "FL_thigh": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "FL_calf": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_KNEE_FMAX),
        # Rear-Right (RR)
        "RR_hip": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "RR_thigh": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "RR_calf": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_KNEE_FMAX),
        # Rear-Left (RL)
        "RL_hip": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "RL_thigh": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_HIP_FMAX),
        "RL_calf": BaseActuatorCfg(kp=_HIP_KP, kd=_HIP_KD, force_limit=_KNEE_FMAX),
    }

    # ------------------------------------------------------------------
    # Joint limits (rad) – copied from the XML <default> blocks
    # ------------------------------------------------------------------
    joint_limits: dict[str, tuple[float, float]] = {
        # Abduction (hip-yaw) joints
        "FR_hip": (-0.863, 0.863),
        "FL_hip": (-0.863, 0.863),
        "RR_hip": (-0.863, 0.863),
        "RL_hip": (-0.863, 0.863),
        # Hip-pitch joints
        "FR_thigh": (-0.686, 4.501),
        "FL_thigh": (-0.686, 4.501),
        "RR_thigh": (-0.686, 4.501),
        "RL_thigh": (-0.686, 4.501),
        # Knees (negative range = flexion only)
        "FR_calf": (-2.818, -0.888),
        "FL_calf": (-2.818, -0.888),
        "RR_calf": (-2.818, -0.888),
        "RL_calf": (-2.818, -0.888),
    }
