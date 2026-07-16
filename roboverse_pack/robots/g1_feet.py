"""Go1-feet-only robot configuration for MuJoCo-Playground / MetaSim.

• 12 position actuators (3 per leg).
• Default PID gains and force limits are copied verbatim from the XML:
    - Global default:   kp = 35, kd = 0.5, force ±23.7 N·m
    - Knees override:   force ±35.55 N·m
"""

from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Go1FeetCfg(RobotCfg):
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
    #
    # ``BaseActuatorCfg`` exposes the PD gains as ``stiffness`` (position gain, kp) and
    # ``damping`` (velocity gain, kd), and the torque cap as ``effort_limit_sim`` (the
    # per-backend force limit). The XML's kp/kd/force values map onto those fields.
    _HIP_KP = 35.0  # -> stiffness
    _HIP_KD = 0.5  # -> damping
    _HIP_FMAX = 23.7  # -> effort_limit_sim
    _KNEE_FMAX = 35.55  # -> effort_limit_sim (knee override)

    actuators: dict[str, BaseActuatorCfg] = {
        # Front-Right (FR)
        "FR_hip": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "FR_thigh": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "FR_calf": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_KNEE_FMAX),
        # Front-Left (FL)
        "FL_hip": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "FL_thigh": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "FL_calf": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_KNEE_FMAX),
        # Rear-Right (RR)
        "RR_hip": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "RR_thigh": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "RR_calf": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_KNEE_FMAX),
        # Rear-Left (RL)
        "RL_hip": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "RL_thigh": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_HIP_FMAX),
        "RL_calf": BaseActuatorCfg(stiffness=_HIP_KP, damping=_HIP_KD, effort_limit_sim=_KNEE_FMAX),
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
