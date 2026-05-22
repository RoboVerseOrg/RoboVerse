"""RobotCfg for mjlab's Unitree Go1 quadruped. Same pattern as MjlabCartpoleCfg.

Lets velocity_flat_go1_v2 task work on Newton GPU backend (1000+ envs).
Joint set + default pose match beyondmimic / mjlab Go1 convention.
"""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass

_GO1_JOINTS = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# Default stance (matches velocity_go1_v2._GO1_DEFAULT_POSE_NP)
_GO1_DEFAULTS = {
    name: val
    for name, val in zip(
        _GO1_JOINTS,
        [0.0, 0.9, -1.8] * 4,
    )
}


@configclass
class MjlabGo1Cfg(RobotCfg):
    """Unitree Go1 from mjlab's go1.xml (12 hinge joints + floating base).

    Name aligns with velocity_go1_v2's SceneEntityCfg ("go1") so the same
    task code path works on mujoco scene-MJCF AND newton RobotCfg backends.
    """

    name: str = "go1"
    mjcf_path: str = "roboverse_data/robots/mjlab/asset_zoo/robots/unitree_go1/xmls/go1.xml"
    urdf_path: str = mjcf_path
    usd_path: str = mjcf_path

    num_joints: int = 12
    fix_base_link: bool = False  # Go1 is mobile, base floats
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        name: BaseActuatorCfg(
            stiffness=15.9 if "calf" not in name else 35.8,
            damping=1.0 if "calf" not in name else 2.3,
            effort_limit_sim=23.7 if "calf" not in name else 35.55,
            velocity_limit=20.0,
        )
        for name in _GO1_JOINTS
    }
    joint_limits: dict[str, tuple[float, float]] = {
        # hip ±50°, thigh -39° to 258°, calf -161° to -51° (mjlab go1.xml)
        **{j: (-0.863, 0.863) for j in _GO1_JOINTS if "hip" in j and "thigh" not in j},
        **{j: (-0.686, 4.501) for j in _GO1_JOINTS if "thigh" in j},
        **{j: (-2.818, -0.888) for j in _GO1_JOINTS if "calf" in j},
    }
    default_joint_positions: dict[str, float] = _GO1_DEFAULTS
    default_joint_velocities: dict[str, float] = {n: 0.0 for n in _GO1_JOINTS}
    control_type: dict[str, Literal["position", "effort"]] = {n: "position" for n in _GO1_JOINTS}
    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.42)
    default_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
