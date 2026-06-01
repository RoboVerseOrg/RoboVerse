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


# Default stance — MUST match mjlab INIT_STATE (go1_constants.py): thigh=0.9,
# calf=-1.8, and asymmetric hips R_hip=+0.1 / L_hip=-0.1. The earlier symmetric
# hip=0.0 here (Newton path never got the r7 ±0.1 fix) gave an unstable stance
# that, together with the wrong 0.42 init height, made go1 collapse under PD on
# Newton (couldn't even stand). FR/RR are the "R" legs, FL/RL the "L" legs.
def _go1_default(name: str) -> float:
    if "thigh" in name:
        return 0.9
    if "calf" in name:
        return -1.8
    if "hip" in name:
        return 0.1 if name[1] == "R" else -0.1
    return 0.0


_GO1_DEFAULTS = {name: _go1_default(name) for name in _GO1_JOINTS}


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

    # EXACT mjlab compiled actuator spec (from _mjlab_actuator_spec.json, dumped
    # from mjlab's compiled go1 MjModel). The Newton backend applies stiffness→
    # joint_target_ke, damping→joint_target_kd, ARMATURE→joint_armature,
    # effort_limit_sim→joint_effort_limit. The armature was previously unset on
    # the RobotCfg → Newton ran without mjlab's reflected inertia, so a trained
    # walker collapsed on the Newton path (and training learned a degenerate
    # gait). hip/thigh use the 5020-class actuator, calf the 7520_14-class.
    actuators: dict[str, BaseActuatorCfg] = {
        name: BaseActuatorCfg(
            stiffness=35.7643 if "calf" in name else 15.8952,
            damping=2.2768 if "calf" in name else 1.0119,
            armature=0.00906 if "calf" in name else 0.00403,
            effort_limit_sim=35.55 if "calf" in name else 23.7,
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
    default_pos: tuple[float, float, float] = (
        0.0,
        0.0,
        0.278,
    )  # mjlab INIT_STATE height (was 0.42 -> 14cm drop -> collapse)
    default_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
