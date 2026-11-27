"""RobotCfg for mjlab's i2rt YAM 6-DOF arm."""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass

_YAM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


_FINGER_LIMITS = {"left_finger": (-0.00205, 0.037524), "right_finger": (-0.037524, 0.00205)}


@configclass
class MjlabYamCfg(RobotCfg):
    """mjlab's bare copy of the YAM arm for the mjlab lift-cube ports; ``name='yam'`` matches the tasks' entities.

    The six arm joints are driven from the config (the asset ships no actuators). The two gripper joints are
    in the joint vector like every joint of the asset but are *passive*: no actuator and no gains, coupled by
    the asset's equality constraint, so the action/observation width is 8 on every backend and a target in a
    finger slot is ignored. For a controllable gripper use ``YamCfg``.
    """

    name: str = "yam"
    mjcf_path: str = "roboverse_data/robots/mjlab/asset_zoo/robots/i2rt_yam/xmls/yam.xml"
    urdf_path: str = mjcf_path
    usd_path: str = mjcf_path

    num_joints: int = 8
    fix_base_link: bool = True
    enabled_gravity: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        n: BaseActuatorCfg(
            stiffness=20.0 if "1" in n or "2" in n or "3" in n else 10.0,
            damping=2.0,
            effort_limit_sim=30.0,
            velocity_limit=10.0,
        )
        for n in _YAM_JOINTS
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-2.61799, 3.05433),
        "joint2": (0.0, 3.66519),
        "joint3": (0.0, 3.66519),
        "joint4": (-1.5708, 1.5708),
        "joint5": (-1.5708, 1.5708),
        "joint6": (-2.0944, 2.0944),
        **_FINGER_LIMITS,
    }
    default_joint_positions: dict[str, float] = {n: 0.0 for n in (*_YAM_JOINTS, *_FINGER_LIMITS)}
    default_joint_velocities: dict[str, float] = {n: 0.0 for n in (*_YAM_JOINTS, *_FINGER_LIMITS)}
    control_type: dict[str, Literal["position", "effort"]] = {n: "position" for n in _YAM_JOINTS}
    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
