"""RobotCfg for mjlab's i2rt YAM 6-DOF arm."""

from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass

_YAM_ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# yam.xml also declares the two prismatic gripper-finger joints (left/right
# finger, coupled by an MJCF equality). The v2 obs/joint SceneEntityCfg
# references all 8 joints, so the Newton-imported joint set must include the
# fingers too (otherwise ``left_finger`` resolves to no joint on Newton).
_YAM_FINGER_JOINTS = ("left_finger", "right_finger")
_YAM_JOINTS = _YAM_ARM_JOINTS + _YAM_FINGER_JOINTS


@configclass
class MjlabYamCfg(RobotCfg):
    """YAM 6-DOF arm + 2-finger gripper. name='yam' matches v2 task SceneEntityCfg."""

    name: str = "yam"
    mjcf_path: str = "roboverse_data/robots/mjlab/asset_zoo/robots/i2rt_yam/xmls/yam.xml"
    urdf_path: str = mjcf_path
    usd_path: str = mjcf_path

    num_joints: int = 8
    fix_base_link: bool = True
    enabled_gravity: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        **{
            n: BaseActuatorCfg(
                stiffness=20.0 if "1" in n or "2" in n or "3" in n else 10.0,
                damping=2.0,
                effort_limit_sim=30.0,
                velocity_limit=10.0,
            )
            for n in _YAM_ARM_JOINTS
        },
        **{
            n: BaseActuatorCfg(
                stiffness=10.0,
                damping=1.0,
                effort_limit_sim=30.0,
                velocity_limit=10.0,
            )
            for n in _YAM_FINGER_JOINTS
        },
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "joint1": (-2.61799, 3.05433),
        "joint2": (0.0, 3.66519),
        "joint3": (0.0, 3.66519),
        "joint4": (-1.5708, 1.5708),
        "joint5": (-1.5708, 1.5708),
        "joint6": (-2.0944, 2.0944),
        "left_finger": (-0.00205, 0.037524),
        "right_finger": (-0.037524, 0.00205),
    }
    default_joint_positions: dict[str, float] = {n: 0.0 for n in _YAM_JOINTS}
    default_joint_velocities: dict[str, float] = {n: 0.0 for n in _YAM_JOINTS}
    control_type: dict[str, Literal["position", "effort"]] = {n: "position" for n in _YAM_JOINTS}
    default_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
