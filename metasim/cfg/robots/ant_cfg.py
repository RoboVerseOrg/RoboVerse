from __future__ import annotations

from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class AntCfg(BaseRobotCfg):
    name: str = "ant"
    num_joints: int = 8
    fix_base_link: bool = False
    scale: list[float] = [1.0, 1.0, 1.0]
    mjcf_path: str = "roboverse_data/robots/ant/mjcf/nv_ant.xml"
    urdf_path: str = "roboverse_data/robots/ant/urdf/ant.urdf"
    usd_path: str = "roboverse_data/robots/ant/urdf/ant.urdf"
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False
    isaacgym_read_mjcf: bool = True
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = None

    joint_limits: dict[str, tuple[float, float]] = None

    default_joint_positions: dict[str, float] = None

    control_type: dict[str, Literal["position", "effort"]] = None

    def __post_init__(self):
        super().__post_init__()

        if self.joint_limits is None:
            # Joint limits from nv_ant.xml, converted from degrees to radians
            deg_to_rad = 3.14159 / 180.0
            self.joint_limits = {
                "hip_1": (-40 * deg_to_rad, 40 * deg_to_rad),  # -0.698, 0.698 rad
                "ankle_1": (30 * deg_to_rad, 100 * deg_to_rad),  # 0.524, 1.745 rad
                "hip_2": (-40 * deg_to_rad, 40 * deg_to_rad),  # -0.698, 0.698 rad
                "ankle_2": (-100 * deg_to_rad, -30 * deg_to_rad),  # -1.745, -0.524 rad
                "hip_3": (-40 * deg_to_rad, 40 * deg_to_rad),  # -0.698, 0.698 rad
                "ankle_3": (-100 * deg_to_rad, -30 * deg_to_rad),  # -1.745, -0.524 rad
                "hip_4": (-40 * deg_to_rad, 40 * deg_to_rad),  # -0.698, 0.698 rad
                "ankle_4": (30 * deg_to_rad, 100 * deg_to_rad),  # 0.524, 1.745 rad
            }

        if self.default_joint_positions is None:
            # Set initial positions based on joint limits (following IsaacGymEnvs logic)
            deg_to_rad = 3.14159 / 180.0
            self.default_joint_positions = {
                "hip_1": 0.0,  # limit: -40 to 40, use 0
                "ankle_1": 30 * deg_to_rad,  # limit: 30 to 100, use lower limit
                "hip_2": 0.0,  # limit: -40 to 40, use 0
                "ankle_2": -30 * deg_to_rad,  # limit: -100 to -30, use upper limit
                "hip_3": 0.0,  # limit: -40 to 40, use 0
                "ankle_3": -30 * deg_to_rad,  # limit: -100 to -30, use upper limit
                "hip_4": 0.0,  # limit: -40 to 40, use 0
                "ankle_4": 30 * deg_to_rad,  # limit: 30 to 100, use lower limit
            }

        if self.actuators is None:
            # From MJCF: gear="15" for all motors, so motor_effort = 15
            # For direct torque control, set stiffness=1.0, damping=0.0
            # This makes the PD controller act as direct torque (torque = 1.0 * action)
            self.actuators = {
                "hip_1": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "ankle_1": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "hip_2": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "ankle_2": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "hip_3": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "ankle_3": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "hip_4": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
                "ankle_4": BaseActuatorCfg(velocity_limit=30.0, torque_limit=15.0, stiffness=1.0, damping=0.0),
            }

        if self.control_type is None:
            self.control_type = {
                "hip_1": "effort",
                "ankle_1": "effort",
                "hip_2": "effort",
                "ankle_2": "effort",
                "hip_3": "effort",
                "ankle_3": "effort",
                "hip_4": "effort",
                "ankle_4": "effort",
            }
