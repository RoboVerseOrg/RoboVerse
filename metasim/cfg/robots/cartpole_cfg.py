from __future__ import annotations

from typing import Literal

from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg
from metasim.utils import configclass


@configclass
class CartpoleCfg(BaseRobotCfg):
    name = "cartpole"
    urdf_path = "roboverse_data/robots/cartpole/urdf/cartpole.urdf"
    mjcf_path = "roboverse_data/robots/cartpole/urdf/cartpole.urdf"
    usd_path = "roboverse_data/robots/cartpole/urdf/cartpole.urdf"
    num_joints = 2  # slider joint for cart, revolute joint for pole

    # Physical properties
    fix_base_link = True  # The base/world is fixed

    actuators: dict[str, BaseActuatorCfg] = None
    joint_limits: dict[str, tuple[float, float]] = None
    default_joint_positions: dict[str, float] = None
    control_type: dict[str, Literal["position", "effort"]] = None
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.5)

    def __post_init__(self):
        super().__post_init__()

        if self.joint_limits is None:
            self.joint_limits = {
                "slider_to_cart": (-4.0, 4.0),  # From URDF: lower="-4" upper="4"
                "cart_to_pole": (-3.14159, 3.14159),  # Continuous joint, use +/- pi
            }

        if self.default_joint_positions is None:
            self.default_joint_positions = {
                "slider_to_cart": 0.0,  # cart position
                "cart_to_pole": 0.0,  # pole angle
            }

        if self.actuators is None:
            # Only the cart is actuated with force control
            # The pole joint is passive (not actuated)
            self.actuators = {
                "slider_to_cart": BaseActuatorCfg(
                    velocity_limit=100.0,  # From URDF
                    torque_limit=400.0,  # Max force for cart (from IsaacGymEnvs)
                    stiffness=0.0,  # Direct force control
                    damping=0.0,  # No damping for direct control
                ),
                "cart_to_pole": BaseActuatorCfg(
                    velocity_limit=8.0,  # From URDF
                    torque_limit=0.0,  # Passive joint - no actuation
                    stiffness=0.0,
                    damping=0.0,
                    fully_actuated=False,  # This is a passive joint
                ),
            }

        if self.control_type is None:
            self.control_type = {
                "slider_to_cart": "effort",  # Force control
                "cart_to_pole": "effort",  # Even though passive, needs to be specified
            }
