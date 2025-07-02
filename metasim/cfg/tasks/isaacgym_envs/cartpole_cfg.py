from __future__ import annotations

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.cartpole_cfg import CartpoleCfg as CartpoleRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class CartpoleCfg(BaseTaskCfg):
    name = "isaacgym_envs:Cartpole"
    episode_length = 500
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION  # Using closest available type

    # Task parameters
    reset_dist = 3.0
    max_push_effort = 400.0

    # Robot configuration
    robot: CartpoleRobotCfg = CartpoleRobotCfg()

    # No additional objects needed
    objects: list[RigidObjCfg] = []

    # Control configuration
    control: ControlCfg = ControlCfg(
        action_scale=400.0,  # Max push effort from IsaacGymEnvs
        action_offset=False,
        torque_limit_scale=1.0,
    )

    checker = EmptyChecker()

    # Observation space: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    observation_space = {"shape": [4]}
    action_space = {"shape": [1]}  # Single action for cart force

    randomize = {
        "robot": {
            "cartpole": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.1,  # 0.2 * 0.5 = 0.1
                    "high": 0.1,
                },
                "joint_qvel": {
                    "type": "uniform",
                    "low": -0.25,  # 0.5 * 0.5 = 0.25
                    "high": 0.25,
                },
            }
        }
    }
