from __future__ import annotations

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.ant_cfg import AntCfg as AntRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class AntIsaacGymCfg(BaseTaskCfg):
    name = "isaacgym_envs:AntIsaacGym"
    episode_length = 1000
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    initial_height = 0.55

    dof_vel_scale = 0.2
    contact_force_scale = 0.1
    power_scale = 1.0
    heading_weight = 0.5
    up_weight = 0.1
    actions_cost_scale = 0.005
    energy_cost_scale = 0.05
    joints_at_limit_cost_scale = 0.1
    death_cost = -2.0
    termination_height = 0.31

    robot: AntRobotCfg = AntRobotCfg()

    objects: list[RigidObjCfg] = []

    control: ControlCfg = ControlCfg(action_scale=15.0, action_offset=False, torque_limit_scale=1.0)

    checker = EmptyChecker()

    observation_space = {"shape": [60]}

    randomize = {
        "robot": {
            "ant": {
                "pos": {
                    "type": "gaussian",
                    "mean": [0.0, 0.0, 0.55],
                    "std": [0.0, 0.0, 0.0],
                },
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                },
                "joint_qvel": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                },
            }
        }
    }
