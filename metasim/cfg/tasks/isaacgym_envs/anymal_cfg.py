from __future__ import annotations

import logging

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.anymal_cfg import AnymalCfg as AnymalRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class AnymalCfg(BaseTaskCfg):
    name = "isaacgym_envs:Anymal"
    episode_length = 1000
    traj_filepath = None
    task_type = TaskType.LOCOMOTION

    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.5

    lin_vel_xy_reward_scale = 1.0
    ang_vel_z_reward_scale = 0.5
    torque_reward_scale = -0.000025

    command_x_range = [-2.0, 2.0]
    command_y_range = [-1.0, 1.0]
    command_yaw_range = [-1.0, 1.0]

    base_contact_force_threshold = 1.0
    knee_contact_force_threshold = 1.0

    robot: AnymalRobotCfg = AnymalRobotCfg()

    objects: list[RigidObjCfg] = []

    observation_space = {"shape": [48]}

    randomize = {
        "robot": {
            "anymal": {
                "joint_qpos": {"type": "scaling", "low": 0.5, "high": 1.5, "base": "default"},
                "joint_qvel": {
                    "type": "uniform",
                    "low": -0.1,
                    "high": 0.1,
                },
            }
        }
    }
