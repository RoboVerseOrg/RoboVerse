from __future__ import annotations

import logging

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class AllegroHandCfg(BaseTaskCfg):
    name = "isaacgym_envs:AllegroHand"
    episode_length = 600
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    object_type = "block"

    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250.0
    success_tolerance = 0.1
    fall_dist = 0.24
    fall_penalty = 0.0
    rot_eps = 0.1
    av_factor = 0.1
    max_consecutive_successes = 0

    use_relative_control = False
    actions_moving_average = 1.0
    dof_speed_scale = 20.0

    obs_type = "full_no_vel"

    reset_position_noise = 0.01
    reset_rotation_noise = 0.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0

    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    objects: list[RigidObjCfg] | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.objects is None:
            self.objects = [
                RigidObjCfg(
                    name="block",
                    usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/usd/cube_multicolor_allegro.usd",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    default_position=(0.0, -0.20000000298023224, 0.6600000023841858),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                RigidObjCfg(
                    name="goal",
                    usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/usd/cube_multicolor_allegro.usd",
                    mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
                    default_position=(-0.20000000298023224, -0.25999999046325684, 0.6399999856948853),
                    default_orientation=(1.0, 0.0, 0.0, 0.0),
                    physics=PhysicStateType.XFORM,
                ),
            ]

    observation_space = {"full_no_vel": 50, "full": 72, "full_state": 88}

    randomize = {
        "robot": {
            "allegro_hand": {
                "joint_qpos": {
                    "type": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                }
            }
        },
        "object": {
            "block": {
                "position": {
                    "x": [0.0, 0.0],
                    "y": [-0.21, -0.19],
                    "z": [0.56, 0.56],
                },
                "orientation": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                    "w": [-1.0, 1.0],
                },
            },
            "goal": {
                "orientation": {
                    "x": [-1.0, 1.0],
                    "y": [-1.0, 1.0],
                    "z": [-1.0, 1.0],
                    "w": [-1.0, 1.0],
                }
            },
        },
    }
