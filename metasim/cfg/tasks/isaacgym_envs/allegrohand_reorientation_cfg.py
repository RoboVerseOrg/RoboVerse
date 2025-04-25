import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass
from metasim.utils.math import quat_invert, quat_mul

from ..base_task_cfg import BaseTaskCfg


@configclass
class AllegroHandReorientationCfg(BaseTaskCfg):
    episode_length = 600
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    objects = [
        RigidObjCfg(
            name="block",
            usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            default_position=(0.0, -0.2, 0.56),
            default_orientation=(1.0, 0.0, 0.0, 0.0),
        ),
        RigidObjCfg(
            name="goal",
            usd_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            mjcf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            urdf_path="roboverse_data/assets/isaacgymenvs/block_allegrohand_multicolor/urdf/cube_multicolor_allegro.urdf",
            default_position=(0.0, 0.0, 0.92),
            default_orientation=torch.nn.functional.normalize(torch.rand(4), p=2, dim=0),
            physics=PhysicStateType.XFORM,
        ),
    ]

    observation_space = {
        "robot": {
            "joint_qpos": {
                "low": float("-inf"),
                "high": float("inf"),
                "shape": (16,),
            },
        },
        "objects": {
            "block": {
                "pos": {
                    "low": float("-inf"),
                    "high": float("inf"),
                    "shape": (3,),
                },
                "rot": {
                    "low": float("-inf"),
                    "high": float("inf"),
                    "shape": (4,),
                },
            },
        },
    }

    randomize = {
        "object": {
            "block": {
                "orientation": {
                    "x": [-1.0, +1.0],
                    "y": [-1.0, +1.0],
                    "z": [-1.0, +1.0],
                    "w": [-1.0, +1.0],
                },
            },
            "goal": {
                "orientation": {
                    "x": [-1.0, +1.0],
                    "y": [-1.0, +1.0],
                    "z": [-1.0, +1.0],
                    "w": [-1.0, +1.0],
                },
            },
        },
    }

    ignore_z_rotation = True

    def reward_fn(self, states, actions):
        # Reward constants
        dist_reward_scale = -10.0
        rot_reward_scale = 1.0
        rot_eps = 0.1
        action_penalty_scale = -0.0002
        success_tolerance = 0.1
        reach_goal_bonus = 250.0
        fall_dist = 0.24
        fall_penalty = 0.0
        success_tolerance = 0.1

        rewards = []
        if self.ignore_z_rotation:
            success_tolerance = 2.0 * success_tolerance
        for i, env_state in enumerate(states):
            object_state = env_state["objects"]["block"]
            goal_state = env_state["objects"]["goal"]
            action = list(actions[i]["dof_pos_target"].values())

            object_pos = object_state["pos"]
            object_rot = object_state["rot"]

            goal_pos = goal_state["pos"]
            goal_rot = goal_state["rot"]

            goal_dist = torch.norm(object_pos - goal_pos, p=2)

            quat_diff = quat_mul(object_rot, quat_invert(goal_rot))
            rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[0:3], p=2, dim=-1), max=1.0))

            dist_rew = goal_dist * dist_reward_scale
            rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

            action_penalty = torch.sum(torch.tensor(action) ** 2, dim=-1)

            reward = -dist_rew + rot_rew - action_penalty * action_penalty_scale

            if goal_dist < success_tolerance and rot_dist < success_tolerance:
                reward += reach_goal_bonus

            if goal_dist >= fall_dist:
                reward += fall_penalty

            rewards.append(reward)

        return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        terminations = []
        for env_state in states:
            robot_state = env_state["robots"]["allegro_hand"]
            block_state = env_state["objects"]["block"]
            robot_pos = robot_state["pos"]
            block_pos = block_state["pos"]

            fall_dist = 0.24

            goal_dist = torch.norm(block_pos - robot_pos, p=2)
            terminate = goal_dist >= fall_dist
            terminations.append(terminate)

        return torch.tensor(terminations) if terminations else torch.tensor([False])
