import gym
import numpy as np
import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class AllegroHandCfg(BaseTaskCfg):
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

    observation_space = gym.spaces.Dict({
        "joint_qpos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
    })

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

            quat_diff = self._quat_mul(object_rot, self._quat_conjugate(goal_rot))
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

    def _quat_mul(self, a, b):
        """Multiply two quaternions."""
        assert a.shape == b.shape
        x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
        x2, y2, z2, w2 = b[0], b[1], b[2], b[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.tensor([x, y, z, w])

    def _quat_conjugate(self, q):
        """Conjugate of quaternion."""
        return torch.tensor([-q[0], -q[1], -q[2], q[3]])

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
