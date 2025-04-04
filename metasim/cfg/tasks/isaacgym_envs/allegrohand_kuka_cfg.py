import torch

from metasim.constants import TaskType
from metasim.utils import configclass

from ..base_task_cfg import BaseTaskCfg


@configclass
class AllegroHandReorientationCfg(BaseTaskCfg):
    episode_length = 600
    objects = []
    traj_filepath = None
    task_type = TaskType.TABLETOP_MANIPULATION

    def reward_fn(self, states):
        # Reward constants
        dist_reward_scale = 10.0
        rot_reward_scale = 1.0
        rot_eps = 0.1
        action_penalty_scale = 0.01
        success_tolerance = 0.1
        reach_goal_bonus = 10.0
        fall_dist = 0.3
        fall_penalty = -20.0

        # Handle both multi-env (IsaacGym) and single-env (Mujoco) formats
        rewards = []
        for env_state in states:
            # Extract necessary state information from env_state
            allegro_hand_state = env_state.get("allegro_hand", {})
            kuka_state = env_state.get("kuka", {})
            object_state = env_state.get("object", {})
            goal_state = env_state.get("goal", {})
            actions = env_state.get("actions", torch.zeros(16))  # Default to zeros if actions not provided

            # Extract object and goal poses
            object_pos = object_state.get("pos", torch.zeros(3))
            object_rot = object_state.get("rot", torch.tensor([1.0, 0.0, 0.0, 0.0]))  # w,x,y,z quaternion

            goal_pos = goal_state.get("pos", torch.zeros(3))
            goal_rot = goal_state.get("rot", torch.tensor([1.0, 0.0, 0.0, 0.0]))

            # Distance from object to goal
            goal_dist = torch.norm(object_pos - goal_pos, p=2)

            # Orientation alignment between object and goal
            # Convert quaternions to rotation matrices and compute angular distance
            quat_diff = self._quat_mul(object_rot, self._quat_conjugate(goal_rot))
            rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[0:3]), max=1.0))

            # Calculate reward components
            dist_rew = goal_dist * dist_reward_scale
            rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

            # Action penalty to encourage smooth motions
            action_penalty = torch.sum(actions**2)

            # Total reward
            reward = -dist_rew + rot_rew - action_penalty * action_penalty_scale

            # Success bonus
            if goal_dist < success_tolerance and rot_dist < success_tolerance:
                reward += reach_goal_bonus

            # Fall penalty if object is too far from goal
            if goal_dist >= fall_dist:
                reward += fall_penalty

            rewards.append(reward)

        # Return concatenated rewards
        return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def _quat_mul(self, a, b):
        """Multiply two quaternions."""
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
