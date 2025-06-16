from __future__ import annotations

import logging

import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.constants import PhysicStateType, TaskType
from metasim.utils import configclass
from metasim.utils.math import quat_inv, quat_mul

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class AllegroHandCfg(BaseTaskCfg):
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
        self._prev_actions = None
        self._goal_rotations = None

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

    def get_observation(self, states):
        observations = []

        for i, env_state in enumerate(states):
            robot_state = env_state["robots"]["allegro_hand"]
            block_state = env_state["objects"]["block"]
            goal_state = env_state["objects"]["goal"]

            if "joint_qpos" in robot_state:
                hand_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32)
            elif "dof_pos" in robot_state:
                hand_pos = torch.tensor([v for v in robot_state["dof_pos"].values()], dtype=torch.float32)
            else:
                hand_pos = torch.zeros(16, dtype=torch.float32)

            hand_base_pos = torch.tensor(robot_state.get("pos", [0.0, 0.0, 0.5]), dtype=torch.float32)

            object_pos = torch.tensor(block_state["pos"], dtype=torch.float32)
            object_rot = torch.tensor(block_state["rot"], dtype=torch.float32)

            target_pos = torch.tensor(goal_state["pos"], dtype=torch.float32)
            target_rot = torch.tensor(goal_state["rot"], dtype=torch.float32)

            object_rot_norm = torch.norm(object_rot, p=2, dim=-1, keepdim=True)
            if object_rot_norm > 0:
                object_rot = object_rot / object_rot_norm
            else:
                object_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            target_rot_norm = torch.norm(target_rot, p=2, dim=-1, keepdim=True)
            if target_rot_norm > 0:
                target_rot = target_rot / target_rot_norm
            else:
                target_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            quat_diff = quat_mul(object_rot, quat_inv(target_rot))
            quat_diff_norm = torch.norm(quat_diff, p=2, dim=-1, keepdim=True)
            if quat_diff_norm > 0:
                quat_diff = quat_diff / quat_diff_norm
            else:
                quat_diff = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            if self.obs_type == "full_no_vel":
                obs = torch.cat([
                    hand_pos,
                    object_pos,
                    object_rot,
                    target_pos,
                    target_rot,
                    quat_diff,
                    self._get_prev_actions(i)
                    if hasattr(self, "_prev_actions") and self._prev_actions is not None
                    else torch.zeros(16, dtype=torch.float32),
                ])
            elif self.obs_type == "full":
                if "joint_qvel" in robot_state:
                    hand_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32)
                elif "dof_vel" in robot_state:
                    hand_vel = torch.tensor([v for v in robot_state["dof_vel"].values()], dtype=torch.float32)
                else:
                    hand_vel = torch.zeros(16, dtype=torch.float32)
                object_vel = torch.tensor(
                    block_state.get("lin_vel", block_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32
                )
                object_ang_vel = torch.tensor(block_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32)

                obs = torch.cat([
                    hand_pos,
                    hand_vel * 0.2,
                    object_pos,
                    object_rot,
                    object_vel,
                    object_ang_vel * 0.2,
                    target_pos,
                    target_rot,
                    quat_diff,
                    self._get_prev_actions(i)
                    if hasattr(self, "_prev_actions") and self._prev_actions is not None
                    else torch.zeros(16, dtype=torch.float32),
                ])
            elif self.obs_type == "full_state":
                if "joint_qvel" in robot_state:
                    hand_vel = torch.tensor(robot_state["joint_qvel"])
                elif "dof_vel" in robot_state:
                    hand_vel = torch.tensor([v for v in robot_state["dof_vel"].values()])
                else:
                    hand_vel = torch.zeros(16)
                object_vel = torch.tensor(block_state.get("lin_vel", block_state.get("vel", [0.0, 0.0, 0.0])))
                object_ang_vel = torch.tensor(block_state.get("ang_vel", [0.0, 0.0, 0.0]))

                obs = torch.cat([
                    hand_pos,
                    hand_vel * 0.2,
                    torch.zeros(16),
                    object_pos,
                    object_rot,
                    object_vel,
                    object_ang_vel * 0.2,
                    target_pos,
                    target_rot,
                    quat_diff,
                    self._get_prev_actions(i)
                    if hasattr(self, "_prev_actions") and self._prev_actions is not None
                    else torch.zeros(16, dtype=torch.float32),
                ])

            if torch.isnan(obs).any():
                log.warning("NaN detected in observation. Replacing with zeros.")
                obs = torch.nan_to_num(obs, nan=0.0)

            observations.append(obs)

        return torch.stack(observations) if observations else torch.zeros((0, 50))

    def reward_fn(self, states, actions):
        if not hasattr(self, "_debug_printed_reward"):
            self._debug_printed_reward = True

        if hasattr(self, "_prev_actions"):
            num_envs = len(actions)
            if self._prev_actions is None or self._prev_actions.shape[0] != num_envs:
                self._prev_actions = torch.zeros((num_envs, 16), dtype=torch.float32)

            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        self._prev_actions[i] = torch.tensor(action_values, dtype=torch.float32)
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            if not hasattr(self, "_debug_printed_objects"):
                log.debug(f"Objects in states: {list(states.objects.keys())}")
                for obj_name, obj_state in states.objects.items():
                    log.debug(f"Object '{obj_name}' has attributes: {dir(obj_state)}")
                    if hasattr(obj_state, "root_state"):
                        log.debug(f"Object '{obj_name}' root_state shape: {obj_state.root_state.shape}")
                        log.debug(f"Object '{obj_name}' root_state sample: {obj_state.root_state[0]}")
                self._debug_printed_objects = True

            object_state = states.objects.get("block")
            goal_state = states.objects.get("goal")

            if object_state is None or goal_state is None:
                log.error(f"Missing objects - block: {object_state is not None}, goal: {goal_state is not None}")
                return torch.zeros(
                    len(actions),
                    device=actions[0][next(iter(actions[0].keys()))]["dof_pos_target"][
                        next(iter(actions[0][next(iter(actions[0].keys()))]["dof_pos_target"].keys()))
                    ].device
                    if isinstance(actions[0], dict)
                    else torch.device("cpu"),
                )

            object_pos = object_state.root_state[:, :3]
            object_rot = object_state.root_state[:, 3:7]
            goal_pos = goal_state.root_state[:, :3]
            goal_rot = goal_state.root_state[:, 3:7]

            num_envs = object_pos.shape[0]
            action_tensor = torch.zeros((num_envs, 16), device=object_pos.device)

            for i, act in enumerate(actions):
                if isinstance(act, dict):
                    robot_name = next(iter(act.keys()))
                    if "dof_pos_target" in act[robot_name]:
                        action_values = list(act[robot_name]["dof_pos_target"].values())
                        action_tensor[i] = torch.tensor(action_values, device=object_pos.device)

            pos_dist = torch.norm(object_pos - goal_pos, p=2, dim=1)
            dist_reward = self.dist_reward_scale * pos_dist

            if torch.isnan(object_pos).any() or torch.isnan(goal_pos).any():
                log.warning(f"NaN in positions - object_pos: {object_pos[0]}, goal_pos: {goal_pos[0]}")

            object_rot = torch.nn.functional.normalize(object_rot, p=2, dim=1)
            goal_rot = torch.nn.functional.normalize(goal_rot, p=2, dim=1)

            if torch.isnan(object_rot).any() or torch.isnan(goal_rot).any():
                log.warning(f"NaN in rotations - object_rot: {object_rot[0]}, goal_rot: {goal_rot[0]}")

            quat_dot = torch.abs(torch.sum(object_rot * goal_rot, dim=1))
            quat_dot = torch.clamp(quat_dot, min=0.0, max=1.0)

            rot_dist = torch.where(quat_dot >= 1.0, torch.zeros_like(quat_dot), 2.0 * torch.acos(quat_dot))

            rot_reward = self.rot_reward_scale / (torch.abs(rot_dist) + self.rot_eps)

            action_penalty = torch.sum(action_tensor**2, dim=1) * self.action_penalty_scale

            rewards = dist_reward + rot_reward + action_penalty

            success_mask = torch.abs(rot_dist) <= self.success_tolerance
            rewards[success_mask] += self.reach_goal_bonus

            fall_mask = pos_dist >= self.fall_dist
            rewards[fall_mask] += self.fall_penalty

            return rewards

        else:
            rewards = []
            for i, env_state in enumerate(states):
                object_state = env_state["objects"]["block"]
                goal_state = env_state["objects"]["goal"]

                if isinstance(actions[i], dict):
                    robot_name = next(iter(actions[i].keys()))
                    if "dof_pos_target" in actions[i][robot_name]:
                        action = torch.tensor(list(actions[i][robot_name]["dof_pos_target"].values()))
                    else:
                        action = torch.zeros(16)
                else:
                    action = torch.zeros(16)

                object_pos = torch.tensor(object_state["pos"])
                object_rot = torch.tensor(object_state["rot"])

                goal_pos = torch.tensor(goal_state["pos"])
                goal_rot = torch.tensor(goal_state["rot"])

                pos_dist = torch.norm(object_pos - goal_pos, p=2)
                dist_reward = self.dist_reward_scale * pos_dist

                object_rot = torch.nn.functional.normalize(object_rot, p=2, dim=0)
                goal_rot = torch.nn.functional.normalize(goal_rot, p=2, dim=0)

                quat_dot = torch.abs(torch.dot(object_rot, goal_rot))
                quat_dot = torch.clamp(quat_dot, max=1.0)
                rot_dist = 2.0 * torch.acos(quat_dot)

                rot_reward = self.rot_reward_scale / (torch.abs(rot_dist) + self.rot_eps)

                action_penalty = torch.sum(action**2) * self.action_penalty_scale

                reward = dist_reward + rot_reward + action_penalty

                if torch.abs(rot_dist) <= self.success_tolerance:
                    reward += self.reach_goal_bonus

                if pos_dist >= self.fall_dist:
                    reward += self.fall_penalty

                rewards.append(reward.item())

            return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["allegro_hand"]
            block_state = states.objects["block"]

            robot_pos = robot_state.root_state[:, :3]
            block_pos = block_state.root_state[:, :3]

            has_nan = torch.isnan(robot_pos).any(dim=1) | torch.isnan(block_pos).any(dim=1)

            goal_dist = torch.norm(block_pos - robot_pos, p=2, dim=1)
            terminations = (goal_dist >= self.fall_dist) | has_nan

            return terminations

        else:
            terminations = []
            for env_state in states:
                robot_state = env_state["robots"]["allegro_hand"]
                block_state = env_state["objects"]["block"]
                robot_pos = torch.tensor(robot_state["pos"])
                block_pos = torch.tensor(block_state["pos"])

                goal_dist = torch.norm(block_pos - robot_pos, p=2)
                terminate = goal_dist >= self.fall_dist

                terminations.append(terminate.item() if isinstance(terminate, torch.Tensor) else terminate)

            return torch.tensor(terminations) if terminations else torch.tensor([False])

    def build_scene(self, config=None):
        self._prev_actions = None
        self._success_count = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            num_envs = 1
        else:
            num_envs = len(env_ids)

        rand_floats = torch.rand((num_envs, 4)) * 2.0 - 1.0
        new_rot = torch.nn.functional.normalize(rand_floats, p=2, dim=1)

        mask = new_rot[:, 0] < 0
        new_rot[mask] = -new_rot[mask]

        self._goal_rotations = new_rot

        if self._prev_actions is None:
            self._prev_actions = torch.zeros((num_envs, 16), dtype=torch.float32)
        else:
            if env_ids is not None:
                self._prev_actions[env_ids] = 0.0

    def post_reset(self):
        pass

    def _get_prev_actions(self, env_idx):
        if hasattr(self, "_prev_actions") and self._prev_actions is not None:
            if env_idx < len(self._prev_actions):
                return self._prev_actions[env_idx]
        return torch.zeros(16, dtype=torch.float32)
