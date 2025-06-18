from __future__ import annotations

import logging

import torch

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.anymal_cfg import AnymalCfg as AnymalRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass
from metasim.utils.math import quat_rotate, quat_rotate_inverse

from ..base_task_cfg import BaseTaskCfg

log = logging.getLogger(__name__)


@configclass
class AnymalCfg(BaseTaskCfg):
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

    def __post_init__(self):
        super().__post_init__()
        self._commands = None

    def get_observation(self, states):
        observations = []

        for i, env_state in enumerate(states):
            robot_state = env_state["robots"]["anymal"]

            base_quat = torch.tensor(robot_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
            base_lin_vel = torch.tensor(
                robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32
            )
            base_ang_vel = torch.tensor(robot_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32)

            if not hasattr(self, "_debug_printed"):
                log.debug(f"robot_state keys: {robot_state.keys() if hasattr(robot_state, 'keys') else 'Not a dict'}")
                log.debug(f"base_quat: {base_quat}")
                self._debug_printed = True

            base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel) * self.lin_vel_scale
            base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel) * self.ang_vel_scale

            gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
            projected_gravity = quat_rotate(base_quat, gravity_vec)

            if "joint_qpos" in robot_state:
                dof_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32)
            elif "dof_pos" in robot_state:
                dof_pos = torch.tensor([v for v in robot_state["dof_pos"].values()], dtype=torch.float32)
            else:
                dof_pos = torch.zeros(12, dtype=torch.float32)

            default_pos = torch.tensor(
                [0.03, 0.03, -0.03, -0.03, 0.4, -0.4, 0.4, -0.4, -0.8, 0.8, -0.8, 0.8], dtype=torch.float32
            )

            dof_pos_scaled = (dof_pos - default_pos) * self.dof_pos_scale

            if "joint_qvel" in robot_state:
                dof_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32)
            elif "dof_vel" in robot_state:
                dof_vel = torch.tensor([v for v in robot_state["dof_vel"].values()], dtype=torch.float32)
            else:
                dof_vel = torch.zeros(12, dtype=torch.float32)

            dof_vel_scaled = dof_vel * self.dof_vel_scale

            if self._commands is not None and i < len(self._commands):
                commands = self._commands[i].to(base_quat.device)
            else:
                commands = torch.zeros(3, dtype=torch.float32, device=base_quat.device)

            commands_scaled = commands * torch.tensor(
                [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                dtype=torch.float32,
                device=base_quat.device,
            )

            if hasattr(self, "_prev_actions") and self._prev_actions is not None and i < len(self._prev_actions):
                prev_actions = self._prev_actions[i].to(base_quat.device)
            else:
                prev_actions = torch.zeros(12, dtype=torch.float32, device=base_quat.device)

            obs = torch.cat([
                base_lin_vel_base,
                base_ang_vel_base,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel_scaled,
                prev_actions,
            ])

            observations.append(obs)

        return torch.stack(observations) if observations else torch.zeros((0, 48))

    def reward_fn(self, states, actions):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["anymal"]

            base_quat = robot_state.root_state[:, 3:7]
            base_lin_vel = robot_state.root_state[:, 7:10]
            base_ang_vel = robot_state.root_state[:, 10:13]

            base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
            base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)

            if self._commands is not None:
                commands = self._commands
            else:
                num_envs = base_quat.shape[0]
                commands = torch.zeros((num_envs, 3), device=base_quat.device)

            lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
            ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])

            rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.lin_vel_xy_reward_scale
            rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.ang_vel_z_reward_scale

            if hasattr(robot_state, "torques"):
                torques = robot_state.torques
            else:
                torques = torch.zeros((base_quat.shape[0], 12), device=base_quat.device)

            rew_torque = torch.sum(torch.square(torques), dim=1) * self.torque_reward_scale

            rewards = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
            rewards = torch.clamp(rewards, min=0.0)

            return rewards

        else:
            rewards = []
            for i, env_state in enumerate(states):
                robot_state = env_state["robots"]["anymal"]

                base_quat = torch.tensor(robot_state["rot"], dtype=torch.float32)
                base_lin_vel = torch.tensor(
                    robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32
                )
                base_ang_vel = torch.tensor(robot_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32)

                base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
                base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)

                if self._commands is not None and i < len(self._commands):
                    commands = self._commands[i]
                else:
                    commands = torch.zeros(3, dtype=torch.float32)

                lin_vel_error = torch.sum(torch.square(commands[:2] - base_lin_vel[:2]))
                ang_vel_error = torch.square(commands[2] - base_ang_vel[2])

                rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.lin_vel_xy_reward_scale
                rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.ang_vel_z_reward_scale

                rew_torque = 0.0

                reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
                reward = torch.clamp(reward, min=0.0)

                rewards.append(reward.item())

            return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["anymal"]

            if hasattr(robot_state, "contact_forces"):
                contact_forces = robot_state.contact_forces

                base_contact = torch.norm(contact_forces[:, 0, :], dim=1) > self.base_contact_force_threshold

                terminations = base_contact
            else:
                num_envs = robot_state.root_state.shape[0]
                terminations = torch.zeros(num_envs, dtype=torch.bool, device=robot_state.root_state.device)

            return terminations

        else:
            terminations = []
            for env_state in states:
                terminations.append(False)

            return torch.tensor(terminations) if terminations else torch.tensor([False])

    def build_scene(self, config=None):
        self._commands = None
        self._prev_actions = None

    def reset(self, env_ids=None):
        if env_ids is None:
            num_envs = 1
        else:
            num_envs = len(env_ids)

        device = (
            self._commands.device
            if self._commands is not None
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        commands = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        commands[:, 0] = (
            torch.rand(num_envs, device=device) * (self.command_x_range[1] - self.command_x_range[0])
            + self.command_x_range[0]
        )
        commands[:, 1] = (
            torch.rand(num_envs, device=device) * (self.command_y_range[1] - self.command_y_range[0])
            + self.command_y_range[0]
        )
        commands[:, 2] = (
            torch.rand(num_envs, device=device) * (self.command_yaw_range[1] - self.command_yaw_range[0])
            + self.command_yaw_range[0]
        )

        if self._commands is None:
            self._commands = commands
        else:
            if env_ids is not None and len(env_ids) > 0:
                for i, env_id in enumerate(env_ids):
                    if env_id < len(self._commands):
                        self._commands[env_id] = commands[i]
            else:
                self._commands = commands

        if self._prev_actions is None:
            self._prev_actions = torch.zeros((num_envs, 12), dtype=torch.float32, device=device)
        else:
            if env_ids is not None and len(env_ids) > 0:
                for env_id in env_ids:
                    if env_id < len(self._prev_actions):
                        self._prev_actions[env_id] = 0.0

    def post_reset(self):
        pass
