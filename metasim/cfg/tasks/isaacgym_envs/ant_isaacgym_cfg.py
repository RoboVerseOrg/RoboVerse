from __future__ import annotations

import torch

from metasim.cfg.checkers import EmptyChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots.ant_cfg import AntCfg as AntRobotCfg
from metasim.constants import TaskType
from metasim.utils import configclass
from metasim.utils.math import (
    euler_xyz_from_quat,
    normalize,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
)

from ..base_task_cfg import BaseTaskCfg
from .isaacgym_task_base import IsaacGymTaskBase


@configclass
class AntIsaacGymCfg(BaseTaskCfg, IsaacGymTaskBase):
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

    def __post_init__(self):
        super().__post_init__()
        self._targets = None
        self._potentials = None
        self._prev_potentials = None
        self._up_vec = None
        self._heading_vec = None
        self._inv_start_rot = None
        self._basis_vec0 = None
        self._basis_vec1 = None
        self._actions = None
        self._joint_gears = None

    def get_observation(self, states):
        observations = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i, env_state in enumerate(states):
            robot_state = env_state["robots"]["ant"]

            torso_pos = torch.tensor(
                robot_state.get("pos", [0.0, 0.0, self.initial_height]), dtype=torch.float32, device=device
            )
            torso_rot = torch.tensor(robot_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
            lin_vel = torch.tensor(
                robot_state.get("lin_vel", robot_state.get("vel", [0.0, 0.0, 0.0])), dtype=torch.float32, device=device
            )
            ang_vel = torch.tensor(robot_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)

            if "joint_qpos" in robot_state:
                dof_pos = torch.tensor(robot_state["joint_qpos"], dtype=torch.float32, device=device)
            elif "dof_pos" in robot_state:
                dof_pos = torch.tensor([v for v in robot_state["dof_pos"].values()], dtype=torch.float32, device=device)
            else:
                dof_pos = torch.zeros(8, dtype=torch.float32, device=device)

            if "joint_qvel" in robot_state:
                dof_vel = torch.tensor(robot_state["joint_qvel"], dtype=torch.float32, device=device)
            elif "dof_vel" in robot_state:
                dof_vel = torch.tensor([v for v in robot_state["dof_vel"].values()], dtype=torch.float32, device=device)
            else:
                dof_vel = torch.zeros(8, dtype=torch.float32, device=device)

            if self._targets is not None and i < len(self._targets):
                target = self._targets[i]
            else:
                target = torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float32, device=torso_pos.device)

            to_target = target - torso_pos
            to_target[2] = 0.0

            if self._potentials is not None and i < len(self._potentials):
                potential = self._potentials[i]
                prev_potential = self._prev_potentials[i] if self._prev_potentials is not None else potential
            else:
                dt = 0.01667
                potential = -torch.norm(to_target, p=2) / dt
                prev_potential = potential

            if self._inv_start_rot is not None:
                inv_start_rot = self._inv_start_rot[0] if len(self._inv_start_rot.shape) > 1 else self._inv_start_rot
            else:
                start_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=torso_pos.device)
                inv_start_rot = quat_inv(start_rot)

            if self._basis_vec0 is not None and i < len(self._basis_vec0):
                basis_vec0 = self._basis_vec0[i]
                basis_vec1 = self._basis_vec1[i]
            else:
                basis_vec0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=torso_pos.device)
                basis_vec1 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=torso_pos.device)

            torso_quat, up_proj, heading_proj, up_vec, heading_vec = self.compute_heading_and_up(
                torso_rot.unsqueeze(0),
                inv_start_rot.unsqueeze(0),
                to_target.unsqueeze(0),
                basis_vec0.unsqueeze(0),
                basis_vec1.unsqueeze(0),
                2,
            )
            torso_quat = torso_quat.squeeze(0)
            up_proj = up_proj.squeeze()
            heading_proj = heading_proj.squeeze()

            vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = self.compute_rot(
                torso_quat.unsqueeze(0),
                lin_vel.unsqueeze(0),
                ang_vel.unsqueeze(0),
                target.unsqueeze(0),
                torso_pos.unsqueeze(0),
            )
            vel_loc = vel_loc.squeeze(0)
            angvel_loc = angvel_loc.squeeze(0)
            roll = roll.squeeze()
            pitch = pitch.squeeze()
            yaw = yaw.squeeze()
            angle_to_target = angle_to_target.squeeze()

            if hasattr(self, "_dof_limits_lower") and hasattr(self, "_dof_limits_upper"):
                dof_pos_scaled = (2.0 * dof_pos - self._dof_limits_upper - self._dof_limits_lower) / (
                    self._dof_limits_upper - self._dof_limits_lower
                )
            else:
                deg_to_rad = 3.14159 / 180.0
                dof_limits_lower = torch.tensor(
                    [
                        -40 * deg_to_rad,
                        30 * deg_to_rad,
                        -40 * deg_to_rad,
                        -100 * deg_to_rad,
                        -40 * deg_to_rad,
                        -100 * deg_to_rad,
                        -40 * deg_to_rad,
                        30 * deg_to_rad,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                dof_limits_upper = torch.tensor(
                    [
                        40 * deg_to_rad,
                        100 * deg_to_rad,
                        40 * deg_to_rad,
                        -30 * deg_to_rad,
                        40 * deg_to_rad,
                        -30 * deg_to_rad,
                        40 * deg_to_rad,
                        100 * deg_to_rad,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                dof_pos_scaled = (2.0 * dof_pos - dof_limits_upper - dof_limits_lower) / (
                    dof_limits_upper - dof_limits_lower
                )

            sensor_forces = torch.zeros(24, dtype=torch.float32, device=device)
            if "sensor_forces" in robot_state:
                sensor_forces = torch.tensor(
                    robot_state["sensor_forces"], dtype=torch.float32, device=device
                ).flatten()[:24]

            if self._actions is not None and i < len(self._actions):
                actions = self._actions[i]
            else:
                actions = torch.zeros(8, dtype=torch.float32, device=device)

            obs = torch.cat([
                torso_pos[2:3],
                vel_loc,
                angvel_loc,
                yaw.unsqueeze(0),
                roll.unsqueeze(0),
                angle_to_target.unsqueeze(0),
                up_proj.unsqueeze(0),
                heading_proj.unsqueeze(0),
                dof_pos_scaled,
                dof_vel * self.dof_vel_scale,
                sensor_forces * self.contact_force_scale,
                actions,
            ])

            observations.append(obs)

        return torch.stack(observations) if observations else torch.zeros((0, 60))

    def reward_fn(self, states, actions):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["ant"]

            torso_pos = robot_state.root_state[:, 0:3]
            torso_rot = robot_state.root_state[:, 3:7]
            lin_vel = robot_state.root_state[:, 7:10]
            ang_vel = robot_state.root_state[:, 10:13]

            dof_pos = robot_state.joint_pos
            dof_vel = robot_state.joint_vel

            if self._targets is None:
                num_envs = torso_pos.shape[0]
                self._targets = torch.tensor([1000.0, 0.0, 0.0], device=torso_pos.device).repeat((num_envs, 1))

            to_target = self._targets - torso_pos
            to_target[:, 2] = 0.0

            dt = 0.01667
            if self._prev_potentials is not None:
                self._prev_potentials = self._potentials.clone()
            self._potentials = -torch.norm(to_target, p=2, dim=-1) / dt

            if self._prev_potentials is None:
                self._prev_potentials = self._potentials.clone()

            if self._inv_start_rot is None:
                start_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=torso_pos.device)
                self._inv_start_rot = quat_inv(start_rot).repeat((torso_pos.shape[0], 1))

            if self._basis_vec0 is None:
                self._basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=torso_pos.device).repeat((
                    torso_pos.shape[0],
                    1,
                ))
                self._basis_vec1 = torch.tensor([0.0, 0.0, 1.0], device=torso_pos.device).repeat((
                    torso_pos.shape[0],
                    1,
                ))

            torso_quat, up_proj, heading_proj, up_vec, heading_vec = self.compute_heading_and_up(
                torso_rot, self._inv_start_rot, to_target, self._basis_vec0, self._basis_vec1, 2
            )

            self._up_vec = up_vec
            self._heading_vec = heading_vec

            heading_weight_tensor = torch.ones_like(heading_proj) * self.heading_weight
            heading_reward = torch.where(
                heading_proj > 0.8, heading_weight_tensor, self.heading_weight * heading_proj / 0.8
            )

            up_reward = torch.zeros_like(heading_reward)
            up_reward = torch.where(up_proj > 0.93, up_reward + self.up_weight, up_reward)

            if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
                actions_list = []
                for act in actions:
                    if self.robot.name in act and "dof_pos_target" in act[self.robot.name]:
                        joint_actions = act[self.robot.name]["dof_pos_target"]
                        action_values = [
                            joint_actions[f"hip_{i // 2 + 1}" if i % 2 == 0 else f"ankle_{i // 2 + 1}"]
                            for i in range(8)
                        ]
                        actions_list.append(torch.tensor(action_values, device=torso_pos.device))
                    else:
                        actions_list.append(torch.zeros(8, device=torso_pos.device))
                actions_tensor = torch.stack(actions_list)
            elif isinstance(actions, torch.Tensor):
                actions_tensor = actions.to(torso_pos.device)
            else:
                actions_tensor = torch.zeros((torso_pos.shape[0], 8), device=torso_pos.device)

            actions_cost = torch.sum(actions_tensor**2, dim=-1)
            electricity_cost = torch.sum(torch.abs(actions_tensor * dof_vel), dim=-1)

            deg_to_rad = 3.14159 / 180.0
            dof_limits_lower = torch.tensor(
                [
                    -40 * deg_to_rad,
                    30 * deg_to_rad,
                    -40 * deg_to_rad,
                    -100 * deg_to_rad,
                    -40 * deg_to_rad,
                    -100 * deg_to_rad,
                    -40 * deg_to_rad,
                    30 * deg_to_rad,
                ],
                device=torso_pos.device,
            )
            dof_limits_upper = torch.tensor(
                [
                    40 * deg_to_rad,
                    100 * deg_to_rad,
                    40 * deg_to_rad,
                    -30 * deg_to_rad,
                    40 * deg_to_rad,
                    -30 * deg_to_rad,
                    40 * deg_to_rad,
                    100 * deg_to_rad,
                ],
                device=torso_pos.device,
            )
            dof_pos_scaled = (2.0 * dof_pos - dof_limits_upper - dof_limits_lower) / (
                dof_limits_upper - dof_limits_lower
            )
            dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.99, dim=-1)

            alive_reward = torch.ones_like(self._potentials) * 0.5
            progress_reward = self._potentials - self._prev_potentials

            total_reward = (
                progress_reward
                + alive_reward
                + up_reward
                + heading_reward
                - self.actions_cost_scale * actions_cost
                - self.energy_cost_scale * electricity_cost
                - dof_at_limit_cost * self.joints_at_limit_cost_scale
            )

            total_reward = torch.where(
                torso_pos[:, 2] < self.termination_height, torch.ones_like(total_reward) * self.death_cost, total_reward
            )

            return total_reward

        else:
            rewards = []
            for i, env_state in enumerate(states):
                robot_state = env_state["robots"]["ant"]

                torso_pos = torch.tensor(robot_state["pos"], dtype=torch.float32)

                reward = 1.0
                if torso_pos[2] < self.termination_height:
                    reward = self.death_cost

                rewards.append(reward)

            return torch.tensor(rewards) if rewards else torch.tensor([0.0])

    def termination_fn(self, states):
        if hasattr(states, "__class__") and states.__class__.__name__ == "TensorState":
            robot_state = states.robots["ant"]
            torso_height = robot_state.root_state[:, 2]
            terminations = torso_height < self.termination_height
            return terminations
        else:
            terminations = []
            for env_state in states:
                robot_state = env_state["robots"]["ant"]
                torso_pos = torch.tensor(robot_state["pos"], dtype=torch.float32)
                terminations.append(torso_pos[2] < self.termination_height)
            return torch.tensor(terminations) if terminations else torch.tensor([False])

    def build_scene(self, config=None):
        self._targets = None
        self._potentials = None
        self._prev_potentials = None
        self._actions = None

    def reset(self, env_ids=None):
        if env_ids is None:
            num_envs = 1
            env_ids = list(range(num_envs))
        else:
            num_envs = len(env_ids)

        if self._targets is None:
            self._targets = torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float32).repeat((num_envs, 1))

        if self._potentials is None:
            dt = 0.01667
            self._potentials = torch.zeros(num_envs, dtype=torch.float32) - 1000.0 / dt
            self._prev_potentials = self._potentials.clone()

        if env_ids is not None and self._potentials is not None:
            for i, env_id in enumerate(env_ids):
                if env_id < len(self._potentials):
                    self._potentials[env_id] = -1000.0 / 0.01667
                    if self._prev_potentials is not None:
                        self._prev_potentials[env_id] = self._potentials[env_id]

    def post_reset(self):
        pass

    def initialize_buffers(self, num_envs, device):
        self._targets = torch.tensor([1000.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
        dt = 0.01667
        self._potentials = torch.zeros(num_envs, device=device) - 1000.0 / dt
        self._prev_potentials = self._potentials.clone()

        start_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        self._inv_start_rot = quat_inv(start_rot).repeat((num_envs, 1))

        self._basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
        self._basis_vec1 = torch.tensor([0.0, 0.0, 1.0], device=device).repeat((num_envs, 1))

        self._up_vec = torch.tensor([0.0, 0.0, 1.0], device=device).repeat((num_envs, 1))
        self._heading_vec = torch.tensor([1.0, 0.0, 0.0], device=device).repeat((num_envs, 1))

        deg_to_rad = 3.14159 / 180.0
        self._dof_limits_lower = torch.tensor(
            [
                -40 * deg_to_rad,
                30 * deg_to_rad,
                -40 * deg_to_rad,
                -100 * deg_to_rad,
                -40 * deg_to_rad,
                -100 * deg_to_rad,
                -40 * deg_to_rad,
                30 * deg_to_rad,
            ],
            device=device,
        )
        self._dof_limits_upper = torch.tensor(
            [
                40 * deg_to_rad,
                100 * deg_to_rad,
                40 * deg_to_rad,
                -30 * deg_to_rad,
                40 * deg_to_rad,
                -30 * deg_to_rad,
                40 * deg_to_rad,
                100 * deg_to_rad,
            ],
            device=device,
        )

        self._joint_gears = torch.ones(8, device=device) * 150.0

    def set_actions(self, actions):
        self._actions = actions

    def compute_heading_and_up(self, torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx):
        num_envs = torso_rotation.shape[0]
        target_dirs = normalize(to_target)

        torso_quat = quat_mul(torso_rotation, inv_start_rot)
        up_vec = quat_rotate(torso_quat, vec1).view(num_envs, 3)
        heading_vec = quat_rotate(torso_quat, vec0).view(num_envs, 3)
        up_proj = up_vec[:, up_idx]
        heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

        return torso_quat, up_proj, heading_proj, up_vec, heading_vec

    def compute_rot(self, torso_quat, velocity, ang_velocity, targets, torso_positions):
        vel_loc = quat_rotate_inverse(torso_quat, velocity)
        angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)

        roll, pitch, yaw = euler_xyz_from_quat(torso_quat)

        walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2], targets[:, 0] - torso_positions[:, 0])
        angle_to_target = walk_target_angle - yaw

        return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target
