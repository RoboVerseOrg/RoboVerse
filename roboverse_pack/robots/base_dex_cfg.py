from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import torch

from metasim.scenario.robot import RobotCfg
from metasim.utils import configclass, math
from metasim.utils.state import TensorState


@configclass
class BaseDexCfg(RobotCfg):
    """Base cfg for dexterous hand robot.
    Actuator should be set in alphabetical order of joint names.
    """

    num_arm_joints: int = MISSING
    """Number of arm joints."""

    hand_urdf_path: Path | None = None
    """Path to the hand URDF file. Used for solving ik"""

    robot_controller: str | None = None
    """Controller for dexterous hand."""

    fingertips: list[str] | None = None
    """List of fingertip names."""

    fingertips_offset: list[float, float, float] | None = None
    """List of fingertip offsets."""

    ee_base_link: str | None = None
    """Name of the end-effector base link."""

    wrist: str | None = None
    """Name of the wrist link."""

    wrist_offset: list[float, float, float] | None = None
    """Offset for the wrist link."""

    palm: str | None = None
    """Name of the palm link."""

    palm_offset: list[float, float, float] | None = None
    """Offset for the palm link."""

    root_link: str | None = None
    """Name of the root link."""

    def __post_init__(self):
        super().__post_init__()
        sorted_joint_names = sorted(list(self.joint_limits.keys()))
        self.actuators = {k: self.actuators[k] for k in sorted_joint_names}
        self.joint_limits = {
            k: self.joint_limits[k] for k in sorted_joint_names
        }
        self.default_joint_positions = {
            k: self.default_joint_positions[k] for k in sorted_joint_names
        }
        self.num_fingertips = len(self.fingertips) if self.fingertips else 0
        self.observation_shape = self.num_joints * 2 + 7 + self.num_fingertips * 7
        self.dof_names: list[str] = list(self.default_joint_positions.keys())
        self.hand_dof_names: list[str] = [name for name in self.dof_names if name not in self.arm_dof_names]
        self.hand_dof_idx = [self.dof_names.index(name) for name in self.hand_dof_names]
        self.arm_dof_idx = [self.dof_names.index(name) for name in self.arm_dof_names]
        self.hand_acutuated_idx = [
            idx for idx, name in enumerate(self.hand_dof_names) if self.actuators[name].fully_actuated
        ]
        self.num_actuated_joints = len(self.hand_acutuated_idx) + self.num_arm_joints
        self.joint_limits_lower = [self.joint_limits[name][0] for name in self.dof_names]
        self.joint_limits_upper = [self.joint_limits[name][1] for name in self.dof_names]

    def update_state(self, envstates: TensorState):
        if not hasattr(self, "ft_index"):
            self.ft_index = [envstates.robots[self.name].body_names.index(name) for name in self.fingertips]
        if not hasattr(self, "ee_base_link_index"):
            self.ee_base_link_index = envstates.robots[self.name].body_names.index(self.ee_base_link)
        if not hasattr(self, "wrist_index"):
            self.wrist_index = envstates.robots[self.name].body_names.index(self.wrist)
        if not hasattr(self, "palm_index"):
            self.palm_index = envstates.robots[self.name].body_names.index(self.palm)
        if not hasattr(self, "root_link_index"):
            self.root_link_index = envstates.robots[self.name].body_names.index(self.root_link)
        self.ft_states = envstates.robots[self.name].body_state[:, self.ft_index, :]
        if isinstance(self.joint_limits_lower, list):
            self.joint_limits_lower = torch.tensor(self.joint_limits_lower, device=self.ft_states.device)
            self.joint_limits_upper = torch.tensor(self.joint_limits_upper, device=self.ft_states.device)
        if isinstance(self.fingertips_offset, list):
            self.fingertips_offset = (
                torch.tensor(self.fingertips_offset, device=self.ft_states.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.ft_states.shape[0], self.num_fingertips, 1)
            )
        if isinstance(self.wrist_offset, list):
            self.wrist_offset = (
                torch.tensor(self.wrist_offset, device=self.ft_states.device)
                .unsqueeze(0)
                .repeat(self.ft_states.shape[0], 1)
            )
        if isinstance(self.palm_offset, list):
            self.palm_offset = (
                torch.tensor(self.palm_offset, device=self.ft_states.device)
                .unsqueeze(0)
                .repeat(self.ft_states.shape[0], 1)
            )   
        if isinstance(self.hand_dof_idx, list):
            self.hand_dof_idx = torch.tensor(self.hand_dof_idx, device=self.ft_states.device)
            self.arm_dof_idx = torch.tensor(self.arm_dof_idx, device=self.ft_states.device)
            self.hand_acutuated_idx = torch.tensor(self.hand_acutuated_idx, device=self.ft_states.device)
        self.wrist_state = envstates.robots[self.name].body_state[:, self.wrist_index, :]
        self.ft_pos = self.ft_states[:, :, :3]
        self.ft_rot = self.ft_states[:, :, 3:7]
        if self.fingertips_offset is not None:
            self.ft_pos = self.ft_pos + math.quat_apply(
                self.ft_rot, self.fingertips_offset
            )  # (num_envs, num_fingertips, 3)
        self.wrist_pos = self.wrist_state[:, :3]
        self.wrist_rot = self.wrist_state[:, 3:7]
        if self.wrist_offset is not None:
            self.wrist_pos = self.wrist_pos + math.quat_apply(
                self.wrist_rot, self.palm_offset
            )
        self.ft_relative_rot = math.quat_mul(
            math.quat_inv(self.wrist_rot).unsqueeze(1).repeat(1, self.num_fingertips, 1),
            self.ft_rot,
        )
        self.palm_state = envstates.robots[self.name].body_state[:, self.palm_index, :]
        self.palm_pos = self.palm_state[:, :3]
        self.palm_rot = self.palm_state[:, 3:7]
        if self.palm_offset is not None:
            self.palm_pos = self.palm_pos + math.quat_apply(
                self.palm_rot, self.palm_offset
            )
        self.dof_pos = envstates.robots[self.name].joint_pos
        self.dof_vel = envstates.robots[self.name].joint_vel

    def observation(self, use_palm: bool = False):
        """
        Return the proceptive observation of the robot. The observation includes:
            - Arm joint positions and velocities
            - Hand joint positions and velocities
            - Wrist position and orientation
            - Fingertip positions and orientations
        """
        obs = torch.zeros((self.ft_states.shape[0], self.observation_shape), device=self.ft_states.device)
        obs[:, : self.num_joints] = math.scale_transform(self.dof_pos, self.joint_limits_lower, self.joint_limits_upper)
        obs[:, self.num_joints : 2 * self.num_joints] = self.dof_vel * self.vel_obs_scale
        if use_palm:
            obs[:, 2 * self.num_joints : 2 * self.num_joints + 3] = self.palm_pos
            obs[:, 2 * self.num_joints + 3 : 2 * self.num_joints + 7] = self.palm_rot
        else:
            obs[:, 2 * self.num_joints : 2 * self.num_joints + 3] = self.wrist_pos
            obs[:, 2 * self.num_joints + 3 : 2 * self.num_joints + 7] = self.wrist_rot
        ft_state = torch.cat([self.ft_pos, self.ft_rot], dim=-1).view(self.ft_states.shape[0], -1)
        obs[:, 2 * self.num_joints + 7 :] = ft_state
        return obs

    def reward(self, target_pos, use_palm):
        """Reward based on the distance between the fingertips and the target position.

        Args:
            target_pos: (num_envs, 3) target position
        Returns:
            reward: (num_envs,) reward
        """
        raise NotImplementedError
