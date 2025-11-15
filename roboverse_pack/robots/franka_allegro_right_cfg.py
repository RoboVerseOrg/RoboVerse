from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import torch

from metasim.scenario.robot import BaseActuatorCfg
from metasim.utils import configclass, math
from metasim.utils.bidex_util import _solve_ik_single_env, jax_to_torch, torch_to_jax

from .base_dex_cfg import BaseDexCfg


@configclass
class FrankaAllegroHandRightCfg(BaseDexCfg):
    """Cfg for the franka with right allegro hand robot."""

    name: str = "franka_allegro_right"
    num_joints: int = 23
    num_arm_joints: int = 7
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/franka_allegro_hand/urdf/franka_allegro_right.urdf"
    usd_path: str = "roboverse_data/robots/franka_allegro_hand/usd/franka_allegro_right.usd"
    project_root: Path = Path(__file__).resolve().parents[2]
    hand_urdf_path: Path = (
        project_root / "roboverse_data" / "robots" / "franka_allegro_hand" / "urdf" / "allegro_right.urdf"
    )
    isaacgym_read_mjcf: bool = False
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    use_vhacd: bool = True
    dof_drive_mode: Literal["none", "position", "effort"] = "position"
    angular_damping: float = None
    linear_damping: float = None
    tendon_limit_stiffness: float = 30
    tendon_damping: float = 0.1
    friction = None  # Use default friction from MJCF
    hand_controller: Literal["ik", "dof_pos", "dof_effort"] = "ik"
    arm_controller: Literal["ik", "ik_abs", "dof_pos"] = "ik"
    use_relative_control: bool = False
    arm_translation_scale: float = 0.005
    arm_orientation_scale: float = 0.05
    hand_translation_scale: float = 0.05
    hand_orientation_scale: float = 0.05
    dof_speed_scale: float = 20
    fingertips = ["link_3", "link_7_tip", "link_11_tip", "link_15_tip"]
    fingertips_offset = [0.0, 0.0, 0.0]
    ee_base_link = "panda_link7"
    wrist = "base_link"
    palm = "base_link"
    root_link = "panda_link0"
    allegro_stiffness: float = 3.0
    allegro_damping: float = 0.1
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 1.0  # Scale for force and torque observations

    actuators: dict[str, BaseActuatorCfg] = {
        "joint_0": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_1": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_10": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_11": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_12": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_13": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_14": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_15": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_2": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_3": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_4": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_5": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_6": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_7": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_8": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "joint_9": BaseActuatorCfg(stiffness=allegro_stiffness, damping=allegro_damping),
        "panda_joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(stiffness=800, damping=50, velocity_limit=2.61),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "joint_0": (-0.57, 0.57),
        "joint_1": (-0.296, 1.71),
        "joint_10": (-0.274, 1.809),
        "joint_11": (-0.327, 1.718),
        "joint_12": (0.363, 1.496),
        "joint_13": (-0.205, 1.163),
        "joint_14": (-0.289, 1.644),
        "joint_15": (-0.262, 1.819),
        "joint_2": (-0.274, 1.809),
        "joint_3": (-0.327, 1.718),
        "joint_4": (-0.57, 0.57),
        "joint_5": (-0.296, 1.71),
        "joint_6": (-0.274, 1.809),
        "joint_7": (-0.327, 1.718),
        "joint_8": (-0.57, 0.57),
        "joint_9": (-0.296, 1.71),
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
    }

    arm_dof_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # set False for visualization correction. Also see https://forums.developer.nvidia.com/t/how-to-flip-collision-meshes-in-isaac-gym/260779 for another example.
    isaacgym_flip_visual_attachments = False

    default_joint_positions: dict[str, float] = {
        "joint_0": 0.0,
        "joint_1": 0.0,
        "joint_10": 0.0,
        "joint_11": 0.0,
        "joint_12": 0.363,
        "joint_13": 0.0,
        "joint_14": 1.64,
        "joint_15": 0.0,
        "joint_2": 0.0,
        "joint_3": 0.0,
        "joint_4": 0.0,
        "joint_5": 0.0,
        "joint_6": 0.0,
        "joint_7": 0.0,
        "joint_8": 0.0,
        "joint_9": 0.0,
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 3.1415926,
        "panda_joint7": -2.356194,
    }

    control_type = {}

    def __post_init__(self):
        super().__post_init__()
        self.load_robot_for_ik()

    def scale_hand_action(self, actions: torch.Tensor) -> torch.Tensor:
        if self.hand_controller != "dof_pos":
            raise ValueError("hand_controller must be 'dof_pos' to use scale_hand_action")
        if actions.shape[1] != len(self.hand_acutuated_idx):
            raise ValueError(
                f"action shape {actions.shape} does not match hand dof {self.num_joints - self.num_arm_joints}"
            )
        hand_dof = math.unscale_transform(
            actions,
            self.joint_limits_lower[self.hand_dof_idx][self.hand_acutuated_idx],
            self.joint_limits_upper[self.hand_dof_idx][self.hand_acutuated_idx],
        )
        control_actions = torch.zeros((self.dof_pos.shape[0], len(self.hand_dof_idx)), device=hand_dof.device)
        control_actions[:, self.hand_acutuated_idx] = hand_dof
        return control_actions

    def control_arm(self, dpose, num_envs: int, device: str):
        # Set controller parameters
        # IK params
        if self.arm_controller in ["ik"]:
            dpose[:, :3] = dpose[:, :3] * self.arm_translation_scale
            dpose[:, 3:] = dpose[:, 3:] * self.arm_orientation_scale
            dpose = dpose.unsqueeze(-1)
            damping = 0.05
            jacobian_tensor = self.jacobian[:, self.jacobian_body_reindex, :, :][
                ..., self.jacobian_joint_reindex
            ]  # (num_envs, num_bodies, 6, num_dofs)
            # solve damped least squares
            if self.fix_base_link:
                ik_idx = (
                    self.ee_base_link_index - 1
                    if self.root_link_index < self.ee_base_link_index
                    else self.ee_base_link_index
                )
                j_eef = jacobian_tensor[:, ik_idx, :, self.arm_dof_idx]
            else:
                j_eef = jacobian_tensor[:, self.ee_base_link_index, :, self.arm_dof_idx]
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=device) * (damping**2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
            u += self.dof_pos[:, self.arm_dof_idx]
            u = torch.clamp(u, self.joint_limits_lower[self.arm_dof_idx], self.joint_limits_upper[self.arm_dof_idx])
            return u
        elif self.arm_controller in ["ik_abs"]:
            pos_err = dpose[:, :3] * self.arm_translation_scale
            dpose[:, 3:] = dpose[:, 3:] / dpose[:, 3:].norm(p=2, dim=-1, keepdim=True)
            orr_err = math.orientation_error(dpose[:, 3:], self.wrist_rot)
            dpose = torch.cat([pos_err, orr_err], dim=-1).unsqueeze(-1)
            damping = 0.05
            jacobian_tensor = self.jacobian[:, self.jacobian_body_reindex, :, :][
                ..., self.jacobian_joint_reindex
            ]  # (num_envs, num_bodies, 6, num_dofs)
            # solve damped least squares
            if self.fix_base_link:
                ik_idx = (
                    self.ee_base_link_index - 1
                    if self.root_link_index < self.ee_base_link_index
                    else self.ee_base_link_index
                )
                j_eef = jacobian_tensor[:, ik_idx, :, self.arm_dof_idx]
            else:
                j_eef = jacobian_tensor[:, self.ee_base_link_index, :, self.arm_dof_idx]
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=device) * (damping**2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
            u += self.dof_pos[:, self.arm_dof_idx]
            u = torch.clamp(u, self.joint_limits_lower[self.arm_dof_idx], self.joint_limits_upper[self.arm_dof_idx])
            return u
        elif self.arm_controller in ["dof_pos"]:
            if self.use_relative_control:
                u = self.dof_pos[:, self.arm_dof_idx] + self.dof_speed_scale * dpose
                return torch.clamp(
                    u, self.joint_limits_lower[self.arm_dof_idx], self.joint_limits_upper[self.arm_dof_idx]
                )
            else:
                u = math.unscale_transform(
                    dpose,
                    self.joint_limits_lower[self.arm_dof_idx],
                    self.joint_limits_upper[self.arm_dof_idx],
                )
                return u

    def load_robot_for_ik(self):
        import pyroki as pk
        import yourdfpy

        robot_urdf_path = self.hand_urdf_path

        def filename_handler(fname: str) -> str:
            base_path = robot_urdf_path.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)

        try:
            urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"File {robot_urdf_path} not found.") from err

        # Create robot.
        self.hand = pk.Robot.from_urdf(urdf)
        self.solver = jax.jit(jax.vmap(_solve_ik_single_env, in_axes=(None, 0, 0, 0, None), out_axes=0))
        self.hand_dof_names_ik = self.hand.joints.actuated_names  # skip the first joint which is world joint
        self.ik_reindex = [
            self.hand_dof_names.index(name) for name in self.hand_dof_names_ik if name in self.hand_dof_names
        ]
        self.ik_reindex_inverse = [
            self.hand_dof_names_ik.index(name) for name in self.hand_dof_names if name in self.hand_dof_names_ik
        ]

    def control_hand_ik(self, target_pos, target_rot):
        if self.hand_controller != "ik":
            raise ValueError("hand_controller must be 'ik' to use control_hand_ik")
        target_pos = target_pos * self.hand_translation_scale
        target_rot = target_rot * self.hand_orientation_scale
        init_q = torch_to_jax(self.dof_pos[:, self.hand_dof_idx][:, self.ik_reindex])
        target_wxyz = torch_to_jax(
            math.quat_mul(
                math.quat_from_euler_xyz(target_rot[..., 0], target_rot[..., 1], target_rot[..., 2]),
                self.ft_relative_rot,
            )
        )
        target_position = target_pos + self.ft_pos
        target_position = torch_to_jax(
            math.quat_apply(
                math.quat_inv(self.wrist_rot).unsqueeze(1).repeat(1, self.num_fingertips, 1),
                target_position - self.wrist_pos.unsqueeze(1),
            )
        )
        if not hasattr(self, "target_link_indices"):
            self.target_link_indices = jnp.array(
                [self.hand.links.names.index(name) for name in self.fingertips], dtype=jnp.int32
            )

        solution = self.solver(
            self.hand,
            init_q,
            target_wxyz,
            target_position,
            self.target_link_indices,
        )
        hand_dof = jax_to_torch(solution)[:, self.ik_reindex_inverse]
        hand_dof = torch.clamp(
            hand_dof, self.joint_limits_lower[self.hand_dof_idx], self.joint_limits_upper[self.hand_dof_idx]
        )
        action = torch.zeros((self.dof_pos.shape[0], len(self.hand_dof_idx)), device=hand_dof.device)
        action[:, self.hand_acutuated_idx] = hand_dof[:, self.hand_acutuated_idx]
        return action

    def reward(self, target_pos, use_palm: bool = False):
        """Reward based on the distance between the fingertips and the target position.

        Args:
            target_pos: (num_envs, 3) target position
        Returns:
            reward: (num_envs,) reward
        """
        if not use_palm:
            dists = self.ft_pos - target_pos.unsqueeze(1)  # (num_envs, num_fingertips, 3)
            dists = torch.norm(dists, p=2, dim=-1)  # (num_envs, num_fingertips)
            mean_dists = torch.mean(dists, dim=-1)  # (num_envs,)
            sum_dists = torch.sum(dists, dim=-1)  # (num_envs,)
            reward = 1.2 - sum_dists
            return reward, mean_dists
        else:
            palm_dist = torch.norm(self.palm_state[:, :3] - target_pos, p=2, dim=-1)
            reward = 0.1 - palm_dist
            return reward, palm_dist
