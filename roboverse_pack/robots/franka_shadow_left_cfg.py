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
class FrankaShadowHandLeftCfg(BaseDexCfg):
    """Cfg for the franka with left shadow hand robot."""

    name: str = "franka_shadow_left"
    num_joints: int = 31
    num_arm_joints: int = 7
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/franka_shadow_hand/urdf/franka_shadow_left.urdf"
    mjcf_path: str = "roboverse_data/robots/franka_shadow_hand/mjcf/franka_shadow_left.xml"
    usd_path: str = "roboverse_data/robots/franka_shadow_hand/usd/franka_shadow_left.usd"
    project_root: Path = Path(__file__).resolve().parents[2]
    hand_urdf_path: Path = (
        project_root / "roboverse_data" / "robots" / "franka_shadow_hand" / "urdf" / "shadow_left.urdf"
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
    arm_orientation_scale: float = 0.1
    hand_translation_scale: float = 0.05
    hand_orientation_scale: float = 0.05
    dof_speed_scale: float = 20
    fingertips = ["ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"]
    fingertips_offset = [0.0, 0.005, 0.02]
    ee_base_link = "panda_link7"
    wrist = "forearm"
    palm = "palm"
    palm_offset = [0.0, -0.02, 0.08]
    root_link = "panda_link0"
    shadow_hand_wrist_stiffness: float = 5
    shadow_hand_wrist_damping: float = 0.5
    shadow_hand_finger_stiffness: float = 1.0
    shadow_hand_finger_damping: float = 0.1
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 1.0  # Scale for force and torque observations

    actuators: dict[str, BaseActuatorCfg] = {
        "FFJ1": BaseActuatorCfg(),
        "FFJ2": BaseActuatorCfg(),
        "FFJ3": BaseActuatorCfg(),
        "FFJ4": BaseActuatorCfg(),
        "LFJ1": BaseActuatorCfg(),
        "LFJ2": BaseActuatorCfg(),
        "LFJ3": BaseActuatorCfg(),
        "LFJ4": BaseActuatorCfg(),
        "LFJ5": BaseActuatorCfg(),
        "MFJ1": BaseActuatorCfg(),
        "MFJ2": BaseActuatorCfg(),
        "MFJ3": BaseActuatorCfg(),
        "MFJ4": BaseActuatorCfg(),
        "RFJ1": BaseActuatorCfg(),
        "RFJ2": BaseActuatorCfg(),
        "RFJ3": BaseActuatorCfg(),
        "RFJ4": BaseActuatorCfg(),
        "THJ1": BaseActuatorCfg(),
        "THJ2": BaseActuatorCfg(),
        "THJ3": BaseActuatorCfg(),
        "THJ4": BaseActuatorCfg(),
        "THJ5": BaseActuatorCfg(),
        "WRJ1": BaseActuatorCfg(),
        "WRJ2": BaseActuatorCfg(),
        "panda_joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(stiffness=800, damping=50, velocity_limit=2.61),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "FFJ1": (0, 1.571),
        "FFJ2": (0, 1.571),
        "FFJ3": (0, 1.571),
        "FFJ4": (-0.349, 0.349),
        "LFJ1": (0, 1.571),
        "LFJ2": (0, 1.571),
        "LFJ3": (0, 1.571),
        "LFJ4": (-0.349, 0.349),
        "LFJ5": (0, 0.785),
        "MFJ1": (0, 1.571),
        "MFJ2": (0, 1.571),
        "MFJ3": (0, 1.571),
        "MFJ4": (-0.349, 0.349),
        "RFJ1": (0, 1.571),
        "RFJ2": (0, 1.571),
        "RFJ3": (0, 1.571),
        "RFJ4": (-0.349, 0.349),
        "THJ1": (0, 1.571),
        "THJ2": (-0.524, 0.524),
        "THJ3": (-0.209, 0.209),
        "THJ4": (0, 1.222),
        "THJ5": (-0.2, 1.047),
        "WRJ1": (-0.698, 0.489),
        "WRJ2": (-0.489, 0.14),
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
        "FFJ1": 0.0,
        "FFJ2": 0.0,
        "FFJ3": 0.0,
        "FFJ4": 0.0,
        "LFJ1": 0.0,
        "LFJ2": 0.0,
        "LFJ3": 0.0,
        "LFJ4": 0.0,
        "LFJ5": 0.0,
        "MFJ1": 0.0,
        "MFJ2": 0.0,
        "MFJ3": 0.0,
        "MFJ4": 0.0,
        "RFJ1": 0.0,
        "RFJ2": 0.0,
        "RFJ3": 0.0,
        "RFJ4": 0.0,
        "THJ1": 0.0,
        "THJ2": 0.0,
        "THJ3": 0.0,
        "THJ4": 0.0,
        "THJ5": 0.0,
        "WRJ1": 0.0,
        "WRJ2": 0.0,
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
        if self.isaacgym_read_mjcf:
            self.actuators["LFJ1"].fully_actuated = False
            self.actuators["RFJ1"].fully_actuated = False
            self.actuators["MFJ1"].fully_actuated = False
            self.actuators["FFJ1"].fully_actuated = False
        for name in ["WRJ1", "WRJ2"]:
            self.actuators[name].stiffness = self.shadow_hand_wrist_stiffness
            self.actuators[name].damping = self.shadow_hand_wrist_damping
        for name in [
            "FFJ1",
            "FFJ2",
            "FFJ3",
            "FFJ4",
            "LFJ1",
            "LFJ2",
            "LFJ3",
            "LFJ4",
            "LFJ5",
            "MFJ1",
            "MFJ2",
            "MFJ3",
            "MFJ4",
            "RFJ1",
            "RFJ2",
            "RFJ3",
            "RFJ4",
            "THJ1",
            "THJ2",
            "THJ3",
            "THJ4",
            "THJ5",
        ]:
            self.actuators[name].stiffness = self.shadow_hand_finger_stiffness
            self.actuators[name].damping = self.shadow_hand_finger_damping
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
        self.hand_dof_names_ik = self.hand.joints.names[1:]  # skip the first joint which is world joint
        self.ik_reindex = [self.hand_dof_names.index(name) for name in self.hand_dof_names_ik]
        self.ik_reindex_inverse = [self.hand_dof_names_ik.index(name) for name in self.hand_dof_names]

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
            palm_dist = torch.norm(self.palm_pos - target_pos, p=2, dim=-1)
            reward = 0.1 - palm_dist
            return reward, palm_dist

