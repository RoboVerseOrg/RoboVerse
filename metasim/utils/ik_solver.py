"""Unified IK solver supporting both curobo and pyroki backends."""

from __future__ import annotations

from typing import Literal

import torch

from metasim.scenario.robot import RobotCfg


class IKSolver:
    """Unified IK solver with curobo/pyroki backends."""

    def __init__(self, robot_cfg: RobotCfg, solver: Literal["curobo", "pyroki"] = "pyroki", no_gnd: bool = False):
        self.robot_cfg = robot_cfg
        self.solver = solver
        self.no_gnd = no_gnd

        # Robot properties
        self.joint_names = list(robot_cfg.joint_limits.keys())
        self.n_robot_dof = len(self.joint_names)
        self.ee_n_dof = len(robot_cfg.gripper_open_q)
        self.n_dof_ik = len(robot_cfg.actuators) - len(robot_cfg.gripper_open_q)

        self._setup_solver()

    def _setup_solver(self):
        if self.solver == "curobo":
            from metasim.utils.kinematics import get_curobo_models

            *_, self.backend_solver = get_curobo_models(self.robot_cfg, self.no_gnd)
        elif self.solver == "pyroki":
            from metasim.utils.kinematics import get_pyroki_model

            self.backend_solver = get_pyroki_model(self.robot_cfg)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def solve_ik_batch(
        self, ee_pos_target: torch.Tensor, ee_quat_target: torch.Tensor, seed_q: torch.Tensor | None = None
    ):
        """Solve IK for batch of poses. Returns (q_solution, ik_succ).

        Args:
            ee_pos_target: Target end-effector positions (B, 3)
            ee_quat_target: Target end-effector quaternions (B, 4)
            seed_q: Seed joint configuration for IK initialization (B, n_dof). Required for curobo.
        """
        num_envs = ee_pos_target.shape[0]

        if self.solver == "curobo":
            from curobo.types.math import Pose

            if seed_q is None:
                raise ValueError("seed_q required for curobo solver")

            seed_config = seed_q[:, : self.n_dof_ik].unsqueeze(1).tile([1, self.backend_solver._num_seeds, 1])
            result = self.backend_solver.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

            ik_succ = result.success.squeeze(1)
            q_solution = torch.zeros((num_envs, self.n_dof_ik), device=ee_pos_target.device)
            q_solution[ik_succ] = result.solution[ik_succ, 0].clone()
            return q_solution, ik_succ

        else:  # pyroki
            q_list = [self.backend_solver["solve_ik"](ee_pos_target[i], ee_quat_target[i]) for i in range(num_envs)]
            q_solution = torch.stack(q_list, dim=0)[:, : self.n_dof_ik]
            ik_succ = torch.ones(num_envs, dtype=torch.bool, device=ee_pos_target.device)
            return q_solution, ik_succ

    def compose_full_joint_command(
        self,
        q_solution: torch.Tensor,
        gripper_widths: torch.Tensor,
        current_q: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | list:
        """Compose full joint command with arm + gripper.

        Args:
            q_solution: IK solution for arm joints (B, n_dof_ik)
            gripper_widths: Gripper joint positions (B, ee_n_dof) or (B,) or (ee_n_dof,)
            current_q: Current full joint state to preserve non-IK joints (B, n_robot_dof). Optional.
            return_dict: If True, return action dictionaries; if False, return tensor (default).

        Returns:
            If return_dict=False: torch.Tensor of shape (B, n_robot_dof) in dictionary order
            If return_dict=True: list of action dictionaries for env execution
        """
        num_envs = q_solution.shape[0]
        device = q_solution.device

        # Get joint names in different orders
        joint_names_dict_order = list(self.robot_cfg.actuators.keys())  # Original dict order
        joint_names_alpha_order = sorted(self.robot_cfg.actuators.keys())  # Alphabetical order

        # Build q_full in original dict order first
        q_full_dict_order = (
            current_q[:, : self.n_robot_dof].clone()
            if current_q is not None
            else torch.zeros((num_envs, self.n_robot_dof), device=device)
        )

        # Place IK solution in the first n_dof_ik positions
        q_full_dict_order[:, : self.n_dof_ik] = q_solution

        # Handle gripper dimensions
        if gripper_widths.dim() == 0:
            gripper_widths = gripper_widths.expand(num_envs, self.ee_n_dof)
        elif gripper_widths.dim() == 1:
            if gripper_widths.shape[0] == num_envs:
                gripper_widths = gripper_widths.unsqueeze(-1).expand(-1, self.ee_n_dof)
            elif gripper_widths.shape[0] == self.ee_n_dof:
                gripper_widths = gripper_widths.unsqueeze(0).expand(num_envs, -1)

        # Place gripper positions in the last ee_n_dof positions
        q_full_dict_order[:, -self.ee_n_dof :] = gripper_widths

        if return_dict:
            # Return action dictionaries using original dict order
            return [
                {
                    self.robot_cfg.name: {
                        "dof_pos_target": dict(zip(joint_names_dict_order, q_full_dict_order[i].tolist()))
                    }
                }
                for i in range(q_full_dict_order.shape[0])
            ]
        else:
            # Convert from dict order to alphabetical order
            # Create reindexing mapping: from dict order to alphabetical order
            dict_to_alpha_idx = [joint_names_dict_order.index(name) for name in joint_names_alpha_order]
            q_full_alpha_order = q_full_dict_order[:, dict_to_alpha_idx]
            return q_full_alpha_order


def setup_ik_solver(
    robot_cfg: RobotCfg, solver: Literal["curobo", "pyroki"] = "pyroki", no_gnd: bool = False
) -> IKSolver:
    """Setup IK solver."""
    return IKSolver(robot_cfg, solver, no_gnd)


def process_gripper_command(
    gripper_binary: torch.Tensor, robot_cfg: RobotCfg, device: torch.device | str
) -> torch.Tensor:
    """Convert binary gripper command to joint widths."""
    if gripper_binary.dim() == 0:
        gripper_binary = gripper_binary.unsqueeze(0)

    num_envs = gripper_binary.shape[0]
    gripper_widths = torch.zeros((num_envs, len(robot_cfg.gripper_open_q)), device=device)

    open_mask = gripper_binary > 0.5
    close_mask = ~open_mask

    if open_mask.any():
        gripper_widths[open_mask] = torch.tensor(robot_cfg.gripper_open_q, device=device)
    if close_mask.any():
        gripper_widths[close_mask] = torch.tensor(robot_cfg.gripper_close_q, device=device)

    return gripper_widths
