from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from yourdfpy import URDF

from metasim.scenario.robot import RobotCfg
from metasim.utils.math import quat_from_matrix, quat_inv, quat_mul


@dataclass
class JointInfo:
    name: str
    joint_type: str
    axis: torch.Tensor
    origin_T: torch.Tensor
    is_actuated: bool
    q_index: int | None


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rot = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )
    return rot


def _make_transform(rpy: tuple[float, float, float], xyz: tuple[float, float, float]) -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
    return T


def _axis_angle_transform(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-9)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1.0 - c
    x, y, z = axis
    R = torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ]
    )
    T = torch.eye(4, dtype=axis.dtype, device=axis.device)
    T[:3, :3] = R
    return T


def _translation_transform(axis: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-9)
    T = torch.eye(4, dtype=axis.dtype, device=axis.device)
    T[:3, 3] = axis * distance
    return T


class BuiltinIKSolver:
    def __init__(self, robot_cfg: RobotCfg):
        self.robot_cfg = robot_cfg
        self.urdf = URDF.load(robot_cfg.urdf_path, load_meshes=False)
        self.ee_link = robot_cfg.ee_body_name
        self.default_map = robot_cfg.default_joint_positions or {}

        actuators_order = list(robot_cfg.actuators.keys())
        ee_joints = set(robot_cfg.ee_joint_names or [])
        self.arm_joint_names = [jn for jn in actuators_order if jn not in ee_joints]
        self.n_dof = len(self.arm_joint_names)

        limits = torch.tensor([robot_cfg.joint_limits[jn] for jn in self.arm_joint_names], dtype=torch.float32)
        self.lower_limits = limits[:, 0]
        self.upper_limits = limits[:, 1]

        self.default_q = torch.tensor([self.default_map.get(jn, 0.0) for jn in self.arm_joint_names], dtype=torch.float32)

        self.chain = self._build_chain()

        self.position_threshold = 1e-3
        self.rotation_threshold = math.radians(1.0)
        self.damping = 1e-2
        self.max_step = 0.2
        self.max_iters = 80

    def _build_chain(self) -> List[JointInfo]:
        base_link = self.urdf.base_link.name  # type: ignore[attr-defined]
        chain_joint_names = self.urdf.get_chain(base_link, self.ee_link, joints=True, links=False, fixed=True)
        joint_infos: List[JointInfo] = []
        arm_set = set(self.arm_joint_names)

        for name in chain_joint_names:
            joint = self.urdf.joint_map[name]
            origin = joint.origin
            if origin is not None:
                xyz = tuple(origin.xyz) if origin.xyz is not None else (0.0, 0.0, 0.0)
                rpy = tuple(origin.rpy) if origin.rpy is not None else (0.0, 0.0, 0.0)
            else:
                xyz = (0.0, 0.0, 0.0)
                rpy = (0.0, 0.0, 0.0)
            origin_T = _make_transform(rpy, xyz)
            axis = torch.tensor(joint.axis if joint.axis is not None else [0.0, 0.0, 1.0], dtype=torch.float32)
            joint_type = joint.joint_type
            is_actuated = joint_type in ("revolute", "prismatic") and name in arm_set
            q_index = self.arm_joint_names.index(name) if is_actuated else None

            joint_infos.append(
                JointInfo(
                    name=name,
                    joint_type=joint_type,
                    axis=axis,
                    origin_T=origin_T,
                    is_actuated=is_actuated,
                    q_index=q_index,
                )
            )

        return joint_infos

    def _motion_transform(
        self, info: JointInfo, value: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if info.joint_type == "revolute":
            axis = info.axis.to(device=device, dtype=dtype)
            return _axis_angle_transform(axis, value)
        if info.joint_type == "prismatic":
            axis = info.axis.to(device=device, dtype=dtype)
            return _translation_transform(axis, value)
        return torch.eye(4, dtype=dtype, device=device)

    def _forward_kinematics(
        self, q: torch.Tensor, compute_jacobian: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        dtype = q.dtype
        device = q.device
        T = torch.eye(4, dtype=dtype, device=device)

        axes_world: list[torch.Tensor] = []
        joint_positions: list[torch.Tensor] = []
        joint_types: list[str] = []

        for info in self.chain:
            origin_T = info.origin_T.to(device=device, dtype=dtype)
            T = T @ origin_T

            if info.is_actuated:
                value = q[info.q_index]  # type: ignore[index]
            else:
                value = torch.tensor(self.default_map.get(info.name, 0.0), dtype=dtype, device=device)

            if compute_jacobian and info.is_actuated:
                R_world = T[:3, :3]
                axis_world = R_world @ info.axis.to(device=device, dtype=dtype)
                axes_world.append(axis_world)
                joint_positions.append(T[:3, 3].clone())
                joint_types.append(info.joint_type)

            motion_T = self._motion_transform(info, value, dtype, device)
            T = T @ motion_T

        pos = T[:3, 3]
        rot = T[:3, :3]
        quat = quat_from_matrix(rot.unsqueeze(0))[0]

        if not compute_jacobian:
            return pos, quat, None

        p_ee = pos
        J = torch.zeros((6, self.n_dof), dtype=dtype, device=device)

        col = 0
        for axis_world, joint_pos, joint_type in zip(axes_world, joint_positions, joint_types):
            if joint_type == "revolute":
                J[:3, col] = torch.cross(axis_world, p_ee - joint_pos)
                J[3:, col] = axis_world
            elif joint_type == "prismatic":
                J[:3, col] = axis_world
                J[3:, col] = torch.zeros(3, dtype=dtype, device=device)
            col += 1

        return pos, quat, J

    def _orientation_error(self, quat_current: torch.Tensor, quat_target: torch.Tensor) -> torch.Tensor:
        q_err = quat_mul(quat_target, quat_inv(quat_current))
        if q_err[0].item() < 0:
            q_err = -q_err
        sin_half = torch.linalg.norm(q_err[1:])
        if sin_half.item() < 1e-8:
            return torch.zeros(3, dtype=quat_current.dtype, device=quat_current.device)
        axis = q_err[1:] / sin_half
        angle = 2.0 * torch.atan2(sin_half, q_err[0].clamp_min(1e-8))
        return axis * angle

    def solve_single(
        self, pos_target: torch.Tensor, quat_target: torch.Tensor, seed_q: torch.Tensor | None
    ) -> tuple[torch.Tensor, bool]:
        device = pos_target.device
        dtype = pos_target.dtype
        if seed_q is None:
            q = self.default_q.to(device=device, dtype=dtype).clone()
        else:
            q = seed_q.to(device=device, dtype=dtype).clone()

        success = False
        lower = self.lower_limits.to(device=device, dtype=dtype)
        upper = self.upper_limits.to(device=device, dtype=dtype)

        for _ in range(self.max_iters):
            pos_curr, quat_curr, J = self._forward_kinematics(q, compute_jacobian=True)
            pos_error = pos_target - pos_curr
            rot_error = self._orientation_error(quat_curr, quat_target)

            if pos_error.norm() <= self.position_threshold and rot_error.norm() <= self.rotation_threshold:
                success = True
                break

            if J is None:
                break

            error = torch.cat([pos_error, rot_error], dim=0)
            JT = J.transpose(0, 1)
            JJt = J @ JT
            damping = (self.damping ** 2) * torch.eye(6, dtype=dtype, device=device)
            try:
                delta = torch.linalg.solve(JJt + damping, error)
            except RuntimeError:
                pinv = torch.linalg.pinv(J)
                dq = pinv @ error
            else:
                dq = JT @ delta

            dq = dq.clamp(-self.max_step, self.max_step)
            q = q + dq
            q = torch.max(torch.min(q, upper), lower)

            if dq.norm().item() < 1e-4:
                break

        return q, success

    def solve_batch(
        self, ee_pos_target: torch.Tensor, ee_quat_target: torch.Tensor, seed_q: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        solutions: list[torch.Tensor] = []
        success_flags = []

        for idx in range(ee_pos_target.shape[0]):
            seed = seed_q[idx] if seed_q is not None else None
            sol, succ = self.solve_single(ee_pos_target[idx], ee_quat_target[idx], seed)
            solutions.append(sol)
            success_flags.append(succ)

        q_solution = torch.stack(solutions, dim=0)
        ik_succ = torch.tensor(success_flags, dtype=torch.bool, device=ee_pos_target.device)
        return q_solution, ik_succ
