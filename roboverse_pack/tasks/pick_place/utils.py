"""Franka task utilities: quaternion and vector operations."""

from __future__ import annotations

import torch
from metasim.utils.math import matrix_from_quat


class Utils:
    """Utility functions for Franka robot tasks."""

    @staticmethod
    def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Compute quaternion conjugate. Assumes format (w, x, y, z).

        Args:
            q: Quaternion tensor with shape (..., 4)

        Returns:
            Conjugate quaternion with same shape
        """
        q_flat = q.reshape(-1, 4)
        conj_flat = torch.stack([q_flat[:, 0], -q_flat[:, 1], -q_flat[:, 2], -q_flat[:, 3]], dim=-1)
        return conj_flat.reshape(q.shape)

    @staticmethod
    def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication. Assumes format (w, x, y, z).

        Args:
            q: First quaternion tensor
            r: Second quaternion tensor

        Returns:
            Product quaternion q * r
        """
        q_b, r_b = torch.broadcast_tensors(q, r)
        q_flat = q_b.reshape(-1, 4)
        r_flat = r_b.reshape(-1, 4)

        w1, x1, y1, z1 = q_flat[:, 0], q_flat[:, 1], q_flat[:, 2], q_flat[:, 3]
        w2, x2, y2, z2 = r_flat[:, 0], r_flat[:, 1], r_flat[:, 2], r_flat[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = torch.stack([w, x, y, z], dim=-1)
        return result.reshape(q_b.shape)

    @staticmethod
    def quat_angle(q: torch.Tensor) -> torch.Tensor:
        """Compute rotation angle from quaternion. Assumes format (w, x, y, z).

        Args:
            q: Quaternion tensor with shape (..., 4)

        Returns:
            Rotation angle in radians with shape (...,)
        """
        q_flat = q.reshape(-1, 4)
        imag_norm = torch.norm(q_flat[:, 1:], dim=-1)
        angle = 2.0 * torch.atan2(imag_norm, torch.clamp(q_flat[:, 0].abs(), min=1e-6))
        return angle.reshape(q.shape[:-1])

    @staticmethod
    def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by quaternion. Assumes q format (w, x, y, z).

        Args:
            q: Quaternion tensor with shape (..., 4)
            v: Vector tensor with shape (..., 3)

        Returns:
            Rotated vector with shape (..., 3)
        """
        if v.shape[-1] != 3:
            raise ValueError(f"Expected vector with last dim 3, got {v.shape}")

        prefix_shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
        q_expanded = torch.broadcast_to(q, (*prefix_shape, 4))
        v_expanded = torch.broadcast_to(v, (*prefix_shape, 3))

        q_flat = q_expanded.reshape(-1, 4)
        v_flat = v_expanded.reshape(-1, 3)

        # q format: (w, x, y, z)
        q_w = q_flat[:, 0].unsqueeze(-1)
        q_vec = q_flat[:, 1:4]

        a = v_flat * (2.0 * q_w**2 - 1.0)
        b = torch.cross(q_vec, v_flat, dim=-1) * q_w * 2.0
        c = q_vec * torch.sum(q_vec * v_flat, dim=-1, keepdim=True) * 2.0

        rotated = a + b + c
        return rotated.reshape(*prefix_shape, 3)

    @staticmethod
    def quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to tangent and normal vector representation.

        Assumes format (w, x, y, z).

        Args:
            quat: Quaternion tensor with shape (..., 4)

        Returns:
            Concatenated tangent and normal vectors with shape (..., 6)
        """
        if quat.shape[-1] != 4:
            raise ValueError(f"Expected quaternion shape (..., 4), got {quat.shape}")

        quat_flat = quat.reshape(-1, 4)
        rot_mats = matrix_from_quat(quat_flat)

        ref_tan = torch.zeros((quat_flat.shape[0], 3), device=quat.device, dtype=quat.dtype)
        ref_tan[:, 0] = 1.0
        tan = torch.bmm(rot_mats, ref_tan.unsqueeze(-1)).squeeze(-1)

        ref_norm = torch.zeros_like(ref_tan)
        ref_norm[:, 2] = 1.0
        norm = torch.bmm(rot_mats, ref_norm.unsqueeze(-1)).squeeze(-1)

        tan = tan.reshape(*quat.shape[:-1], 3)
        norm = norm.reshape(*quat.shape[:-1], 3)
        return torch.cat([tan, norm], dim=-1)

