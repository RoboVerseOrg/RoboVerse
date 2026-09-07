"""Sub-module containing utilities for camera parameters."""

import numpy as np
import torch

from metasim.utils.math import quat_from_matrix

#: Change of basis from an OpenGL camera frame (right +X, up +Y, backward +Z) to the world-convention
#: frame ``CameraState.quat_world`` uses (forward +X, left +Y, up +Z): forward = -Z_gl, left = -X_gl,
#: up = +Y_gl. ``convert_camera_frame_orientation_convention(origin="opengl", target="world")`` is the same
#: rotation (a test pins that); this is the matrix form, one matmul per call on the state-read path.
_OPENGL_TO_WORLD = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def camera_quat_world_from_opengl(c2w_rotation: torch.Tensor) -> torch.Tensor:
    """``CameraState.quat_world`` (w, x, y, z) from a camera-to-world rotation in the OpenGL convention.

    MuJoCo's ``cam_xmat`` and every OpenGL-style renderer describe the camera frame as right +X, up +Y,
    looking down -Z; the state reports the world convention (forward +X, up +Z) that Isaac Sim's
    ``quat_w_world`` uses, so a consumer rotates +X by the quaternion to get the viewing direction on
    every backend. The inverse is ``CameraState.quat_opengl``.

    Args:
        c2w_rotation: ``(..., 3, 3)`` rotation matrices whose columns are the camera's right, up and
            backward axes in world coordinates (any float dtype; the result is float32).
    """
    return quat_from_matrix(c2w_rotation.to(torch.float32) @ _OPENGL_TO_WORLD)


def camera_pose_fields(intrinsics: np.ndarray, c2w: np.ndarray) -> dict[str, torch.Tensor]:
    """The ``pos`` / ``quat_world`` / ``intrinsics`` of a single-env ``CameraState`` from a backend's matrices.

    Every backend that renders a Gaussian-splat background already computes ``(Ks, c2w)`` for it: a
    ``(3, 3)`` intrinsics matrix and a ``(4, 4)`` camera-to-world transform in the OpenGL convention.
    This is the one place they become state fields (each ``(1, ...)``, float32).
    """
    c2w_t = torch.from_numpy(np.asarray(c2w, dtype=np.float32))
    return {
        "pos": c2w_t[:3, 3].clone().unsqueeze(0),
        "quat_world": camera_quat_world_from_opengl(c2w_t[:3, :3]).unsqueeze(0),
        "intrinsics": torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).unsqueeze(0),
    }


def get_cam_params(
    cam_pos: torch.Tensor,
    cam_look_at: torch.Tensor,
    width=640,
    height=480,
    focal_length=24,
    horizontal_aperture=20.955,
    vertical_aperture=None,
):
    """Get the camera parameters.

    Args:
        cam_pos: The camera position.
        cam_look_at: The camera look at point.
        width: The width of the image.
        height: The height of the image.
        focal_length: The focal length of the camera.
        horizontal_aperture: The horizontal aperture of the camera.
        vertical_aperture: The vertical aperture of the camera.

    Returns:
        The camera parameters.
    """
    if vertical_aperture is None:
        vertical_aperture = horizontal_aperture * height / width

    device = cam_pos.device
    num_envs = len(cam_pos)
    cam_front = cam_look_at - cam_pos
    cam_right = torch.cross(cam_front, torch.tensor([[0.0, 0.0, 1.0]], device=device), dim=1)
    cam_up = torch.cross(cam_right, cam_front)

    cam_right = cam_right / (torch.norm(cam_right, dim=-1, keepdim=True) + 1e-12)
    cam_front = cam_front / (torch.norm(cam_front, dim=-1, keepdim=True) + 1e-12)
    cam_up = cam_up / (torch.norm(cam_up, dim=-1, keepdim=True) + 1e-12)

    # Camera convention difference between ROS and Isaac Sim
    R = torch.stack([cam_right, -cam_up, cam_front], dim=1)  # .transpose(-1, -2)
    t = -torch.bmm(R, cam_pos.unsqueeze(-1)).squeeze()
    extrinsics = torch.eye(4, device=device).unsqueeze(0).tile([num_envs, 1, 1])
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = t

    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    cx = width * 0.5
    cy = height * 0.5

    intrinsics = (
        torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device).unsqueeze(0).tile([num_envs, 1, 1])
    )

    return extrinsics, intrinsics
