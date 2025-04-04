"""Sub-module containing utilities for camera parameters."""

import torch


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
