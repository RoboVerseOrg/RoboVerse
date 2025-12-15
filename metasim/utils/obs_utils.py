"""Utils for get_started scripts."""

from __future__ import annotations

import os

import imageio.v2 as iio
import numpy as np
import torch
from loguru import logger as log
from numpy.typing import NDArray
from torchvision.utils import make_grid, save_image

from metasim.utils.state import TensorState

try:
    import cv2
except ImportError:
    cv2 = None


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = torch.concat([cam.rgb for cam in state.cameras.values()], dim=2)  # horizontal concat
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self, fps: int = 30):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=fps)


try:
    import open3d as o3d
except ImportError:
    pass


def get_depth_from_normalized(depth_normalized, depth_min, depth_max):
    """Get the depth from the normalized depth."""
    assert depth_normalized.min() >= 0.0
    assert depth_normalized.max() <= 1.0
    depth = depth_normalized * (depth_max - depth_min) + depth_min
    return depth


def get_pcd_from_rgbd(depth, rgb_img, cam_intr_mat, cam_extr_mat):
    """Get the point cloud from the RGBD image."""
    if type(cam_intr_mat) is not np.ndarray:
        cam_intr_mat = np.array(cam_intr_mat)
    if type(cam_extr_mat) is not np.ndarray:
        cam_extr_mat = np.array(cam_extr_mat)

    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_img).astype(np.uint8))
    rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )

    cam_intr = o3d.camera.PinholeCameraIntrinsic(
        width=depth.shape[1],
        height=depth.shape[0],
        fx=cam_intr_mat[0, 0],
        fy=cam_intr_mat[1, 1],
        cx=cam_intr_mat[0, 2],
        cy=cam_intr_mat[1, 2],
    )
    cam_extr = np.array(cam_extr_mat)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_o3d,
        cam_intr,
        cam_extr,
    )

    return pcd


@torch.jit.script
def depth_image_to_point_cloud_GPU(
    camera_tensor,
    camera_view_matrix_inv,
    camera_proj_matrix,
    u,
    v,
    width: float,
    height: float,
    depth_bar: float,
    device: torch.device,
):
    """Convert a depth image to a point cloud using the camera parameters."""
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points


def convert_to_ply(points, filename):
    """Convert a point cloud (NumPy array or PyTorch tensor) to a PLY file using Open3D.

    :param points: NumPy array or PyTorch tensor of shape (N, 3) or (N, 6)
                where N is the number of points.
    :param filename: Name of the output PLY file.
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points. Assuming the first 3 columns are x, y, z coordinates
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # If the points array has 6 columns, assume the last 3 are RGB values
    if points.shape[1] == 6:
        # Normalize color values to [0, 1] if they are not already
        colors = points[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Write to a PLY file
    o3d.io.write_point_cloud(filename, pcd)
    log.info(f"Point cloud saved to '{filename}'.")


def display_obs(obs, width: int, height: int, window_name: str = "Camera View - Real-time Robot View") -> bool:
    """Display camera observations using OpenCV - split screen for multiple cameras.

    Args:
        obs: Observation object with cameras attribute containing camera data
        width: Display window width
        height: Display window height
        window_name: Name of the OpenCV window

    Returns:
        bool: True if should continue running, False if ESC key was pressed
    """
    if cv2 is None:
        log.warning("OpenCV not available for camera display")
        return True

    if not hasattr(obs, "cameras") or len(obs.cameras) == 0:
        # Create a blank dark gray image if no camera data
        blank_img = np.full((height, width, 3), 50, dtype=np.uint8)
        cv2.imshow(window_name, blank_img)
        return True

    camera_names = list(obs.cameras.keys())
    num_cameras = len(camera_names)

    # Create display image
    display_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Split the display area into two halves for dual camera view
    half_width = width // 2
    half_height = height

    # Display first camera on the left
    if num_cameras >= 1:
        camera_name_1 = camera_names[0]
        rgb_data_1 = obs.cameras[camera_name_1].rgb

        if rgb_data_1 is not None:
            # Convert to numpy array and handle different formats
            if isinstance(rgb_data_1, torch.Tensor):
                rgb_np_1 = rgb_data_1.cpu().numpy()
                if rgb_np_1.max() <= 1.0:
                    rgb_np_1 = (rgb_np_1 * 255).astype(np.uint8)
            else:
                rgb_np_1 = np.array(rgb_data_1)

            # Handle different shapes
            if len(rgb_np_1.shape) == 4:  # (N, C, H, W)
                rgb_np_1 = rgb_np_1[0]  # Take first environment
            if len(rgb_np_1.shape) == 3 and rgb_np_1.shape[0] == 3:  # (C, H, W)
                rgb_np_1 = np.transpose(rgb_np_1, (1, 2, 0))  # (H, W, C)

            try:
                # Resize image to fit the left half
                if rgb_np_1.shape[:2] != (half_height, half_width):
                    rgb_resized_1 = cv2.resize(rgb_np_1, (half_width, half_height))
                else:
                    rgb_resized_1 = rgb_np_1
                display_img[:, :half_width] = rgb_resized_1
            except Exception as e:
                log.warning(f"Error displaying camera 1 image: {e}")
                # Draw error rectangle on left half
                cv2.rectangle(display_img, (0, 0), (half_width, half_height), (50, 50, 100), -1)

    # Display second camera on the right
    if num_cameras >= 2:
        camera_name_2 = camera_names[1]
        rgb_data_2 = obs.cameras[camera_name_2].rgb

        if rgb_data_2 is not None:
            # Convert to numpy array and handle different formats
            if isinstance(rgb_data_2, torch.Tensor):
                rgb_np_2 = rgb_data_2.cpu().numpy()
                if rgb_np_2.max() <= 1.0:
                    rgb_np_2 = (rgb_np_2 * 255).astype(np.uint8)
            else:
                rgb_np_2 = np.array(rgb_data_2)

            # Handle different shapes
            if len(rgb_np_2.shape) == 4:  # (N, C, H, W)
                rgb_np_2 = rgb_np_2[0]  # Take first environment
            if len(rgb_np_2.shape) == 3 and rgb_np_2.shape[0] == 3:  # (C, H, W)
                rgb_np_2 = np.transpose(rgb_np_2, (1, 2, 0))  # (H, W, C)

            try:
                # Resize image to fit the right half
                if rgb_np_2.shape[:2] != (half_height, half_width):
                    rgb_resized_2 = cv2.resize(rgb_np_2, (half_width, half_height))
                else:
                    rgb_resized_2 = rgb_np_2
                display_img[:, half_width:] = rgb_resized_2
            except Exception as e:
                log.warning(f"Error displaying camera 2 image: {e}")
                # Draw error rectangle on right half
                cv2.rectangle(display_img, (half_width, 0), (width, half_height), (50, 50, 100), -1)

    # Fill areas if cameras are missing
    if num_cameras == 0:
        display_img.fill(50)  # Dark gray
    elif num_cameras == 1:
        # Fill right half with darker gray
        display_img[:, half_width:] = 30

    # Show the combined image
    cv2.imshow(window_name, display_img)

    # Handle key events for OpenCV window
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        return False  # Signal to exit
    return True  # Continue running
