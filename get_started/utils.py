"""Utils for get_started scripts."""

from __future__ import annotations

import os

import imageio.v2 as iio
import numpy as np
import torch
from loguru import logger as log
from numpy.typing import NDArray
from torchvision.utils import make_grid, save_image

from metasim.types import EnvState


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, states: list[EnvState]):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        rgb_data_list = []
        for state in states:
            for camera_name, camera_data in state["cameras"].items():
                if "rgb" in camera_data:
                    rgb_data_list.append(camera_data["rgb"])
                    continue

        if not rgb_data_list:
            return

        rgb_data = torch.stack(rgb_data_list, dim=0)
        image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)
