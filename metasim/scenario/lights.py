"""Configuration classes for lights used in the simulation."""

from __future__ import annotations

import math

import torch
from loguru import logger as log

from metasim.utils import configclass
from metasim.utils.math import quat_from_euler_xyz


@configclass
class BaseLightCfg:
    """Base configuration for a light."""

    intensity: float = 500.0
    """Intensity of the light"""
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Color of the light"""
    is_global: bool = False
    """Whether the light is a global light that is not copied to each environment"""


@configclass
class DistantLightCfg(BaseLightCfg):
    """Configuration for a distant light. The default direction is (0, 0, -1), pointing towards Z- direction."""

    polar: float = 0.0
    """Polar angle of the light (in degrees). Default is 0, which means the light is pointing towards Z- direction."""
    azimuth: float = 0.0
    """Azimuth angle of the light (in degrees). Default is 0."""
    is_global: bool = True
    """Whether the light is a global light that is not copied to each environment. For distant light, it must be global."""

    @property
    def quat(self) -> tuple[float, float, float, float]:
        """Quaternion of the light direction. (1, 0, 0, 0)a means the light is pointing towards Z- direction."""
        roll = torch.tensor(self.polar / 180.0 * math.pi)
        pitch = torch.tensor(0.0)
        yaw = torch.tensor(self.azimuth / 180.0 * math.pi)
        return tuple(quat_from_euler_xyz(roll, pitch, yaw).squeeze(0).tolist())

    def __post_init__(self):
        """Post-initialization hook to check if the light is global."""
        if not self.is_global:
            log.warning("Distant light must be global, overriding the value.")
            self.is_global = True


@configclass
class CylinderLightCfg(BaseLightCfg):
    """Configuration for a cylinder light."""

    length: float = 1.0
    """Length of the cylinder (in m). Default is 1.0m."""
    radius: float = 0.5
    """Radius of the cylinder (in m). Default is 0.5m."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the cylinder (in m). Default is (0.0, 0.0, 0.0)."""
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation of the cylinder. Default is (1.0, 0.0, 0.0, 0.0)."""
