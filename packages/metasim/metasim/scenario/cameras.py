"""Sub-module containing the camera configuration."""

from __future__ import annotations

import math
from typing import Literal

from metasim.utils.configclass import configclass


@configclass
class BaseCameraCfg:
    """Base camera configuration."""

    name: str = "camera0"
    """Name of the camera. Defaults to "camera0". Different cameras should have different names, so if you add multiple cameras, make sure to give them unique names."""
    data_types: list[Literal["rgb", "depth", "instance_seg", "instance_id_seg"]] = ["rgb", "depth"]
    """List of sensor types to enable for the camera. Defaults to ["rgb", "depth"]."""
    width: int = 256
    """Width of the image in pixels. Defaults to 256."""
    height: int = 256
    """Height of the image in pixels. Defaults to 256."""
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Position of the camera in the world frame. Defaults to (0.0, 0.0, 1.0)."""
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look at point of the camera in the world frame. Defaults to (0.0, 0.0, 0.0)."""
    mount_to: str | tuple[str, str] | None = None
    """Mount the camera to a specific object or robot. Defaults to None."""
    mount_link: str | tuple[str, str] | None = None
    """Specify the link to mount the camera to. Defaults to None."""
    mount_pos: tuple[float, float, float] | None = None
    """Position of the camera on the mount. Defaults to None."""
    mount_quat: tuple[float, float, float, float] | None = None
    """Quaternion of the camera on the mount. Defaults to None."""
    intrinsic: list[list[float]] | None = None
    """Explicit 3x3 pinhole intrinsic [[fx,0,cx],[0,fy,cy],[0,0,1]]. When set, backends that
    support it (sapien2) apply the exact focal lengths / principal point instead of deriving them
    from fov/aperture. Defaults to None (derive from fov)."""

    def __post_init__(self) -> None:
        """A camera needs a positive pixel size and a finite position.

        A zero-sized buffer or a NaN pose fails deep inside the renderer with no reference to the
        camera that caused it.
        """
        from ._validate import finite_triple, positive_finite_or_none, positive_int

        owner = f"{type(self).__name__}({self.name!r})"
        positive_int(owner, "width", self.width)
        positive_int(owner, "height", self.height)
        finite_triple(owner, "pos", self.pos)
        # the intrinsics below divide by these; a zero used to surface as a bare ZeroDivisionError
        for field_name in ("focal_length", "horizontal_aperture"):
            if hasattr(self, field_name):
                positive_finite_or_none(owner, field_name, getattr(self, field_name))
        if getattr(self, "look_at", None) is not None:
            finite_triple(owner, "look_at", self.look_at)


@configclass
class PinholeCameraCfg(BaseCameraCfg):
    """Pinhole camera configuration."""

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0 cm."""
    focus_distance: float = 400.0
    """Distance from the camera to the focus plane (in m). Defaults to 400.0."""
    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm."""
    clipping_range: tuple[float, float] = (0.05, 1e5)
    """Near and far clipping distances (in m). Defaults to (0.05, 1e5)."""

    @property
    def vertical_aperture(self) -> float:
        """Vertical aperture (in cm)."""
        return self.horizontal_aperture * self.height / self.width

    @property
    def horizontal_fov(self) -> float:
        """Horizontal field of view, in degrees."""
        return 2 * math.atan(self.horizontal_aperture / (2 * self.focal_length)) / math.pi * 180

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view, in degrees."""
        return 2 * math.atan(self.vertical_aperture / (2 * self.focal_length)) / math.pi * 180

    @property
    def intrinsics(self) -> list[list[float]]:
        """Intrinsics matrix of the camera. Type is 3x3 nested list of floats."""
        fx = self.width * self.focal_length / self.horizontal_aperture
        fy = self.height * self.focal_length / self.vertical_aperture
        cx = self.width * 0.5
        cy = self.height * 0.5
        return [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]


REALSENSE_CAMERA = PinholeCameraCfg(
    name="realsense_camera",
    data_types=["rgb", "depth"],
    width=640,
    height=360,
    focal_length=1.88,
    horizontal_aperture=float(2 * 1.88 * math.tan(71.28 / 180 * math.pi / 2)),
)
