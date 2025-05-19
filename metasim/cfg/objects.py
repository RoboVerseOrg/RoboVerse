"""Configuration classes for various types of objects used in the simulation.

Configurations include the object name, source file, geometries (scaling, radius, etc.),
and physics (mass, density, etc.).
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from metasim.constants import PhysicStateType
from metasim.utils import configclass


@configclass
class BaseObjCfg:
    """Base class for object cfg."""

    name: str = MISSING
    """Object name"""
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Default position of the object, default is (0.0, 0.0, 0.0)"""
    default_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    """Default orientation of the object, default is (1.0, 0.0, 0.0, 0.0)"""


# File-based object
@configclass
class RigidObjCfg(BaseObjCfg):
    """Rigid object cfg.

    The file source should be specified, from a USD, URDF, MJCF, or mesh file,
    with the path specified by the members below.

    Attributes:
        usd_path: Path to the USD file
        urdf_path: Path to the URDF file
        mjcf_path: Path to the MJCF file
        mesh_path: Path to the Mesh file
        physics: Specify the physics APIs applied on the object
    """

    usd_path: str | None = None
    urdf_path: str | None = None
    mjcf_path: str | None = None
    mjx_mjcf_path: str | None = None
    """Path to the MJCF file only used for MJX. If not specified, it will be the same as mjcf_path."""
    mesh_path: str | None = None
    collision_enabled: bool = True
    physics: PhysicStateType | None = None
    fix_base_link: bool = False
    """Whether to fix the base link of the object, default is False"""
    scale: float | tuple[float, float, float] = 1.0
    """Object scaling (in scalar) for the object, default is 1.0"""

    def __post_init__(self):
        super().__post_init__()

        ## Parse the physics state to the enabled and fix_base_link.
        if self.physics is not None:
            if self.physics == PhysicStateType.XFORM:
                self.collision_enabled = False
                self.fix_base_link = True
            elif self.physics == PhysicStateType.GEOM:
                self.collision_enabled = True
                self.fix_base_link = True
            elif self.physics == PhysicStateType.RIGIDBODY:
                self.collision_enabled = True
                self.fix_base_link = False
            else:
                raise ValueError(f"Invalid physics type: {self.physics}")

        ## Transform the 1d scale to a tuple of (x-scale, y-scale, z-scale).
        if isinstance(self.scale, float):
            self.scale = (self.scale, self.scale, self.scale)

        ## Set the mjx_mjcf_path if it is not specified.
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path


@configclass
class NonConvexRigidObjCfg(RigidObjCfg):
    """Non-convex rigid object class."""

    mesh_pose: list[float] = MISSING


@configclass
class ArticulationObjCfg(BaseObjCfg):
    """Articulation object cfg."""

    usd_path: str | None = None
    urdf_path: str | None = None
    mjcf_path: str | None = None
    mjx_mjcf_path: str | None = None
    """Path to the MJCF file only used for MJX. If not specified, it will be the same as mjcf_path."""
    fix_base_link: bool = False
    """Whether to fix the base link of the object, default is False"""
    scale: float | tuple[float, float, float] = 1.0
    """Object scaling (in scalar) for the object, default is 1.0"""

    def __post_init__(self):
        super().__post_init__()

        ## Transform the 1d scale to a tuple of (x-scale, y-scale, z-scale).
        if isinstance(self.scale, float):
            self.scale = (self.scale, self.scale, self.scale)

        ## Set the mjx_mjcf_path if it is not specified.
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path


# Primitive object are all rigid objects
@configclass
class PrimitiveCubeCfg(RigidObjCfg):
    """Primitive cube object cfg."""

    mass: float = 0.1
    """Mass of the object (in kg), default is 0.1 kg"""
    color: list[float] = MISSING
    """Color of the object in RGB"""
    size: list[float] = MISSING
    """Size of the object (in m)"""
    physics: PhysicStateType = MISSING
    """Physics state of the object"""
    mjcf_path: str | None = None  # TODO: remove this field

    @property
    def half_size(self) -> list[float]:
        """Half of the extend, for SAPIEN usage."""
        return [size / 2 for size in self.size]

    @property
    def density(self) -> float:
        """Object density, for SAPIEN usage."""
        return self.mass / (self.size[0] * self.size[1] * self.size[2])


@configclass
class PrimitiveSphereCfg(RigidObjCfg):
    """Primitive sphere object cfg."""

    mass: float = 0.1
    color: list[float] = MISSING
    radius: float = MISSING
    physics: PhysicStateType = MISSING

    @property
    def density(self) -> float:
        """For SAPIEN usage."""
        return self.mass / (4 / 3 * math.pi * self.radius**3)


@configclass
class PrimitiveFrameCfg(RigidObjCfg):
    """Primitive coordinate frame cfg."""

    # TODO: This is object shouldn't inherit from RigidObjCfg?
    name: str = MISSING
    scale: float = 1.0
    """Scale of the frame"""
    base_link: str | tuple[str, str] | None = None
    """Base link to attach the frame.
        If ``None``, the frame will be attached to the world origin.
        If a ``str``, the frame will be attached to the root link of the object specified by the name.
        If a ``tuple[str, str]``, the frame will be attached to the object specified by the first str and the body link specified by the second str.
    """


@configclass
class PrimitiveCylinderCfg(RigidObjCfg):
    """Primitive cylinder object cfg."""

    mass: float = 0.1
    color: list[float] = MISSING
    radius: float = MISSING
    height: float = MISSING
    physics: PhysicStateType = MISSING

    @property
    def density(self) -> float:
        """For SAPIEN usage."""
        return self.mass / (math.pi * self.radius**2 * self.height)
