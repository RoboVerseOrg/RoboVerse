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
    fix_base_link: bool = False
    """Whether to fix the base link of the object, default is False"""
    scale: float | tuple[float, float, float] = 1.0
    """Object scaling (in scalar) for the object, default is 1.0"""
    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Default position of the object, default is (0.0, 0.0, 0.0)"""
    default_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    """Default orientation of the object, default is (1.0, 0.0, 0.0, 0.0)"""

    def __post_init__(self):
        """Transform the 1d scale to a tuple of (x-scale, y-scale, z-scale)."""
        if isinstance(self.scale, float):
            self.scale = (self.scale, self.scale, self.scale)


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
    mesh_path: str | None = None
    collision_enabled: bool = True
    physics: PhysicStateType | None = None

    def __post_init__(self):
        """Parse the physics state to the enabled and fix_base_link."""
        super().__post_init__()
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
