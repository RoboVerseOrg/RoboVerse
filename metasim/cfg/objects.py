"""Configuration classes for various types of objects."""

from __future__ import annotations

import math
from dataclasses import MISSING

from metasim.constants import PhysicStateType
from metasim.utils import configclass

##################################################
# Mixins: File-based or Primitive
##################################################


@configclass
class FileBasedMixin:
    """File-based mixin."""

    mesh_path: str | None = None
    """Path to the mesh file."""

    usd_path: str | None = None
    """Path to the USD file."""

    urdf_path: str | None = None
    """Path to the URDF file."""

    mjcf_path: str | None = None
    """Path to the MJCF file."""

    mjx_mjcf_path: str | None = None
    """Path to the MJCF file only used for MJX. If not specified, it will be the same as mjcf_path."""

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


@configclass
class PrimitiveMixin:
    """Primitive mixin."""

    mass: float = 0.1
    """Mass of the object (in kg), default is 0.1 kg"""

    color: list[float] = MISSING
    """Color of the object in RGB"""

    @property
    def volume(self) -> float:
        """Volume of the object."""
        raise NotImplementedError

    @property
    def density(self) -> float:
        """Density of the object."""
        return self.mass / self.volume


##################################################
# Level 0: Base
##################################################


@configclass
class BaseObjCfg:
    """Base class for object cfg."""

    name: str = MISSING
    """Object name"""

    default_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Default position of the object, default is (0.0, 0.0, 0.0)"""

    default_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    """Default orientation of the object, default is (1.0, 0.0, 0.0, 0.0)"""

    fix_base_link: bool = False
    """Whether to fix the base link of the object, default is False"""


##################################################
# Level 1: Base rigid object and base articulation object
##################################################


@configclass
class BaseRigidObjCfg(BaseObjCfg):
    """Base rigid object cfg."""

    collision_enabled: bool = True
    """Whether to enable collision."""

    physics: PhysicStateType | None = None
    """IsaacSim's convention for collision and gravity state. Default to None. If specified, it will be translated to :attr:`collision_enabled` and :attr:`fix_base_link`."""

    def __post_init__(self):
        super().__post_init__()

        ## Parse physics to collision_enabled and fix_base_link.
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
class BaseArticulationObjCfg(BaseObjCfg):
    """Base articulation object cfg."""


##################################################
# Level 2: Concrete object
##################################################


@configclass
class RigidObjCfg(FileBasedMixin, BaseRigidObjCfg):
    """Rigid object cfg."""


@configclass
class ArticulationObjCfg(FileBasedMixin, BaseArticulationObjCfg):
    """Articulation object cfg."""


@configclass
class PrimitiveCubeCfg(PrimitiveMixin, BaseRigidObjCfg):
    """Primitive cube object cfg."""

    size: list[float] = MISSING
    """Size of the object (in m)."""

    @property
    def half_size(self) -> list[float]:
        """Half of the extend, for SAPIEN usage."""
        return [size / 2 for size in self.size]

    @property
    def volume(self) -> float:
        """Volume of the cube."""
        return self.size[0] * self.size[1] * self.size[2]


@configclass
class PrimitiveSphereCfg(PrimitiveMixin, BaseRigidObjCfg):
    """Primitive sphere object cfg."""

    radius: float = MISSING
    """Radius of the sphere (in m)."""

    @property
    def volume(self) -> float:
        """Volume of the sphere."""
        return 4 / 3 * math.pi * self.radius**3


@configclass
class PrimitiveCylinderCfg(PrimitiveMixin, BaseRigidObjCfg):
    """Primitive cylinder object cfg."""

    radius: float = MISSING
    """Radius of the cylinder (in m)."""

    height: float = MISSING
    """Height of the cylinder (in m)."""

    @property
    def volume(self) -> float:
        """Volume of the cylinder."""
        return math.pi * self.radius**2 * self.height


##################################################
# Other objects
##################################################


@configclass
class PrimitiveFrameCfg(RigidObjCfg):
    """Primitive coordinate frame cfg.

    .. warning::
        This class is experimental and subject to change.
    """

    # TODO: This is object shouldn't inherit from RigidObjCfg?
    base_link: str | tuple[str, str] | None = None
    """Base link to attach the frame.
        If ``None``, the frame will be attached to the world origin.
        If a ``str``, the frame will be attached to the root link of the object specified by the name.
        If a ``tuple[str, str]``, the frame will be attached to the object specified by the first str and the body link specified by the second str.
    """


@configclass
class NonConvexRigidObjCfg(RigidObjCfg):
    """Non-convex rigid object class.

    .. warning::
        This class is deprecated and will be removed in the future.
    """

    # TODO: remove this
    mesh_pose: list[float] = MISSING
