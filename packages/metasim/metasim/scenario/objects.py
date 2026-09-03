"""Configuration classes for various types of objects."""

from __future__ import annotations

import math
from dataclasses import MISSING

from metasim.constants import PhysicStateType
from metasim.utils import configclass

##################################################
# Mixins: File-based or Primitive
##################################################


# Default file_type dictionary that can be accessed at class level
_DEFAULT_FILE_TYPE = {
    "isaaclab": "usd",
    "isaacsim": "usd",
    "blender": "usd",
    "pybullet": "urdf",
    "sapien2": "urdf",
    "sapien3": "urdf",
    "genesis": "urdf",
    "isaacgym": "urdf",
    "mujoco": "mjcf",
    "mjx": "mjx_mjcf",
    "newton": "urdf",
    "superdex": "urdf",
}


@configclass
class _FileBasedMixin:
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

    # Instance variable for file_type (will be initialized from class variable if not provided)
    file_type: dict[str, str] = _DEFAULT_FILE_TYPE.copy()
    """Instance variable for file_type mapping. Defaults to class variable value."""

    # isaacgym_read_mjcf: bool = False
    # """By default, Isaac Gym will read from URDF files. If this is set to True, Isaac Gym will read from MJCF files."""

    # genesis_read_mjcf: bool = False
    # """By default, Genesis will read from URDF files. If this is set to True, Genesis will read from MJCF files."""

    extra_resources: list[str] = []
    """Extra resources to load for the object. This is used to load additional resources for the object, such as textures, materials, etc."""

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()

        ## Set the mjx_mjcf_path if it is not specified.
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path

    def file_name(self, sim_name):
        file_type = self.file_type[sim_name]
        if file_type == "usd":
            if self.usd_path is not None:
                return self.usd_path
            if sim_name in {"isaacsim", "isaaclab"}:
                return self.urdf_path
            return None
        elif file_type == "urdf":
            if self.urdf_path is not None:
                return self.urdf_path
            # sapien2 also natively loads GLB scenes (NonConvexRigidObjCfg.usd_path) and mesh-based
            # rigid objects (RigidObjCfg.mesh_path); resolve to what the handler actually loads.
            if sim_name == "sapien2":
                return self.usd_path or getattr(self, "mesh_path", None)
            return self.urdf_path
        elif file_type == "mjcf":
            return self.mjcf_path
        elif file_type == "mjx_mjcf":
            return self.mjx_mjcf_path
        elif file_type == "mesh":
            return self.mesh_path
        else:
            raise ValueError(f"Invalid file type: {file_type}")


@configclass
class _PrimitiveMixin:
    """Primitive mixin."""

    mass: float = 0.1
    """Mass of the object (in kg), default is 0.1 kg"""

    color: list[float] = MISSING
    """Color of the object in RGB"""

    def __post_init__(self):
        super().__post_init__()

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

    fix_base_link: bool | None = None
    """Whether to fix the base link of the object. If None, will be inferred from physics parameter. Default is None."""

    enabled_gravity: bool = True
    """Whether to enable gravity. Default to True. If False, the robot will not be affected by gravity."""

    scale: float | tuple[float, float, float] = 1.0
    """Object scaling (in scalar) for the object, default is 1.0"""

    def __post_init__(self):
        # Set default value for fix_base_link if not explicitly set
        if self.fix_base_link is None:
            self.fix_base_link = False

        if isinstance(self.scale, float):
            self.scale = (self.scale, self.scale, self.scale)


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

    # Optional rigid-body physical properties (opt-in; default None = backend default). Used by
    # mesh-based rigid objects in backends that support per-object material/density (sapien2).
    mesh_density: float | None = None
    """Uniform density (kg/m^3) used to derive mass/inertia from the collision mesh of a mesh-based
    rigid object. Named ``mesh_density`` to avoid clashing with the primitive ``density`` property.
    Default None."""
    collision_mesh_path: str | None = None
    """Separate collision mesh for mesh-based rigid objects whose collision geometry differs from the
    visual (e.g. SimplerEnv objects: collision.obj + textured.dae). Falls back to the visual mesh."""
    static_friction: float | None = None
    dynamic_friction: float | None = None
    restitution: float | None = None
    linear_damping: float | None = None
    angular_damping: float | None = None

    def __post_init__(self):
        # Parse physics parameter first (if provided)
        if self.physics is not None:
            if self.physics == PhysicStateType.XFORM:
                self.collision_enabled = False
                # Only set fix_base_link from physics if not explicitly set by user
                if self.fix_base_link is None:
                    self.fix_base_link = True
            elif self.physics == PhysicStateType.GEOM:
                self.collision_enabled = True
                if self.fix_base_link is None:
                    self.fix_base_link = True
            elif self.physics == PhysicStateType.RIGIDBODY:
                self.collision_enabled = True
                if self.fix_base_link is None:
                    self.fix_base_link = False
            else:
                raise ValueError(f"Invalid physics type: {self.physics}")

        # Call parent __post_init__ to set default value if still None
        super().__post_init__()


@configclass
class BaseArticulationObjCfg(BaseObjCfg):
    """Base articulation object cfg."""

    def __post_init__(self):
        super().__post_init__()


##################################################
# Level 2: Concrete object
##################################################


@configclass
class RigidObjCfg(_FileBasedMixin, BaseRigidObjCfg):
    """Rigid object cfg."""

    collapse_fixed_joints: bool = False
    """Whether to collapse fixed joints when loading the object. Default is False."""


@configclass
class ArticulationObjCfg(_FileBasedMixin, BaseArticulationObjCfg):
    """Articulation object cfg."""


# Set file_type as a class variable for ArticulationObjCfg and its subclasses
# This allows accessing it as RobotCfg.file_type
_FileBasedMixin.file_type = _DEFAULT_FILE_TYPE.copy()
ArticulationObjCfg.file_type = _DEFAULT_FILE_TYPE.copy()


@configclass
class PrimitiveCubeCfg(_PrimitiveMixin, BaseRigidObjCfg):
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
class PrimitiveSphereCfg(_PrimitiveMixin, BaseRigidObjCfg):
    """Primitive sphere object cfg."""

    radius: float = MISSING
    """Radius of the sphere (in m)."""

    @property
    def volume(self) -> float:
        """Volume of the sphere."""
        return 4 / 3 * math.pi * self.radius**3


@configclass
class PrimitiveCylinderCfg(_PrimitiveMixin, BaseRigidObjCfg):
    """Primitive cylinder object cfg."""

    radius: float = MISSING
    """Radius of the cylinder (in m)."""

    height: float = MISSING
    """Height of the cylinder (in m)."""

    @property
    def volume(self) -> float:
        """Volume of the cylinder."""
        return math.pi * self.radius**2 * self.height


@configclass
class PrimitiveMultiBoxCfg(BaseRigidObjCfg):
    """A single rigid body composed of several box collision shapes at local offsets.

    For compound primitive geometry that a single box can't express — an L-shaped tool, a
    box-with-a-hole frame, a multi-prong plug — without needing a mesh asset. Each entry of
    :attr:`boxes` is ``{"half_size": [hx, hy, hz], "pos": [x, y, z], "quat": [w, x, y, z]}``
    (``pos``/``quat`` default to the body origin / identity). ``mass`` sets the total mass and is
    distributed as a uniform density across the boxes so the inertia is consistent.

    Backend support: SAPIEN 3. Other backends that don't implement it skip the object (same as any
    unsupported asset). Default ``fix_base_link=False`` builds a dynamic body; ``True`` builds it
    kinematic (e.g. a fixed frame).
    """

    boxes: list = MISSING
    """List of box shape dicts: ``{"half_size": [3], "pos": [3] (opt), "quat": [4] wxyz (opt)}``."""

    mass: float = 0.1
    """Total mass (kg); spread as uniform density over the boxes."""

    color: list[float] = MISSING
    """Visual colour in RGB."""

    @property
    def total_volume(self) -> float:
        """Sum of the box volumes (m^3)."""
        v = 0.0
        for b in self.boxes:
            hs = b["half_size"]
            v += 8.0 * hs[0] * hs[1] * hs[2]
        return v

    @property
    def density(self) -> float:
        """Uniform density (kg/m^3) implied by the total mass and box volumes."""
        vol = self.total_volume
        return self.mass / vol if vol > 0 else 0.0


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
