# Objects Config

Objects are the basics of a simulation. Every simulated entity is represented as an object. Different objects have different properties. We abstract all the properties into the following hierarchy of inheritance:

```python
@configclass
class BaseObjCfg:
    """Base class for object cfg."""

    name: str = MISSING
    """Object name"""
    fix_base_link: bool = False
    """Whether to fix the base link of the object, default is False"""
    scale: float | tuple[float, float, float] = 1.0
    """Object scaling (in scalar) for the object, default is 1.0"""

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
    physics: PhysicStateType = MISSING


@configclass
class NonConvexRigidObjCfg(RigidObjCfg):
    """Non-convex rigid object class, used by some simulators."""

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
    """Primitive cube object cfg.

    This class specifies configuration parameters of a primitive cube.

    Attributes:
        mass: Mass of the object, in kg, default is 0.1
        color: Color of the object in RGB
        size: Size of the object, extent in m
    """

    mass: float = 0.1
    color: list[float] = MISSING
    size: list[float] = MISSING
    physics: PhysicStateType = MISSING
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

```

For better understanding, we can define a cube as follows:

```python
PrimitiveCubeCfg(
    name="block_red",
    mass=0.1,
    size=(0.07, 0.05, 0.05),
    color=(1.0, 0.0, 0.0),
    physics=PhysicStateType.RIGIDBODY,
    scale=0.8,
)
```

As you can see, some fields are not placed reasonably, which is still a focus of our development. This means, our configuration system may need some ground-breaking changes, get prepared and stay tuned!
