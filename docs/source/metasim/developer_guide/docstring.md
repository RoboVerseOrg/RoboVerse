# Docstrings

When contributing to the RoboVerse project, the developers are recommended to include the docstrings for their classes, members, functions. The [Python PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/) provides typical conventions, and the [Google Python Style Guide on the Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) provides more details.

Docstrings should be added for:

- all modules, as well as the functions and classes exported by them
- public methods, including the `__init__` constructors
- `__init__.py`, describing the module that it's for

Please refer to the [Sphinx autodoc docs](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) for the format of writing the docstrings, so that they can be automatically included by Sphinx.

After finishing the docstrings for new modules, classes, or functions, you can add register the module within `docs/` following the [Sphinx autodoc docs](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) so that the docs can automatically generate the docs page for you.

## Hints

Here we provide several examples with explainations, so you can get started quickly.

### Modules

Every module file should begin with a module-level docstring describing its purpose. For example, for `metasim.cfg.objects`:


```python
"""Configuration classes for various types of objects used in the simulation.

Configurations include the object name, source file, geometries (scaling, radius, etc.),
and physics (mass, density, etc.).
"""

# imports ...
```

Also include license boilerplate at the top of the file.

### Classes

Use multi-line docstrings for classes. The first line should be a concise summary for autosummary in Sphinx. Follow with a blank line and a more detailed description.

Also, document each class attribute with a clear description, including units and default values where applicable. Inherited attributes need re-documentation only if their meaning or usage changes in the derived class.

```python
@configclass
class RigidObjMetaCfg(BaseObjMetaCfg):
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
class PrimitiveCubeMetaCfg(RigidObjMetaCfg):
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

    @property
    def half_size(self) -> list[float]:
        """Half of the extend, for SAPIEN usage."""
        return [size / 2 for size in self.size]

    @property
    def density(self) -> float:
        """Object density, for SAPIEN usage."""
        return self.mass / (self.size[0] * self.size[1] * self.size[2])

```


### Methods and Functions

Document methods and functions (including properties) with a clear description of their functionality, followed by explanations of arguments, return values, and exceptions. Leverage type hints in the method signature to reduce redundancy in docstrings.

```python
class BaseSimHandler
    # ...
    def get_joint_names(self, object: ArticulationObjMetaCfg) -> list[str]:
        """Get the joint names for a specified articulated object.

        Args:
            object (ArticulationObjMetaCfg): The desired articulated object.

        Returns:
            list[str]: A list of joint names (strings) including the joint names.

        Raises:
            AssertionError: The provided `object` is not an ArticulationObjMetaCfg.
        """
        assert isinstance(object, ArticulationObjMetaCfg)
        pass

    # ...
```
