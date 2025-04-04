## Modified from IsaacLab/_isaac_sim/exts/omni.isaac.core/omni/isaac/core/objects/cuboid.py

# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, Sequence

import numpy as np
import omni.kit.commands
from pxr import UsdGeom

try:
    from omni.isaac.core.materials.physics_material import PhysicsMaterial
    from omni.isaac.core.materials.preview_surface import PreviewSurface
    from omni.isaac.core.materials.visual_material import VisualMaterial
    from omni.isaac.core.prims import GeometryPrim, RigidPrim
    from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
    from omni.isaac.core.utils.string import find_unique_string_name
except ModuleNotFoundError:
    from isaacsim.core.api.materials.physics_material import PhysicsMaterial
    from isaacsim.core.api.materials.preview_surface import PreviewSurface
    from isaacsim.core.api.materials.visual_material import VisualMaterial
    from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
    from isaacsim.core.utils.string import find_unique_string_name


class VisualCuboid(GeometryPrim):
    """High level wrapper to create/encapsulate a visual cuboid

    .. note::

        Visual cuboids (Cube shape) have no collisions (Collider API) or rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "visual_cube".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        size (Optional[float], optional): length of each cube edge. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.

    Example:

    .. code-block:: python

        >>> from omni.isaac.core.objects import VisualCuboid
        >>> import numpy as np
        >>>
        >>> # create a red visual cube at the given path
        >>> prim = VisualCuboid(prim_path="/World/Xform/Cube", color=np.array([1.0, 0.0, 0.0]))
        >>> prim
        <omni.isaac.core.objects.cuboid.VisualCuboid object at 0x7f12e756fa00>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "visual_cube",
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
    ) -> None:
        if is_prim_path_valid(prim_path):
            prim = get_prim_at_path(prim_path)
            if not prim.IsA(UsdGeom.Cube):
                raise Exception(f"The prim at path {prim_path} cannot be parsed as a Cube object")
            cubeGeom = UsdGeom.Cube(prim)
        else:
            result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
            omni.kit.commands.execute("MovePrim", path_from=path, path_to=prim_path)
            if size is None:
                size = 1.0
            if visible is None:
                visible = True
            if visual_material is None:
                if color is None:
                    color = np.array([0.5, 0.5, 0.5])
                visual_prim_path = find_unique_string_name(
                    initial_name="/World/Looks/visual_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
                visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
        GeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            collision=False,
        )
        if visual_material is not None:
            VisualCuboid.apply_visual_material(self, visual_material)
        if size is not None:
            VisualCuboid.set_size(self, size)
        size = VisualCuboid.get_size(self)
        extent_size = size * VisualCuboid.get_local_scale(self)
        return

    def set_size(self, size: float) -> None:
        """Set the length of each cube edge

        Args:
            size (float): edge length

        Example:

        .. code-block:: python

            >>> prim.set_size(2.0)
        """
        self.size = size
        return

    def get_size(self) -> np.ndarray:
        """Get the length of each cube edge

        Returns:
            float: edge length

        Example:

        .. code-block:: python

            >>> prim.get_size()
            1.0
        """
        return self.size


class FixedCuboid(VisualCuboid):
    """High level wrapper to create/encapsulate a fixed cuboid

    .. note::

        Fixed cuboids (Cube shape) have collisions (Collider API) but no rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "fixed_cube".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        size (Optional[float], optional): length of each cube edge. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.
        physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
                                Defaults to None. If not specified, a default physics material will be added.

    Example:

    .. code-block:: python

        >>> from omni.isaac.core.objects import FixedCuboid
        >>> import numpy as np
        >>>
        >>> # create a red fixed cube at the given path
        >>> prim = FixedCuboid(prim_path="/World/Xform/Cube", color=np.array([1.0, 0.0, 0.0]))
        >>> prim
        <omni.isaac.core.objects.cuboid.FixedCuboid object at 0x7f7b4d91da80>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "fixed_cube",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        set_offsets = False
        if not is_prim_path_valid(prim_path):
            # set default values if no physics material given
            if physics_material is None:
                static_friction = 0.2
                dynamic_friction = 1.0
                restitution = 0.0
                physics_material_path = find_unique_string_name(
                    initial_name="/World/Physics_Materials/physics_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                physics_material = PhysicsMaterial(
                    prim_path=physics_material_path,
                    dynamic_friction=dynamic_friction,
                    static_friction=static_friction,
                    restitution=restitution,
                )
            set_offsets = True
        VisualCuboid.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
        )
        GeometryPrim.set_collision_enabled(self, True)
        if physics_material is not None:
            FixedCuboid.apply_physics_material(self, physics_material)
        if set_offsets:
            FixedCuboid.set_rest_offset(self, 0.0)
            FixedCuboid.set_contact_offset(self, 0.1)
            FixedCuboid.set_torsional_patch_radius(self, 1.0)
            FixedCuboid.set_min_torsional_patch_radius(self, 0.8)
        return


class DynamicCuboid(RigidPrim, FixedCuboid):
    """High level wrapper to create/encapsulate a dynamic cuboid

    .. note::

        Dynamic cuboids (Cube shape) have collisions (Collider API) and rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "fixed_cube".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        size (Optional[float], optional): length of each cube edge. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.
        physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
                                Defaults to None. If not specified, a default physics material will be added.
        mass (Optional[float], optional): mass in kg. Defaults to None.
        density (Optional[float], optional): density. Defaults to None.
        linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
        angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.

    Example:

    .. code-block:: python

        >>> from omni.isaac.core.objects import DynamicCuboid
        >>> import numpy as np
        >>>
        >>> # create a red dynamic cube of mass 1kg at the given path
        >>> prim = DynamicCuboid(prim_path="/World/Xform/Cube", color=np.array([1.0, 0.0, 0.0]), mass=1.0)
        >>> prim
        <omni.isaac.core.objects.cuboid.DynamicCuboid object at 0x7ff14c04d990>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "dynamic_cube",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        linear_velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
    ) -> None:
        if not is_prim_path_valid(prim_path):
            if mass is None:
                mass = 0.02
        FixedCuboid.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            size=size,
            visual_material=visual_material,
            physics_material=physics_material,
        )
        RigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            mass=mass,
            density=density,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
