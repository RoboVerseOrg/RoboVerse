"""Scene randomizer for domain randomization.

This module provides functionality to randomize scene geometry and surface materials,
including walls, floors, ceilings, and tabletops when no predefined scene exists.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass


@configclass
class SceneGeometryCfg:
    """Configuration for scene geometry elements.

    Args:
        enabled: Whether to create this geometry element
        size: Size of the geometry (x, y, z) in meters
        position: Position of the geometry (x, y, z) in meters
        material_randomization: Whether to randomize material for this element
    """

    enabled: bool = True
    size: tuple[float, float, float] = (1.0, 1.0, 0.1)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    material_randomization: bool = True


@configclass
class SceneMaterialPoolCfg:
    """Configuration for scene material pools.

    Args:
        material_paths: List of paths to material files (MDL)
        selection_strategy: How to select from available materials
        weights: Optional weights for weighted selection
    """

    material_paths: list[str] = dataclasses.field(default_factory=list)
    selection_strategy: Literal["random", "sequential", "weighted"] = "random"
    weights: list[float] | None = None

    def __post_init__(self):
        """Validate material pool configuration."""
        if self.selection_strategy == "weighted":
            if self.weights is None or len(self.weights) != len(self.material_paths):
                raise ValueError("weights must be provided and match material_paths length for weighted selection")


@configclass
class SceneRandomCfg:
    """Configuration for scene randomization.

    Args:
        floor: Floor geometry configuration
        walls: Wall geometry configuration (4 walls)
        ceiling: Ceiling geometry configuration
        table: Table/desktop geometry configuration
        floor_materials: Material pool for floor
        wall_materials: Material pool for walls
        ceiling_materials: Material pool for ceiling
        table_materials: Material pool for table/desktop
        only_if_no_scene: Only create scene elements if no predefined scene exists
        env_ids: List of environment IDs to apply randomization to (None = all)
    """

    floor: SceneGeometryCfg | None = None
    walls: SceneGeometryCfg | None = None
    ceiling: SceneGeometryCfg | None = None
    table: SceneGeometryCfg | None = None

    floor_materials: SceneMaterialPoolCfg | None = None
    wall_materials: SceneMaterialPoolCfg | None = None
    ceiling_materials: SceneMaterialPoolCfg | None = None
    table_materials: SceneMaterialPoolCfg | None = None

    only_if_no_scene: bool = True
    env_ids: list[int] | None = None

    def __post_init__(self):
        """Validate scene randomization configuration."""
        # Check if at least one element is enabled
        enabled_elements = [
            elem for elem in [self.floor, self.walls, self.ceiling, self.table] if elem is not None and elem.enabled
        ]
        if not enabled_elements:
            logger.warning("No scene elements enabled in SceneRandomCfg")


class SceneRandomizer(BaseRandomizerType):
    """Randomizer for scene geometry and materials.

    This randomizer creates and randomizes scene elements (floor, walls, ceiling, table)
    when no predefined scene exists. It samples materials from curated ARNOLD and
    vMaterials collections.

    Example:
        >>> cfg = SceneRandomCfg(
        ...     floor=SceneGeometryCfg(enabled=True, size=(10.0, 10.0, 0.1)),
        ...     floor_materials=SceneMaterialPoolCfg(material_paths=["..."])
        ... )
        >>> randomizer = SceneRandomizer(cfg, seed=42)
        >>> randomizer.bind_handler(handler)
        >>> randomizer()  # Apply randomization
    """

    def __init__(self, cfg: SceneRandomCfg, seed: int | None = None):
        """Initialize scene randomizer.

        Args:
            cfg: Scene randomization configuration
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.cfg = cfg
        self._seed = seed

        # Initialize random number generator
        if seed is not None:
            import random

            self._rng = random.Random(seed)
        else:
            import random

            self._rng = random.Random()

        # Initialize material selection state for sequential selection
        self._material_selection_state = {
            "floor_index": 0,
            "wall_index": 0,
            "ceiling_index": 0,
            "table_index": 0,
        }

        # Track created prims to avoid recreating
        self._created_prims = set()

        logger.debug(f"SceneRandomizer initialized with seed {self._seed}")

    def bind_handler(self, handler):
        """Bind the scene randomizer to a simulation handler.

        Args:
            handler: Simulation handler to bind to
        """
        super().bind_handler(handler)

        # Check if scene exists
        if self.cfg.only_if_no_scene:
            self._check_scene_exists()

    def _check_scene_exists(self) -> bool:
        """Check if a predefined scene exists.

        Returns:
            True if scene exists, False otherwise
        """
        # Check if handler has a scene object
        if hasattr(self.handler, "scenario") and hasattr(self.handler.scenario, "scene"):
            if self.handler.scenario.scene is not None:
                logger.info("Predefined scene detected, SceneRandomizer will skip geometry creation")
                return True
            else:
                logger.error("Predefined scene is None")
                return False
        return False

    def __call__(self, env_ids: list[int] | None = None):
        """Apply scene randomization.

        Args:
            env_ids: Optional list of environment IDs to randomize. If None, uses cfg.env_ids
        """
        if not self.handler:
            raise RuntimeError("Handler not bound. Call bind_handler() first.")

        # Skip if scene exists and only_if_no_scene is True
        if self.cfg.only_if_no_scene and self._check_scene_exists():
            # Still apply material randomization to existing elements
            self._randomize_materials_only(env_ids)
            return

        # Get environment IDs to randomize
        target_env_ids = env_ids if env_ids is not None else self.cfg.env_ids
        if target_env_ids is None:
            target_env_ids = list(range(self.handler.num_envs))

        # Create and randomize scene elements for each environment
        for env_id in target_env_ids:
            self._randomize_scene_for_env(env_id)

    def _randomize_scene_for_env(self, env_id: int):
        """Randomize scene for a specific environment.

        Args:
            env_id: Environment ID
        """
        env_prim_path = f"/World/envs/env_{env_id}"

        # Create/randomize floor
        if self.cfg.floor is not None and self.cfg.floor.enabled:
            self._create_or_update_floor(env_prim_path, env_id)

        # Create/randomize walls
        if self.cfg.walls is not None and self.cfg.walls.enabled:
            self._create_or_update_walls(env_prim_path, env_id)

        # Create/randomize ceiling
        if self.cfg.ceiling is not None and self.cfg.ceiling.enabled:
            self._create_or_update_ceiling(env_prim_path, env_id)

        # Create/randomize table
        if self.cfg.table is not None and self.cfg.table.enabled:
            self._create_or_update_table(env_prim_path, env_id)

    def _create_or_update_floor(self, env_prim_path: str, env_id: int):
        """Create or update floor geometry and material.

        We always create our own large floor plane positioned slightly above z=0
        to ensure materials are visible, rather than trying to modify IsaacSim's terrain.

        Args:
            env_prim_path: Environment prim path
            env_id: Environment ID
        """
        # Create our own floor geometry at World level (shared across envs)
        floor_path = "/World/scene_floor"

        if floor_path not in self._created_prims:
            # Use configured size and position
            floor_size = self.cfg.floor.size
            floor_position = self.cfg.floor.position

            self._create_cube_prim(floor_path, floor_size, floor_position)
            self._created_prims.add(floor_path)
            logger.info(f"Created custom floor plane at {floor_path} (size={floor_size}, pos={floor_position})")

        # Randomize material every time this is called
        if self.cfg.floor.material_randomization and self.cfg.floor_materials is not None:
            material_path = self._select_material(self.cfg.floor_materials, "floor_index")
            if material_path:
                material_name = material_path.split("/")[-1].replace(".mdl", "")
                logger.info(f"Applying floor material: {material_name} to {floor_path}")
                self._apply_material_to_prim(material_path, floor_path)

    def _create_or_update_walls(self, env_prim_path: str, env_id: int):
        """Create or update wall geometry and materials (4 walls).

        Args:
            env_prim_path: Environment prim path
            env_id: Environment ID
        """
        # Wall naming: front, back, left, right
        wall_configs = self._generate_wall_configs(self.cfg.walls.size, self.cfg.walls.position)

        logger.debug(f"Wall configs: {wall_configs}")

        # Select material once for all walls (same material for all 4 walls)
        material_path = None
        if self.cfg.walls.material_randomization and self.cfg.wall_materials is not None:
            material_path = self._select_material(self.cfg.wall_materials, "wall_index")

        for wall_name, (size, position) in wall_configs.items():
            wall_path = f"{env_prim_path}/scene_wall_{wall_name}"

            logger.debug(f"Creating wall '{wall_name}' at {wall_path}: size={size}, position={position}")

            # Create wall if it doesn't exist
            if wall_path not in self._created_prims:
                self._create_cube_prim(wall_path, size, position)
                self._created_prims.add(wall_path)
                logger.debug(f"Wall '{wall_name}' created and added to _created_prims")
            else:
                logger.debug(f"Wall '{wall_name}' already exists, skipping creation")

            # Apply the selected material to this wall
            if material_path:
                self._apply_material_to_prim(material_path, wall_path)

    def _create_or_update_ceiling(self, env_prim_path: str, env_id: int):
        """Create or update ceiling geometry and material.

        Args:
            env_prim_path: Environment prim path
            env_id: Environment ID
        """
        ceiling_path = f"{env_prim_path}/scene_ceiling"

        # Create ceiling if it doesn't exist
        if ceiling_path not in self._created_prims:
            self._create_cube_prim(ceiling_path, self.cfg.ceiling.size, self.cfg.ceiling.position)
            self._created_prims.add(ceiling_path)

        # Randomize material
        if self.cfg.ceiling.material_randomization and self.cfg.ceiling_materials is not None:
            material_path = self._select_material(self.cfg.ceiling_materials, "ceiling_index")
            if material_path:
                self._apply_material_to_prim(material_path, ceiling_path)

    def _create_or_update_table(self, env_prim_path: str, env_id: int):
        """Create or update table/desktop geometry and material.

        Args:
            env_prim_path: Environment prim path
            env_id: Environment ID
        """
        table_path = f"{env_prim_path}/scene_table"

        # Create table if it doesn't exist
        if table_path not in self._created_prims:
            self._create_cube_prim(table_path, self.cfg.table.size, self.cfg.table.position)
            self._created_prims.add(table_path)

        # Randomize material
        if self.cfg.table.material_randomization and self.cfg.table_materials is not None:
            material_path = self._select_material(self.cfg.table_materials, "table_index")
            if material_path:
                self._apply_material_to_prim(material_path, table_path)

    def _generate_wall_configs(
        self, base_size: tuple[float, float, float], base_position: tuple[float, float, float]
    ) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Generate configurations for 4 walls around a room.

        Args:
            base_size: Base wall size (room_length, thickness, height)
            base_position: Center position of walls (x, y, z_center)

        Returns:
            Dictionary mapping wall names to (size, position) tuples
        """
        room_length, thickness, height = base_size
        cx, cy, cz = base_position

        # Calculate wall positions and sizes to form a complete enclosure
        # The room is centered at (cx, cy), with room_length x room_length size
        # Walls are placed at the edges, with thickness extending outward

        half_room = room_length / 2.0  # Distance from center to edge
        half_thickness = thickness / 2.0

        # Front/Back walls extend in X direction, should cover full width INCLUDING side wall thickness
        # Left/Right walls extend in Y direction, fit BETWEEN front/back walls

        return {
            # Front wall (positive Y side)
            # - Extends in X direction: width = room_length + 2*thickness (to cover side walls)
            # - Center Y position: cy + half_room + half_thickness (outer edge of room)
            "front": (
                (room_length + 2 * thickness, thickness, height),  # Cover corners
                (cx, cy + half_room + half_thickness, cz),
            ),
            # Back wall (negative Y side)
            "back": (
                (room_length + 2 * thickness, thickness, height),  # Cover corners
                (cx, cy - half_room - half_thickness, cz),
            ),
            # Left wall (negative X side)
            # - Extends in Y direction: length = room_length (fits between front/back)
            # - Center X position: cx - half_room - half_thickness (outer edge of room)
            "left": (
                (thickness, room_length, height),  # Fits between front/back walls
                (cx - half_room - half_thickness, cy, cz),
            ),
            # Right wall (positive X side)
            "right": (
                (thickness, room_length, height),  # Fits between front/back walls
                (cx + half_room + half_thickness, cy, cz),
            ),
        }

    def _create_cube_prim(self, prim_path: str, size: tuple[float, float, float], position: tuple[float, float, float]):
        """Create a cube primitive with physics collision using IsaacSim commands.

        Args:
            prim_path: USD prim path
            size: Size of the cube (x, y, z)
            position: Position of the cube (x, y, z)
        """
        try:
            # Lazy import IsaacSim modules
            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            from pxr import Gf, UsdGeom, UsdPhysics

            # Get stage
            stage = prim_utils.get_current_stage()
            if not stage:
                logger.warning("No stage available")
                return

            # Create cube directly using USD
            cube_prim = stage.DefinePrim(prim_path, "Cube")
            if not cube_prim:
                logger.warning(f"Failed to define cube at {prim_path}")
                return

            cube = UsdGeom.Cube(cube_prim)

            # Set cube size to 2.0 (default is 2.0, gives us a 2x2x2 cube, then we scale by 0.5*desired)
            # Actually, USD Cube size=2.0 means each edge is 2.0 units
            # So we need to scale by (desired_size / 2.0)
            cube.GetSizeAttr().Set(2.0)

            # Add xform ops for scale and translation
            xform = UsdGeom.Xformable(cube_prim)

            # Clear any existing xform ops to start fresh
            xform.ClearXformOpOrder()

            # Add translation operation first
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*position))

            # Then add scale operation (scale is relative to the translated position)
            scale_op = xform.AddScaleOp()
            # USD Cube with size=2.0 has edges of length 2.0
            # So to get desired size, we scale by (desired / 2.0)
            scale_factor = tuple(s / 2.0 for s in size)
            scale_op.Set(Gf.Vec3d(*scale_factor))

            # Add physics collision - IMPORTANT for table to support objects!
            # For static geometry (floor, walls, table), we ONLY need CollisionAPI
            # DO NOT add RigidBodyAPI - that makes it dynamic and it will fall!
            collision_api = UsdPhysics.CollisionAPI.Apply(cube_prim)

            logger.debug(
                f"Created cube at {prim_path} with size {size} (scale={scale_factor}) and position {position} with collision"
            )

        except Exception as e:
            logger.warning(f"Failed to create cube prim {prim_path}: {e}")

    def _select_material(self, material_pool: SceneMaterialPoolCfg, state_key: str) -> str | None:
        """Select a material from the pool based on selection strategy.

        Args:
            material_pool: Material pool configuration
            state_key: Key for tracking sequential selection state

        Returns:
            Selected material path or None
        """
        if not material_pool.material_paths:
            return None

        if material_pool.selection_strategy == "random":
            return self._rng.choice(material_pool.material_paths)

        elif material_pool.selection_strategy == "sequential":
            idx = self._material_selection_state[state_key] % len(material_pool.material_paths)
            self._material_selection_state[state_key] += 1
            return material_pool.material_paths[idx]

        elif material_pool.selection_strategy == "weighted":
            return self._rng.choices(material_pool.material_paths, weights=material_pool.weights, k=1)[0]

        return None

    def _calculate_uv_tile_scale(self, prim, prim_path: str) -> float:
        """Calculate appropriate UV tile scale based on geometry size.

        For large surfaces like walls and floors, we want textures to repeat
        rather than stretch. This function calculates a tile_scale that makes
        textures repeat approximately every 1-2 meters.

        Args:
            prim: USD prim
            prim_path: Path to the prim (for logging)

        Returns:
            tile_scale: Scale factor for UV coordinates (smaller = more repetitions)
        """
        try:
            from pxr import UsdGeom

            prim_type = prim.GetTypeName()

            if prim_type == "Cube":
                # Get cube size and scale
                cube = UsdGeom.Cube(prim)
                size = cube.GetSizeAttr().Get() or 2.0  # Default USD cube size is 2.0

                # Get scale from xform
                xformable = UsdGeom.Xformable(prim)
                xform_ops = xformable.GetOrderedXformOps()
                scale = [1.0, 1.0, 1.0]
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_value = op.Get()
                        if scale_value:
                            scale = list(scale_value)
                        break

                # Calculate actual dimensions
                actual_sizes = [size * s for s in scale]

                # For walls, we care about the two largest dimensions (width and height)
                # For floors/ceilings, same applies
                sorted_sizes = sorted(actual_sizes, reverse=True)
                max_dimension = sorted_sizes[0]  # Largest dimension

                # We want textures to repeat approximately every 1 meter
                # tile_scale controls how many times the 0-1 UV range is used
                # Larger dimensions need larger tile_scale values for more repetitions
                target_texture_size = 1.0  # Target: texture repeats every 1 meter
                tile_scale = max_dimension / target_texture_size

                logger.debug(
                    f"Cube {prim_path}: size={size}, scale={scale}, actual={actual_sizes}, tile_scale={tile_scale:.2f}"
                )
                return tile_scale

            else:
                # For other geometry types, use a default moderate tiling
                return 2.0

        except Exception as e:
            logger.warning(f"Failed to calculate tile scale for {prim_path}: {e}")
            return 1.0  # Default fallback

    def _apply_material_to_prim(self, material_path: str, prim_path: str):
        """Apply MDL material to a primitive using MaterialRandomizer's proven method.

        Args:
            material_path: Path to MDL material file
            prim_path: USD prim path
        """
        # try:
        # First, check and download the material file if needed
        from metasim.utils.hf_util import check_and_download_recursive

        logger.debug(f"Checking and downloading material: {material_path}")
        check_and_download_recursive([material_path])

        # Get absolute path to MDL file
        import os

        abs_material_path = os.path.abspath(material_path)
        if not os.path.exists(abs_material_path):
            logger.warning(f"Material file not found: {abs_material_path}")
            return

        # Lazy import IsaacSim modules
        try:
            import omni.isaac.core.utils.prims as prim_utils
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils

        from pxr import UsdGeom

        # Find all mesh prims under the target path
        target_prim = prim_utils.get_prim_at_path(prim_path)
        if not target_prim or not target_prim.IsValid():
            logger.warning(f"Target prim not found: {prim_path}")
            return

        mesh_prims_paths = []

        # Check if target is a geometric primitive (Mesh, Cube, Sphere, etc.) or Xform
        prim_type = target_prim.GetTypeName()

        if target_prim.IsA(UsdGeom.Mesh):
            mesh_prims_paths.append(prim_path)
            logger.debug(f"Target prim {prim_path} is a Mesh")
        elif prim_type in ["Cube", "Sphere", "Cylinder", "Cone", "Capsule"]:
            # USD geometric primitives can accept materials directly
            mesh_prims_paths.append(prim_path)
            logger.debug(f"Target prim {prim_path} is a {prim_type} primitive")
        else:
            logger.debug(f"Target prim {prim_path} is {prim_type}, searching for Mesh/Primitive children...")

            # Recursively find all mesh/primitive children
            def find_renderables(prim):
                prim_type = prim.GetTypeName()
                if prim.IsA(UsdGeom.Mesh) or prim_type in ["Cube", "Sphere", "Cylinder", "Cone", "Capsule"]:
                    mesh_path = str(prim.GetPath())
                    mesh_prims_paths.append(mesh_path)
                    logger.debug(f"  Found {prim_type}: {mesh_path}")
                for child in prim.GetChildren():
                    find_renderables(child)

            find_renderables(target_prim)

        if not mesh_prims_paths:
            logger.warning(f"No renderable prims found under {prim_path}, applying to prim itself...")
            mesh_prims_paths = [prim_path]

        # Use MaterialRandomizer's proven method to apply material to each mesh
        from metasim.randomization.material_randomizer import MaterialRandomizer

        dummy_randomizer = MaterialRandomizer.__new__(MaterialRandomizer)
        dummy_randomizer.handler = self.handler

        for mesh_path in mesh_prims_paths:
            # Ensure UV coordinates for all scene geometry (floor, walls, ceiling, table)
            mesh_prim = prim_utils.get_prim_at_path(mesh_path)
            if mesh_prim and any(
                keyword in mesh_path.lower()
                for keyword in ["ground", "terrain", "floor", "wall", "ceiling", "table", "scene_"]
            ):
                logger.debug(f"Ensuring UV coordinates for scene geometry {mesh_path}")
                try:
                    # Calculate appropriate tile_scale based on geometry size
                    tile_scale = self._calculate_uv_tile_scale(mesh_prim, mesh_path)
                    dummy_randomizer._ensure_uv_for_hierarchy(mesh_prim, tile_scale=tile_scale)
                    logger.debug(f"Successfully ensured UV coordinates with tile_scale={tile_scale}")
                except Exception as e:
                    logger.warning(f"Failed to ensure UV coordinates: {e}")

            dummy_randomizer._apply_mdl_to_prim(material_path, mesh_path)
            logger.debug(f"Applied material to mesh {mesh_path}")

        logger.info(f"Successfully applied MDL material to {len(mesh_prims_paths)} mesh(es) under {prim_path}")

        # except Exception as e:
        #     logger.warning(f"Failed to apply material {material_path} to {prim_path}: {e}")

    def _randomize_materials_only(self, env_ids: list[int] | None = None):
        """Apply material randomization to existing scene elements only.

        This is used when a predefined scene exists but we still want to
        randomize materials on existing geometry.

        Args:
            env_ids: Optional list of environment IDs to randomize
        """
        # Get environment IDs to randomize
        target_env_ids = env_ids if env_ids is not None else self.cfg.env_ids
        if target_env_ids is None:
            target_env_ids = list(range(self.handler.num_envs))

        logger.debug("Applying material randomization to existing scene elements")

        # This would require detecting existing scene elements
        # For now, we skip this in favor of explicit material randomization
        pass

    def get_scene_properties(self) -> dict:
        """Get current scene properties.

        Returns:
            Dictionary containing scene element properties
        """
        properties = {
            "created_prims": list(self._created_prims),
            "num_elements": len(self._created_prims),
        }

        return properties
