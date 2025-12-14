"""Scene Randomizer - Dynamic object lifecycle manager.

The SceneRandomizer is responsible for managing dynamic objects that can be
created, deleted, and switched at runtime. It operates on three hierarchical layers:

Layers:
- Environment: Backgrounds, rooms, walls, floors, ceilings
- Workspace: Tables, desktops, manipulation surfaces
- Objects: Static distractor objects (cups, fruits, etc.)

Key features:
- Direct USD Stage manipulation (bypasses Handler)
- Supports Manual Geometry (procedural) and USD Assets
- Registers all created objects to ObjectRegistry
- Built-in material randomization for Manual Geometry

Note: All objects created by SceneRandomizer are pure visual (disable_physics=True)
      and cannot be added to Handler's scene structure due to IsaacLab limitations.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.randomization.core.object_registry import ObjectMetadata, ObjectRegistry
from metasim.utils.configclass import configclass

# =============================================================================
# Scene Element Configurations
# =============================================================================


@configclass
class ManualGeometryCfg:
    """Manual procedural geometry configuration.

    SceneRandomizer creates geometry with optional default material.
    For material randomization, use MaterialRandomizer.

    Attributes:
        name: Unique element name
        geometry_type: Primitive type (cube, sphere, cylinder, plane)
        size: Geometry size (x, y, z) in meters
        position: World position (x, y, z)
        rotation: Orientation quaternion (w, x, y, z)
        add_collision: Whether to add collision (usually False for scene elements)
        default_material: Default MDL material path (applied once at creation, optional)
        enabled: Whether this element is active

    Example:
        # Create table with default material
        table = ManualGeometryCfg(
            name="table",
            geometry_type="cube",
            size=(1.8, 1.8, 0.1),
            default_material="roboverse_data/materials/arnold/Wood/Plywood.mdl"  # Optional
        )

        # Later: Randomize material (separate step)
        table_mat = MaterialRandomizer(
            MaterialPresets.mdl_family_object("table", family=("wood", "stone"))
        )
    """

    name: str = dataclasses.MISSING
    geometry_type: Literal["cube", "sphere", "cylinder", "plane"] = "cube"
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    add_collision: bool = False
    default_material: str | None = None
    enabled: bool = True


@configclass
class USDAssetCfg:
    """Single USD asset configuration.

    Attributes:
        name: Unique element name
        usd_path: Path to USD file
        position: World position (x, y, z)
        rotation: Orientation quaternion (w, x, y, z)
        scale: Scale factor (x, y, z)
        auto_download: Enable automatic asset download
        add_collision: Whether to add static collision (prevents penetration without physics)
        enabled: Whether this element is active

    Note: Scene objects are always pure visual (physics disabled)
    """

    name: str = dataclasses.MISSING
    usd_path: str = dataclasses.MISSING
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    auto_download: bool = True
    add_collision: bool = False
    enabled: bool = True

    def __post_init__(self):
        if not self.usd_path:
            raise ValueError(f"USDAssetCfg '{self.name}': usd_path cannot be empty")


@configclass
class USDAssetPoolCfg:
    """USD asset pool configuration.

    Randomly selects one USD from a pool. Supports per-path configuration overrides.

    Attributes:
        name: Pool name
        usd_paths: List of USD file paths (will be converted to candidates)
        per_path_overrides: Dict mapping USD path to override config
        position: Default position for all USDs
        rotation: Default rotation for all USDs
        scale: Default scale for all USDs
        selection_strategy: Selection strategy (random, sequential)
        auto_download: Enable automatic download
        add_collision: Whether to add static collision (prevents penetration without physics)
        enabled: Whether this pool is active
    """

    name: str = dataclasses.MISSING
    usd_paths: list[str] | None = None
    per_path_overrides: dict[str, dict] | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    selection_strategy: Literal["random", "sequential"] = "random"
    auto_download: bool = True
    add_collision: bool = False
    enabled: bool = True
    candidates: list[USDAssetCfg] | None = None  # Will be auto-generated

    def __post_init__(self):
        if not self.usd_paths:
            raise ValueError(f"USDAssetPoolCfg '{self.name}': usd_paths cannot be empty")

        # Convert usd_paths to candidates
        self.candidates = []
        for i, path in enumerate(self.usd_paths):
            cfg_kwargs = {
                "name": f"{self.name}_{i}",
                "usd_path": path,
                "position": self.position,
                "rotation": self.rotation,
                "scale": self.scale,
                "auto_download": self.auto_download,
                "add_collision": self.add_collision,
                "enabled": self.enabled,
            }

            # Apply per-path overrides (filter out invalid keys)
            if self.per_path_overrides:
                from pathlib import Path

                override = self.per_path_overrides.get(path) or self.per_path_overrides.get(Path(path).name)
                if override:
                    # Only update valid USDAssetCfg fields
                    valid_keys = {"position", "rotation", "scale", "auto_download", "add_collision", "enabled"}
                    for k, v in override.items():
                        if k in valid_keys:
                            cfg_kwargs[k] = v

            self.candidates.append(USDAssetCfg(**cfg_kwargs))


# =============================================================================
# Layer Configurations
# =============================================================================


@configclass
class SceneLayerCfg:
    """Scene layer configuration.

    Attributes:
        elements: List of scene elements (Manual Geometry or USD Assets)
        shared: Whether elements are shared across all environments
        z_offset: Z-axis offset to apply to all elements
        enabled: Whether this layer is active
    """

    elements: list[ManualGeometryCfg | USDAssetCfg | USDAssetPoolCfg] = dataclasses.field(default_factory=list)
    shared: bool = True
    z_offset: float = 0.0
    enabled: bool = True


# Type aliases for clarity
EnvironmentLayerCfg = SceneLayerCfg
WorkspaceLayerCfg = SceneLayerCfg
ObjectsLayerCfg = SceneLayerCfg


@configclass
class SceneRandomCfg:
    """Scene randomization configuration.

    Attributes:
        environment_layer: Environment layer (floor, walls, ceiling)
        workspace_layer: Workspace layer (tables, desktops)
        objects_layer: Objects layer (distractors)
        auto_flush_visuals: Auto flush visual updates after material changes
        only_if_no_scene: Only create scene if none exists
    """

    environment_layer: SceneLayerCfg | None = None
    workspace_layer: SceneLayerCfg | None = None
    objects_layer: SceneLayerCfg | None = None
    auto_flush_visuals: bool = True
    only_if_no_scene: bool = False


# =============================================================================
# Scene Randomizer Implementation
# =============================================================================


class SceneRandomizer(BaseRandomizerType):
    """Scene randomizer for dynamic object lifecycle management.

    Responsibilities:
    - Create dynamic objects (Manual Geometry and USD Assets)
    - Delete dynamic objects
    - Switch dynamic objects (USD switching)
    - Set dynamic object transforms
    - NOT responsible for: Material randomization (use MaterialRandomizer)

    Characteristics:
    - Operates directly on USD Stage (bypasses Handler)
    - All created objects are pure visual (disable_physics=True)
    - Registers objects to ObjectRegistry for unified access
    - Supports Hybrid simulation (uses render_handler)

    Usage:
        cfg = SceneRandomCfg(
            workspace_layer=SceneLayerCfg(
                shared=True,
                elements=[USDAssetPoolCfg(name="table", usd_paths=[...])]
            )
        )
        randomizer = SceneRandomizer(cfg, seed=42)
        randomizer.bind_handler(handler)
        randomizer()  # Create/switch scene objects
    """

    REQUIRES_HANDLER = "render"  # Use render_handler for Hybrid

    def __init__(self, cfg: SceneRandomCfg, seed: int | None = None):
        """Initialize scene randomizer.

        Args:
            cfg: Scene randomization configuration
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.cfg = cfg
        self.registry: ObjectRegistry | None = None

        # USD Stage (lazy initialized)
        self.stage = None
        self.prim_utils = None

        # Track created prims for switching
        self._created_prims: dict[str, list[str]] = {}
        self._loaded_usds: dict[str, str] = {}  # {prim_path: usd_path}

        self._sub_init()

    def _sub_init(self):
        # Initialize USD
        try:
            import omni.isaac.core.utils.prims as prim_utils
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils

        self.prim_utils = prim_utils
        self.stage = prim_utils.get_current_stage()

    def __call__(self, env_ids: list[int] | None = None):
        """Execute scene randomization.

        Args:
            env_ids: Environment IDs to randomize (None = all environments)
        """
        if not self._actual_handler:
            raise RuntimeError("Handler not bound. Call bind_handler() first.")

        # Skip if scene exists and only_if_no_scene is True
        if self.cfg.only_if_no_scene and self._check_scene_exists():
            return

        # Get target environment IDs
        target_env_ids = env_ids if env_ids is not None else list(range(self._actual_handler.num_envs))

        # Process layers
        if self.cfg.environment_layer and self.cfg.environment_layer.enabled:
            self._process_layer(self.cfg.environment_layer, "environment", target_env_ids)

        if self.cfg.workspace_layer and self.cfg.workspace_layer.enabled:
            self._process_layer(self.cfg.workspace_layer, "workspace", target_env_ids)

        if self.cfg.objects_layer and self.cfg.objects_layer.enabled:
            self._process_layer(self.cfg.objects_layer, "objects", target_env_ids)

        # Auto-flush visual updates
        if self.cfg.auto_flush_visuals:
            self._flush_visual_updates()

    def _check_scene_exists(self) -> bool:
        """Check if scene already exists."""
        # Check if any layer has created prims
        return len(self._created_prims) > 0

    def _process_layer(
        self,
        layer_cfg: SceneLayerCfg,
        layer_name: str,
        env_ids: list[int],
    ):
        """Process a scene layer.

        Args:
            layer_cfg: Layer configuration
            layer_name: Layer name (environment, workspace, objects)
            env_ids: Environment IDs to process
        """
        for element in layer_cfg.elements:
            if not element.enabled:
                continue

            if isinstance(element, ManualGeometryCfg):
                self._process_manual_geometry(element, layer_name, layer_cfg, env_ids)
            elif isinstance(element, USDAssetCfg):
                self._process_usd_asset(element, layer_name, layer_cfg, env_ids)
            elif isinstance(element, USDAssetPoolCfg):
                self._process_usd_pool(element, layer_name, layer_cfg, env_ids)

    # -------------------------------------------------------------------------
    # Manual Geometry Processing
    # -------------------------------------------------------------------------

    def _process_manual_geometry(
        self,
        element: ManualGeometryCfg,
        layer_name: str,
        layer_cfg: SceneLayerCfg,
        env_ids: list[int],
    ):
        """Process manual geometry element.

        Args:
            element: Manual geometry configuration
            layer_name: Layer name
            layer_cfg: Layer configuration
            env_ids: Environment IDs
        """
        if layer_cfg.shared:
            # Shared: create one prim for all environments
            prim_path = f"/World/scene_{layer_name}_{element.name}"
            self._create_geometry_prim(element, prim_path, layer_cfg.z_offset)
            prim_paths = [prim_path]
        else:
            # Per-env: create one prim per environment
            prim_paths = []
            for env_id in env_ids:
                prim_path = f"/World/envs/env_{env_id}/scene_{layer_name}_{element.name}"
                self._create_geometry_prim(element, prim_path, layer_cfg.z_offset)
                prim_paths.append(prim_path)

        # Register to ObjectRegistry (only on first creation)
        if element.name not in self._created_prims:
            self.registry.register(
                ObjectMetadata(
                    name=element.name,
                    category="scene_element",
                    lifecycle="dynamic",
                    prim_paths=prim_paths,
                    shared=layer_cfg.shared,
                    has_physics=False,
                    layer=layer_name,
                )
            )
            self._created_prims[element.name] = prim_paths

    def _create_geometry_prim(self, element: ManualGeometryCfg, prim_path: str, z_offset: float):
        """Create a procedural geometry prim.

        Args:
            element: Manual geometry configuration
            prim_path: USD prim path
            z_offset: Z-axis offset
        """
        from pxr import Gf, UsdGeom

        if not self.stage:
            logger.warning("No stage available for geometry creation")
            return

        # Adjust position with z_offset
        pos = list(element.position)
        pos[2] += z_offset

        # Create geometry based on type
        if element.geometry_type == "cube":
            geom = UsdGeom.Cube.Define(self.stage, prim_path)
            geom.GetSizeAttr().Set(2.0)  # Unit cube, will scale via transform
        elif element.geometry_type == "sphere":
            geom = UsdGeom.Sphere.Define(self.stage, prim_path)
            geom.GetRadiusAttr().Set(element.size[0])
        elif element.geometry_type == "cylinder":
            geom = UsdGeom.Cylinder.Define(self.stage, prim_path)
            geom.GetRadiusAttr().Set(element.size[0])
            geom.GetHeightAttr().Set(element.size[2])
        elif element.geometry_type == "plane":
            geom = UsdGeom.Mesh.Define(self.stage, prim_path)
            # Define plane vertices
            points = [
                (-element.size[0] / 2, -element.size[1] / 2, 0),
                (element.size[0] / 2, -element.size[1] / 2, 0),
                (element.size[0] / 2, element.size[1] / 2, 0),
                (-element.size[0] / 2, element.size[1] / 2, 0),
            ]
            geom.CreatePointsAttr().Set(points)
            geom.CreateFaceVertexCountsAttr().Set([4])
            geom.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        else:
            logger.error(f"Unsupported geometry type: {element.geometry_type}")
            return

        # Set transform (position, rotation, scale)
        xform = UsdGeom.Xformable(geom)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddOrientOp().Set(
            Gf.Quatf(element.rotation[0], Gf.Vec3f(element.rotation[1], element.rotation[2], element.rotation[3]))
        )
        # Scale to actual size (cube is 2.0 units, scale to desired size)
        scale_factor = (element.size[0] / 2.0, element.size[1] / 2.0, element.size[2] / 2.0)
        xform.AddScaleOp().Set(Gf.Vec3d(*scale_factor))

        # Add collision if requested
        if element.add_collision:
            from pxr import UsdPhysics

            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

        # Apply default material if specified (once, not randomized)
        if element.default_material:
            try:
                if not hasattr(self, "adapter"):
                    from metasim.randomization.core.isaacsim_adapter import IsaacSimAdapter

                    self.adapter = IsaacSimAdapter(self._actual_handler)

                # Apply material (adapter will handle unique naming internally)
                self.adapter.apply_mdl_material(prim_path, element.default_material)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # USD Asset Processing
    # -------------------------------------------------------------------------

    def _process_usd_pool(
        self,
        element: USDAssetPoolCfg,
        layer_name: str,
        layer_cfg: SceneLayerCfg,
        env_ids: list[int],
    ):
        """Process USD asset pool (select one and load).

        Args:
            element: USD asset pool configuration
            layer_name: Layer name
            layer_cfg: Layer configuration
            env_ids: Environment IDs
        """
        # Select from candidates (usd_paths was converted to candidates in __post_init__)
        if not element.candidates:
            logger.error(f"USD pool '{element.name}' has no candidates")
            return

        if element.selection_strategy == "random":
            asset_cfg = self.rng.choice(element.candidates)
        else:  # sequential
            if not hasattr(self, "_usd_pool_indices"):
                self._usd_pool_indices = {}
            pool_key = element.name
            if pool_key not in self._usd_pool_indices:
                self._usd_pool_indices[pool_key] = 0
            idx = self._usd_pool_indices[pool_key] % len(element.candidates)
            self._usd_pool_indices[pool_key] += 1
            asset_cfg = element.candidates[idx]

        # Process the selected USD asset (use pool name for consistent prim_path)
        self._process_usd_asset(asset_cfg, layer_name, layer_cfg, env_ids, override_name=element.name)

    def _process_usd_asset(
        self,
        element: USDAssetCfg,
        layer_name: str,
        layer_cfg: SceneLayerCfg,
        env_ids: list[int],
        override_name: str | None = None,
    ):
        """Process single USD asset.

        Args:
            element: USD asset configuration
            layer_name: Layer name
            layer_cfg: Layer configuration
            env_ids: Environment IDs
            override_name: Override name for consistent prim paths (used by pools)
        """
        # Handle auto-download if needed
        if element.auto_download:
            if not hasattr(self, "_prepared_usds"):
                self._prepared_usds = set()

            if not self._ensure_usd_downloaded(element.usd_path, self._prepared_usds, auto_download=True):
                logger.error(f"Failed to download USD: {element.usd_path}")
                return

        # Use override_name (from pool) for consistent prim_path across randomizations
        element_name = override_name if override_name else element.name

        if layer_cfg.shared:
            # Shared: create one prim for all environments
            prim_path = f"/World/scene_{layer_name}_{element_name}"
            self._load_or_replace_usd(prim_path, element, layer_cfg.z_offset)
            prim_paths = [prim_path]
        else:
            # Per-env: create one prim per environment
            prim_paths = []
            for env_id in env_ids:
                prim_path = f"/World/envs/env_{env_id}/scene_{layer_name}_{element_name}"
                self._load_or_replace_usd(prim_path, element, layer_cfg.z_offset)
                prim_paths.append(prim_path)

        # Register to ObjectRegistry (use element_name for consistency)
        if element_name not in self._created_prims:
            self.registry.register(
                ObjectMetadata(
                    name=element_name,  # Use pool name, not candidate name
                    category="scene_element",
                    lifecycle="dynamic",
                    prim_paths=prim_paths,
                    shared=layer_cfg.shared,
                    has_physics=False,
                    layer=layer_name,
                )
            )
            self._created_prims[element_name] = prim_paths

    def _load_or_replace_usd(self, prim_path: str, element: USDAssetCfg, z_offset: float):
        """Load or replace USD asset at prim_path.

        Args:
            prim_path: USD prim path
            element: USD asset configuration
            z_offset: Z-axis offset
        """
        # Check if we need to replace
        if prim_path in self._loaded_usds:
            if self._loaded_usds[prim_path] != element.usd_path:
                # Different USD: replace
                logger.info(f"Replacing USD at {prim_path}")
                logger.info(f"   Old: {self._loaded_usds[prim_path]}")
                logger.info(f"   New: {element.usd_path}")
                self._delete_usd(prim_path)
                self._load_usd(prim_path, element, z_offset)
                self._loaded_usds[prim_path] = element.usd_path
                logger.info("USD replaced successfully")
            else:
                # Same USD: skip (already loaded)
                pass
        else:
            # First time: load
            self._load_usd(prim_path, element, z_offset)
            self._loaded_usds[prim_path] = element.usd_path

    def _load_usd(self, prim_path: str, element: USDAssetCfg, z_offset: float):
        """Load USD asset.

        Args:
            prim_path: USD prim path
            element: USD asset configuration
            z_offset: Z-axis offset
        """
        import os

        from pxr import Gf, UsdGeom

        # Check for Kujiale path remapping (RoboVerse -> InteriorAgent folder)
        usd_to_load = element.usd_path
        if hasattr(self, "_usda_path_mapping") and element.usd_path in self._usda_path_mapping:
            usd_to_load = self._usda_path_mapping[element.usd_path]

        # Check if URDF (needs conversion)
        is_urdf = usd_to_load.endswith(".urdf")

        if is_urdf:
            # URDF: Convert to USD first (using MESH converter)
            usd_path = self._convert_urdf_to_usd(usd_to_load)
            if not usd_path:
                logger.error(f"Failed to convert URDF: {element.usd_path}")
                # Create empty Xform as placeholder
                self.stage.DefinePrim(prim_path, "Xform")
                return

            # Load the converted USD (use absolute path)
            usd_path_abs = os.path.abspath(usd_path)
            try:
                try:
                    from omni.isaac.core.utils.stage import add_reference_to_stage
                except ImportError:
                    import isaacsim.core.utils.stage as stage_utils

                    add_reference_to_stage = stage_utils.add_reference_to_stage

                add_reference_to_stage(usd_path_abs, prim_path)
            except Exception as e:
                logger.error(f"Failed to load converted USD: {e}")
                return
        else:
            # USD: Use reference (absolute path)
            usd_path_abs = os.path.abspath(usd_to_load)
            try:
                try:
                    from omni.isaac.core.utils.stage import add_reference_to_stage
                except ImportError:
                    import isaacsim.core.utils.stage as stage_utils

                    add_reference_to_stage = stage_utils.add_reference_to_stage

                add_reference_to_stage(usd_path_abs, prim_path)
            except Exception as e:
                # Fallback: use USD API directly
                ref_prim = self.stage.DefinePrim(prim_path, "Xform")
                if not ref_prim:
                    logger.error(f"Failed to create prim at {prim_path}")
                    return
                # Use absolute path and don't specify defaultPrim (USD will auto-find)
                ref_prim.GetReferences().AddReference(usd_path_abs)

        # Set transform
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            xform = UsdGeom.Xformable(prim)

            # Check for existing xform ops (from converted USD)
            existing_ops = xform.GetOrderedXformOps()

            if existing_ops:
                # Update existing ops (preserve precision)
                pos = list(element.position)
                pos[2] += z_offset

                for op in existing_ops:
                    op_name = op.GetOpName()
                    if "translate" in op_name:
                        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                            op.Set(Gf.Vec3d(*pos))
                        else:
                            op.Set(Gf.Vec3f(*pos))
                    elif "scale" in op_name:
                        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                            op.Set(Gf.Vec3d(*element.scale))
                        else:
                            op.Set(Gf.Vec3f(*element.scale))
                    elif "orient" in op_name and element.rotation != (1.0, 0.0, 0.0, 0.0):
                        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                            op.Set(Gf.Quatd(element.rotation[0], Gf.Vec3d(*element.rotation[1:])))
                        else:
                            op.Set(Gf.Quatf(element.rotation[0], Gf.Vec3f(*element.rotation[1:])))
            else:
                # No existing ops, create new (use double for consistency with URDF converter)
                xform.ClearXformOpOrder()
                pos = list(element.position)
                pos[2] += z_offset
                xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
                if element.rotation != (1.0, 0.0, 0.0, 0.0):
                    xform.AddOrientOp().Set(
                        Gf.Quatd(
                            element.rotation[0], Gf.Vec3d(element.rotation[1], element.rotation[2], element.rotation[3])
                        )
                    )
                xform.AddScaleOp().Set(Gf.Vec3d(*element.scale))

            # Disable physics (remove RigidBodyAPI, optionally keep CollisionAPI)
            self._disable_physics_for_prim(prim, keep_collision=element.add_collision)

    def _delete_usd(self, prim_path: str):
        """Delete USD prim.

        Args:
            prim_path: USD prim path
        """
        try:
            import omni.isaac.core.utils.prims as prim_utils
        except ModuleNotFoundError:
            import isaacsim.core.utils.prims as prim_utils

        if prim_utils.is_prim_path_valid(prim_path):
            prim_utils.delete_prim(prim_path)

    def _disable_physics_for_prim(self, prim, keep_collision: bool = False, is_root: bool = True):
        """Recursively disable physics for a prim.

        Args:
            prim: USD prim
            keep_collision: If True, apply CollisionAPI to root prim (for static collision)
            is_root: Whether this is the root prim of the USD asset
        """
        from pxr import UsdPhysics

        # Always remove RigidBodyAPI (we don't want dynamic physics)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

        # For root prim: apply or remove CollisionAPI based on keep_collision
        if is_root:
            if keep_collision:
                # Apply CollisionAPI to the root prim
                # PhysX will use the visual geometry for collision
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
            else:
                # Remove CollisionAPI if exists
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
        else:
            # For non-root prims: always remove CollisionAPI to avoid conflicts
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)

        # Recursively process all children (no longer root)
        for child in prim.GetAllChildren():
            self._disable_physics_for_prim(child, keep_collision=keep_collision, is_root=False)

    # -------------------------------------------------------------------------
    # URDF Conversion
    # -------------------------------------------------------------------------

    def _ensure_usd_downloaded(self, usd_path: str, prepared_usds: set[str], auto_download: bool) -> bool:
        """Download USD/URDF and dependencies if needed.

        Args:
            usd_path: Path to USD/URDF file
            prepared_usds: Cache set
            auto_download: Whether to download

        Returns:
            True if ready, False otherwise
        """
        if usd_path in prepared_usds:
            return True

        # Special: Kujiale scenes
        if usd_path.endswith(".usda") and "kujiale" in usd_path.lower():
            success = self._download_kujiale_scene_folder(usd_path)
            if success:
                prepared_usds.add(usd_path)
            return success

        # Check if file exists
        if not os.path.exists(usd_path):
            if not auto_download:
                return False

            # Special: EmbodiedGen URDF (download folder)
            if usd_path.endswith(".urdf") and "EmbodiedGenData" in usd_path:
                success = self._download_embodiedgen_asset_folder(usd_path)
                if success:
                    prepared_usds.add(usd_path)
                return success

            # Generic: single file
            try:
                from metasim.utils.hf_util import check_and_download_single

                check_and_download_single(usd_path)
            except Exception:
                return False

        prepared_usds.add(usd_path)
        return True

    def _download_embodiedgen_asset_folder(self, urdf_path: str) -> bool:
        """Download complete EmbodiedGen asset folder (URDF + mesh/ + textures)."""
        from pathlib import Path

        from huggingface_hub import snapshot_download

        try:
            path_obj = Path(urdf_path)
            parts = path_obj.parts

            if "EmbodiedGenData" not in parts:
                return False

            idx = parts.index("EmbodiedGenData")
            asset_folder_parts = parts[idx + 1 : -1]  # e.g., dataset/basic_furniture/table/uuid
            asset_folder = "/".join(asset_folder_parts)

            local_base = Path(*parts[: idx + 1])

            logger.info(f"Downloading EmbodiedGen folder: {asset_folder}")

            snapshot_download(
                repo_id="HorizonRobotics/EmbodiedGenData",
                repo_type="dataset",
                local_dir=str(local_base),
                allow_patterns=[f"{asset_folder}/*"],
                local_dir_use_symlinks=False,
            )

            return path_obj.exists()

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _download_kujiale_scene_folder(self, usda_path: str) -> bool:
        """Download RoboVerse main USDA + InteriorAgent assets.

        Strategy:
        1. Download RoboVerse USDA file (main scene description)
        2. Download InteriorAgent assets (meshes, textures that USDA references)
        3. Use RoboVerse USDA as primary (it references InteriorAgent assets)

        Args:
            usda_path: Path to RoboVerse USDA (e.g., roboverse_data/scenes/kujiale/003.usda)

        Returns:
            True if both main USDA and assets are available
        """
        from pathlib import Path

        from huggingface_hub import snapshot_download

        try:
            path_obj = Path(usda_path)
            scene_num = int(path_obj.stem)

            # Step 1: Download RoboVerse main USDA file
            if not path_obj.exists():
                logger.info(f"Downloading RoboVerse Kujiale USDA: {path_obj.name}")

                # Determine local directory structure
                if "roboverse_data" in str(path_obj):
                    local_dir = "roboverse_data"
                    remote_path = str(path_obj.relative_to("roboverse_data"))
                else:
                    # Fallback
                    local_dir = str(path_obj.parent)
                    remote_path = path_obj.name

                snapshot_download(
                    repo_id="RoboVerseOrg/roboverse_data",
                    repo_type="dataset",
                    local_dir=local_dir,
                    allow_patterns=[remote_path],
                    local_dir_use_symlinks=False,
                )

                if not path_obj.exists():
                    logger.warning(f"RoboVerse USDA not found after download: {usda_path}")

            # Step 2: Download InteriorAgent assets (meshes, textures)
            remote_folder = f"kujiale_{scene_num:04d}"
            local_base = Path("third_party/InteriorAgent")
            downloaded_folder = local_base / remote_folder

            if not downloaded_folder.exists():
                local_base.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading InteriorAgent assets: {remote_folder}")

                snapshot_download(
                    repo_id="spatialverse/InteriorAgent",
                    repo_type="dataset",
                    local_dir=str(local_base),
                    allow_patterns=[f"{remote_folder}/*"],
                    local_dir_use_symlinks=False,
                )

            # Verify: RoboVerse USDA exists (primary file)
            if not path_obj.exists():
                logger.error(f"RoboVerse USDA missing: {usda_path}")
                return False

            # Verify: InteriorAgent assets exist (referenced assets)
            if not downloaded_folder.exists():
                logger.error(f"InteriorAgent assets missing: {downloaded_folder}")
                return False

            # Step 3: Copy RoboVerse USDA to InteriorAgent folder
            # RoboVerse USDA uses relative refs: "./Meshes/..."
            # By copying to InteriorAgent folder, refs resolve correctly
            usda_in_interior = downloaded_folder / path_obj.name

            if not usda_in_interior.exists():
                try:
                    import shutil

                    shutil.copy2(path_obj, usda_in_interior)
                except Exception as e:
                    logger.error(f"Failed to copy USDA to InteriorAgent folder: {e}")
                    return False

            # Update path to use the USDA in InteriorAgent folder
            # (This will be used by _load_usd via element.usd_path remapping)
            if not hasattr(self, "_usda_path_mapping"):
                self._usda_path_mapping = {}
            self._usda_path_mapping[usda_path] = str(usda_in_interior)
            return True

        except Exception as e:
            logger.error(f"Kujiale download failed: {e}")
            return False

    def _convert_urdf_to_usd(self, urdf_path: str) -> str | None:
        """Convert URDF to USD using AssetConverterFactory (MESH source type).

        Args:
            urdf_path: Path to URDF file

        Returns:
            Path to converted USD file, or None if failed
        """
        from pathlib import Path

        try:
            urdf_path_obj = Path(urdf_path)
            usd_output = urdf_path_obj.parent / (urdf_path_obj.stem + ".usd")

            # If already converted, use existing
            if usd_output.exists():
                return str(usd_output)

            logger.info(f"Converting URDF to USD: {urdf_path}")

            # Use AssetConverterFactory with MESH source type
            try:
                from generation.asset_converter import AssetConverterFactory
                from generation.enums import AssetType

                converter = AssetConverterFactory.create(
                    target_type=AssetType.USD,
                    source_type=AssetType.MESH,  # KEY: MESH not URDF!
                    simulation_app=None,  # Already running
                    exit_close=False,
                    force_usd_conversion=True,
                    make_instanceable=True,
                )

                converter.convert(str(urdf_path), str(usd_output))

                if not usd_output.exists():
                    logger.error(f"USD not created: {usd_output}")
                    return None

                logger.info(f"Converted: {usd_output}")
                return str(usd_output)

            except Exception as e:
                logger.error(f"Conversion failed: {e}")
                return None

        except Exception as e:
            logger.error(f"URDF conversion error: {e}")
            return None

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _flush_visual_updates(self):
        """Flush visual updates to ensure materials are visible.

        Respects global defer flag for atomic multi-randomizer operations.
        """
        # Check global defer flag (set by apply_randomization for 22→1 flush optimization)
        if (
            hasattr(self._actual_handler, "_defer_all_visual_flushes")
            and self._actual_handler._defer_all_visual_flushes
        ):
            return  # Skip flush, will be done by apply_randomization

        if hasattr(self._actual_handler, "flush_visual_updates"):
            self._actual_handler.flush_visual_updates()

    def get_table_bounds(self, env_id: int = 0) -> dict[str, float] | None:
        """Get workspace table bounding box.

        This is a utility method for positioning task objects relative to the table.

        Args:
            env_id: Environment ID to query

        Returns:
            Dict with keys: height, x_min, x_max, y_min, y_max (or None if no workspace)
        """
        if not self.cfg.workspace_layer or not self.cfg.workspace_layer.elements:
            return None

        from pxr import UsdGeom

        # Get first workspace element
        element = self.cfg.workspace_layer.elements[0]

        if self.cfg.workspace_layer.shared:
            prim_path = f"/World/scene_workspace_{element.name}"
        else:
            prim_path = f"/World/envs/env_{env_id}/scene_workspace_{element.name}"

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None

        # For ManualGeometry, use the config directly (more reliable than bounding box)
        if isinstance(element, ManualGeometryCfg):
            pos = element.position
            size = element.size

            # Table bounds from geometry config
            half_x = size[0] / 2
            half_y = size[1] / 2
            height = pos[2] + size[2] / 2  # Top surface

            return {
                "height": float(height),
                "x_min": float(pos[0] - half_x),
                "x_max": float(pos[0] + half_x),
                "y_min": float(pos[1] - half_y),
                "y_max": float(pos[1] + half_y),
            }

        # For USD assets, use bounding box
        bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()

        min_point = bbox_range.GetMin()
        max_point = bbox_range.GetMax()

        return {
            "height": float(max_point[2]),
            "x_min": float(min_point[0]),
            "x_max": float(max_point[0]),
            "y_min": float(min_point[1]),
            "y_max": float(max_point[1]),
        }
