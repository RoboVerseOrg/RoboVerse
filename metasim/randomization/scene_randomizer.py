"""Scene randomizer for domain randomization.

This module provides a 3-layer hierarchical architecture for scene randomization:
- Layer 0 (Environment): Backgrounds, rooms, walls, floors, ceilings
- Layer 1 (Workspace): Tables, desktops, manipulation surfaces
- Layer 2 (Objects): Static distractor objects

Each layer can contain multiple elements, where each element can be either:
- Manual geometry (cube, sphere, etc.) with full material randomization support
- USD asset with optional material randomization

Users are responsible for positioning elements to avoid overlaps.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

from loguru import logger

from metasim.randomization.base import BaseRandomizerType
from metasim.utils.configclass import configclass

# =============================================================================
# Material Pool Configuration (shared)
# =============================================================================


@configclass
class SceneMaterialPoolCfg:
    """Configuration for scene material pools.

    Args:
        material_paths: List of paths to material files (MDL). Can use "path/to/file.mdl::MaterialName"
                       to specify a particular material variant within a file
        selection_strategy: How to select from available materials
        weights: Optional weights for weighted selection
        randomize_material_variant: If True, randomly select from all material variants in each MDL file.
                                   This expands diversity significantly. Default: True to maximize diversity.
                                   Fully reproducible with seed.
    """

    material_paths: list[str] = dataclasses.field(default_factory=list)
    selection_strategy: Literal["random", "sequential", "weighted"] = "random"
    weights: list[float] | None = None
    randomize_material_variant: bool = True

    def __post_init__(self):
        """Validate material pool configuration."""
        if self.selection_strategy == "weighted":
            if self.weights is None or len(self.weights) != len(self.material_paths):
                raise ValueError("weights must be provided and match material_paths length for weighted selection")


# =============================================================================
# Element Configurations (Manual Geometry and USD Asset)
# =============================================================================


@configclass
class ManualGeometryCfg:
    """Manual geometry element configuration.

    Creates procedural geometry (cube, sphere, etc.) with full material randomization support.

    Args:
        name: Unique element name (e.g., "floor", "wall_front", "table")
        geometry_type: Type of primitive geometry
        size: Geometry size (x, y, z) in meters
        position: World position (x, y, z) in meters
        rotation: Orientation as quaternion (w, x, y, z)
        add_collision: Whether to add collision geometry
        fix_base_link: If True, object is static; if False, object is dynamic
        material_randomization: Enable material randomization for this element
        material_pool: Material pool configuration for randomization
        enabled: Whether this element is active
    """

    name: str = dataclasses.MISSING
    geometry_type: Literal["cube", "sphere", "cylinder", "plane"] = "cube"
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    add_collision: bool = True
    fix_base_link: bool = True
    material_randomization: bool = True
    material_pool: SceneMaterialPoolCfg | None = None
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.material_randomization and self.material_pool is None:
            logger.warning(f"ManualGeometryCfg '{self.name}': material_randomization enabled but material_pool is None")


@configclass
class USDAssetCfg:
    """Single USD asset configuration (analogous to a single material path).

    Represents one USD asset with its transform and properties.
    For multiple USD options, use USDAssetPoolCfg.

    Args:
        name: Unique element name (e.g., "table", "room_1")
        usd_path: Path to USD file
        position: World position (x, y, z) in meters
        rotation: Orientation as quaternion (w, x, y, z)
        scale: Scale factor (x, y, z)
        fix_base_link: If True, object is static; if False, object is dynamic
        disable_physics: If True, removes all physics APIs from loaded USD (for visual-only background scenes)
        material_randomization: Enable material randomization for USD prims
        material_target_patterns: List of prim path patterns to apply materials to (e.g., ["*/Looks/*"])
        material_pool: Material pool configuration for randomization
        auto_download: Enable automatic download of missing assets
        enabled: Whether this element is active
    """

    name: str = dataclasses.MISSING
    usd_path: str = dataclasses.MISSING

    # Transform properties
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Physical properties
    fix_base_link: bool = True
    disable_physics: bool = False

    # Material randomization (optional for USD)
    material_randomization: bool = False
    material_target_patterns: list[str] | None = None
    material_pool: SceneMaterialPoolCfg | None = None

    # Auto-download configuration
    auto_download: bool = True

    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not self.usd_path:
            raise ValueError(f"USDAssetCfg '{self.name}': usd_path cannot be empty")

        if self.material_randomization:
            if self.material_pool is None:
                logger.warning(f"USDAssetCfg '{self.name}': material_randomization enabled but material_pool is None")
            if not self.material_target_patterns:
                logger.warning(
                    f"USDAssetCfg '{self.name}': material_randomization enabled but no material_target_patterns specified"
                )


@configclass
class USDAssetPoolCfg:
    """USD asset pool configuration (analogous to SceneMaterialPoolCfg).

    Randomly selects one USD asset from a pool. Supports three usage modes:

    **Mode 1: Shortcut (shared config for all paths)**
        Use `usd_paths` for multiple USD files that share the same transform/properties.
        ```python
        USDAssetPoolCfg(
            name="kujiale_scenes",
            usd_paths=["room1.usd", "room2.usd", "room3.usd"],  # Quick way
            position=(0, 0, 0),     # Shared by all
            disable_physics=True,   # Shared by all
        )
        ```

    **Mode 2: Shortcut with per-path overrides (flexible!)**
        Use `usd_paths` + `per_path_overrides` to override specific paths.
        ```python
        USDAssetPoolCfg(
            name="kujiale_scenes",
            usd_paths=["room1.usd", "room2.usd", "room3.usd"],
            position=(0, 0, 0),     # Default for all
            disable_physics=True,   # Default for all
            per_path_overrides={
                "room2.usd": {"position": (1, 0, 0), "scale": (1.5, 1.5, 1.5)},  # Override room2
                "room3.usd": {"rotation": (0.707, 0, 0, 0.707)},  # Override room3 rotation only
            }
        )
        ```

    **Mode 3: Detailed (full control)**
        Use `candidates` for fine-grained control over each USD asset.
        ```python
        USDAssetPoolCfg(
            name="kujiale_scenes",
            candidates=[
                USDAssetCfg(name="room1", usd_path="room1.usd", position=(0,0,0), scale=(2,2,2)),
                USDAssetCfg(name="room2", usd_path="room2.usd", position=(1,0,0), scale=(1.5,1.5,1.5)),
            ]
        )
        ```

    Args:
        name: Pool name (e.g., "kujiale_scenes")
        usd_paths: Shortcut - list of USD paths (auto-creates USDAssetCfg with shared config)
        per_path_overrides: Dict mapping USD path to override config (only with usd_paths)
                           Keys match entries in usd_paths (exact match or basename match)
                           Values are dicts with optional keys: 'position', 'rotation', 'scale', 'disable_physics', etc.
        candidates: Detailed - list of pre-configured USDAssetCfg objects
        selection_strategy: How to select from pool ("random", "sequential")

        # Shared config (only used with usd_paths shortcut)
        position: Default position for all USD paths
        rotation: Default rotation for all USD paths
        scale: Default scale for all USD paths
        disable_physics: Default disable_physics for all USD paths
        fix_base_link: Default fix_base_link for all USD paths
        auto_download: Default auto_download for all USD paths

        enabled: Whether this pool is active
    """

    name: str = dataclasses.MISSING

    # Mode 1-2: Shortcut (mutually exclusive with candidates)
    usd_paths: list[str] | None = None
    per_path_overrides: dict[str, dict] | None = None  # Override specific paths

    # Mode 3: Detailed (mutually exclusive with usd_paths)
    candidates: list[USDAssetCfg] | None = None

    # Selection strategy
    selection_strategy: Literal["random", "sequential"] = "random"

    # Shared config for usd_paths shortcut (ignored if using candidates)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    disable_physics: bool = False
    fix_base_link: bool = True
    auto_download: bool = True

    enabled: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure exactly one mode is used
        if self.usd_paths is not None and self.candidates is not None:
            raise ValueError(
                f"USDAssetPoolCfg '{self.name}': Cannot specify both 'usd_paths' and 'candidates'. "
                "Use 'usd_paths' for shared config or 'candidates' for individual configs."
            )

        if self.usd_paths is None and self.candidates is None:
            raise ValueError(f"USDAssetPoolCfg '{self.name}': Must specify either 'usd_paths' or 'candidates'")

        # Mode 1-2: Auto-expand usd_paths to candidates (with optional per-path overrides)
        if self.usd_paths is not None:
            if len(self.usd_paths) == 0:
                raise ValueError(f"USDAssetPoolCfg '{self.name}': usd_paths cannot be empty")

            # Warn if per_path_overrides used with candidates
            if self.per_path_overrides and self.candidates:
                logger.warning(f"USDAssetPoolCfg '{self.name}': per_path_overrides ignored when using candidates")

            logger.debug(f"USDAssetPoolCfg '{self.name}': Auto-expanding {len(self.usd_paths)} paths to candidates")

            self.candidates = []
            for i, path in enumerate(self.usd_paths):
                # Start with default config
                cfg_kwargs = {
                    "name": f"{self.name}_{i}",
                    "usd_path": path,
                    "position": self.position,
                    "rotation": self.rotation,
                    "scale": self.scale,
                    "disable_physics": self.disable_physics,
                    "fix_base_link": self.fix_base_link,
                    "auto_download": self.auto_download,
                }

                # Apply per-path overrides if specified
                if self.per_path_overrides:
                    # Try exact match first
                    override = self.per_path_overrides.get(path)

                    # If no exact match, try basename match (e.g., "room1.usd" matches "path/to/room1.usd")
                    if override is None:
                        from pathlib import Path

                        basename = Path(path).name
                        override = self.per_path_overrides.get(basename)

                    # Apply overrides
                    if override:
                        logger.debug(f"  Applying overrides to '{path}': {override}")
                        for key, value in override.items():
                            if key in cfg_kwargs:
                                cfg_kwargs[key] = value
                            else:
                                logger.warning(f"Unknown override key '{key}' for '{path}', ignoring")

                self.candidates.append(USDAssetCfg(**cfg_kwargs))

            # Clear usd_paths to avoid confusion
            self.usd_paths = None

        # Mode 3: Validate candidates
        if self.candidates is not None and len(self.candidates) == 0:
            raise ValueError(f"USDAssetPoolCfg '{self.name}': candidates cannot be empty")


# =============================================================================
# Layer Configurations (3 Layers)
# =============================================================================


@configclass
class EnvironmentLayerCfg:
    """Layer 0: Environment layer configuration.

    Contains background scenes, rooms, walls, floors, ceilings, etc.

    Args:
        elements: List of elements (ManualGeometryCfg, USDAssetCfg, or USDAssetPoolCfg)
        enabled: Whether this layer is active
        z_offset: Z-axis offset applied to all elements in this layer
        apply_to_all_envs: If True, elements are shared across all envs; if False, each env gets its own copy
    """

    elements: list[ManualGeometryCfg | USDAssetCfg | USDAssetPoolCfg] = dataclasses.field(default_factory=list)
    enabled: bool = True
    z_offset: float = 0.0
    apply_to_all_envs: bool = True


@configclass
class WorkspaceLayerCfg:
    """Layer 1: Workspace layer configuration.

    Contains tables, desktops, manipulation surfaces, etc.

    Args:
        elements: List of elements (ManualGeometryCfg, USDAssetCfg, or USDAssetPoolCfg)
        enabled: Whether this layer is active
        z_offset: Z-axis offset applied to all elements in this layer
        apply_to_all_envs: If True, elements are shared across all envs; if False, each env gets its own copy
    """

    elements: list[ManualGeometryCfg | USDAssetCfg | USDAssetPoolCfg] = dataclasses.field(default_factory=list)
    enabled: bool = True
    z_offset: float = 0.0
    apply_to_all_envs: bool = True


@configclass
class ObjectsLayerCfg:
    """Layer 2: Objects layer configuration.

    Contains static distractor objects, decorative items, etc.

    Args:
        elements: List of elements (ManualGeometryCfg, USDAssetCfg, or USDAssetPoolCfg)
        enabled: Whether this layer is active
        z_offset: Z-axis offset applied to all elements in this layer
        apply_to_all_envs: If True, elements are shared across all envs; if False, each env gets its own copy
    """

    elements: list[ManualGeometryCfg | USDAssetCfg | USDAssetPoolCfg] = dataclasses.field(default_factory=list)
    enabled: bool = True
    z_offset: float = 0.0
    apply_to_all_envs: bool = False


# =============================================================================
# Main Scene Randomization Configuration
# =============================================================================


@configclass
class SceneRandomCfg:
    """Scene randomization configuration with 3-layer hierarchical architecture.

    Layers:
        - Layer 0 (Environment): Backgrounds, rooms, walls, floors, ceilings
        - Layer 1 (Workspace): Tables, desktops, manipulation surfaces
        - Layer 2 (Objects): Static distractor objects

    Each layer can contain multiple elements. Each element can be:
        - ManualGeometryCfg: Procedural geometry with full material randomization
        - USDAssetCfg: USD asset with optional material randomization

    Users are responsible for positioning elements to avoid overlaps.

    Args:
        environment_layer: Layer 0 configuration
        workspace_layer: Layer 1 configuration
        objects_layer: Layer 2 configuration
        only_if_no_scene: If True, skip scene creation if scenario already has a scene
        env_ids: List of environment IDs to apply randomization to (None = all)
        auto_flush_visuals: Automatically flush visual updates after material changes
    """

    environment_layer: EnvironmentLayerCfg | None = None
    workspace_layer: WorkspaceLayerCfg | None = None
    objects_layer: ObjectsLayerCfg | None = None

    only_if_no_scene: bool = True
    env_ids: list[int] | None = None
    auto_flush_visuals: bool = True

    def __post_init__(self):
        """Validate scene randomization configuration."""
        enabled_layers = []
        if self.environment_layer is not None and self.environment_layer.enabled:
            enabled_layers.append("environment")
        if self.workspace_layer is not None and self.workspace_layer.enabled:
            enabled_layers.append("workspace")
        if self.objects_layer is not None and self.objects_layer.enabled:
            enabled_layers.append("objects")

        if not enabled_layers:
            logger.warning("No layers enabled in SceneRandomCfg")
        else:
            logger.debug(f"SceneRandomCfg: enabled layers = {enabled_layers}")


# =============================================================================
# Scene Randomizer Implementation
# =============================================================================


class SceneRandomizer(BaseRandomizerType):
    """Scene randomizer with 3-layer hierarchical architecture.

    Creates and randomizes scene elements across three layers:
    - Environment: backgrounds, rooms, walls, floors, ceilings
    - Workspace: tables, desktops, manipulation surfaces
    - Objects: static distractor objects

    Each element can be manual geometry (procedural) or USD asset.
    Manual geometry supports full material randomization from MDL collections.
    USD assets support optional material randomization on specified prims.

    Example:
        >>> from metasim.randomization import SceneRandomizer, SceneRandomCfg
        >>> from metasim.randomization import EnvironmentLayerCfg, ManualGeometryCfg
        >>> cfg = SceneRandomCfg(
        ...     environment_layer=EnvironmentLayerCfg(
        ...         elements=[
        ...             ManualGeometryCfg(
        ...                 name="floor",
        ...                 size=(10.0, 10.0, 0.1),
        ...                 position=(0.0, 0.0, 0.005),
        ...                 material_randomization=True,
        ...                 material_pool=SceneMaterialPoolCfg(material_paths=[...]),
        ...             ),
        ...         ],
        ...     ),
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
        self.cfg = cfg

        # Material selection cache (for random/weighted strategies) and state (for sequential strategy)
        self._material_selection_state = {}

        # Track created prims to avoid recreating
        self._created_prims = set()

        # USD pool selection state (for sequential selection in USDAssetPoolCfg)
        self._usd_pool_selection_state = {}

        # Cache for downloaded USD assets (symmetric to Material's prepared_mdls)
        self._prepared_usds = set()

        # Track which USD is loaded at each prim_path (for replacement detection)
        self._loaded_usds: dict[str, str] = {}

        super().__init__(seed=seed)

        logger.debug(f"SceneRandomizer initialized with seed {self._seed}")

    def set_seed(self, seed: int | None) -> None:
        """Set seed and reset selection state."""
        super().set_seed(seed)
        self._material_selection_state.clear()
        self._usd_pool_selection_state.clear()

    def get_table_bounds(self, env_id: int = 0) -> dict[str, float] | None:
        """Get the bounding box of the workspace table/desktop.

        Args:
            env_id: Environment ID

        Returns:
            Dictionary with 'height' (Z max), 'x_min', 'x_max', 'y_min', 'y_max', 'z_min'
            or None if no table exists
        """
        try:
            from pxr import UsdGeom

            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            # Try multiple possible table locations
            # 1. Shared USD table (created for all envs) - try with and without suffix
            # 2. Per-env USD table
            # 3. Per-env manual geometry table
            # Note: USD tables may have numeric suffixes (_1, _2, etc.) when replaced during randomization
            possible_paths = [
                "/World/scene_workspace_table",  # Shared USD table (base name)
                f"/World/envs/env_{env_id}/scene_workspace_table",  # Per-env USD table
                f"/World/envs/env_{env_id}/scene_table",  # Per-env manual table
            ]

            prim = None
            table_prim_path = None
            logger.debug(f"Searching for table in env {env_id}, trying paths:")

            # First try exact paths
            for path in possible_paths:
                test_prim = prim_utils.get_prim_at_path(path)
                logger.debug(
                    f"  {path}: exists={test_prim is not None}, valid={test_prim.IsValid() if test_prim else False}"
                )
                if test_prim and test_prim.IsValid():
                    prim = test_prim
                    table_prim_path = path
                    logger.info(f"Found table at: {path}")
                    break

            # If not found, search for table with numeric suffix (e.g., table_1, table_2)
            # This happens when USD tables are replaced during randomization
            if not prim or not prim.IsValid():
                stage = prim_utils.get_current_stage()
                for candidate in stage.Traverse():
                    candidate_path = str(candidate.GetPath())
                    # Match /World/scene_workspace_table* or /World/envs/env_X/scene_workspace_table*
                    if (
                        candidate_path.startswith("/World/scene_workspace_table")
                        or candidate_path.startswith(f"/World/envs/env_{env_id}/scene_workspace_table")
                        or candidate_path.startswith(f"/World/envs/env_{env_id}/scene_table")
                    ):
                        if candidate.IsValid():
                            prim = candidate
                            table_prim_path = candidate_path
                            logger.info(f"Found table with suffix at: {candidate_path}")
                            break

            if not prim or not prim.IsValid():
                logger.warning(f"No table found in env {env_id}. Tried paths: {possible_paths}")
                # List all scene prims for debugging
                try:
                    stage = prim_utils.get_current_stage()
                    scene_prims = [p.GetPath() for p in stage.Traverse() if "scene" in str(p.GetPath()).lower()]
                    if scene_prims:
                        logger.debug(f"Available scene prims: {scene_prims[:10]}")  # Show first 10
                except:
                    pass
                return None

            # Compute world-space bounding box
            bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()

            min_point = bbox_range.GetMin()
            max_point = bbox_range.GetMax()

            bounds = {
                "height": float(max_point[2]),  # Table surface Z
                "x_min": float(min_point[0]),
                "x_max": float(max_point[0]),
                "y_min": float(min_point[1]),
                "y_max": float(max_point[1]),
                "z_min": float(min_point[2]),
                "z_max": float(max_point[2]),
            }

            logger.debug(f"Table bounds for env {env_id}: {bounds}")
            return bounds

        except Exception as e:
            logger.error(f"Failed to compute table bounds: {e}")
            logger.debug("Traceback:", exc_info=True)
            return None

    def bind_handler(self, handler):
        """Bind the scene randomizer to a simulation handler.

        Args:
            handler: Simulation handler to bind to
        """
        super().bind_handler(handler)

        # Check if scene exists (for only_if_no_scene logic)
        if self.cfg.only_if_no_scene:
            self._check_scene_exists()

    def _check_scene_exists(self) -> bool:
        """Check if a predefined scene exists in the scenario.

        Returns:
            True if scene exists, False otherwise
        """
        if hasattr(self.handler, "scenario") and hasattr(self.handler.scenario, "scene"):
            if self.handler.scenario.scene is not None:
                logger.info("Predefined scene detected, SceneRandomizer will skip if only_if_no_scene=True")
                return True
            else:
                logger.debug("No predefined scene; SceneRandomizer will create scene elements")
                return False
        return False

    def __call__(self, env_ids: list[int] | None = None):
        """Apply scene randomization across all layers.

        Args:
            env_ids: Optional list of environment IDs to randomize. If None, uses cfg.env_ids
        """
        if not self.handler:
            raise RuntimeError("Handler not bound. Call bind_handler() first.")

        # Clear material selection cache at the start of each randomization cycle
        # This ensures that each randomization gets fresh material selections
        # BUT shared material_pool instances will still get the same material
        self._material_selection_state.clear()

        # Skip if scene exists and only_if_no_scene is True
        if self.cfg.only_if_no_scene and self._check_scene_exists():
            logger.debug("Skipping scene creation due to existing scene and only_if_no_scene=True")
            return

        # Get environment IDs to randomize
        target_env_ids = env_ids if env_ids is not None else self.cfg.env_ids
        if target_env_ids is None:
            target_env_ids = list(range(self.handler.num_envs))

        # Process layers in order
        if self.cfg.environment_layer is not None and self.cfg.environment_layer.enabled:
            self._process_layer(self.cfg.environment_layer, "environment", target_env_ids)

        if self.cfg.workspace_layer is not None and self.cfg.workspace_layer.enabled:
            self._process_layer(self.cfg.workspace_layer, "workspace", target_env_ids)

        if self.cfg.objects_layer is not None and self.cfg.objects_layer.enabled:
            self._process_layer(self.cfg.objects_layer, "objects", target_env_ids)

        # Auto-flush visual updates after material changes (if enabled)
        if self.cfg.auto_flush_visuals:
            self._flush_visual_updates()

    def _process_layer(
        self,
        layer_cfg: EnvironmentLayerCfg | WorkspaceLayerCfg | ObjectsLayerCfg,
        layer_name: str,
        env_ids: list[int],
    ):
        """Process a single layer by creating/updating all its elements.

        Args:
            layer_cfg: Layer configuration
            layer_name: Layer name for prim path construction
            env_ids: List of environment IDs to process
        """
        logger.debug(f"Processing layer '{layer_name}' with {len(layer_cfg.elements)} elements")

        for element in layer_cfg.elements:
            if not element.enabled:
                logger.debug(f"Skipping disabled element '{element.name}' in layer '{layer_name}'")
                continue

            if isinstance(element, ManualGeometryCfg):
                self._process_manual_geometry(element, layer_name, layer_cfg, env_ids)
            elif isinstance(element, USDAssetPoolCfg):
                # Pool: select one candidate and process it
                # CRITICAL: Use pool's name (not candidate's name) for prim path
                # This ensures the same prim path across randomizations, enabling proper USD replacement
                selected_cfg = self._select_from_usd_pool(element)
                if selected_cfg:
                    self._process_usd_asset(selected_cfg, layer_name, layer_cfg, env_ids, override_name=element.name)
            elif isinstance(element, USDAssetCfg):
                self._process_usd_asset(element, layer_name, layer_cfg, env_ids)
            else:
                logger.warning(f"Unknown element type: {type(element)}")

    def _process_manual_geometry(
        self,
        element: ManualGeometryCfg,
        layer_name: str,
        layer_cfg: EnvironmentLayerCfg | WorkspaceLayerCfg | ObjectsLayerCfg,
        env_ids: list[int],
    ):
        """Process a manual geometry element.

        Args:
            element: Manual geometry configuration
            layer_name: Layer name for prim path construction
            layer_cfg: Layer configuration (for z_offset and apply_to_all_envs)
            env_ids: List of environment IDs to process
        """
        if layer_cfg.apply_to_all_envs:
            # Shared across all envs
            prim_path = f"/World/scene_{layer_name}_{element.name}"

            # Create geometry only once
            if prim_path not in self._created_prims:
                self._create_geometry_prim(element, prim_path, layer_cfg.z_offset)
                self._created_prims.add(prim_path)
                logger.debug(f"Created shared manual geometry: {prim_path}")

            # Randomize material every time (if enabled)
            if element.material_randomization and element.material_pool:
                # Use element name prefix as state_key to ensure related elements get the same material
                # (e.g., all "wall_*" elements share the same material)
                # Note: @configclass deepcopies all attributes, so we can't rely on id()
                state_key = element.name.rsplit("_", 1)[0] if "_" in element.name else element.name
                self._apply_material_to_prim(prim_path, element.material_pool, f"{layer_name}_{state_key}")
        else:
            # Per-env copy
            for env_id in env_ids:
                prim_path = f"/World/envs/env_{env_id}/scene_{layer_name}_{element.name}"

                # Create geometry
                if prim_path not in self._created_prims:
                    self._create_geometry_prim(element, prim_path, layer_cfg.z_offset)
                    self._created_prims.add(prim_path)
                    logger.debug(f"Created per-env manual geometry: {prim_path}")

                # Randomize material
                if element.material_randomization and element.material_pool:
                    # Use element name prefix as state_key for consistency
                    state_key = element.name.rsplit("_", 1)[0] if "_" in element.name else element.name
                    self._apply_material_to_prim(prim_path, element.material_pool, f"{layer_name}_{state_key}")

    def _process_usd_asset(
        self,
        element: USDAssetCfg,
        layer_name: str,
        layer_cfg: EnvironmentLayerCfg | WorkspaceLayerCfg | ObjectsLayerCfg,
        env_ids: list[int],
        override_name: str | None = None,
    ):
        """Process a USD asset element with support for USD replacement.

        Args:
            element: USD asset configuration
            layer_name: Layer name for prim path construction
            layer_cfg: Layer configuration (for z_offset and apply_to_all_envs)
            env_ids: List of environment IDs to process
            override_name: Override element.name for prim path (used by USDAssetPoolCfg to ensure consistent prim path)
        """
        # USD path is directly specified in element (no selection needed)
        selected_usd = element.usd_path

        # Use override_name if provided (for pool-based selection), otherwise use element.name
        element_name = override_name if override_name is not None else element.name

        if layer_cfg.apply_to_all_envs:
            # Shared across all envs
            prim_path = f"/World/scene_{layer_name}_{element_name}"

            # Check if USD needs to be replaced
            if prim_path in self._created_prims:
                # Prim exists, check if it's the same USD
                if prim_path in self._loaded_usds and self._loaded_usds[prim_path] != selected_usd:
                    logger.info(f"Replacing USD at {prim_path}: {self._loaded_usds[prim_path]} -> {selected_usd}")
                    self._replace_usd_reference(selected_usd, prim_path, element, layer_cfg.z_offset)
                    self._loaded_usds[prim_path] = selected_usd
            else:
                # First time creation
                self._load_usd_reference(selected_usd, prim_path, element, layer_cfg.z_offset)
                self._created_prims.add(prim_path)
                self._loaded_usds[prim_path] = selected_usd
                logger.debug(f"Created shared USD asset: {prim_path}")

            # Randomize USD materials (if enabled)
            if element.material_randomization and element.material_pool and element.material_target_patterns:
                self._apply_material_to_usd(prim_path, element)
        else:
            # Per-env copy
            for env_id in env_ids:
                prim_path = f"/World/envs/env_{env_id}/scene_{layer_name}_{element_name}"

                # Check if USD needs to be replaced
                if prim_path in self._created_prims:
                    # Prim exists, check if it's the same USD
                    if prim_path in self._loaded_usds and self._loaded_usds[prim_path] != selected_usd:
                        logger.info(f"Replacing USD at {prim_path}: {self._loaded_usds[prim_path]} -> {selected_usd}")
                        self._replace_usd_reference(selected_usd, prim_path, element, layer_cfg.z_offset)
                        self._loaded_usds[prim_path] = selected_usd
                else:
                    # First time creation
                    self._load_usd_reference(selected_usd, prim_path, element, layer_cfg.z_offset)
                    self._created_prims.add(prim_path)
                    self._loaded_usds[prim_path] = selected_usd
                    logger.debug(f"Created per-env USD asset: {prim_path}")

                # Randomize USD materials
                if element.material_randomization and element.material_pool and element.material_target_patterns:
                    self._apply_material_to_usd(prim_path, element)

    def _create_geometry_prim(self, element: ManualGeometryCfg, prim_path: str, z_offset: float):
        """Create a procedural geometry primitive.

        Args:
            element: Manual geometry configuration
            prim_path: USD prim path for the geometry
            z_offset: Z-axis offset to apply
        """
        try:
            # Lazy import IsaacSim modules
            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            from pxr import Gf, UsdGeom, UsdPhysics

            stage = prim_utils.get_current_stage()
            if not stage:
                logger.warning("No stage available for geometry creation")
                return

            # Adjust position with z_offset
            pos = list(element.position)
            pos[2] += z_offset

            # Create geometry based on type
            if element.geometry_type == "cube":
                geom_prim = stage.DefinePrim(prim_path, "Cube")
                geom = UsdGeom.Cube(geom_prim)
                geom.GetSizeAttr().Set(2.0)  # Default USD cube size

                # Scale to desired size
                scale_factor = tuple(s / 2.0 for s in element.size)

            elif element.geometry_type == "sphere":
                geom_prim = stage.DefinePrim(prim_path, "Sphere")
                geom = UsdGeom.Sphere(geom_prim)
                # Sphere radius is average of size dimensions
                avg_radius = (element.size[0] + element.size[1] + element.size[2]) / 3.0
                geom.GetRadiusAttr().Set(avg_radius)
                scale_factor = (1.0, 1.0, 1.0)

            elif element.geometry_type == "cylinder":
                geom_prim = stage.DefinePrim(prim_path, "Cylinder")
                geom = UsdGeom.Cylinder(geom_prim)
                # Use size[0] and size[1] for radius, size[2] for height
                avg_radius = (element.size[0] + element.size[1]) / 2.0
                geom.GetRadiusAttr().Set(avg_radius)
                geom.GetHeightAttr().Set(element.size[2])
                scale_factor = (1.0, 1.0, 1.0)

            elif element.geometry_type == "plane":
                # Plane is essentially a thin cube
                geom_prim = stage.DefinePrim(prim_path, "Cube")
                geom = UsdGeom.Cube(geom_prim)
                geom.GetSizeAttr().Set(2.0)
                scale_factor = tuple(s / 2.0 for s in element.size)

            else:
                logger.warning(f"Unsupported geometry type: {element.geometry_type}")
                return

            # Apply transform
            xform = UsdGeom.Xformable(geom_prim)
            xform.ClearXformOpOrder()

            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3f(*pos))

            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3f(*scale_factor))

            # Apply rotation
            if element.rotation != (1.0, 0.0, 0.0, 0.0):
                orient_op = xform.AddOrientOp()
                orient_op.Set(Gf.Quatf(element.rotation[0], Gf.Vec3f(*element.rotation[1:])))

            # Add physics collision (static geometry)
            if element.add_collision:
                UsdPhysics.CollisionAPI.Apply(geom_prim)

            logger.debug(f"Created {element.geometry_type} geometry at {prim_path}")

        except Exception as e:
            logger.error(f"Failed to create geometry prim {prim_path}: {e}")

    def _load_usd_reference(self, usd_path: str, prim_path: str, element: USDAssetCfg, z_offset: float):
        """Load USD asset as a reference.

        Implements on-demand downloading for USD assets, mimicking Material's behavior.

        Args:
            usd_path: Path to USD file
            prim_path: USD prim path for the reference
            element: USD asset configuration
            z_offset: Z-axis offset to apply
        """
        try:
            from pathlib import Path

            # Lazy import IsaacSim modules
            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            from pxr import Gf, UsdGeom

            # On-demand download if file doesn't exist (with caching to avoid re-downloading)
            if not self._ensure_usd_downloaded(usd_path, self._prepared_usds, element.auto_download):
                logger.warning(f"Failed to download USD asset {usd_path}")
                return

            # Check if we downloaded a complete Kujiale scene folder
            # If so, use the actual downloaded path instead of the requested path
            if hasattr(self, "_kujiale_scene_mapping") and usd_path in self._kujiale_scene_mapping:
                usd_path = self._kujiale_scene_mapping[usd_path]
                logger.debug("Using complete Kujiale scene with dependencies")

            # Convert URDF to USD if necessary
            path_obj = Path(usd_path)
            if path_obj.suffix.lower() == ".urdf":
                converted_usd = self._convert_urdf_to_usd(str(path_obj))
                if not converted_usd:
                    logger.error(f"Failed to convert URDF to USD: {path_obj}")
                    logger.info("Suggestion: Use --scene-mode 0 or 1 to avoid URDF assets")
                    return
                usd_path = converted_usd  # Use converted USD

            stage = prim_utils.get_current_stage()
            if not stage:
                logger.warning("No stage available for USD reference")
                return

            # Create xform prim
            xform_prim = stage.DefinePrim(prim_path, "Xform")

            # Add USD reference (use absolute path for correct asset resolution)
            references = xform_prim.GetReferences()
            abs_usd_path = os.path.abspath(usd_path)
            references.AddReference(abs_usd_path)
            logger.debug(f"Added USD reference: {abs_usd_path}")

            # Apply transform
            # Note: For converted USD files (from URDF), the prim may already have xform ops with double precision
            # We need to handle both cases: adding new ops or using existing ones
            pos = list(element.position)
            pos[2] += z_offset

            xform = UsdGeom.Xformable(xform_prim)

            # Check if there are existing xform ops
            existing_ops = xform.GetOrderedXformOps()

            if existing_ops:
                # Use existing ops (likely from converted USD with double precision)
                # Find and update translate, scale, orient ops
                logger.debug(f"Found {len(existing_ops)} existing xform ops, updating values")

                for op in existing_ops:
                    op_name = op.GetOpName()

                    if "translate" in op_name:
                        # Use the existing precision (likely double)
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
                # No existing ops, create new ones with float precision
                xform.ClearXformOpOrder()

                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3f(*pos))

                scale_op = xform.AddScaleOp()
                scale_op.Set(Gf.Vec3f(*element.scale))

                if element.rotation != (1.0, 0.0, 0.0, 0.0):
                    orient_op = xform.AddOrientOp()
                    orient_op.Set(Gf.Quatf(element.rotation[0], Gf.Vec3f(*element.rotation[1:])))

            # Configure physics properties
            if element.disable_physics:
                # Visual-only: remove all physics APIs
                self._disable_physics_for_prim(xform_prim)
            else:
                # Physics enabled: ensure RigidBodyAPI and set kinematic state
                self._configure_physics_for_prim(xform_prim, fix_base_link=element.fix_base_link)

            logger.debug(f"Loaded USD reference from {usd_path} to {prim_path}")

        except Exception as e:
            logger.error(f"Failed to load USD reference {usd_path} to {prim_path}: {e}")

    def _replace_usd_reference(
        self,
        usd_path: str,
        prim_path: str,
        element: USDAssetCfg,
        z_offset: float,
    ):
        """Replace an existing USD reference with a new one (for randomization).

        Args:
            usd_path: New USD file path
            prim_path: Target prim path (existing)
            element: USD asset configuration
            z_offset: Z-axis offset to apply
        """
        try:
            # Lazy import IsaacSim modules
            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            # Delete the old prim (this will remove the old USD reference)
            if prim_utils.is_prim_path_valid(prim_path):
                prim_utils.delete_prim(prim_path)
                logger.debug(f"Deleted old USD prim: {prim_path}")

            # Load the new USD reference
            self._load_usd_reference(usd_path, prim_path, element, z_offset)
            logger.debug(f"Replaced USD reference at {prim_path}")

        except Exception as e:
            logger.error(f"Failed to replace USD reference at {prim_path}: {e}")

    def _disable_physics_for_prim(self, prim):
        """Recursively disable all physics on a prim and its descendants.

        Removes PhysicsRigidBodyAPI and PhysicsCollisionAPI from all prims,
        making them purely visual without physics simulation.

        Args:
            prim: USD prim to disable physics for
        """
        try:
            from pxr import UsdPhysics

            # Recursively process all descendants
            for descendant in prim.GetAllChildren():
                self._disable_physics_for_prim(descendant)

            # Remove physics APIs from current prim
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                # logger.debug(f"Removed RigidBodyAPI from {prim.GetPath()}")

            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                # logger.debug(f"Removed CollisionAPI from {prim.GetPath()}")

        except Exception as e:
            logger.warning(f"Failed to disable physics for prim {prim.GetPath()}: {e}")

    def _configure_physics_for_prim(self, prim, fix_base_link: bool):
        """Configure physics for USD object (apply to root prim only).

        For dynamic objects (fix_base_link=False), applies RigidBodyAPI to make them fall.
        For static objects (fix_base_link=True), sets kinematicEnabled=True.

        Note: We only apply physics to the ROOT prim, not individual meshes.
        PhysX will automatically handle the collision geometry.

        Args:
            prim: USD root prim (Xform) to configure physics for
            fix_base_link: If True, object is static (kinematic); if False, object is dynamic
        """
        try:
            from pxr import UsdPhysics

            # Only configure the root prim, not descendants
            # PhysX will automatically find collision geometry in descendants

            # Apply CollisionAPI first (required for physics)
            # This tells PhysX to use descendant meshes for collision
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                logger.info(f"Applied CollisionAPI to {prim.GetPath()}")

            # Apply or get RigidBodyAPI on root
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                logger.info(f"Applied RigidBodyAPI to {prim.GetPath()}")
            else:
                rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
                logger.debug(f"RigidBodyAPI already exists on {prim.GetPath()}")

            # Set kinematic state
            kinematic_attr = rigid_body_api.GetKinematicEnabledAttr()
            if not kinematic_attr:
                kinematic_attr = rigid_body_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(fix_base_link)

            # For dynamic objects, ensure mass is set
            if not fix_base_link:
                mass_attr = rigid_body_api.GetMassAttr()
                if not mass_attr or mass_attr.Get() == 0.0:
                    if not mass_attr:
                        mass_attr = rigid_body_api.CreateMassAttr()
                    mass_attr.Set(1.0)  # Default 1kg
                    logger.info(f"Set mass=1.0 for dynamic object {prim.GetPath()}")

            logger.info(f"Configured physics: kinematic={fix_base_link}, prim={prim.GetPath()}")

        except Exception as e:
            logger.warning(f"Error configuring physics for {prim.GetPath()}: {e}")

    def _convert_urdf_to_usd(self, urdf_path: str) -> str | None:
        """Convert URDF to USD using the same converter as download_asset.py.

        Uses AssetConverterFactory with MESH source type to ensure compatibility
        with EmbodiedGen assets.

        Args:
            urdf_path: Path to URDF file

        Returns:
            Path to converted USD file, or None if conversion failed
        """
        from pathlib import Path

        try:
            urdf_path_obj = Path(urdf_path)
            usd_output = urdf_path_obj.parent / (urdf_path_obj.stem + ".usd")

            # If already converted, use existing USD
            if usd_output.exists():
                logger.info(f"Using existing converted USD: {usd_output}")
                return str(usd_output)

            logger.info(f"Converting URDF to USD: {urdf_path}")

            # Use the same conversion approach as download_asset.py
            # This ensures compatibility with EmbodiedGen assets
            try:
                from generation.asset_converter import AssetConverterFactory
                from generation.enums import AssetType

                # Create converter (same as download_asset.py line 72-76)
                # simulation_app is already running, so pass None and don't close it
                converter = AssetConverterFactory.create(
                    target_type=AssetType.USD,
                    source_type=AssetType.MESH,  # Key: MESH not URDF!
                    simulation_app=None,  # Already running
                    exit_close=False,  # Don't close the app
                    force_usd_conversion=True,
                    make_instanceable=True,
                )

                # Convert URDF -> USD (extracts mesh and converts)
                converter.convert(str(urdf_path), str(usd_output))

                if not usd_output.exists():
                    logger.error(f"USD file was not created: {usd_output}")
                    return None

                logger.info(f"Successfully converted URDF to USD: {usd_output}")
                return str(usd_output)

            except ImportError as e:
                logger.error(f"Required modules not available: {e}")
                logger.error("Please ensure Isaac Lab and generation modules are installed")
                return None
            except Exception as e:
                logger.error(f"URDF to USD conversion failed: {e}")
                logger.debug("Full traceback:", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"URDF conversion error for {urdf_path}: {e}")
            return None

    def _apply_material_to_prim(self, prim_path: str, material_pool: SceneMaterialPoolCfg, state_key: str):
        """Apply material randomization to a prim.

        Args:
            prim_path: USD prim path
            material_pool: Material pool configuration
            state_key: Key for tracking selection state
        """
        material_path = self._select_material(material_pool, state_key)
        if material_path:
            self._apply_mdl_material(material_path, prim_path, material_pool)
        else:
            logger.warning(f"No material selected for {prim_path} (pool may be empty)")

    def _apply_material_to_usd(self, base_prim_path: str, element: USDAssetCfg):
        """Apply material randomization to USD asset prims matching patterns.

        Args:
            base_prim_path: Base USD prim path
            element: USD asset configuration
        """
        if not element.material_target_patterns or not element.material_pool:
            return

        try:
            # Find matching prims
            matching_prims = []
            for pattern in element.material_target_patterns:
                prims = self._find_prims_by_pattern(base_prim_path, pattern)
                matching_prims.extend(prims)

            # Apply material to each matching prim
            for prim_path in matching_prims:
                # Use element name as state_key for USD assets
                self._apply_material_to_prim(prim_path, element.material_pool, f"usd_{element.name}")

        except Exception as e:
            logger.warning(f"Failed to apply USD material randomization: {e}")

    def _find_prims_by_pattern(self, base_path: str, pattern: str) -> list[str]:
        """Find prims matching a glob pattern.

        Args:
            base_path: Base prim path to search from
            pattern: Glob pattern (e.g., "*/Looks/*")

        Returns:
            List of matching prim paths
        """
        try:
            import fnmatch

            try:
                import omni.isaac.core.utils.prims as prim_utils
            except ModuleNotFoundError:
                import isaacsim.core.utils.prims as prim_utils

            stage = prim_utils.get_current_stage()
            if not stage:
                return []

            base_prim = stage.GetPrimAtPath(base_path)
            if not base_prim:
                return []

            matching = []
            for prim in base_prim.GetAllChildren():
                rel_path = str(prim.GetPath()).replace(base_path, "")
                if fnmatch.fnmatch(rel_path, pattern):
                    matching.append(str(prim.GetPath()))

            return matching

        except Exception as e:
            logger.warning(f"Failed to find prims by pattern: {e}")
            return []

    def _select_from_usd_pool(self, pool: USDAssetPoolCfg) -> USDAssetCfg | None:
        """Select a USD asset configuration from the pool based on selection strategy.

        Args:
            pool: USD asset pool configuration

        Returns:
            Selected USDAssetCfg or None
        """
        if not pool.candidates:
            logger.warning(f"USD pool '{pool.name}' has no candidates!")
            return None

        # Use pool name as state key for tracking sequential index
        state_key = f"usd_pool_{pool.name}"

        if pool.selection_strategy == "random":
            # Random selection (no caching needed as pool selection happens once per randomization cycle)
            selected = self._rng.choice(pool.candidates)
            logger.debug(f"USD pool '{pool.name}': randomly selected '{selected.name}'")
            return selected

        elif pool.selection_strategy == "sequential":
            # Sequential selection with wrapping
            if state_key not in self._usd_pool_selection_state:
                self._usd_pool_selection_state[state_key] = 0
            idx = self._usd_pool_selection_state[state_key] % len(pool.candidates)
            self._usd_pool_selection_state[state_key] += 1
            selected = pool.candidates[idx]
            logger.debug(f"USD pool '{pool.name}': sequentially selected '{selected.name}' (index {idx})")
            return selected

        return None

    def _select_material(self, material_pool: SceneMaterialPoolCfg, state_key: str) -> str | None:
        """Select a material from the pool based on selection strategy.

        Args:
            material_pool: Material pool configuration
            state_key: Key for tracking selection state (used for caching random/weighted selections
                      and indexing sequential selections). State key is derived from element name
                      prefix (e.g., all "wall_*" elements share "environment_wall" as state_key),
                      ensuring consistent material selection within a single randomization cycle.

        Returns:
            Selected material path or None
        """
        if not material_pool.material_paths:
            logger.warning(f"Material pool for '{state_key}' is empty!")
            return None

        if material_pool.selection_strategy == "random":
            # For random strategy, cache the selection per state_key within a single randomization cycle
            # This ensures that elements with the same name prefix get the same material
            if state_key not in self._material_selection_state:
                self._material_selection_state[state_key] = self._rng.choice(material_pool.material_paths)
            return self._material_selection_state[state_key]

        elif material_pool.selection_strategy == "sequential":
            if state_key not in self._material_selection_state:
                self._material_selection_state[state_key] = 0
            idx = self._material_selection_state[state_key] % len(material_pool.material_paths)
            self._material_selection_state[state_key] += 1
            return material_pool.material_paths[idx]

        elif material_pool.selection_strategy == "weighted":
            # For weighted strategy, also cache to ensure consistency
            if state_key not in self._material_selection_state:
                self._material_selection_state[state_key] = self._rng.choices(
                    material_pool.material_paths, weights=material_pool.weights, k=1
                )[0]
            return self._material_selection_state[state_key]

        return None

    def _apply_mdl_material(self, material_path: str, prim_path: str, pool_cfg: SceneMaterialPoolCfg):
        """Apply MDL material to a prim with optional material variant randomization.

        Args:
            material_path: Path to MDL file
            prim_path: USD prim path
            pool_cfg: Material pool configuration
        """
        from metasim.randomization.material_randomizer import MaterialRandomizer, list_materials_in_mdl

        # Convert to absolute path
        material_path = os.path.abspath(material_path)

        # Download MDL and textures using MaterialRandomizer's shared method
        dummy_randomizer = MaterialRandomizer.__new__(MaterialRandomizer)
        dummy_randomizer.handler = self.handler

        prepared_mdls = set()
        if not dummy_randomizer._ensure_mdl_downloaded(material_path, prepared_mdls, auto_download=True):
            logger.warning(f"Failed to download MDL or textures for {material_path}")
            return

        # Select material variant if enabled
        material_name = None
        if pool_cfg.randomize_material_variant and "::" not in material_path:
            available_materials = list_materials_in_mdl(material_path)
            if len(available_materials) > 1:
                material_name = self._rng.choice(available_materials)
                logger.debug(
                    f"Selected material variant '{material_name}' from "
                    f"{len(available_materials)} variants in {os.path.basename(material_path)}"
                )

        # Apply material
        dummy_randomizer._apply_mdl_to_prim(material_path, prim_path, material_name)

    def _flush_visual_updates(self):
        """Flush visual updates to ensure material changes are applied."""
        self._mark_visual_dirty()
        flush_fn = getattr(self.handler, "flush_visual_updates", None)
        if callable(flush_fn):
            try:
                flush_fn(wait_for_materials=True, settle_passes=2)
                logger.debug("Visual updates flushed")
            except Exception as e:
                logger.debug(f"Failed to flush visual updates: {e}")

    def _mark_visual_dirty(self):
        """Mark visual cache as dirty (handler-specific)."""
        if hasattr(self.handler, "_visual_cache_dirty"):
            self.handler._visual_cache_dirty = True

    def get_scene_properties(self) -> dict:
        """Get current scene properties.

        Returns:
            Dictionary containing scene element properties
        """
        return {
            "created_prims": list(self._created_prims),
            "num_elements": len(self._created_prims),
            "layers": {
                "environment": self.cfg.environment_layer.enabled if self.cfg.environment_layer else False,
                "workspace": self.cfg.workspace_layer.enabled if self.cfg.workspace_layer else False,
                "objects": self.cfg.objects_layer.enabled if self.cfg.objects_layer else False,
            },
        }

    def _ensure_usd_downloaded(self, usd_path: str, prepared_usds: set[str], auto_download: bool) -> bool:
        """Download USD/URDF file and its dependencies if missing.

        For URDF files from EmbodiedGen, downloads the entire asset folder (URDF + mesh/ + textures).
        This method is symmetric to MaterialRandomizer._ensure_mdl_downloaded().

        Args:
            usd_path: Absolute path to USD/URDF file
            prepared_usds: Set of already prepared USD/URDF paths (for caching)
            auto_download: Whether to auto-download missing files

        Returns:
            True if USD/URDF and dependencies are available, False if download failed
        """
        # If already processed, return True
        if usd_path in prepared_usds:
            return True

        # Special handling for Kujiale scenes: ALWAYS download complete folder and set mapping
        # Even if the roboverse_data version exists, we need the InteriorAgent version with Meshes/
        if usd_path.endswith(".usda") and "kujiale" in usd_path.lower():
            success = self._download_kujiale_scene_folder(usd_path)
            if success:
                prepared_usds.add(usd_path)
            return success

        # Download the USD/URDF file if it doesn't exist
        if not os.path.exists(usd_path):
            if not auto_download:
                logger.warning(f"USD/URDF file not found and auto_download is disabled: {usd_path}")
                return False

            # For URDF files from EmbodiedGen, download entire folder
            if usd_path.endswith(".urdf") and "EmbodiedGenData" in usd_path:
                success = self._download_embodiedgen_asset_folder(usd_path)
                if success:
                    prepared_usds.add(usd_path)
                return success

            # For other files, download single file
            try:
                from metasim.utils.hf_util import check_and_download_single

                logger.info(f"Downloading USD/URDF file: {os.path.basename(usd_path)}")
                check_and_download_single(usd_path)
            except Exception as e:
                logger.warning(f"Failed to download USD/URDF {usd_path}: {e}")
                return False

        prepared_usds.add(usd_path)
        return True

    def _download_embodiedgen_asset_folder(self, urdf_path: str) -> bool:
        """Download entire EmbodiedGen asset folder using snapshot_download (like download_asset.py).

        EmbodiedGen assets are organized as complete folders:
            dataset/basic_furniture/table/<uuid>/
                ├── <uuid>.urdf
                └── mesh/
                    ├── model.obj (or other .obj files)
                    └── texture.png

        Args:
            urdf_path: Local path to URDF file (e.g., EmbodiedGenData/dataset/.../uuid.urdf)

        Returns:
            True if folder downloaded successfully, False otherwise
        """
        from pathlib import Path

        from huggingface_hub import snapshot_download

        try:
            path_obj = Path(urdf_path)
            parts = path_obj.parts

            # Find EmbodiedGenData in path
            if "EmbodiedGenData" not in parts:
                logger.error(f"Path does not appear to be from EmbodiedGen: {urdf_path}")
                return False

            embodiedgen_idx = parts.index("EmbodiedGenData")

            # Extract: dataset/basic_furniture/table/<uuid>
            asset_folder_parts = parts[embodiedgen_idx + 1 : -1]  # Skip 'EmbodiedGenData' and filename
            asset_folder_remote = "/".join(asset_folder_parts)  # e.g., dataset/basic_furniture/table/uuid

            # Local base directory: EmbodiedGenData
            local_base = Path(parts[0])
            for i in range(1, embodiedgen_idx + 1):
                local_base = local_base / parts[i]

            logger.info(f"Downloading EmbodiedGen asset folder: {asset_folder_remote}")

            # Download entire folder (URDF + mesh/ + all contents)
            snapshot_download(
                repo_id="HorizonRobotics/EmbodiedGenData",
                repo_type="dataset",
                local_dir=str(local_base),
                allow_patterns=[f"{asset_folder_remote}/*"],  # Download entire folder
                local_dir_use_symlinks=False,
            )

            # Verify URDF file was downloaded
            if not path_obj.exists():
                logger.error(f"URDF file not found after download: {urdf_path}")
                return False

            logger.info("Successfully downloaded EmbodiedGen asset folder with all dependencies")
            return True

        except Exception as e:
            logger.error(f"Failed to download EmbodiedGen asset folder: {e}")
            logger.debug("Traceback:", exc_info=True)
            return False

    def _download_kujiale_scene_folder(self, usda_path: str) -> bool:
        """Download entire Kujiale scene folder from InteriorAgent dataset.

        Kujiale scenes are organized as complete folders in spatialverse/InteriorAgent:
            kujiale_0032/
                ├── kujiale_0032.usda      ← Main scene file
                ├── Meshes/                ← Sub-USD meshes (pillow_0032.usd, etc.)
                ├── Materials/             ← Material files (.mdl, textures)
                └── buikslotermeerplein_4k.hdr

        Args:
            usda_path: Local path to USDA file (e.g., roboverse_data/scenes/kujiale/032.usda)

        Returns:
            True if folder downloaded successfully, False otherwise
        """
        from pathlib import Path

        from huggingface_hub import snapshot_download

        try:
            path_obj = Path(usda_path)

            # Extract scene number from filename (e.g., "032.usda" -> "032")
            scene_name = path_obj.stem
            if not scene_name.isdigit():
                logger.error(f"Cannot extract scene number from filename: {path_obj.name}")
                return False

            # Construct remote folder name with 4-digit padding (e.g., "kujiale_0032")
            # HuggingFace uses 4-digit format: kujiale_0003, kujiale_0032, etc.
            scene_num = int(scene_name)
            remote_folder = f"kujiale_{scene_num:04d}"

            # Download to InteriorAgent location
            # InteriorAgent dataset structure: kujiale_0032/<files>
            local_temp_base = Path("third_party/InteriorAgent")
            downloaded_folder = local_temp_base / remote_folder
            downloaded_usda = downloaded_folder / f"{remote_folder}.usda"

            # Check if already downloaded
            if downloaded_usda.exists() and (downloaded_folder / "Meshes").exists():
                logger.info(f"Kujiale scene already downloaded: {downloaded_folder}")
            else:
                local_temp_base.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading Kujiale scene folder: {remote_folder}")
                logger.info("  This includes .usda, Meshes/, Materials/, and HDR files")

                # Download entire scene folder from spatialverse/InteriorAgent
                snapshot_download(
                    repo_id="spatialverse/InteriorAgent",
                    repo_type="dataset",
                    local_dir=str(local_temp_base),
                    allow_patterns=[f"{remote_folder}/*"],
                    local_dir_use_symlinks=False,
                )

            # Verify downloaded folder exists
            if not downloaded_folder.exists():
                logger.error(f"Scene folder not found after download: {downloaded_folder}")
                return False

            # Verify USDA file exists
            if not downloaded_usda.exists():
                # Try alternative naming
                possible_usdas = list(downloaded_folder.glob("*.usda"))
                if possible_usdas:
                    downloaded_usda = possible_usdas[0]
                else:
                    logger.error(f"USDA file not found in downloaded folder: {downloaded_folder}")
                    return False

            # Replace with RoboVerse optimized USDA if available
            # RoboVerse version has optimized scene structure (fewer objects, simplified settings)
            roboverse_usda = Path(usda_path)
            if roboverse_usda.exists():
                import shutil

                backup_usda = downloaded_usda.with_suffix(".usda.original")
                if not backup_usda.exists():
                    shutil.copy2(downloaded_usda, backup_usda)
                    logger.debug(f"Backed up original InteriorAgent USDA: {backup_usda}")

                shutil.copy2(roboverse_usda, downloaded_usda)
                logger.info(f"Replaced with RoboVerse optimized USDA: {roboverse_usda.name}")
            else:
                logger.debug("RoboVerse optimized USDA not found, using InteriorAgent version")

            # Set up mapping for Kujiale scenes to use the complete version with dependencies
            if not hasattr(self, "_kujiale_scene_mapping"):
                self._kujiale_scene_mapping = {}

            # Store mapping with multiple path formats
            self._kujiale_scene_mapping[usda_path] = str(downloaded_usda)
            self._kujiale_scene_mapping[str(path_obj.absolute())] = str(downloaded_usda)
            self._kujiale_scene_mapping[str(path_obj)] = str(downloaded_usda)

            logger.debug(f"Mapped Kujiale scene: {usda_path} → {downloaded_usda}")

            return True

        except Exception as e:
            logger.error(f"Failed to download Kujiale scene folder: {e}")
            logger.debug("Traceback:", exc_info=True)
            return False
