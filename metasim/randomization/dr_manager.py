"""
Unified Domain Randomization Manager for Demo Collection and Policy Evaluation.

This module provides a centralized DR system that can be used by:
- collect_demo.py (demo data collection)
- IL evaluation runners (ACT, Diffusion Policy)
- 12_domain_randomization.py (continues to use individual randomizers directly)

Architecture:
- Static Objects: Handler-managed (Robot, task objects, Camera, Light)
- Dynamic Objects: SceneRandomizer-managed (Floor, Table, Distractors)
- Level system: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera
- Mode system: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from loguru import logger as log

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

# Try to import randomization components
try:
    from .camera_randomizer import (
        CameraIntrinsicsRandomCfg,
        CameraPositionRandomCfg,
        CameraRandomCfg,
        CameraRandomizer,
    )
    from .light_randomizer import (
        LightColorRandomCfg,
        LightIntensityRandomCfg,
        LightOrientationRandomCfg,
        LightPositionRandomCfg,
        LightRandomCfg,
        LightRandomizer,
    )
    from .material_randomizer import MaterialRandomizer
    from .scene_randomizer import (
        EnvironmentLayerCfg,
        ManualGeometryCfg,
        ObjectsLayerCfg,
        SceneRandomCfg,
        SceneRandomizer,
        USDAssetPoolCfg,
        WorkspaceLayerCfg,
    )
    from .presets.camera_presets import CameraProperties
    from .presets.material_presets import MaterialPresets
    from .presets.scene_presets import ScenePresets, SceneUSDCollections

    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Randomization components not available: {e}")
    RANDOMIZATION_AVAILABLE = False


@dataclass
class DRConfig:
    """Domain Randomization configuration.
    
    Attributes:
        level: Randomization level (0=None, 1=Scene+Material, 2=+Light, 3=+Camera)
        scene_mode: Scene mode (0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD)
        randomization_seed: Seed for reproducibility. If None, uses random seed
    """

    level: Literal[0, 1, 2, 3] = 0
    scene_mode: Literal[0, 1, 2, 3] = 0
    randomization_seed: int | None = None


class DomainRandomizationManager:
    """Unified Domain Randomization Manager.
    
    Can be used for both demo collection and policy evaluation.
    """

    def __init__(
        self,
        config: DRConfig,
        scenario: ScenarioCfg,
        handler,
        init_states: list | None = None,
        render_cfg=None,
    ):
        """Initialize DR Manager.
        
        Args:
            config: DR configuration
            scenario: Scenario configuration
            handler: Simulation handler
            init_states: Initial states for position adjustment (optional)
            render_cfg: Render configuration for light intensity adjustment (optional)
        """
        self.config = config
        self.scenario = scenario
        self.handler = handler
        self.init_states = init_states or []
        self.render_cfg = render_cfg
        self.randomizers = {}

        # Store original camera positions BEFORE any randomization
        self.original_camera_positions = {}
        for camera in self.handler.cameras:
            self.original_camera_positions[camera.name] = {
                "pos": tuple(camera.pos),
                "look_at": tuple(camera.look_at),
            }

        # Store original positions for ALL demos (for position adjustment)
        self.original_positions = {}
        if init_states:
            for demo_idx, init_state in enumerate(init_states):
                demo_key = f"demo_{demo_idx}"
                self.original_positions[demo_key] = {}

                if "objects" in init_state:
                    for obj_name, obj_state in init_state["objects"].items():
                        self.original_positions[demo_key][f"obj_{obj_name}"] = {
                            "x": float(obj_state["pos"][0]),
                            "y": float(obj_state["pos"][1]),
                            "z": float(obj_state["pos"][2]),
                        }

                if "robots" in init_state:
                    for robot_name, robot_state in init_state["robots"].items():
                        self.original_positions[demo_key][f"robot_{robot_name}"] = {
                            "x": float(robot_state["pos"][0]),
                            "y": float(robot_state["pos"][1]),
                            "z": float(robot_state["pos"][2]),
                        }

        # Early validation
        if not self._validate_setup():
            return

        log.info("=" * 50)
        log.info("DOMAIN RANDOMIZATION SETUP: Initializing randomizers")
        self._setup_randomizers()
        log.info(f"Setup complete: Randomizers ready (Level {config.level}, Mode {config.scene_mode})")

    def _validate_setup(self) -> bool:
        """Validate if randomization can be set up."""
        if self.config.level == 0:
            log.info("Domain randomization disabled (level=0)")
            return False

        if not RANDOMIZATION_AVAILABLE:
            log.warning("Domain randomization requested but components not available")
            return False

        return True

    def _setup_randomizers(self):
        """Initialize all randomizers based on level and mode."""
        seed = self.config.randomization_seed
        self._setup_reproducibility(seed)

        self.randomizers = {
            "scene": None,
            "material_dynamic": [],
            "light": [],
            "camera": [],
        }

        # Scene Randomizer
        self._setup_scene_randomizer(seed)

        # Material Randomization
        self._setup_material_randomizers(seed)

        # Light Randomization (Level 2+)
        if self.config.level >= 2:
            self._setup_light_randomizers(seed)

        # Camera Randomization (Level 3+)
        if self.config.level >= 3:
            self._setup_camera_randomizers(seed)

    def _setup_reproducibility(self, seed: int | None):
        """Setup global reproducibility if seed is provided."""
        if seed is not None:
            log.info(f"Setting up reproducible randomization with seed: {seed}")
            torch.manual_seed(seed)
            import random

            import numpy as np

            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def _setup_scene_randomizer(self, seed: int | None):
        """Setup SceneRandomizer based on scene_mode."""
        mode = self.config.scene_mode
        level = self.config.level

        log.info(f"\nScene Creation (Mode {mode})")
        log.info("-" * 50)

        # Environment Layer
        if mode >= 2:
            # USD Scene
            scene_paths, scene_configs = SceneUSDCollections.kujiale_scenes(return_configs=True)
            log.info(f"Environment: Kujiale USD ({len(scene_paths)} scenes)")
            env_element = USDAssetPoolCfg(
                name="kujiale_scene",
                usd_paths=scene_paths,
                per_path_overrides=scene_configs,
                selection_strategy="random" if level >= 1 else "sequential",
            )
            environment_layer = EnvironmentLayerCfg(elements=[env_element])
        else:
            # Manual Scene
            log.info("Environment: Manual geometry (10m x 10m x 5m)")
            base_cfg = ScenePresets.empty_room(room_size=10.0, wall_height=5.0)
            environment_layer = base_cfg.environment_layer

        # Workspace Layer
        if mode >= 1:
            # USD Table
            table_paths, table_configs = SceneUSDCollections.table785(return_configs=True)
            log.info(f"Workspace: Table785 USD ({len(table_paths)} tables)")
            workspace_element = USDAssetPoolCfg(
                name="table",
                usd_paths=table_paths,
                per_path_overrides=table_configs,
                selection_strategy="random" if level >= 1 else "sequential",
                add_collision=True,
            )
            workspace_layer = WorkspaceLayerCfg(elements=[workspace_element])
        else:
            # Manual Table
            log.info("Workspace: Manual table (Plywood default, randomized in level 1+)")
            workspace_layer = WorkspaceLayerCfg(
                elements=[
                    ManualGeometryCfg(
                        name="table",
                        geometry_type="cube",
                        size=(1.8, 1.8, 0.1),
                        position=(0.0, 0.0, 0.7 - 0.05),
                        default_material="roboverse_data/materials/arnold/Wood/Plywood.mdl",
                        add_collision=True,
                    )
                ]
            )

        # Objects Layer
        if mode >= 3:
            object_paths, object_configs = SceneUSDCollections.desktop_supplies(return_configs=True)
            log.info(f"Objects: Desktop supplies ({len(object_paths)} items, placing 3)")
            objects_layer = ObjectsLayerCfg(
                elements=[
                    USDAssetPoolCfg(
                        name=f"desktop_object_{i + 1}",
                        usd_paths=object_paths,
                        per_path_overrides=object_configs,
                        selection_strategy="random" if level >= 1 else "sequential",
                        add_collision=True,
                    )
                    for i in range(3)
                ]
            )
        else:
            objects_layer = None

        # Create SceneRandomizer
        scene_cfg = SceneRandomCfg(
            environment_layer=environment_layer,
            workspace_layer=workspace_layer,
            objects_layer=objects_layer,
        )

        scene_rand = SceneRandomizer(scene_cfg, seed=seed)
        scene_rand.bind_handler(self.handler)
        self.randomizers["scene"] = scene_rand
        log.info("SceneRandomizer created")

    def _setup_material_randomizers(self, seed: int | None):
        """Setup material randomizers for dynamic objects (environment)."""
        mode = self.config.scene_mode
        level = self.config.level

        if level == 0:
            return

        log.info("\nMaterial Randomization")
        log.info("-" * 50)

        # Dynamic Objects (Manual geometry only)
        if mode == 0:
            # Manual table
            table_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object("table", family=("wood", "metal")),
                seed=seed + 2 if seed is not None else None,
            )
            table_mat.bind_handler(self.handler)
            self.randomizers["material_dynamic"].append(table_mat)
            log.info("  Dynamic Object: table (Manual)")

        # Manual environment (mode < 2 and level >= 1)
        if mode < 2 and level >= 1:
            # Floor
            floor_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object("floor", family=("carpet", "wood", "stone")),
                seed=seed + 101 if seed is not None else None,
            )
            floor_mat.bind_handler(self.handler)
            self.randomizers["material_dynamic"].append(floor_mat)

            # Walls
            wall_seed = seed + 102 if seed is not None else None
            for wall_name in ["wall_front", "wall_back", "wall_left", "wall_right"]:
                wall_mat = MaterialRandomizer(
                    MaterialPresets.mdl_family_object(wall_name, family=("masonry", "architecture")),
                    seed=wall_seed,
                )
                wall_mat.bind_handler(self.handler)
                self.randomizers["material_dynamic"].append(wall_mat)

            # Ceiling
            ceiling_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object("ceiling", family=("architecture", "wall_board")),
                seed=seed + 103 if seed is not None else None,
            )
            ceiling_mat.bind_handler(self.handler)
            self.randomizers["material_dynamic"].append(ceiling_mat)

            log.info("  Dynamic Objects: floor + 4 walls + ceiling")

    def _setup_light_randomizers(self, seed: int | None):
        """Setup light randomizers (Level 2+)."""
        from metasim.scenario.lights import DiskLightCfg, DomeLightCfg, SphereLightCfg

        log.info("\nLight Randomization")
        log.info("-" * 50)

        lights = getattr(self.scenario, "lights", [])
        if not lights:
            log.info("  No lights found")
            return

        # Determine intensity ranges based on render mode
        if self.render_cfg and hasattr(self.render_cfg, "mode") and self.render_cfg.mode == "pathtracing":
            main_range = (22000.0, 40000.0)
            corner_range = (10000.0, 18000.0)
        else:
            main_range = (16000.0, 30000.0)
            corner_range = (6000.0, 12000.0)

        for i, light in enumerate(lights):
            light_name = getattr(light, "name", f"light_{i}")

            if isinstance(light, DiskLightCfg):
                # Main ceiling light with orientation
                light_rand = LightRandomizer(
                    LightRandomCfg(
                        light_name=light_name,
                        intensity=LightIntensityRandomCfg(intensity_range=main_range, enabled=True),
                        color=LightColorRandomCfg(
                            temperature_range=(3000.0, 6000.0), use_temperature=True, enabled=True
                        ),
                        orientation=LightOrientationRandomCfg(
                            angle_range=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
                            relative_to_origin=True,
                            distribution="uniform",
                            enabled=True,
                        ),
                    ),
                    seed=seed + 4 + i if seed is not None else None,
                )
            elif isinstance(light, SphereLightCfg):
                # Corner lights with position and color
                light_rand = LightRandomizer(
                    LightRandomCfg(
                        light_name=light_name,
                        intensity=LightIntensityRandomCfg(intensity_range=corner_range, enabled=True),
                        color=LightColorRandomCfg(
                            temperature_range=(2700.0, 5500.0), use_temperature=True, enabled=True
                        ),
                        position=LightPositionRandomCfg(
                            position_range=((-0.5, 0.5), (-0.5, 0.5), (-0.3, 0.3)),
                            relative_to_origin=True,
                            distribution="uniform",
                            enabled=True,
                        ),
                    ),
                    seed=seed + 5 + i if seed is not None else None,
                )
            elif isinstance(light, DomeLightCfg):
                # Dome light (ambient)
                from metasim.randomization.presets.light_presets import LightPresets

                config = LightPresets.dome_ambient(light_name)
                light_rand = LightRandomizer(config, seed=seed + 4 + i if seed else None)
            else:
                log.warning(f"  Unknown light type: {light_name}")
                continue

            light_rand.bind_handler(self.handler)
            self.randomizers["light"].append(light_rand)
            log.info(f"  Light: {light_name}")

    def _setup_camera_randomizers(self, seed: int | None):
        """Setup camera randomizers (Level 3+)."""
        log.info("\nCamera Randomization")
        log.info("-" * 50)

        cameras = getattr(self.scenario, "cameras", [])
        if not cameras:
            log.info("  No cameras found")
            return

        for camera in cameras:
            camera_name = getattr(camera, "name", "camera")

            # Orbit camera configuration (no roll to keep camera horizontal)
            # Z range kept positive to ensure camera stays above table (no upward view)
            cam_config = CameraRandomCfg(
                camera_name=camera_name,
                position=CameraPositionRandomCfg(
                    delta_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.2)),  # Z: only upward movement
                    use_delta=True,
                    distribution="uniform",
                    enabled=True,
                ),
                intrinsics=CameraIntrinsicsRandomCfg(
                    fov_range=CameraProperties.FOV_NORMAL,
                    use_fov=True,
                    distribution="uniform",
                    enabled=True,
                ),
            )

            cam_rand = CameraRandomizer(cam_config, seed=seed + 10 if seed is not None else None)
            cam_rand.bind_handler(self.handler)
            self.randomizers["camera"].append(cam_rand)
            log.info(f"  Camera: {camera_name}")

    def apply_randomization(self, demo_idx: int = 0, is_initial: bool = False):
        """Apply randomization with global deferred visual flush.

        Args:
            demo_idx: Demo index for logging
            is_initial: Whether this is the initial call (always creates scene)
        """
        if self.config.level == 0 or not self.randomizers:
            return

        log.info("=" * 50)
        log.info(f"DOMAIN RANDOMIZATION: Demo {demo_idx}")

        # Enable global defer flag
        if self.handler:
            self.handler._defer_all_visual_flushes = True

        try:
            # Scene creation/switching
            if self.randomizers["scene"]:
                if is_initial or self.config.level >= 1:
                    scene_rand = self.randomizers["scene"]
                    original_auto_flush = scene_rand.cfg.auto_flush_visuals
                    scene_rand.cfg.auto_flush_visuals = False
                    scene_rand()
                    scene_rand.cfg.auto_flush_visuals = original_auto_flush
                    log.info("  Applied SceneRandomizer")

            # Level 1+: Material randomization (environment only)
            if self.config.level >= 1:
                for mat_rand in self.randomizers["material_dynamic"]:
                    mat_rand()
                if self.randomizers["material_dynamic"]:
                    log.info(f"  Applied MaterialRandomizers ({len(self.randomizers['material_dynamic'])})")

            # Level 2+: Lighting
            if self.config.level >= 2:
                for light_rand in self.randomizers["light"]:
                    light_rand()
                if self.randomizers["light"]:
                    log.info(f"  Applied LightRandomizers ({len(self.randomizers['light'])})")

            # Level 3+: Camera
            if self.config.level >= 3:
                for cam_rand in self.randomizers["camera"]:
                    cam_rand()
                if self.randomizers["camera"]:
                    log.info(f"  Applied CameraRandomizers ({len(self.randomizers['camera'])})")

        finally:
            # Disable global defer and flush once
            if self.handler:
                self.handler._defer_all_visual_flushes = False
                if hasattr(self.handler, "flush_visual_updates"):
                    try:
                        self.handler.flush_visual_updates(wait_for_materials=True, settle_passes=2)
                    except Exception as e:
                        log.debug(f"Failed to flush visual updates: {e}")

    def update_camera_look_at(self, env_id: int = 0):
        """Update camera position and look_at to focus on table after scene switch.

        Adjusts both camera position and look-at point to maintain the same relative
        viewing angle but account for the table's height.
        """
        if not self.randomizers.get("scene"):
            return

        table_bounds = self.randomizers["scene"].get_table_bounds(env_id=env_id)
        if not table_bounds or abs(table_bounds.get("height", 0)) > 100:
            return

        table_height = table_bounds["height"]

        # Use original camera positions stored in __init__ (before any randomization)
        clearance = 0.05
        target_look_at_z = table_height + clearance

        for camera in self.handler.cameras:
            orig = self.original_camera_positions[camera.name]
            orig_look_at_z = orig["look_at"][2]
            orig_pos_z = orig["pos"][2]

            # Compute Z offset needed
            z_offset = target_look_at_z - orig_look_at_z

            # Apply same offset to both position and look_at
            camera.pos = (orig["pos"][0], orig["pos"][1], orig_pos_z + z_offset)
            camera.look_at = (orig["look_at"][0], orig["look_at"][1], target_look_at_z)

        if hasattr(self.handler, "_update_camera_pose"):
            self.handler._update_camera_pose()

    def update_positions_to_table(self, demo_idx: int, env_id: int = 0):
        """Update object positions to align with current table after scene switch.

        Maintains relative positions of all objects and robots (rigid body translation).
        The entire system is translated such that the original ground level aligns with the table surface.
        """
        if not self.randomizers.get("scene"):
            return

        # Get current state
        if demo_idx >= len(self.init_states):
            return

        init_state = self.init_states[demo_idx]

        # Get this demo's original positions (stored in __init__)
        demo_key = f"demo_{demo_idx}"
        if demo_key not in self.original_positions:
            log.warning(f"No original positions found for demo {demo_idx}")
            return

        demo_original_positions = self.original_positions[demo_key]

        # Get table bounds
        table_bounds = self.randomizers["scene"].get_table_bounds(env_id=env_id)
        if not table_bounds or abs(table_bounds.get("height", 0)) > 100:
            return

        table_height = table_bounds["height"]
        table_center_x = (table_bounds["x_min"] + table_bounds["x_max"]) / 2
        table_center_y = (table_bounds["y_min"] + table_bounds["y_max"]) / 2

        # Compute system center (XY)
        all_x = [demo_original_positions[k]["x"] for k in demo_original_positions]
        all_y = [demo_original_positions[k]["y"] for k in demo_original_positions]
        system_center_x = sum(all_x) / len(all_x)
        system_center_y = sum(all_y) / len(all_y)

        # Compute XY offset (to center system on table)
        offset_x = table_center_x - system_center_x
        offset_y = table_center_y - system_center_y

        # Compute Z offset: move the reference plane from ground to table
        # Find the ground level (minimum Z in original trajectory)
        all_z = [demo_original_positions[k]["z"] for k in demo_original_positions]
        ground_level = min(all_z)

        # The reference plane (ground) moves to table surface
        # All objects and robots maintain their relative height from this plane
        z_offset = table_height - ground_level

        # Apply same offset to everything (rigid body translation)
        for obj_name, obj_state in init_state["objects"].items():
            orig = demo_original_positions[f"obj_{obj_name}"]
            obj_state["pos"][0] = orig["x"] + offset_x
            obj_state["pos"][1] = orig["y"] + offset_y
            obj_state["pos"][2] = orig["z"] + z_offset

        for robot_name, robot_state in init_state["robots"].items():
            orig = demo_original_positions[f"robot_{robot_name}"]
            robot_state["pos"][0] = orig["x"] + offset_x
            robot_state["pos"][1] = orig["y"] + offset_y
            robot_state["pos"][2] = orig["z"] + z_offset

