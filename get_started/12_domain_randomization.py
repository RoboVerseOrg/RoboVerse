"""Domain Randomization Demo with 4 Scene Modes

Replays close_box task trajectories with progressive domain randomization.
Features 4 symmetric scene modes with increasing USD integration.

4 Scene Modes (--scene-mode):
┌──────┬─────────────┬─────────────────┬──────────────┬──────────────┐
│ Mode │ Name        │ Environment     │ Workspace    │ Objects      │
├──────┼─────────────┼─────────────────┼──────────────┼──────────────┤
│  0   │ Manual      │ Manual geometry │ Manual table │ None         │
│  1   │ USD Table   │ Manual geometry │ Table785 USD │ None         │
│  2   │ USD Scene   │ Kujiale USD     │ Table785 USD │ None         │
│  3   │ Full USD    │ Kujiale USD     │ Table785 USD │ Desktop USD  │
└──────┴─────────────┴─────────────────┴──────────────┴──────────────┘

Randomization Levels (--level):
- Level 0: Baseline (fixed scene, no randomization)
- Level 1: Scene/Material randomization
  * Mode 0-1: Material randomization
  * Mode 2-3: USD asset selection randomization
- Level 2: Level 1 + Lighting randomization
- Level 3: Level 2 + Camera randomization

Lighting (5 lights, fixed positions):
- 1 central DiskLight at (0, 0, 2.8) - main directional light
- 4 corner SphereLight at (±1, ±1, 2.5) - ambient coverage
- Conservative positioning for typical room sizes (4-6m)
- Intensity for render mode:
  * PathTracing: 18K + 8Kx4 = 50K total
  * RayTracing: 12K + 5Kx4 = 32K total

Examples:
- Manual geometry:    python 12_domain_randomization.py --scene-mode 0 --level 1
- USD table:          python 12_domain_randomization.py --scene-mode 1 --level 1
- USD scene + table:  python 12_domain_randomization.py --scene-mode 2 --level 2
- Full USD (3 layers): python 12_domain_randomization.py --scene-mode 3 --level 3
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
import random
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.randomization import (
    CameraPresets,
    CameraRandomizer,
    EnvironmentLayerCfg,
    LightRandomizer,
    MaterialPresets,
    MaterialRandomizer,
    ObjectPresets,
    ObjectRandomizer,
    ObjectsLayerCfg,
    SceneRandomizer,
    WorkspaceLayerCfg,
)
from metasim.randomization.presets.scene_presets import (
    SceneMaterialCollections,
    ScenePresets,
    SceneUSDCollections,
)
from metasim.randomization.scene_randomizer import SceneMaterialPoolCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.render import RenderCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.obs_utils import ObsSaver

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def get_actions(all_actions, action_idx: int, num_envs: int):
    """Get actions for all environments at a given step."""
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_runout(all_actions, action_idx: int):
    """Check if all trajectories have run out of actions."""
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout


def create_env(args):
    """Create task environment."""
    task_name = "close_box"
    task_cls = get_task_class(task_name)

    # Initial table height estimate (will be updated dynamically after randomization)
    initial_table_height = 0.7

    camera = PinholeCameraCfg(
        name="main_camera",
        width=1024,
        height=1024,
        pos=(1.2, -1.2, 1.5),  # Closer: ~2.0m → better for Kujiale scenes
        look_at=(0.0, 0.0, initial_table_height + 0.05),  # Will be updated with actual table height
        focal_length=18.0,  # Wider FOV: ~54° (was 24.0 → 41.7°)
    )

    # Lighting configuration
    # Fixed positions designed for small rooms (Kujiale 4-6m) while also working for large rooms
    # All lights positioned to be inside typical room bounds
    if args.render_mode == "pathtracing":
        ceiling_main = 18000.0  # Reduced from 28K
        ceiling_corners = 8000.0  # Reduced from 12K
    else:
        ceiling_main = 12000.0  # Reduced from 20K
        ceiling_corners = 5000.0  # Reduced from 7K

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),  # Conservative height for typical ceiling (2.5-3m)
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(1.0, 1.0, 2.5),  # Conservative offset (1m) fits in 4m+ rooms
        ),
        SphereLightCfg(
            name="ceiling_nw",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-1.0, 1.0, 2.5),
        ),
        SphereLightCfg(
            name="ceiling_sw",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-1.0, -1.0, 2.5),
        ),
        SphereLightCfg(
            name="ceiling_se",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(1.0, -1.0, 2.5),
        ),
    ]

    render_cfg = RenderCfg(mode=args.render_mode)

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        lights=lights,
        simulator=args.sim,
        renderer=args.renderer,
        render=render_cfg,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    return env


def get_init_states(level, num_envs):
    """Get initial states for objects and robot based on level."""
    box_base_height = 0.15
    table_surface_z = 0.7

    objects = {
        "box_base": {
            "pos": torch.tensor([-0.2, 0.0, table_surface_z + box_base_height / 2]),
            "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
            "dof_pos": {"box_joint": 0.0},
        },
    }

    robot = {
        "franka": {
            "pos": torch.tensor([0.0, -0.4, table_surface_z]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785398,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356194,
                "panda_joint5": 0.0,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        },
    }

    return [{"objects": objects, "robots": robot}] * num_envs


def initialize_randomizers(handler, args):
    """Initialize all randomizers based on scene mode and randomization level."""
    randomizers = {
        "object": [],
        "material": [],
        "light": [],
        "camera": [],
        "scene": None,
    }

    mode = args.scene_mode
    level = args.level

    MODE_NAMES = {0: "Manual", 1: "USD Table", 2: "USD Scene", 3: "Full USD"}

    log.info("=" * 70)
    log.info(f"Scene Mode: {mode} ({MODE_NAMES[mode]})")
    log.info(f"Randomization Level: {level}")
    log.info("=" * 70)

    log.info("\n[Scene Configuration]")
    log.info("-" * 70)

    # Build scene configuration based on scene mode
    from metasim.randomization.scene_randomizer import (
        ManualGeometryCfg,
        SceneRandomCfg,
        USDAssetCfg,
        USDAssetPoolCfg,
    )

    scene_cfg = SceneRandomCfg()

    # ========================================================================
    # Mode 0-1: Manual Environment (floor, walls, ceiling)
    # Mode 2-3: USD Environment (Kujiale scenes)
    # ========================================================================
    if mode >= 2:
        # USD Environment
        # Get paths and configs together (convenient!)
        scene_paths, scene_configs = SceneUSDCollections.kujiale_scenes(auto_download=True, return_configs=True)
        log.info(f"  Environment: Kujiale USD ({len(scene_paths)} scenes)")
        if scene_paths:
            if level >= 1:
                # Randomize: use pool with per-path configs
                env_element = USDAssetPoolCfg(
                    name="kujiale_scene",
                    usd_paths=scene_paths,
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    disable_physics=True,
                    per_path_overrides=scene_configs,  # Apply per-scene calibrations
                    selection_strategy="random",
                )
                log.info("    Selection: Random (with per-scene configs)")
            else:
                # Fixed: use single USD
                env_element = USDAssetCfg(
                    name="kujiale_scene",
                    usd_path=scene_paths[0],
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    disable_physics=True,
                )
                log.info("    Selection: Fixed")

            scene_cfg.environment_layer = EnvironmentLayerCfg(elements=[env_element])
    else:
        # Manual Environment
        log.info("  Environment: Manual geometry (10m x 10m x 5m)")

        if level < 1:
            # Level 0: Fixed materials (baseline, deterministic)
            base_cfg = ScenePresets.empty_room(
                room_size=10.0,
                wall_height=5.0,
                floor_families=("carpet",),  # Single family for determinism
                wall_families=("masonry",),
                ceiling_families=("architecture",),
            )
            scene_cfg.environment_layer = base_cfg.environment_layer

            # Override with single fixed material for complete determinism
            for element in scene_cfg.environment_layer.elements:
                if element.name == "floor":
                    element.material_pool = SceneMaterialPoolCfg(
                        material_paths=["roboverse_data/materials/arnold/Carpet/Carpet_Beige.mdl"],
                        selection_strategy="sequential",
                    )
                elif element.name.startswith("wall_"):
                    element.material_pool = SceneMaterialPoolCfg(
                        material_paths=["roboverse_data/materials/arnold/Masonry/Stucco.mdl"],
                        selection_strategy="sequential",
                    )
                elif element.name == "ceiling":
                    element.material_pool = SceneMaterialPoolCfg(
                        material_paths=["roboverse_data/materials/arnold/Architecture/Ceiling_Tiles.mdl"],
                        selection_strategy="sequential",
                    )
            log.info("    Materials: Fixed (baseline)")
        else:
            # Level 1+: Randomized materials
            base_cfg = ScenePresets.empty_room(
                room_size=10.0,
                wall_height=5.0,
                floor_families=("carpet", "wood", "stone", "concrete", "architecture"),
                wall_families=("architecture", "wall_board", "masonry", "paint", "composite"),
                ceiling_families=("architecture", "wall_board", "wood"),
            )
            scene_cfg.environment_layer = base_cfg.environment_layer
            log.info("    Materials: Randomized")

    # ========================================================================
    # Mode 0: Manual Workspace (table)
    # Mode 1-3: USD Workspace (Table785)
    # ========================================================================
    if mode >= 1:
        # USD Workspace
        # Get paths and configs together (convenient!)
        table_paths, table_configs = SceneUSDCollections.table785(auto_download=True, return_configs=True)
        log.info(f"  Workspace: Table785 USD ({len(table_paths)} tables)")
        if table_paths:
            if level >= 1:
                # Randomize: use pool with per-path configs
                workspace_element = USDAssetPoolCfg(
                    name="table",
                    usd_paths=table_paths,
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    disable_physics=True,  # Pure visual - no physics collision
                    per_path_overrides=table_configs,  # Apply per-table calibrations
                    selection_strategy="random",
                )
                log.info("    Selection: Random (with per-table configs)")
            else:
                # Fixed: use single USD
                workspace_element = USDAssetCfg(
                    name="table",
                    usd_path=table_paths[0],
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    disable_physics=True,  # Pure visual - no physics collision
                )
                log.info("    Selection: Fixed")

            scene_cfg.workspace_layer = WorkspaceLayerCfg(elements=[workspace_element])
    else:
        # Manual Workspace
        log.info("  Workspace: Manual table (1.8m x 1.8m at z=0.7m)")
        scene_cfg.workspace_layer = WorkspaceLayerCfg(
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=(1.8, 1.8, 0.1),
                    position=(0.0, 0.0, 0.7 - 0.05),
                    material_randomization=True,
                    material_pool=SceneMaterialPoolCfg(
                        material_paths=["roboverse_data/materials/arnold/Wood/Plywood.mdl"]
                        if level < 1
                        else SceneMaterialCollections.table_materials(
                            families=("wood", "stone", "plastic", "ceramic", "metal")
                        ),
                        selection_strategy="sequential" if level < 1 else "random",
                    ),
                ),
            ],
        )
        log.info(f"    Materials: {'Fixed' if level < 1 else 'Randomized'}")

    # ========================================================================
    # Mode 0-2: No Objects
    # Mode 3: USD Objects (Desktop supplies - 3 objects on table)
    # ========================================================================
    if mode >= 3:
        # USD Objects
        # Get paths and configs together (convenient!)
        object_paths, object_configs = SceneUSDCollections.desktop_supplies(auto_download=True, return_configs=True)
        log.info(f"  Objects: Desktop supplies USD ({len(object_paths)} objects, placing 3 on table)")
        if object_paths:
            # Create 3 separate object elements for variety
            object_elements = []

            for i in range(3):
                if level >= 1:
                    # Randomize: use pool with per-path configs
                    object_element = USDAssetPoolCfg(
                        name=f"desktop_object_{i + 1}",  # Unique name for each object
                        usd_paths=object_paths,
                        fix_base_link=True,  # Static (not used when disable_physics=True)
                        disable_physics=True,  # Pure visual - no physics (like scene/table)
                        per_path_overrides=object_configs,  # Apply per-object calibrations
                        selection_strategy="random",
                    )
                else:
                    # Fixed: use sequential USD for variety
                    object_element = USDAssetCfg(
                        name=f"desktop_object_{i + 1}",
                        usd_path=object_paths[i % len(object_paths)],
                        fix_base_link=True,  # Static (not used when disable_physics=True)
                        disable_physics=True,  # Pure visual - no physics (like scene/table)
                    )
                object_elements.append(object_element)

            scene_cfg.objects_layer = ObjectsLayerCfg(elements=object_elements)
            log.info(f"    Selection: Random (3 objects from {len(object_paths)} candidates)")
    else:
        # No Objects
        log.info("  Objects: None")

    scene_rand = SceneRandomizer(scene_cfg, seed=args.seed)
    scene_rand.bind_handler(handler)
    randomizers["scene"] = scene_rand

    # Object randomization for box_base (physics only, no pose changes)
    box_rand = ObjectRandomizer(
        ObjectPresets.heavy_object("box_base"),
        seed=args.seed,
    )
    # Disable all pose randomization for deterministic behavior
    box_rand.cfg.pose.rotation_range = (0, 0)
    box_rand.cfg.pose.position_range = [(0, 0), (0, 0), (0, 0)]  # No position jitter
    box_rand.bind_handler(handler)
    box_rand()

    log.info("\n[Level 1+] Material Randomization")
    log.info("-" * 70)

    box_mat = MaterialRandomizer(
        MaterialPresets.mdl_family_object("box_base", family=("paper", "wood")),
        seed=args.seed,
    )
    box_mat.bind_handler(handler)
    randomizers["material"].append(box_mat)
    log.info("  box_base: wood material")

    log.info("\n[Level 2+] Light Randomization")
    log.info("-" * 70)

    from metasim.randomization import (
        LightColorRandomCfg,
        LightIntensityRandomCfg,
        LightOrientationRandomCfg,
        LightPositionRandomCfg,
        LightRandomCfg,
    )

    if args.render_mode == "pathtracing":
        ceiling_main_range = (22000.0, 40000.0)
        ceiling_corner_range = (10000.0, 18000.0)
    else:
        ceiling_main_range = (16000.0, 30000.0)
        ceiling_corner_range = (6000.0, 12000.0)

    main_light_cfg = LightRandomCfg(
        light_name="ceiling_main",
        intensity=LightIntensityRandomCfg(
            intensity_range=ceiling_main_range,
            distribution="uniform",
            enabled=True,
        ),
        color=LightColorRandomCfg(
            temperature_range=(3000.0, 6000.0),
            use_temperature=True,
            distribution="uniform",
            enabled=True,
        ),
        position=LightPositionRandomCfg(
            position_range=((-1.0, 1.0), (-1.0, 1.0), (-0.2, 0.2)),
            relative_to_origin=True,
            distribution="uniform",
            enabled=True,
        ),
        orientation=LightOrientationRandomCfg(
            angle_range=((-20.0, 20.0), (-20.0, 20.0), (-180.0, 180.0)),
            relative_to_origin=True,
            distribution="uniform",
            enabled=True,
        ),
        randomization_mode="combined",
    )
    main_light_rand = LightRandomizer(main_light_cfg, seed=args.seed)
    main_light_rand.bind_handler(handler)
    randomizers["light"].append(main_light_rand)

    for light_name in ["ceiling_ne", "ceiling_nw", "ceiling_sw", "ceiling_se"]:
        light_cfg = LightRandomCfg(
            light_name=light_name,
            intensity=LightIntensityRandomCfg(
                intensity_range=ceiling_corner_range,
                distribution="uniform",
                enabled=True,
            ),
            color=LightColorRandomCfg(
                temperature_range=(2700.0, 5500.0),
                use_temperature=True,
                distribution="uniform",
                enabled=True,
            ),
            position=LightPositionRandomCfg(
                position_range=((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2)),
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
            ),
            randomization_mode="combined",
        )
        light_rand = LightRandomizer(light_cfg, seed=args.seed)
        light_rand.bind_handler(handler)
        randomizers["light"].append(light_rand)

    log.info(f"  Configured {len(randomizers['light'])} light randomizers")
    log.info(f"    DiskLight (main): {ceiling_main_range[0] / 1000:.0f}K-{ceiling_main_range[1] / 1000:.0f}K")
    log.info(f"    SphereLight (corners): {ceiling_corner_range[0] / 1000:.0f}K-{ceiling_corner_range[1] / 1000:.0f}K")
    log.info("    Randomization: intensity, color, position, orientation (DiskLight only)")

    log.info("\n[Level 3+] Camera Randomization")
    log.info("-" * 70)

    camera_rand = CameraRandomizer(
        CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
        seed=args.seed,
    )
    camera_rand.bind_handler(handler)
    randomizers["camera"].append(camera_rand)
    log.info("  Camera: surveillance preset")

    log.info("\n" + "=" * 70)
    return randomizers


def update_positions_based_on_table(env, scene_randomizer, init_state, table_bounds=None, _original_z_cache=None):
    """Update object and robot positions based on current table bounds.

    Preserves relative positions between robot and objects, only translates them
    to center on the table and adjusts Z to table surface.

    Args:
        env: Environment instance
        scene_randomizer: Scene randomizer instance
        init_state: Initial state dictionary to update
        table_bounds: Pre-computed table bounds (optional, will compute if None)
        _original_z_cache: Internal cache for original Z values (relative to ground)
    """
    if scene_randomizer is None:
        return

    # Initialize cache if not provided
    if _original_z_cache is None:
        _original_z_cache = {}

    # Use provided bounds or compute them
    if table_bounds is None:
        table_bounds = scene_randomizer.get_table_bounds(env_id=0)

    if not table_bounds:
        return

    table_height = table_bounds["height"]
    table_center_x = (table_bounds["x_min"] + table_bounds["x_max"]) / 2
    table_center_y = (table_bounds["y_min"] + table_bounds["y_max"]) / 2

    log.debug(
        f"Updating positions - Table height: {table_height:.3f}, Center: ({table_center_x:.3f}, {table_center_y:.3f})"
    )

    # First call: save original ground-relative Z values
    if not _original_z_cache:
        for obj_name, obj_state in init_state["objects"].items():
            z_val = obj_state["pos"][2].item() if hasattr(obj_state["pos"][2], "item") else float(obj_state["pos"][2])
            _original_z_cache[f"obj_{obj_name}"] = z_val
        for robot_name, robot_state in init_state["robots"].items():
            z_val = (
                robot_state["pos"][2].item() if hasattr(robot_state["pos"][2], "item") else float(robot_state["pos"][2])
            )
            _original_z_cache[f"robot_{robot_name}"] = z_val
        log.debug(f"Saved original Z values: {_original_z_cache}")

    # Compute center of all objects and robots (their original relative positions)
    all_entities = []
    for obj_name, obj_state in init_state["objects"].items():
        all_entities.append(obj_state["pos"][:2])  # Only X, Y
    for robot_name, robot_state in init_state["robots"].items():
        all_entities.append(robot_state["pos"][:2])  # Only X, Y

    if not all_entities:
        return

    # Calculate the center of the robot-object system
    import torch

    entities_tensor = torch.stack(all_entities)
    system_center_x = entities_tensor[:, 0].mean().item()
    system_center_y = entities_tensor[:, 1].mean().item()

    # Compute offset to align system center with table center
    offset_x = table_center_x - system_center_x
    offset_y = table_center_y - system_center_y

    log.debug(
        f"System center: ({system_center_x:.3f}, {system_center_y:.3f}), Offset: ({offset_x:.3f}, {offset_y:.3f})"
    )

    # Apply offset to all objects (preserving relative positions)
    for obj_name, obj_state in init_state["objects"].items():
        # Translate X, Y by offset (preserve relative position)
        obj_state["pos"][0] += offset_x
        obj_state["pos"][1] += offset_y

        # Adjust Z to table surface using cached ground-relative height
        original_z = _original_z_cache[f"obj_{obj_name}"]
        obj_state["pos"][2] = table_height + original_z

    # Apply offset to all robots (preserving relative positions)
    for robot_name, robot_state in init_state["robots"].items():
        # Translate X, Y by offset (preserve relative position)
        robot_state["pos"][0] += offset_x
        robot_state["pos"][1] += offset_y

        # Adjust Z to table surface using cached ground-relative height
        original_z = _original_z_cache[f"robot_{robot_name}"]
        robot_state["pos"][2] = table_height + original_z

    # Apply updated states to all environments
    num_envs = env.scenario.num_envs
    env.handler.set_states([init_state] * num_envs, env_ids=list(range(num_envs)))


def apply_randomization(randomizers, level, handler) -> None:
    """Apply all randomizers simultaneously with deferred visual flush.

    Ensures all randomizations (scene, object, material, light, camera) are
    applied atomically before flushing visuals, preventing intermediate states
    from being captured in video recordings.

    Args:
        randomizers: Dictionary of randomizers
        level: Randomization level
        handler: Environment handler
    """
    # Temporarily disable auto-flush in scene randomizer
    scene_rand = randomizers["scene"]
    if scene_rand:
        original_auto_flush = scene_rand.cfg.auto_flush_visuals
        scene_rand.cfg.auto_flush_visuals = False

        scene_rand()

        scene_rand.cfg.auto_flush_visuals = original_auto_flush

    # Apply object randomization (only level 1+)
    # Level 0 should be completely deterministic
    if level >= 1:
        for rand in randomizers["object"]:
            rand()

    # Apply material randomization with deferred flush
    if level >= 1:
        for rand in randomizers["material"]:
            if hasattr(rand, "_defer_visual_flush"):
                rand._defer_visual_flush = True
            rand()
            if hasattr(rand, "_defer_visual_flush"):
                rand._defer_visual_flush = False

    # Apply light randomization
    if level >= 2:
        for rand in randomizers["light"]:
            rand()

    # Apply camera randomization
    if level >= 3:
        for rand in randomizers["camera"]:
            rand()

    # Single comprehensive flush after all randomizations complete
    flush_fn = getattr(handler, "flush_visual_updates", None)
    if callable(flush_fn):
        try:
            flush_fn(wait_for_materials=True, settle_passes=3)
        except Exception as e:
            log.debug(f"Failed to flush visual updates: {e}")


def get_states(all_states, action_idx: int, num_envs: int):
    """Get states for all environments at a given step."""
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def run_replay_with_randomization(env, randomizers, init_state, all_actions, all_states, args):
    """Replay trajectory with periodic randomization."""
    os.makedirs("get_started/output", exist_ok=True)

    # Generate video filename based on scene mode
    if args.object_states:
        mode_tag = "states"
    else:
        mode_names = {0: "manual", 1: "usd_table", 2: "usd_scene", 3: "full_usd"}
        mode_tag = f"mode{args.scene_mode}_{mode_names[args.scene_mode]}_level{args.level}"

    video_path = f"get_started/output/12_dr_{mode_tag}_{args.sim}.mp4"

    obs_saver = ObsSaver(video_path=video_path)

    log.info("\n" + "=" * 70)
    log.info("Trajectory Replay with Domain Randomization")
    log.info("=" * 70)
    log.info(f"Video output: {video_path}")
    log.info(f"Randomization interval: every {args.randomize_interval} steps")

    traj_length = len(all_actions[0]) if all_actions else (len(all_states[0]) if all_states else 0)
    log.info(f"Trajectory length: {traj_length} steps")

    randomization_enabled = not args.object_states

    # Note: Initial randomization already applied in main() before calling this function
    obs, extras = env.reset(states=[init_state] * args.num_envs)

    step = 0
    num_envs = env.scenario.num_envs

    while True:
        if randomization_enabled and step % args.randomize_interval == 0 and step > 0:
            log.info(f"Step {step}: Applying randomizations (including objects)")
            # Apply randomization (scene, table, objects, materials, lights, camera)
            # Objects will naturally fall and interact with physics in real-time
            apply_randomization(randomizers, args.level, env.handler)

            # Note: We do NOT update positions here during trajectory replay!
            # The trajectory is relative to the initial position, so moving
            # robot/objects mid-execution would break the trajectory.
            # Position adjustment only happens once at initialization.

        if args.object_states:
            if all_states is None:
                raise ValueError("State playback requested but no states were loaded from trajectory")

            states = get_states(all_states, step, num_envs)
            env.handler.set_states(states, env_ids=list(range(num_envs)))
            env.handler.refresh_render()
            obs = env.handler.get_states()

            if hasattr(env, "checker"):
                success = env.checker.check(env.handler, obs)
            else:
                success = torch.zeros(num_envs, dtype=torch.bool)

            time_out = torch.zeros_like(success)
        else:
            actions = get_actions(all_actions, step, num_envs)
            obs, reward, success, time_out, extras = env.step(actions)

        if success.any():
            log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded")

        if time_out.any():
            log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out")

        if success.all() or time_out.all():
            log.info("All environments terminated")
            break

        obs_saver.add(obs)

        if args.object_states:
            if get_runout(all_states, step + 1):
                log.info("Trajectory ended")
                break
        else:
            if get_runout(all_actions, step + 1):
                log.info("Trajectory ended")
                break

        step += 1

    obs_saver.save()
    log.info(f"\nVideo saved: {video_path}")


def main():
    @configclass
    class Args:
        sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaacsim"
        renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None
        robot: str = "franka"
        scene: str | None = None
        num_envs: int = 1
        headless: bool = False
        seed: int | None = 42

        scene_mode: Literal[0, 1, 2, 3] = 0
        """Scene mode:
        0 - Manual (all manual geometry, no objects)
        1 - USD Table (USD table, manual environment, no objects)
        2 - USD Scene (USD environment + table, no objects)
        3 - Full USD (USD environment + table + objects)
        """

        level: Literal[0, 1, 2, 3] = 1
        """Randomization level:
        0 - Baseline (fixed scene, no randomization)
        1 - Scene/Material randomization
        2 - Level 1 + Lighting randomization
        3 - Level 2 + Camera randomization
        """

        randomize_interval: int = 60
        """Randomization interval in steps."""

        object_states: bool = False
        """If True, replay using object states (deterministic)."""

        render_mode: Literal["raytracing", "pathtracing"] = "raytracing"
        """Rendering mode:
        - raytracing: Fast with shadows
        - pathtracing: Highest quality (slower)
        """

    args = tyro.cli(Args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    MODE_NAMES = {0: "Manual", 1: "USD Table", 2: "USD Scene", 3: "Full USD"}
    MODE_DESCRIPTIONS = {
        0: "Manual environment + Manual table + No objects",
        1: "Manual environment + USD table + No objects",
        2: "USD environment + USD table + No objects",
        3: "USD environment + USD table + USD objects",
    }

    log.info("=" * 70)
    log.info("Domain Randomization Demo with 4 Scene Modes")
    log.info("=" * 70)
    log.info("\nConfiguration:")
    log.info(f"  Simulator: {args.sim}")
    log.info(f"  Render mode: {args.render_mode}")
    log.info(f"  Robot: {args.robot}")
    log.info(f"  Seed: {args.seed}")

    log.info(f"\nScene Mode: {args.scene_mode} ({MODE_NAMES[args.scene_mode]})")
    log.info(f"  {MODE_DESCRIPTIONS[args.scene_mode]}")
    log.info(f"\nRandomization Level: {args.level}")

    log.info(f"\nLighting ({args.render_mode}):")
    if args.render_mode == "pathtracing":
        log.info("  DiskLight (main): 18K, 4x SphereLight (corners): 8K each")
        log.info("  Total: ~50K")
    else:
        log.info("  DiskLight (main): 12K, 4x SphereLight (corners): 5K each")
        log.info("  Total: ~32K")
    log.info("  Positions: Fixed at (0,0,2.8) and (±1,±1,2.5)")

    env = create_env(args)
    handler = env.handler

    traj_filepath = env.traj_filepath
    log.info(f"\nLoading trajectory: {traj_filepath}")
    assert os.path.exists(traj_filepath), f"Trajectory file not found: {traj_filepath}"

    init_states, all_actions, all_states = get_traj(traj_filepath, env.scenario.robots[0], handler)
    init_state = init_states[0]

    log.info(f"Loaded {len(all_actions[0]) if all_actions else 0} actions")

    # Initialize randomizers first (this creates the scene with table)
    randomizers = initialize_randomizers(handler, args)

    # Apply initial randomization to create the scene
    if not args.object_states:
        apply_randomization(randomizers, args.level, env.handler)

    # Dynamically compute table bounds and adjust initial positions
    scene_randomizer = randomizers["scene"]
    table_bounds = None  # Will be computed once and reused

    if scene_randomizer:
        table_bounds = scene_randomizer.get_table_bounds(env_id=0)
        if table_bounds:
            log.info("\nInitial Dynamic Table Bounds:")
            log.info(f"  Height: {table_bounds['height']:.3f}")
            log.info(f"  X range: [{table_bounds['x_min']:.3f}, {table_bounds['x_max']:.3f}]")
            log.info(f"  Y range: [{table_bounds['y_min']:.3f}, {table_bounds['y_max']:.3f}]")

            # Update camera look_at to focus on actual table height
            actual_table_height = table_bounds["height"]
            for camera in env.handler.cameras:
                if camera.name == "main_camera":
                    camera.look_at = (0.0, 0.0, actual_table_height + 0.05)
                    log.info(f"Updated camera look_at to table height: {actual_table_height:.3f}")

            # Apply camera pose update
            if hasattr(env.handler, "_update_camera_pose"):
                env.handler._update_camera_pose()
                log.debug("Camera pose updated with new look_at position")
        else:
            log.warning("Could not compute table bounds, using default Z offset +0.7")
            for obj_name, obj_state in init_state["objects"].items():
                obj_state["pos"][2] += 0.7
            for robot_name, robot_state in init_state["robots"].items():
                robot_state["pos"][2] += 0.7
    else:
        log.warning("No scene randomizer found, using default Z offset +0.7")
        for obj_name, obj_state in init_state["objects"].items():
            obj_state["pos"][2] += 0.7
        for robot_name, robot_state in init_state["robots"].items():
            robot_state["pos"][2] += 0.7

    # Apply the unified position update logic (reuse computed table_bounds)
    update_positions_based_on_table(env, scene_randomizer, init_state, table_bounds=table_bounds)

    if args.object_states:
        log.info("\nWARNING: State-based replay mode (no randomization applied)")

    run_replay_with_randomization(env, randomizers, init_state, all_actions, all_states, args)

    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()

    log.info("\nDemo completed")


if __name__ == "__main__":
    main()
