"""Domain Randomization Demo - Refactored Architecture

This demo showcases the refactored Domain Randomization architecture with clean
separation of concerns and unified object access.

Architecture Highlights:
1. Two Object Types:
   - Static Objects: Handler-managed (Robot, box_base, Camera, Light)
   - Dynamic Objects: SceneRandomizer-managed (Floor, Table, Distractors)

2. Two Randomizer Types:
   - Lifecycle Manager: SceneRandomizer (create/delete/switch)
   - Property Editors: Material/Object/Light/Camera (edit properties)

3. Unified Access:
   - ObjectRegistry: Automatic, transparent access to all objects
   - MaterialRandomizer can randomize Dynamic Objects (table, floor)

4. Hybrid Support:
   - Automatic handler dispatch based on REQUIRES_HANDLER
   - Zero configuration needed

Performance Optimization:
- Global defer mechanism: 22 flushes → 1 flush (~15-30x speedup)
- Unified settle_passes=2 for quality-performance balance

Scene Modes:
- Mode 0: Manual (all manual geometry)
- Mode 1: USD Table (USD table + manual environment)
- Mode 2: USD Scene (Kujiale + Table785)
- Mode 3: Full USD (Kujiale + Table785 + Desktop objects)

Randomization Levels:
- Level 0: Baseline (no randomization)
- Level 1: Scene/Material randomization
- Level 2: Level 1 + Lighting randomization (intensity/color/position/orientation)
- Level 3: Level 2 + Camera randomization

Run:
    python get_started/12_domain_randomization.py --scene_mode 0 --level 2
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
    CameraIntrinsicsRandomCfg,
    CameraPositionRandomCfg,
    # Randomizers
    CameraRandomCfg,
    CameraRandomizer,
    # Scene Configuration
    EnvironmentLayerCfg,
    LightRandomizer,
    ManualGeometryCfg,
    MaterialPresets,
    MaterialRandomizer,
    ObjectPresets,
    ObjectRandomizer,
    # Core (NEW - usually transparent)
    ObjectRegistry,
    ObjectsLayerCfg,
    SceneRandomCfg,
    SceneRandomizer,
    USDAssetPoolCfg,
    WorkspaceLayerCfg,
)
from metasim.randomization.presets.scene_presets import (
    ScenePresets,
    SceneUSDCollections,
)
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.render import RenderCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.obs_utils import ObsSaver

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def create_env(args):
    """Create task environment with lights and camera."""
    task_name = "close_box"
    task_cls = get_task_class(task_name)

    camera = PinholeCameraCfg(
        name="main_camera",
        width=1024,
        height=1024,
        pos=(1.2, -1.2, 1.5),
        look_at=(0.0, 0.0, 0.75),
        focal_length=18.0,
    )

    # Lighting setup
    if args.render_mode == "pathtracing":
        ceiling_main = 18000.0
        ceiling_corners = 8000.0
    else:
        ceiling_main = 12000.0
        ceiling_corners = 5000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),
            rot=(0.7071, 0.0, 0.0, 0.7071),
            shared=False,  # Per-environment lights
        ),
        SphereLightCfg(
            name="ceiling_ne",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(1.0, 1.0, 2.5),
            shared=False,  # Per-environment lights
        ),
        SphereLightCfg(
            name="ceiling_nw",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-1.0, 1.0, 2.5),
            shared=False,  # Per-environment lights
        ),
        SphereLightCfg(
            name="ceiling_sw",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-1.0, -1.0, 2.5),
            shared=False,  # Per-environment lights
        ),
        SphereLightCfg(
            name="ceiling_se",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(1.0, -1.0, 2.5),
            shared=False,  # Per-environment lights
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
        env_spacing=args.env_spacing,
        headless=args.headless,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    return env


def initialize_randomizers(handler, args):
    """Initialize all randomizers showcasing the new architecture."""
    mode = args.scene_mode
    level = args.level

    log.info("=" * 70)
    log.info("NEW ARCHITECTURE: Static vs Dynamic Objects")
    log.info("=" * 70)
    log.info("Static Objects (Handler-managed):")
    log.info("  - Robot (franka)")
    log.info("  - Task Object (box_base)")
    log.info("  - Camera (main_camera)")
    log.info("  - Lights (5 lights)")
    log.info("")
    log.info("Dynamic Objects (SceneRandomizer-managed):")
    log.info("  - Environment (Floor/Walls/Ceiling or Kujiale scene)")
    log.info("  - Workspace (Table)")
    log.info("  - Objects (Desktop items)")
    log.info("=" * 70)

    randomizers = {
        "scene": None,
        "scene_env_ids": None,  # Store env_ids for SceneRandomizer
        "object_physics": [],
        "material_static": [],  # Materials for Static Objects
        "material_dynamic": [],  # Materials for Dynamic Objects
        "light": [],
        "camera": [],
    }

    # =========================================================================
    # STEP 1: Create Scene (SceneRandomizer - Lifecycle Manager)
    # =========================================================================

    log.info("\n[STEP 1] Scene Creation (SceneRandomizer)")
    log.info("-" * 70)

    if mode >= 2:
        # USD Scene
        scene_paths, scene_configs = SceneUSDCollections.kujiale_scenes(return_configs=True)
        log.info(f"Environment: Kujiale USD ({len(scene_paths)} scenes, per-env)")

        env_element = USDAssetPoolCfg(
            name="kujiale_scene",
            usd_paths=scene_paths,
            per_path_overrides=scene_configs,
            selection_strategy="random" if level >= 1 else "sequential",
        )
        environment_layer = EnvironmentLayerCfg(
            elements=[env_element],
            shared=False,  # Per-env
            per_env=False,  # All envs get the same USD scene
            env_ids=None,  # Apply to all envs
        )
    else:
        # Manual Scene - Complete room
        log.info("Environment: Manual geometry (10m x 10m x 5m, per-env, all envs)")
        base_cfg = ScenePresets.empty_room(
            room_size=10.0,
            wall_height=5.0,
        )
        environment_layer = base_cfg.environment_layer
        environment_layer.shared = False  # Per-env
        environment_layer.per_env = True  # Each env gets different material (via MaterialRandomizer)
        environment_layer.env_ids = None  # Apply to all envs

    if mode >= 1:
        # USD Table
        table_paths, table_configs = SceneUSDCollections.table785(return_configs=True)

        # Calculate first half of environments dynamically
        num_envs = handler.num_envs
        first_half_envs = list(range(num_envs // 2))

        log.info(f"Workspace: Table785 USD ({len(table_paths)} tables)")
        log.info(f"  - env_ids: {first_half_envs} (first half of {num_envs} environments)")
        log.info("  - per_env: False (all envs get the same table)")

        workspace_element = USDAssetPoolCfg(
            name="table",
            usd_paths=table_paths,
            per_path_overrides=table_configs,
            selection_strategy="random" if level >= 1 else "sequential",
        )
        workspace_layer = WorkspaceLayerCfg(
            elements=[workspace_element],
            shared=False,  # Per-env
            per_env=False,  # All envs get the same table
            env_ids=first_half_envs,  # Only apply to first half of envs
        )
    else:
        # Manual Table (with default Plywood, MaterialRandomizer will randomize in level 1+)
        log.info("Workspace: Manual table (Plywood default, randomized in level 1+, per-env, all envs)")
        workspace_layer = WorkspaceLayerCfg(
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=(1.8, 1.8, 0.1),
                    position=(0.0, 0.0, 0.7 - 0.05),  # 0.65m (table surface at 0.7m)
                    default_material="roboverse_data/materials/arnold/Wood/Plywood.mdl",
                )
            ],
            shared=False,  # Per-env
            per_env=False,  # Manual geometry, material randomization via MaterialRandomizer
            env_ids=None,  # Apply to all envs
        )

    if mode >= 3:
        # Desktop Objects
        object_paths, object_configs = SceneUSDCollections.desktop_supplies(return_configs=True)
        log.info(f"Objects: Desktop supplies ({len(object_paths)} items, placing 3, per-env, all envs)")

        objects_layer = ObjectsLayerCfg(
            elements=[
                USDAssetPoolCfg(
                    name=f"desktop_object_{i + 1}",
                    usd_paths=object_paths,
                    per_path_overrides=object_configs,
                    selection_strategy="random" if level >= 1 else "sequential",
                )
                for i in range(3)
            ],
            shared=False,  # Per-env
            per_env=False,  # All envs get the same objects
            env_ids=None,  # Apply to all envs
        )
    else:
        objects_layer = None

    # Create SceneRandomizer
    scene_cfg = SceneRandomCfg(
        environment_layer=environment_layer,
        workspace_layer=workspace_layer,
        objects_layer=objects_layer,
        env_ids=None,  # Default env_ids for all layers (can be overridden by layer.env_ids)
    )

    scene_rand = SceneRandomizer(scene_cfg, seed=args.seed)
    scene_rand.bind_handler(handler)
    randomizers["scene"] = scene_rand

    # No longer need scene_env_ids, as each layer has its own env_ids
    log.info("SceneRandomizer created (layers have independent env_ids)")

    # =========================================================================
    # STEP 2: Material Randomization (NEW: Works for ALL objects)
    # =========================================================================

    log.info("\n[STEP 2] Material Randomization (MaterialRandomizer)")
    log.info("-" * 70)

    # Static Object material
    box_mat_cfg = MaterialPresets.mdl_family_object("box_base", family=("wood", "plastic"))
    box_mat_cfg.mdl.per_env = True  # Each environment gets different material
    box_mat = MaterialRandomizer(box_mat_cfg, seed=args.seed + 1)
    box_mat.bind_handler(handler)
    randomizers["material_static"].append(box_mat)
    log.info("Static Object: box_base (wood/plastic/metal/ceramic materials, per_env=True)")

    # Dynamic Object materials (NEW FEATURE!)
    # Note: Only for Mode 0 (manual table)
    # Mode 1+ use USD tables with their own materials
    if mode == 0:
        table_mat_cfg = MaterialPresets.mdl_family_object("table", family=("wood", "metal"))
        table_mat_cfg.mdl.per_env = True  # Each environment gets different material
        table_mat = MaterialRandomizer(table_mat_cfg, seed=args.seed + 2)
        table_mat.bind_handler(handler)
        randomizers["material_dynamic"].append(table_mat)
        log.info("Dynamic Object: table (Manual, wood/metal materials, per_env=True)")

    # Manual geometry materials (floor, walls, ceiling)
    # Only for modes with manual environment (mode < 2) and level >= 1
    if mode < 2 and level >= 1:
        # Floor
        floor_mat_cfg = MaterialPresets.mdl_family_object("floor", family=("carpet", "wood", "stone"))
        floor_mat_cfg.mdl.per_env = True  # Each environment gets different material
        floor_mat = MaterialRandomizer(floor_mat_cfg, seed=args.seed + 101)
        floor_mat.bind_handler(handler)
        randomizers["material_dynamic"].append(floor_mat)

        # Walls (all 4 share same seed for consistency)
        wall_seed = args.seed + 102
        for wall_name in ["wall_front", "wall_back", "wall_left", "wall_right"]:
            wall_mat_cfg = MaterialPresets.mdl_family_object(wall_name, family=("masonry", "architecture"))
            wall_mat_cfg.mdl.per_env = True  # Each environment gets different material
            wall_mat = MaterialRandomizer(wall_mat_cfg, seed=wall_seed)  # Same seed for all walls
            wall_mat.bind_handler(handler)
            randomizers["material_dynamic"].append(wall_mat)

        # Ceiling
        ceiling_mat_cfg = MaterialPresets.mdl_family_object("ceiling", family=("architecture", "wall_board"))
        ceiling_mat_cfg.mdl.per_env = True  # Each environment gets different material
        ceiling_mat = MaterialRandomizer(ceiling_mat_cfg, seed=args.seed + 103)
        ceiling_mat.bind_handler(handler)
        randomizers["material_dynamic"].append(ceiling_mat)

        log.info("Dynamic Objects: floor + 4 walls + ceiling (manual geometry materials, per_env=True)")

    # =========================================================================
    # STEP 3: Physics Randomization (ObjectRandomizer - Static Objects only)
    # =========================================================================

    log.info("\n[STEP 3] Physics Randomization (ObjectRandomizer)")
    log.info("-" * 70)

    box_physics = ObjectRandomizer(
        ObjectPresets.heavy_object("box_base"),
        seed=args.seed + 3,
    )
    box_physics.cfg.pose.rotation_delta_range = (0, 0)  # Disable rotation for stability
    box_physics.cfg.pose.position_delta_range = [(0, 0), (0, 0), (0, 0)]  # Disable position jitter
    box_physics.bind_handler(handler)
    box_physics()  # Apply once at start
    randomizers["object_physics"].append(box_physics)
    log.info("Static Object: box_base (mass randomization)")

    # =========================================================================
    # STEP 4: Light Randomization (LightRandomizer)
    # =========================================================================

    log.info("\n[STEP 4] Light Randomization (LightRandomizer)")
    log.info("-" * 70)

    from metasim.randomization import (
        LightColorRandomCfg,
        LightIntensityRandomCfg,
        LightOrientationRandomCfg,
        LightPositionRandomCfg,
        LightRandomCfg,
    )

    if args.render_mode == "pathtracing":
        main_range = (22000.0, 40000.0)
        corner_range = (10000.0, 18000.0)
    else:
        main_range = (16000.0, 30000.0)
        corner_range = (6000.0, 12000.0)

    # Main light with orientation randomization (simulates different lighting angles)
    main_light = LightRandomizer(
        LightRandomCfg(
            light_name="ceiling_main",
            intensity=LightIntensityRandomCfg(
                intensity_delta_range=main_range, use_delta=True, enabled=True, per_env=True
            ),
            color=LightColorRandomCfg(
                temperature_delta_range=(3000.0, 6000.0),
                use_temperature=True,
                use_delta=True,
                enabled=True,
                per_env=True,
            ),
            orientation=LightOrientationRandomCfg(
                angle_delta_range=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),  # Small angle variations
                use_delta=True,
                distribution="uniform",
                enabled=True,
                per_env=True,
            ),
        ),
        seed=args.seed + 4,
    )
    main_light.bind_handler(handler)
    randomizers["light"].append(main_light)

    # Corner lights with position and orientation randomization
    for i, light_name in enumerate(["ceiling_ne", "ceiling_nw", "ceiling_sw", "ceiling_se"]):
        corner_light = LightRandomizer(
            LightRandomCfg(
                light_name=light_name,
                intensity=LightIntensityRandomCfg(
                    intensity_delta_range=corner_range, use_delta=True, enabled=True, per_env=True
                ),
                color=LightColorRandomCfg(
                    temperature_delta_range=(2700.0, 5500.0),
                    use_temperature=True,
                    use_delta=True,
                    enabled=True,
                    per_env=True,
                ),
                position=LightPositionRandomCfg(
                    position_delta_range=((-0.5, 0.5), (-0.5, 0.5), (-0.3, 0.3)),  # Small position jitter
                    use_delta=True,
                    distribution="uniform",
                    enabled=True,
                    per_env=True,
                ),
            ),
            seed=args.seed + 5 + i,
        )
        corner_light.bind_handler(handler)
        randomizers["light"].append(corner_light)

    log.info(f"Configured {len(randomizers['light'])} lights (with position/orientation randomization)")

    # =========================================================================
    # STEP 5: Camera Randomization (CameraRandomizer)
    # =========================================================================

    log.info("\n[STEP 5] Camera Randomization (CameraRandomizer)")
    log.info("-" * 70)

    # Split environments into two groups for different camera randomization
    num_envs = handler.num_envs
    half = num_envs // 2
    first_half_envs = list(range(half))  # e.g., [0, 1] for 4 envs
    second_half_envs = list(range(half, num_envs))  # e.g., [2, 3] for 4 envs

    # Group 1: First half - Small orbit movements (conservative)
    camera_rand_group1 = CameraRandomizer(
        CameraRandomCfg(
            camera_name="main_camera",
            position=CameraPositionRandomCfg(
                position_delta_range=((-0.3, 0.3), (-0.3, 0.3), (-0.2, 0.2)),  # Small delta
                use_delta=True,
                distribution="uniform",
                enabled=True,
                per_env=False,  # Same random value for all envs in this group
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=(45, 60),  # Narrow FOV range
                use_fov=True,
                distribution="uniform",
                enabled=True,
                per_env=False,  # Same FOV for all envs in this group
            ),
            env_ids=first_half_envs,  # Only apply to first half
        ),
        seed=args.seed + 10,
    )
    camera_rand_group1.bind_handler(handler)
    randomizers["camera"].append(camera_rand_group1)
    log.info(f"Camera Group 1 (envs {first_half_envs}): Small orbit (±0.3m), FOV 45-60°, consistent within group")

    # Group 2: Second half - Large orbit movements (aggressive)
    camera_rand_group2 = CameraRandomizer(
        CameraRandomCfg(
            camera_name="main_camera",
            position=CameraPositionRandomCfg(
                position_delta_range=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),  # Large delta
                use_delta=True,
                distribution="uniform",
                enabled=True,
                per_env=False,  # Same random value for all envs in this group
            ),
            intrinsics=CameraIntrinsicsRandomCfg(
                fov_range=(60, 90),  # Wide FOV range
                use_fov=True,
                distribution="uniform",
                enabled=True,
                per_env=False,  # Same FOV for all envs in this group
            ),
            env_ids=second_half_envs,  # Only apply to second half
        ),
        seed=args.seed + 11,
    )
    camera_rand_group2.bind_handler(handler)
    randomizers["camera"].append(camera_rand_group2)
    log.info(f"Camera Group 2 (envs {second_half_envs}): Large orbit (±1.0m), FOV 60-90°, consistent within group")

    log.info("\n" + "=" * 70)
    log.info("All Randomizers Initialized")
    log.info("=" * 70)

    # Inspect ObjectRegistry
    if level >= 1:
        registry = ObjectRegistry.get_instance()
        log.info("\nObjectRegistry Contents:")
        log.info(f"  Static Objects: {registry.list_objects(lifecycle='static')}")
        log.info(
            f"  Dynamic Objects: {registry.list_objects(lifecycle='dynamic')} (will be populated after scene_rand())"
        )

    return randomizers


def apply_randomization(randomizers, level, handler=None, is_initial=False):
    """Apply all randomizers with global deferred visual flush.

    New Strategy (Performance Optimized):
    - Set global defer flag on handler to block ALL internal flushes
    - This includes: MaterialRandomizer, LightRandomizer, force_pose_nudge, etc.
    - Single atomic flush at the end (settle_passes=2)
    - Result: ~22 flushes → 1 flush (~15-30x speedup)

    Ensures all randomizations are applied atomically before flushing visuals,
    preventing intermediate states from being captured in recordings.

    Args:
        randomizers: Dictionary of randomizers
        level: Randomization level (0-3)
        handler: Simulation handler
        is_initial: Whether this is the initial call (for scene creation)
    """
    # Enable global defer flag (blocks ALL internal flush calls)
    if handler:
        handler._defer_all_visual_flushes = True

    try:
        # Scene creation/switching logic:
        # - Initial call: Always create scene (even level 0)
        # - Periodic call: Only switch scene at level 1+ (level 0: no switching)
        if randomizers["scene"]:
            if is_initial or level >= 1:
                scene_rand = randomizers["scene"]
                original_auto_flush = scene_rand.cfg.auto_flush_visuals
                scene_rand.cfg.auto_flush_visuals = False
                scene_rand()  # Each layer uses its own env_ids
                scene_rand.cfg.auto_flush_visuals = original_auto_flush

        # Level 1+: Material randomization
        if level >= 1:
            for mat_rand in randomizers["material_static"]:
                mat_rand()

            for mat_rand in randomizers["material_dynamic"]:
                mat_rand()

        # Level 2+: Lighting
        if level >= 2:
            for light_rand in randomizers["light"]:
                light_rand()

        # Level 3+: Camera
        if level >= 3:
            for cam_rand in randomizers["camera"]:
                cam_rand()

    finally:
        # Disable global defer flag and perform single comprehensive flush
        if handler:
            handler._defer_all_visual_flushes = False
            if hasattr(handler, "flush_visual_updates"):
                try:
                    # Unified settle_passes=2 balances quality and performance
                    handler.flush_visual_updates(wait_for_materials=True, settle_passes=2)
                except Exception as e:
                    log.debug(f"Failed to flush visual updates: {e}")


def run_replay(env, randomizers, init_state, all_actions, args):
    """Run trajectory replay with randomization."""
    os.makedirs("get_started/output", exist_ok=True)

    mode_names = {0: "manual", 1: "usd_table", 2: "usd_scene", 3: "full_usd"}
    video_path = f"get_started/output/12_dr_mode{args.scene_mode}_{mode_names[args.scene_mode]}_level{args.level}.mp4"

    obs_saver = ObsSaver(video_path=video_path)

    log.info("\n" + "=" * 70)
    log.info("Trajectory Replay with Domain Randomization")
    log.info("=" * 70)
    log.info(f"Video: {video_path}")
    log.info(f"Randomization interval: {args.randomize_interval} steps")

    # Initial randomization (create scene)
    apply_randomization(randomizers, args.level, env.handler, is_initial=True)

    # Store original positions for later updates
    # get_states() returns LOCAL coordinates (already subtracted env_origins)
    # For multi-env, we only need env_0's local coords as the reference template

    original_positions = {}
    for obj_name, obj_state in init_state["objects"].items():
        # obj_state["pos"] is a tensor: [num_envs, 3] for multi-env, [3,] for single-env
        # We use env_0's local coordinates as the reference
        pos = obj_state["pos"]
        if pos.ndim == 1:
            # Single environment: pos is [3,]
            pos_env0 = pos
        else:
            # Multi-environment: pos is [num_envs, 3], get env_0
            pos_env0 = pos[0]

        original_positions[f"obj_{obj_name}"] = {
            "x": float(pos_env0[0]),
            "y": float(pos_env0[1]),
            "z": float(pos_env0[2]),
        }

    for robot_name, robot_state in init_state["robots"].items():
        pos = robot_state["pos"]
        if pos.ndim == 1:
            pos_env0 = pos
        else:
            pos_env0 = pos[0]

        original_positions[f"robot_{robot_name}"] = {
            "x": float(pos_env0[0]),
            "y": float(pos_env0[1]),
            "z": float(pos_env0[2]),
        }

    # Update positions to match table (center + height)
    def update_positions_to_table():
        """
        Simple approach: Adjust only Z height to match table surface.

        Key insights:
        - get_states() returns LOCAL coordinates (already subtracted env_origins)
        - set_states() expects LOCAL coordinates (will add env_origins automatically)
        - get_table_bounds() returns WORLD coordinates

        So we only need to:
        1. Convert table bounds from world to local (subtract env_origin_0)
        2. Adjust Z in local coordinates
        3. set_states() will handle the rest
        """
        if not randomizers["scene"]:
            return

        # Get table bounds from env_0 (world coordinates)
        table_bounds = randomizers["scene"].get_table_bounds(env_id=0)
        if not table_bounds or abs(table_bounds.get("height", 0)) > 100:
            return

        # Convert world table height to local coordinates
        env_origin_0 = env.handler.scene.env_origins[0].cpu().numpy()
        table_height_local = table_bounds["height"] - env_origin_0[2]

        # Compute Z average from original positions (already in local coords)
        all_z = [original_positions[k]["z"] for k in original_positions]
        avg_z = sum(all_z) / len(all_z)

        log.info(f"Adjusting Z to table surface: {table_height_local:.3f}m (local coords)")

        # Directly modify init_state (which is already in local coords)
        # Handle both single-env ([3,]) and multi-env ([num_envs, 3]) cases
        for obj_name, obj_state in init_state["objects"].items():
            orig = original_positions[f"obj_{obj_name}"]
            new_z = table_height_local + (orig["z"] - avg_z) + 0.05
            if obj_state["pos"].ndim == 1:
                # Single environment
                obj_state["pos"][2] = new_z
            else:
                # Multi-environment: broadcast to all envs
                obj_state["pos"][:, 2] = new_z

        for robot_name, robot_state in init_state["robots"].items():
            orig = original_positions[f"robot_{robot_name}"]
            new_z = table_height_local + (orig["z"] - avg_z) + 0.05
            if robot_state["pos"].ndim == 1:
                # Single environment
                robot_state["pos"][2] = new_z
            else:
                # Multi-environment: broadcast to all envs
                robot_state["pos"][:, 2] = new_z

        # Set states (will add env_origins automatically)
        env.handler.set_states([init_state] * env.scenario.num_envs)

    # Initial position update
    update_positions_to_table()

    # Reset environment first (this may update camera using default look_at)
    obs, _ = env.reset(states=[init_state] * args.num_envs)

    # Update camera look_at AFTER reset: each environment can have different table height
    if randomizers["scene"]:
        # Collect table heights for each environment
        table_heights = []
        for env_id in range(env.scenario.num_envs):
            table_bounds = randomizers["scene"].get_table_bounds(env_id=env_id)
            if table_bounds and abs(table_bounds.get("height", 0)) < 100:
                table_heights.append(table_bounds["height"])
            else:
                # Fallback to default height
                table_heights.append(0.75)

        # Directly set camera poses for each environment (bypass _update_camera_pose)
        for camera in env.handler.cameras:
            if camera.name == "main_camera":
                camera_inst = env.handler.scene.sensors[camera.name]

                # Base camera position (local coordinates)
                base_pos = torch.tensor(camera.pos, device=env.handler.device)

                # Per-environment look_at (local Z varies by table height)
                look_at_list = []
                for env_id, table_height in enumerate(table_heights):
                    look_at_list.append([0.0, 0.0, table_height + 0.05])

                base_lookat = torch.tensor(look_at_list, device=env.handler.device)

                # Add env_origins to get world coordinates
                position_tensor = base_pos.unsqueeze(0) + env.handler.scene.env_origins
                camera_lookat_tensor = base_lookat + env.handler.scene.env_origins

                camera_inst.set_world_poses_from_view(position_tensor, camera_lookat_tensor)

                log.info(
                    "Camera look_at per env: " + ", ".join([f"env_{i}: Z={h:.3f}" for i, h in enumerate(table_heights)])
                )

    step = 0
    while step < len(all_actions[0]):
        # Periodic randomization
        if step % args.randomize_interval == 0 and step > 0:
            log.info(f"Step {step}: Applying randomization")
            apply_randomization(randomizers, args.level, env.handler)

        # Execute action
        actions = [all_actions[0][step]] * args.num_envs
        obs, reward, success, time_out, extras = env.step(actions)
        obs_saver.add(obs)

        if success.any() or time_out.any():
            log.info("Task completed")
            break

        step += 1

    obs_saver.save()
    log.info(f"\nVideo saved: {video_path}")

    # Show final Registry state
    if args.level >= 1:
        registry = ObjectRegistry.get_instance()
        log.info("\nFinal ObjectRegistry State:")
        log.info(f"  Total objects: {len(registry.list_objects())}")
        log.info(f"  Static: {registry.list_objects(lifecycle='static')}")
        log.info(f"  Dynamic: {registry.list_objects(lifecycle='dynamic')}")


def main():
    @configclass
    class Args:
        sim: Literal["isaacsim"] = "isaacsim"
        renderer: str | None = None
        robot: str = "franka"
        scene: str | None = None
        num_envs: int = 1
        env_spacing: float = 5.0
        headless: bool = False
        seed: int = 42
        scene_mode: Literal[0, 1, 2, 3] = 3
        level: Literal[0, 1, 2, 3] = 1
        randomize_interval: int = 60
        render_mode: Literal["raytracing", "pathtracing"] = "raytracing"

    args = tyro.cli(Args)

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info("=" * 70)
    log.info("Domain Randomization Demo")
    log.info("=" * 70)
    log.info(f"Scene Mode: {args.scene_mode}")
    log.info(f"Randomization Level: {args.level}")
    log.info(f"Seed: {args.seed}")

    # Create environment
    env = create_env(args)
    handler = env.handler

    # Load trajectory
    traj_filepath = env.traj_filepath
    init_states, all_actions, _ = get_traj(traj_filepath, env.scenario.robots[0], handler)
    init_state = init_states[0]

    # Initialize randomizers (NEW: Auto-initializes and populates ObjectRegistry)
    randomizers = initialize_randomizers(handler, args)

    # Run replay
    run_replay(env, randomizers, init_state, all_actions, args)

    # Cleanup
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()

    log.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
