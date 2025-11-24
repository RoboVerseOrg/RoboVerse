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
    # Randomizers
    CameraPresets,
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
        ),
        SphereLightCfg(
            name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
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
        log.info(f"Environment: Kujiale USD ({len(scene_paths)} scenes)")

        env_element = USDAssetPoolCfg(
            name="kujiale_scene",
            usd_paths=scene_paths,
            per_path_overrides=scene_configs,
            selection_strategy="random" if level >= 1 else "sequential",
        )
        environment_layer = EnvironmentLayerCfg(elements=[env_element])
    else:
        # Manual Scene - Complete room
        log.info("Environment: Manual geometry (10m x 10m x 5m)")
        base_cfg = ScenePresets.empty_room(
            room_size=10.0,
            wall_height=5.0,
        )
        environment_layer = base_cfg.environment_layer

    if mode >= 1:
        # USD Table
        table_paths, table_configs = SceneUSDCollections.table785(return_configs=True)
        log.info(f"Workspace: Table785 USD ({len(table_paths)} tables)")

        workspace_element = USDAssetPoolCfg(
            name="table",
            usd_paths=table_paths,
            per_path_overrides=table_configs,
            selection_strategy="random" if level >= 1 else "sequential",
        )
        workspace_layer = WorkspaceLayerCfg(elements=[workspace_element])
    else:
        # Manual Table (with default Plywood, MaterialRandomizer will randomize in level 1+)
        log.info("Workspace: Manual table (Plywood default, randomized in level 1+)")
        workspace_layer = WorkspaceLayerCfg(
            elements=[
                ManualGeometryCfg(
                    name="table",
                    geometry_type="cube",
                    size=(1.8, 1.8, 0.1),
                    position=(0.0, 0.0, 0.7 - 0.05),  # 0.65m (table surface at 0.7m)
                    default_material="roboverse_data/materials/arnold/Wood/Plywood.mdl",
                )
            ]
        )

    if mode >= 3:
        # Desktop Objects
        object_paths, object_configs = SceneUSDCollections.desktop_supplies(return_configs=True)
        log.info(f"Objects: Desktop supplies ({len(object_paths)} items, placing 3)")

        objects_layer = ObjectsLayerCfg(
            elements=[
                USDAssetPoolCfg(
                    name=f"desktop_object_{i + 1}",
                    usd_paths=object_paths,
                    per_path_overrides=object_configs,
                    selection_strategy="random" if level >= 1 else "sequential",
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

    scene_rand = SceneRandomizer(scene_cfg, seed=args.seed)
    scene_rand.bind_handler(handler)
    randomizers["scene"] = scene_rand

    log.info("SceneRandomizer created (manages Dynamic Objects lifecycle)")

    # =========================================================================
    # STEP 2: Material Randomization (NEW: Works for ALL objects)
    # =========================================================================

    log.info("\n[STEP 2] Material Randomization (MaterialRandomizer)")
    log.info("-" * 70)

    # Static Object material
    box_mat = MaterialRandomizer(
        MaterialPresets.mdl_family_object("box_base", family=("wood", "plastic")),
        seed=args.seed + 1,
    )
    box_mat.bind_handler(handler)
    randomizers["material_static"].append(box_mat)
    log.info("Static Object: box_base (wood/plastic/metal/ceramic materials)")

    # Dynamic Object materials (NEW FEATURE!)
    # Note: Only for Mode 0 (manual table)
    # Mode 1+ use USD tables with their own materials
    if mode == 0:
        table_mat = MaterialRandomizer(
            MaterialPresets.mdl_family_object("table", family=("wood", "metal")),
            seed=args.seed + 2,
        )
        table_mat.bind_handler(handler)
        randomizers["material_dynamic"].append(table_mat)
        log.info("Dynamic Object: table (Manual, wood/metal materials)")

    # Manual geometry materials (floor, walls, ceiling)
    # Only for modes with manual environment (mode < 2) and level >= 1
    if mode < 2 and level >= 1:
        # Floor
        floor_mat = MaterialRandomizer(
            MaterialPresets.mdl_family_object("floor", family=("carpet", "wood", "stone")),
            seed=args.seed + 101,
        )
        floor_mat.bind_handler(handler)
        randomizers["material_dynamic"].append(floor_mat)

        # Walls (all 4 share same seed for consistency)
        wall_seed = args.seed + 102
        for wall_name in ["wall_front", "wall_back", "wall_left", "wall_right"]:
            wall_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object(wall_name, family=("masonry", "architecture")),
                seed=wall_seed,  # Same seed for all walls
            )
            wall_mat.bind_handler(handler)
            randomizers["material_dynamic"].append(wall_mat)

        # Ceiling
        ceiling_mat = MaterialRandomizer(
            MaterialPresets.mdl_family_object("ceiling", family=("architecture", "wall_board")),
            seed=args.seed + 103,
        )
        ceiling_mat.bind_handler(handler)
        randomizers["material_dynamic"].append(ceiling_mat)

        log.info("Dynamic Objects: floor + 4 walls + ceiling (manual geometry materials)")

    # =========================================================================
    # STEP 3: Physics Randomization (ObjectRandomizer - Static Objects only)
    # =========================================================================

    log.info("\n[STEP 3] Physics Randomization (ObjectRandomizer)")
    log.info("-" * 70)

    box_physics = ObjectRandomizer(
        ObjectPresets.heavy_object("box_base"),
        seed=args.seed + 3,
    )
    box_physics.cfg.pose.rotation_range = (0, 0)  # Disable rotation for stability
    box_physics.cfg.pose.position_range = [(0, 0), (0, 0), (0, 0)]  # Disable position jitter
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
            intensity=LightIntensityRandomCfg(intensity_range=main_range, enabled=True),
            color=LightColorRandomCfg(temperature_range=(3000.0, 6000.0), use_temperature=True, enabled=True),
            orientation=LightOrientationRandomCfg(
                angle_range=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),  # Small angle variations
                relative_to_origin=True,
                distribution="uniform",
                enabled=True,
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
                intensity=LightIntensityRandomCfg(intensity_range=corner_range, enabled=True),
                color=LightColorRandomCfg(temperature_range=(2700.0, 5500.0), use_temperature=True, enabled=True),
                position=LightPositionRandomCfg(
                    position_range=((-0.5, 0.5), (-0.5, 0.5), (-0.3, 0.3)),  # Small position jitter
                    relative_to_origin=True,
                    distribution="uniform",
                    enabled=True,
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

    camera_rand = CameraRandomizer(
        CameraPresets.orbit_camera("main_camera"),
        seed=args.seed + 10,
    )
    camera_rand.bind_handler(handler)
    randomizers["camera"].append(camera_rand)
    log.info("Camera: orbit preset (circles around table center)")

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
                scene_rand()
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
    original_positions = {}
    for obj_name, obj_state in init_state["objects"].items():
        original_positions[f"obj_{obj_name}"] = {
            "x": float(obj_state["pos"][0]),
            "y": float(obj_state["pos"][1]),
            "z": float(obj_state["pos"][2]),
        }
    for robot_name, robot_state in init_state["robots"].items():
        original_positions[f"robot_{robot_name}"] = {
            "x": float(robot_state["pos"][0]),
            "y": float(robot_state["pos"][1]),
            "z": float(robot_state["pos"][2]),
        }

    # Update positions to match table (center + height)
    def update_positions_to_table():
        if not randomizers["scene"]:
            return

        table_bounds = randomizers["scene"].get_table_bounds(env_id=0)
        if not table_bounds or abs(table_bounds.get("height", 0)) > 100:
            return

        table_height = table_bounds["height"]
        table_center_x = (table_bounds["x_min"] + table_bounds["x_max"]) / 2
        table_center_y = (table_bounds["y_min"] + table_bounds["y_max"]) / 2

        # Compute system center (robot + objects)
        all_x = [original_positions[k]["x"] for k in original_positions]
        all_y = [original_positions[k]["y"] for k in original_positions]
        system_center_x = sum(all_x) / len(all_x)
        system_center_y = sum(all_y) / len(all_y)

        # Compute offset to align system to table center
        offset_x = table_center_x - system_center_x
        offset_y = table_center_y - system_center_y

        log.info(
            f"Adjusting positions: table center ({table_center_x:.2f}, {table_center_y:.2f}), height {table_height:.3f}"
        )

        # Compute original Z average
        all_z = [original_positions[k]["z"] for k in original_positions]
        avg_z = sum(all_z) / len(all_z)

        # Apply offset (rigid body translation - preserves ALL relative positions)
        for obj_name, obj_state in init_state["objects"].items():
            orig = original_positions[f"obj_{obj_name}"]
            obj_state["pos"][0] = orig["x"] + offset_x  # XY: center alignment
            obj_state["pos"][1] = orig["y"] + offset_y
            obj_state["pos"][2] = table_height + (orig["z"] - avg_z) + 0.05  # Z: preserve relative + clearance

        for robot_name, robot_state in init_state["robots"].items():
            orig = original_positions[f"robot_{robot_name}"]
            robot_state["pos"][0] = orig["x"] + offset_x
            robot_state["pos"][1] = orig["y"] + offset_y
            robot_state["pos"][2] = table_height + (orig["z"] - avg_z) + 0.05

        env.handler.set_states([init_state] * env.scenario.num_envs)

    # Initial position update
    update_positions_to_table()

    # Update camera look_at
    if randomizers["scene"]:
        table_bounds = randomizers["scene"].get_table_bounds(env_id=0)
        if table_bounds:
            for camera in env.handler.cameras:
                if camera.name == "main_camera":
                    camera.look_at = (0.0, 0.0, table_bounds["height"] + 0.05)
            if hasattr(env.handler, "_update_camera_pose"):
                env.handler._update_camera_pose()

    obs, _ = env.reset(states=[init_state] * args.num_envs)

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
