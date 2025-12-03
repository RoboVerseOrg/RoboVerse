from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import tyro
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.render import RenderCfg


@dataclass
class Args:
    render: RenderCfg = field(default_factory=RenderCfg)
    """Renderer options"""
    task: str = "put_banana"
    """Task name"""
    robot: str = "vega"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "isaacsim", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "isaacsim"
    """Simulator backend"""
    demo_start_idx: int | None = None
    """The index of the first demo to collect, None for all demos"""
    # max_demo_idx: int | None = None
    # """Maximum number of demos to collect, None for all demos"""
    num_demo_success: int | None = None
    """Target number of successful demos to collect"""
    retry_num: int = 0
    """Number of retries for a failed demo"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""
    split: Literal["train", "val", "test", "all"] = "all"
    """Split to collect"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    custom_save_dir: str | None = None
    """Custom base path for saving demos. If None, use default structure."""
    scene: str | None = None
    """Scene name"""
    run_all: bool = True
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""
    renderer: Literal["isaaclab", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"

    ## Domain randomization options
    enable_randomization: bool = True
    """Enable domain randomization during demo collection"""
    randomize_materials: bool = True
    """Enable material randomization (when randomization is enabled)"""
    randomize_lights: bool = True
    """Enable light randomization (when randomization is enabled)"""
    randomize_cameras: bool = True
    """Enable camera randomization (when randomization is enabled)"""
    randomize_physics: bool = True
    """Enable physics (mass/friction/pose) randomization using ObjectRandomizer"""
    randomize_zed_camera: bool = False
    """Apply bounded randomization to zed_rgb_camera's mount offset"""
    zed_cam_vertical_delta: float = 0.1
    """Max vertical offset (m) for zed_rgb_camera mount"""
    zed_cam_lateral_delta: float = 0.02
    """Max lateral (left/right) offset (m) for zed_rgb_camera mount"""
    zed_cam_forward_delta: float = 0.0
    """Max forward/backward offset (m) for zed_rgb_camera mount"""
    randomization_frequency: Literal["per_demo", "per_episode"] = "per_demo"
    """When to apply randomization: per_demo (once at start) or per_episode (every episode)"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""
    traj_filepath: str | None = (
        "/home/balen/murphy/isaaclab_rv/2/RoboVerse/eval_trajs/trackgrasphandrelative_vega_eval_20251126_133029_v2.pkl"
    )
    """Path to trajectory file. If None, uses env.traj_filepath"""

    def __post_init__(self):
        assert self.run_all or self.run_unfinished or self.run_failed, (
            "At least one of run_all, run_unfinished, or run_failed must be True"
        )
        # if self.max_demo_idx is None:
        #     self.max_demo_idx = math.inf
        if self.num_demo_success is None:
            self.num_demo_success = 100
        if self.demo_start_idx is None:
            self.demo_start_idx = 0

        # Validate randomization settings
        if self.enable_randomization:
            if not (
                self.randomize_materials or self.randomize_lights or self.randomize_cameras or self.randomize_physics
            ):
                log.warning("Randomization enabled but no randomization types selected, disabling randomization")
                self.enable_randomization = False

        log.info(f"Args: {self}")

        # Log randomization settings
        if self.enable_randomization:
            log.info("=" * 60)
            log.info("DOMAIN RANDOMIZATION CONFIGURATION")
            log.info(f"  Materials: {'✓' if self.randomize_materials else '✗'}")
            log.info(f"  Lights: {'✓' if self.randomize_lights else '✗'}")
            log.info(f"  Cameras: {'✓' if self.randomize_cameras else '✗'}")
            log.info(f"  Physics: {'✓' if self.randomize_physics else '✗'} (ObjectRandomizer)")
            log.info(f"  Frequency: {self.randomization_frequency}")
            log.info(f"  Seed: {self.randomization_seed if self.randomization_seed else 'Random'}")
            log.info("=" * 60)


args = tyro.cli(Args)

import multiprocessing as mp
import os

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch
from tqdm.rich import tqdm_rich as tqdm

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import SphereLightCfg
from metasim.scenario.robot import RobotCfg
from metasim.sim import BaseSimHandler
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu

rootutils.setup_root(__file__, pythonpath=True)

# Import randomization components (after rootutils setup)
try:
    from roboverse_pack.randomization import (
        CameraPresets,
        CameraRandomizer,
        LightPresets,
        LightRandomizer,
        MaterialPresets,
        MaterialRandomizer,
        ObjectPresets,
        ObjectRandomizer,
    )
    from metasim.randomization.camera_randomizer import CameraPositionRandomCfg, CameraRandomCfg
    from metasim.randomization.scene_randomizer import (
        SceneRandomizer,
        SceneRandomCfg,
        ManualGeometryCfg,
        EnvironmentLayerCfg,
        WorkspaceLayerCfg,
        USDAssetPoolCfg,
    )
    from metasim.randomization.presets.scene_presets import ScenePresets
    from metasim.randomization.presets.scene_presets import SceneUSDCollections

    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Randomization components not available: {e}")
    RANDOMIZATION_AVAILABLE = False


def log_randomization_result(
    randomizer_type: str, obj_name: str, property_name: str, before_value, after_value, unit: str = ""
):
    """Log randomization results in a consistent format."""
    if hasattr(before_value, "cpu"):
        before_str = str(before_value.cpu().numpy().round(3) if hasattr(before_value, "numpy") else before_value)
    else:
        before_str = str(before_value)

    if hasattr(after_value, "cpu"):
        after_str = str(after_value.cpu().numpy().round(3) if hasattr(after_value, "numpy") else after_value)
    else:
        after_str = str(after_value)

    log.info(f"  [{randomizer_type}] {obj_name}.{property_name}: {before_str} -> {after_str} {unit}")


def log_randomization_header(randomizer_name: str, description: str = ""):
    """Log a consistent header for randomization sections."""
    log.info("=" * 50)
    if description:
        log.info(f"{randomizer_name}: {description}")
    else:
        log.info(randomizer_name)


class DomainRandomizationManager:
    """Manages domain randomization for demo collection with unified interface."""

    def __init__(self, args: Args, scenario, handler):
        self.args = args
        self.scenario = scenario
        self.handler = handler
        self.randomizers = []
        self.scene_randomizer = None
        self._demo_count = 0

        # Early validation
        if not self._validate_setup():
            return

        log_randomization_header("DOMAIN RANDOMIZATION SETUP", "Initializing randomizers")
        self._setup_randomizers()
        log.info(f"Setup complete: {len(self.randomizers)} randomizers ready")

    def _validate_setup(self) -> bool:
        """Validate if randomization can be set up."""
        if not self.args.enable_randomization:
            log.info("Domain randomization disabled")
            return False

        if not RANDOMIZATION_AVAILABLE:
            log.warning("Domain randomization requested but components not available")
            return False

        return True

    def _setup_randomizers(self):
        """Initialize all randomizers based on configuration."""
        seed = self.args.randomization_seed
        self._setup_reproducibility(seed)

        # Setup scene randomizer first (table + room) - must be before other randomizers
        self._setup_scene_randomizer(seed)

        # Setup each randomization type symmetrically
        if self.args.randomize_materials:
            self._setup_material_randomizers(seed)

        if self.args.randomize_lights:
            self._setup_light_randomizers(seed)

        if self.args.randomize_cameras:
            self._setup_camera_randomizers(seed)

        if self.args.randomize_physics:
            self._setup_physics_randomizers(seed)

    def _setup_reproducibility(self, seed: int | None):
        """Setup global reproducibility if seed is provided."""
        if seed is not None:
            log.info(f"Setting up reproducible randomization with seed: {seed}")
            torch.manual_seed(seed)
            import numpy as np

            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def _setup_scene_randomizer(self, seed: int | None):
        """Setup scene randomizer with custom room and optional manual table."""
        log.info("  Setting up SceneRandomizer (custom room + optional workspace table)")

        # Room (environment layer) - manually create to adjust floor position
        room_size = 5.0
        wall_height = 4.0
        wall_thickness = 0.1
        half_room = room_size / 2.0
        half_thickness = wall_thickness / 2.0
        
        # Floor position: move down 0.05m (from 0.005 to -0.045)
        floor_position = (0.0, 0.0, -0.045)
        
        environment_layer = EnvironmentLayerCfg(
            elements=[
                # Floor (moved down 0.05m)
                ManualGeometryCfg(
                    name="floor",
                    geometry_type="cube",
                    size=(room_size, room_size, wall_thickness),
                    position=floor_position,
                    default_material="roboverse_data/materials/arnold/Carpet/Carpet_Beige.mdl",
                ),
                # Front wall (positive Y)
                ManualGeometryCfg(
                    name="wall_front",
                    geometry_type="cube",
                    size=(room_size + 2 * wall_thickness, wall_thickness, wall_height),
                    position=(0.0, half_room + half_thickness, wall_height / 2),
                    default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                ),
                # Back wall (negative Y)
                ManualGeometryCfg(
                    name="wall_back",
                    geometry_type="cube",
                    size=(room_size + 2 * wall_thickness, wall_thickness, wall_height),
                    position=(0.0, -half_room - half_thickness, wall_height / 2),
                    default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                ),
                # Left wall (negative X)
                ManualGeometryCfg(
                    name="wall_left",
                    geometry_type="cube",
                    size=(wall_thickness, room_size, wall_height),
                    position=(-half_room - half_thickness, 0.0, wall_height / 2),
                    default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                ),
                # Right wall (positive X)
                ManualGeometryCfg(
                    name="wall_right",
                    geometry_type="cube",
                    size=(wall_thickness, room_size, wall_height),
                    position=(half_room + half_thickness, 0.0, wall_height / 2),
                    default_material="roboverse_data/materials/arnold/Masonry/Brick_Pavers.mdl",
                ),
                # Ceiling
                ManualGeometryCfg(
                    name="ceiling",
                    geometry_type="cube",
                    size=(room_size, room_size, wall_thickness),
                    position=(0.0, 0.0, wall_height + wall_thickness / 2),
                    default_material="roboverse_data/materials/arnold/Architecture/Roof_Tiles.mdl",
                ),
            ],
        )

        workspace_layer = self._create_workspace_table_layer() if self.args.table else None

        scene_cfg = SceneRandomCfg(
            workspace_layer=workspace_layer,
            environment_layer=environment_layer,
            only_if_no_scene=False,  # Explicitly set to False to always create scene
            auto_flush_visuals=True,  # Auto flush visual updates
        )

        self.scene_randomizer = SceneRandomizer(scene_cfg, seed=seed)
        self.scene_randomizer.bind_handler(self.handler)
        table_state = "with USD pool table" if workspace_layer is not None else "without table"
        log.info(f"    Added SceneRandomizer (room, floor -0.05m, {table_state})")
        log.info(
            f"    SceneRandomizer config: only_if_no_scene={scene_cfg.only_if_no_scene}, "
            f"auto_flush_visuals={scene_cfg.auto_flush_visuals}"
        )

    def _create_workspace_table_layer(self) -> WorkspaceLayerCfg | None:
        """Create a workspace layer using USD pool table assets."""
        # Use USD Table785 assets from SceneUSDCollections
        table_paths, table_configs = SceneUSDCollections.table785(return_configs=True)
        log.info(f"    Workspace: Table785 USD pool ({len(table_paths)} tables available)")

        workspace_element = USDAssetPoolCfg(
            name="table",
            usd_paths=table_paths,
            per_path_overrides=table_configs,
            selection_strategy="random",  # Random selection for domain randomization
        )
        return WorkspaceLayerCfg(elements=[workspace_element])

    def _setup_material_randomizers(self, seed: int | None):
        """Setup material randomizers for scene geometry (walls, floor, ceiling) only.
        
        Note: Task objects' materials are NOT randomized (texture stays fixed).
        """
        # Skip material randomization for task objects (keep their textures fixed)
        objects = getattr(self.scenario, "objects", [])
        if objects:
            log.info(f"  Skipping material randomization for {len(objects)} task objects (textures stay fixed)")

        # Setup material randomizers for scene geometry (walls, floor, ceiling) only
        # These are created by SceneRandomizer as manual geometry
        log.info("  Setting up material randomizers for scene geometry (walls, floor, ceiling)")
        
        # Floor
        floor_mat = MaterialRandomizer(
            MaterialPresets.mdl_family_object("floor", family=("carpet", "wood", "stone")),
            seed=seed + 101 if seed is not None else None,
        )
        floor_mat.bind_handler(self.handler)
        self.randomizers.append(floor_mat)
        log.info("    Added MaterialRandomizer for floor")

        # Walls (all 4 share same seed for consistency)
        wall_seed = seed + 102 if seed is not None else None
        for wall_name in ["wall_front", "wall_back", "wall_left", "wall_right"]:
            wall_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object(wall_name, family=("masonry", "architecture")),
                seed=wall_seed,  # Same seed for all walls
            )
            wall_mat.bind_handler(self.handler)
            self.randomizers.append(wall_mat)
            log.info(f"    Added MaterialRandomizer for {wall_name}")

        # Ceiling
        ceiling_mat = MaterialRandomizer(
            MaterialPresets.mdl_family_object("ceiling", family=("architecture", "wall_board")),
            seed=seed + 103 if seed is not None else None,
        )
        ceiling_mat.bind_handler(self.handler)
        self.randomizers.append(ceiling_mat)
        log.info("    Added MaterialRandomizer for ceiling")

    def _setup_light_randomizers(self, seed: int | None):
        """Setup light randomizers for all lights except fixed lights."""
        from metasim.scenario.lights import DiskLightCfg, DomeLightCfg, SphereLightCfg

        lights = getattr(self.scenario, "lights", [])
        if not lights:
            log.info("  No lights found for light randomization")
            return

        # Fixed lights that should not be randomized
        fixed_light_names = {"fixed_ceiling_light"}

        log.info(f"  Setting up light randomizers for {len(lights)} lights")
        for light in lights:
            light_name = getattr(light, "name", f"light_{len(self.randomizers)}")
            
            # Skip fixed lights
            if light_name in fixed_light_names:
                log.info(f"    Skipping LightRandomizer for '{light_name}' (fixed light, not randomized)")
                continue

            if isinstance(light, DomeLightCfg):
                config = LightPresets.dome_ambient(light_name)
            elif isinstance(light, (SphereLightCfg, DiskLightCfg)):
                config = LightPresets.sphere_ceiling_light(light_name)
            else:
                log.warning(f"Unknown light type for {light_name}, using sphere_ceiling_light preset")
                config = LightPresets.sphere_ceiling_light(light_name)

            randomizer = LightRandomizer(config, seed=seed)
            randomizer.bind_handler(self.handler)
            self.randomizers.append(randomizer)
            log.info(f"    Added LightRandomizer for {light_name}")

    def _setup_camera_randomizers(self, seed: int | None):
        """Setup camera randomizers for all cameras."""
        cameras = getattr(self.scenario, "cameras", [])
        if not cameras:
            log.info("  No cameras found for camera randomization")
            return

        log.info(f"  Setting up camera randomizers for {len(cameras)} cameras")
        for camera in cameras:
            camera_name = getattr(camera, "name", f"camera_{len(self.randomizers)}")
            if camera_name == "zed_rgb_camera":
                if not self.args.randomize_zed_camera:
                    log.info("    Skipping CameraRandomizer for 'zed_rgb_camera' (locked)")
                    continue

                position_cfg = CameraPositionRandomCfg(
                    delta_range=(
                        (-self.args.zed_cam_forward_delta, self.args.zed_cam_forward_delta),
                        (-self.args.zed_cam_lateral_delta, self.args.zed_cam_lateral_delta),
                        (-self.args.zed_cam_vertical_delta, self.args.zed_cam_vertical_delta),
                    ),
                    use_delta=True,
                    distribution="uniform",
                    enabled=True,
                )
                config = CameraRandomCfg(camera_name=camera_name, position=position_cfg)
                log.info(
                    "    Added bounded CameraRandomizer for 'zed_rgb_camera' "
                    f"(Δx≤{self.args.zed_cam_forward_delta:.3f}m, "
                    f"Δy≤{self.args.zed_cam_lateral_delta:.3f}m, "
                    f"Δz≤{self.args.zed_cam_vertical_delta:.3f}m)"
                )
            else:
                config = CameraPresets.surveillance_camera(camera_name)
                log.info(f"    Added CameraRandomizer preset for '{camera_name}'")

            randomizer = CameraRandomizer(config, seed=seed)
            randomizer.bind_handler(self.handler)
            self.randomizers.append(randomizer)

    def _get_material_config(self, obj_name: str):
        """Get appropriate material configuration based on object type."""
        obj_lower = obj_name.lower()
        if "cube" in obj_lower:
            return MaterialPresets.mdl_family_object(obj_name, family="metal")
        elif "sphere" in obj_lower:
            return MaterialPresets.rubber_object(obj_name)
        else:
            return MaterialPresets.mdl_family_object(obj_name, family="wood")

    def _setup_physics_randomizers(self, seed: int | None):
        """Setup unified ObjectRandomizers for robots and objects."""
        robots = getattr(self.scenario, "robots", [])
        objects = getattr(self.scenario, "objects", [])

        self._setup_object_randomizers(robots, objects, seed)

    def _setup_object_randomizers(self, robots: list, objects: list, seed: int | None):
        """Setup unified ObjectRandomizers for all physical entities."""
        log.info("  Setting up ObjectRandomizers for physics randomization")

        # Robot randomization
        if robots:
            robot_name = robots[0] if isinstance(robots[0], str) else robots[0].name
            robot_randomizer = ObjectRandomizer(ObjectPresets.robot_base(robot_name), seed=seed)
            robot_randomizer.bind_handler(self.handler)
            self.randomizers.append(robot_randomizer)
            log.info(f"    Added ObjectRandomizer for robot {robot_name}")

        # Object randomization
        if objects:
            for obj in objects:
                obj_name = obj.name
                config = self._get_object_physics_config(obj_name)

                obj_randomizer = ObjectRandomizer(config, seed=seed)
                obj_randomizer.bind_handler(self.handler)
                self.randomizers.append(obj_randomizer)
                log.info(f"    Added ObjectRandomizer for {obj_name}")

        if not robots and not objects:
            log.info("    No robots or objects found for physics randomization")

    def _get_object_physics_config(self, obj_name: str):
        """Get appropriate physics configuration based on object type."""
        obj_lower = obj_name.lower()
        if "cube" in obj_lower:
            return ObjectPresets.grasping_target(obj_name)
        elif "sphere" in obj_lower:
            return ObjectPresets.bouncy_object(obj_name)
        else:
            return ObjectPresets.physics_only(obj_name)

    def randomize_for_demo(self, demo_idx: int, seed: int | None = None):
        """Apply randomization for a new demo."""
        if not self._should_randomize(demo_idx):
            return

        # Use provided seed or fall back to args.randomization_seed
        if seed is not None:
            self._setup_reproducibility(seed)
            log_randomization_header("DOMAIN RANDOMIZATION", f"Demo {demo_idx} (seed={seed})")
        else:
            log_randomization_header("DOMAIN RANDOMIZATION", f"Demo {demo_idx}")

        # Apply scene randomizer first (creates room)
        if self.scene_randomizer:
            log.info("  Calling SceneRandomizer to create room...")
            try:
                self.scene_randomizer()
                log.info("  ✓ SceneRandomizer applied successfully (room created)")
            except Exception as e:
                log.error(f"  ✗ SceneRandomizer failed: {e}")
                import traceback
                log.error(traceback.format_exc())

        # Apply all other randomizers and collect statistics
        stats = self._apply_all_randomizers()

        # Log summary
        self._log_randomization_summary(stats)
        self._demo_count += 1

    def _should_randomize(self, demo_idx: int) -> bool:
        """Check if randomization should be applied for this demo."""
        if not self.args.enable_randomization or not self.randomizers:
            return False

        return self.args.randomization_frequency == "per_demo" or (
            self.args.randomization_frequency == "per_episode" and demo_idx == 0
        )

    def _apply_all_randomizers(self) -> dict[str, int]:
        """Apply all randomizers and return statistics."""
        stats = {"ObjectRandomizer": 0, "MaterialRandomizer": 0, "LightRandomizer": 0, "CameraRandomizer": 0}

        for randomizer in self.randomizers:
            try:
                obj_name = self._get_randomizer_target_name(randomizer)
                randomizer_type = type(randomizer).__name__

                # Apply randomization
                randomizer()
                stats[randomizer_type] = stats.get(randomizer_type, 0) + 1
                log.info(f"  Applied {randomizer_type} for {obj_name}")

            except Exception as e:
                log.warning(f"  {type(randomizer).__name__} failed for {obj_name}: {e}")

        return stats

    def _get_randomizer_target_name(self, randomizer) -> str:
        """Extract target object name from randomizer configuration."""
        if not hasattr(randomizer, "cfg"):
            return "unknown"

        cfg = randomizer.cfg
        if hasattr(cfg, "obj_name"):
            return cfg.obj_name
        elif hasattr(cfg, "light_name"):
            return cfg.light_name
        elif hasattr(cfg, "camera_name"):
            return cfg.camera_name
        else:
            return "unknown"

    def _log_randomization_summary(self, stats: dict[str, int]):
        """Log a summary of applied randomizers."""
        applied_types = [f"{name}: {count}" for name, count in stats.items() if count > 0]
        if applied_types:
            log.info(f"Applied randomizers: {', '.join(applied_types)}")
        else:
            log.info("No randomizers were applied")


def get_actions(all_actions, env, demo_idxs: list[int], robot: RobotCfg, single_actions=None, max_frames: int = 70):
    """
    Get actions for current step.
    If single_actions is provided, use it for all environments (hardcoded single trajectory).
    Otherwise, use all_actions[demo_idx] for each environment.
    Limited to max_frames frames.
    """
    action_idxs = env._episode_steps

    actions = []
    for env_id, (demo_idx, action_idx) in enumerate(zip(demo_idxs, action_idxs)):
        # Limit to max_frames
        if action_idx >= max_frames:
            # Use last action if exceeded max_frames
            if single_actions is not None:
                action = single_actions[min(max_frames - 1, len(single_actions) - 1)]
            else:
                action = all_actions[demo_idx][min(max_frames - 1, len(all_actions[demo_idx]) - 1)]
        elif single_actions is not None:
            # Use hardcoded single trajectory
            if action_idx < len(single_actions):
                action = single_actions[action_idx]
            else:
                action = single_actions[-1]
        else:
            # Use original logic with multiple trajectories
            if action_idx < len(all_actions[demo_idx]):
                action = all_actions[demo_idx][action_idx]
            else:
                action = all_actions[demo_idx][-1]

        actions.append(action)

    return actions


def get_run_out(all_actions, env, demo_idxs: list[int], single_actions=None, max_frames: int = 70) -> list[bool]:
    """
    Check if actions have run out.
    If single_actions is provided, use it for all environments (hardcoded single trajectory).
    Otherwise, use all_actions[demo_idx] for each environment.
    Limited to max_frames frames.
    """
    action_idxs = env._episode_steps
    if single_actions is not None:
        # Use hardcoded single trajectory, limit to max_frames
        run_out = [action_idx >= min(len(single_actions), max_frames) for action_idx in action_idxs]
    else:
        # Use original logic with multiple trajectories, limit to max_frames
        run_out = [
            action_idx >= min(len(all_actions[demo_idx]), max_frames)
            for demo_idx, action_idx in zip(demo_idxs, action_idxs)
        ]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue, robot_cfg: RobotCfg, task_desc: str):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo, robot_cfg=robot_cfg, task_desc=task_desc)


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        # Only start checking after minimum steps
        if step >= min_steps:
            if prev_state is not None:
                # Check if key states are stable (focus on articulated objects)
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            # Check DOF positions for articulated objects
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                # Additional validation: check if we're stable at the RIGHT state
                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        # We're stable but at wrong state - force more simulation
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False
                        # Continue simulating to let physics settle properly

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:  # Stable for 2 consecutive steps at correct state
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    # Final validation if we ran out of steps
    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    # Final state refresh
    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True  # Can't validate, assume correct

    # Focus on articulated objects which are most prone to reset issues
    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True  # No critical objects to validate

    tolerance = 5e-3  # Reasonable tolerance for DOF positions

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        # Check DOF positions for articulated objects (most critical for demo consistency)
        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                # Log the specific difference for debugging
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def force_reset_to_state(env, state, env_id):
    """Force reset environment to specific state with validation."""
    env.reset(states=[state], env_ids=[env_id])
    # Pass expected state for validation
    ensure_clean_state(env.handler, expected_state=state)
    # Reset episode counter AFTER stabilization to ensure demo starts from action 0
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0


def reset_to_task_initial_state(env, env_id):
    """Reset environment to task's initial state (not using trajectory initial state)."""
    env.reset(env_ids=[env_id])
    # Wait for environment to stabilize
    ensure_clean_state(env.handler)
    # Reset episode counter AFTER stabilization to ensure demo starts from action 0
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0


global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0


class DemoCollector:
    def __init__(self, handler, robot_cfg, task_desc="", demo_start_idx=0):
        assert isinstance(handler, BaseSimHandler)
        self.handler = handler
        self.robot_cfg = robot_cfg
        self.task_desc = task_desc
        self.cache: dict[int, list[dict]] = {}
        self.save_request_queue = mp.Queue()
        self.save_proc = mp.Process(target=save_demo_mp, args=(self.save_request_queue, robot_cfg, task_desc))
        self.save_proc.start()

        TaskName = args.task
        if args.custom_save_dir:
            self.base_save_dir = args.custom_save_dir
        else:
            additional_str = f"-{args.cust_name}" if args.cust_name else ""
            self.base_save_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

        self.success_counter = demo_start_idx
        self.failed_counter = demo_start_idx
        log.info(
            f"Initialized counters from demo_start_idx={demo_start_idx}: success={self.success_counter}, failed={self.failed_counter}"
        )

    def _get_max_demo_index(self, status: str) -> int:
        status_dir = os.path.join(self.base_save_dir, status)
        if not os.path.exists(status_dir):
            return 0

        max_idx = -1
        for item in os.listdir(status_dir):
            if item.startswith("demo_") and os.path.isdir(os.path.join(status_dir, item)):
                try:
                    idx = int(item.split("_")[1])
                    max_idx = max(max_idx, idx)
                except (ValueError, IndexError):
                    continue

        return max_idx + 1

    def create(self, demo_idx: int, data_dict: dict):
        assert demo_idx not in self.cache
        assert isinstance(demo_idx, int)
        self.cache[demo_idx] = [data_dict]

    def add(self, demo_idx: int, data_dict: dict):
        if data_dict is None:
            log.warning("Skipping adding obs to DemoCollector because obs is None")
        assert demo_idx in self.cache
        self.cache[demo_idx].append(deepcopy(tensor_to_cpu(data_dict)))

    def save(self, demo_idx: int, status: str):
        assert demo_idx in self.cache
        assert status in ["success", "failed"], f"Invalid status: {status}"

        if status == "success":
            continuous_idx = self.success_counter
            self.success_counter += 1
        else:  # failed
            continuous_idx = self.failed_counter
            self.failed_counter += 1

        save_dir = os.path.join(self.base_save_dir, status, f"demo_{continuous_idx:04d}")
        if os.path.exists(os.path.join(save_dir, "status.txt")):
            os.remove(os.path.join(save_dir, "status.txt"))

        os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving demo {demo_idx} (original) as {continuous_idx:04d} (continuous) to {save_dir}")

        ## Option 1: Save immediately, blocking and slower

        from metasim.utils.save_util import save_demo

        save_demo(save_dir, self.cache[demo_idx], self.robot_cfg, self.task_desc)

        if status == "failed":
            with open(os.path.join(save_dir, "status.txt"), "w") as f:
                f.write(status)

        ## Option 2: Save in a separate process, non-blocking, not friendly to KeyboardInterrupt
        # self.save_request_queue.put({"demo": self.cache[demo_idx], "save_dir": save_dir})

    def delete(self, demo_idx: int):
        assert demo_idx in self.cache
        del self.cache[demo_idx]

    def final(self):
        self.save_request_queue.put(None)  # signal to save_demo_mp to exit
        self.save_proc.join()
        assert self.cache == {}


def should_skip(log_dir: str, demo_idx: int):
    demo_name = f"demo_{demo_idx:04d}"
    success_path = os.path.join(log_dir, "success", demo_name, "status.txt")
    failed_path = os.path.join(log_dir, "failed", demo_name, "status.txt")

    if args.run_all:
        return False

    if args.run_unfinished:
        if not os.path.exists(success_path) and not os.path.exists(failed_path):
            return False
        return True

    if args.run_failed:
        if os.path.exists(success_path):
            return is_status_success(log_dir, demo_idx)
        return False

    return True


def is_status_success(log_dir: str, demo_idx: int) -> bool:
    demo_name = f"demo_{demo_idx:04d}"
    status_path = os.path.join(log_dir, "success", demo_name, "status.txt")

    if os.path.exists(status_path):
        return open(status_path).read().strip() == "success"
    return False


class DemoIndexer:
    def __init__(self, save_root_dir: str, start_idx: int, end_idx: int, pbar: tqdm):
        self.save_root_dir = save_root_dir
        self._next_idx = start_idx
        self.end_idx = end_idx
        self.pbar = pbar
        self._skip_if_should()

    @property
    def next_idx(self):
        return self._next_idx

    def _skip_if_should(self):
        while should_skip(self.save_root_dir, self._next_idx):
            global global_step, tot_success, tot_give_up
            if is_status_success(self.save_root_dir, self._next_idx):
                tot_success += 1
            else:
                tot_give_up += 1
            self.pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
            self.pbar.update(1)
            log.info(f"Demo {self._next_idx} already exists, skipping...")
            self._next_idx += 1

    def move_on(self):
        self._next_idx += 1
        self._skip_if_should()


def main():
    global global_step, tot_success, tot_give_up
    task_cls = get_task_class(args.task)

    # Configure Zed RGB camera with mount (same as replay_demo.py)
    from scipy.spatial.transform import Rotation as R
    
    fx = 365.5782165527344
    fy = 365.5782165527344
    cx = 494.15985107421875
    cy = 301.70770263671875
    img_width = 960
    img_height = 600
    focal_length_mm = 2.2112011909484863
    focal_length_cm = focal_length_mm / 10.0
    horizontal_aperture_cm = img_width * focal_length_cm / fx
    
    quat_xyzw = R.from_euler("xyz", [0, 0, 0], degrees=True).as_quat()
    quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  # convert to wxyz
    translation_from_torso_l3 = (0.01742, 0.0302, 0.75528)  # z increased by 0.3m to raise camera view

    camera = PinholeCameraCfg(
        name="zed_rgb_camera",
        width=img_width,
        height=img_height,
        data_types=["rgb"],
        focal_length=focal_length_cm,
        horizontal_aperture=horizontal_aperture_cm,
        mount_to=args.robot,
        mount_link="torso_l3",
        mount_pos=translation_from_torso_l3,
        mount_quat=quat,
    )
    
    # Add fixed light at (0, 0, 2.5) with intensity 10000 (always present, not randomized)
    fixed_light = SphereLightCfg(
        name="fixed_ceiling_light",
        intensity=10000.0,
        color=(1.0, 1.0, 1.0),
        radius=0.5,
        pos=(0.0, 0.0, 2.5),
        normalize=False,  # Don't normalize intensity
    )
    
    # Get existing lights from scenario (if any) and add fixed light
    existing_lights = list(getattr(task_cls.scenario, "lights", [])) if hasattr(task_cls.scenario, "lights") and task_cls.scenario.lights else []
    all_lights = existing_lights + [fixed_light]
    
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        lights=all_lights,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    log.info(f"Added fixed light 'fixed_ceiling_light' at (0, 0, 2.5) with intensity 10000 (not randomized)")
    robot = get_robot(args.robot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    # Initialize domain randomization manager
    randomization_manager = DomainRandomizationManager(args, scenario, env.handler)
    ## Data
    # Use specified trajectory filepath or fall back to env.traj_filepath
    traj_filepath = args.traj_filepath if args.traj_filepath is not None else env.traj_filepath
    assert traj_filepath is not None, "Trajectory filepath must be provided via args.traj_filepath or env.traj_filepath"
    assert os.path.exists(traj_filepath), f"Trajectory file does not exist: {traj_filepath}"
    log.info(f"Loading trajectory from: {traj_filepath}")
    init_states, all_actions, all_states = get_traj(traj_filepath, robot, env.handler)

    # Hardcode: Use only the first trajectory (demo 0) for actions
    hardcoded_demo_idx = 0
    max_frames = 70  # Limit to first 70 frames
    log.info(f"Hardcoded: Using only demo {hardcoded_demo_idx} from trajectory file")
    log.info(f"Total demos in file: {len(all_actions)}, will reuse demo {hardcoded_demo_idx} with different DR")
    log.info(f"Limiting replay to first {max_frames} frames")
    
    # Use only the first demo's actions (initial state will come from task, not trajectory)
    single_actions = all_actions[hardcoded_demo_idx]
    single_states = all_states[hardcoded_demo_idx] if all_states else None
    
    # Note: We no longer use trajectory's initial state for reset.
    # Instead, we use task's initial state via env.reset() without states parameter.
    log.info("Using task's initial state for reset (not trajectory initial state)")
    
    # Log trajectory length info
    if single_actions:
        original_length = len(single_actions)
        effective_length = min(original_length, max_frames)
        log.info(f"Trajectory length: {original_length} frames, will replay {effective_length} frames")
    
    n_demo = 1  # Only one demo, but will be reused
    log.info(f"Will collect {args.num_demo_success} demos by reusing demo {hardcoded_demo_idx} with different domain randomization")

    ########################################################
    ## Main
    ########################################################
    # if args.max_demo_idx > n_demo:
    #     log.warning(
    #         f"Max demo {args.max_demo_idx} is greater than the number of demos in the dataset {n_demo}, using {n_demo}"
    #     )
    # max_demo = min(args.max_demo_idx, n_demo)
    max_demo = n_demo
    try_num = args.retry_num + 1

    ## Demo collection state machine:
    ## CollectingDemo -> Success -> FinalizeDemo -> NextDemo
    ## CollectingDemo -> Timeout -> Retry/GiveUp -> NextDemo

    ## Setup
    # Get task description from environment
    task_desc = getattr(env, "task_desc", "")
    collector = DemoCollector(env.handler, robot, task_desc, demo_start_idx=args.demo_start_idx)
    # pbar = tqdm(total=max_demo - args.demo_start_idx, desc="Collecting demos")
    pbar = tqdm(total=args.num_demo_success, desc="Collecting successful demos")

    ## State variables
    failure_count = [0] * env.handler.num_envs
    steps_after_success = [0] * env.handler.num_envs
    finished = [False] * env.handler.num_envs
    TaskName = args.task

    if args.cust_name is not None:
        additional_str = f"-{args.cust_name}"
    else:
        additional_str = ""

    if args.custom_save_dir:
        save_root_dir = args.custom_save_dir
    else:
        save_root_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    demo_indexer = DemoIndexer(
        save_root_dir=save_root_dir,
        start_idx=args.demo_start_idx,
        end_idx=args.num_demo_success,  # Will create demos up to num_demo_success
        pbar=pbar,
    )
    
    # Hardcode: Always use demo 0 from trajectory, but create new demo indices for saving
    demo_idxs = []
    for env_id in range(env.handler.num_envs):
        demo_idxs.append(demo_indexer.next_idx)
        demo_indexer.move_on()
    log.info(f"Initialize with demo idxs: {demo_idxs} (using trajectory demo {hardcoded_demo_idx})")

    # Track continuous demo index for randomization seed
    continuous_demo_idx = 0

    ## Apply initial randomization with seed based on continuous_demo_idx
    base_seed = args.randomization_seed if args.randomization_seed is not None else 42
    current_seed = base_seed + continuous_demo_idx
    log.info(f"Applying randomization with seed {current_seed} (base={base_seed}, offset={continuous_demo_idx})")
    randomization_manager.randomize_for_demo(continuous_demo_idx, seed=current_seed)

    ## Reset to task's initial state (not using trajectory initial state)
    log.info("Resetting to task's initial state (not using trajectory initial state)")
    obs, extras = env.reset()

    ## Wait for environment to stabilize after reset (before counting demo steps)
    # For initial setup, we can't validate individual states easily, so just ensure stability
    ensure_clean_state(env.handler)

    ## Reset episode step counters AFTER stabilization
    if hasattr(env, "_episode_steps"):
        for env_id in range(env.handler.num_envs):
            env._episode_steps[env_id] = 0

    ## Now record the clean, stabilized initial state
    obs = env.handler.get_states()
    obs = state_tensor_to_nested(env.handler, obs)
    for env_id, demo_idx in enumerate(demo_idxs):
        log.info(f"Starting Demo {demo_idx} in Env {env_id}")
        collector.create(demo_idx, obs[env_id])

    ## Main Loop
    stop_flag = False

    while not all(finished):
        if stop_flag:
            pass

        if tot_success >= args.num_demo_success:
            log.info(f"Reached target number of successful demos ({args.num_demo_success}).")
            stop_flag = True

        if demo_indexer.next_idx >= args.num_demo_success:
            log.info(f"Reached target number of demos ({args.num_demo_success}).")
            stop_flag = True
            break

        pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
        max_frames = 70  # Limit to first 70 frames
        actions = get_actions(all_actions, env, demo_idxs, robot, single_actions=single_actions, max_frames=max_frames)
        obs, reward, success, time_out, extras = env.step(actions)
        obs = state_tensor_to_nested(env.handler, obs)
        run_out = get_run_out(all_actions, env, demo_idxs, single_actions=single_actions, max_frames=max_frames)

        for env_id in range(env.handler.num_envs):
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            collector.add(demo_idx, obs[env_id])

        for env_id in success.nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            if steps_after_success[env_id] == 0:
                log.info(f"Demo {demo_idx} in Env {env_id} succeeded!")
                tot_success += 1
                pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

            if not run_out[env_id] and steps_after_success[env_id] < args.tot_steps_after_success:
                steps_after_success[env_id] += 1
            else:
                steps_after_success[env_id] = 0
                collector.save(demo_idx, status="success")
                collector.delete(demo_idx)

                if (not stop_flag) and (demo_indexer.next_idx < args.num_demo_success):
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    continuous_demo_idx = new_demo_idx  # Update continuous index for seed
                    log.info(f"Transitioning Env {env_id}: Demo {demo_idx} to Demo {new_demo_idx} (reusing trajectory demo {hardcoded_demo_idx})")

                    # Apply new randomization with different seed
                    current_seed = base_seed + continuous_demo_idx
                    log.info(f"Applying randomization with seed {current_seed} for demo {new_demo_idx}")
                    randomization_manager.randomize_for_demo(new_demo_idx, seed=current_seed)
                    reset_to_task_initial_state(env, env_id)  # Use task's initial state

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                    run_out[env_id] = False
                else:
                    finished[env_id] = True

        for env_id in (time_out | torch.tensor(run_out, device=time_out.device)).nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            log.info(f"Demo {demo_idx} in Env {env_id} timed out!")
            collector.save(demo_idx, status="failed")
            collector.delete(demo_idx)
            failure_count[env_id] += 1

            if failure_count[env_id] < try_num:
                log.info(f"Demo {demo_idx} failed {failure_count[env_id]} times, retrying...")
                # Apply new randomization with different seed (use demo_idx as offset)
                current_seed = base_seed + demo_idx + failure_count[env_id] * 1000  # Add large offset for retries
                log.info(f"Applying randomization with seed {current_seed} for retry")
                randomization_manager.randomize_for_demo(demo_idx, seed=current_seed)
                reset_to_task_initial_state(env, env_id)  # Use task's initial state

                obs = env.handler.get_states()
                obs = state_tensor_to_nested(env.handler, obs)
                collector.create(demo_idx, obs[env_id])
            else:
                log.error(f"Demo {demo_idx} failed too many times, giving up")
                failure_count[env_id] = 0
                tot_give_up += 1
                # pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                if demo_indexer.next_idx < args.num_demo_success:
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    continuous_demo_idx = new_demo_idx  # Update continuous index for seed
                    # Apply new randomization with different seed
                    current_seed = base_seed + continuous_demo_idx
                    log.info(f"Applying randomization with seed {current_seed} for demo {new_demo_idx}")
                    randomization_manager.randomize_for_demo(new_demo_idx, seed=current_seed)
                    reset_to_task_initial_state(env, env_id)  # Use task's initial state

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                else:
                    finished[env_id] = True

        global_step += 1

    log.info("Finalizing")
    collector.final()
    env.close()


if __name__ == "__main__":
    main()
