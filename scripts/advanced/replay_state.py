from __future__ import annotations

import logging
import os
import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image

from metasim.randomization import (
    LightColorRandomCfg,
    LightIntensityRandomCfg,
    LightOrientationRandomCfg,
    LightPositionRandomCfg,
    LightRandomCfg,
    LightRandomizer,
    SceneRandomizer,
)
from metasim.randomization.presets.scene_presets import ScenePresets
from metasim.randomization.scene_randomizer import SceneMaterialPoolCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.render import RenderCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.state import TensorState

rootutils.setup_root(__file__, pythonpath=True)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    """Replay trajectory for a given task."""

    task: str = "put_banana"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx", "isaacsim"] = (
        "isaacsim"
    )
    renderer: (
        Literal["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3", "isaacsim"] | None
    ) = None

    num_envs: int = 1
    headless: bool = True

    save_image_dir: str | None = None
    save_video_path: str | None = "test_output/test_replay.mp4"

    # Scene randomization options
    enable_scene: bool = True
    """Enable scene randomization (floor, walls, ceiling)"""
    scene_type: Literal["empty_room", "tabletop_workspace", "floor_only"] = "empty_room"
    room_size: float = 4.0
    """Size of the room (square, in meters)"""
    wall_height: float = 5.0
    """Height of walls (in meters)"""
    table_height: float = 0.7
    """Height of table surface from ground (in meters)"""
    randomize_materials: bool = True
    """Enable material randomization for scene elements"""
    randomize_lights: bool = True
    """Enable light randomization (intensity, color, position)"""
    base_seed: int = 1
    """Base random seed for scene randomization"""

    # Multi-variant options
    num_variants: int = 10
    """Number of material variants to generate (each uses different seed)"""

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def _suffix_path(p: str | None, suffix: str) -> str | None:
    if p is None:
        return None
    base, ext = os.path.splitext(p)
    if ext:
        return f"{base}_{suffix}{ext}"
    return f"{p}_{suffix}"


class SceneRandomizationManager:
    """Manages scene randomization across multiple variants without environment restart."""

    def __init__(self, handler, scenario, args):
        self.handler = handler
        self.scenario = scenario
        self.args = args
        self.scene_randomizer = None
        self.light_randomizers = []

        if not args.enable_scene:
            log.info("Scene randomization disabled")
            return

        # Create scene configuration
        log.info("=" * 70)
        log.info("SCENE RANDOMIZATION SETUP")
        log.info("=" * 70)
        log.info(f"  Scene type: {args.scene_type}")
        log.info(f"  Room size: {args.room_size}m x {args.room_size}m")
        log.info(f"  Wall height: {args.wall_height}m")
        log.info(f"  Material randomization: {'Enabled' if args.randomize_materials else 'Disabled'}")
        log.info(f"  Light randomization: {'Enabled' if args.randomize_lights else 'Disabled'}")
        log.info(f"  Base seed: {args.base_seed}")
        log.info(f"  Number of variants: {args.num_variants}")

        # Create scene preset
        if args.scene_type == "empty_room":
            scene_cfg = ScenePresets.empty_room(
                room_size=args.room_size,
                wall_height=args.wall_height,
                wall_thickness=0.1,
            )
        elif args.scene_type == "tabletop_workspace":
            scene_cfg = ScenePresets.tabletop_workspace(
                room_size=args.room_size,
                wall_height=args.wall_height,
                table_size=(1.8, 1.8, 0.1),
                table_height=args.table_height,
            )
        elif args.scene_type == "floor_only":
            scene_cfg = ScenePresets.floor_only(
                floor_size=args.room_size,
                floor_thickness=0.1,
            )
        else:
            log.error(f"Unknown scene type: {args.scene_type}")
            return

        # Override material settings
        if not args.randomize_materials:
            log.info("\nUsing fixed materials")
            scene_cfg.floor_materials = SceneMaterialPoolCfg(
                material_paths=["roboverse_data/materials/arnold/Carpet/Carpet_Beige.mdl"],
                selection_strategy="sequential",
            )
            scene_cfg.wall_materials = SceneMaterialPoolCfg(
                material_paths=["roboverse_data/materials/arnold/Masonry/Stucco.mdl"],
                selection_strategy="sequential",
            )
            scene_cfg.ceiling_materials = SceneMaterialPoolCfg(
                material_paths=["roboverse_data/materials/arnold/Architecture/Ceiling_Tiles.mdl"],
                selection_strategy="sequential",
            )
            scene_cfg.table_materials = SceneMaterialPoolCfg(
                material_paths=["roboverse_data/materials/arnold/Wood/Plywood.mdl"],
                selection_strategy="sequential",
            )
        else:
            log.info("\nUsing randomized materials:")
            log.info("  Floor: ~150 materials")
            log.info("  Walls: ~150 materials")
            log.info("  Ceiling: ~50 materials")

        # Create scene randomizer (will be called multiple times with different seeds)
        self.scene_cfg = scene_cfg

        # Setup light randomizers if enabled
        if args.randomize_lights:
            self._setup_light_randomizers()

        log.info("=" * 70)

    def _setup_light_randomizers(self):
        """Setup light randomizers for all lights."""
        lights = getattr(self.scenario, "lights", [])
        if not lights:
            log.info("  No lights found for light randomization")
            return

        log.info("\n[Light Randomization Setup]")
        log.info(f"  Found {len(lights)} lights to randomize")

        # Light intensity ranges (bright enough for enclosed room!)
        # Following 12_domain_randomization.py settings
        ceiling_main_range = (18000.0, 32000.0)  # Main light: 18K-32K
        ceiling_corner_range = (7000.0, 14000.0)  # Corner lights: 7K-14K each
        # Total range: ~46K to ~88K (ensures bright illumination)

        for light in lights:
            light_name = getattr(light, "name", f"light_{len(self.light_randomizers)}")

            if light_name == "ceiling_main":
                # Main DiskLight - full randomization including orientation
                light_cfg = LightRandomCfg(
                    light_name=light_name,
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
                        position_range=((-2.0, 2.0), (-2.0, 2.0), (-0.5, 0.5)),
                        relative_to_origin=True,
                        distribution="uniform",
                        enabled=True,
                    ),
                    orientation=LightOrientationRandomCfg(
                        angle_range=((-30.0, 30.0), (-30.0, 30.0), (-180.0, 180.0)),
                        relative_to_origin=True,
                        distribution="uniform",
                        enabled=True,
                    ),
                    randomization_mode="combined",
                )
                log.info(
                    f"    Main light: intensity {ceiling_main_range[0] / 1000:.0f}K-{ceiling_main_range[1] / 1000:.0f}K with orientation"
                )
            else:
                # Corner SphereLights - intensity, color, position
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
                        position_range=((-1.5, 1.5), (-1.5, 1.5), (-0.5, 0.5)),
                        relative_to_origin=True,
                        distribution="uniform",
                        enabled=True,
                    ),
                    randomization_mode="combined",
                )

            # Store the config (will create randomizer with seed later)
            self.light_randomizers.append(light_cfg)
            log.info(f"    Added config for {light_name}")

    def randomize_for_variant(self, variant_id: int):
        """Apply randomization for a specific variant.

        Args:
            variant_id: Variant ID (0, 1, 2, ...)
        """
        if not self.args.enable_scene:
            return

        seed = self.args.base_seed + variant_id
        log.info(f"\nApplying randomization for variant {variant_id + 1} (seed={seed})")

        # Apply scene randomization (materials)
        if self.args.randomize_materials:
            scene_randomizer = SceneRandomizer(self.scene_cfg, seed=seed)
            scene_randomizer.bind_handler(self.handler)
            scene_randomizer()
            log.info("  ✓ Scene materials randomized")

        # Apply light randomization
        if self.args.randomize_lights and self.light_randomizers:
            for light_cfg in self.light_randomizers:
                light_rand = LightRandomizer(light_cfg, seed=seed)
                light_rand.bind_handler(self.handler)
                light_rand()
            log.info(f"  ✓ {len(self.light_randomizers)} lights randomized")

        log.info(f"✓ Variant {variant_id + 1} randomization complete")


class ObsSaver:
    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []
        self.image_idx = 0

    def add(self, state: TensorState):
        if self.image_dir is None and self.video_path is None:
            return
        try:
            rgb_data = next(iter(state.cameras.values())).rgb  # (N, H, W, 3) or (1, H, W, 3)
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(max(1, rgb_data.shape[0] ** 0.5)))
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)

    def clear(self):
        """Clear images for next variant."""
        self.images = []
        self.image_idx = 0


def replay_single_variant(env, scenario, all_states, variant_id: int, args) -> str:
    """Replay trajectory for a single variant.

    Args:
        env: Task environment
        scenario: Scenario configuration
        all_states: All trajectory states
        variant_id: Variant ID (0, 1, 2, ...)
        args: Configuration arguments

    Returns:
        Path to saved video
    """
    variant_name = f"variant_{variant_id + 1}"
    log.info("\n" + "=" * 70)
    log.info(f"REPLAYING VARIANT {variant_id + 1}/{args.num_variants}")
    log.info("=" * 70)

    # Setup output paths
    video_path = _suffix_path(args.save_video_path, variant_name)
    image_dir = _suffix_path(args.save_image_dir, variant_name) if args.save_image_dir else None

    saver = ObsSaver(image_dir=image_dir, video_path=video_path)
    log.info(f"Video will be saved to: {video_path}")

    # Use states from first episode
    episode_states = all_states[0]
    total = len(episode_states)
    num_envs = env.scenario.num_envs

    log.info(f"Replaying {total} states from episode 0")

    # Replay states
    for step in range(total):
        log.debug(f"[STATE] Step {step}/{total - 1}")

        # Get state dict for this step
        state_dict = episode_states[step]

        # Set the state in the handler
        env.handler.set_states([state_dict] * num_envs)

        env.handler.refresh_render()
        obs = env.handler.get_states()
        saver.add(obs)

        # Check success
        try:
            success = env.checker.check(env.handler)
            if success.any():
                log.info(f"[STATE] Env {success.nonzero().squeeze(-1).tolist()} succeeded at step {step}!")
            if success.all():
                break
        except Exception as e:
            log.debug(f"Checker error: {e}")
            pass

    # Save video
    saver.save()
    log.info(f"✓ Variant {variant_id + 1} completed")
    log.info(f"  Video saved: {video_path}")

    return video_path


def main():
    """Main entry point - replays multiple variants in one run without restart."""
    log.info("\n" + "=" * 70)
    log.info("STATE REPLAY WITH SCENE RANDOMIZATION")
    log.info("=" * 70)
    log.info(f"Task: {args.task}")
    log.info(f"Simulator: {args.sim}")
    log.info(f"Number of variants: {args.num_variants}")
    log.info("=" * 70)

    # ========================================
    # Step 1: Create environment (ONCE!)
    # ========================================
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 2.0), look_at=(0.0, 0.0, 0.0), width=2048, height=2048)

    # Setup lighting for enclosed scene
    lights = None
    if args.enable_scene and args.scene_type in ["empty_room", "tabletop_workspace"]:
        log.info("\nConfiguring lighting for enclosed scene...")
        log.info("  Base intensities (will be randomized):")
        log.info("    Main light: 25K (range: 18K-32K)")
        log.info("    Corner lights: 10K each (range: 7K-14K)")
        log.info("    Total base: ~65K (range: ~46K-88K)")

        lights = [
            DiskLightCfg(
                name="ceiling_main",
                intensity=25000.0,  # Base: 25K (will randomize to 18K-32K)
                color=(1.0, 1.0, 1.0),
                radius=1.2,
                pos=(0.0, 0.0, args.wall_height - 0.5),
                rot=(0.7071, 0.0, 0.0, 0.7071),  # 45° downward
            ),
            SphereLightCfg(
                name="ceiling_ne",
                intensity=10000.0,  # Base: 10K (will randomize to 7K-14K)
                color=(1.0, 1.0, 1.0),
                radius=0.6,
                pos=(2.0, 2.0, args.wall_height - 1.0),
            ),
            SphereLightCfg(
                name="ceiling_nw",
                intensity=10000.0,
                color=(1.0, 1.0, 1.0),
                radius=0.6,
                pos=(-2.0, 2.0, args.wall_height - 1.0),
            ),
            SphereLightCfg(
                name="ceiling_sw",
                intensity=10000.0,
                color=(1.0, 1.0, 1.0),
                radius=0.6,
                pos=(-2.0, -2.0, args.wall_height - 1.0),
            ),
            SphereLightCfg(
                name="ceiling_se",
                intensity=10000.0,
                color=(1.0, 1.0, 1.0),
                radius=0.6,
                pos=(2.0, -2.0, args.wall_height - 1.0),
            ),
        ]
        log.info("  ✓ Added 5 lights (1 DiskLight + 4 SphereLights)")

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        lights=lights,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    num_envs: int = scenario.num_envs

    device = torch.device("cpu")
    t0 = time.time()
    env = task_cls(scenario, device=device)
    log.trace(f"Time to launch: {time.time() - t0:.2f}s")

    # ========================================
    # Step 2: Initialize scene randomization manager
    # ========================================
    randomization_manager = SceneRandomizationManager(env.handler, scenario, args)

    # ========================================
    # Step 3: Load trajectory (ONCE!)
    # ========================================
    traj_filepath = "/home/priosin/murphy/ui/RoboVerse/teleop_trajs/put_banana_franka_20251029_220717_v2.pkl"
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    t0 = time.time()
    init_states, all_actions, all_states = get_traj(traj_filepath, scenario.robots[0], env.handler)
    log.trace(f"Time to load data: {time.time() - t0:.2f}s")

    # Check if states are available
    if all_states is None or len(all_states) == 0:
        log.error("No states found in trajectory file. Please ensure the trajectory was saved with --save-states")
        env.close()
        return

    log.info(f"Loaded {len(all_states)} episodes with states")
    log.info(f"Episode 0 has {len(all_states[0])} states")

    # Adjust object heights if using tabletop workspace
    if args.enable_scene and args.scene_type == "tabletop_workspace":
        log.info(f"\nAdjusting object heights for table at z={args.table_height}m")
        for episode in all_states:
            for state_dict in episode:
                # Adjust object positions
                if "objects" in state_dict:
                    for obj_name, obj_state in state_dict["objects"].items():
                        if "pos" in obj_state:
                            obj_state["pos"][2] += args.table_height

                # Adjust robot positions
                if "robots" in state_dict:
                    for robot_name, robot_state in state_dict["robots"].items():
                        if "pos" in robot_state:
                            robot_state["pos"][2] += args.table_height

        log.info(f"Adjusted heights for {len(all_states)} episodes")

    os.makedirs("test_output", exist_ok=True)

    # ========================================
    # Step 4: Replay multiple variants (NO RESTART!)
    # ========================================
    saved_videos = []

    for variant_id in range(args.num_variants):
        # Apply randomization for this variant
        randomization_manager.randomize_for_variant(variant_id)

        # Reset environment to initial state
        env.reset()

        # Replay this variant
        video_path = replay_single_variant(env, scenario, all_states, variant_id, args)
        saved_videos.append(video_path)

        # Small delay between variants
        if variant_id < args.num_variants - 1:
            log.info("\nPreparing next variant...")
            time.sleep(0.5)

    # ========================================
    # Step 5: Cleanup
    # ========================================
    env.close()

    # Summary
    log.info("\n" + "=" * 70)
    log.info("ALL VARIANTS COMPLETED")
    log.info("=" * 70)
    log.info(f"Total variants: {args.num_variants}")
    log.info("\nGenerated videos:")
    for video in saved_videos:
        log.info(f"  - {video}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
