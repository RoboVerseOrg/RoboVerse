from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
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
from roboverse_pack.tasks.embodiedgen.tables import table785_config
from torchvision.utils import make_grid, save_image

from metasim.randomization import (
    LightColorRandomCfg,
    LightIntensityRandomCfg,
    LightOrientationRandomCfg,
    LightPositionRandomCfg,
    LightRandomCfg,
    LightRandomizer,
)
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

DEFAULT_TABLE_HEIGHT = 0.900
HIDDEN_TABLE_POSE = {"pos": [100.0, 100.0, -25.0], "rot": [1.0, 0.0, 0.0, 0.0]}


@configclass
class Args:
    """Replay trajectory for a given task."""

    task: str = "close_box"
    robot: str = "franka"
    render: RenderCfg = RenderCfg(mode="raytracing")

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

    # Scene selection options
    scene_names: tuple[str, ...] = (
        "kujiale_scene_0003",
        "kujiale_scene_0004",
        "kujiale_scene_0008",
        "kujiale_scene_0009",
        "kujiale_scene_0020",
        "kujiale_scene_0021",
        "kujiale_scene_0022",
        "kujiale_scene_0024",
        "kujiale_scene_0025",
        "kujiale_scene_0031",
        "kujiale_scene_0032",
        "kujiale_scene_0033",
    )
    """Ordered list of available scenes; index via `scene_index`"""
    scene_index: int = 3
    """Index into scene_names specifying which scene to load"""

    # Table selection options
    table_index: int = 5
    """Index into table_configs specifying which table to use"""

    # Replay / randomization options
    randomize_lights: bool = True
    """Enable light randomization (intensity, color, position)"""
    num_light_variants: int = 1
    """Number of lighting randomizations to replay"""
    base_seed: int = 1
    """Base random seed for lighting randomization"""
    wall_height: float = 2.0
    """Nominal ceiling height used when spawning helper lights"""
    table_height: float = DEFAULT_TABLE_HEIGHT
    """Height offset applied to replayed table/object poses"""

    def __post_init__(self):
        if len(self.scene_names) == 0:
            raise ValueError("scene_names must contain at least one scene identifier.")
        if self.scene_index < 0:
            self.scene_index %= len(self.scene_names)
        elif self.scene_index >= len(self.scene_names):
            log.warning(
                f"scene_index {self.scene_index} out of range for {len(self.scene_names)} scenes; using modulo."
            )
            self.scene_index %= len(self.scene_names)

        if len(table785_config.ALL_TABLE750_CONFIGS) == 0:
            raise ValueError("No table configurations available.")
        if self.table_index < 0:
            self.table_index %= len(table785_config.ALL_TABLE750_CONFIGS)
        elif self.table_index >= len(table785_config.ALL_TABLE750_CONFIGS):
            log.warning(f"table_index {self.table_index} out of range; using modulo of available tables.")
            self.table_index %= len(table785_config.ALL_TABLE750_CONFIGS)

        if self.num_light_variants < 1:
            log.warning("num_light_variants < 1; defaulting to 1.")
            self.num_light_variants = 1

        log.info(f"Args: {self}")


args = tyro.cli(Args)


def _suffix_path(p: str | None, suffix: str) -> str | None:
    if p is None:
        return None
    base, ext = os.path.splitext(p)
    if ext:
        return f"{base}_{suffix}{ext}"
    return f"{p}_{suffix}"


def _to_tensor(value, fallback):
    if value is None:
        value = fallback
    if isinstance(value, torch.Tensor):
        tensor = value.clone().detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).clone().detach()
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


def _maybe_apply_height_offset(tensor: torch.Tensor, offset: float | None) -> torch.Tensor:
    if offset is None:
        return tensor
    if tensor.numel() < 3:
        raise ValueError("Expected a 3D position vector.")
    adjusted = tensor.clone()
    adjusted[2] += offset
    return adjusted


def _preprocess_states(all_states, height_offset: float | None):
    processed_states = []
    table_reference_states = []

    for episode in all_states:
        processed_episode = []
        table_refs_episode = []
        for state in episode:
            state_copy = deepcopy(state)
            objects = state_copy.setdefault("objects", {})

            table_state = objects.pop("table", None) or {}
            table_pos = _to_tensor(table_state.get("pos"), fallback=[0.0, 0.0, 0.0])
            table_rot = _to_tensor(table_state.get("rot"), fallback=[1.0, 0.0, 0.0, 0.0])
            table_pos = _maybe_apply_height_offset(table_pos, height_offset)
            table_refs_episode.append({"pos": table_pos.clone(), "rot": table_rot.clone()})

            for obj_state in objects.values():
                if "pos" in obj_state:
                    obj_state["pos"] = _maybe_apply_height_offset(
                        _to_tensor(obj_state["pos"], fallback=[0.0, 0.0, 0.0]), height_offset
                    )
                if "rot" in obj_state:
                    obj_state["rot"] = _to_tensor(obj_state["rot"], fallback=[1.0, 0.0, 0.0, 0.0])

            for robot_state in state_copy.get("robots", {}).values():
                if "pos" in robot_state:
                    robot_state["pos"] = _maybe_apply_height_offset(
                        _to_tensor(robot_state["pos"], fallback=[0.0, 0.0, 0.0]), height_offset
                    )
                if "rot" in robot_state:
                    robot_state["rot"] = _to_tensor(robot_state["rot"], fallback=[1.0, 0.0, 0.0, 0.0])

            processed_episode.append(state_copy)

        processed_states.append(processed_episode)
        table_reference_states.append(table_refs_episode)

    return processed_states, table_reference_states


def _build_variant_states(
    base_episode_states,
    table_reference_states,
    active_table_cfg,
    hidden_pose: dict[str, list[float]] | dict[str, torch.Tensor],
):
    variant_states = []
    hidden_pos = _to_tensor(hidden_pose["pos"], fallback=[0.0, 0.0, 0.0])
    hidden_rot = _to_tensor(hidden_pose["rot"], fallback=[1.0, 0.0, 0.0, 0.0])

    for step_idx, base_state in enumerate(base_episode_states):
        state = deepcopy(base_state)
        objects = state.setdefault("objects", {})

        ref_pose = table_reference_states[step_idx] if step_idx < len(table_reference_states) else {}
        ref_pos = ref_pose.get("pos")
        ref_rot = ref_pose.get("rot")
        pos_tensor = ref_pos.clone() if isinstance(ref_pos, torch.Tensor) else hidden_pos.clone()
        rot_tensor = ref_rot.clone() if isinstance(ref_rot, torch.Tensor) else hidden_rot.clone()
        objects[active_table_cfg.name] = {"pos": pos_tensor, "rot": rot_tensor}

        variant_states.append(state)

    return variant_states


class LightRandomizationManager:
    """Manages only lighting randomization for replay."""

    def __init__(self, handler, scenario, args):
        self.handler = handler
        self.scenario = scenario
        self.args = args
        self.light_randomizers: list[LightRandomCfg] = []

        if not args.randomize_lights:
            log.info("Light randomization disabled")
            return

        lights = getattr(self.scenario, "lights", [])
        if not lights:
            log.info("  No lights configured; skipping light randomization")
            return

        log.info("\n[Light Randomization Setup]")
        log.info(f"  Found {len(lights)} lights to randomize")

        ceiling_main_range = (18000.0, 32000.0)  # Main light: 18K-32K
        ceiling_corner_range = (7000.0, 14000.0)  # Corner lights: 7K-14K each

        for light in lights:
            light_name = getattr(light, "name", f"light_{len(self.light_randomizers)}")

            if light_name == "ceiling_main":
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
            else:
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

            self.light_randomizers.append(light_cfg)
            log.info(f"    Added config for {light_name}")

    def randomize_lights(self, seed: int):
        if not self.light_randomizers:
            return

        log.info(f"\nRandomizing lights (seed={seed})")
        for idx, light_cfg in enumerate(self.light_randomizers):
            light_rand = LightRandomizer(light_cfg, seed=seed + idx)
            light_rand.bind_handler(self.handler)
            light_rand()
        log.info(f"  ✓ {len(self.light_randomizers)} lights randomized")


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
            rgb_data = torch.concat([cam.rgb for cam in state.cameras.values()], dim=2)  # concatenate horizontally
            image = make_grid(
                rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(max(1, rgb_data.shape[0] ** 0.5))
            )  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
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


def replay_single_variant(
    env,
    scenario,
    episode_states,
    variant_idx: int,
    total_variants: int,
    args,
    table_name: str | None = None,
    variant_label: str | None = None,
) -> str:
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
    variant_name = variant_label or f"variant_{variant_idx + 1}"
    log.info("\n" + "=" * 70)
    log.info(f"REPLAYING VARIANT {variant_idx + 1}/{total_variants}")
    log.info("=" * 70)
    log.info(f"Variant label: {variant_name}")

    if table_name is not None:
        log.info(f"Table asset: {table_name}")

    # Setup output paths
    video_path = _suffix_path(args.save_video_path, variant_name)
    image_dir = _suffix_path(args.save_image_dir, variant_name) if args.save_image_dir else None

    saver = ObsSaver(image_dir=image_dir, video_path=video_path)
    log.info(f"Video will be saved to: {video_path}")

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
    log.info(f"✓ Variant {variant_idx + 1} completed")
    log.info(f"  Video saved: {video_path}")

    return video_path


def main():
    """Main entry point - replays multiple variants in one run without restart."""
    log.info("\n" + "=" * 70)
    log.info("STATE REPLAY WITH SCENE RANDOMIZATION")
    log.info("=" * 70)
    log.info(f"Task: {args.task}")
    log.info(f"Simulator: {args.sim}")
    log.info(f"Lighting variants: {args.num_light_variants}")
    log.info("=" * 70)

    # ========================================
    # Step 1: Create environment (ONCE!)
    # ========================================
    task_cls = get_task_class(args.task)
    camera1 = PinholeCameraCfg(name="camera1", pos=(1.5, -1.5, 1.8), look_at=(0.0, 0.0, 1.0), width=840, height=840)
    camera2 = PinholeCameraCfg(name="camera2", pos=(0.5, -2, 1.8), look_at=(0.0, 0.0, 1.5), width=840, height=840)
    scene_names = list(args.scene_names)
    selected_scene = scene_names[args.scene_index % len(scene_names)]
    log.info(f"Selected scene: {selected_scene}")

    # Setup lighting for enclosed scene
    lights = None
    if args.randomize_lights:
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

    scenario_cfg = deepcopy(task_cls.scenario)
    existing_objects = [deepcopy(obj) for obj in scenario_cfg.objects if obj.name != "table"]
    table_pool = table785_config.ALL_TABLE750_CONFIGS
    active_table_cfg = deepcopy(table_pool[args.table_index % len(table_pool)])
    log.info(f"Selected table asset: {active_table_cfg.name}")
    scenario_cfg.update(
        robots=[args.robot],
        scene=selected_scene,
        cameras=[camera1, camera2],
        lights=lights,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
        objects=existing_objects + [active_table_cfg],
    )
    scenario = scenario_cfg
    num_envs: int = scenario.num_envs

    device = torch.device("cpu")
    t0 = time.time()
    env = task_cls(scenario, device=device)
    log.trace(f"Time to launch: {time.time() - t0:.2f}s")

    # ========================================
    # Step 2: Initialize light randomization manager
    # ========================================
    light_manager = LightRandomizationManager(env.handler, scenario, args)

    traj_filepath = env.traj_filepath
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    t0 = time.time()
    init_states, all_actions, all_states = get_traj(traj_filepath, scenario.robots[0], env.handler)
    log.trace(f"Time to load data: {time.time() - t0:.2f}s")

    # Check if states are available
    if all_states is None or len(all_states) == 0:
        log.error("No states found in trajectory file. Please ensure the trajectory was saved with --save-states")
        env.close()
        return

    all_states, table_reference_states = _preprocess_states(all_states, height_offset=args.table_height)

    log.info(f"Loaded {len(all_states)} episodes with states")
    log.info(f"Episode 0 has {len(all_states[0])} states")

    os.makedirs("test_output", exist_ok=True)

    # ========================================
    # Step 4: Replay with lighting variations
    # ========================================
    saved_videos = []
    base_seed = args.base_seed

    variant_episode_states = _build_variant_states(
        base_episode_states=all_states[0],
        table_reference_states=table_reference_states[0],
        active_table_cfg=active_table_cfg,
        hidden_pose=HIDDEN_TABLE_POSE,
    )

    total_variants = args.num_light_variants if args.randomize_lights else 1

    for light_idx in range(total_variants):
        current_seed = base_seed + light_idx
        if args.randomize_lights:
            light_manager.randomize_lights(current_seed)

        # Reset environment to initial state
        env.reset()

        # Replay this variant
        video_path = replay_single_variant(
            env,
            scenario,
            variant_episode_states,
            light_idx,
            total_variants,
            args,
            table_name=active_table_cfg.name,
            variant_label=f"{active_table_cfg.name}_light_{light_idx + 1}",
        )
        saved_videos.append(video_path)

        # Small delay between variants
        if light_idx < total_variants - 1:
            log.info("\nPreparing next lighting variant...")
            time.sleep(0.5)

    # ========================================
    # Step 5: Cleanup
    # ========================================
    env.close()

    # Summary
    log.info("\n" + "=" * 70)
    log.info("ALL VARIANTS COMPLETED")
    log.info("=" * 70)
    log.info(f"Scene preset: {selected_scene}")
    log.info(f"Table asset: {active_table_cfg.name}")
    log.info(f"Total lighting variants: {total_variants}")
    log.info("\nGenerated videos:")
    for video in saved_videos:
        log.info(f"  - {video}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
