"""Domain Randomization Demo with Trajectory Replay

Replays close_box task trajectories with progressive domain randomization.

Scene Setup:
- Enclosed room: 10m x 10m x 5m with walls and ceiling
- Table: 1.8m x 1.8m at height 0.7m with physics collision
- Objects placed on table surface

Lighting (5 lights, all inside room):
- 1 central DiskLight (main directional light, supports orientation randomization)
- 4 corner SphereLight (even ambient coverage)
- Intensities auto-adjusted for render mode:
  * PathTracing: 28K + 12Kx4 = 76K total
  * RayTracing: 20K + 7Kx4 = 48K total

Randomization Levels:
- Level 0: Baseline (no randomization)
- Level 1: Material randomization
- Level 2: Material + Lighting randomization
- Level 3: Material + Lighting + Camera randomization

All randomizations applied simultaneously every N steps (default: 10)
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.randomization import (
    CameraPresets,
    CameraRandomizer,
    LightRandomizer,
    MaterialPresets,
    MaterialRandomizer,
    ObjectPresets,
    ObjectRandomizer,
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

    table_height = 0.7

    camera = PinholeCameraCfg(
        name="main_camera",
        width=1024,
        height=1024,
        pos=(2.0, -2.0, 2.0),
        look_at=(0.0, 0.0, table_height + 0.05),
    )

    # Lighting configuration for enclosed room (10m x 10m x 5m)
    # All lights positioned inside room to avoid wall blocking
    # Layout: 1 central DiskLight + 4 corner SphereLight
    if args.render_mode == "pathtracing":
        ceiling_main = 28000.0
        ceiling_corners = 12000.0
    else:
        ceiling_main = 20000.0
        ceiling_corners = 7000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 4.5),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(3.0, 3.0, 4.0),
        ),
        SphereLightCfg(
            name="ceiling_nw",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-3.0, 3.0, 4.0),
        ),
        SphereLightCfg(
            name="ceiling_sw",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(-3.0, -3.0, 4.0),
        ),
        SphereLightCfg(
            name="ceiling_se",
            intensity=ceiling_corners,
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(3.0, -3.0, 4.0),
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
    """Initialize all randomizers based on randomization level."""
    randomizers = {
        "object": [],
        "material": [],
        "light": [],
        "camera": [],
        "scene": None,
    }

    level = args.level
    log.info("=" * 70)
    log.info(f"Randomization Level: {level}")
    log.info("=" * 70)

    log.info("\n[Scene Setup]")
    log.info("-" * 70)

    scene_cfg = ScenePresets.tabletop_workspace(
        room_size=10.0,
        wall_height=5.0,
        table_size=(1.8, 1.8, 0.1),
        table_height=0.7,
    )

    if level < 1:
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
        log.info("  Scene with fixed materials")
    else:
        log.info("  Scene with randomized materials")

    scene_rand = SceneRandomizer(scene_cfg, seed=args.seed)
    scene_rand.bind_handler(handler)
    randomizers["scene"] = scene_rand

    log.info("    Room: 10m x 10m x 5m")
    log.info("    Table: 1.8m x 1.8m at z=0.7m with collider")

    box_rand = ObjectRandomizer(
        ObjectPresets.heavy_object("box_base"),
        seed=args.seed,
    )
    box_rand.cfg.pose.rotation_range = (0, 0)
    box_rand.cfg.pose.position_range[2] = (0, 0)
    box_rand.bind_handler(handler)
    box_rand()

    log.info("\n[Level 1+] Material Randomization")
    log.info("-" * 70)

    box_mat = MaterialRandomizer(
        MaterialPresets.wood_object("box_base", use_mdl=True),
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
                position_range=((-1.5, 1.5), (-1.5, 1.5), (-0.5, 0.5)),
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


def apply_randomization(randomizers, level):
    """Apply all active randomizers."""
    if randomizers["scene"]:
        randomizers["scene"]()

    if level >= 0:
        for rand in randomizers["object"]:
            rand()

    if level >= 1:
        for rand in randomizers["material"]:
            rand()

    if level >= 2:
        for rand in randomizers["light"]:
            rand()

    if level >= 3:
        for rand in randomizers["camera"]:
            rand()


def get_states(all_states, action_idx: int, num_envs: int):
    """Get states for all environments at a given step."""
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def run_replay_with_randomization(env, randomizers, init_state, all_actions, all_states, args):
    """Replay trajectory with periodic randomization."""
    os.makedirs("get_started/output", exist_ok=True)
    video_path = f"get_started/output/12_dr_level{args.level}_{args.sim}.mp4"
    obs_saver = ObsSaver(video_path=video_path)

    log.info("\n" + "=" * 70)
    log.info("Trajectory Replay with Domain Randomization")
    log.info("=" * 70)
    log.info(f"Video output: {video_path}")
    log.info(f"Randomization interval: every {args.randomize_interval} steps")

    traj_length = len(all_actions[0]) if all_actions else (len(all_states[0]) if all_states else 0)
    log.info(f"Trajectory length: {traj_length} steps")

    apply_randomization(randomizers, args.level)
    obs, extras = env.reset(states=[init_state] * args.num_envs)

    step = 0
    num_envs = env.scenario.num_envs

    while True:
        if step % args.randomize_interval == 0 and step > 0:
            log.info(f"Step {step}: Applying randomizations")
            apply_randomization(randomizers, args.level)

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

        level: Literal[0, 1, 2, 3] = 1
        """Randomization level:
        0 - Baseline (no DR)
        1 - Material randomization
        2 - Material + Light randomization
        3 - Material + Light + Camera randomization
        """

        randomize_interval: int = 10

        object_states: bool = False
        """If True, replay using object states (deterministic)."""

        render_mode: Literal["raytracing", "pathtracing"] = "raytracing"
        """Rendering mode:
        - raytracing: Fast with shadows
        - pathtracing: Highest quality (slower)
        """

    args = tyro.cli(Args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    log.info("=" * 70)
    log.info("Domain Randomization Demo with Trajectory Replay")
    log.info("=" * 70)
    log.info("\nConfiguration:")
    log.info(f"  Simulator: {args.sim}")
    log.info(f"  Render mode: {args.render_mode}")
    log.info(f"  Robot: {args.robot}")
    log.info(f"  Seed: {args.seed}")
    log.info(f"  Randomization level: {args.level}")

    log.info("\nScene:")
    log.info("  Room: 10m x 10m x 5m (enclosed)")
    log.info("  Table: 1.8m x 1.8m at z=0.7m")

    log.info(f"\nLighting ({args.render_mode}):")
    if args.render_mode == "pathtracing":
        log.info("  DiskLight (main): 28K, 4x SphereLight (corners): 12K each")
        log.info("  Total: ~76K")
    else:
        log.info("  DiskLight (main): 20K, 4x SphereLight (corners): 7K each")
        log.info("  Total: ~48K")
    log.info("  All lights inside room")

    env = create_env(args)
    handler = env.handler

    traj_filepath = env.traj_filepath
    log.info(f"\nLoading trajectory: {traj_filepath}")
    assert os.path.exists(traj_filepath), f"Trajectory file not found: {traj_filepath}"

    init_states, all_actions, all_states = get_traj(traj_filepath, env.scenario.robots[0], handler)
    init_state = init_states[0]

    for obj_name, obj_state in init_state["objects"].items():
        obj_state["pos"][2] += 0.7

    for robot_name, robot_state in init_state["robots"].items():
        robot_state["pos"][2] += 0.7

    env.handler.set_states(init_states, env_ids=list(range(args.num_envs)))
    log.info(f"Loaded {len(all_actions[0]) if all_actions else 0} actions")

    randomizers = initialize_randomizers(handler, args)

    if args.object_states:
        log.info("\nWARNING: State-based replay mode (no randomization applied)")

    run_replay_with_randomization(env, randomizers, init_state, all_actions, all_states, args)

    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()

    log.info("\nDemo completed")


if __name__ == "__main__":
    main()
