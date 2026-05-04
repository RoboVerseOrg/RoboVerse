"""End-to-end Blender backend demo: replay a manipulation trajectory in photoreal Cycles.

What it shows:
  * ``HybridSimHandler`` running mujoco for physics + Blender for rendering.
  * Cycles quality knobs (``--samples``, ``--max-bounces``, ``--adaptive-threshold``).
  * Per-replay augmentation: HDRI swap and (optional) material library re-skinning.
    The camera is intentionally kept fixed so multiple replays are pixel-aligned.
  * One mp4 per replay written under ``--out-dir``.

Heavy assets (HDRIs, ``material_lib_v2.blend``) are looked up locally first, then
fetched on demand from the ``RoboVerseOrg/roboverse_data`` HF dataset under
``blender/`` — see ``metasim/sim/blender/utils/asset_paths.py``.

Examples:

    # Default: stack_cube, 64 samples, 1 replay, sample HDRIs, no material rand.
    python get_started/blender_demo.py

    # High quality, 3 augmented replays, swap look every 30 frames, full HDRI set.
    python get_started/blender_demo.py --samples 256 --replays 3 \\
        --aug-period 30 --randomize-materials --hdri-full

    # Different task and camera placement.
    python get_started/blender_demo.py --task close_box \\
        --cam-pos 1.4 -1.0 0.9 --cam-look-at 0.5 0.2 0.1
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import bpy
import rootutils
import torch
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.sim import HybridSimHandler
from metasim.sim.blender.utils.asset_paths import (
    get_hdri_dir,
    get_material_library,
)
from metasim.sim.blender.utils.material_aug import (
    assign_random_materials_to,
    index_material_library,
)
from metasim.sim.blender.utils.scene_aug import (
    add_ground_plane,
    add_sun_light,
    list_hdri_files,
    setup_world_color,
    setup_world_hdri,
)
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_recursive
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="stack_cube")
    p.add_argument("--robot", default="franka")
    p.add_argument(
        "--sim",
        default="mujoco",
        choices=["mujoco", "isaacsim", "isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3"],
    )
    p.add_argument("--out-dir", default="get_started/output/blender_demo", help="output directory; one mp4 per replay")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--cam-pos", type=float, nargs=3, default=(1.4, -1.0, 0.9))
    p.add_argument("--cam-look-at", type=float, nargs=3, default=(0.3, 0.0, 0.1))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--samples", type=int, default=64, help="Cycles AA samples — higher = cleaner renders, slower")
    p.add_argument(
        "--adaptive-threshold", type=float, default=0.01, help="adaptive sampling noise threshold (smaller = cleaner)"
    )
    p.add_argument("--max-bounces", type=int, default=8, help="max light bounces (raise for HDRI / glass / mirrors)")
    p.add_argument(
        "--ground-size", type=float, default=1.2, help="side length of the ground plane (m); 1.2m = a real-ish tabletop"
    )
    # Augmentation
    p.add_argument(
        "--replays", type=int, default=1, help="how many times to replay the trajectory (each gets its own mp4)"
    )
    p.add_argument(
        "--aug-period",
        type=int,
        default=0,
        help="swap HDRI/materials every N frames inside a replay (0 = swap once per replay)",
    )
    p.add_argument(
        "--randomize-materials",
        action="store_true",
        help="re-skin objects + ground from the material library on every aug step",
    )
    p.add_argument(
        "--material-categories",
        nargs="*",
        default=None,
        help="restrict the material library to these categories (e.g. wood metal rubber)",
    )
    p.add_argument(
        "--hdri-full", action="store_true", help="download the full 318 MB HDRI set instead of the small sample"
    )
    p.add_argument("--no-hdri", action="store_true")
    p.add_argument("--no-ground", action="store_true")
    p.add_argument("--no-download", action="store_true", help="do not fetch HDRI / material assets from HuggingFace")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene set-up + per-replay augmentation
# ---------------------------------------------------------------------------


def base_world(args) -> None:
    """One-time world bootstrap: ground plane + a default world."""
    if args.no_hdri:
        setup_world_color((0.18, 0.20, 0.22))
        add_sun_light()
    if not args.no_ground:
        add_ground_plane(size=args.ground_size, z=0.0, color=(0.55, 0.55, 0.58))


def apply_aug(args, hdri_files, material_lib, scenario, rng):
    """Apply one stochastic augmentation step (HDRI + optional materials).

    Camera is intentionally untouched — replays must be pixel-aligned.
    """
    hdri_used = None
    if not args.no_hdri and hdri_files:
        hdri_path = rng.choice(hdri_files)
        setup_world_hdri(hdri_path, rotation_z=rng.uniform(-3.14159, 3.14159), strength=rng.uniform(0.7, 1.3))
        hdri_used = os.path.basename(hdri_path)

    if args.randomize_materials and material_lib is not None:
        target_objs = []
        for name in [o.name for o in scenario.objects]:
            o = bpy.data.objects.get(name)
            if o is not None and o.type == "MESH":
                target_objs.append(o)
        ground = bpy.data.objects.get("RV_Ground")
        if ground is not None:
            target_objs.append(ground)
        if target_objs:
            assign_random_materials_to(material_lib, target_objs, categories=args.material_categories, rng=rng)
    return hdri_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"task={args.task}  sim={args.sim}  renderer=blender  out={out_dir}")
    log.info(f"replays={args.replays}  aug-period={args.aug_period}  samples={args.samples}")

    # Scenario + handlers
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(
        name="cam",
        pos=tuple(args.cam_pos),
        look_at=tuple(args.cam_look_at),
        width=args.width,
        height=args.height,
        data_types=["rgb"],
    )
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        cameras=[camera],
        simulator=args.sim,
        num_envs=1,
        headless=True,
    )

    log.info(f"loading physics handler ({args.sim})…")
    physics_handler = get_handler(scenario)
    scenario.update(simulator="blender")
    log.info("loading blender render handler…")
    render_handler = get_handler(scenario)
    render_handler.set_quality(
        args.samples,
        adaptive_threshold=args.adaptive_threshold,
        max_bounces=args.max_bounces,
    )
    handler = HybridSimHandler(scenario, physics_handler, render_handler)

    base_world(args)

    # Augmentation resources — local first, then HF (unless --no-download).
    rng = random.Random(args.seed)
    hdri_files: list[str] = []
    if not args.no_hdri:
        hdri_dir = get_hdri_dir(download=not args.no_download, sample_only=not args.hdri_full)
        if hdri_dir is not None:
            hdri_files = list_hdri_files(hdri_dir)
        log.info(f"HDRIs available: {len(hdri_files)} (dir={hdri_dir})")

    material_lib = None
    if args.randomize_materials:
        mat_path = get_material_library(download=not args.no_download)
        if mat_path is not None:
            material_lib = index_material_library(str(mat_path))
            log.info(
                f"material library: {len(material_lib.all_names)} materials, "
                f"categories={list(material_lib.by_category.keys())}"
            )
        else:
            log.warning("material library unavailable; skipping material rand")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(handler, device=device)

    if not os.path.exists(env.traj_filepath):
        log.info(f"downloading trajectory: {env.traj_filepath}")
        check_and_download_recursive([env.traj_filepath], n_processes=4)
    init_states, all_actions, _ = get_traj(env.traj_filepath, scenario.robots[0], handler)
    max_steps = max(len(a) for a in all_actions)
    if args.max_steps:
        max_steps = min(args.max_steps, max_steps)
    log.info(f"replaying {max_steps} steps x {args.replays} replays")

    for replay_idx in range(args.replays):
        out_path = out_dir / f"replay_{replay_idx:02d}.mp4"
        log.info(f"== replay {replay_idx + 1}/{args.replays} -> {out_path} ==")

        handler.set_states(init_states[:1])
        hdri_used = apply_aug(args, hdri_files, material_lib, scenario, rng)
        log.info(f"   initial aug: hdri={hdri_used}")

        obs = handler.get_states(mode="tensor")
        saver = ObsSaver(video_path=str(out_path))
        saver.add(obs)

        for step in range(max_steps):
            if args.aug_period and step > 0 and step % args.aug_period == 0:
                hdri_used = apply_aug(args, hdri_files, material_lib, scenario, rng)
                log.info(f"   step {step}: aug change → hdri={hdri_used}")

            actions = [all_actions[0][min(step, len(all_actions[0]) - 1)]]
            handler.set_dof_targets(actions)
            handler.simulate()
            obs = handler.get_states(mode="tensor")
            saver.add(obs)
            if step % 50 == 0:
                log.info(f"   step {step}/{max_steps}")

        saver.save(fps=args.fps)
        log.info(f"   wrote {len(saver.images)} frames → {out_path}")

    handler.close()
    log.info(f"done — {args.replays} replays in {out_dir}/")


if __name__ == "__main__":
    main()
