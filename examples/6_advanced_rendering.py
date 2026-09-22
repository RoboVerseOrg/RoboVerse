"""Reproducible appearance variants, from primitive scenes to asset-based rendering.

Recipes can be generated without a simulator. Render with Blender or Isaac Sim,
optionally driving the same scene with MuJoCo physics. See Tutorial 6 for commands.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np
import tyro
from loguru import logger as log

from metasim.constants import PhysicStateType, SimType
from metasim.randomization import (
    EnvironmentRandomCfg,
    LightingRandomCfg,
    SensorRandomCfg,
    SurfaceRandomCfg,
    TextureSetCfg,
    ViewRandomCfg,
    VisualRandomizationCfg,
    VisualRandomizer,
    VisualRecipe,
)
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import SphereLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.scenario import ScenarioCfg


@dataclass
class Args:
    """Render variants in independently reproducible, atomically published folders."""

    sim: Literal["blender", "isaacsim"] = "blender"
    physics: Literal["none", "mujoco"] = "none"
    scene: Literal["primitives", "assets"] = "primitives"
    appearance: Literal["studio", "stress"] = "studio"
    robot: str | None = None
    render: RenderCfg = field(default_factory=lambda: RenderCfg(mode="pathtracing", samples=64, view_transform="AgX"))
    num_envs: int = 1
    headless: bool = True
    variants: int = 4
    sample_id: int = 0
    seed: int = 0
    frames: int | None = None
    """Captured frames per variant: 1 by default, or every recorded step of a --task demo."""
    steps_per_frame: int = 4
    settle_steps: int = 0
    """Control intervals simulated once after launch so every variant starts from a settled scene."""
    fps: float | None = None
    """Playback FPS; by default matches the physics time between captured frames."""
    width: int = 512
    height: int = 512
    multiview: bool = False
    shard_index: int = 0
    num_shards: int = 1
    hdri: tuple[str, ...] = ()
    hdri_library: Path | None = None
    """Directory searched recursively for ``.hdr`` / ``.exr`` environment maps, added to --hdri."""
    background: str | None = None
    """Scene config name (e.g. ``kujiale_scene_0003``) rendered around the objects instead of the primitive floor/backdrop."""
    task: str | None = None
    """Registered task (e.g. ``libero.pick_butter``): its scenario replaces the primitive scene and its recorded demo drives the robot."""
    demo: int = 0
    """Index of the demonstration in the task's trajectory file."""
    camera_pos: tuple[float, float, float] = (1.7, -2.0, 1.4)
    look_at: tuple[float, float, float] = (0.1, 0.0, 0.3)
    textures: TextureSetCfg = field(default_factory=TextureSetCfg)
    """One texture set for the BBQ-sauce asset's authored UVs (requires --scene assets)."""
    material_library: Path | None = None
    """Directory of ``<name>_BaseColor.png`` sets (optional ``_Normal``/``_ORM``) for floor and backdrop."""
    depth_of_field: bool = True
    sensor: bool = True
    """Apply the image-space lens/sensor stage; raw captures are kept for the first frame."""
    recipe: Path | None = None
    verify_assets: bool = True
    recipes_only: bool = False
    output: Path = Path("examples/output/visual_variants")

    def __post_init__(self):
        for name in ("variants", "frames", "steps_per_frame", "width", "height", "num_shards", "num_envs"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("fps must be finite and positive")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be in [0, num_shards)")
        if self.sim == "blender" and self.num_envs != 1:
            raise ValueError("Blender supports one environment per process")
        if self.settle_steps < 0 or self.demo < 0:
            raise ValueError("settle_steps and demo must be nonnegative")
        if ((self.frames or 1) > 1 or self.settle_steps) and self.physics == "none":
            raise ValueError("Video frames and settling require --physics mujoco")
        if self.task is not None and (
            self.physics != "mujoco" or self.settle_steps or self.scene == "assets" or self.robot
        ):
            raise ValueError(
                "--task replays a recorded demo through --physics mujoco and defines its own scene and robot"
            )
        if self.recipe is not None and self.num_shards != 1:
            raise ValueError("Replay a single recipe without sharding")
        for name in ("material_library", "hdri_library"):
            path = getattr(self, name)
            if path is not None and not path.is_dir():
                raise FileNotFoundError(f"{name} is not a directory: {path}")
        if self.hdri_library is not None:
            found = sorted(str(p) for p in self.hdri_library.rglob("*") if p.suffix.lower() in {".hdr", ".exr"})
            if not found:
                raise FileNotFoundError(f"No .hdr/.exr files under {self.hdri_library}")
            self.hdri = tuple(dict.fromkeys([*self.hdri, *found]))


def _scenario(args):
    objects = [
        PrimitiveCubeCfg(
            name="floor",
            color=[0.3, 0.3, 0.3],
            size=(6.0, 6.0, 0.1),
            default_position=(0.0, 0.0, -0.05),
            fix_base_link=True,
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="cube",
            color=[0.7, 0.2, 0.1],
            size=(0.16, 0.16, 0.16),
            default_position=(0.0, 0.0, 0.45),
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            color=[0.1, 0.3, 0.7],
            radius=0.1,
            default_position=(0.3, 0.0, 0.65),
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveCubeCfg(
            name="backdrop",
            color=[0.5, 0.5, 0.5],
            size=(2.8, 0.08, 1.6),
            default_position=(0.0, 0.8, 0.8),
            fix_base_link=True,
            physics=PhysicStateType.RIGIDBODY,
        ),
    ]
    if args.robot:
        # Leave the common Franka workspace clear at the initial frame.
        objects[1].default_position = (0.55, -0.35, 0.45)
        objects[2].default_position = (0.75, 0.3, 0.65)
    if args.background:
        # An interior scene supplies its own floor and walls; physics keeps a flat ground at z=0.
        objects = [obj for obj in objects if obj.name not in {"floor", "backdrop"}]
    if args.scene == "assets":
        base = "roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce"
        objects.append(
            RigidObjCfg(
                name="bbq_sauce",
                scale=(2, 2, 2),
                default_position=(0.5, 0.1, 0.2),
                physics=PhysicStateType.RIGIDBODY,
                usd_path=f"{base}/usd/bbq_sauce.usd",
                urdf_path=f"{base}/urdf/bbq_sauce.urdf",
                mjcf_path=f"{base}/mjcf/bbq_sauce.xml",
            )
        )
    look_at = tuple(args.look_at)
    cameras = [
        PinholeCameraCfg(
            name="front",
            width=args.width,
            height=args.height,
            pos=tuple(args.camera_pos),
            look_at=look_at,
            data_types=["rgb"],
        )
    ]
    if args.multiview:
        # The side view mirrors the front view's offset across the look-at point's X axis.
        dx, dy, dz = (c - t for c, t in zip(args.camera_pos, look_at, strict=True))
        cameras.append(
            PinholeCameraCfg(
                name="side",
                width=args.width,
                height=args.height,
                pos=(look_at[0] - dx, look_at[1] + dy * 0.65, look_at[2] + dz * 0.85),
                look_at=look_at,
                data_types=["rgb"],
            )
        )
    key, fill = (700, 180) if args.sim == "blender" else (48000, 14400)
    rig = 0.3 if args.background else 1.0
    shared = {
        "cameras": cameras,
        # Renderer light units differ; recipes multiply this backend-tuned baseline.
        # Interior scenes bring their own emitters and windows, so the studio rig steps back.
        "lights": [
            SphereLightCfg(name="key", pos=(0.5, -1.5, 2.5), intensity=key * rig, radius=0.25),
            SphereLightCfg(name="fill", pos=(-1.5, -0.5, 1.7), intensity=fill * rig, radius=0.6),
        ],
        "simulator": args.sim,
        "render": args.render,
        "num_envs": args.num_envs,
        "env_spacing": 8.0,
        "headless": args.headless,
    }
    if args.task:
        from metasim.task.registry import get_task_class

        base = get_task_class(args.task).scenario
        # replace() deep-copies: update() would mutate the task class's shared scenario.
        # A task scene keeps its own ground unless an interior background supplies the floor.
        return base.replace(
            scene=args.background or base.scene,
            add_default_ground=not args.background and base.add_default_ground,
            **shared,
        )
    return ScenarioCfg(
        robots=[args.robot] if args.robot else [],
        objects=objects,
        scene=args.background,
        add_default_ground=False,
        **shared,
    )


def _texture_library(directory):
    """Coherent sets named ``<name>_BaseColor.png`` with sibling ``_Normal`` / ``_ORM`` maps."""
    sets = []
    for base_color in sorted(directory.rglob("*_BaseColor.png")):
        stem = base_color.name[: -len("_BaseColor.png")]
        normal = base_color.with_name(f"{stem}_Normal.png")
        orm = base_color.with_name(f"{stem}_ORM.png")
        sets.append(
            TextureSetCfg(
                base_color=str(base_color),
                normal=str(normal) if normal.is_file() else None,
                orm=str(orm) if orm.is_file() else None,
            )
        )
    if not sets:
        raise FileNotFoundError(f"No *_BaseColor.png texture sets under {directory}")
    return tuple(sets)


def _sampler(args):
    # The BBQ-sauce asset carries authored UVs; primitives use metric box projection.
    asset_textures = (args.textures,) if any(asdict(args.textures).values()) else ()
    if asset_textures and args.scene != "assets":
        raise ValueError("--textures targets the BBQ-sauce asset; add --scene assets")
    studio = args.appearance == "studio"
    materials = {
        "floor": SurfaceRandomCfg(color=((0.15, 0.25),) * 3, roughness=(0.55, 0.8)),
        "cube": SurfaceRandomCfg(),
        "sphere": SurfaceRandomCfg(metallic=(0.5, 1.0)),
        "backdrop": SurfaceRandomCfg(),
    }
    if studio:
        # Linear reflectances: restrained coated surfaces and unpainted metal.
        # Broad independent RGB sampling remains available as a stress preset.
        materials.update({
            "floor": SurfaceRandomCfg(
                color_palette=((0.18, 0.18, 0.18), (0.28, 0.26, 0.23), (0.12, 0.14, 0.16)), roughness=(0.6, 0.85)
            ),
            "cube": SurfaceRandomCfg(
                color_palette=((0.35, 0.08, 0.04), (0.04, 0.13, 0.24), (0.12, 0.2, 0.09)),
                roughness=(0.3, 0.55),
                ior=(1.4, 1.6),
            ),
            "sphere": SurfaceRandomCfg(
                color_palette=((0.65, 0.65, 0.65), (0.55, 0.4, 0.22)), metallic=(1, 1), roughness=(0.2, 0.4)
            ),
            "backdrop": SurfaceRandomCfg(
                color_palette=((0.45, 0.45, 0.45), (0.35, 0.32, 0.28), (0.25, 0.28, 0.3)), roughness=(0.75, 0.95)
            ),
        })
    if args.task:
        # Task assets keep their authored materials; only lighting, camera and sensor vary.
        materials = {}
    elif args.background:
        for name in ("floor", "backdrop"):
            del materials[name]
    elif args.material_library is not None:
        library = _texture_library(args.material_library)
        # Box UVs are metric: uv_scale is texture tiles per metre, the same on every renderer.
        for name, tiles in (("floor", (0.6, 1.4)), ("backdrop", (0.8, 1.6))):
            materials[name] = SurfaceRandomCfg(
                textures=library, uv_projection="box", uv_scale=(tiles, tiles), uv_rotation=(0.0, math.pi / 2)
            )
    if args.scene == "assets" and asset_textures and not args.task:
        materials["bbq_sauce"] = SurfaceRandomCfg(textures=asset_textures)
    camera_names = ["front", "side"] if args.multiview else ["front"]
    view = ViewRandomCfg(
        f_stop=((2.8, 8.0) if studio else (1.4, 11.0)) if args.depth_of_field else (0.0, 0.0),
        focus_scale=(0.9, 1.1),
    )
    # A camera in continuous auto-exposure mode: interiors, HDRIs and rigs differ widely in brightness.
    sensor = (
        SensorRandomCfg(auto_exposure=(0.2, 0.32))
        if studio
        else SensorRandomCfg(
            auto_exposure=(0.1, 0.35),
            exposure_ev=(-1.0, 1.0),
            white_balance=((0.8, 1.2), (0.8, 1.2)),
            vignetting=(0.0, 0.7),
            distortion=((-0.2, 0.08), (-0.02, 0.02), (-0.003, 0.003), (-0.003, 0.003)),
            blur_sigma=(0.0, 1.6),
            shot_noise=(0.0, 0.004),
            read_noise=(0.0, 0.001),
        )
    )
    return VisualRandomizer(
        VisualRandomizationCfg(
            materials=materials,
            lights={
                "key": LightingRandomCfg(color_temperature=(3200, 6500)),
                "fill": LightingRandomCfg(intensity_scale=(0.8, 1.2), color_temperature=(4000, 7500)),
            },
            cameras=dict.fromkeys(camera_names, view),
            environment=EnvironmentRandomCfg(
                hdri_paths=args.hdri,
                color=((0.35, 0.35),) * 3 if studio else ((0.15, 0.5),) * 3,
            ),
            sensors=dict.fromkeys(camera_names, sensor) if args.sensor else {},
        ),
        seed=args.seed,
    )


def _launch(args, scenario):
    from metasim.sim.hybrid import HybridSimHandler
    from metasim.utils.setup_util import get_sim_handler_class

    renderer = get_sim_handler_class(SimType(args.sim))(scenario)
    handler = renderer
    if args.physics == "mujoco":
        # Interior scenes are render-only; MuJoCo keeps a flat ground at z=0 (the scenes' floor height).
        physics_cfg = scenario.replace(simulator="mujoco", cameras=[], scene=None, add_default_ground=True)
        physics = get_sim_handler_class(SimType.MUJOCO)(physics_cfg)
        handler = HybridSimHandler(physics_cfg, physics, renderer)
    try:
        handler.launch()
    except BaseException:
        handler.close()
        raise
    return handler


_HASHES: dict[tuple[str, int, int], str] = {}


def _hash_files(paths):
    """SHA-256 per path, reusing digests of files unchanged since this process hashed them."""
    result = {}
    for path in sorted(paths):
        stat = os.stat(path)
        key = (path, stat.st_size, stat.st_mtime_ns)
        if key not in _HASHES:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            _HASHES[key] = digest.hexdigest()
        result[path] = _HASHES[key]
    return result


def _asset_hashes(recipe):
    paths = {p for material in recipe.materials.values() for p in material["textures"].values() if p}
    if recipe.environment and recipe.environment["hdri_path"]:
        paths.add(recipe.environment["hdri_path"])
    return _hash_files(paths)


def _scene_asset_hashes(scenario):
    """Mesh/robot description files the scenario references, for manifest provenance."""
    paths = set()
    for cfg in [*scenario.objects, *scenario.robots, *([scenario.scene] if scenario.scene else [])]:
        for name in ("usd_path", "mjcf_path", "urdf_path"):
            path = getattr(cfg, name, None)
            if path and Path(path).is_file():
                paths.add(str(Path(path).resolve()))
    return _hash_files(paths)


def _trajectory(args, scenario):
    """The recorded demo replayed for a task: its initial state, per-step actions and file hash."""
    from metasim.task.registry import get_task_class
    from metasim.utils.demo_util import get_traj

    path = get_task_class(args.task).traj_filepath
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"Task {args.task!r} has no local trajectory file: {path}")
    init_states, all_actions, _ = get_traj(path, scenario.robots[0])
    if args.demo >= len(all_actions):
        raise IndexError(f"Demo {args.demo} out of range; {path} holds {len(all_actions)} demonstrations")
    return {
        "path": str(Path(path).resolve()),
        "sha256": _hash_files([str(Path(path).resolve())]),
        "demo": args.demo,
        "initial_state": init_states[args.demo],
        "actions": all_actions[args.demo],
    }


def _backend_versions(args):
    versions = {}
    if args.sim == "blender":
        import bpy

        versions["blender"] = bpy.app.version_string
    else:
        import carb

        versions["isaacsim"] = carb.settings.get_settings().get("/app/version")
    if args.physics == "mujoco":
        import mujoco

        versions["mujoco"] = mujoco.__version__
    return versions


def _verify_replay(recipe_path, recipe):
    """Check saved recipe and texture/HDRI content before launching a simulator."""
    manifest_path = recipe_path.with_name("manifest.json")
    if not manifest_path.exists():
        # Standalone recipes have no content-hash provenance to compare against.
        return
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(recipe.to_json().encode()).hexdigest() != manifest["recipe_sha256"]:
        raise ValueError("Replay recipe hash differs from manifest; use --no-verify-assets for intentional edits")
    if _asset_hashes(recipe) != manifest["asset_sha256"]:
        raise ValueError("Replay asset hash differs from manifest; use --no-verify-assets for intentional edits")


def _image_metrics(rgb):
    """Image diagnostics, not a perceptual realism score; RGB is display encoded."""
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected uint8 HWC RGB, received {pixels.dtype} {pixels.shape}")
    return {
        "mean_rgb": pixels.mean(axis=(0, 1)).tolist(),
        "std": float(pixels.std()),
        "dark_fraction": float(np.all(pixels <= 2, axis=-1).mean()),
        "clipped_fraction": float(np.any(pixels >= 253, axis=-1).mean()),
    }


def _write_variant(args, handler, sampler, recipe, initial_state, provenance, actions=None):
    name = f"sample_{recipe.sample_id:06d}_variant_{recipe.variant_id:04d}"
    frames = args.frames or 1
    if actions is not None:
        # A recorded demo sets the length: every action is replayed, one image per steps_per_frame.
        available = 1 + math.ceil(len(actions) / args.steps_per_frame)
        frames = available if args.frames is None else min(args.frames, available)
    destination = args.output / name
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a fresh --output directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{name}_", dir=args.output))
    started = time.perf_counter()
    writers = {}
    frame_records = []
    try:
        (staging / "recipe.json").write_text(recipe.to_json() + "\n")
        hashes = _asset_hashes(recipe)
        physics = getattr(handler, "physics_handler", None)
        control_dt = physics.physics_dt * physics.scenario.decimation if physics is not None else None
        capture_fps = args.fps or (1 / (args.steps_per_frame * control_dt) if control_dt else None)
        if handler is not None:
            # Appearance first, without a render: restoring the physics state renders once through the hybrid.
            sampler.apply(recipe, render=initial_state is None)
            if initial_state is not None:
                handler.set_states(initial_state)
            elapsed_steps = 0
            for frame_index in range(frames):
                if frame_index and actions is None:
                    handler.simulate_steps(steps=args.steps_per_frame)
                    elapsed_steps += args.steps_per_frame
                elif frame_index:
                    chunk = actions[elapsed_steps : elapsed_steps + args.steps_per_frame]
                    handler.simulate_steps(steps=len(chunk), actions=[[step] * handler.num_envs for step in chunk])
                    elapsed_steps += len(chunk)
                state = handler.get_states(mode="tensor")
                frame_metrics = {}
                sensors = {}
                for view, camera in state.cameras.items():
                    for env_id, rgb in enumerate(camera.rgb.cpu().numpy()):
                        key = f"{view}_env_{env_id}"
                        if view in recipe.sensors:
                            capture = recipe.apply_sensor(view, rgb, camera.intrinsics[env_id], frame=frame_index)
                            sensors.setdefault(view, []).append({
                                "model": capture.model,
                                "intrinsics": capture.intrinsics,
                                "distortion": capture.distortion,
                                "exposure_gain": capture.exposure_gain,
                            })
                            if frame_index == 0:
                                imageio.imwrite(staging / f"{key}_raw.png", rgb)
                            rgb = capture.rgb
                        frame_metrics[key] = _image_metrics(rgb)
                        if frame_index == 0:
                            imageio.imwrite(staging / f"{key}.png", rgb)
                        if frames > 1:
                            if key not in writers:
                                # macro_block_size=1: ffmpeg must not resize frames to a multiple of 16,
                                # or the video would no longer match the per-frame calibration.
                                writers[key] = imageio.get_writer(
                                    staging / f"{key}.mp4", fps=capture_fps, macro_block_size=1
                                )
                            writers[key].append_data(rgb)
                # Calibration accompanies every frame; keep pose/focal jitter observable.
                # ``render`` is the renderer's pinhole; ``sensor`` describes the processed image.
                calibration = {
                    view: {
                        key: getattr(cam, key).cpu().tolist() if getattr(cam, key) is not None else None
                        for key in ("pos", "quat_world", "intrinsics")
                    }
                    for view, cam in state.cameras.items()
                }
                (staging / f"camera_{frame_index:05d}.json").write_text(
                    json.dumps({"render": calibration, "sensor": sensors}, indent=2)
                )
                frame_records.append({
                    "index": frame_index,
                    "control_steps": elapsed_steps,
                    "simulation_time_s": elapsed_steps * control_dt if control_dt else None,
                    "video_time_s": frame_index / capture_fps if capture_fps else None,
                    "images": frame_metrics,
                })
        for writer in writers.values():
            writer.close()
        writers.clear()
        manifest = {
            "status": "sampled" if handler is None else "rendered",
            "backend": args.sim,
            "physics": args.physics,
            "recipe_sha256": hashlib.sha256(recipe.to_json().encode()).hexdigest(),
            "asset_sha256": hashes,
            **provenance,
            "arguments": asdict(args),
            "frames": frame_records,
            "capture_fps": capture_fps,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        os.replace(staging, destination)
        log.info(f"Saved {destination} in {manifest['elapsed_seconds']:.2f}s")
    finally:
        for writer in writers.values():
            writer.close()
        if staging.exists():
            shutil.rmtree(staging)


def main(args: Args):
    """Sample or render complete variants, publishing only finished output folders."""
    sampler = _sampler(args)
    recipes = (
        [VisualRecipe.from_json(args.recipe.read_text())]
        if args.recipe
        else [
            sampler.sample(sample_id=args.sample_id, variant_id=i)
            for i in range(args.shard_index, args.variants, args.num_shards)
        ]
    )
    if args.recipe and args.verify_assets:
        _verify_replay(args.recipe, recipes[0])
    args.output.mkdir(parents=True, exist_ok=True)
    handler = None
    actions = None
    trajectory = None
    provenance = {"scene_asset_sha256": {}, "backend_versions": {}, "trajectory": None}
    try:
        if not args.recipes_only and recipes:
            scenario = _scenario(args)
            trajectory = _trajectory(args, scenario) if args.task else None
            handler = _launch(args, scenario)
            sampler.bind_handler(handler)
            provenance = {
                "scene_asset_sha256": _scene_asset_hashes(scenario),
                "backend_versions": _backend_versions(args),
                "trajectory": None,
            }
            if trajectory is not None:
                provenance["trajectory"] = {
                    "task": args.task,
                    "path": trajectory["path"],
                    "sha256": trajectory["sha256"],
                    "demo": args.demo,
                    "action_steps": len(trajectory["actions"]),
                }
        initial_state = None
        if handler is not None and args.physics == "mujoco":
            if trajectory is not None:
                handler.set_states([trajectory["initial_state"]] * handler.num_envs)
                actions = trajectory["actions"]
            if args.settle_steps:
                handler.simulate_steps(steps=args.settle_steps)
            initial_state = copy.deepcopy(handler.physics_handler.get_states(mode="tensor"))
        for recipe in recipes:
            _write_variant(args, handler, sampler, recipe, initial_state, provenance, actions)
        # Contact sheet includes only this invocation's completed stills.
        paths = [
            args.output / f"sample_{r.sample_id:06d}_variant_{r.variant_id:04d}" / "front_env_0.png" for r in recipes
        ]
        if paths and all(p.exists() for p in paths):
            imageio.imwrite(
                args.output / f"contact_sheet_shard_{args.shard_index}.png",
                np.concatenate([imageio.imread(p) for p in paths[:8]], axis=1),
            )
    finally:
        if handler is not None:
            handler.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
