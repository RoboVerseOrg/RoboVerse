"""The front door works: the getting-started examples run headless on MuJoCo, end to end.

Each example is run as a subprocess from the repository root (asset paths are cwd-relative), one or two
envs, and must exit 0 without a traceback, including at interpreter exit (a renderer collected after the
EGL display used to end every camera example in an ``EGLError`` traceback). MuJoCo rendering needs a GL
backend, which the hosted CI runners lack (``MUJOCO_GL=disable``); there the lane skips with that reason.
Elsewhere the subprocesses run with ``MUJOCO_GL=egl`` (headless), so the lane runs on any machine with EGL.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("mujoco")

ROOT = pathlib.Path(__file__).resolve().parents[1]
# the assets these examples load (their scenarios name them); the marker skips, naming the file, when
# one is absent, so the suite never fetches from the Hub as a side effect of being collected
COMMON_ASSETS = (
    "robots/franka/mjcf/panda.xml",
    "robots/h1/mjcf/h1.xml",
    "assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
    "assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
)
EMBODIEDGEN = pathlib.Path("roboverse_data/assets/EmbodiedGenData/demo_assets")
EXAMPLES = [
    ("examples/0_static_scene.py", ["--sim", "mujoco", "--headless"]),
    ("examples/1_control_robot.py", ["--sim", "mujoco", "--headless"]),
    ("examples/2_add_new_robot.py", ["--sim", "mujoco", "--headless"]),
    ("examples/3_parallel_envs.py", ["--sim", "mujoco", "--headless", "--num_envs", "2"]),
    ("examples/7_multiple_robots.py", ["--sim", "mujoco", "--headless"]),
    ("examples/9_cfg_task.py", ["--sim", "mujoco", "--headless", "--device", "cpu", "--no-save_video"]),
    ("examples/10_mount_camera.py", ["--sim", "mujoco", "--headless"]),
    ("examples/13_get_exras.py", ["--sim", "mujoco", "--headless"]),
    ("examples/14_real_assets.py", ["--sim", "mujoco", "--headless"]),
]


@pytest.mark.examples
@pytest.mark.requires_asset(*COMMON_ASSETS)
@pytest.mark.skipif(
    os.environ.get("MUJOCO_GL", "").strip().lower() in ("disable", "disabled", "off", "false", "0"),
    reason="MuJoCo rendering is disabled (MUJOCO_GL): the examples render cameras",
)
@pytest.mark.parametrize(("script", "args"), EXAMPLES, ids=[s.split("/")[-1] for s, _ in EXAMPLES])
def test_example_runs_headless_on_mujoco(script, args):
    if script.endswith("14_real_assets.py") and not (ROOT / EMBODIEDGEN).is_dir():
        pytest.skip(f"{EMBODIEDGEN} is absent; the example would download it from the Hub (run it once by hand)")
    proc = subprocess.run(
        [sys.executable, script, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        # the examples are run headless: EGL is MuJoCo's display-less backend (GLFW needs a display)
        env={**os.environ, "MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1"},
        check=False,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"{script} exited {proc.returncode}:\n{tail}"
    assert "Traceback" not in proc.stderr and "Exception ignored" not in proc.stderr, (
        f"{script} printed a traceback:\n{tail}"
    )


def test_visual_example_sampling_replay_and_failed_output_are_atomic(tmp_path):
    """Recipe-only mode exercises the real CLI without launching a renderer."""
    import json

    def run(*args):
        return subprocess.run(
            [sys.executable, "examples/6_advanced_rendering.py", "--recipes-only", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

    output = tmp_path / "sampled"
    proc = run("--variants", "3", "--output", str(output))
    assert proc.returncode == 0, proc.stderr
    dirs = sorted(output.glob("sample_*"))
    assert len(dirs) == 3
    recipe = dirs[1] / "recipe.json"
    saved = recipe.read_text()
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    assert manifest["status"] == "sampled"
    assert not list(output.glob(".*"))

    replay = tmp_path / "replay"
    proc = run("--recipe", str(recipe), "--output", str(replay))
    assert proc.returncode == 0, proc.stderr
    assert (replay / dirs[1].name / "recipe.json").read_text() == saved
    proc = run("--output", str(output))
    assert proc.returncode != 0 and "Output already exists" in proc.stderr
    assert recipe.read_text() == saved
    assert not list(output.glob(".*"))
    data = json.loads(saved)
    data["materials"]["cube"]["roughness"] = 0.1
    recipe.write_text(json.dumps(data))
    rejected = tmp_path / "rejected"
    proc = run("--recipe", str(recipe), "--output", str(rejected))
    assert proc.returncode != 0 and "recipe hash differs" in proc.stderr
    assert not rejected.exists()


@pytest.mark.parametrize("scene", ["primitives", "assets"])
def test_visual_example_constructs_scene_without_optional_backends(scene):
    """Recipe export must not hide invalid primitive configs in the rendering path."""
    import runpy

    example = runpy.run_path(str(ROOT / "examples/6_advanced_rendering.py"))
    args = example["Args"](scene=scene, multiview=True)
    scenario = example["_scenario"](args)
    assert {obj.name for obj in scenario.objects} >= {"cube", "sphere", "backdrop"}
    assert [camera.name for camera in scenario.cameras] == ["front", "side"]
    assert not scenario.add_default_ground
    floor = next(obj for obj in scenario.objects if obj.name == "floor")
    assert floor.default_position[2] + floor.size[2] / 2 == 0


def test_visual_example_background_scene_resolves_for_both_renderers():
    """Interior scene configs must expose a Blender file type; the pack lacked it and raised KeyError."""
    import runpy

    example = runpy.run_path(str(ROOT / "examples/6_advanced_rendering.py"))
    for sim in ("blender", "isaacsim"):
        scenario = example["_scenario"](example["Args"](sim=sim, background="kujiale_scene_0009"))
        assert scenario.scene.file_name(sim).endswith("kujiale_0009/009.usda")
        assert not scenario.add_default_ground
        assert {obj.name for obj in scenario.objects} == {"cube", "sphere"}
        assert [light.name for light in scenario.lights] == ["key", "fill"]


def test_visual_replay_rejects_changed_texture_and_image_metrics_detect_extremes(tmp_path):
    import hashlib
    import json
    import runpy

    import numpy as np

    from metasim.randomization import SurfaceRandomCfg, TextureSetCfg, VisualRandomizationCfg, VisualRandomizer

    example = runpy.run_path(str(ROOT / "examples/6_advanced_rendering.py"))
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"original")
    recipe = VisualRandomizer(
        VisualRandomizationCfg(materials={"cube": SurfaceRandomCfg(textures=(TextureSetCfg(base_color=str(texture)),))})
    ).sample(sample_id=0)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(recipe.to_json())
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "recipe_sha256": hashlib.sha256(recipe.to_json().encode()).hexdigest(),
            "asset_sha256": example["_asset_hashes"](recipe),
        })
    )
    example["_verify_replay"](recipe_path, recipe)
    texture.write_bytes(b"modified")
    with pytest.raises(ValueError, match="asset hash differs"):
        example["_verify_replay"](recipe_path, recipe)
    assert example["_image_metrics"](np.zeros((4, 4, 3), np.uint8))["dark_fraction"] == 1
    assert example["_image_metrics"](np.full((4, 4, 3), 255, np.uint8))["clipped_fraction"] == 1
