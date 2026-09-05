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
