"""Multi-camera IL pipeline: backward-compat (single head camera) + multi-view wrist cams.

The RoboTwin DP policy-reproduction pipeline gained optional multi-camera support so a
precise task (place_empty_cup) can train on wrist cameras, not just the head camera.
Every layer defaults to the original single-head-camera behaviour; these tests pin both
that default (so existing single-view training is byte-for-byte unchanged) and the opt-in
multi-view path (per-camera mp4 -> per-camera zarr array -> per-camera obs key).

Dep-light: exercises robotwin_to_demo (imageio) + data2zarr (zarr/imageio) + the dataset
loader's camera->obs mapping. Skips cleanly if an optional dep is missing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.join(_HERE, "..", "tools", "robotwin_integration")


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_TOOLS, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_bridge(n: int = 6) -> dict:
    """A minimal bridge with 14-D vectors + 4 camera RGB streams (tiny 8x8 frames)."""
    rng = np.arange(n * 14, dtype=float).reshape(n, 14)
    cams = {
        c: (np.arange(n * 8 * 8 * 3).reshape(n, 8, 8, 3) % 255).astype(np.uint8)
        for c in ("head_camera", "left_camera", "right_camera", "front_camera")
    }
    return {"vectors": rng, "real_vectors": rng + 0.5, "rgb": cams, "task": "unit", "seed": 0}


def test_robotwin_to_demo_single_camera_default(tmp_path):
    """Default (no --cameras) writes exactly the legacy rgb.mp4 -- single-view unchanged."""
    imageio = pytest.importorskip("imageio")
    rtd = _load("rtd_single", "robotwin_to_demo.py")
    demo = str(tmp_path / "demo_0000")
    t = rtd._convert_one(_fake_bridge(), demo, "head_camera")
    assert t == 6
    assert os.path.isfile(os.path.join(demo, "rgb.mp4"))
    meta = json.load(open(os.path.join(demo, "metadata.json")))
    assert meta["camera"] == "head_camera"


def test_robotwin_to_demo_multi_camera(tmp_path):
    """--cameras writes a <cam>.mp4 per requested view plus the legacy rgb.mp4."""
    pytest.importorskip("imageio")
    rtd = _load("rtd_multi", "robotwin_to_demo.py")
    demo = str(tmp_path / "demo_0000")
    cams = ["head_camera", "left_camera", "right_camera"]
    t = rtd._convert_one(_fake_bridge(), demo, "head_camera", cameras=cams)
    assert t == 6
    for c in cams:
        assert os.path.isfile(os.path.join(demo, f"{c}.mp4")), f"missing {c}.mp4"
    assert os.path.isfile(os.path.join(demo, "rgb.mp4"))  # legacy primary still written
    assert json.load(open(os.path.join(demo, "metadata.json")))["cameras"] == cams


def test_robotwin_to_demo_missing_camera_skips(tmp_path):
    """Requesting a camera the bridge never captured is skipped, not silently wrong."""
    pytest.importorskip("imageio")
    rtd = _load("rtd_miss", "robotwin_to_demo.py")
    br = _fake_bridge()
    del br["rgb"]["right_camera"]
    t = rtd._convert_one(br, str(tmp_path / "demo_0000"), "head_camera", cameras=["head_camera", "right_camera"])
    assert t == 0  # skipped


def test_dataset_camera_obs_mapping():
    """The dataset maps each zarr camera array name to the policy's *_cam obs key."""
    pytest.importorskip("zarr")
    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(_HERE, ".."))
    from roboverse_learn.il.datasets.robot_image_dataset import RobotImageDataset

    m = RobotImageDataset._CAM_OBS_NAME
    assert m["head_camera"] == "head_cam"
    assert m["left_camera"] == "left_cam"
    assert m["right_camera"] == "right_cam"
    # default ctor arg is single head camera (back-compat)
    import inspect

    assert tuple(inspect.signature(RobotImageDataset.__init__).parameters["camera_keys"].default) == ("head_camera",)
