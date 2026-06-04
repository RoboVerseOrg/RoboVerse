"""Dep-free regression tests for the RoboTwin bridge -> RoboVerse IL demo converter.

``robotwin_to_demo._convert_one`` is the bridge between the data collector
(``collect_bridge.py --rgb``) and the *unmodified* IL data pipeline
(``roboverse_learn/il/data2zarr_dp.py``). It must emit exactly the on-disk
contract data2zarr consumes -- ``demo_<id>/metadata.json`` with ``joint_qpos``
(achieved state) + ``joint_qpos_target`` (command action) and a ``rgb.mp4`` whose
frame count matches -- so these lock that contract without needing a RoboTwin
checkout, a GPU, or any sim. Only imageio + numpy are required.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONV = os.path.join(_HERE, "..", "tools", "robotwin_integration", "robotwin_to_demo.py")


def _load_converter():
    spec = importlib.util.spec_from_file_location("rt_to_demo", _CONV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_bridge(t=12, cams=("head_camera", "left_camera")):
    """A minimal bridge dict shaped like collect_bridge.py --rgb output."""
    rng = np.random.RandomState(0)
    return {
        "task": "beat_block_hammer",
        "seed": 3,
        "vectors": rng.rand(t, 14).astype(float),  # command target -> action
        "real_vectors": rng.rand(t, 14).astype(float),  # achieved state -> obs
        "rgb": {c: (rng.rand(t, 16, 24, 3) * 255).astype(np.uint8) for c in cams},
    }


@pytest.mark.general
def test_convert_one_writes_data2zarr_contract(tmp_path):
    """A demo dir has metadata.json (joint_qpos/_target, T x 14) + a T-frame rgb.mp4."""
    import imageio.v2 as imageio

    conv = _load_converter()
    bridge = _fake_bridge(t=10)
    demo_dir = str(tmp_path / "demo_0000")
    n = conv._convert_one(bridge, demo_dir, "head_camera")
    assert n == 10

    meta = json.load(open(os.path.join(demo_dir, "metadata.json")))
    # data2zarr reads exactly these two keys for observation/action space = joint_pos.
    assert len(meta["joint_qpos"]) == 10 and len(meta["joint_qpos"][0]) == 14
    assert len(meta["joint_qpos_target"]) == 10 and len(meta["joint_qpos_target"][0]) == 14
    # state is the achieved vector, action is the command vector (non-circular).
    assert np.allclose(meta["joint_qpos"], bridge["real_vectors"])
    assert np.allclose(meta["joint_qpos_target"], bridge["vectors"])
    # rgb.mp4 frame count must equal len(joint_qpos) (data2zarr zips them per-frame).
    frames = imageio.mimread(os.path.join(demo_dir, "rgb.mp4"))
    assert len(frames) == 10


@pytest.mark.general
def test_convert_one_aligns_ragged_lengths(tmp_path):
    """A frame missing its achieved capture must not desync state/action/rgb."""
    conv = _load_converter()
    bridge = _fake_bridge(t=12)
    bridge["real_vectors"] = bridge["real_vectors"][:9]  # achieved capture short by 3
    demo_dir = str(tmp_path / "demo_0001")
    n = conv._convert_one(bridge, demo_dir, "head_camera")
    assert n == 9  # min(12 action, 9 state, 12 rgb)
    meta = json.load(open(os.path.join(demo_dir, "metadata.json")))
    assert len(meta["joint_qpos"]) == len(meta["joint_qpos_target"]) == 9


@pytest.mark.general
def test_convert_one_skips_missing_camera(tmp_path):
    """Requesting a camera the bridge never captured is a clean skip, not a crash."""
    conv = _load_converter()
    bridge = _fake_bridge(cams=("head_camera",))
    n = conv._convert_one(bridge, str(tmp_path / "demo_x"), "right_camera")
    assert n == 0
    assert not os.path.isdir(str(tmp_path / "demo_x"))  # nothing written on skip
