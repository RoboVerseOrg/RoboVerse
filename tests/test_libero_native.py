"""Native LIBERO task is libero/robosuite-free + checker geometry is correct."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np


def test_native_modules_import_without_libero_or_robosuite():
    before = set(sys.modules)
    importlib.import_module("roboverse_pack.tasks.libero_native.checker")
    importlib.import_module("roboverse_pack.tasks.libero_native.osc")
    importlib.import_module("roboverse_pack.tasks.libero_native.native_task")
    pulled = {m.split(".")[0] for m in set(sys.modules) - before}
    assert "libero" not in pulled, "native task must not import libero"
    assert "robosuite" not in pulled, "native task must not import robosuite"


def test_in_box_matches_libero_semantics():
    from roboverse_pack.tasks.libero_native.checker import in_box

    pos, mat, half = np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.1])
    assert in_box(pos, mat, half, np.array([0.0, 0.0, 0.0]))  # center -> inside
    assert not in_box(pos, mat, half, np.array([0.2, 0.0, 0.0]))  # outside x
    # lower-z is extended by 1cm (verbatim from LIBERO SiteObject.in_box)
    assert in_box(pos, mat, half, np.array([0.0, 0.0, -0.105]))
    assert not in_box(pos, mat, half, np.array([0.0, 0.0, -0.12]))


def test_under_matches_libero_semantics():
    from roboverse_pack.tasks.libero_native.checker import under

    pos, mat, size = np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.05])
    # resting just above the site floor: size[2]-0.005 < dz < size[2]+0.10
    assert under(pos, mat, size, np.array([0.0, 0.0, 0.06]))
    assert not under(pos, mat, size, np.array([0.0, 0.0, 0.20]))  # too high
    assert not under(pos, mat, size, np.array([0.0, 0.0, 0.0]))  # below floor band
    assert not under(pos, mat, size, np.array([0.15, 0.0, 0.06]))  # outside xy


def test_from_vendored_loads_without_libero_or_robosuite():
    """A vendored task loads + runs from roboverse_data with libero/robosuite blocked.

    Skips when the vendored assets aren't present locally (they live on HF
    ``RoboVerseOrg/roboverse_data`` under ``libero/`` — see migrate_libero_assets).
    """
    import builtins
    import glob

    import pytest

    from roboverse_pack.tasks.libero_native._locator import _data_dirs

    vendored = next((d for d in _data_dirs() if (d / "libero" / "tasks").exists()), None)
    if vendored is None:
        pytest.skip("vendored roboverse_data/libero not present locally")
    xmls = glob.glob(str(vendored / "libero" / "tasks" / "*" / "*.xml"))
    suite_task = os.path.relpath(xmls[0], vendored / "libero" / "tasks")[:-4].split(os.sep)

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name.split(".")[0] in ("libero", "robosuite"):
            raise ImportError(f"{name} is blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        from roboverse_pack.tasks.libero_native.native_task import LiberoNativeTask

        tk = LiberoNativeTask.from_vendored(suite_task[0], suite_task[1])
        assert tk.model.nq > 0 and tk.success() in (True, False)
        assert "libero" not in sys.modules and "robosuite" not in sys.modules
    finally:
        builtins.__import__ = real_import


def test_articulated_and_contact_predicates_eval():
    """The 4 articulated reductions + on/on_site dict forms evaluate (no libero)."""
    import mujoco

    from roboverse_pack.tasks.libero_native.checker import eval_predicate

    xml = """
    <mujoco><worldbody>
      <body name="drawer"><joint name="dj" type="slide" axis="1 0 0"/>
        <geom name="dg" type="box" size=".05 .05 .05"/></body>
      <body name="cube" pos="0 0 .2"><freejoint/><geom name="cg" type="box" size=".02 .02 .02"/></body>
    </worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    addr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "dj")]
    d.qpos[addr] = 0.2
    mujoco.mj_forward(m, d)
    # open if qpos > 0.1 (any joint); closed otherwise
    assert eval_predicate(m, d, {"fn": "joint", "reduce": "any", "op": "gt", "thr": 0.1, "qpos_addr": [int(addr)]})
    assert not eval_predicate(m, d, {"fn": "joint", "reduce": "all", "op": "lt", "thr": 0.1, "qpos_addr": [int(addr)]})
