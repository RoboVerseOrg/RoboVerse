"""Native LIBERO tasks (roboverse_pack.tasks.libero.native_libero) — library-free.

Covers the BDDL success checker semantics and that a registered native task loads
and runs with the ``libero``/``robosuite`` packages absent (assets are vendored to
``roboverse_data``; the test skips when they aren't present locally).
"""

from __future__ import annotations

import importlib
import os

import mujoco


def test_in_region_matches_libero_in_box():
    from roboverse_pack.tasks.libero._native_util import check_bddl_success

    xml = """
    <mujoco><worldbody>
      <site name="reg" type="box" pos="0 0 0" size="0.1 0.1 0.05"/>
      <body name="cube_main" pos="0 0 0"><freejoint/><geom type="box" size=".01 .01 .01"/></body>
    </worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    addr = m.jnt_qposadr[0]
    d.qpos[addr : addr + 3] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    assert check_bddl_success(m, d, [("In", ["cube", "reg"])])  # centred -> inside
    d.qpos[addr : addr + 3] = [0.2, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    assert not check_bddl_success(m, d, [("In", ["cube", "reg"])])  # outside x
    # LIBERO extends the lower z bound by 1cm: -0.055 is inside only because of it
    # (site half-z 0.05 -> floor -0.05, extended to -0.06), -0.07 stays outside.
    d.qpos[addr : addr + 3] = [0.0, 0.0, -0.055]
    mujoco.mj_forward(m, d)
    assert check_bddl_success(m, d, [("In", ["cube", "reg"])])
    d.qpos[addr : addr + 3] = [0.0, 0.0, -0.07]
    mujoco.mj_forward(m, d)
    assert not check_bddl_success(m, d, [("In", ["cube", "reg"])])


def test_open_close_thresholds_have_a_dead_zone():
    """Open/Close are not complements (a dead zone between the ranges)."""
    from roboverse_pack.tasks.libero._native_util import _is_close, _is_open

    xml = """
    <mujoco><worldbody>
      <body name="wooden_cabinet_1_main"><joint name="wooden_cabinet_1_top_level" type="slide" axis="1 0 0"/>
        <geom type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    region = "wooden_cabinet_1_top_region"
    for q, want_open, want_close in [(-0.15, True, False), (-0.10, False, False), (0.01, False, True)]:
        d.qpos[0] = q
        mujoco.mj_forward(m, d)
        assert _is_open(m, d, region) is want_open and _is_close(m, d, region) is want_close, q


def test_registered_native_task_runs_without_libero():
    """A registered native LIBERO task loads from roboverse_data + steps, no libero."""
    import builtins

    import pytest

    from roboverse_pack.tasks.libero._native_util import roboverse_data_assets

    if not os.path.isdir(roboverse_data_assets()):
        pytest.skip("vendored roboverse_data/libero assets not present locally")
    libero_pkg = os.path.dirname(importlib.import_module("roboverse_pack.tasks.libero").__file__)
    bdir = os.path.join(libero_pkg, "native_bundles")
    bundles = [b for b in sorted(os.listdir(bdir)) if os.path.exists(os.path.join(bdir, b, "model.xml"))]
    if not bundles:
        pytest.skip("no native bundles present")

    import torch

    from roboverse_pack.tasks.libero.native_libero import NativeLiberoEnv  # pre-import deps (no libero)

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        # any *new* import of libero/robosuite during load+run would raise -> proves
        # the native task needs neither (already-loaded modules can't be unloaded
        # reliably mid-pytest, so we assert on "does not import", not on sys.modules).
        if name.split(".")[0] in ("libero", "robosuite"):
            raise ImportError(f"{name} blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        cls = type("T", (NativeLiberoEnv,), {"bundle": bundles[0]})
        env = cls()
        env.reset()
        _, _, _, _, info = env.step(torch.zeros(7))
        assert "success" in info
        env.close()
    finally:
        builtins.__import__ = real_import
