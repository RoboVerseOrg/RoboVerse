"""Regression test: MetaSim reproduces native ManiSkill object state bit-for-bit.

This pins the ManiSkill 1:1 integration claim — that the reproduction recipe
in ``tools/maniskill_integration/recipe.py`` rebuilds a native ManiSkill
(``physx_cpu``) tabletop scene exactly.  Native ManiSkill on the CPU backend is
itself bit-deterministic, so the object pose delta after replaying identical
actions should be exactly 0.

The test is heavy (imports ``mani_skill`` + SAPIEN and steps physics), so it
skips cleanly when those aren't installed; it is meant to run in the dedicated
``maniskill1to1`` env.  Forward/backward compat: purely additive.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("sapien")


@pytest.mark.parametrize("task_id", ["PickCube-v1", "PushCube-v1"])
def test_object_state_bitwise(task_id):
    from tools.maniskill_integration.parity_native import run_task

    # A short, smooth (demo-like) rollout — enough to move the arm and settle
    # the objects while keeping the run fast.
    r = run_task(task_id, steps=10, seed=0)

    assert r["objects"], f"{task_id}: no objects captured"
    for name, o in r["objects"].items():
        # Every primitive-geometry object must be frame-bitwise.
        assert o["bitwise"], f"{task_id}: object {name} not bitwise (Δ={o['delta_max']:.2e})"

    # The robot rides on float32 solver roundoff; assert it stays small rather
    # than exactly zero (this is the documented, expected behaviour).
    assert r["qpos_delta_max"] < 1e-3, f"{task_id}: qpos delta too large ({r['qpos_delta_max']:.2e})"
    assert np.isfinite(r["qpos_delta_max"])
