"""Regression: the shipped ManiSkill panda fingers carry ManiSkill's grasp friction (2.0).

ManiSkill sets the panda finger contact material to ``static = dynamic = 2.0`` (+ patch radius 0.1)
in code via ``Panda.urdf_config`` — NOT in the URDF — so a plain SAPIEN3 URDF load leaves the fingers
at the scene-default 0.3 and grasped objects slip ~1 cm during a lift, flipping demo-replay success
(8/12 → 22/25 on PickCube). ``recipe.apply_maniskill_gripper_friction`` re-applies it after load and
the base task calls it on reset. These tests lock that in.

Heavy (SAPIEN); skips without it. Run in the ``maniskill1to1`` env.
"""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("sapien")


def _finger_frictions(robot):
    """Static frictions of the panda finger-link collision shapes (raw or wrapped articulation)."""
    out = {}
    for link in robot.get_links():
        name = link.get_name() if hasattr(link, "get_name") else link.name
        if "finger" not in name:
            continue
        raw = link._objs[0] if hasattr(link, "_objs") else link
        shapes = getattr(raw, "collision_shapes", None) or raw.get_collision_shapes()
        if shapes:
            out[name] = shapes[0].get_physical_material().get_static_friction()
    return out


def _task(name):
    import roboverse_pack.tasks.maniskill  # noqa: F401
    from metasim.task.registry import get_task_class

    cls = get_task_class(name)
    sc = copy.deepcopy(cls.scenario)
    sc.simulator = "sapien3"
    sc.num_envs = 1
    sc.headless = True
    sc.cameras = []
    return cls(sc)


def test_shipped_panda_fingers_have_grasp_friction():
    """After reset, the shipped pick_cube panda fingers are at ManiSkill's friction 2.0, not 0.3."""
    t = _task("maniskill.pick_cube_native")
    t.reset(seed=0)
    fr = _finger_frictions(t.handler.object_ids["panda"])
    t.close()
    assert fr, "no finger links found"
    for link, mu in fr.items():
        assert abs(mu - 2.0) < 1e-4, f"{link} friction {mu} != 2.0 (gripper material not applied)"


def test_apply_is_idempotent_and_counts_shapes():
    """The helper returns the number of shapes updated and re-applying is a no-op on count parity."""
    from roboverse_pack.tasks.maniskill._native.recipe import apply_maniskill_gripper_friction

    t = _task("maniskill.pick_cube_native")
    t.reset(seed=0)
    robot = t.handler.object_ids["panda"]
    n1 = apply_maniskill_gripper_friction(robot)
    n2 = apply_maniskill_gripper_friction(robot)
    fr = _finger_frictions(robot)
    t.close()
    assert n1 > 0 and n1 == n2
    assert all(abs(mu - 2.0) < 1e-4 for mu in fr.values())


def test_no_op_for_gripperless_robot():
    """panda_stick has no finger links — the helper applies nothing and raises nothing."""
    from roboverse_pack.tasks.maniskill._native.recipe import apply_maniskill_gripper_friction

    t = _task("maniskill.push_t_native")
    t.reset(seed=0)
    n = apply_maniskill_gripper_friction(t.handler.object_ids["panda"])
    t.close()
    assert n == 0
