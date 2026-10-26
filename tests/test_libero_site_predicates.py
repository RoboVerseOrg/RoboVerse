"""LIBERO-90's ``In`` / ``On`` site predicates reproduce LIBERO's own expressions, quirks included."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from metasim.utils.site_util import SitePose
from roboverse_pack.tasks.libero_90._site_predicates import site_contains, site_supports

_SIZE = (0.03, 0.05, 0.02)


def _site(yaw: float, pos=(0.1, -0.2, 0.3)) -> SitePose:
    quat = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], dtype=torch.float64)
    return SitePose(pos=torch.tensor([pos], dtype=torch.float64), quat=quat, size=_SIZE)


def _in_box(t, mat, size, p):
    """LIBERO SiteObject.in_box."""
    total = np.abs(mat @ np.asarray(size))
    ub, lb = t + total, t - total
    lb[2] -= 0.01
    return bool(np.all(p > lb) and np.all(p < ub))


def _under(t, mat, size, p, other_height=0.10):
    """LIBERO SiteObject.under."""
    d = mat @ (p - t)
    return bool(size[2] - 0.005 < d[2] < size[2] + other_height and np.all(np.abs(d[:2]) < np.asarray(size)[:2]))


@pytest.mark.parametrize("yaw", [0.0, 0.4, math.pi / 2, 2.5])
def test_predicates_agree_with_libero_on_random_points(yaw):
    site = _site(yaw)
    t, mat = site.pos[0].numpy(), site.rot[0].numpy()
    points = torch.as_tensor(np.random.default_rng(0).uniform(-0.15, 0.15, (4000, 3)), dtype=torch.float64) + site.pos
    many = SitePose(pos=site.pos.expand(len(points), 3), quat=site.quat.expand(len(points), 4), size=_SIZE)
    got_in, got_on = site_contains(many, points).numpy(), site_supports(many, points).numpy()
    want_in = np.array([_in_box(t, mat, _SIZE, p) for p in points.numpy()])
    want_on = np.array([_under(t, mat, _SIZE, p) for p in points.numpy()])
    assert want_in.any() and want_on.any()  # the sample reaches both regions
    np.testing.assert_array_equal(got_in, want_in)
    np.testing.assert_array_equal(got_on, want_on)


def test_in_lowers_the_floor_by_a_centimetre():
    site = _site(0.0)
    below = site.pos + torch.tensor([[0.0, 0.0, -_SIZE[2] - 0.005]], dtype=torch.float64)
    assert bool(site_contains(site, below)[0])


def test_on_spans_from_the_top_face_to_ten_centimetres_above():
    site = _site(0.0)
    for dz, expected in [(_SIZE[2] - 0.004, True), (_SIZE[2] + 0.099, True), (_SIZE[2] + 0.101, False), (0.0, False)]:
        p = site.pos + torch.tensor([[0.0, 0.0, dz]], dtype=torch.float64)
        assert bool(site_supports(site, p)[0]) is expected, dz


def test_a_site_without_a_box_size_is_an_error():
    site = SitePose(pos=torch.zeros(1, 3), quat=torch.tensor([[1.0, 0, 0, 0]]), size=(0.01,))
    with pytest.raises(ValueError, match="3-number size"):
        site_contains(site, torch.zeros(1, 3))


def test_every_libero_90_goal_names_objects_its_scenario_has():
    """A task's ``goal`` is checked against its own scenario (the constructor runs the same check)."""
    import importlib
    import pathlib

    from roboverse_pack.tasks.libero_90._goal import check_goal
    from roboverse_pack.tasks.libero_90.libero_90_base import Libero90BaseTask

    root = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks" / "libero_90"
    goals = 0
    for path in sorted(root.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "libero_90_base":
            continue
        mod = importlib.import_module(f"roboverse_pack.tasks.libero_90.{path.stem}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, Libero90BaseTask) and obj.__module__ == mod.__name__:
                if obj.goal is not None:
                    check_goal(obj.goal, {o.name for o in obj.scenario.objects})
                    goals += 1
                else:
                    assert obj.checker is not None or "_terminated" in vars(obj), f"{obj.__name__} defines no success"
    assert goals >= 62


def test_goal_terms_are_validated():
    from roboverse_pack.tasks.libero_90._goal import check_goal

    names = {"bowl", "wooden_cabinet", "flat_stove"}
    check_goal((("Close", "wooden_cabinet", "top_level"), ("On", "bowl", "wooden_cabinet", "top_side")), names)
    with pytest.raises(ValueError, match="not in the scenario"):
        check_goal((("In", "cup", "wooden_cabinet", "top_region"),), names)
    with pytest.raises(ValueError, match="thresholds are known for"):
        check_goal((("Turnon", "wooden_cabinet", "top_level"),), names)
    with pytest.raises(ValueError, match="expected one of the forms"):
        check_goal((("Under", "bowl", "flat_stove"),), names)


def test_goal_is_the_conjunction_per_env():
    import types

    from metasim.types import ObjectState, TensorState
    from roboverse_pack.tasks.libero_90._goal import goal_holds

    site = _site(0.0, pos=(0.0, 0.0, 0.0))
    states = TensorState(
        objects={
            "stove": ObjectState(
                root_state=torch.zeros(2, 13), joint_pos=torch.tensor([[0.7], [0.7]]), joint_vel=torch.zeros(2, 1)
            ),
            "pan": ObjectState(root_state=torch.tensor([[0, 0, 0.05] + [0.0] * 10, [0.2, 0, 0.05] + [0.0] * 10])),
        },
        robots={},
        cameras={},
    )
    task = types.SimpleNamespace(
        handler=types.SimpleNamespace(get_joint_names=lambda obj, sort=True: ["button"]),
        _site_pose=lambda s, obj, name: SitePose(
            pos=site.pos.float().expand(2, 3), quat=site.quat.float().expand(2, 4), size=_SIZE
        ),
    )
    goal = (("Turnon", "flat_stove", "button"), ("On", "pan", "flat_stove", "cook_region"))
    states.objects["flat_stove"] = states.objects.pop("stove")
    assert goal_holds(task, states, goal).tolist() == [True, False]  # env 1's pan is 20 cm off the region
