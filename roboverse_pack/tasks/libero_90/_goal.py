"""A LIBERO-90 task's success condition as data: the terms of its BDDL ``(:goal (And ...))``.

A task declares, in RoboVerse object names::

    goal = (
        ("Close", "wooden_cabinet", "top_level"),
        ("On", "akita_black_bowl", "wooden_cabinet", "top_side"),
    )

and :class:`~roboverse_pack.tasks.libero_90.libero_90_base.Libero90BaseTask` evaluates it with LIBERO's own
predicates (``_site_predicates``) on the state. Term forms:

- ``("In", obj, region_obj, site)`` / ``("On", obj, region_obj, site)``: ``obj``'s root in / on a site region;
- ``("On", obj, base_obj)``: ``obj`` resting on another object;
- ``("Open", obj, joint)`` / ``("Close", obj, joint)``: a cabinet drawer or the microwave door;
- ``("Turnon", obj, joint)`` / ``("Turnoff", obj, joint)``: the stove's knob joint.
"""

from __future__ import annotations

import torch

from metasim.utils.site_util import site_sizes
from metasim.utils.xml_safe import ET

from ._site_predicates import (
    CABINET_CLOSED_ABOVE,
    CABINET_OPEN_BELOW,
    MICROWAVE_CLOSED_ABOVE,
    MICROWAVE_OPEN_BELOW,
    STOVE_OFF_BELOW,
    STOVE_ON_FROM,
    object_on,
    site_contains,
    site_supports,
)

__all__ = ["GoalTerm", "check_goal", "goal_holds"]

GoalTerm = tuple[str, ...]

#: LIBERO's joint thresholds, per (predicate, asset).
_JOINT_TESTS = {
    ("Open", "wooden_cabinet"): lambda q: q < CABINET_OPEN_BELOW,
    ("Open", "white_cabinet"): lambda q: q < CABINET_OPEN_BELOW,
    ("Open", "microwave"): lambda q: q < MICROWAVE_OPEN_BELOW,
    ("Close", "wooden_cabinet"): lambda q: q > CABINET_CLOSED_ABOVE,
    ("Close", "white_cabinet"): lambda q: q > CABINET_CLOSED_ABOVE,
    ("Close", "microwave"): lambda q: q > MICROWAVE_CLOSED_ABOVE,
    ("Turnon", "flat_stove"): lambda q: q >= STOVE_ON_FROM,
    ("Turnoff", "flat_stove"): lambda q: q < STOVE_OFF_BELOW,
}
_JOINT_KINDS = {kind for kind, _ in _JOINT_TESTS}


def check_goal(goal: tuple[GoalTerm, ...], objects: dict, *, read_assets: bool = True) -> None:
    """Fail at construction on a goal that cannot be evaluated on this scenario.

    ``objects`` maps scenario object names to their configs. Checked: the goal is not empty, each term has a
    known form, its objects exist, a region is a box site the region object's MJCF declares, a joint is one its
    MJCF declares, and LIBERO's thresholds are known for a joint predicate's asset. ``read_assets=False``
    skips the two checks that read the MJCF files.
    """
    if not goal:
        raise ValueError("an empty goal never decides success; leave goal=None to use a checker")
    shapes = {"In": (3,), "On": (2, 3), **{k: (2,) for k in _JOINT_KINDS}}
    for term in goal:
        kind, *args = term
        if kind not in shapes or len(args) not in shapes[kind]:
            raise ValueError(f"goal term {term!r}: expected one of the forms in {__name__}")
        names = args[:2] if kind in ("In", "On") else args[:1]
        missing = [n for n in names if n not in objects]
        if missing:
            raise ValueError(f"goal term {term!r} names objects not in the scenario: {missing}")
        if read_assets and kind in ("In", "On") and len(args) == 3:
            sizes = site_sizes(obj=objects[args[1]])
            if args[2] not in sizes:
                raise ValueError(f"goal term {term!r}: {args[1]!r} declares no site {args[2]!r} ({sorted(sizes)})")
            if sizes[args[2]] is None or len(sizes[args[2]]) != 3:
                raise ValueError(f"goal term {term!r}: site {args[2]!r} is not a box (size={sizes[args[2]]})")
        if kind in _JOINT_KINDS:
            if (kind, args[0]) not in _JOINT_TESTS:
                known = sorted(asset for k, asset in _JOINT_TESTS if k == kind)
                raise ValueError(f"goal term {term!r}: LIBERO's {kind} thresholds are known for {known}")
            if not read_assets:
                continue
            joints = {j.get("name") for j in ET.parse(objects[args[0]].mjcf_path).getroot().iter("joint")}
            if args[1] not in joints:
                raise ValueError(
                    f"goal term {term!r}: {args[0]!r} has no joint {args[1]!r} ({sorted(joints - {None})})"
                )


def goal_holds(task, states, goal: tuple[GoalTerm, ...]) -> torch.Tensor:
    """(N,) bool: every term of ``goal`` holds in ``states``."""
    ok = None
    for kind, *args in goal:
        if kind in ("In", "On") and len(args) == 3:
            obj, region_obj, site = args
            point = states.objects[obj].root_state[:, :3]
            pose = task._site_pose(states, region_obj, site)
            held = site_contains(pose, point) if kind == "In" else site_supports(pose, point)
        elif kind == "On":
            obj, base = args
            held = object_on(states.objects[obj].root_state[:, :3], states.objects[base].root_state[:, :3])
        else:
            obj, joint = args
            names = task.handler.get_joint_names(obj, sort=True)
            if joint not in names:
                raise KeyError(f"goal term {(kind, *args)!r}: {obj!r} has no joint {joint!r} (joints: {names})")
            held = _JOINT_TESTS[kind, obj](states.objects[obj].joint_pos[:, names.index(joint)])
        ok = held if ok is None else ok & held
    return ok
