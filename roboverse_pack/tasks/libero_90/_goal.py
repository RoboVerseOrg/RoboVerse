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
- ``("Open", obj, joint)`` / ``("Close", obj, joint)``: a cabinet drawer joint;
- ``("Turnon", obj, joint)`` / ``("Turnoff", obj, joint)``: the stove's knob joint.
"""

from __future__ import annotations

import torch

from ._site_predicates import (
    CABINET_CLOSED_ABOVE,
    CABINET_OPEN_BELOW,
    STOVE_OFF_BELOW,
    STOVE_ON_FROM,
    object_on,
    site_contains,
    site_supports,
)

__all__ = ["GoalTerm", "check_goal", "goal_holds"]

GoalTerm = tuple[str, ...]

#: Joint predicates and the assets whose LIBERO thresholds they carry.
_JOINT_TESTS = {
    "Open": (("wooden_cabinet", "white_cabinet"), lambda q: q < CABINET_OPEN_BELOW),
    "Close": (("wooden_cabinet", "white_cabinet"), lambda q: q > CABINET_CLOSED_ABOVE),
    "Turnon": (("flat_stove",), lambda q: q >= STOVE_ON_FROM),
    "Turnoff": (("flat_stove",), lambda q: q < STOVE_OFF_BELOW),
}


def check_goal(goal: tuple[GoalTerm, ...], object_names: set[str]) -> None:
    """Fail at construction on a term this module cannot evaluate or that names a missing object."""
    for term in goal:
        kind, *args = term
        shapes = {"In": (3,), "On": (2, 3), **{k: (2,) for k in _JOINT_TESTS}}
        if kind not in shapes or len(args) not in shapes[kind]:
            raise ValueError(f"goal term {term!r}: expected one of the forms in {__name__}")
        names = args[:2] if kind in ("In", "On") else args[:1]
        missing = [n for n in names if n not in object_names]
        if missing:
            raise ValueError(f"goal term {term!r} names objects not in the scenario: {missing}")
        if kind in _JOINT_TESTS and args[0] not in _JOINT_TESTS[kind][0]:
            raise ValueError(f"goal term {term!r}: LIBERO's {kind} thresholds are known for {_JOINT_TESTS[kind][0]}")


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
            held = _JOINT_TESTS[kind][1](states.objects[obj].joint_pos[:, names.index(joint)])
        ok = held if ok is None else ok & held
    return ok
