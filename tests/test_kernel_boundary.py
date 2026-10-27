"""Content talks to a simulator through the kernel handler surface, and the exceptions only shrink.

``metasim.kernel`` defines what a benchmark port may rely on: ``BaseSimHandler``'s public methods,
properties and the attributes its ``__init__`` sets (scenario, robots, objects, num_envs, ...).
Anything else read off a handler -- ``handler.physics`` (MuJoCo), ``handler.gym`` / ``_envs``
(Isaac Gym), ``handler.scene`` (Isaac Sim), ``object_ids`` (SAPIEN) -- ties the code to one backend
and to that backend's internals, which is how a task that declares two simulators ends up working
on one.

This test counts those reach-ins per content area and holds each area to its budget. Moving a
read onto ``get_states`` / a query lowers the count: lower the budget with it. A new area starts
at zero. Raising a budget needs a reason that belongs in the PR description.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest

from metasim.sim.base import BaseSimHandler

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Reach-ins past the kernel handler surface, per content area (1.0.0 started at 354).
BUDGET = {
    "tasks/mjlab": 61,  # +2: the Newton end-effector fallback reads tree depth from the body labels
    "roboverse_pack/queries": 44,
    "tasks/maniskill": 35,
    "roboverse_pack/randomization": 24,
    "tasks/libero": 21,
    "tasks/simpler_env": 13,
    "tasks/beyondmimic": 10,
    "tasks/robosuite": 7,
    "roboverse_pack/callback_funcs": 6,
    "roboverse_pack/utils": 2,
}


def _kernel_surface() -> set[str]:
    names = {n for n in dir(BaseSimHandler) if not n.startswith("_")}
    tree = ast.parse((_ROOT / "packages/metasim/metasim/sim/base.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and not node.attr.startswith("_")
        ):
            names.add(node.attr)
    return names


def _area(path: pathlib.Path) -> str:
    parts = path.relative_to(_ROOT).parts
    if parts[:2] == ("roboverse_pack", "tasks") and len(parts) > 3:
        return f"tasks/{parts[2]}"
    return "/".join(parts[:2])


def _reach_ins() -> dict[str, collections.Counter]:
    surface = _kernel_surface()
    found: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for top in ("roboverse_pack", "roboverse_learn"):
        for path in sorted((_ROOT / top).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or node.attr in surface:
                    continue
                owner = node.value
                if (isinstance(owner, ast.Name) and owner.id == "handler") or (
                    isinstance(owner, ast.Attribute) and owner.attr == "handler"
                ):
                    found[_area(path)][node.attr] += 1
    return found


def test_handler_reach_ins_stay_within_budget():
    found = _reach_ins()
    over = {
        area: f"{sum(c.values())} > {BUDGET.get(area, 0)} ({dict(c.most_common(4))})"
        for area, c in found.items()
        if sum(c.values()) > BUDGET.get(area, 0)
    }
    assert not over, (
        "content reads simulator internals past the kernel handler surface; use get_states / a query "
        f"instead (see tests/test_kernel_boundary.py): {over}"
    )


def test_budget_is_tight():
    """A budget above the count is slack a regression could hide in: lower it when you fix reach-ins."""
    found = _reach_ins()
    slack = {area: (sum(found[area].values()), n) for area, n in BUDGET.items() if sum(found[area].values()) < n}
    assert not slack, f"lower these budgets to the current count (count, budget): {slack}"


@pytest.mark.parametrize("name", ["get_states", "set_states", "set_dof_targets", "simulate", "launch", "num_envs"])
def test_the_kernel_verbs_are_on_the_surface(name):
    assert name in _kernel_surface()
