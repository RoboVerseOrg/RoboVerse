"""Regression guard: ``joint_pos_target`` must be reported in the same
alphabetically-sorted joint order as ``joint_pos``/``joint_vel`` on every
backend that materializes it from a name-keyed action cache.

Motivation: ``joint_pos``/``joint_vel``/``joint_effort_target`` are emitted in
sorted-name order (via ``joint_reindex`` / ``_get_joint_ids_reindex``), but
several backends assembled the reported ``joint_pos_target`` by iterating their
*native* URDF joint order instead. Whenever a robot's native joint order is not
already alphabetical (e.g. numeric names ``joint_2``/``joint_10``, or ``A,C,B``),
``joint_pos_target[i]`` then referred to a different joint than ``joint_pos[i]``
— a silent index misalignment. Commit 92755f6 fixed this on sapien2/genesis;
isaacgym and pybullet were the remaining offenders.

The faithful check needs a live backend (GPU/import order), which CI can't run,
so this is a static AST guard instead: for each backend it pins that the
``joint_pos_target`` materializer iterates ``_get_joint_names(..., sort=True)``
and no longer references the native-order joint list. Pure-Python, no sim env,
no GPU — runs under ``-k general``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SIM_ROOT = Path(__file__).resolve().parents[1].joinpath("sim")


def _find_function(tree: ast.AST, class_name: str, func_name: str) -> ast.FunctionDef:
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == func_name)


def _calls_get_joint_names_sorted(fn: ast.FunctionDef) -> bool:
    """True if ``fn`` calls ``*._get_joint_names(...)`` / ``*.get_joint_names(...)`` with ``sort=True``.

    Accepts either the keyword form ``sort=True`` or the positional form
    ``_get_joint_names(obj_name, True)`` — both mean sorted order.
    """
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "get_action_joint_names":
            return True  # the handler-order layout: sorted joint names per robot by construction
        if node.func.attr not in ("_get_joint_names", "get_joint_names"):
            continue
        for kw in node.keywords:
            if kw.arg == "sort" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
        # positional sort is the 2nd arg after obj_name
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value is True:
            return True
    return False


def _references_attr(fn: ast.FunctionDef, attr: str) -> bool:
    return any(isinstance(node, ast.Attribute) and node.attr == attr for node in ast.walk(fn))


# (source file, class, function that materializes joint_pos_target, native-order
#  attribute that must NOT be used to build it).
_CASES = [
    # Isaac Gym and Genesis delegate to the one base implementation, which is checked in their place
    pytest.param("base.py", "BaseSimHandler", "_action_spans", "_joint_info", id="base"),  # the slice per robot
    pytest.param("pybullet/pybullet.py", "SinglePybulletHandler", "_get_states", "object_joint_order", id="pybullet"),
]


@pytest.mark.general
@pytest.mark.parametrize("rel_path,class_name,func_name,native_attr", _CASES)
def test_joint_pos_target_uses_sorted_joint_order(rel_path: str, class_name: str, func_name: str, native_attr: str):
    """The ``joint_pos_target`` materializer must iterate sorted joint names.

    Fails if a backend reverts to iterating its native joint order, which would
    re-open the silent ``joint_pos_target``/``joint_pos`` index misalignment
    fixed for sapien2/genesis in 92755f6 and here for isaacgym/pybullet.
    """
    source = _SIM_ROOT.joinpath(rel_path).read_text(encoding="utf-8")
    fn = _find_function(ast.parse(source), class_name, func_name)

    assert _calls_get_joint_names_sorted(fn), (
        f"{class_name}.{func_name} must build joint_pos_target from "
        f"_get_joint_names(..., sort=True) so it aligns with joint_pos "
        f"(sorted-name order); no such call found."
    )
    assert not _references_attr(fn, native_attr), (
        f"{class_name}.{func_name} still references native joint order "
        f"({native_attr!r}) — joint_pos_target[i] would refer to a different "
        f"joint than joint_pos[i] whenever the native order is not alphabetical."
    )


@pytest.mark.general
def test_genesis_and_isaacgym_read_the_target_from_the_base_helper():
    """Their own materialisers (native joint order, None for tensor actions) are gone; a re-inlined copy
    would escape the base-helper check above."""
    for rel in ("genesis/genesis.py", "isaacgym/isaacgym.py"):
        source = _SIM_ROOT.joinpath(rel).read_text(encoding="utf-8")
        assert "_joint_pos_target_from_action_cache(" in source, rel
        assert "def _joint_pos_target_from_cache" not in source, rel


@pytest.mark.general
def test_every_backend_reports_no_position_target_for_an_effort_driven_robot():
    """One predicate from the robot config gates ``joint_pos_target`` at every backend's report site: an
    effort-driven robot's applied tensor (or ``ctrl``) is a torque and must not be reported as a position."""
    from types import SimpleNamespace

    from metasim.sim.base import BaseSimHandler

    class _H(BaseSimHandler):
        def _set_states(self, states, env_ids=None):
            pass

        def _set_dof_targets(self, actions):
            pass

        def _get_states(self, env_ids=None):
            return None

        def _simulate(self):
            pass

    h = _H.__new__(_H)
    h.object_dict = {
        "pos": SimpleNamespace(control_type={"j1": "position"}),
        "mixed": SimpleNamespace(control_type={"j1": "position", "j2": "effort"}),
        "none": SimpleNamespace(control_type=None),
    }
    assert h._robot_reports_position_target("pos") and h._robot_reports_position_target("none")
    assert not h._robot_reports_position_target("mixed")
    for rel in (
        "mujoco/mujoco.py",
        "mjx/mjx.py",
        "sapien/sapien3.py",
        "sapien/sapien2.py",
        "pybullet/pybullet.py",
        "isaacsim/isaacsim.py",
        "newton/newton.py",
        "superdex/superdex.py",
    ):
        assert "_robot_reports_position_target(" in _SIM_ROOT.joinpath(rel).read_text(encoding="utf-8"), rel
    assert "_robot_reports_position_target(" in _SIM_ROOT.joinpath("base.py").read_text(encoding="utf-8")
