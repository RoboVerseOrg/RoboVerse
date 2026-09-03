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
    """True if ``fn`` calls ``*._get_joint_names(...)`` with ``sort=True``.

    Accepts either the keyword form ``sort=True`` or the positional form
    ``_get_joint_names(obj_name, True)`` — both mean sorted order.
    """
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "_get_joint_names":
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
    pytest.param(
        "isaacgym/isaacgym.py", "IsaacgymHandler", "_joint_pos_target_from_cache", "_joint_info", id="isaacgym"
    ),
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
