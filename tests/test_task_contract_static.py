"""The task contract, checked without importing a simulator: what every registered task must look like in source.

- one registration per class, every registered name unique across the packs;
- a registered class has a class-level ``scenario`` (or inherits one), so the harness and the gate can build it;
- a module registers at most ``MAX_CLASSES_PER_MODULE`` classes: variants are instances or parameters, not
  generated classes (two generated families accounted for 2587 of the 3021 classes retired on 2026-11-27);
- a ``traj_filepath`` literal is a ``roboverse_data/...`` path (the Hub check in ``tools/assets/check_manifest.py``
  verifies that it exists).
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections import Counter, defaultdict

import pytest

TASKS = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks"
MAX_CLASSES_PER_MODULE = 16
pytestmark = pytest.mark.general


def _registrations():
    """[(module path, class name, (registered names), class node, module tree)] for every @register_task class."""
    out = []
    for py in sorted(TASKS.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as err:  # pragma: no cover - a syntax error is its own failure
            pytest.fail(f"{py}: {err}")
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for d in node.decorator_list:
                func = d.func if isinstance(d, ast.Call) else d
                name = getattr(func, "id", getattr(func, "attr", ""))
                if name == "register_task" and isinstance(d, ast.Call):
                    names = tuple(a.value for a in d.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
                    out.append((py.relative_to(TASKS).as_posix(), node.name, names, node, tree))
    return out


REGISTRATIONS = _registrations()


def test_there_are_registered_tasks():
    assert len(REGISTRATIONS) >= 40, "the content packs register fewer tasks than the curated baseline"


def test_registered_names_are_unique():
    seen = Counter(n for _, _, names, _, _ in REGISTRATIONS for n in names)
    dupes = sorted(n for n, c in seen.items() if c > 1)
    assert not dupes, f"registered names used by more than one class: {dupes}"


def test_a_class_is_registered_once():
    per_class = Counter((m, c) for m, c, _, _, _ in REGISTRATIONS)
    twice = sorted(f"{m}:{c}" for (m, c), n in per_class.items() if n > 1)
    assert not twice, f"classes with several @register_task decorators: {twice}"


def test_no_generated_variant_families():
    per_module = Counter(m for m, _, _, _, _ in REGISTRATIONS)
    big = {m: n for m, n in per_module.items() if n > MAX_CLASSES_PER_MODULE}
    assert not big, (
        f"modules registering more than {MAX_CLASSES_PER_MODULE} classes (make variants parameters, not classes): {big}"
    )


def _class_assigns(node: ast.ClassDef) -> set[str]:
    names = set()
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            names |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    return names


def test_registered_classes_have_a_scenario_in_source_or_a_base_in_the_pack():
    """A registered class defines ``scenario`` itself, or inherits from a class defined in the packs (which the
    runtime contract test checks); a bare class with neither cannot be built by `bench run`."""
    pack_classes = {c for _, c, _, _, _ in REGISTRATIONS}
    all_defs = defaultdict(set)
    for py in TASKS.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            for node in ast.parse(py.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.ClassDef):
                    all_defs[node.name] |= _class_assigns(node)
        except SyntaxError:
            continue
    bare = []
    for m, c, names, node, _ in REGISTRATIONS:
        bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
        has = "scenario" in _class_assigns(node) or any("scenario" in all_defs.get(b, set()) for b in bases)
        if not has and not bases & (pack_classes | set(all_defs)):
            bare.append(f"{m}:{c} {names}")
    assert not bare, "registered classes without a class-level scenario or a pack base class:\n  " + "\n  ".join(bare)


def test_traj_filepath_literals_point_into_roboverse_data():
    bad = []
    for m, c, _, node, _ in REGISTRATIONS:
        for stmt in node.body:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                targets = [stmt.target.id]
            if (
                "traj_filepath" in targets
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                if not re.match(r"roboverse_data/", stmt.value.value):
                    bad.append(f"{m}:{c} -> {stmt.value.value!r}")
    assert not bad, "traj_filepath must be a roboverse_data/... path:\n  " + "\n  ".join(bad)
