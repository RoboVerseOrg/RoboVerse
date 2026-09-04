"""Every first-party import in the content tree resolves: the module exists and defines the imported names.

Function-local imports are not executed at import time, so a module that moved (``il/act`` became
``il/policies/act``) breaks only when the branch runs, typically deep inside an evaluation with
domain randomization. Resolving the targets statically, without importing anything, catches the
stale path in CI regardless of which optional dependencies are installed.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("roboverse_learn", "roboverse_pack", "scripts", "examples", "tools")
PACKAGE_DIRS = {
    "roboverse_learn": ROOT / "roboverse_learn",
    "roboverse_pack": ROOT / "roboverse_pack",
    "metasim": ROOT / "packages" / "metasim" / "metasim",
}


def _module_path(module: str) -> Path | None:
    """``pkg.a.b`` -> ``<pkg dir>/a/b.py`` or the package directory ``<pkg dir>/a/b``; None if absent."""
    parts = module.split(".")
    base = PACKAGE_DIRS.get(parts[0])
    if base is None:
        return None
    rel = base.joinpath(*parts[1:]) if len(parts) > 1 else base
    if rel.is_dir():  # a package wins over a same-named module, as in Python
        return rel
    if rel.with_suffix(".py").is_file():
        return rel.with_suffix(".py")
    return None


def _names_in(tree_body) -> set[str] | None:
    """Names bound at the top level of a module body, recursing into every compound statement
    (try / if / for / while / with / match); None when the module binds names in a way that cannot be
    followed statically (``from x import *``, ``globals()[...] = ...``, a module ``__getattr__``)."""
    names: set[str] = set()
    for node in tree_body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "__getattr__":
                return None
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "globals" for n in ast.walk(target)):
                    return None
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
            if any(isinstance(n, ast.Name) and n.id == "__all__" for t in targets for n in ast.walk(t)):
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple)):  # ``__all__`` is a declaration of exports
                    names.update(
                        e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                return None
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        else:
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                for child in getattr(node, field, None) or []:
                    inner = getattr(child, "body", None) if not isinstance(child, ast.stmt) else [child]
                    sub = _names_in(inner or [])
                    if sub is None:
                        return None
                    names |= sub
            if isinstance(node, (ast.For, ast.AsyncFor)):
                names.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            for n in ast.walk(node):
                if isinstance(n, ast.NamedExpr):
                    names.add(n.target.id)
    return names


@functools.cache
def _defined_names(path: Path) -> set[str] | None:
    """Top-level names a module (or package) defines; None when a ``import *`` makes that unknowable."""
    submodules: set[str] = set()
    if path.is_dir():
        submodules = {p.stem for p in path.iterdir() if (p.suffix == ".py" and p.stem != "__init__") or p.is_dir()}
        path = path / "__init__.py"
        if not path.is_file():
            return submodules
    try:
        names = _names_in(ast.parse(path.read_text(encoding="utf-8")).body)
    except SyntaxError:
        return None
    return None if names is None else names | submodules


def _first_party_imports() -> list[tuple[str, int, str, tuple[str, ...]]]:
    found = []
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            rel = str(path.relative_to(ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    if node.module.split(".")[0] in PACKAGE_DIRS:
                        found.append((rel, node.lineno, node.module, tuple(a.name for a in node.names)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in PACKAGE_DIRS:
                            found.append((rel, node.lineno, alias.name, ()))
    return found


@pytest.mark.parametrize(("file", "line", "module", "names"), _first_party_imports(), ids=lambda v: str(v))
def test_first_party_import_targets_exist(file, line, module, names):
    path = _module_path(module)
    assert path is not None, f"{file}:{line}: module {module!r} does not exist on disk"
    if "*" in names:
        return
    defined = _defined_names(path)
    if defined is None:
        return  # the module re-exports with ``import *``; its names cannot be resolved statically
    missing = [n for n in names if n not in defined]
    assert not missing, f"{file}:{line}: {module!r} defines no {missing}"


def test_resolution_rejects_a_removed_module_and_accepts_moved_one():
    assert _module_path("roboverse_learn.il.act.act_eval_runner") is None
    assert _module_path("roboverse_learn.il.act") is None
    assert _module_path("roboverse_learn.il.policies.act.act_eval_runner") is not None
    assert "ensure_clean_state" in _defined_names(_module_path("roboverse_learn.il.utils.clean_state"))
    assert _module_path("metasim.utils.state") is not None
    assert _names_in(ast.parse("for n in X:\n    FOO = 1").body) == {"n", "FOO"}
    assert _names_in(ast.parse("globals()['G'] = 1").body) is None
    assert _names_in(ast.parse("if a:\n    A = 1\nelse:\n    B = 2").body) == {"A", "B"}
