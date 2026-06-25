"""Backend-free conformance guardrail: every Tier-1 task ``reset`` accepts ``seed``.

The gym bridge (``metasim.task.gym_registration``) forwards ``env.reset(seed=)``
to a task only when the task's ``reset`` signature accepts it — tasks that
override ``reset`` without a ``seed`` parameter silently drop the seed, so
rollouts are not reproducible on the simulator side.

This test statically (no sim backend, no instantiation) scans every task file
and asserts the contract on **Tier-1 tasks only**:

* **Tier 1** — classes in the unified contract: they (transitively) subclass
  ``BaseTaskEnv`` / ``RLTaskEnv`` / ``ManagerBasedRVEnv``. Only these are checked.
* **Tier 3** — external/native passthrough adapters (modules under ``_native``/
  ``_passthrough`` or classes named ``Native*``/``Passthrough*``) are an
  explicitly-exempt compatibility tier and are skipped.
* Non-task helpers (controllers, sensors, command/actuator managers, success
  evaluators, sessions) are not in the inheritance graph to ``BaseTaskEnv`` and
  are therefore ignored.

Assertions:

* no *new* Tier-1 ``reset`` override lacks ``seed`` beyond the P1-cleanup
  allowlist (ratchet: the set may only shrink);
* the allowlist has no stale entries (forces it to shrink as P1 lands);
* the unified family base classes already honor the contract.

When you fix a file's ``reset`` to accept ``seed`` (or stop overriding
``reset``), delete it from ``KNOWN_NO_SEED_RESET`` below.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_TASKS = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks"

# Terminal in-contract roots: a class is Tier-1 if it transitively subclasses one
# of these. BaseTaskEnv/RLTaskEnv live in MetaSim; ManagerBasedRVEnv in
# roboverse_learn — they are not defined under tasks/, so they anchor the graph.
_CONTRACT_ROOTS = {"BaseTaskEnv", "RLTaskEnv", "ManagerBasedRVEnv"}

# Family base classes fixed in the reset(seed) unification — must stay conformant.
_FIXED_BASES = {
    "libero/libero_base.py",
    "libero_90/libero_90_base.py",
    "maniskill/maniskill_base.py",
    "rlbench/rl_bench.py",
    "embodiedgen/base.py",
    "calvin/base_table.py",
    "pick_place/base.py",
    "humanoid/base/base_legged_robot.py",
}

# P1 cleanup targets: Tier-1 task files whose ``reset`` override still drops
# ``seed``. The contract is now COMPLETE — every Tier-1 task accepts ``seed`` —
# so this is empty and the guardrail is zero-tolerance. It may only ever be
# extended with a NEW genuine violator pending its fix; prefer fixing instead.
KNOWN_NO_SEED_RESET = frozenset()


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _build_class_graph() -> dict[str, set[str]]:
    """Map every class defined under tasks/ to its (immediate) base-class names."""
    graph: dict[str, set[str]] = {}
    for path in _TASKS.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            graph.setdefault(cls.name, set()).update(b for b in (_base_name(x) for x in cls.bases) if b is not None)
    return graph


def _in_contract(name: str, graph: dict[str, set[str]], seen: set[str] | None = None) -> bool:
    """True if ``name`` transitively subclasses a contract root (Tier-1)."""
    if name in _CONTRACT_ROOTS:
        return True
    seen = seen or set()
    if name in seen or name not in graph:
        return False
    seen.add(name)
    return any(_in_contract(b, graph, seen) for b in graph[name])


def _is_tier3(rel_path: str, class_name: str) -> bool:
    """Native/passthrough adapters are an explicitly-exempt compatibility tier."""
    p = "/" + rel_path
    if "/_native/" in p or "_passthrough" in rel_path:
        return True
    return class_name.startswith(("Native", "Passthrough"))


def _reset_lacks_seed(fn: ast.FunctionDef) -> bool:
    a = fn.args
    names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    return "seed" not in names and a.kwarg is None  # no seed= and no **kwargs


def _all_violators() -> set[str]:
    graph = _build_class_graph()
    out: set[str] = set()
    for path in _TASKS.rglob("*.py"):
        rel = path.relative_to(_TASKS).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if _is_tier3(rel, cls.name) or not _in_contract(cls.name, graph):
                continue
            for fn in cls.body:
                if (
                    isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and fn.name == "reset"
                    and _reset_lacks_seed(fn)
                ):
                    out.add(rel)
    return out


@pytest.mark.general
def test_no_new_reset_without_seed():
    """No Tier-1 task may add a ``reset`` override that drops ``seed`` (ratchet)."""
    new = sorted(_all_violators() - KNOWN_NO_SEED_RESET)
    assert not new, (
        "These Tier-1 task files override reset() without a seed parameter, breaking "
        "the env.reset(seed=) contract. Add seed=None and forward it to super().reset:\n  " + "\n  ".join(new)
    )


@pytest.mark.general
def test_reset_seed_allowlist_has_no_stale_entries():
    """Allowlisted files must still be violators — forces the list to shrink."""
    stale = sorted(KNOWN_NO_SEED_RESET - _all_violators())
    assert not stale, (
        "These files no longer violate the reset(seed) contract — remove them from "
        "KNOWN_NO_SEED_RESET:\n  " + "\n  ".join(stale)
    )


@pytest.mark.general
def test_unified_bases_accept_seed():
    """The fixed family base classes must keep accepting seed (regression guard)."""
    regressed = []
    for rel in sorted(_FIXED_BASES):
        tree = ast.parse((_TASKS / rel).read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "reset" and _reset_lacks_seed(fn):
                    regressed.append(rel)
    assert not regressed, f"Base classes regressed to reset() without seed: {sorted(set(regressed))}"
