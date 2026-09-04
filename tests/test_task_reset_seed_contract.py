"""Backend-free conformance guardrail: every Tier-1 task ``reset`` is substitutable for the base.

The base contract is ``BaseTaskEnv.reset(states=None, env_ids=None, seed=None)``, and callers rely
on the *whole* signature, not just one parameter:

* ``states`` — the IL/VLA eval runners replay demo initial states with
  ``env.reset(states=...)`` (``roboverse_learn/il/runners/default_runner.py``,
  ``roboverse_learn/vla/{OpenVLA,SmolVLA,pi0}/*_eval.py``). An override that drops ``states``
  makes those runners die with ``TypeError: reset() got an unexpected keyword argument 'states'``.
* ``env_ids`` — partial resets.
* ``seed`` — the gym bridge (``metasim.task.gym_registration``) forwards ``env.reset(seed=)`` to a
  task only when its signature accepts it, so an override without ``seed`` silently drops it and
  rollouts are not reproducible on the simulator side.

This guardrail originally checked ``seed`` alone; eight Tier-1 overrides passed it while dropping
``states`` (six SimplerEnv tasks took ``options`` in its place). It now checks full
substitutability — accepted parameters, their positional order, and that no override adds a
*required* parameter of its own — so that class of drift cannot recur.

This test statically (no sim backend, no instantiation) scans every task file
and asserts the contract on **Tier-1 tasks only**:

* **Tier 1** — classes in the unified contract: they (transitively) subclass
  ``BaseTaskEnv`` / ``RLTaskEnv`` / ``ManagerBasedRVEnv``. Only these are checked.
* **Tier 3** — external/native passthrough adapters under dedicated ``_native``/
  ``_passthrough`` paths are an explicitly-exempt compatibility tier and skipped.
  (Exemption is PATH-based only — a first-party in-contract task is checked even
  if its class name starts with ``Native``; see ``_is_tier3``.)
* Non-task helpers (controllers, sensors, command/actuator managers, success
  evaluators, sessions) are not in the inheritance graph to ``BaseTaskEnv`` and
  are therefore ignored.

Assertions (ratchets — each allowlist may only SHRINK, and may not go stale):

* ``reset`` — no *new* Tier-1 override breaks substitutability (allowlist now empty →
  zero-tolerance); the unified family base classes honor the contract.
* ``step`` — no *new* Tier-1 override (action transforms belong in a pre-physics
  action hook / ``pre_physics_step_callback``); current overrides are tracked
  in ``KNOWN_STEP_OVERRIDES`` as P1-cleanup targets.
* ``close`` — no *new* Tier-1 override (teardown belongs in ``close_callback``).

When you fix a file (honor the reset contract / drop the ``step``/``close`` override),
delete it from the corresponding allowlist below.

A task may still take *extra* parameters (e.g. SimplerEnv's ``options`` episode spec) as long as
they come after the contract's three and have defaults, so a plain ``reset(states=...)`` call works.
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

# The base signature: BaseTaskEnv.reset(states=None, env_ids=None, seed=None). Order matters —
# callers may pass these positionally.
_CONTRACT_PARAMS = ("states", "env_ids", "seed")

# Family base classes / tasks fixed in the reset-contract unification — must stay conformant.
_FIXED_BASES = {
    "libero/libero_base.py",
    "libero_90/libero_90_base.py",
    "maniskill/maniskill_base.py",
    "rlbench/rl_bench.py",
    "embodiedgen/base.py",
    "calvin/base_table.py",
    "pick_place/base.py",
    "humanoid/base/base_legged_robot.py",
    "beyondmimic/metasim/envs/base_legged_robot.py",
    # SimplerEnv: took `options` in place of `states`, so VLA eval on SimplerEnv — the canonical
    # pairing for that benchmark — was a hard TypeError. They keep `options` after the contract's
    # three parameters.
    "simpler_env/_metasim/coke_task.py",
    "simpler_env/_metasim/drawer_task.py",
    "simpler_env/_metasim/grasp_objects_task.py",
    "simpler_env/_metasim/place_task.py",
    "simpler_env/_metasim/put_on_task.py",
}

# P1 cleanup targets: Tier-1 task files whose ``reset`` override is not substitutable for the base
# signature. The contract is now COMPLETE — every Tier-1 task accepts states/env_ids/seed — so this
# is empty and the guardrail is zero-tolerance. It may only ever be extended with a NEW genuine
# violator pending its fix; prefer fixing instead.
KNOWN_NONCONFORMANT_RESET = frozenset()

# The BaseTaskEnv contract says tasks should NOT override step()/close() — action
# transforms belong in a pre-physics action hook, teardown in close_callback. These
# allowlists are the current P1-cleanup targets; they may only SHRINK. The ratchet
# blocks any NEW Tier-1 step()/close() override so the architecture cannot degrade
# while the existing ones are migrated. Remove an entry when you drop its override.
KNOWN_STEP_OVERRIDES = frozenset({
    "libero/native_libero.py",  # native LIBERO replay port (Tier-1 by base, kept explicit)
    "mujoco_playground/open_cabinet.py",
    "mujoco_playground/pick.py",
    "beyondmimic/metasim/envs/base_legged_robot.py",
    "calvin/base_table.py",
    "humanoid/base/base_legged_robot.py",
    "mjlab/cartpole_train.py",
    "mujoco_playground/handover.py",
    "pick_place/approach_grasp.py",
    "pick_place/approach_grasp_ceramic_teapot.py",
    "pick_place/approach_grasp_knife.py",
    "pick_place/base.py",
    "pick_place/hand_trajectory.py",
    "pick_place/track_banana.py",
    "pick_place/track_ceramic_teapot.py",
    "pick_place/track_knife.py",
    "pick_place/track_screwdriver.py",
    "pick_place/track_spoon.py",
    "robosuite/robosuite_env.py",
    "simpler_env/_metasim/base.py",
    "simpler_env/_metasim/coke_task.py",
    "simpler_env/_metasim/place_task.py",
})
KNOWN_CLOSE_OVERRIDES = frozenset({
    "libero/native_libero.py",
    "robosuite/robosuite_env.py",
})


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
    """Native/passthrough adapters are an explicitly-exempt compatibility tier.

    Exemption is PATH-based only (dedicated ``_native``/``_passthrough`` dirs/files).
    A NAME-based exemption (``Native*``/``Passthrough*``) was removed: it let
    first-party in-contract tasks (e.g. ``NativeLiberoEnv(BaseTaskEnv)``, which
    lives in ``libero/native_libero.py`` — not a ``_native/`` dir) silently dodge
    the contract. In-contract classes are now always checked regardless of name;
    only genuine adapters under ``_native``/``_passthrough`` paths are exempt.
    """
    p = "/" + rel_path
    return "/_native/" in p or "_passthrough" in rel_path


def _reset_violations(fn: ast.FunctionDef) -> list[str]:
    """Reasons ``fn`` is not substitutable for ``reset(states=None, env_ids=None, seed=None)``.

    Three ways an override breaks a caller of the base signature:

    1. it does not accept a contract parameter at all (``reset(states=...)`` → TypeError);
    2. it accepts them positionally in a different order (a positional caller silently binds the
       wrong value — worse than a TypeError, because it corrupts instead of failing);
    3. it adds a parameter of its own with no default (``reset(states=...)`` → TypeError for the
       missing argument).
    """
    a = fn.args
    positional = [p.arg for p in (*a.posonlyargs, *a.args)][1:]  # drop self
    kwonly = [p.arg for p in a.kwonlyargs]
    names = [*positional, *kwonly]
    bad: list[str] = []

    # (1) accepted at all? **kwargs absorbs anything, so it satisfies the contract.
    if a.kwarg is None:
        missing = [p for p in _CONTRACT_PARAMS if p not in names]
        if missing:
            bad.append(f"does not accept {', '.join(missing)}")

    # (2) contract params that are positional must keep the base's relative order.
    ordered = [p for p in positional if p in _CONTRACT_PARAMS]
    if ordered != [p for p in _CONTRACT_PARAMS if p in ordered]:
        bad.append(f"positional order is {tuple(ordered)}, must follow {_CONTRACT_PARAMS}")

    # (3) any extra parameter must have a default (positional defaults bind to the tail).
    n_pos_defaults = len(a.defaults)
    required_positional = positional[: len(positional) - n_pos_defaults]
    required_kwonly = [p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=False) if d is None]
    extra_required = [p for p in (*required_positional, *required_kwonly) if p not in _CONTRACT_PARAMS]
    if extra_required:
        bad.append(f"adds required parameter(s) {', '.join(extra_required)} (give them defaults)")

    return bad


def _all_violators() -> dict[str, list[str]]:
    """Tier-1 task files whose ``reset`` override is not substitutable → why."""
    graph = _build_class_graph()
    out: dict[str, list[str]] = {}
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
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "reset":
                    bad = _reset_violations(fn)
                    if bad:
                        out.setdefault(rel, []).extend(f"{cls.name}.reset {b}" for b in bad)
    return out


def _files_overriding(method: str) -> set[str]:
    """Tier-1 task files (Tier-3 exempt) that define a class method named ``method``."""
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
            if any(isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == method for fn in cls.body):
                out.add(rel)
    return out


@pytest.mark.general
def test_no_new_nonconformant_reset():
    """No Tier-1 task may add a ``reset`` override that is not substitutable for the base (ratchet).

    Covers all three contract parameters. ``states`` is the one that bit us: the IL and VLA eval
    runners call ``env.reset(states=...)`` unconditionally, so an override that drops it turns
    every evaluation on that task into a TypeError.
    """
    violators = _all_violators()
    new = sorted(set(violators) - KNOWN_NONCONFORMANT_RESET)
    detail = "\n  ".join(f"{rel}: {'; '.join(violators[rel])}" for rel in new)
    assert not new, (
        "These Tier-1 task files override reset() incompatibly with the base contract "
        "reset(states=None, env_ids=None, seed=None). Accept all three (extra params go last, with "
        "defaults) and forward them to super().reset:\n  " + detail
    )


@pytest.mark.general
def test_reset_allowlist_has_no_stale_entries():
    """Allowlisted files must still be violators — forces the list to shrink."""
    stale = sorted(KNOWN_NONCONFORMANT_RESET - set(_all_violators()))
    assert not stale, (
        "These files no longer violate the reset() contract — remove them from "
        "KNOWN_NONCONFORMANT_RESET:\n  " + "\n  ".join(stale)
    )


@pytest.mark.general
def test_unified_bases_honor_reset_contract():
    """The fixed family bases/tasks must keep the full reset signature (regression guard)."""
    regressed = []
    for rel in sorted(_FIXED_BASES):
        tree = ast.parse((_TASKS / rel).read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in cls.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == "reset":
                    regressed += [f"{rel}: {cls.name}.reset {b}" for b in _reset_violations(fn)]
    assert not regressed, "Base classes regressed off the reset contract:\n  " + "\n  ".join(sorted(regressed))


@pytest.mark.general
@pytest.mark.parametrize(
    ("sig", "conformant"),
    [
        ("def reset(self, states=None, env_ids=None, seed=None): ...", True),
        # SimplerEnv: extra episode-spec param, after the contract's three, with a default.
        ("def reset(self, states=None, env_ids=None, seed=0, options=None): ...", True),
        ("def reset(self, states=None, env_ids=None, **kwargs): ...", True),  # **kwargs absorbs seed
        ("def reset(self, env_ids=None, seed=None): ...", False),  # legged-robot bases, pre-fix
        ("def reset(self, env_ids=None, options=None, seed=0): ...", False),  # SimplerEnv, pre-fix
        ("def reset(self, env_ids=None, states=None, seed=None): ...", False),  # swapped positionals
        # extra param with no default: reset(states=...) would TypeError on the missing argument
        ("def reset(self, states=None, env_ids=None, seed=None, *, options): ...", False),
    ],
)
def test_reset_violation_checker(sig, conformant):
    """The checker itself — it must reject exactly the signatures the seed-only guardrail let pass."""
    fn = ast.parse(sig).body[0]
    assert (not _reset_violations(fn)) is conformant


@pytest.mark.general
def test_no_new_step_override():
    """No Tier-1 task may add a new step() override (ratchet; use the action hook)."""
    new = sorted(_files_overriding("step") - KNOWN_STEP_OVERRIDES)
    assert not new, (
        "These Tier-1 task files add a step() override (discouraged — transform actions in a "
        "pre-physics action hook / pre_physics_step_callback instead):\n  " + "\n  ".join(new)
    )


@pytest.mark.general
def test_step_override_allowlist_has_no_stale_entries():
    """Allowlisted step() overrides must still exist — forces the list to shrink."""
    stale = sorted(KNOWN_STEP_OVERRIDES - _files_overriding("step"))
    assert not stale, (
        "These files no longer override step() — remove them from KNOWN_STEP_OVERRIDES:\n  " + "\n  ".join(stale)
    )


@pytest.mark.general
def test_no_new_close_override():
    """No Tier-1 task may add a new close() override (ratchet; use close_callback)."""
    new = sorted(_files_overriding("close") - KNOWN_CLOSE_OVERRIDES)
    assert not new, (
        "These Tier-1 task files add a close() override (use close_callback for teardown):\n  " + "\n  ".join(new)
    )


@pytest.mark.general
def test_close_override_allowlist_has_no_stale_entries():
    """Allowlisted close() overrides must still exist — forces the list to shrink."""
    stale = sorted(KNOWN_CLOSE_OVERRIDES - _files_overriding("close"))
    assert not stale, (
        "These files no longer override close() — remove them from KNOWN_CLOSE_OVERRIDES:\n  " + "\n  ".join(stale)
    )


def _unseeded_generator_calls(path: pathlib.Path) -> list[str]:
    """``np.random.default_rng()`` / ``RandomState()`` / ``Generator`` built without a seed in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in ("default_rng", "RandomState"):
            continue
        seeded = any(not (isinstance(a, ast.Constant) and a.value is None) for a in node.args) or any(
            k.arg == "seed" and not (isinstance(k.value, ast.Constant) and k.value is None) for k in node.keywords
        )
        if not seeded:
            hits.append(f"{path.relative_to(_TASKS.parent.parent)}:{node.lineno}")
    return hits


@pytest.mark.general
def test_no_task_builds_an_unseeded_numpy_generator():
    """Task reset / DR noise must come from the RNG ``handler.set_seed`` seeds (global ``np.random``
    or a generator seeded from it). A fresh ``np.random.default_rng()`` ignores ``reset(seed=N)`` and
    makes the episode unreproducible while the seed plumbing looks correct (eight mjlab samplers did
    this; ``cartpole_v2`` documents the fix)."""
    offenders = [hit for path in sorted(_TASKS.rglob("*.py")) for hit in _unseeded_generator_calls(path)]
    assert not offenders, f"unseeded NumPy generators ignore reset(seed=): {offenders}"
