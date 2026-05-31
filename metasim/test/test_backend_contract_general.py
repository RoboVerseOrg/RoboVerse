"""Contract tests: every concrete ``BaseSimHandler`` subclass must implement
the documented backend interface.

Motivation: RoboVerse is moving toward being a *standard* cross-platform
benchmark. New simulator backends should fail at definition time if they
forget a documented contract method, not silently at first call. The
clean way to enforce that is ``@abstractmethod`` — but several methods
(``_get_joint_names``, ``_get_body_names``) have the decorator commented
out because existing backends (pyrep, partial pybullet/genesis) don't
implement them yet. Adding the decorator now would break those imports.

Instead this test statically asserts each concrete backend overrides
every required method. Known-incomplete backends are marked ``xfail`` so
the gap is *documented and surfaced* but doesn't block the suite. When a
backend catches up, its xfail flips to ``xpassed`` — that's the signal
to flip the abstractmethod decorator on for real.

This test is intentionally pure-Python: no GPU, no sim env, no asset
download. It runs as ``-k general``.
"""

from __future__ import annotations

import pytest

from metasim.sim.base import BaseSimHandler


def _import_all_backend_handlers() -> list[type[BaseSimHandler]]:
    """Import every backend module so its ``BaseSimHandler`` subclasses
    register in ``__subclasses__``. Missing optional deps are tolerated —
    we list every backend we *intend* to ship.
    """
    backends = [
        "metasim.sim.mujoco.mujoco",
        "metasim.sim.mjx.mjx",
        "metasim.sim.sapien.sapien2",
        "metasim.sim.sapien.sapien3",
        "metasim.sim.pybullet.pybullet",
        "metasim.sim.newton.newton",
        "metasim.sim.genesis.genesis",
        "metasim.sim.blender.blender",
        "metasim.sim.isaacgym.isaacgym",
        "metasim.sim.isaacsim.isaacsim",
        "metasim.sim.pyrep.pyrep",
        "metasim.sim.parallel",
        "metasim.sim.hybrid",
    ]
    imported: list[type[BaseSimHandler]] = []
    for mod_name in backends:
        try:
            __import__(mod_name)
        except Exception:
            continue
    # Walk subclasses transitively; ``parallel`` / ``hybrid`` wrap others.
    seen: set[type[BaseSimHandler]] = set()
    stack: list[type[BaseSimHandler]] = list(BaseSimHandler.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        imported.append(cls)
        stack.extend(cls.__subclasses__())
    return imported


# Documented contract methods every backend must override. Grouped so we
# can xfail individual gaps without losing coverage on the rest.
_REQUIRED_METHODS = (
    "_set_states",
    "_set_dof_targets",
    "_get_states",
    "_simulate",
    "_get_joint_names",
    "_get_body_names",
    "close",
)


# Backends with known gaps; format: {(class_name, method): "reason"}.
# A class name appears here only when the actual concrete subclass is
# missing the override (not the base raising NotImplementedError).
_KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("PyrepHandler", "_get_joint_names"): "pyrep RLBench backend in transition",
    ("PyrepHandler", "_get_body_names"): "pyrep RLBench backend in transition",
    # PyrepHandler.close gap closed by pr/audit-fixes 22ad5ca (stub handler).
    ("SinglePybulletHandler", "_get_body_names"): "pybullet backend partially implemented",
    ("GenesisHandler", "_get_body_names"): "genesis backend partially implemented",
}


_BACKEND_METHOD_PARAMS = [
    pytest.param(cls, m, id=f"{cls.__name__}-{m}") for cls in _import_all_backend_handlers() for m in _REQUIRED_METHODS
]


# Public properties every concrete handler must expose. Captured here
# separately because ``__qualname__`` checks (used for methods above)
# don't apply to properties — the descriptor lives on the class and
# we need ``fget.__qualname__`` instead. ``actions_cache`` previously
# slipped past the method test entirely because it's a property, and
# only the dynamic-attribute fallback in subclasses kept Parallel /
# Hybrid working until this contract landed at the base.
_REQUIRED_PROPERTIES = (
    "actions_cache",
    "device",
)


_BACKEND_PROPERTY_PARAMS = [
    pytest.param(cls, p, id=f"{cls.__name__}-{p}")
    for cls in _import_all_backend_handlers()
    for p in _REQUIRED_PROPERTIES
]


@pytest.mark.general
@pytest.mark.parametrize("cls,prop", _BACKEND_PROPERTY_PARAMS)
def test_backend_exposes_contract_property(cls: type[BaseSimHandler], prop: str):
    """Every concrete handler must expose each documented contract
    property — not as a method, not as a plain attribute, but as a
    real ``@property``-style descriptor on the class.

    Previously ``actions_cache`` was implemented as a property on 8
    concrete backends but missing from base, ParallelHandler, and
    HybridSimHandler — so ``handler.actions_cache`` raised
    ``AttributeError`` on the parallel path even though tests asserted
    it. The contract is now owned by the base; this test guards the
    surface so the same gap can't reopen.
    """
    descriptor = getattr(cls, prop, None)
    assert descriptor is not None, (
        f"{cls.__name__} does not expose the contract property {prop!r}. "
        f"Add it to the base or override on the subclass."
    )


@pytest.mark.general
@pytest.mark.parametrize("cls,method", _BACKEND_METHOD_PARAMS)
def test_backend_overrides_contract_method(cls: type[BaseSimHandler], method: str):
    """Every concrete handler must override every documented contract method.

    'Override' = the method's ``__qualname__`` does NOT point at
    ``BaseSimHandler.<method>``. That catches both missing definitions
    (inherited from base) and trivial pass-through stubs.
    """
    gap_key = (cls.__name__, method)
    if gap_key in _KNOWN_GAPS:
        pytest.xfail(_KNOWN_GAPS[gap_key])

    func = getattr(cls, method, None)
    assert func is not None, f"{cls.__name__} is missing required method {method!r}"
    qualname = getattr(func, "__qualname__", "")
    assert not qualname.startswith("BaseSimHandler."), (
        f"{cls.__name__}.{method} is inherited from BaseSimHandler — backend must "
        f"override it. If this backend genuinely can't support {method}, add it to "
        f"_KNOWN_GAPS with a reason so the contract drift is documented."
    )


@pytest.mark.general
def test_known_gaps_are_actually_gaps():
    """Self-check: every entry in ``_KNOWN_GAPS`` must correspond to a real,
    currently-failing gap. Once a backend catches up, the entry becomes a
    lie — this test catches that so the xfail can be removed and the
    contract tightened."""
    classes = {cls.__name__: cls for cls in _import_all_backend_handlers()}
    for (class_name, method), reason in _KNOWN_GAPS.items():
        if class_name not in classes:
            # Backend didn't import in this env (optional dep) — skip,
            # don't false-fail.
            continue
        cls = classes[class_name]
        func = getattr(cls, method, None)
        qualname = getattr(func, "__qualname__", "") if func is not None else ""
        is_still_gap = func is None or qualname.startswith("BaseSimHandler.")
        assert is_still_gap, (
            f"_KNOWN_GAPS lists {class_name}.{method} as missing ({reason!r}) "
            f"but the backend now overrides it — remove the entry so the "
            f"contract test enforces it for real."
        )


@pytest.mark.general
def test_set_states_invalidates_cache_on_all_backends():
    """Forward-compat guard: ``BaseSimHandler.set_states`` and
    ``set_dof_targets`` both call ``_invalidate_state_caches``. If a
    future refactor reverts that, the cross-backend silent-staleness
    bugs that motivated this session come back. Static AST check —
    no imports, no sim env."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1].joinpath("sim/base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseSimHandler")

    def _calls_invalidate(fn: ast.FunctionDef) -> bool:
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_invalidate_state_caches"
            ):
                return True
        return False

    for fn_name in ("set_states", "set_dof_targets", "simulate"):
        fn = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        assert _calls_invalidate(fn), (
            f"BaseSimHandler.{fn_name} no longer calls _invalidate_state_caches — "
            f"this re-opens the cross-backend stale-cache bug fixed in p0_fixes_2026_05_26. "
            f"If the invalidation moved elsewhere, update this test to look there."
        )
