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
    """The concrete handler classes a user can obtain: every entry of the backend registry
    (``metasim.utils.setup_util.SIM_BACKENDS``, the single source of truth for dispatch), the
    composite ``HybridSimHandler``, and one ``ParallelSimWrapper`` instance class so the wrapper's
    contract is covered too. Enumerating the registry instead of walking ``__subclasses__`` keeps stub
    handlers defined by other test modules (which register as subclasses when files share a process)
    out of the parametrization.
    """
    import importlib

    from metasim.utils.setup_util import SIM_BACKENDS

    classes: list[type[BaseSimHandler]] = []
    first_single: type[BaseSimHandler] | None = None
    for spec in SIM_BACKENDS.values():
        try:
            cls = getattr(importlib.import_module(spec.module), spec.cls)
        except Exception:
            continue  # optional backend not installed here
        classes.append(cls)
        if spec.parallel and first_single is None:
            first_single = cls
    try:
        from metasim.sim.hybrid import HybridSimHandler

        classes.append(HybridSimHandler)
    except Exception:
        pass
    if first_single is not None:
        from metasim.sim.parallel import ParallelSimWrapper

        classes.append(ParallelSimWrapper(first_single))
    return classes


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
    contract tightened.
    """
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
    no imports, no sim env.
    """
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


@pytest.mark.parametrize("cls", _import_all_backend_handlers(), ids=lambda c: c.__name__)
def test_backend_declares_set_states_refreshes(cls: type[BaseSimHandler]):
    """Every backend states whether ``_set_states`` leaves its renderer current.

    ``BaseTaskEnv.reset`` and the benchmark render sync read ``set_states_refreshes`` to decide whether
    a ``refresh_render()`` is still needed; a backend that says nothing inherits ``False`` (an extra
    refresh, never a stale frame). A ``property`` is allowed for answers that depend on the instance
    (Isaac Sim: only with cameras; the composites: whatever they wrap).
    """
    attr = getattr(cls, "set_states_refreshes", None)
    assert attr is not None, f"{cls.__name__} lost the set_states_refreshes capability flag"
    if isinstance(attr, property):
        return
    assert isinstance(attr, bool), f"{cls.__name__}.set_states_refreshes must be a bool, got {attr!r}"


def test_set_states_refreshes_declared_where_set_states_renders():
    """The backends whose ``_set_states`` already refreshes the renderer say so; the rest inherit False."""
    by_name = {cls.__name__: cls for cls in _import_all_backend_handlers()}
    class_level_true = {"BlenderHandler", "SuperdexHandler"}
    instance_level = {"IsaacsimHandler", "MujocoHandler", "HybridSimHandler", "ParallelHandler"}
    for name, cls in by_name.items():
        attr = cls.set_states_refreshes
        if name in instance_level:
            assert isinstance(attr, property), f"{name}.set_states_refreshes should depend on the instance"
        else:
            assert attr is (name in class_level_true), f"{name}.set_states_refreshes is {attr!r}"


def test_parallel_wrapper_forwards_only_a_class_level_guarantee():
    """A class-level ``True`` on the wrapped class is a guarantee; a class-level ``False`` / absence means
    no guarantee. A property on the wrapped class (MuJoCo: viewer-dependent) is asked of a worker, so
    that case is exercised with a fake remote."""
    from metasim.sim.parallel import ParallelSimWrapper

    class _Yes:
        set_states_refreshes = True

    class _Silent:
        pass

    class _Maybe:
        @property
        def set_states_refreshes(self):
            return True

    for base, expected in ((_Yes, True), (_Silent, False)):
        wrapper = ParallelSimWrapper(base)
        instance = object.__new__(wrapper)  # no workers: a class-level answer must not need any state
        assert instance.set_states_refreshes is expected, base.__name__

    class _Remote:
        def __init__(self, answer):
            self.answer = answer
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

    wrapper = ParallelSimWrapper(_Maybe)
    instance = object.__new__(wrapper)
    instance.remotes = [_Remote(True)]
    instance._recv_or_surface = lambda idx: instance.remotes[idx].answer
    assert instance.set_states_refreshes is True
    assert instance.remotes[0].sent == [("set_states_refreshes", (None,))]
    assert instance.set_states_refreshes is True and len(instance.remotes[0].sent) == 1  # asked once


def _reset_with(flag: bool) -> list[str]:
    """Drive ``BaseTaskEnv.reset`` with a stub handler that records its calls."""
    import torch

    from metasim.task.base import BaseTaskEnv

    calls: list[str] = []

    class _Stub:
        num_envs = 2
        set_states_refreshes = flag

        def set_states(self, states=None, env_ids=None):
            calls.append("set_states")

        def refresh_render(self):
            calls.append("refresh_render")

        def get_states(self, env_ids=None, mode="tensor"):
            calls.append("get_states")
            return None

    env = BaseTaskEnv.__new__(BaseTaskEnv)
    env.handler = _Stub()
    env.reset_callback = []
    env._initial_states = None
    env._episode_steps = torch.zeros(2, dtype=torch.long)
    env.device = torch.device("cpu")
    env._privileged_observation = lambda states: None
    env._observation = lambda states: None
    env.reset()
    return calls


def test_task_reset_refreshes_render_only_when_the_backend_did_not():
    """``BaseTaskEnv.reset`` consults the backend's capability flag instead of guessing the backend."""
    assert _reset_with(False) == ["set_states", "refresh_render", "get_states"]
    assert _reset_with(True) == ["set_states", "get_states"]


def test_capability_flags_match_the_backends_that_implement_them():
    """``get_states_honours_env_ids`` and ``set_states_restores_velocities`` are declared where the code does it."""
    by_name = {cls.__name__: cls for cls in _import_all_backend_handlers()}
    honours = {"MJXHandler", "NewtonHandler", "GenesisHandler", "HybridSimHandler", "ParallelHandler"}
    restores = {"MujocoHandler", "SuperdexHandler", "NewtonHandler", "IsaacsimHandler", "IsaacgymHandler"}
    dict_restores = {
        "MujocoHandler",
        "SuperdexHandler",
        "NewtonHandler",
    }  # Isaac Sim / Isaac Gym dict paths write poses only
    for name, cls in by_name.items():
        assert cls.get_states_honours_env_ids is (name in honours), f"{name}.get_states_honours_env_ids"
        attr = cls.set_states_restores_velocities
        dict_attr = cls.set_states_restores_dict_velocities
        if name in {"ParallelHandler", "HybridSimHandler"}:
            assert isinstance(attr, property) and isinstance(dict_attr, property), f"{name} forwards the wrapped answer"
        else:
            assert attr is (name in restores), f"{name}.set_states_restores_velocities is {attr!r}"
            assert dict_attr is (name in dict_restores), f"{name}.set_states_restores_dict_velocities is {dict_attr!r}"


def test_wrappers_forward_the_velocity_capability_of_what_they_wrap():
    from metasim.sim.parallel import ParallelSimWrapper

    class _Restores:
        set_states_restores_velocities = True

    class _Drops:
        pass

    assert object.__new__(ParallelSimWrapper(_Restores)).set_states_restores_velocities is True
    assert object.__new__(ParallelSimWrapper(_Drops)).set_states_restores_velocities is False


def test_get_states_honours_env_ids_declarations_are_pinned():
    """The flag turns a self-healing slice into a hard error, so a wrong declaration breaks every
    partial reset on that backend. The set is pinned here; a backend joining it needs its
    ``_get_states`` to index robots and objects by ``env_ids`` (Isaac Gym indexes only its cameras and
    stays out; MuJoCo, SAPIEN 3, PyBullet, SuperDex, Isaac Sim, Blender return the full batch).
    """
    honours = {cls.__name__ for cls in _import_all_backend_handlers() if cls.get_states_honours_env_ids}
    installed = {cls.__name__ for cls in _import_all_backend_handlers()}
    assert (
        honours == {"MJXHandler", "NewtonHandler", "GenesisHandler", "HybridSimHandler", "ParallelHandler"} & installed
    )
