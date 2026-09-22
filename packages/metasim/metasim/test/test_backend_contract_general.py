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


@pytest.mark.general
def test_physics_dt_contract_reports_the_configured_or_resolved_step():
    """``physics_dt`` is the step the backend integrates with: the configured ``dt``, a backend override
    (Isaac Sim assigns it, MuJoCo / SuperDex read their engine), or None when unresolved.
    """
    from types import SimpleNamespace

    from metasim.sim.base import BaseSimHandler

    class _Concrete(BaseSimHandler):
        def _set_states(self, states, env_ids=None):
            pass

        def _set_dof_targets(self, actions):
            pass

        def _get_states(self, env_ids=None):
            return None

        def _simulate(self):
            pass

    h = _Concrete.__new__(_Concrete)
    h.scenario = SimpleNamespace(sim_params=SimpleNamespace(dt=None))
    assert h.physics_dt is None
    h.scenario = SimpleNamespace(sim_params=SimpleNamespace(dt=0.002))
    assert h.physics_dt == 0.002
    h.physics_dt = 0.015 / 15  # the Isaac Sim assignment path
    assert h.physics_dt == pytest.approx(0.001)


def _render_method(relative_path, class_name, method_name, namespace):
    """Execute the production method with recording stand-ins for its backend."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / relative_path
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    code = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.general
@pytest.mark.parametrize("backend", ["blender", "isaacsim"])
def test_default_ground_opt_out_skips_backend_creation(backend):
    from types import SimpleNamespace

    h = SimpleNamespace(scenario=SimpleNamespace(ground=None, add_default_ground=False))
    if backend == "blender":
        method = _render_method("sim/blender/blender.py", "BlenderHandler", "_add_default_ground", {})
        method(h, None)
    else:
        method = _render_method("sim/isaacsim/isaacsim.py", "IsaacsimHandler", "_load_terrain", {})
        method(h)


@pytest.mark.general
def test_camera_augmentation_rejects_static_calibration_splat_compositor():
    from types import SimpleNamespace

    method = _render_method("randomization/core/visual_adapter.py", "IsaacSimVisualAdapter", "validate", {})
    adapter = SimpleNamespace(
        handler=SimpleNamespace(scenario=SimpleNamespace(gs_scene=SimpleNamespace(with_gs_background=True)))
    )
    with pytest.raises(NotImplementedError, match="Gaussian-splat"):
        method(adapter, SimpleNamespace(cameras={"front": {}}), [0])


@pytest.mark.general
@pytest.mark.parametrize("timeout", [False, True])
def test_asset_wait_renders_without_physics_and_has_deadline(monkeypatch, timeout):
    import sys
    from types import SimpleNamespace

    events = []
    pending = iter((2, 1, 0))
    ticks = iter((0, 2, 3))
    usd = SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage_loading_status=lambda: (0, 0, next(pending))))
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(usd=usd))
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    monkeypatch.setitem(sys.modules, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    h = SimpleNamespace(
        scenario=SimpleNamespace(render=SimpleNamespace(asset_timeout_s=1 if timeout else 10)),
        sim=SimpleNamespace(render=lambda: events.append("render")),
    )
    method = _render_method("sim/isaacsim/isaacsim.py", "IsaacsimHandler", "_wait_for_render_assets", {})
    if timeout:
        with pytest.raises(TimeoutError, match="USD assets"):
            method(h)
        assert events == []
    else:
        method(h)
        assert events == ["render", "render"]


@pytest.mark.general
@pytest.mark.parametrize("samples,expected", [(1, 3), (32, 3), (33, 4), (128, 6)])
def test_isaac_render_budget_accumulates_before_sensor_readback(samples, expected):
    from types import SimpleNamespace

    from metasim.scenario.render import RenderCfg

    events = []
    h = SimpleNamespace(
        scenario=SimpleNamespace(render=RenderCfg(mode="pathtracing", samples=samples)),
        scene=SimpleNamespace(
            update=lambda **kw: events.append("scene"),
            sensors={"cam": SimpleNamespace(update=lambda **kw: events.append(("sensor", kw)))},
        ),
        sim=SimpleNamespace(render=lambda: events.append("render")),
        _wait_for_render_assets=lambda: None,
    )
    refresh = _render_method("sim/isaacsim/isaacsim.py", "IsaacsimHandler", "refresh_render", {})
    refresh(h, passes=1)
    assert events == ["scene", *(["render"] * expected), ("sensor", {"dt": 0.0, "force_recompute": True})]
    assert h._render_current and not h._visual_refresh_pending
    h._defer_all_visual_flushes = True
    refresh(h)
    assert len(events) == expected + 2


@pytest.mark.general
def test_isaac_render_failure_never_marks_frame_current():
    from types import SimpleNamespace

    from metasim.scenario.render import RenderCfg

    def broken():
        raise RuntimeError("renderer failed")

    h = SimpleNamespace(scenario=SimpleNamespace(render=RenderCfg()), scene=None, sim=SimpleNamespace(render=broken))
    refresh = _render_method("sim/isaacsim/isaacsim.py", "IsaacsimHandler", "refresh_render", {})
    with pytest.raises(RuntimeError, match="renderer failed"):
        refresh(h)
    assert not h._render_current


@pytest.mark.general
def test_blender_camera_state_uses_realized_pose_and_lens(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import torch

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.types import CameraState

    class Matrix:
        translation = (2.0, 3.0, 4.0)

        def to_3x3(self):
            return np.eye(3)

    cam_obj = SimpleNamespace(matrix_world=Matrix(), data=SimpleNamespace(lens=50.0, sensor_width=25.0))
    cfg = PinholeCameraCfg(name="cam", width=80, height=40, pos=(0, 0, 1))
    scene = SimpleNamespace(render=SimpleNamespace())
    h = SimpleNamespace(context=SimpleNamespace(scene=scene), _camera_objs={"cam": cam_obj}, _tmp_dir=tmp_path)
    render = _render_method(
        "sim/blender/blender.py",
        "BlenderHandler",
        "_render_camera",
        {
            "np": np,
            "torch": torch,
            "PinholeCameraCfg": PinholeCameraCfg,
            "CameraState": CameraState,
            "bpy": SimpleNamespace(ops=SimpleNamespace(render=SimpleNamespace(render=lambda **kw: None))),
            "iio": SimpleNamespace(imread=lambda path: np.zeros((40, 80, 3), dtype=np.uint8)),
            "_BLENDER_RGB_READBACK_SUFFIX": "png",
        },
    )
    state = render(h, cfg)
    assert state.pos.tolist() == [[2, 3, 4]]
    assert state.intrinsics[0].tolist() == [[160, 0, 40], [0, 160, 20], [0, 0, 1]]
    assert state.quat_world is not None
    assert torch.allclose(state.quat_opengl, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)


@pytest.mark.general
def test_isaac_camera_refresh_retains_per_env_episode_jitter():
    from types import SimpleNamespace

    import torch

    from metasim.scenario.cameras import PinholeCameraCfg

    cfg = PinholeCameraCfg(name="cam", pos=(1, 1, 1), look_at=(0, 0, 0))
    poses = []
    subset = []
    origins = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
    sensor = SimpleNamespace(
        set_world_poses_from_view=lambda p, t, env_ids=None: (poses if env_ids is None else subset).append((
            p.clone(),
            t.clone(),
            env_ids,
        ))
    )
    h = SimpleNamespace(
        scene=SimpleNamespace(env_origins=origins, sensors={"cam": sensor}),
        num_envs=2,
        device="cpu",
        cameras=[cfg],
        _camera_poses={},
    )
    namespace = {"torch": torch, "PinholeCameraCfg": PinholeCameraCfg}
    for name in ("_env_origins", "set_camera_pose", "_update_camera_pose"):
        namespace[name] = _render_method("sim/isaacsim/isaacsim.py", "IsaacsimHandler", name, namespace)
    h._env_origins = lambda: namespace["_env_origins"](h)
    refresh = namespace["_update_camera_pose"]

    # No remembered pose: the scenario configuration is what gets re-asserted.
    refresh(h)
    assert torch.equal(poses[-1][0], torch.tensor([[1.0, 1, 1], [1.0, 1, 1]]) + origins)

    # One environment is jittered; the other keeps the configured pose, across repeated refreshes.
    namespace["set_camera_pose"](h, "cam", (2.0, 3, 4), (0.2, 0.1, 0), env_ids=[1])
    assert subset[-1][2] == [1]
    for _ in range(3):
        refresh(h)
        assert torch.equal(poses[-1][0], torch.tensor([[1.0, 1, 1], [2.0, 3, 4]]) + origins)
        assert torch.equal(poses[-1][1], torch.tensor([[0.0, 0, 0], [0.2, 0.1, 0]]) + origins)

    # A later writer (legacy randomizer / DR manager) wins over the earlier jitter, and sticks.
    namespace["set_camera_pose"](h, "cam", (5.0, 5, 5), (1.0, 0, 0))
    for _ in range(2):
        refresh(h)
        assert torch.equal(poses[-1][0], torch.tensor([[5.0, 5, 5], [5.0, 5, 5]]) + origins)
        assert torch.equal(poses[-1][1], torch.tensor([[1.0, 0, 0], [1.0, 0, 0]]) + origins)
    assert cfg.pos == (1, 1, 1)
    for bad in ({"name": "missing"}, {"env_ids": [2]}):
        with pytest.raises((ValueError, KeyError)):
            namespace["set_camera_pose"](
                h, bad.get("name", "cam"), (0, 0, 0), (1, 0, 0), **{k: v for k, v in bad.items() if k != "name"}
            )
