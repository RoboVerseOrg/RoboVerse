"""Shared pytest utilities for simulator-backed tests.

This module provides:
- global marker registration (isaacsim/isaacgym/mujoco/mjx/sim/general)
- a registry to associate test package prefixes with a scenario_fn
- a generic `handler` fixture that creates a handler using the registered
  scenario builder for that package.

Suites opt in by calling `register_shared_suite(pkg_prefix, scenario_fn)`
from their own conftest.py.

IMPORTANT: Tests marked with @pytest.mark.general should NOT request the
`handler` fixture. General tests are for pure unit tests that don't
need any simulator or handler.
"""

from __future__ import annotations

from typing import Callable

import pytest
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg
from metasim.sim.sim_context import HandlerContext
from metasim.test.test_utils import get_test_parameters

_SUPPORTED_SIMS = {"isaacgym", "isaacsim", "mujoco", "mjx", "sapien3"}

# pkg_prefix -> scenario_fn
_SUITE_REGISTRY: dict[str, Callable[[str, int], ScenarioCfg]] = {}

# Global map of running handler contexts keyed by (scenario_fn, sim, num_envs)
_shared_handler_contexts: dict[tuple, HandlerContext] = {}

# Track the current IsaacSim simulation context
_isaacsim_context: dict = {}


def register_shared_suite(
    pkg_prefix: str,
    scenario_fn: Callable[[str, int], ScenarioCfg],
) -> None:
    """Register a test suite package for handler support."""
    _SUITE_REGISTRY[pkg_prefix] = scenario_fn


def pytest_configure(config):
    """Register sim markers to avoid unknown-mark warnings."""
    for name, desc in [
        ("isaacsim", "tests that require or target IsaacSim"),
        ("isaacgym", "tests that require or target IsaacGym"),
        ("mujoco", "tests that require or target MuJoCo"),
        ("mjx", "tests that require or target MJX"),
        ("sapien3", "tests that require or target SAPIEN3"),
        ("sim(*sims)", "specify one or more simulator backends for a test"),
        ("general", "tests that require no simulator/handler"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")


def _extract_sim_markers(metafunc: pytest.Metafunc) -> list[str]:
    """Return list of simulators requested via markers."""
    sims: set[str] = set()
    if any(marker.name == "general" for marker in metafunc.definition.iter_markers()):
        return ["_general"]
    for marker in metafunc.definition.iter_markers():
        if marker.name in _SUPPORTED_SIMS:
            sims.add(marker.name)
        elif marker.name == "sim":
            for arg in marker.args:
                if arg in _SUPPORTED_SIMS:
                    sims.add(arg)
    return sorted(sims)


def _find_suite_for_module(module_name: str):
    """Find the registered suite whose prefix matches the module."""
    matches = [(prefix, fn) for prefix, fn in _SUITE_REGISTRY.items() if module_name.startswith(prefix)]
    if not matches:
        return None
    # Pick the longest matching prefix for specificity
    return max(matches, key=lambda x: len(x[0]))[1]


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Parametrize handler for suites that registered support."""
    if "handler" not in metafunc.fixturenames:
        return

    suite = _find_suite_for_module(metafunc.definition.module.__name__)
    if suite is None:
        return
    scenario_fn = suite

    sims = _extract_sim_markers(metafunc)

    if sims == ["_general"]:
        metafunc.parametrize("handler", [{"general": True}], indirect=True, ids=["general"], scope="session")
        return

    params = get_test_parameters()
    if sims:
        params = [p for p in params if p[0] in sims]
    if not params:
        return

    wrapped_params = [{"sim": sim, "num_envs": num_envs, "scenario_fn": scenario_fn} for (sim, num_envs) in params]
    ids = [f"{p['sim']}-{p['num_envs']}" for p in wrapped_params]
    metafunc.parametrize("handler", wrapped_params, indirect=True, ids=ids, scope="session")


def _clear_isaacsim_context():
    """Clear the current IsaacSim simulation context if one exists."""
    if "sim_context" in _isaacsim_context:
        sim_context = _isaacsim_context.pop("sim_context")
        log.debug("Clearing IsaacSim simulation context")
        try:
            sim_context.clear_all_callbacks()
            sim_context.clear_instance()
        except Exception:
            log.exception("Error clearing IsaacSim simulation context")
        log.debug("IsaacSim simulation context cleared")


def _create_isaacsim_context():
    """Create a new IsaacSim simulation context."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils

    log.debug("Creating new IsaacSim stage")
    stage_utils.create_new_stage()
    log.debug("New IsaacSim stage created")

    sim_cfg = sim_utils.SimulationCfg()
    sim_context = sim_utils.SimulationContext(sim_cfg)
    sim_context._app_control_on_stop_handle = None

    _isaacsim_context["sim_context"] = sim_context
    return sim_context


def _get_or_create_handler(param, request, isaacsim_app):
    """Get or create a handler for the given parameters using HandlerContext."""
    if param.get("general"):
        pytest.skip("No simulator required for @pytest.mark.general tests")

    sim = param["sim"]
    num_envs = param["num_envs"]
    scenario_fn = param["scenario_fn"]

    if sim not in _SUPPORTED_SIMS:
        pytest.skip(f"Skipping handler for unsupported sim '{sim}'")

    key = (scenario_fn.__module__, scenario_fn.__name__, sim, num_envs)

    if key not in _shared_handler_contexts:
        # Pre-flight asset check
        try:
            scenario = scenario_fn(sim, num_envs)
            scenario.check_assets()
        except Exception as e:
            pytest.skip(f"Skipping handler for {key} because assets are missing or failed to download: {e}")

        log.info(f"[handler] Creating handler for {sim}, num_envs={num_envs}")

        # For IsaacSim: clear any existing context and create a new one
        if sim == "isaacsim":
            _clear_isaacsim_context()
            _create_isaacsim_context()

        # Use HandlerContext to create and launch the handler
        handler_context = HandlerContext(scenario, simulation_app=isaacsim_app)
        handler = handler_context.__enter__()

        log.info("[handler] Handler created via HandlerContext")

        _shared_handler_contexts[key] = handler_context

        def _cleanup():
            ctx = _shared_handler_contexts.get(key)
            if ctx is None:
                return
            log.info(f"[handler] Cleaning up handler for {key}")
            try:
                # Use HandlerContext's __exit__ for proper cleanup
                ctx.__exit__(None, None, None)
                if sim == "isaacsim":
                    _clear_isaacsim_context()
                log.info(f"[handler] Handler for {key} cleaned up")
            except Exception:
                log.exception(f"[handler] Error cleaning up handler for {key}")
            finally:
                _shared_handler_contexts.pop(key, None)

        request.addfinalizer(_cleanup)

    return _shared_handler_contexts[key].handler


@pytest.fixture(scope="session")
def isaacsim_app(request):
    """Create an IsaacSim app if any test in the session uses 'isaacsim'."""
    needs_isaacsim = False

    # Check if any test uses isaacsim
    session = request.session
    if hasattr(session, "items"):
        for item in session.items:
            if hasattr(item, "callspec") and "handler" in item.callspec.params:
                handler_param = item.callspec.params["handler"]
                if isinstance(handler_param, dict) and handler_param.get("sim") == "isaacsim":
                    needs_isaacsim = True
                    break

    if needs_isaacsim:
        from isaaclab.app import AppLauncher

        log.info("[isaacsim_app] Launching IsaacSim application")
        app = AppLauncher(headless=True, enable_cameras=True).app
        yield app
        # NOTE: Don't call app.close(), otherwise pytest summary will be skipped!
        log.info("[isaacsim_app] IsaacSim session complete")
    else:
        yield None


@pytest.fixture(scope="session")
def handler(request, isaacsim_app):
    """Fixture providing a shared handler instance for the test session."""
    if not hasattr(request, "param"):
        # For session-scoped fixtures, request.node is the session, so get info from fspath
        module_info = getattr(request, "fspath", "unknown")
        pytest.fail(
            f"Test uses 'handler' fixture but no scenario is registered. "
            f"Ensure your test module's conftest.py calls register_shared_suite() "
            f"with the correct module prefix (fspath: {module_info}). "
            f"Registered prefixes: {list(_SUITE_REGISTRY.keys())}"
        )
    return _get_or_create_handler(request.param, request, isaacsim_app)
