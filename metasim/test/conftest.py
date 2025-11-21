"""Shared pytest utilities for simulator-backed tests.

This module provides:
- global marker registration (isaacsim/isaacgym/mujoco/mjx/sim/general)
- a registry to associate test package prefixes with a scenario_fn
- a generic `shared_handler` fixture that spins up a handler in a child process
  using the registered scenario builder for that package.

Suites opt in by calling `register_shared_suite(pkg_prefix, scenario_fn)`
from their own conftest.py.

IMPORTANT: Tests marked with @pytest.mark.general should NOT request the
`shared_handler` fixture. General tests are for pure unit tests that don't
need any simulator or handler.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import get_context
from typing import Callable

import pytest
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.test_utils import get_test_parameters

_MP_CTX = get_context("spawn")
_SUPPORTED_SIMS = {"isaacgym", "isaacsim", "mujoco", "mjx"}

# pkg_prefix -> scenario_fn
_SUITE_REGISTRY: dict[str, Callable[[str, int], ScenarioCfg]] = {}

# Global map of running handler processes keyed by (scenario_fn, sim, num_envs)
_shared_handler_processes: dict = {}


def register_shared_suite(
    pkg_prefix: str,
    scenario_fn: Callable[[str, int], ScenarioCfg],
) -> None:
    """Register a test suite package for shared_handler support."""
    _SUITE_REGISTRY[pkg_prefix] = scenario_fn


def pytest_configure(config):
    """Register sim markers to avoid unknown-mark warnings."""
    for name, desc in [
        ("isaacsim", "tests that require or target IsaacSim"),
        ("isaacgym", "tests that require or target IsaacGym"),
        ("mujoco", "tests that require or target MuJoCo"),
        ("mjx", "tests that require or target MJX"),
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
    if "handler" in metafunc.fixturenames:
        target_fixture = "handler"
    else:
        return

    suite = _find_suite_for_module(metafunc.definition.module.__name__)
    if suite is None:
        return
    scenario_fn = suite

    sims = _extract_sim_markers(metafunc)

    if sims == ["_general"]:
        metafunc.parametrize(target_fixture, [{"general": True}], indirect=True, ids=["general"], scope="session")
        return

    params = get_test_parameters()
    if sims:
        params = [p for p in params if p[0] in sims]
    if not params:
        return

    wrapped_params = [{"sim": sim, "num_envs": num_envs, "scenario_fn": scenario_fn} for (sim, num_envs) in params]
    ids = [f"{p['sim']}-{p['num_envs']}" for p in wrapped_params]
    metafunc.parametrize(target_fixture, wrapped_params, indirect=True, ids=ids, scope="session")


def _run_test_in_process(task_queue: mp.Queue, result_queue: mp.Queue, sim: str, num_envs: int, scenario_fn):
    """Child process target: create handler and execute requested test functions."""
    import traceback

    from metasim.utils.setup_util import get_handler

    log.info(f"[shared-handler] Creating simulation handler for {sim}, num_envs={num_envs}")
    scenario = scenario_fn(sim, num_envs)
    handler = get_handler(scenario)
    log.info("[shared-handler] Handler created")

    # Notify ready
    result_queue.put({"status": "ready"})

    try:
        while True:
            task = task_queue.get()
            if task is None:
                log.info("[shared-handler] Received None -> shutdown")
                break

            if task.get("command") == "close":
                log.info("[shared-handler] Received close command")
                break

            func = task.get("func")
            args = task.get("args", [])
            kwargs = task.get("kwargs", {})

            try:
                if func is None:
                    raise RuntimeError("No function provided to run in handler process")

                func_name = getattr(func, "__name__", "<unknown>")
                ret = func(handler, *args, **kwargs)
                result_queue.put({"status": "success", "func": func_name, "result": ret})
            except Exception as e:
                func_name = getattr(func, "__name__", "<unknown>")
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                log.exception("[shared-handler] Error running func %s", func_name)
                result_queue.put({
                    "status": "error",
                    "func": func_name,
                    "error": str(e),
                    "traceback": tb,
                })
    finally:
        try:
            log.info("[shared-handler] Closing handler...")
            handler.close()
            log.info("[shared-handler] Handler closed")
        except Exception:
            log.exception("[shared-handler] Error closing handler")
        result_queue.put({"status": "closed"})


class HandlerProxy:
    """Proxy object given to tests. Sends tasks to the handler process and waits for results."""

    def __init__(self, task_queue: mp.Queue, result_queue: mp.Queue, timeout: float = 60.0):
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._timeout = timeout

    def run_test(
        self,
        func: Callable,
        timeout: float | None = None,
        *args,
        **kwargs,
    ):
        """Request the handler process to run `func(handler, *args, **kwargs)`."""
        if timeout is None:
            timeout = self._timeout

        task = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }
        self._task_queue.put(task)

        try:
            res = self._result_queue.get(timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Timeout waiting for result of {getattr(func, '__name__', '<unknown>')}: {e}") from e

        if res.get("status") == "error":
            raise RuntimeError(f"Child error running {res.get('func')}: {res.get('error')}\n{res.get('traceback', '')}")
        return res

    def close(self, timeout: float = 10.0):
        """Request graceful shutdown of child handler process and wait for confirmation."""
        self._task_queue.put({"command": "close"})
        try:
            res = self._result_queue.get(timeout=timeout)
            return res
        except Exception:
            return {"status": "timeout"}


def _get_or_create_handler(param, request):
    if param.get("general"):
        pytest.skip("No simulator required for @pytest.mark.general tests")
    sim = param["sim"]
    num_envs = param["num_envs"]
    scenario_fn = param["scenario_fn"]

    if sim not in _SUPPORTED_SIMS:
        pytest.skip(f"Skipping shared handler for unsupported sim '{sim}'")

    key = (scenario_fn.__module__, scenario_fn.__name__, sim, num_envs)

    if key not in _shared_handler_processes:
        # Pre-flight asset check in the main process
        try:
            scenario = scenario_fn(sim, num_envs)
            scenario.check_assets()
        except Exception as e:
            pytest.skip(f"Skipping shared handler for {key} because assets are missing or failed to download: {e}")

        task_q = _MP_CTX.Queue()
        result_q = _MP_CTX.Queue()

        proc = _MP_CTX.Process(
            target=_run_test_in_process,
            args=(task_q, result_q, sim, num_envs, scenario_fn),
            daemon=False,
        )
        proc.start()

        try:
            ready = result_q.get(timeout=600)
        except Exception as e:
            proc.terminate()
            raise RuntimeError(f"Handler process failed to start for {key}: {e}") from e

        if ready.get("status") != "ready":
            proc.terminate()
            raise RuntimeError(f"Handler process for {key} returned unexpected ready message: {ready}")

        _shared_handler_processes[key] = {
            "proc": proc,
            "task_queue": task_q,
            "result_queue": result_q,
        }
        log.info(f"[shared-handler/main] Started handler process for {key}")

        def _cleanup():
            info = _shared_handler_processes.get(key)
            if not info:
                return
            log.info(f"[shared-handler/main] Cleaning up handler process for {key}")
            try:
                info["task_queue"].put({"command": "close"})
                try:
                    msg = info["result_queue"].get(timeout=20)
                    log.info(f"[shared-handler/main] Child closed status: {msg}")
                except Exception:
                    log.warning(f"[shared-handler/main] No closed message from child for {key} (timed out)")

                info["proc"].join(timeout=10)
                if info["proc"].is_alive():
                    log.warning(f"[shared-handler/main] Child process still alive for {key}; terminating")
                    info["proc"].terminate()
                log.info(f"[shared-handler/main] Handler process for {key} terminated")
            except Exception:
                log.exception(f"[shared-handler/main] Exception during cleanup for {key}")
            finally:
                _shared_handler_processes.pop(key, None)

        request.addfinalizer(_cleanup)

    info = _shared_handler_processes[key]
    proxy = HandlerProxy(info["task_queue"], info["result_queue"])
    return sim, proxy


@pytest.fixture(scope="session")
def handler(request):
    """Alias fixture for shared_handler (for tests preferring `handler` name)."""
    return _get_or_create_handler(request.param, request)


def pytest_pyfunc_call(pyfuncitem):
    """Run simulator-marked tests inside the child process automatically."""
    markers = {m.name for m in pyfuncitem.iter_markers()}
    if "general" in markers:
        return None  # default behavior

    if "handler" not in pyfuncitem.funcargs:
        return None

    sim, proxy = pyfuncitem.funcargs["handler"]
    # Collect other fixture-provided args in order, excluding the handler fixture itself
    argnames = pyfuncitem._fixtureinfo.argnames or []
    other_args = [pyfuncitem.funcargs[name] for name in argnames if name != "handler"]
    proxy.run_test(pyfuncitem.obj, *other_args)
    return True
