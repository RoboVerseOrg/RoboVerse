"""conftest for query tests.

We start a real simulator handler in a separate process (like the
randomization tests) and expose a small HandlerProxy so tests can
run functions against the live handler without importing fixtures
from other directories.

All query tests share this single handler pipeline. Which tests
actually run is controlled by the simulator name (`sim`) coming
from `get_test_parameters()`:

- sim == "isaacsim": ContactForces + SitePos tests
- sim == "mujoco":   ContactForces + SitePos-on-MuJoCo tests
- sim == "mjx":      SitePos-on-MJX tests
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import get_context
from typing import Callable

import pytest
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg

# Use 'spawn' context for safety with GPU/C++ resources (avoid fork-related issues)
_MP_CTX = get_context("spawn")

# Global map of running handler processes keyed by (sim, num_envs)
_shared_handler_processes: dict = {}


def get_test_parameters():
    """Generate test parameters with different num_envs for different simulators."""
    # MuJoCo & MJX only support num_envs=1 in these tests
    # Other simulators can test with multiple environments
    isaacsim_params = [("isaacsim", num_envs) for num_envs in [1, 2, 4]]
    mujoco_params = [("mujoco", 1)]
    mjx_params = [("mjx", 1)]
    # Order matters for reproducible node ids
    return mujoco_params + isaacsim_params + mjx_params


def get_query_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create a standard scenario configuration for query tests.

    Default scenario mirrors the Unitree G1 locomotion task
    (see `roboverse_pack.tasks.unitree_rl.locomotion.walk_g1_dof29.WalkG1Dof29Task`)
    but is reused across simulators.
    """

    if sim not in {"isaacsim", "mujoco", "mjx"}:
        raise ValueError(f"Unsupported simulator '{sim}' for query tests")

    from metasim.scenario.lights import DomeLightCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from roboverse_pack.robots.g1_cfg import G1Dof29Cfg

    sim_params = SimParamCfg(
        dt=0.005,
        substeps=1,
        num_threads=10,
        solver_type=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        contact_offset=0.01,
        rest_offset=0.0,
        bounce_threshold_velocity=0.5,
        max_depenetration_velocity=1.0,
        default_buffer_size_multiplier=5,
        replace_cylinder_with_capsule=True,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
    )

    return ScenarioCfg(
        robots=[G1Dof29Cfg()],
        objects=[],
        cameras=[],
        num_envs=num_envs,
        simulator=sim,
        headless=True,
        env_spacing=2.5,
        decimation=1,
        sim_params=sim_params,
        lights=[
            DomeLightCfg(
                intensity=800.0,
                color=(0.85, 0.9, 1.0),
            )
        ],
    )


def _run_test_in_process(task_queue: mp.Queue, result_queue: mp.Queue, sim: str, num_envs: int):
    """Child process target: create handler and execute requested test functions."""
    # Import inside child process
    import traceback

    from metasim.utils.setup_util import get_handler

    log.info(f"[queries/handler-process] Creating simulation handler for {sim}, num_envs={num_envs}")
    scenario = get_query_scenario(sim, num_envs)
    handler = get_handler(scenario)
    log.info("[queries/handler-process] Handler created")

    # Notify ready
    result_queue.put({"status": "ready"})

    try:
        while True:
            task = task_queue.get()
            if task is None:
                log.info("[queries/handler-process] Received None -> shutdown")
                break

            if task.get("command") == "close":
                log.info("[queries/handler-process] Received close command")
                break

            func = task.get("func")
            args = task.get("args", []) or []
            kwargs = task.get("kwargs", {}) or {}

            try:
                if func is None:
                    raise RuntimeError("No function provided to run in handler process")

                ret = func(handler, *args, **kwargs)
                result_queue.put({"status": "success", "func": func.__name__, "result": ret})
            except Exception as e:
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                log.exception("[queries/handler-process] Error running func %s", getattr(func, "__name__", "<unknown>"))
                result_queue.put({
                    "status": "error",
                    "func": getattr(func, "__name__", "<unknown>"),
                    "error": str(e),
                    "traceback": tb,
                })
    finally:
        try:
            log.info("[queries/handler-process] Closing handler...")
            handler.close()
            log.info("[queries/handler-process] Handler closed")
        except Exception:
            log.exception("[queries/handler-process] Error closing handler")
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


def _select_params_for_test(test_name: str, doc: str | None) -> list[tuple[str, int]]:
    """Return subset of (sim, num_envs) pairs relevant for this test.

    We infer the simulators a test cares about from its name/docstring to avoid
    spinning up handlers for unrelated backends (e.g. running only MuJoCo tests
    with `-k mujoco` will now only create MuJoCo handlers).
    """
    all_params = get_test_parameters()
    text = f"{test_name}\n{doc or ''}"

    # Simulators that are actually supported by this queries suite
    supported_sims = ["isaacsim", "mujoco", "mjx"]
    # Simulators that might be mentioned in test names/docstrings but are not
    # backed by real query fixtures here (they will be skipped early).
    unsupported_sims = ["isaacgym"]

    # First, look for supported simulators mentioned in the test
    requested_supported: list[str] = [sim for sim in supported_sims if sim in text]
    if requested_supported:
        return [p for p in all_params if p[0] in requested_supported]

    # If the test explicitly targets an unsupported simulator (e.g. isaacgym),
    # parametrize it with a dummy entry so it gets reported as skipped without
    # accidentally binding to isaacsim/mujoco.
    mentioned_unsupported = [sim for sim in unsupported_sims if sim in text]
    if mentioned_unsupported:
        return [(sim, 1) for sim in mentioned_unsupported]

    # Fallback: default to isaacsim + mujoco if a test does not explicitly
    # mention any simulator (generic tests).
    fallback = ["isaacsim", "mujoco"]
    return [p for p in all_params if p[0] in fallback]


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Dynamically parametrize the shared_handler fixture per test.

    Each test only gets (sim, num_envs) combinations that match the simulator
    mentioned in its name or docstring. This keeps `-k` filtering efficient and
    avoids starting unused simulators.
    """
    if "shared_handler" not in metafunc.fixturenames:
        return

    params = _select_params_for_test(metafunc.function.__name__, metafunc.function.__doc__)
    if not params:
        # No relevant params -> leave test un-parametrized; pytest will error
        # loudly rather than silently doing something unexpected.
        return

    ids = [f"{sim}-{num_envs}" for sim, num_envs in params]
    metafunc.parametrize("shared_handler", params, indirect=True, ids=ids, scope="session")


@pytest.fixture(scope="session")
def shared_handler(request):
    """Start or reuse a handler process for this (sim, num_envs).

    Returns:
        tuple[str, HandlerProxy]: The simulator name and a proxy to the child-process handler.
    """
    sim, num_envs = request.param
    if sim not in ["isaacsim", "mujoco", "mjx"]:
        pytest.skip(f"Skipping query tests for unsupported sim '{sim}' in queries suite")
    key = (sim, num_envs)

    if key not in _shared_handler_processes:
        # Pre-flight asset check in the main process so we either
        # download once up front or gracefully skip if assets are missing.
        try:
            scenario = get_query_scenario(sim, num_envs)
            scenario.check_assets()
        except Exception as e:
            pytest.skip(
                f"Skipping query tests for {key} because required assets are missing or failed to download: {e}"
            )

        task_q = _MP_CTX.Queue()
        result_q = _MP_CTX.Queue()

        proc = _MP_CTX.Process(
            target=_run_test_in_process,
            args=(task_q, result_q, sim, num_envs),
            daemon=False,
        )
        proc.start()

        try:
            # Handler creation can trigger large asset downloads on first run;
            # use a generous timeout here and rely on the pre-flight check above
            # to surface missing-asset issues quickly.
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
        log.info(f"[queries/main] Started handler process for {key}")

        def _cleanup():
            info = _shared_handler_processes.get(key)
            if not info:
                return
            log.info(f"[queries/main] Cleaning up handler process for {key}")
            try:
                info["task_queue"].put({"command": "close"})
                try:
                    msg = info["result_queue"].get(timeout=20)
                    log.info(f"[queries/main] Child closed status: {msg}")
                except Exception:
                    log.warning(f"[queries/main] No closed message from child for {key} (timed out)")

                info["proc"].join(timeout=10)
                if info["proc"].is_alive():
                    log.warning(f"[queries/main] Child process still alive for {key}; terminating")
                    info["proc"].terminate()
                log.info(f"[queries/main] Handler process for {key} terminated")
            except Exception:
                log.exception(f"[queries/main] Exception during cleanup for {key}")
            finally:
                _shared_handler_processes.pop(key, None)

        request.addfinalizer(_cleanup)

    info = _shared_handler_processes[key]
    proxy = HandlerProxy(info["task_queue"], info["result_queue"])
    return sim, proxy
