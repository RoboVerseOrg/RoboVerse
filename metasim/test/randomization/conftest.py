# conftest.py
"""conftest: start one handler process per (sim, num_envs) param combination.

Handler lives in a separate process; tests in pytest process use HandlerProxy.run_test
to request the child process to run named functions (module + function name).
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import get_context
from typing import Callable

import pytest
from loguru import logger as log

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.test_utils import get_test_parameters

# Use 'spawn' context for safety with GPU/C++ resources (avoid fork-related issues)
_MP_CTX = get_context("spawn")

# Global map of running handler processes keyed by (sim, num_envs)
_shared_handler_processes: dict = {}


def get_shared_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create a standard scenario configuration for randomization tests."""
    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
    from roboverse_pack.robots.franka_cfg import FrankaCfg

    return ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.4, -0.6, 0.05],
            ),
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.5, 0.0, 0.5],
            ),
        ],
        lights=[
            DiskLightCfg(
                name="test_light",  # Changed from "light" to avoid naming conflict
                intensity=20000.0,
                color=(1.0, 1.0, 1.0),
                radius=1.2,
                pos=(0.0, 0.0, 4.5),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        robots=[FrankaCfg()],
        cameras=[
            PinholeCameraCfg(
                name="test_camera",
                width=1024,
                height=1024,
                pos=(2.0, -2.0, 2.0),
                look_at=(0.0, 0.0, 0.05),
            )
        ],
    )


def _run_test_in_process(task_queue: mp.Queue, result_queue: mp.Queue, sim: str, num_envs: int):
    """Child process target: create handler and execute requested test functions.

    Protocol:
      - When ready, child puts {"status": "ready"} into result_queue.
      - Main process sends tasks like:
          {"func_name": "camera_seed_reproducibility", "module": "metasim.test.randomization.test_camera_randomizer", "args": [...], "kwargs": {...}}
        or control messages: {"command": "close"}.
      - Child executes the function (it must be importable) and returns {"status": "success", "func": name, "result": ...}
        or on exception returns {"status": "error", "func": name, "error": str(e), "traceback": ...}.
    """
    # Import inside child process
    import traceback

    from metasim.utils.setup_util import get_handler

    log.info(f"[handler-process] Creating simulation handler for {sim}, num_envs={num_envs}")
    scenario = get_shared_scenario(sim, num_envs)
    handler = get_handler(scenario)
    log.info("[handler-process] Handler created")

    # Notify ready
    result_queue.put({"status": "ready"})

    try:
        while True:
            task = task_queue.get()
            if task is None:
                # treat None as close
                log.info("[handler-process] Received None -> shutdown")
                break

            if task.get("command") == "close":
                log.info("[handler-process] Received close command")
                break

            # Expect func (or global func name)
            func = task.get("func")
            args = task.get("args", []) or []
            kwargs = task.get("kwargs", {}) or {}

            try:
                if func is None:
                    raise RuntimeError(f"Function {func.__name__} not found")

                # Call the function with handler as first positional arg
                ret = func(handler, *args, **kwargs)

                # If return value is not picklable, it's ok — we won't require it.
                result_queue.put({"status": "success", "func": func.__name__, "result": ret})
            except Exception as e:
                tb = traceback.format_exc()
                log.exception(f"[handler-process] Exception running {func.__name__}")
                result_queue.put({
                    "status": "error",
                    "func": func.__name__,
                    "error": str(e),
                    "traceback": tb,
                })
    finally:
        # Ensure handler closed in child process
        try:
            log.info("[handler-process] Closing handler...")
            handler.close()
            log.info("[handler-process] Handler closed")
        except Exception:
            log.exception("[handler-process] Error closing handler")
        # Signal closed
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
        """Request the handler process to run `module.func_name(handler, *args, **kwargs)`.

        Returns the child process' result dict on success. Raises RuntimeError if child reported error
        or if waiting times out.
        """
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
            raise RuntimeError(f"Timeout waiting for result of {func.__name__}: {e}") from e

        if res.get("status") == "error":
            raise RuntimeError(f"Child error running {res.get('func')}: {res.get('error')}\n{res.get('traceback', '')}")
        return res

    def close(self, timeout: float = 10.0):
        """Request graceful shutdown of child handler process and wait for confirmation."""
        # send close command
        self._task_queue.put({"command": "close"})
        try:
            res = self._result_queue.get(timeout=timeout)
            return res
        except Exception:
            return {"status": "timeout"}


@pytest.fixture(scope="session", params=get_test_parameters())
def shared_handler(request):
    """Start or reuse a handler process for this (sim, num_envs) and return a HandlerProxy.

    The returned object is a proxy; tests must call proxy.run_test(...) to execute code that
    needs the handler (handler itself lives in child process).
    """
    sim, num_envs = request.param
    if sim not in ["isaacsim"]:
        pytest.skip(f"Skipping randomization tests for unsupported sim '{sim}'")
    key = (sim, num_envs)

    # If not started yet, start process for this key
    if key not in _shared_handler_processes:
        task_q = _MP_CTX.Queue()
        result_q = _MP_CTX.Queue()

        proc = _MP_CTX.Process(
            target=_run_test_in_process,
            args=(task_q, result_q, sim, num_envs),
            daemon=False,
        )
        proc.start()

        # Wait for ready
        try:
            ready = result_q.get(timeout=120)
        except Exception as e:
            proc.terminate()
            raise RuntimeError(f"Handler process failed to start: {e}") from e

        if ready.get("status") != "ready":
            proc.terminate()
            raise RuntimeError(f"Handler process for {key} returned unexpected ready message: {ready}")

        _shared_handler_processes[key] = {
            "proc": proc,
            "task_queue": task_q,
            "result_queue": result_q,
        }
        log.info(f"[main] Started handler process for {key}")

        # register finalizer to ensure child process is closed at session end
        def _cleanup():
            info = _shared_handler_processes.get(key)
            if not info:
                return
            log.info(f"[main] Cleaning up handler process for {key}")
            try:
                info["task_queue"].put({"command": "close"})
                # wait for child to report closed
                try:
                    msg = info["result_queue"].get(timeout=20)
                    log.info(f"[main] Child closed status: {msg}")
                except Exception:
                    log.warning(f"[main] No closed message from child for {key} (timed out)")

                info["proc"].join(timeout=10)
                if info["proc"].is_alive():
                    log.warning(f"[main] Child process still alive for {key}; terminating")
                    info["proc"].terminate()
                log.info(f"[main] Handler process for {key} terminated")
            except Exception:
                log.exception(f"[main] Exception during cleanup for {key}")
            finally:
                _shared_handler_processes.pop(key, None)

        request.addfinalizer(_cleanup)

    info = _shared_handler_processes[key]
    proxy = HandlerProxy(info["task_queue"], info["result_queue"])
    return proxy
