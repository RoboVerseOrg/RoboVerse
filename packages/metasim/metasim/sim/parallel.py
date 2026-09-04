"""A module that implements parallel simulation using multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
import platform
import sys
import time
import traceback
from copy import deepcopy
from functools import partial
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

import torch
from loguru import logger as log

if TYPE_CHECKING:
    from metasim.scenario.scenario import ScenarioCfg

from metasim.queries.base import BaseQueryType
from metasim.sim.base import BaseSimHandler
from metasim.types import CompatActionInput, DictEnvState, TensorState
from metasim.utils.state import join_tensor_states


def _worker(
    rank: int,
    remote: Connection,
    parent_remote: Connection,
    error_queue: mp.Queue,
    handler_class: type[BaseSimHandler],
):
    parent_remote.close()

    # Bind ``env`` before the try-block so the ``finally`` can guard on it:
    # if ``handler_class()`` raises, ``env.close()`` must not shadow the
    # construction error with a NameError.
    env: BaseSimHandler | None = None
    try:
        env = handler_class()
        while True:
            cmd, data = remote.recv()
            if cmd == "launch":
                env.launch()
            elif cmd == "render":
                env.render()
            elif cmd == "refresh_render":
                env.refresh_render()
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "set_states":
                env.set_states(data[0])
            elif cmd == "get_states":
                states = env.get_states(**data[0])
                remote.send(states)
            elif cmd == "simulate":
                env.simulate()
            elif cmd == "get_joint_names":
                names = env.get_joint_names(data[0], data[1])
                remote.send(names)
            elif cmd == "get_body_names":
                names = env.get_body_names(data[0], data[1])
                remote.send(names)
            elif cmd == "set_dof_targets":
                env.set_dof_targets(data[0])
            elif cmd == "set_seed":
                env.set_seed(data[0])
            elif cmd == "handshake":
                # This is used to make sure that the environment is initialized before sending any commands
                remote.send("handshake")
            elif cmd == "device":
                remote.send(env.device)
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
    except KeyboardInterrupt:
        log.info("Worker KeyboardInterrupt")
    except EOFError:
        log.info("Worker EOF")
    except Exception as err:
        log.error(err)
        tb_str = traceback.format_exception(type(err), err, err.__traceback__)
        error_queue.put((type(err).__name__, str(err), tb_str))
        sys.exit(1)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_err:
                log.warning(f"env.close() raised during worker teardown: {close_err}")


def ParallelSimWrapper(base_cls: type[BaseSimHandler]) -> type[BaseSimHandler]:
    """A parallel simulation handler that uses multiprocessing to run multiple simulations in parallel."""

    class ParallelHandler(BaseSimHandler):
        # Workers receive states over the pipe as a per-env list of plain dicts
        # (``set_states`` in ``_worker`` forwards ``data[0]`` straight to the
        # single-env backend), and ``_set_states`` below indexes/len()s that
        # list. So the wrapper must consume the *dict* batch form: declare it
        # here so the base ``set_states`` normaliser converts a TensorState
        # (e.g. RLTaskEnv's ``list_state_to_tensor`` output) into
        # ``list[DictEnvState]`` BEFORE it reaches ``_set_states``. Without
        # this, ``_set_states`` got a raw TensorState and died with
        # ``TypeError: object of type 'TensorState' has no len()`` whenever
        # num_envs>1 (the only case where this wrapper class is used).
        _set_states_input_type = "dict"

        @property
        def set_states_refreshes(self) -> bool:  # type: ignore[override]
            """The workers run ``base_cls``; only a class-level ``True`` on it is a guarantee."""
            return getattr(base_cls, "set_states_refreshes", False) is True

        def __new__(cls, scenario: ScenarioCfg, extra_spec: dict[str, BaseQueryType] | None = None):
            """If num_envs is one, simply use the original single-thread class."""
            if scenario.num_envs == 1:
                return base_cls(scenario, extra_spec)
            else:
                return super().__new__(cls)

        def __init__(self, scenario: ScenarioCfg, extra_spec: dict[str, BaseQueryType] | None = None):
            sub_scenario = deepcopy(scenario)
            sub_scenario.num_envs = 1
            super().__init__(scenario, extra_spec)

            self.waiting = False
            self.closed = False

            # Fork is not a thread safe method
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available and platform.system() != "Darwin" else "spawn"
            ctx = mp.get_context(start_method)
            self.error_queue = ctx.Queue()

            # Initialize workers
            self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)], strict=False)
            self.processes = []
            for rank in range(self.num_envs):
                work_remote = self.work_remotes[rank]
                remote = self.remotes[rank]
                # Pass ``extra_spec`` into the worker handler so its
                # ``optional_queries`` are populated. Without it the worker was
                # built as ``base_cls(sub_scenario)`` with no queries, so every
                # worker's ``get_extra()`` returned ``{}`` and ``state.extras``
                # was always empty in parallel (num_envs>1) mode — i.e. site /
                # contact / custom queries were silently dead in exactly the
                # config used for eval. The query objects are unbound
                # (``handler is None``) until each worker's ``launch()`` binds
                # them, so they pickle cleanly into the worker here.
                args = (rank, work_remote, remote, self.error_queue, partial(base_cls, sub_scenario, extra_spec))
                # daemon=True: if the main process crashes, we should not cause things to hang
                process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
                process.start()
                self.processes.append(process)
                work_remote.close()

            # Every worker must reach its command loop before the wrapper is usable. A worker whose
            # handler construction raised has already exited, so the handshake reads through
            # ``_recv_or_surface`` and raises its real traceback (from ``error_queue``) instead of
            # a bare ``EOFError``; the surviving workers are torn down so nothing is left behind.
            try:
                self._handshake()
            except BaseException:
                self._terminate_workers()
                raise

        def _check_error(self):
            """Drain the error queue and detect dead workers.

            Before: only called from ``launch``, so a worker that died on any
            other call (OOM, GPU failure, asset error) left the parent either
            hanging on ``remote.recv()`` or eventually raising a cryptic
            ``EOFError`` — the real traceback in ``error_queue`` was never
            surfaced. Now every public method calls this both before/after
            wire I/O, and the parent additionally checks ``process.is_alive``
            so an externally-killed worker (SIGKILL, OOM-kill) is caught
            even when the queue is empty.
            """
            if self.closed:
                return
            # 1. Drain anything the workers explicitly reported.
            if not self.error_queue.empty():
                err_type, err_msg, tb = self.error_queue.get()
                tb_text = "".join(tb) if isinstance(tb, list) else str(tb)
                log.error(f"Parallel worker error ({err_type}): {err_msg}\n{tb_text}")
                raise RuntimeError(f"Parallel worker error ({err_type}): {err_msg}\n{tb_text}")
            # 2. Detect workers that died without reporting (e.g. OOM-killed).
            dead = [i for i, p in enumerate(self.processes) if not p.is_alive()]
            if dead:
                raise RuntimeError(
                    f"Parallel worker(s) {dead} died without reporting an error "
                    f"(exit codes: {[self.processes[i].exitcode for i in dead]}). "
                    f"Common cause: OOM-kill or segfault inside the simulator backend."
                )

        def _recv_or_surface(self, remote_idx: int):
            """``recv`` with worker-death detection. A dead worker closes the
            pipe end, which raises ``EOFError`` or ``ConnectionResetError`` —
            translate that into the real underlying error from the queue
            instead of letting the user chase a cryptic IPC traceback."""
            try:
                return self.remotes[remote_idx].recv()
            except (EOFError, ConnectionResetError, BrokenPipeError) as err:
                # Worker died — _check_error will raise with the real message.
                self._wait_for_error_report()
                self._check_error()
                raise RuntimeError(
                    f"Parallel worker {remote_idx} closed its pipe without reporting an error. "
                    f"Worker may have been killed externally."
                ) from err

        def launch(self):
            for remote in self.remotes:
                self._send(remote, ("launch", (None,)))
            self.waiting = False
            # ``launch`` sends no reply; the handshake round-trips so every worker has finished
            # launching before the wrapper returns, and surfaces a worker that died in ``launch``.
            self._handshake()

        def close(self):
            if self.closed:
                return
            for remote in self.remotes:
                self._send(remote, ("close", (None,)))
            self._terminate_workers()

        @staticmethod
        def _send(remote: Connection, msg) -> None:
            """``send`` that tolerates a dead worker: the following ``recv`` / ``_check_error``
            reports the worker's real error, which a ``BrokenPipeError`` here would only hide."""
            try:
                remote.send(msg)
            except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                pass

        def _handshake(self) -> None:
            """Round-trip every worker; raises the worker's own error if one has died."""
            for remote in self.remotes:
                self._send(remote, ("handshake", (None,)))
            for idx in range(len(self.remotes)):
                self._recv_or_surface(idx)
            self._check_error()

        def _wait_for_error_report(self, timeout: float = 2.0) -> None:
            """Give a dying worker's queue feeder thread time to deliver its traceback.

            ``mp.Queue.put`` hands the item to a background thread; the pipe end can close before
            the report is readable on this side, which would turn a real traceback into
            "died without reporting an error".
            """
            deadline = time.monotonic() + timeout
            while self.error_queue.empty() and time.monotonic() < deadline:
                if any(p.is_alive() for p in self.processes):
                    time.sleep(0.01)
                else:
                    break

        def _terminate_workers(self, join_timeout: float = 10.0) -> None:
            """Join every worker, force-terminating any that does not exit in time."""
            for process in self.processes:
                process.join(timeout=join_timeout)
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=join_timeout)
            self.closed = True

        def _set_states(self, states: list[DictEnvState], env_ids: list[int] | None = None) -> None:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            if len(states) == 1:
                payload = states[0]
                for env_idx in env_ids:
                    self.remotes[env_idx].send(("set_states", ([payload],)))
                self._check_error()
                return

            if len(states) != len(env_ids):
                raise ValueError(
                    f"states nums {len(states)} and env_ids nums {len(env_ids)} are inconsistent, "
                    "in parallel mode, it must be one-to-one or single broadcast"
                )

            for local_idx, env_idx in enumerate(env_ids):
                self.remotes[env_idx].send(("set_states", ([states[local_idx]],)))
            self._check_error()

        def _get_states(self, env_ids: list[int] | None = None, **kwargs) -> TensorState:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            # ``join_tensor_states`` requires per-worker TensorStates. The public
            # ``SimHandler.get_states`` calls ``_get_states(env_ids=...)`` WITHOUT
            # forwarding ``mode``, so workers would default to ``mode="dict"`` and
            # return the legacy list/dict format → join would crash with
            # "'list'/'dict' object has no attribute 'objects'". Force tensor mode
            # on the workers; the public method still handles dict-mode conversion
            # + caching on top, so this stays backward-compatible.
            worker_kwargs = {**kwargs, "mode": "tensor"}
            for i in env_ids:
                self.remotes[i].send(("get_states", (worker_kwargs,)))
            states_list = []
            for i in env_ids:
                states = self._recv_or_surface(i)
                states_list.append(states)
            concat_states = join_tensor_states(states_list)
            return concat_states

        def _set_dof_targets(self, actions: CompatActionInput) -> None:
            if isinstance(actions, list):
                action_batch = actions
                if len(action_batch) == 1 and len(self.remotes) > 1:
                    action_batch = action_batch * len(self.remotes)
                if len(action_batch) != len(self.remotes):
                    raise ValueError(
                        f"Expected {len(self.remotes)} per-env actions, got {len(action_batch)} after normalization."
                    )
                for i, remote in enumerate(self.remotes):
                    remote.send(("set_dof_targets", ([action_batch[i]],)))
                self._check_error()
                return

            action_tensor = torch.as_tensor(actions)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)
            if action_tensor.shape[0] == 1 and len(self.remotes) > 1:
                for remote in self.remotes:
                    remote.send(("set_dof_targets", (action_tensor.clone(),)))
                self._check_error()
                return
            if action_tensor.shape[0] != len(self.remotes):
                raise ValueError(f"Expected tensor batch size {len(self.remotes)}, got {action_tensor.shape[0]}.")
            for i, remote in enumerate(self.remotes):
                remote.send(("set_dof_targets", (action_tensor[i : i + 1].clone(),)))
            self._check_error()

        def _simulate(self):
            for remote in self.remotes:
                remote.send(("simulate", (None,)))
            # ``simulate`` sends no reply, so a worker that dies mid-step
            # would stay silent until the next blocking ``recv`` — drain
            # the error queue now instead.
            self._check_error()

        def set_seed(self, seed: int) -> None:
            """Seed the parent + every worker process.

            ``BaseSimHandler.set_seed`` seeds the parent's NumPy / Torch /
            random — but each multiprocessing worker has its own process-
            level RNG that the parent's seed never reaches. Forward to
            each worker so a single ``env.reset(seed=42)`` makes the whole
            parallel rollout reproducible, not just the parent's view.
            """
            super().set_seed(seed)
            for remote in self.remotes:
                remote.send(("set_seed", (seed,)))
            self._check_error()

        def refresh_render(self):
            for remote in self.remotes:
                remote.send(("refresh_render", (None,)))
            self._check_error()

        def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
            self.remotes[0].send(("get_joint_names", (obj_name, sort)))
            return self._recv_or_surface(0)

        def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
            self.remotes[0].send(("get_body_names", (obj_name, sort)))
            return self._recv_or_surface(0)

        # the public names above are answered by worker 0 over the pipe; the private hooks the base
        # class would call are routed to them so the backend contract holds for the wrapper too
        def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
            return self.get_joint_names(obj_name, sort)

        def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
            return self.get_body_names(obj_name, sort)

        @property
        def device(self) -> torch.device:
            self.remotes[0].send(("device", (None,)))
            return self._recv_or_surface(0)

    return ParallelHandler
