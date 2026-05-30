"""A module that implements parallel simulation using multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
import platform
import sys
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

    try:
        env: BaseSimHandler = handler_class()
        while True:
            cmd, data = remote.recv()
            if cmd == "launch":
                env.launch()
            elif cmd == "render":
                env.render()
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
                names = env.get_body_names(data[0])
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
        env.close()


def ParallelSimWrapper(base_cls: type[BaseSimHandler]) -> type[BaseSimHandler]:
    """A parallel simulation handler that uses multiprocessing to run multiple simulations in parallel."""

    class ParallelHandler(BaseSimHandler):
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
            self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
            self.processes = []
            for rank in range(self.num_envs):
                work_remote = self.work_remotes[rank]
                remote = self.remotes[rank]
                args = (rank, work_remote, remote, self.error_queue, partial(base_cls, sub_scenario))
                # daemon=True: if the main process crashes, we should not cause things to hang
                process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
                process.start()
                self.processes.append(process)
                work_remote.close()

            # To make sure environments are initialized in all workers
            for remote in self.remotes:
                remote.send(("handshake", (None,)))
            for remote in self.remotes:
                remote.recv()

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
                self._check_error()
                raise RuntimeError(
                    f"Parallel worker {remote_idx} closed its pipe without reporting an error. "
                    f"Worker may have been killed externally."
                ) from err

        def launch(self):
            for remote in self.remotes:
                remote.send(("launch", (None,)))
            self.waiting = False
            self._check_error()

        def close(self):
            if self.closed:
                return
            for remote in self.remotes:
                remote.send(("close", (None,)))
            for process in self.processes:
                process.join()
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
            self._check_error()

        def set_seed(self, seed: int) -> None:
            """Seed the parent + every worker process.

            ``BaseSimHandler.set_seed`` seeds the parent's NumPy / Torch /
            random — but each multiprocessing worker has its own process-
            level RNG that the parent's seed never reaches. Forward to
            each worker so a single ``env.reset(seed=42)`` makes the whole
            parallel rollout reproducible, not just the parent's view.

            Note that deriving a distinct per-worker seed (e.g. ``seed + rank``)
            would also be defensible — we use the same seed everywhere
            because that matches what most gym wrappers do; if a caller
            wants per-env seeds they can call ``set_seed`` per remote
            themselves, which still works because each worker's
            ``env.set_seed`` is forwarded one-to-one.
            """
            super().set_seed(seed)
            for remote in self.remotes:
                remote.send(("set_seed", (seed,)))
            self._check_error()

        def refresh_render(self):
            log.error("Rendering not supported in parallel mode")

        def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
            self.remotes[0].send(("get_joint_names", (obj_name, sort)))
            return self._recv_or_surface(0)

        def get_body_names(self, obj_name: str) -> list[str]:
            self.remotes[0].send(("get_body_names", (obj_name,)))
            return self._recv_or_surface(0)

        @property
        def device(self) -> torch.device:
            self.remotes[0].send(("device", (None,)))
            return self._recv_or_surface(0)

    return ParallelHandler
