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
from metasim.types import Action, DictEnvState, TensorState
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
                names = env.get_joint_names(data[0])
                remote.send(names)
            elif cmd == "get_body_names":
                names = env.get_body_names(data[0])
                remote.send(names)
            elif cmd == "set_dof_targets":
                env.set_dof_targets(data[0])
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
            if not self.closed:
                while not self.error_queue.empty():
                    error = self.error_queue.get()
                    log.error(f"Worker error: {error}")
                    raise RuntimeError(f"Worker error: {error}")

        def launch(self):
            for remote in self.remotes:
                remote.send(("launch", (None,)))
            self.waiting = False

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
                return

            if len(states) != len(env_ids):
                raise ValueError(
                    f"states nums {len(states)} and env_ids nums {len(env_ids)} are inconsistent, "
                    "in parallel mode, it must be one-to-one or single broadcast"
                )

            for local_idx, env_idx in enumerate(env_ids):
                self.remotes[env_idx].send(("set_states", ([states[local_idx]],)))

        def _get_states(self, env_ids: list[int] | None = None, **kwargs) -> TensorState:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            for i in env_ids:
                self.remotes[i].send(("get_states", (kwargs,)))
            states_list = []
            for i in env_ids:
                states = self.remotes[i].recv()
                states_list.append(states)
            concat_states = join_tensor_states(states_list)
            return concat_states

        def _set_dof_targets(self, actions: list[Action]) -> None:
            for i, remote in enumerate(self.remotes):
                remote.send(("set_dof_targets", (actions[i],)))

        def _simulate(self):
            for remote in self.remotes:
                remote.send(("simulate", (None,)))

        def refresh_render(self):
            log.error("Rendering not supported in parallel mode")

        def get_joint_names(self, obj_name: str) -> list[str]:
            self.remotes[0].send(("get_joint_names", (obj_name,)))
            names = self.remotes[0].recv()
            return names

        def get_body_names(self, obj_name: str) -> list[str]:
            self.remotes[0].send(("get_body_names", (obj_name,)))
            names = self.remotes[0].recv()
            return names

        @property
        def device(self) -> torch.device:
            self.remotes[0].send(("device", (None,)))
            return self.remotes[0].recv()

    return ParallelHandler
