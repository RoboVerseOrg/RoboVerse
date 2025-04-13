"""A module that implements parallel simulation using multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from copy import deepcopy
from functools import partial
from multiprocessing.connection import Connection

import torch
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import Action, EnvState, Extra, Obs, Reward, Success


def _worker(
    rank: int,
    remote: Connection,
    parent_remote: Connection,
    error_queue: mp.Queue,
    handler_class: type[BaseSimHandler],
):
    # NOTE(jigu): Set environment variables for ManiSkill2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    parent_remote.close()

    try:
        env: BaseSimHandler = handler_class()
        while True:
            cmd, data = remote.recv()
            if cmd == "launch":
                env.launch()
            elif cmd == "step":
                obs, reward, done, extra = env.step(data)
                remote.send((obs, reward, done, extra))
            elif cmd == "reset":
                obs, extra = env.reset()
                remote.send((obs, extra))
            elif cmd == "render":
                env.render()
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "set_states":
                env.set_states(data[0])
            elif cmd == "set_dof_targets":
                env.set_dof_targets(data[0], data[1])
            elif cmd == "set_pose":
                env.set_pose(data[0], data[1], data[2])
            elif cmd == "get_states":
                states = env.get_states()
                remote.send(states)
            # elif cmd == "get_vel":
            #     states = env.get_vel(data[0])
            #     remote.send(states)
            # elif cmd == "get_pos":
            #     states = env.get_pos(data[0])
            #     remote.send(states)
            # elif cmd == "get_rot":
            #     states = env.get_rot(data[0])
            #     remote.send(states)
            # elif cmd == "get_dof_pos":
            #     states = env.get_dof_pos(data[0], data[1])
            #     remote.send(states)
            elif cmd == "simulate":
                env.simulate()
            elif cmd == "get_reward":
                reward = env.get_reward()
                remote.send(reward)
            elif cmd == "get_object_joint_names":
                names = env.get_object_joint_names(data[0])
                remote.send(names)
            elif cmd == "handshake":
                # This is used to make sure that the environment is initialized
                # before sending any commands
                remote.send("handshake")
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
        def __new__(cls, scenario: ScenarioCfg):
            """If num_envs is one, simply use the original single-thread class."""
            if scenario.num_envs == 1:
                return base_cls(scenario)
            else:
                return super().__new__(cls)

        def __init__(self, scenario: ScenarioCfg):
            sub_scenario = deepcopy(scenario)
            sub_scenario.num_envs = 1
            super().__init__(scenario)

            self.waiting = False
            self.closed = False

            # Fork is not a thread safe method
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
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

        def step(self, actions: list[Action]) -> tuple[list[Obs], list[Reward], list[Success], list[Extra]]:
            for remote, action in zip(self.remotes, actions):
                remote.send(("step", ([action],)))
            obs_list, reward_list, done_list, extra_list = [], [], [], []
            for remote in self.remotes:
                obs, reward, done, extra = remote.recv()
                obs_list.append(obs[0])
                reward_list.append(reward[0])
                done_list.append(done[0])
                extra_list.append(extra[0])
            return obs_list, reward_list, done_list, extra_list

        def reset(self, env_ids: list[int] | None = None) -> tuple[list[Obs], list[Extra]]:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            for i in env_ids:
                self.remotes[i].send(("reset", (None,)))

            obs_list, extra_list = [], []
            for i in env_ids:
                obs, extra = self.remotes[i].recv()
                obs_list.append(obs[0])
                extra_list.append(extra[0])
            return obs_list, extra_list

        def close(self):
            if self.closed:
                return
            for remote in self.remotes:
                remote.send(("close", (None,)))
            for process in self.processes:
                process.join()
            self.closed = True

        def set_states(self, states: list[EnvState], env_ids: list[int] | None = None) -> None:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            for i in env_ids:
                self.remotes[i].send(("set_states", ([states[i]],)))

        def set_dof_targets(self, obj_name: str, targets: list[EnvState], env_ids: list[int] | None = None) -> None:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            for i in env_ids:
                self.remotes[i].send(("set_dof_targets", (obj_name, [targets[i]])))

        def set_pose(
            self,
            obj_name: str,
            pos: list[torch.Tensor],
            rot: list[torch.Tensor],
            env_ids: list[int] | None = None,
        ) -> None:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            for i in env_ids:
                self.remotes[i].send(("set_pose", (obj_name, [pos[i]], [rot[i]])))

        def get_states(self, env_ids: list[int] | None = None) -> list[EnvState]:
            if env_ids is None:
                env_ids = list(range(self.num_envs))

            for i in env_ids:
                self.remotes[i].send(("get_states", (None,)))

            states_list = []
            for i in env_ids:
                states = self.remotes[i].recv()
                states_list.append(states[0])
            return states_list

        # def get_vel(self, obj_name: str, env_ids: list[int] | None = None) -> list[EnvState]:
        #     if env_ids is None:
        #         env_ids = list(range(self.num_envs))

        #     for i in env_ids:
        #         self.remotes[i].send(("get_vel", (obj_name,)))

        #     states_list = []
        #     for i in env_ids:
        #         states = self.remotes[i].recv()
        #         states_list.append(states[0])
        #     return states_list

        # def get_pos(self, obj_name: str, env_ids: list[int] | None = None) -> list[EnvState]:
        #     if env_ids is None:
        #         env_ids = list(range(self.num_envs))

        #     for i in env_ids:
        #         self.remotes[i].send(("get_pos", (obj_name,)))

        #     # for i in env_ids:
        #     #     self.processes[i].join()
        #     # self._check_error()

        #     states_list = []
        #     for i in env_ids:
        #         states = self.remotes[i].recv()
        #         states_list.append(states[0])
        #     return states_list

        # def get_rot(self, obj_name: str, env_ids: list[int] | None = None) -> list[EnvState]:
        #     if env_ids is None:
        #         env_ids = list(range(self.num_envs))

        #     for i in env_ids:
        #         self.remotes[i].send(("get_rot", (obj_name,)))

        #     states_list = []
        #     for i in env_ids:
        #         states = self.remotes[i].recv()
        #         states_list.append(states[0])
        #     return states_list

        # def get_dof_pos(self, obj_name: str, joint_name: list[str], env_ids: list[int] | None = None) -> list[EnvState]:
        #     if env_ids is None:
        #         env_ids = list(range(self.num_envs))

        #     for i in env_ids:
        #         self.remotes[i].send((
        #             "get_dof_pos",
        #             (
        #                 obj_name,
        #                 joint_name,
        #             ),
        #         ))

        #     states_list = []
        #     for i in env_ids:
        #         states = self.remotes[i].recv()
        #         states_list.append(states[0])
        #     return states_list

        def simulate(self):
            for remote in self.remotes:
                remote.send(("simulate", (None,)))

        def refresh_render(self):
            log.error("Rendering not supported in parallel mode")

        def get_reward(self):
            log.error("get_reward not supported in parallel mode")

        def get_object_joint_names(self, obj_name: str) -> list[str]:
            self.remotes[0].send(("get_object_joint_names", (obj_name,)))
            names = self.remotes[0].recv()
            return names[0]

    return ParallelHandler
