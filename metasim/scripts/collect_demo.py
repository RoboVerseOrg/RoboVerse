## XXX:
## 1. Currently we use global variables to track the progress, which is not a good practice.
## TODO:
## 1. Check the missing demos first, then collect the missing part? In this way, there won't be any global variables
## 2. Or, combine tot_success, tot_give_up, global_step and pbar into a seperate class, maybe called ProgressManager. In this way, there won't be any global variables

from __future__ import annotations

#########################################
## Setup logging
#########################################
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


#########################################
### Add command line arguments
#########################################
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import tyro

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg


@dataclass
class Args:
    random: RandomizationCfg
    """Domain randomization options"""
    render: RenderCfg
    """Renderer options"""
    task: str
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "isaaclab"
    """Simulator backend"""
    demo_start_idx: int | None = None
    """The index of the first demo to collect, None for all demos"""
    max_demo_idx: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    retry_num: int = 0
    """Number of retries for a failed demo"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""
    split: Literal["train", "val", "test", "all"] = "all"
    """Split to collect"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    scene: str | None = None
    """Scene name"""
    run_all: bool = False
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""

    def __post_init__(self):
        assert self.run_all or self.run_unfinished or self.run_failed, (
            "At least one of run_all, run_unfinished, or run_failed must be True"
        )
        if self.random.table and not self.table:
            log.warning("Cannot enable table randomization without a table, disabling table randomization")
            self.random.table = False

        if self.max_demo_idx is None:
            self.max_demo_idx = math.inf

        if self.demo_start_idx is None:
            self.demo_start_idx = 0

        log.info(f"Args: {self}")


args = tyro.cli(Args)


#########################################
### Import packages
#########################################
import multiprocessing as mp
import os

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
from tqdm.rich import tqdm_rich as tqdm

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu


###########################################################
## Utils
###########################################################
def get_actions(all_actions, env: EnvWrapper[BaseSimHandler], demo_idxs: list[int], robot: BaseRobotCfg):
    action_idxs = env.episode_length_buf

    actions = [
        all_actions[demo_idx][action_idx] if action_idx < len(all_actions[demo_idx]) else all_actions[demo_idx][-1]
        for demo_idx, action_idx in zip(demo_idxs, action_idxs)
    ]
    return actions


def get_run_out(all_actions, env: EnvWrapper[BaseSimHandler], demo_idxs: list[int]) -> list[bool]:
    action_idxs = env.episode_length_buf
    run_out = [action_idx >= len(all_actions[demo_idx]) for demo_idx, action_idx in zip(demo_idxs, action_idxs)]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo)


###########################################################
## Global Variables
###########################################################
global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0


###########################################################
## Core Utils
###########################################################
class DemoCollector:
    def __init__(self, handler):
        from metasim.sim import BaseSimHandler

        assert isinstance(handler, BaseSimHandler)
        self.handler = handler
        self.cache: dict[int, list[dict]] = {}
        self.save_request_queue = mp.Queue()
        self.save_proc = mp.Process(target=save_demo_mp, args=(self.save_request_queue,))
        self.save_proc.start()

        TaskName = self.handler.task.__class__.__name__.replace("Cfg", "")
        if args.cust_name is not None:
            additional_str = "-" + str(args.cust_name)
        else:
            additional_str = ""
        self.base_save_dir = (
            f"roboverse_demo/demo_{args.sim}/{TaskName}-Level{args.random.level}{additional_str}/robot-{args.robot}"
        )

    def create(self, demo_idx: int, data_dict: dict):
        assert demo_idx not in self.cache
        assert isinstance(demo_idx, int)
        self.cache[demo_idx] = [data_dict]

    def add(self, demo_idx: int, data_dict: dict):
        if data_dict is None:
            log.warning("Skipping adding obs to DemoCollector because obs is None")
        assert demo_idx in self.cache
        self.cache[demo_idx].append(deepcopy(tensor_to_cpu(data_dict)))

    def save(self, demo_idx: int):
        assert demo_idx in self.cache

        save_dir = os.path.join(self.base_save_dir, f"demo_{demo_idx:04d}")
        if os.path.exists(os.path.join(save_dir, "status.txt")):
            os.remove(os.path.join(save_dir, "status.txt"))

        os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving demo {demo_idx} to {save_dir}")

        ## Option 1: Save immediately, blocking and slower

        from metasim.utils.save_util import save_demo

        save_demo(save_dir, self.cache[demo_idx])

        ## Option 2: Save in a separate process, non-blocking, not friendly to KeyboardInterrupt, TODO: fix
        ## TODO: see https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html#preventing-memory-leaks-in-the-simulator
        # self.save_request_queue.put({"demo": self.cache[demo_idx], "save_dir": save_dir})

    def mark_fail(self, demo_idx: int):
        assert demo_idx in self.cache
        save_dir = os.path.join(self.base_save_dir, f"demo_{demo_idx:04d}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "status.txt"), "w+") as f:
            f.write("failed")

    def delete(self, demo_idx: int):
        assert demo_idx in self.cache
        del self.cache[demo_idx]

    def final(self):
        self.save_request_queue.put(None)  # signal to save_demo_mp to exit
        self.save_proc.join()
        assert self.cache == {}


def should_skip(log_dir):
    if args.run_all:
        return False
    if args.run_unfinished and not os.path.exists(os.path.join(log_dir, "status.txt")):
        return False
    if args.run_failed and (
        not os.path.exists(os.path.join(log_dir, "status.txt"))
        or open(os.path.join(log_dir, "status.txt")).read() != "success"
    ):
        return False
    return True


def is_status_success(log_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(log_dir, "status.txt"))
        and open(os.path.join(log_dir, "status.txt")).read() == "success"
    )


class DemoIndexer:
    def __init__(self, save_root_dir: str, start_idx: int, end_idx: int, pbar: tqdm):
        self.save_root_dir = save_root_dir
        self._next_idx = start_idx
        self.end_idx = end_idx
        self.pbar = pbar
        self._skip_if_should()

    @property
    def next_idx(self):
        return self._next_idx

    def _skip_if_should(self):
        while should_skip(f"{self.save_root_dir}/demo_{self._next_idx:04d}"):
            global global_step, tot_success, tot_give_up
            if is_status_success(f"{self.save_root_dir}/demo_{self._next_idx:04d}"):
                tot_success += 1
            else:
                tot_give_up += 1
            self.pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
            self.pbar.update(1)
            log.info(f"Demo {self._next_idx} already exists, skipping...")
            self._next_idx += 1

    def move_on(self):
        self._next_idx += 1
        self._skip_if_should()


###########################################################
## Main
###########################################################
def main():
    global global_step, tot_success, tot_give_up
    handler_class = get_sim_env_class(SimType(args.sim))
    task = get_task(args.task)
    robot = get_robot(args.robot)
    camera = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(
        task=task,
        robots=[robot],
        scene=args.scene,
        cameras=[camera],
        random=args.random,
        try_add_table=args.table,
        render=args.render,
        split=args.split,
        sim=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    env = handler_class(scenario)

    ## Data
    assert os.path.exists(task.traj_filepath), f"Trajectory file does not exist: {task.traj_filepath}"
    init_states, all_actions, all_states = get_traj(task, robot, env.handler)

    tot_demo = len(all_actions)
    if args.split == "train":
        init_states = init_states[: int(tot_demo * 0.9)]
        all_actions = all_actions[: int(tot_demo * 0.9)]
        all_states = all_states[: int(tot_demo * 0.9)]
    elif args.split == "val" or args.split == "test":
        init_states = init_states[int(tot_demo * 0.9) :]
        all_actions = all_actions[int(tot_demo * 0.9) :]
        all_states = all_states[int(tot_demo * 0.9) :]

    n_demo = len(all_actions)
    log.info(f"Collecting from {args.split} split, {n_demo} out of {tot_demo} demos")

    ########################################################
    ## Main
    ########################################################
    if args.max_demo_idx > n_demo:
        log.warning(
            f"Max demo {args.max_demo_idx} is greater than the number of demos in the dataset {n_demo}, using {n_demo}"
        )
    max_demo = min(args.max_demo_idx, n_demo)
    try_num = args.retry_num + 1

    ###########################################################
    ##   State Machine Diagram
    ###########################################################
    ##   CollectingDemo --> Success: env success
    ##   CollectingDemo --> Timeout: env timeout or run_out
    ##
    ##   Success --> FinalizeCollectingDemo
    ##
    ##   FinalizeCollectingDemo --> NextDemo: run_out or steps_after_success >= args.tot_steps_after_success
    ##
    ##   Timeout --> CollectingDemo: failure_count < try_num
    ##   Timeout --> NextDemo: failure_count >= try_num
    ##
    ##   NextDemo --> CollectingDemo: next_demo_idx < max_demo
    ##   NextDemo --> Finished: next_demo_idx >= max_demo
    ##
    ##   All Finished --> Exit

    ## Setup
    collector = DemoCollector(env.handler)
    pbar = tqdm(total=max_demo - args.demo_start_idx, desc="Collecting demos")

    ## State variables
    failure_count = [0] * env.handler.num_envs
    steps_after_success = [0] * env.handler.num_envs
    finished = [False] * env.handler.num_envs
    TaskName = env.handler.task.__class__.__name__.replace("Cfg", "")

    if args.cust_name is not None:
        additional_str = "-" + str(args.cust_name)
    else:
        additional_str = ""
    demo_indexer = DemoIndexer(
        save_root_dir=f"roboverse_demo/demo_{args.sim}/{TaskName}-Level{args.random.level}{additional_str}/robot-{args.robot}",
        start_idx=0,
        end_idx=max_demo,
        pbar=pbar,
    )
    demo_idxs = []
    for demo_idx in range(env.handler.num_envs):
        demo_idxs.append(demo_indexer.next_idx)
        demo_indexer.move_on()
    log.info(f"Initialize with demo idxs: {demo_idxs}")

    ## Reset before first step
    obs, extras = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs])
    obs = state_tensor_to_nested(env.handler, obs)

    ## Initialize
    for env_id, demo_idx in enumerate(demo_idxs):
        collector.create(demo_idx, obs[env_id])

    ## Main Loop
    while not all(finished):
        pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
        actions = get_actions(all_actions, env, demo_idxs, robot)
        obs, reward, success, time_out, extras = env.step(actions)
        obs = state_tensor_to_nested(env.handler, obs)
        run_out = get_run_out(all_actions, env, demo_idxs)

        for env_id in range(env.handler.num_envs):
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            collector.add(demo_idx, obs[env_id])

        for env_id in success.nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            ## CollectingDemo --> Success
            demo_idx = demo_idxs[env_id]
            if steps_after_success[env_id] == 0:
                log.info(f"Demo {demo_idx} in Env {env_id} succeeded!")
                tot_success += 1
                pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

            ## Success --> FinalizeCollectingDemo
            if not run_out[env_id] and steps_after_success[env_id] < args.tot_steps_after_success:
                steps_after_success[env_id] += 1
            else:
                ## FinalizeCollectingDemo --> NextDemo
                steps_after_success[env_id] = 0
                collector.save(demo_idx)
                collector.delete(demo_idx)

                if demo_indexer.next_idx < max_demo:
                    ## NextDemo --> CollectingDemo
                    demo_idxs[env_id] = demo_indexer.next_idx
                    obs, _ = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs], env_ids=[env_id])
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(demo_indexer.next_idx, obs[env_id])
                    demo_indexer.move_on()
                    run_out[env_id] = False
                else:
                    ## NextDemo --> Finished
                    finished[env_id] = True

        for env_id in (time_out | torch.tensor(run_out, device=time_out.device)).nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            ## CollectingDemo --> Timeout
            demo_idx = demo_idxs[env_id]
            log.info(f"Demo {demo_idx} in Env {env_id} timed out!")
            collector.mark_fail(demo_idx)
            collector.delete(demo_idx)
            failure_count[env_id] += 1

            if failure_count[env_id] < try_num:
                ## Timeout --> CollectingDemo
                log.info(f"Demo {demo_idx} failed {failure_count[env_id]} times, retrying...")
                obs, _ = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs], env_ids=[env_id])
                obs = state_tensor_to_nested(env.handler, obs)
                collector.create(demo_idx, obs[env_id])
            else:
                ## Timeout --> NextDemo
                log.error(f"Demo {demo_idx} failed too many times, giving up")
                failure_count[env_id] = 0
                tot_give_up += 1
                pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                if demo_indexer.next_idx < max_demo:
                    ## NextDemo --> CollectingDemo
                    demo_idxs[env_id] = demo_indexer.next_idx
                    obs, _ = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs], env_ids=[env_id])
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(demo_indexer.next_idx, obs[env_id])
                    demo_indexer.move_on()
                else:
                    ## NextDemo --> Finished
                    finished[env_id] = True

        global_step += 1

    log.info("Finalizing")
    collector.final()
    env.close()


if __name__ == "__main__":
    main()
