import argparse
import os
import pickle as pkl

import numpy as np
import torch
from loguru import logger as log

from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler
from metasim.utils.demo_util import save_traj_file
from metasim.utils.demo_util.demo_util_v1 import (
    get_actions,
    get_all_states,
    get_init_states,
)
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source_path", type=str)
    args = parser.parse_args()
    return args


def tensor_to_list(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_to_list(v) for v in data]
    return data


def check_type(data):
    if not (isinstance(data, float) or isinstance(data, list) or isinstance(data, dict)):
        return False
    if isinstance(data, dict):
        for k, v in data.items():
            if not check_type(v):
                return False
    if isinstance(data, list):
        for v in data:
            if not check_type(v):
                return False
    return True


def do_it(task: BaseTaskCfg, robot: BaseRobotCfg, handler: BaseSimHandler, source_path: str, target_path: str):
    log.info(f"Loading trajectory from {source_path}")
    demo_original = pkl.load(open(source_path, "rb"))["demos"][robot.name]
    init_states = get_init_states(demo_original, len(demo_original), task, handler, robot)
    all_states = get_all_states(demo_original, task, handler, robot)
    all_actions = get_actions(demo_original, robot, handler)

    trajs_new = []
    for demo_idx, init_state in enumerate(init_states):
        actions = all_actions[demo_idx]
        states = all_states[demo_idx] if all_states is not None else None
        traj_new = {
            "actions": actions,
            "init_state": init_state,
            "states": states,
        }
        trajs_new.append(traj_new)

    trajs_new = tensor_to_list(trajs_new)
    final = {robot.name: trajs_new}
    log.info(f"Saving to {target_path}")
    save_traj_file(final, target_path)
    if not check_type(final):
        raise ValueError("Traj format is not valid")


def main():
    args = parse_args()
    env_class = get_sim_env_class(SimType.ISAACLAB)
    task = get_task(args.task)

    if args.source_path is None:
        if task.traj_filepath.endswith("_v2.pkl"):
            if os.path.exists(task.traj_filepath) and not args.overwrite:
                log.info(f"Trajectory {task.traj_filepath} is already in v2 format, skipping")
                log.info(
                    "If you don't have v2 trajectory file, please download from google drive, or remove the '_v2'"
                    " suffix for trajectory file in the task cfg and run this script again"
                )
                return
            else:
                task.traj_filepath = task.traj_filepath.replace("_v2.pkl", ".pkl")
                if os.path.exists(task.traj_filepath):
                    log.info("The v2 trajectory file does not exist, but the v1 trajectory file exists, converting...")
                else:
                    raise FileNotFoundError("The v1 trajectory file does not exist, please check the path")
        source_path = task.traj_filepath
    else:
        source_path = args.source_path

    if source_path.endswith("_v1.pkl"):
        target_path = source_path.replace("_v1.pkl", "_v2.pkl.gz")
    else:
        target_path = source_path.replace(".pkl", "_v2.pkl.gz")

    robot = get_robot(args.robot)
    scenario = ScenarioCfg(task=task, robot=robot, num_envs=1, headless=True)
    env = env_class(scenario)
    do_it(task, robot, env.handler, source_path, target_path)
    log.info("Convert successfully! Please update the task cfg to use the new trajectory file")
    env.close()


if __name__ == "__main__":
    main()
