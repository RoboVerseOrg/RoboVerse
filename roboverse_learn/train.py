from __future__ import annotations

import argparse
import os
import time

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.cfg.scenario import ScenarioCfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--sim",
        type=str,
        default="isaaclab",
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    parser.add_argument(
        "--render",
        type=str,
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_envs: int = args.num_envs

    import torch

    from metasim.cfg.sensors import PinholeCameraCfg
    from metasim.constants import SimType
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task

    task = get_task(args.task)
    robot = get_robot(args.robot)
    camera = PinholeCameraCfg(
        name="camera",
        data_types=["rgb", "depth"],
        width=256,
        height=256,
        pos=(1.5, 0.0, 1.5),
        look_at=(0.0, 0.0, 0.0),
    )
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        cameras=[camera],
        sim=args.sim,
        renderer=args.render,
        num_envs=args.num_envs,
        try_add_table=True,
    )

    tic = time.time()
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    ## Data
    tic = time.time()
    assert os.path.exists(task.traj_filepath), f"Trajectory file: {task.traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(task, robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    # save_obs(obs, 0)

    step = 0
    while True:
        log.debug(f"Step {step}")
        robot_joint_limits = robot.joint_limits
        actions = [
            {
                "dof_pos_target": {
                    joint_name: (
                        torch.rand(1).item() * (robot_joint_limits[joint_name][1] - robot_joint_limits[joint_name][0])
                        + robot_joint_limits[joint_name][0]
                    )
                    for joint_name in robot_joint_limits.keys()
                }
            }
            for _ in range(num_envs)
        ]
        env.step(actions)
        env.handler.refresh_render()
        step += 1

    env.close()


if __name__ == "__main__":
    main()
