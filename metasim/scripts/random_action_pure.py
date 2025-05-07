from __future__ import annotations

import argparse
import os

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--sim", type=str, default="isaaclab", choices=["isaaclab", "isaacgym", "mujoco"])
    parser.add_argument("--task", type=str, default="PickCube")
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        choices=[
            "franka",
            "ur5e_2f85",
            "sawyer",
            "franka_with_gripper_extension",
            "h1_2_without_hand",
            "h1",
            "h1_simple_hand",
            "h1_hand",
            "sawyer_mujoco",
            "fetch",
        ],
    )
    parser.add_argument("--add_table", action="store_true")
    parser.add_argument(
        "--joints", default=None, nargs="+", type=str, help="Joints to randomize, if None, randomize all joints"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_envs: int = args.num_envs
    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        try_add_table=args.add_table,
        sim=args.sim,
        num_envs=num_envs,
        # headless=True, # NOTE: uncomment this to run in headless mode with no camera, maximizes performance
        # cameras=[],
    )

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    ## Main
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, _, _ = get_traj(scenario.task, scenario.robot, env.handler)
    if len(init_states) < num_envs:
        init_states = init_states * (num_envs // len(init_states)) + init_states[: num_envs % len(init_states)]
    else:
        init_states = init_states[:num_envs]
    env.reset(states=init_states)

    robot_joint_limits = scenario.robot.joint_limits
    step = 0
    while True:
        log.debug(f"Step {step}")
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
        env.render()
        step += 1

    env.handler.close()


if __name__ == "__main__":
    main()
