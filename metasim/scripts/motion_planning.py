from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import argparse
import os
import time

import torch
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.math import quat_from_euler_xyz
from metasim.utils.setup_util import get_robot, get_sim_env_class

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
        task=args.task, robots=[args.robot], try_add_table=args.add_table, sim=args.sim, num_envs=num_envs
    )
    robot = get_robot(args.robot)

    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_open_q)

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)

    ## Main
    tic = time.time()
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robots[0], env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")
    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # Generate random actions
    random_gripper_widths = torch.rand((num_envs, len(robot.gripper_open_q)))
    random_gripper_widths = torch.tensor(robot.gripper_open_q) + random_gripper_widths * (
        torch.tensor(robot.gripper_close_q) - torch.tensor(robot.gripper_open_q)
    )

    ee_rot_target = torch.rand((num_envs, 3), device="cuda:0") * torch.pi
    ee_quat_target = quat_from_euler_xyz(ee_rot_target[..., 0], ee_rot_target[..., 1], ee_rot_target[..., 2])

    # Compute targets
    curr_robot_q = obs["joint_qpos"].cuda()
    robot_root_state = obs["robot_root_state"].cuda()
    robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]

    if robot.name == "iiwa":
        ee_pos_target = torch.distributions.Uniform(-0.5, 0.5).sample((num_envs, 3)).to("cuda:0")
        ee_pos_target[:, 2] += 0.5
    elif robot.name == "franka":
        ee_pos_target = torch.distributions.Uniform(-0.5, 0.5).sample((num_envs, 3)).to("cuda:0")
        ee_pos_target[:, 2] += 0.5
    elif robot.name == "sawyer":
        ee_pos_target = torch.stack(
            [
                torch.distributions.Uniform(-0.8, 0.8).sample((num_envs, 1)),
                torch.distributions.Uniform(-0.8, 0.8).sample((num_envs, 1)),
                torch.distributions.Uniform(0.2, 0.8).sample((num_envs, 1)),
            ],
            dim=-1,
        ).to("cuda:0")
    else:
        raise ValueError(f"Unsupported robot: {robot.name}")

    # Solve IK
    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    # Compose robot command
    q = curr_robot_q.clone()
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = random_gripper_widths

    actions = [
        {robot.name: {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))}}
        for i_env in range(num_envs)
    ]

    robot_joint_limits = scenario.robots[0].joint_limits
    # actions = [
    #     {
    #         "dof_pos_target": {
    #             joint_name: (
    #                 torch.rand(1).item() * (robot_joint_limits[joint_name][1] - robot_joint_limits[joint_name][0])
    #                 + robot_joint_limits[joint_name][0]
    #             )
    #             for joint_name in robot_joint_limits.keys()
    #         }
    #     }
    #     for _ in range(num_envs)
    # ]
    for _ in range(10):
        env.step(actions)
        env.render()

    step = 0
    while True:
        log.debug(f"Step {step}")
        actions[0][robot.name]["dof_pos_target"].update({
            joint_name: (
                torch.rand(1).item() * (robot_joint_limits[joint_name][1] - robot_joint_limits[joint_name][0])
                + robot_joint_limits[joint_name][0]
            )
            for joint_name in (args.joints if args.joints is not None else robot_joint_limits.keys())
        })
        env.step(actions)
        env.render()
        step += 1

    env.handler.close()


if __name__ == "__main__":
    main()
