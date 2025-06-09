from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.math import quat_apply, quat_inv
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from metasim.utils.teleop_utils import (
    TRANSFORMATION_MATRIX,
    PhoneServer,
    quaternion_to_rotation_matrix,
    transform_orientation,
)

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_envs: int = args.num_envs
    device = "cuda:0"
    task = get_task(args.task)
    robot = get_robot(args.robot)
    camera = PinholeCameraCfg(pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(task=task, robots=[robot], cameras=[camera], num_envs=num_envs)

    tic = time.time()
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    # data
    tic = time.time()
    assert os.path.exists(task.traj_filepath), f"Trajectory file: {task.traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(task, robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    # reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # cuRobo IKSysFont()
    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_open_q)

    sensor_server = PhoneServer(translation_step=0.01, host="0.0.0.0", port=8765, update_dt=1 / 30)
    sensor_server.start_server()

    obs = env.handler.get_observation()
    robot_ee_state = obs["robot_ee_state"].cuda()

    gripper_actuate_tensor = torch.tensor(robot.gripper_close_q, dtype=torch.float32, device=device)
    gripper_release_tensor = torch.tensor(robot.gripper_open_q, dtype=torch.float32, device=device)

    # todo: add mobile phone teleopration instructions

    step = 0
    running = True
    while running:
        if not running:
            break

        q_world, delta_posi, gripper_flag = sensor_server.get_latest_data()
        q_world_new = transform_orientation(q_world)

        obs = env.handler.get_observation()

        # compute target
        curr_robot_q = obs["joint_qpos"].cuda()
        robot_ee_state = obs["robot_ee_state"].cuda()
        robot_root_state = obs["robot_root_state"].cuda()
        robot_pos, robot_quat = robot_root_state[:, 0:3], robot_root_state[:, 3:7]
        curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]

        curr_ee_pos = quat_apply(quat_inv(robot_quat), curr_ee_pos - robot_pos)

        ee_to_world_matpose = np.eye(4)
        ee_to_world_matpose[:3, :3] = quaternion_to_rotation_matrix(
            transform_orientation(curr_ee_quat[0].cpu().numpy().copy())
        )
        ee_to_world_matpose[:3, 3] = curr_ee_pos.cpu().numpy().copy()
        goal_to_ee_matpose = np.eye(4)
        goal_to_ee_matpose[:3, :3] = TRANSFORMATION_MATRIX
        goal_to_ee_matpose[:3, 3] = delta_posi
        goal = np.dot(ee_to_world_matpose, goal_to_ee_matpose)
        ee_pos_target = goal[:3, 3]
        ee_pos_target_tensor = torch.tensor(ee_pos_target, dtype=torch.float32, device=device)

        close_gripper = gripper_flag
        gripper_q = gripper_actuate_tensor if close_gripper else gripper_release_tensor
        ee_quat_target_local_tensor = torch.tensor(q_world_new, dtype=torch.float32, device=device)

        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
        result = robot_ik.solve_batch(
            Pose(ee_pos_target_tensor.unsqueeze(0), ee_quat_target_local_tensor.unsqueeze(0)), seed_config=seed_config
        )

        ik_succ = result.success.squeeze(1)
        q = curr_robot_q.clone()  # shape: [num_envs, robot.num_joints]
        q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
        q[:, -ee_n_dof:] = gripper_q

        # XXX: this may not work for all simulators, since the order of joints may be different
        actions = [
            {"dof_pos_target": dict(zip(robot.joint_limits.keys(), q[i_env].tolist()))} for i_env in range(num_envs)
        ]
        env.step(actions)
        env.handler.render()

        step += 1
        log.debug(f"Step {step}")

    env.handler.close()
    sys.exit()


if __name__ == "__main__":
    main()
