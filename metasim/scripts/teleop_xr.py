from __future__ import annotations

import argparse
import os
import sys
import time

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
from metasim.utils.math import quat_apply, quat_inv, quat_mul
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from metasim.utils.teleop_utils import (
    R_HEADSET_TO_WORLD,
    XrClient,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    rotation_matrix_to_quaternion,
)

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
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
    scale_factor = 1.75

    tic = time.time()
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    # data
    tic = time.time()
    assert os.path.exists(task.traj_filepath), (
        f"Trajectory file: {task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(task, robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    # reset before first step
    tic = time.time()
    states, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # cuRobo IKSysFont()
    *_, robot_ik = get_curobo_models(robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(robot.gripper_open_q)

    xr_client = XrClient()

    gripper_actuate_tensor = torch.tensor(robot.gripper_close_q, dtype=torch.float32, device=device)
    gripper_release_tensor = torch.tensor(robot.gripper_open_q, dtype=torch.float32, device=device)

    step = 0
    running = True
    prev_controller_pos = None
    prev_controller_quat = None
    prev_ee_pos_local = None
    prev_ee_quat_local = None
    while running:
        if not running:
            break

        if xr_client.get_button_state_by_name("B"):
            log.debug("Exiting simulation...")
            running = False
            break

        # compute target
        reorder_idx = env.handler.get_joint_reindex(robot.name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        curr_robot_q = states.robots[robot.name].joint_pos[:, inverse_reorder_idx]
        ee_idx = states.robots[robot.name].body_names.index(robot.ee_body_name)
        robot_pos, robot_quat = states.robots[robot.name].root_state[0, :7].split([3, 4])
        curr_ee_pos, curr_ee_quat = (
            states.robots[robot.name].body_state[0, ee_idx, :7].split([3, 4])
        )

        curr_ee_pos_local = quat_apply(quat_inv(robot_quat), curr_ee_pos - robot_pos)
        curr_ee_quat_local = quat_mul(quat_inv(robot_quat), curr_ee_quat)

        right_grip = xr_client.get_key_value_by_name("right_grip")
        teleop_active = right_grip > 0.5
        gripper_close = xr_client.get_key_value_by_name("right_trigger") > 0.5

        if teleop_active:
            controller_pose = xr_client.get_pose_by_name("right_controller")
            controller_pos = np.array(controller_pose[:3])
            controller_quat = np.array(
                [
                    controller_pose[6],
                    controller_pose[3],
                    controller_pose[4],
                    controller_pose[5],
                ],
            )

            controller_pos = R_HEADSET_TO_WORLD @ controller_pos
            R_quat = rotation_matrix_to_quaternion(R_HEADSET_TO_WORLD)
            # controller_quat = quaternion_multiply(R_quat, controller_quat)
            controller_quat = quaternion_multiply(
                quaternion_multiply(R_quat, controller_quat),
                quaternion_conjugate(R_quat),
            )
            if prev_controller_pos is None:
                prev_controller_pos = controller_pos
                prev_controller_quat = controller_quat
                prev_ee_pos_local = curr_ee_pos_local
                prev_ee_quat_local = curr_ee_quat_local
                delta_pos = np.zeros(3)
                delta_quat = np.array([1, 0, 0, 0])
            else:
                delta_pos = (controller_pos - prev_controller_pos) * scale_factor
                delta_quat = quaternion_multiply(
                    quaternion_inverse(prev_controller_quat), controller_quat
                )

        else:
            prev_controller_pos = None
            prev_controller_quat = None
            prev_ee_pos_local = curr_ee_pos_local
            prev_ee_quat_local = curr_ee_quat_local
            delta_pos = np.zeros(3)
            delta_quat = np.array([1, 0, 0, 0])

        delta_pos_tensor = torch.tensor(delta_pos, dtype=torch.float32, device=device)
        delta_quat_tensor = torch.tensor(delta_quat, dtype=torch.float32, device=device)
        gripper_q = gripper_actuate_tensor if gripper_close else gripper_release_tensor

        ee_pos_target = prev_ee_pos_local + delta_pos_tensor
        ee_quat_target = quat_mul(delta_quat_tensor, prev_ee_quat_local)

        seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])
        result = robot_ik.solve_batch(
            Pose(ee_pos_target.unsqueeze(0), ee_quat_target.unsqueeze(0)), seed_config=seed_config
        )
        ik_succ = result.success.squeeze(1)
        q = curr_robot_q.clone()  # shape: [num_envs, robot.num_joints]
        q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
        q[:, -ee_n_dof:] = gripper_q

        # XXX: this may not work for all simulators, since the order of joints may be different
        actions = [
            {robot.name: {"dof_pos_target": dict(zip(robot.joint_limits.keys(), q[i_env].tolist()))}}
            for i_env in range(num_envs)
        ]
        states, _, _, _, _ = env.step(actions)

        step += 1
        log.debug(f"Step {step}")

    xr_client.close()
    env.handler.close()
    sys.exit()


if __name__ == "__main__":
    main()
