from __future__ import annotations

import os
import sys
import time
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.render import RenderCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.ik_solver import IKSolver, process_gripper_command
from metasim.utils.math import quat_apply, quat_inv, quat_mul
from metasim.utils.teleop_utils import (
    R_HEADSET_TO_WORLD,
    XrClient,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    rotation_matrix_to_quaternion,
)

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "stack_cube"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    ## Handlers
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    headless: bool = False

    ## IK Solver
    ik_solver: Literal["curobo", "pyroki"] = "pyroki"
    no_gnd: bool = False

    ## XR Settings
    scale_factor: float = 1.75

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def main():
    task_cls = get_task_class(args.task)
    camera = PinholeCameraCfg(pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    traj_filepath = env.traj_filepath
    ## Data
    tic = time.time()
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(
        traj_filepath, scenario.robots[0], env.handler
    )  # XXX: only support one robot
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # Setup IK Solver
    ik_solver = IKSolver(scenario.robots[0], solver=args.ik_solver, no_gnd=args.no_gnd)

    xr_client = XrClient()

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
        reorder_idx = env.handler.get_joint_reindex(scenario.robots[0].name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        curr_robot_q = obs.robots[scenario.robots[0].name].joint_pos[:, inverse_reorder_idx]
        ee_idx = obs.robots[scenario.robots[0].name].body_names.index(scenario.robots[0].ee_body_name)
        robot_pos, robot_quat = obs.robots[scenario.robots[0].name].root_state[0, :7].split([3, 4])
        curr_ee_pos, curr_ee_quat = obs.robots[scenario.robots[0].name].body_state[0, ee_idx, :7].split([3, 4])
        curr_robot_q = curr_robot_q.to(device)
        robot_pos = robot_pos.to(device)
        robot_quat = robot_quat.to(device)
        curr_ee_pos = curr_ee_pos.to(device)
        curr_ee_quat = curr_ee_quat.to(device)

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
                delta_pos = (controller_pos - prev_controller_pos) * args.scale_factor
                delta_quat = quaternion_multiply(quaternion_inverse(prev_controller_quat), controller_quat)

        else:
            prev_controller_pos = None
            prev_controller_quat = None
            prev_ee_pos_local = curr_ee_pos_local
            prev_ee_quat_local = curr_ee_quat_local
            delta_pos = np.zeros(3)
            delta_quat = np.array([1, 0, 0, 0])

        delta_pos_tensor = torch.tensor(delta_pos, dtype=torch.float32, device=device)
        delta_quat_tensor = torch.tensor(delta_quat, dtype=torch.float32, device=device)

        ee_pos_target = prev_ee_pos_local + delta_pos_tensor
        ee_quat_target = quat_mul(delta_quat_tensor, prev_ee_quat_local)

        # Solve IK using the modern IKSolver
        q_solution, ik_succ = ik_solver.solve_ik_batch(
            ee_pos_target.unsqueeze(0), ee_quat_target.unsqueeze(0), seed_q=curr_robot_q
        )

        # Process gripper command
        gripper_widths = process_gripper_command(
            torch.tensor(gripper_close, dtype=torch.float32, device=device), scenario.robots[0], device
        )

        # Compose joint action
        actions = ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)

        obs, reward, success, time_out, extras = env.step(actions)

        step += 1
        log.debug(f"Step {step}")

    xr_client.close()
    env.close()
    sys.exit()


if __name__ == "__main__":
    main()
