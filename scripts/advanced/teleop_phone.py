from __future__ import annotations

import os
import sys
import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

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
from metasim.utils.math import quat_apply, quat_inv
from metasim.utils.teleop_utils import (
    TRANSFORMATION_MATRIX,
    PhoneServer,
    quaternion_to_rotation_matrix,
    transform_orientation,
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

    ## Phone Server
    host: str = "0.0.0.0"
    port: int = 8765
    update_dt: float = 1 / 30
    translation_step: float = 0.01

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

    sensor_server = PhoneServer(
        translation_step=args.translation_step, host=args.host, port=args.port, update_dt=args.update_dt
    )
    sensor_server.start_server()

    step = 0
    running = True
    while running:
        if not running:
            break

        q_world, delta_posi, gripper_flag = sensor_server.get_latest_data()
        q_world_new = transform_orientation(q_world)

        # compute target
        reorder_idx = env.handler.get_joint_reindex(scenario.robots[0].name)
        inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
        curr_robot_q = obs.robots[scenario.robots[0].name].joint_pos[:, inverse_reorder_idx]
        ee_idx = obs.robots[scenario.robots[0].name].body_names.index(scenario.robots[0].ee_body_name)
        robot_pos, robot_quat = obs.robots[scenario.robots[0].name].root_state[0, :7].split([3, 4])
        curr_ee_pos, curr_ee_quat = obs.robots[scenario.robots[0].name].body_state[0, ee_idx, :7].split([3, 4])
        curr_robot_q = curr_robot_q.to(device)
        curr_ee_pos = curr_ee_pos.to(device)
        curr_ee_quat = curr_ee_quat.to(device)
        robot_pos = robot_pos.to(device)
        robot_quat = robot_quat.to(device)

        curr_ee_pos = quat_apply(quat_inv(robot_quat), curr_ee_pos - robot_pos)

        ee_to_world_matpose = np.eye(4)
        ee_to_world_matpose[:3, :3] = quaternion_to_rotation_matrix(
            transform_orientation(curr_ee_quat.cpu().numpy().copy())
        )
        ee_to_world_matpose[:3, 3] = curr_ee_pos.cpu().numpy().copy()
        goal_to_ee_matpose = np.eye(4)
        goal_to_ee_matpose[:3, :3] = TRANSFORMATION_MATRIX
        goal_to_ee_matpose[:3, 3] = delta_posi
        goal = np.dot(ee_to_world_matpose, goal_to_ee_matpose)
        ee_pos_target = goal[:3, 3]
        ee_pos_target_tensor = torch.tensor(ee_pos_target, dtype=torch.float32, device=device)

        close_gripper = gripper_flag
        ee_quat_target_local_tensor = torch.tensor(q_world_new, dtype=torch.float32, device=device)

        # Solve IK using the modern IKSolver
        q_solution, ik_succ = ik_solver.solve_ik_batch(
            ee_pos_target_tensor.unsqueeze(0), ee_quat_target_local_tensor.unsqueeze(0), seed_q=curr_robot_q
        )

        # Process gripper command
        gripper_widths = process_gripper_command(
            torch.tensor(close_gripper, dtype=torch.float32, device=device), scenario.robots[0], device
        )

        # Compose joint action
        actions = ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)

        obs, reward, success, time_out, extras = env.step(actions)

        step += 1
        log.debug(f"Step {step}")

    sensor_server.close()
    env.close()
    sys.exit()


if __name__ == "__main__":
    main()
