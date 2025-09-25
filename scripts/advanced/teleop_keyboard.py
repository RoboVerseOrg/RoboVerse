from __future__ import annotations

import os
import sys
import time
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import pygame
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
from metasim.utils.math import matrix_from_euler, quat_apply, quat_from_matrix, quat_inv, quat_mul
from metasim.utils.teleop_utils import PygameKeyboardClient, process_kb_input

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

    keyboard_client = PygameKeyboardClient(width=670, height=870, title="Keyboard Control")

    for line, instruction in enumerate(keyboard_client.instructions):
        log.info(f"{line:2d}: {instruction}")

    step = 0
    running = True
    while running:
        # update keyboard events every frame
        running = keyboard_client.update()
        if not running:
            break

        if keyboard_client.is_pressed(pygame.K_ESCAPE):
            log.debug("Exiting simulation...")
            running = False
            break

        keyboard_client.draw_instructions()

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
        curr_ee_quat_local = quat_mul(quat_inv(robot_quat), curr_ee_quat)

        d_pos, d_rot_local, close_gripper = process_kb_input(keyboard_client, dpos=0.01, drot=0.05)
        d_pos_tensor = torch.tensor(d_pos, dtype=torch.float32, device=device)
        d_rot_tensor = torch.tensor(d_rot_local, dtype=torch.float32, device=device)

        # delta quaternion
        d_rot_mat_local = matrix_from_euler(d_rot_tensor.unsqueeze(0), "XYZ")
        d_quat_local = quat_from_matrix(d_rot_mat_local)[0]  # (4,)
        ee_pos_target = curr_ee_pos + d_pos_tensor
        ee_quat_target_local = quat_mul(curr_ee_quat_local, d_quat_local)

        # Solve IK using the modern IKSolver
        q_solution, ik_succ = ik_solver.solve_ik_batch(
            ee_pos_target.unsqueeze(0), ee_quat_target_local.unsqueeze(0), seed_q=curr_robot_q
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

    keyboard_client.close()
    env.close()
    sys.exit()


if __name__ == "__main__":
    main()
