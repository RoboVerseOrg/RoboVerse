from __future__ import annotations

import os
import sys
import time
from typing import Literal

import cv2  # OpenCV for camera display
import numpy as np
import pygame
import torch
import tyro
from loguru import logger as log
from pynput import keyboard  # For keyboard input without pygame display
from rich.logging import RichHandler

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

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
    headless: bool = True

    ## IK Solver
    ik_solver: Literal["curobo", "pyroki"] = "pyroki"
    no_gnd: bool = False

    ## Display
    display_camera: bool = True  # Whether to display camera view in real-time
    display_width: int = 1200  # Display window width (adjusted for dual camera split-screen)
    display_height: int = 600  # Display window height

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def display_camera_observation_opencv(obs, width, height):
    """Display camera observations using OpenCV - split screen for two cameras."""
    if not hasattr(obs, "cameras") or len(obs.cameras) == 0:
        # Create a blank dark gray image if no camera data
        blank_img = np.full((height, width, 3), 50, dtype=np.uint8)
        cv2.imshow("Camera View - Real-time Robot View", blank_img)
        return

    camera_names = list(obs.cameras.keys())
    num_cameras = len(camera_names)

    # Create display image
    display_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Split the display area into two halves
    half_width = width // 2
    half_height = height

    # Display first camera on the left
    if num_cameras >= 1:
        camera_name_1 = camera_names[0]
        rgb_data_1 = obs.cameras[camera_name_1].rgb

        if rgb_data_1 is not None:
            # Debug: Print RGB data range and info
            if isinstance(rgb_data_1, torch.Tensor):
                rgb_np_1 = rgb_data_1.cpu().numpy()
                if rgb_np_1.max() <= 1.0:
                    rgb_np_1 = (rgb_np_1 * 255).astype(np.uint8)
            else:
                rgb_np_1 = np.array(rgb_data_1)

            # Handle different shapes
            if len(rgb_np_1.shape) == 4:  # (N, C, H, W)
                rgb_np_1 = rgb_np_1[0]  # Take first environment
            if len(rgb_np_1.shape) == 3 and rgb_np_1.shape[0] == 3:  # (C, H, W)
                rgb_np_1 = np.transpose(rgb_np_1, (1, 2, 0))  # (H, W, C)

            try:
                # Resize image to fit the left half
                if rgb_np_1.shape[:2] != (half_height, half_width):
                    rgb_resized_1 = cv2.resize(rgb_np_1, (half_width, half_height))
                else:
                    rgb_resized_1 = rgb_np_1
                display_img[:, :half_width] = rgb_resized_1
            except Exception as e:
                log.warning(f"Error displaying camera 1 image: {e}")
                # Draw error rectangle on left half
                cv2.rectangle(display_img, (0, 0), (half_width, half_height), (50, 50, 100), -1)

    # Display second camera on the right
    if num_cameras >= 2:
        camera_name_2 = camera_names[1]
        rgb_data_2 = obs.cameras[camera_name_2].rgb

        if rgb_data_2 is not None:
            # Debug: Print RGB data range and info
            if isinstance(rgb_data_2, torch.Tensor):
                rgb_np_2 = rgb_data_2.cpu().numpy()
                if rgb_np_2.max() <= 1.0:
                    rgb_np_2 = (rgb_np_2 * 255).astype(np.uint8)
            else:
                rgb_np_2 = np.array(rgb_data_2)

            # Handle different shapes
            if len(rgb_np_2.shape) == 4:  # (N, C, H, W)
                rgb_np_2 = rgb_np_2[0]  # Take first environment
            if len(rgb_np_2.shape) == 3 and rgb_np_2.shape[0] == 3:  # (C, H, W)
                rgb_np_2 = np.transpose(rgb_np_2, (1, 2, 0))  # (H, W, C)

            try:
                # Resize image to fit the right half
                if rgb_np_2.shape[:2] != (half_height, half_width):
                    rgb_resized_2 = cv2.resize(rgb_np_2, (half_width, half_height))
                else:
                    rgb_resized_2 = rgb_np_2
                display_img[:, half_width:] = rgb_resized_2
            except Exception as e:
                log.warning(f"Error displaying camera 2 image: {e}")
                # Draw error rectangle on right half
                cv2.rectangle(display_img, (half_width, 0), (width, half_height), (50, 50, 100), -1)

    # Fill areas if cameras are missing
    if num_cameras == 0:
        display_img.fill(50)  # Dark gray
    elif num_cameras == 1:
        # Fill right half with darker gray
        display_img[:, half_width:] = 30

    # Show the combined image
    cv2.imshow("Camera View - Real-time Robot View", display_img)

    # Handle key events for OpenCV window
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        return False  # Signal to exit
    return True  # Continue running


def main():
    task_cls = get_task_class(args.task)
    # Create two cameras with different viewpoints
    camera1 = PinholeCameraCfg(name="camera_1", pos=(1.0, -1.0, 1.0), look_at=(0.0, 0.0, 0.0))
    camera2 = PinholeCameraCfg(name="camera_2", pos=(1.5, -0.2, 0.5), look_at=(0.0, 0.0, 0.0))
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera1, camera2],
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs

    tic = time.time()
    device = torch.device("cpu")
    env = task_cls(scenario, device=device)
    from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

    env = TaskViserWrapper(env)
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

    # Setup camera display and keyboard control
    keyboard_client = None
    space_pressed = False  # Track space key state for gripper control

    # Keyboard state tracking for pynput
    key_states = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "e": False,
        "d": False,
        "q": False,
        "w": False,
        "a": False,
        "s": False,
        "z": False,
        "x": False,
        "space": False,
    }

    def on_key_press(key):
        """Handle key press events"""
        nonlocal running, space_pressed
        try:
            key_name = key.char.lower() if hasattr(key, "char") and key.char else str(key).split(".")[-1]
            if key_name in key_states:
                key_states[key_name] = True
            if key_name == "space":
                space_pressed = True
        except AttributeError:
            if str(key) == "<Key.esc>":
                log.debug("ESC pressed, exiting simulation...")
                running = False

    def on_key_release(key):
        """Handle key release events"""
        nonlocal space_pressed
        try:
            key_name = key.char.lower() if hasattr(key, "char") and key.char else str(key).split(".")[-1]
            if key_name in key_states:
                key_states[key_name] = False
            if key_name == "space":
                space_pressed = False
        except AttributeError:
            pass

    # Setup keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)

    if args.display_camera:
        # Initialize OpenCV window for camera display
        cv2.namedWindow("Camera View - Real-time Robot View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View - Real-time Robot View", args.display_width, args.display_height)
        log.info(f"OpenCV camera display window initialized ({args.display_width}x{args.display_height})")
        keyboard_listener.start()  # Start keyboard listener
    else:
        # Setup keyboard interface when not displaying camera
        keyboard_client = PygameKeyboardClient(width=670, height=870, title="Keyboard Control")

        for line, instruction in enumerate(keyboard_client.instructions):
            log.info(f"{line:2d}: {instruction}")
        keyboard_listener.start()  # Start keyboard listener

    step = 0
    running = True
    while running:
        # Handle keyboard events
        if keyboard_client is not None:
            # update keyboard events every frame
            running = keyboard_client.update()
            if not running:
                break

            if keyboard_client.is_pressed(pygame.K_ESCAPE):
                log.debug("Exiting simulation...")
                running = False
                break

            keyboard_client.draw_instructions()
        # Keyboard input is now handled by pynput listener in background

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

        if keyboard_client is not None:
            d_pos, d_rot_local, close_gripper = process_kb_input(keyboard_client, dpos=0.01, drot=0.05)
        else:
            # Handle keyboard input using pynput key states
            d_pos = [0.0, 0.0, 0.0]
            d_rot_local = [0.0, 0.0, 0.0]
            close_gripper = 0

            # Movement controls (pynput key mapping)
            if key_states["up"]:
                d_pos[0] += 0.01  # Move +X
            if key_states["down"]:
                d_pos[0] -= 0.01  # Move -X
            if key_states["left"]:
                d_pos[1] += 0.01  # Move +Y
            if key_states["right"]:
                d_pos[1] -= 0.01  # Move -Y
            if key_states["e"]:
                d_pos[2] += 0.01  # Move +Z
            if key_states["d"]:
                d_pos[2] -= 0.01  # Move -Z

            # Rotation controls (pynput key mapping)
            if key_states["q"]:
                d_rot_local[0] += 0.05  # Roll +
            if key_states["w"]:
                d_rot_local[0] -= 0.05  # Roll -
            if key_states["a"]:
                d_rot_local[1] += 0.05  # Pitch +
            if key_states["s"]:
                d_rot_local[1] -= 0.05  # Pitch -
            if key_states["z"]:
                d_rot_local[2] += 0.05  # Yaw +
            if key_states["x"]:
                d_rot_local[2] -= 0.05  # Yaw -

            # Gripper controls (space_pressed tracks key state)
            # Note: space_pressed=True means close gripper, False means open
            close_gripper = 1 if space_pressed else 0
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

        # Process gripper command (convert boolean to float for consistency)
        gripper_widths = process_gripper_command(
            torch.tensor(float(close_gripper), dtype=torch.float32, device=device), scenario.robots[0], device
        )

        # Compose joint action
        actions = ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)

        obs, reward, success, time_out, extras = env.step(actions)

        # Display camera observation if requested
        if args.display_camera:
            running = display_camera_observation_opencv(obs, args.display_width, args.display_height)
            if not running:
                break

        step += 1

    # Close OpenCV camera display window if it exists
    if args.display_camera:
        cv2.destroyAllWindows()
        log.info("OpenCV camera display window closed")

    # Stop keyboard listener
    keyboard_listener.stop()
    log.info("Keyboard listener stopped")

    # Close keyboard client if it exists
    if keyboard_client is not None:
        keyboard_client.close()

    env.close()
    sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
