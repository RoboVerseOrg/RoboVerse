"""
Phone Teleoperation Script using Lerobot Phone Teleoperator

This script integrates lerobot's phone teleoperator with RoboVerse simulation environment.
It supports both iOS (via HEBI Mobile I/O app) and Android (via WebXR) phones.

Usage:
    # For Android phone:
    python scripts/advanced/teleop_phone_lerobot.py --task stack_cube --robot franka --phone-os android

    # For iOS phone:
    python scripts/advanced/teleop_phone_lerobot.py --task stack_cube --robot franka --phone-os ios

Requirements:
    - Install lerobot with phone support: pip install lerobot[phone]
    - For iOS: Install and open HEBI Mobile I/O app on your iPhone
    - For Android: Open the URL printed by the script in your phone's browser

The script will:
    1. Connect to your phone
    2. Prompt for calibration (hold phone in correct orientation)
    3. Start teleoperation loop
    4. Control robot end-effector based on phone pose and inputs
"""

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

# Import lerobot phone teleoperator
try:
    from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
    from lerobot.teleoperators.phone.teleop_phone import Phone
    from lerobot.utils.rotation import Rotation

    LEROBOT_AVAILABLE = True
except ImportError as e:
    log.error(f"Failed to import lerobot phone teleoperator: {e}")
    log.error("Please install lerobot with phone support: pip install lerobot[phone]")
    LEROBOT_AVAILABLE = False
    PhoneConfig = None
    PhoneOS = None
    Phone = None
    Rotation = None

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

    ## Phone Teleoperator (lerobot)
    phone_os: Literal["ios", "android"] = "android"
    phone_id: str = "phone_1"

    ## Axis mapping (adjust if phone axes don't match robot axes)
    # Options:
    #   "direct": phone_x -> robot_x, phone_y -> robot_y, phone_z -> robot_z (手机向前=机器人向前)
    #   "lerobot_default": robot_x = -phone_y, robot_y = phone_x (lerobot默认，手机向前=机器人左右)
    #   "swap_xy": phone_x -> robot_y, phone_y -> robot_x (交换x和y)
    #   "invert_x": phone_x -> -robot_x, phone_y -> robot_y (反转x轴)
    #   "invert_y": phone_x -> robot_x, phone_y -> -robot_y (反转y轴)
    axis_mapping: str = "direct"  # Default: direct mapping for intuitive control

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def rotation_to_quaternion(rot: Rotation) -> np.ndarray:
    """Convert lerobot Rotation to numpy quaternion (w, x, y, z)."""
    quat_xyzw = rot.as_quat()  # Returns (x, y, z, w)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to (w, x, y, z)


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = quat
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def main():
    if not LEROBOT_AVAILABLE:
        log.error("lerobot phone teleoperator is not available. Exiting.")
        sys.exit(1)

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

    # Setup lerobot Phone Teleoperator
    phone_os = PhoneOS.IOS if args.phone_os.lower() == "ios" else PhoneOS.ANDROID
    phone_config = PhoneConfig(phone_os=phone_os, id=args.phone_id)
    phone_teleop = Phone(phone_config)

    log.info(f"Connecting to {phone_os.value} phone...")
    try:
        phone_teleop.connect()
        log.info("Phone connected successfully!")
    except Exception as e:
        log.error(f"Failed to connect phone: {e}")
        log.error("For iOS: Make sure HEBI Mobile I/O app is open.")
        log.error("For Android: The script will print a URL - open it on your phone.")
        sys.exit(1)

    # Calibration is done automatically in connect() for lerobot Phone
    if not phone_teleop.is_calibrated:
        log.warning("Phone is not calibrated. Calibration should happen automatically.")

    step = 0
    running = True
    ee_reference_pos = None
    ee_reference_quat = None
    reference_captured = False
    last_target_pos = None
    last_target_quat = None
    teleop_active = False  # Track whether teleop should stay enabled (even during gripper press)

    log.info("Starting teleoperation loop. Press Ctrl+C to exit.")

    try:
        while running:
            # Get action from phone teleoperator
            phone_action = phone_teleop.get_action()

            if not phone_action:
                # No valid action yet, skip this step
                time.sleep(0.01)
                continue

            raw_enabled = bool(phone_action.get("phone.enabled", False))
            phone_pos = phone_action.get("phone.pos")
            phone_rot = phone_action.get("phone.rot")
            raw_inputs = phone_action.get("phone.raw_inputs", {})

            # Get current robot state first (needed for gripper even when disabled)
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

            # Convert current EE position to robot local frame
            curr_ee_pos_local = quat_apply(quat_inv(robot_quat), curr_ee_pos - robot_pos)
            curr_ee_pos_local_np = curr_ee_pos_local.cpu().numpy()
            curr_ee_quat_np = curr_ee_quat.cpu().numpy()

            # Check if we have valid phone pose data
            # If not, we can still process gripper commands but skip pose updates
            has_valid_pose = phone_pos is not None and phone_rot is not None

            # Determine whether there is gripper input (used to keep teleop active)
            if args.phone_os.lower() == "ios":
                gripper_input_value = float(raw_inputs.get("a3", 0.0))
                has_gripper_input = abs(gripper_input_value) > 0.01
            else:
                button_a = bool(raw_inputs.get("reservedButtonA", False))
                button_b = bool(raw_inputs.get("reservedButtonB", False))
                has_gripper_input = button_a or button_b

            # Update teleop active state:
            # - When raw_enabled is True (Move/B1 pressed), teleop becomes active.
            # - While operating the gripper, keep teleop active even if Move button momentarily reports False.
            # - When neither Move nor gripper is pressed, deactivate teleop.
            if raw_enabled:
                teleop_active = True
            elif not has_gripper_input:
                teleop_active = False

            enabled = teleop_active

            # Capture reference pose ONLY when the Move/B1 button (raw enable) is pressed
            # This ensures the reference position doesn't reset when re-enabling
            if raw_enabled and not reference_captured and has_valid_pose:
                ee_reference_pos = curr_ee_pos_local_np.copy()
                ee_reference_quat = curr_ee_quat_np.copy()
                reference_captured = True
                log.info("Reference pose captured. Starting teleoperation.")

            if not enabled or not has_valid_pose:
                # When disabled or no valid pose, keep the last commanded pose
                # Don't reset reference_captured, so we don't re-capture on next enable
                if last_target_pos is not None:
                    # Use the last computed target pose to maintain continuity
                    ee_pos_target_local = last_target_pos.copy()
                    ee_quat_target_local = last_target_quat.copy()
                elif ee_reference_pos is not None:
                    # Fallback to reference if no last target exists
                    ee_pos_target_local = ee_reference_pos.copy()
                    ee_quat_target_local = ee_reference_quat.copy()
                else:
                    # Fallback to current pose if no reference exists
                    ee_pos_target_local = curr_ee_pos_local_np.copy()
                    ee_quat_target_local = curr_ee_quat_np.copy()
            else:
                # Phone position is in calibrated frame (relative to calibration pose)
                # Map phone coordinates to robot coordinates
                # Adjust mapping based on axis_mapping setting to match your setup
                if args.axis_mapping == "direct":
                    # Direct mapping: phone_x -> robot_x, phone_y -> robot_y, phone_z -> robot_z
                    # Phone forward = Robot forward (intuitive)
                    delta_pos_robot_frame = np.array([phone_pos[0], phone_pos[1], phone_pos[2]])
                elif args.axis_mapping == "lerobot_default":
                    # Lerobot default: robot_x = -phone_y, robot_y = phone_x
                    # Phone forward = Robot left/right (lerobot convention)
                    delta_pos_robot_frame = np.array([-phone_pos[1], phone_pos[0], phone_pos[2]])
                elif args.axis_mapping == "swap_xy":
                    # Swap x and y: phone_x -> robot_y, phone_y -> robot_x
                    delta_pos_robot_frame = np.array([phone_pos[1], phone_pos[0], phone_pos[2]])
                elif args.axis_mapping == "invert_x":
                    # Invert x axis: phone_x -> -robot_x, phone_y -> robot_y
                    delta_pos_robot_frame = np.array([-phone_pos[0], phone_pos[1], phone_pos[2]])
                elif args.axis_mapping == "invert_y":
                    # Invert y axis: phone_x -> robot_x, phone_y -> -robot_y
                    delta_pos_robot_frame = np.array([phone_pos[0], -phone_pos[1], phone_pos[2]])
                else:
                    # Default to direct mapping
                    log.warning(f"Unknown axis_mapping '{args.axis_mapping}', using direct mapping")
                    delta_pos_robot_frame = np.array([phone_pos[0], phone_pos[1], phone_pos[2]])

                # Ensure reference position is set (should be set on first enable)
                if ee_reference_pos is None:
                    ee_reference_pos = curr_ee_pos_local_np.copy()
                    ee_reference_quat = curr_ee_quat_np.copy()
                    log.warning("Reference position was None, using current position.")

                # Add delta to reference position to get absolute target
                # The reference position is captured only once on first enable
                ee_pos_target_local = ee_reference_pos + delta_pos_robot_frame

                # Phone rotation is already calibrated (relative to calibration pose)
                # Convert phone rotation to quaternion and combine with reference
                phone_quat = rotation_to_quaternion(phone_rot)  # (w, x, y, z)
                ref_quat = ee_reference_quat  # (w, x, y, z)

                # Convert to rotation matrices for composition
                ref_rot_matrix = quaternion_to_rotation_matrix(ref_quat)
                phone_rot_matrix = quaternion_to_rotation_matrix(phone_quat)
                # Compose rotations: target = reference * phone_delta
                target_rot_matrix = ref_rot_matrix @ phone_rot_matrix

                # Convert rotation matrix back to quaternion
                ee_quat_target_local = rotation_matrix_to_quaternion(target_rot_matrix)

                # Store the computed target for use when disabled
                last_target_pos = ee_pos_target_local.copy()
                last_target_quat = ee_quat_target_local.copy()

            # Convert to tensor
            ee_pos_target_tensor = torch.tensor(ee_pos_target_local, dtype=torch.float32, device=device)
            ee_quat_target_tensor = torch.tensor(ee_quat_target_local, dtype=torch.float32, device=device)

            # Get gripper command
            if args.phone_os.lower() == "ios":
                gripper_vel = float(raw_inputs.get("a3", 0.0))
                # Convert velocity to binary for now (can be improved)
                close_gripper = gripper_vel > 0.5
            else:  # Android
                button_a = bool(raw_inputs.get("reservedButtonA", False))
                button_b = bool(raw_inputs.get("reservedButtonB", False))
                close_gripper = button_a and not button_b

            # Solve IK
            q_solution, ik_succ = ik_solver.solve_ik_batch(
                ee_pos_target_tensor.unsqueeze(0),
                ee_quat_target_tensor.unsqueeze(0),
                seed_q=curr_robot_q,
            )

            if not ik_succ[0]:
                log.warning(f"IK failed at step {step}, using previous joint positions")
                q_solution = curr_robot_q

            # Process gripper command
            gripper_widths = process_gripper_command(
                torch.tensor(close_gripper, dtype=torch.float32, device=device),
                scenario.robots[0],
                device,
            )

            # Compose joint action
            actions = ik_solver.compose_joint_action(
                q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True
            )

            # Step environment
            obs, reward, success, time_out, extras = env.step(actions)

            step += 1
            if step % 100 == 0:
                log.debug(f"Step {step}, enabled: {enabled}")

    except KeyboardInterrupt:
        log.info("Interrupted by user. Shutting down...")
        running = False
    except Exception as e:
        log.error(f"Error in teleoperation loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        try:
            phone_teleop.disconnect()
            log.info("Phone disconnected.")
        except Exception as e:
            log.warning(f"Error disconnecting phone: {e}")

        env.close()
        log.info("Environment closed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
