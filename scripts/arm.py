from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Literal

import cv2  # OpenCV for camera display
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
from metasim.utils.demo_util import save_traj_file
from metasim.utils.kinematics import get_pyroki_model
from metasim.utils.math import matrix_from_euler, quat_apply, quat_from_matrix, quat_inv, quat_mul
from metasim.utils.obs_utils import display_obs
from metasim.utils.teleop_utils import PygameKeyboardClient, process_kb_input
from roboverse_pack.robots.vega_cfg import VegaCfg

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "put_banana"
    robot: str = "vega"  # Vega only
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    ## Handlers
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaacsim"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    headless: bool = True

    ## Viser Visualization
    enable_viser: bool = False  # Enable real-time Viser 3D visualization
    viser_port: int = 8080  # Port for Viser server

    ## Display
    display_camera: bool = True  # Whether to display camera view in real-time
    display_width: int = 1200  # Display window width (adjusted for dual camera split-screen)
    display_height: int = 600  # Display window height

    ## Trajectory saving
    save_traj: bool = True  # Whether to save trajectory
    traj_dir: str = "teleop_trajs"  # Directory to save trajectories
    save_states: bool = True  # Whether to save full states (not just actions)
    save_every_n_steps: int = 5  # Save every N steps (1=save all, 2=save every other step, etc.)

    ## Step timing
    min_step_time: float = 0.001  # Minimum time per step in seconds (controls operation speed)

    ## Finger control
    finger_bend_increment: float = 0.5  # Finger bend increment per space press (10% of range)

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def extract_state_dict(obs, scenario):
    """Extract state dictionary from TensorState observation.

    Args:
        obs: TensorState observation
        scenario: Scenario configuration to get joint names

    Returns:
        Dictionary containing positions, rotations, and joint positions for all objects and robots
    """
    state_dict = {}

    # Create lookup dicts for configurations
    obj_cfg_dict = {obj.name: obj for obj in scenario.objects}
    robot_cfg_dict = {robot.name: robot for robot in scenario.robots}

    # Extract object states
    for obj_name, obj_state in obs.objects.items():
        pos = obj_state.root_state[0, :3].cpu().numpy()  # [x, y, z]
        quat = obj_state.root_state[0, 3:7].cpu().numpy()  # [w, x, y, z]

        state_entry = {
            "pos": pos,
            "rot": quat,
        }

        # Add joint positions if the object has joints
        if obj_state.joint_pos is not None and obj_name in obj_cfg_dict:
            obj_cfg = obj_cfg_dict[obj_name]
            if hasattr(obj_cfg, "actuators") and obj_cfg.actuators is not None:
                # Joint names are sorted alphabetically (standard in handlers)
                joint_names = sorted(obj_cfg.actuators.keys())
                joint_positions = obj_state.joint_pos[0].cpu().numpy()
                state_entry["dof_pos"] = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}

        state_dict[obj_name] = state_entry

    # Extract robot states
    for robot_name, robot_state in obs.robots.items():
        pos = robot_state.root_state[0, :3].cpu().numpy()  # [x, y, z]
        quat = robot_state.root_state[0, 3:7].cpu().numpy()  # [w, x, y, z]

        state_entry = {
            "pos": pos,
            "rot": quat,
        }

        # Add joint positions for robot
        if robot_name in robot_cfg_dict:
            robot_cfg = robot_cfg_dict[robot_name]
            if robot_cfg.actuators is not None:
                # Joint names are sorted alphabetically (standard in handlers)
                joint_names = sorted(robot_cfg.actuators.keys())
                joint_positions = robot_state.joint_pos[0].cpu().numpy()
                state_entry["dof_pos"] = {name: float(pos) for name, pos in zip(joint_names, joint_positions)}

        state_dict[robot_name] = state_entry

    return state_dict


def get_finger_joint_names(robot_cfg):
    """Get finger joint names (excluding arm joints) for Vega."""
    all_joints = sorted(robot_cfg.actuators.keys())

    # Vega robot: left arm joints are L_arm_j1 to L_arm_j7
    arm_joints = [f"L_arm_j{i}" for i in range(1, 8)]
    finger_joints = [j for j in all_joints if j not in arm_joints]
    return finger_joints


def update_finger_positions(
    current_finger_positions: dict[str, float],
    robot_cfg: VegaCfg,
    increment: float,
) -> dict[str, float]:
    """Update finger positions by incrementing 10% of their range.

    Args:
        current_finger_positions: Current finger joint positions
        robot_cfg: Robot configuration with joint limits
        increment: Increment factor (0.1 = 10%)

    Returns:
        Updated finger positions
    """
    finger_joints = get_finger_joint_names(robot_cfg)
    updated_positions = current_finger_positions.copy()

    for joint_name in finger_joints:
        if joint_name not in robot_cfg.joint_limits:
            continue

        lower, upper = robot_cfg.joint_limits[joint_name]
        joint_range = upper - lower
        increment_amount = increment * joint_range

        current_pos = current_finger_positions.get(joint_name, robot_cfg.default_joint_positions.get(joint_name, lower))
        new_pos = current_pos + increment_amount

        # Clamp to joint limits
        new_pos = max(lower, min(upper, new_pos))
        updated_positions[joint_name] = new_pos

    return updated_positions


def main():
    task_cls = get_task_class(args.task)
    # Create two cameras with different viewpoints
    camera1 = PinholeCameraCfg(name="camera_1", pos=(2.0, -2.0, 3.0), look_at=(0.0, 0.0, 1.0))
    camera2 = PinholeCameraCfg(name="camera_2", pos=(2.5, -1.2, 3.0), look_at=(0.0, 0.0, 1.0))
    # Select robot configuration based on args.robot
    if args.robot != "vega":
        raise ValueError(f"Unsupported robot '{args.robot}'. This script currently only supports 'vega'.")
    robot_cfg_cls = VegaCfg

    scenario = task_cls.scenario.update(
        robots=[robot_cfg_cls()],
        scene=args.scene,
        cameras=[camera1, camera2],
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    num_envs: int = scenario.num_envs
    robot_cfg = scenario.robots[0]

    tic = time.time()
    device = torch.device("cpu")
    env = task_cls(scenario, device=device)

    # Optionally wrap with Viser visualization
    if args.enable_viser:
        from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

        env = TaskViserWrapper(env, port=args.viser_port)
        log.info(f"Viser visualization enabled on port {args.viser_port}")

    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")
    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # Initialize trajectory recording - support multiple episodes
    all_episodes = []  # List of completed trajectories
    current_episode_actions = []
    current_episode_states = [] if args.save_states else None
    current_episode_init_state = None
    episode_count = 1

    if args.save_traj:
        # Record initial state for first episode
        current_episode_init_state = extract_state_dict(obs, scenario)
        log.info(f"Episode {episode_count}: Initial state recorded")

    # Initialize finger positions from default
    finger_joint_names = get_finger_joint_names(robot_cfg)
    current_finger_positions = {name: robot_cfg.default_joint_positions.get(name, 0.0) for name in finger_joint_names}

    # Setup Pyroki model (direct URDF-order IK)
    pyroki_model = get_pyroki_model(robot_cfg)
    pyroki_solver = pyroki_model["solve_ik_with_seed"]
    pyroki_joint_names = pyroki_model.get("joint_names", [])
    if not pyroki_joint_names:
        pyroki_joint_names = [name for name in robot_cfg.actuators.keys()]
    pyroki_joint_names = [name for name in pyroki_joint_names if name in robot_cfg.actuators]
    pyroki_joint_to_idx = {name: idx for idx, name in enumerate(pyroki_joint_names)}

    # Setup camera display and keyboard control
    keyboard_client = None
    space_pressed = False  # Track space key state for finger control
    space_just_pressed = False  # Track if space was just pressed (edge trigger)
    space_just_released = False  # Track if space was just released (edge trigger)

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
        "r": False,  # Reset key (discard current episode)
    }

    # Episode control flags
    reset_requested = False  # Reset and discard current episode (R key)
    complete_requested = False  # Complete current episode like timeout (V key)
    save_to_file_requested = False  # Save all episodes to file (S key)

    def on_key_press(key):
        """Handle key press events"""
        nonlocal running, space_pressed, space_just_pressed, reset_requested, complete_requested, save_to_file_requested
        try:
            key_name = key.char.lower() if hasattr(key, "char") and key.char else str(key).split(".")[-1]
            if key_name in key_states:
                key_states[key_name] = True
            if key_name == "space":
                if not space_pressed:  # Edge trigger: only trigger on first press
                    space_just_pressed = True
                space_pressed = True
            if key_name == "r":
                reset_requested = True
                log.info("Reset requested (discard current episode)")
            if key_name == "v":
                complete_requested = True
                log.info("Complete requested (save current episode and reset)")
        except AttributeError:
            # Handle special keys (ESC, etc.)
            if str(key) == "Key.esc":
                log.debug("ESC pressed, exiting simulation...")
                running = False

    def on_key_release(key):
        """Handle key release events"""
        nonlocal space_pressed, space_just_released, reset_requested, complete_requested, save_to_file_requested
        try:
            key_name = key.char.lower() if hasattr(key, "char") and key.char else str(key).split(".")[-1]
            if key_name in key_states:
                key_states[key_name] = False
            if key_name == "space":
                if space_pressed:  # Edge trigger: only trigger on first release
                    space_just_released = True
                space_pressed = False
            if key_name == "r":
                reset_requested = False
            if key_name == "v":
                complete_requested = False
        except AttributeError:
            pass

    def save_current_episode():
        """Save current episode to the episodes list"""
        nonlocal current_episode_init_state, current_episode_actions, current_episode_states
        nonlocal all_episodes

        if len(current_episode_actions) > 0:
            episode_data = {
                "init_state": current_episode_init_state,
                "actions": current_episode_actions,
                "states": current_episode_states if args.save_states else None,
            }
            all_episodes.append(episode_data)
            log.info(f"Episode {episode_count} saved ({len(current_episode_actions)} steps)")
            return True
        else:
            log.warning(f"Episode {episode_count} has no actions, not saved")
            return False

    def reset_episode():
        """Reset episode tracking variables"""
        nonlocal current_episode_init_state, current_episode_actions, current_episode_states
        nonlocal episode_count, current_finger_positions

        current_episode_actions = []
        current_episode_states = [] if args.save_states else None
        current_episode_init_state = None
        episode_count += 1
        # Reset finger positions to default
        current_finger_positions = {
            name: robot_cfg.default_joint_positions.get(name, 0.0) for name in finger_joint_names
        }

    def save_all_episodes_to_file():
        """Save all collected episodes to file"""
        if len(all_episodes) == 0:
            log.warning("No episodes to save")
            return

        # Organize in v2 format
        trajs = {scenario.robots[0].name: all_episodes}

        # Create output directory
        os.makedirs(args.traj_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.task}_{scenario.robots[0].name}_{timestamp}_v2.pkl"
        filepath = os.path.join(args.traj_dir, filename)

        # Save trajectory
        save_traj_file(trajs, filepath)
        log.info(f"All episodes saved to {filepath}")
        log.info(f"  - Total episodes: {len(all_episodes)}")
        log.info(f"  - Total steps: {sum(len(ep['actions']) for ep in all_episodes)}")

    # Setup keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)

    # Print control instructions
    log.info("=" * 60)
    log.info("KEYBOARD CONTROLS:")
    log.info("  Arm Movement: Arrow keys (↑↓←→), E/D (up/down)")
    log.info("  Arm Rotation: Q/W (roll), A/S (pitch), Z/X (yaw)")
    log.info("  Fingers: SPACE (bend 10% per press, release to reset)")
    log.info("  Episode: V (complete & save), R (reset & discard)")
    log.info("  Exit: ESC (save all and quit)")
    log.info("=" * 60)

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
    # Pre-compute joint name mappings for efficiency
    all_joint_names = sorted(robot_cfg.actuators.keys())
    # Vega robot: left arm joints
    arm_joint_names = [f"L_arm_j{i}" for i in range(1, 8)]

    pyroki_to_alpha_idx = [all_joint_names.index(name) for name in pyroki_joint_names if name in all_joint_names]
    if len(pyroki_to_alpha_idx) != len(pyroki_joint_names):
        missing_pyroki = sorted(set(pyroki_joint_names) - set(all_joint_names))
        if missing_pyroki:
            log.warning(f"Pyroki joints missing from actuator list: {missing_pyroki}")

    pyroki_left_arm_indices = [pyroki_joint_to_idx[name] for name in arm_joint_names if name in pyroki_joint_to_idx]
    if len(pyroki_left_arm_indices) != len(arm_joint_names):
        log.warning(
            f"Left arm mapping mismatch: expected {len(arm_joint_names)} joints, "
            f"found {len(pyroki_left_arm_indices)} in Pyroki joint list."
        )

    while running:
        # Record step start time for timing control
        step_start_time = time.time()

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
            d_pos, d_rot_local, _ = process_kb_input(keyboard_client, dpos=0.0005, drot=0.01)
        else:
            # Handle keyboard input using pynput key states
            d_pos = [0.0, 0.0, 0.0]
            d_rot_local = [0.0, 0.0, 0.0]

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

        # Handle finger control (edge-triggered on space press)
        if space_just_pressed:
            current_finger_positions = update_finger_positions(
                current_finger_positions, robot_cfg, args.finger_bend_increment
            )
            space_just_pressed = False  # Reset edge trigger
            log.debug(f"Fingers bent: {current_finger_positions}")

        # Handle finger release (edge-triggered on space release)
        if space_just_released:
            # Reset finger positions to default
            current_finger_positions = {
                name: robot_cfg.default_joint_positions.get(name, 0.0) for name in finger_joint_names
            }
            space_just_released = False  # Reset edge trigger
            log.debug(f"Fingers reset to default: {current_finger_positions}")

        d_pos_tensor = torch.tensor(d_pos, dtype=torch.float32, device=device)
        d_rot_tensor = torch.tensor(d_rot_local, dtype=torch.float32, device=device)

        # delta quaternion
        d_rot_mat_local = matrix_from_euler(d_rot_tensor.unsqueeze(0), "XYZ")
        d_quat_local = quat_from_matrix(d_rot_mat_local)[0]  # (4,)
        ee_pos_target = curr_ee_pos + d_pos_tensor
        ee_quat_target_local = quat_mul(curr_ee_quat_local, d_quat_local)

        # Solve IK using Pyroki directly (returns URDF joint ordering)
        curr_q_alpha = curr_robot_q[0].clone()
        if pyroki_to_alpha_idx:
            seed_q_pyroki = curr_q_alpha[pyroki_to_alpha_idx].clone()

            pos_cpu = ee_pos_target.detach().cpu()
            quat_cpu = ee_quat_target_local.detach().cpu()
            seed_q_cpu = seed_q_pyroki.detach().cpu()

            try:
                pyroki_solution = pyroki_solver(pos_cpu, quat_cpu, seed_q_cpu).to(dtype=seed_q_pyroki.dtype)
            except Exception as exc:
                log.warning(f"Pyroki IK failed: {exc}")
                pyroki_solution = seed_q_pyroki.clone()

            if torch.isnan(pyroki_solution).any() or pyroki_solution.shape[0] != seed_q_pyroki.shape[0]:
                log.warning("Pyroki IK produced invalid output; falling back to seed configuration.")
                pyroki_solution = seed_q_pyroki.clone()

            updated_pyroki = seed_q_pyroki.clone()
            if pyroki_left_arm_indices:
                updated_pyroki[pyroki_left_arm_indices] = pyroki_solution[pyroki_left_arm_indices]

            # Preserve finger joints explicitly (use tracked values)
            for finger_name in finger_joint_names:
                idx = pyroki_joint_to_idx.get(finger_name)
                if idx is not None:
                    updated_pyroki[idx] = torch.tensor(
                        current_finger_positions.get(
                            finger_name, robot_cfg.default_joint_positions.get(finger_name, 0.0)
                        ),
                        dtype=updated_pyroki.dtype,
                    )

            # Map Pyroki joint ordering back to alphabetical actuator ordering
            updated_alpha = curr_q_alpha.clone()
            for pyroki_idx, alpha_idx in enumerate(pyroki_to_alpha_idx):
                updated_alpha[alpha_idx] = updated_pyroki[pyroki_idx]
        else:
            log.warning("Pyroki joint mapping is empty; skipping IK update and using current joint positions.")
            updated_alpha = curr_q_alpha

        # Compose full joint dictionary from updated positions
        full_joint_positions = {
            joint_name: float(updated_alpha[idx].item()) for idx, joint_name in enumerate(all_joint_names)
        }

        # Create action dict
        actions = [
            {
                scenario.robots[0].name: {
                    "dof_pos_target": full_joint_positions,
                }
            }
        ]

        obs, reward, success, time_out, extras = env.step(actions)

        # Update finger positions from observation (to keep in sync)
        # Joint positions in observation are in alphabetical order
        curr_all_joint_pos = obs.robots[scenario.robots[0].name].joint_pos[0].cpu().numpy()
        for i, joint_name in enumerate(all_joint_names):
            if joint_name in finger_joint_names:
                current_finger_positions[joint_name] = float(curr_all_joint_pos[i])

        # Record trajectory data (with downsampling)
        if args.save_traj and (step % args.save_every_n_steps == 0):
            # Extract robot action from action list
            robot_action = actions[0][scenario.robots[0].name]

            # Record action in v2 format
            action_record = {
                "dof_pos_target": {k: float(v) for k, v in robot_action.get("dof_pos_target", {}).items()},
                "dof_effort_target": {k: float(v) for k, v in robot_action.get("dof_effort_target", {}).items()}
                if "dof_effort_target" in robot_action
                else None,
            }
            current_episode_actions.append(action_record)

            # Record state if requested
            if args.save_states:
                current_state = extract_state_dict(obs, scenario)
                current_episode_states.append(current_state)

        # Check for episode completion
        episode_done = False

        # Note: success and timeout just notify, don't auto-save or reset
        # User must press V to save or R to discard
        if success.any():
            log.info(f"Episode {episode_count}: Task succeeded! Press V to save, or R to discard and reset")

        if time_out.any():
            log.info(f"Episode {episode_count}: Timeout! Press V to save, or R to discard and reset")

        # Handle manual complete request (save current episode)
        if complete_requested:
            log.info(f"Episode {episode_count}: Saving episode...")
            if args.save_traj:
                save_current_episode()
            episode_done = True

        # Handle manual reset request (discard current episode)
        if reset_requested:
            log.info(f"Episode {episode_count}: Discarding episode and resetting...")
            episode_done = True
            # Don't save the episode, just reset

        # Reset environment if episode is done
        if episode_done:
            reset_episode()
            obs, extras = env.reset()
            if args.save_traj:
                current_episode_init_state = extract_state_dict(obs, scenario)
                log.info(f"Episode {episode_count}: Started (total episodes collected: {len(all_episodes)})")
            step = 0
            reset_requested = False
            complete_requested = False
            continue

        # Display camera observation if requested
        if args.display_camera:
            running = display_obs(obs, args.display_width, args.display_height)
            if not running:
                break

        # Enforce minimum step time to control operation speed
        step_elapsed_time = time.time() - step_start_time
        if step_elapsed_time < args.min_step_time:
            time.sleep(args.min_step_time - step_elapsed_time)

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

    # Save current episode if it has data (before exiting)
    if args.save_traj and len(current_episode_actions) > 0:
        log.info("Saving current episode before exit...")
        save_current_episode()

    # Save all collected episodes
    if args.save_traj:
        save_all_episodes_to_file()

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
