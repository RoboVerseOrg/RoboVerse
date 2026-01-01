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
from metasim.utils.ik_solver import IKSolver, process_gripper_command
from metasim.utils.math import matrix_from_euler, quat_apply, quat_from_matrix, quat_inv, quat_mul
from metasim.utils.obs_utils import display_obs
from metasim.utils.teleop_utils import PygameKeyboardClient, process_kb_input

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "put_banana"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    ## Handlers
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "sapien3"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    ## Others
    num_envs: int = 1
    headless: bool = True

    ## IK Solver
    ik_solver: Literal["curobo", "pyroki"] = "pyroki"
    no_gnd: bool = False

    ## Visualization
    enable_viser: bool = True  # Enable real-time Viser 3D visualization
    enable_rerun: bool = False  # Enable real-time Rerun 3D visualization
    viser_port: int = 8080  # Port for Viser server

    ## Display
    display_camera: bool = True  # Whether to display camera view in real-time
    display_width: int = 2000  # Display window width (adjusted for dual camera split-screen)
    display_height: int = 1200  # Display window height

    ## Trajectory saving
    save_traj: bool = True  # Whether to save trajectory
    traj_dir: str = "teleop_trajs"  # Directory to save trajectories
    save_states: bool = True  # Whether to save full states (not just actions)
    save_every_n_steps: int = 10  # Save every N steps (1=save all, 2=save every other step, etc.)

    ## Step timing
    min_step_time: float = 0.001  # Minimum time per step in seconds (controls operation speed)

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


def extract_state_dict(obs, scenario, joint_orders: dict[str, list[str]] | None = None):
    """Extract state dictionary from TensorState observation.

    Args:
        obs: TensorState observation
        scenario: Scenario configuration to get joint names
        joint_orders: Optional mapping from object/robot name to joint order list. This is required
            for articulated objects that don't expose actuators in the config (e.g., RLBench assets).

    Returns:
        Dictionary containing positions, rotations, and joint positions for all objects and robots
    """
    state_dict = {}

    # Create lookup dicts for configurations
    obj_cfg_dict = {obj.name: obj for obj in scenario.objects}
    robot_cfg_dict = {robot.name: robot for robot in scenario.robots}
    joint_orders = joint_orders or {}

    def _build_dof_pos(names: list[str] | None, joint_pos_tensor):
        if not names:
            return None
        joint_values = None
        if joint_pos_tensor is not None:
            joint_values = joint_pos_tensor[0].cpu().numpy()
        dof_pos = {}
        for idx, name in enumerate(names):
            value = joint_values[idx] if joint_values is not None and idx < len(joint_values) else 0.0
            dof_pos[name] = float(value)
        # ensure deterministic ordering (dictionary order)
        return {key: dof_pos[key] for key in sorted(dof_pos.keys())}

    # Extract object states
    for obj_name, obj_state in obs.objects.items():
        pos = obj_state.root_state[0, :3].cpu().numpy()  # [x, y, z]
        quat = obj_state.root_state[0, 3:7].cpu().numpy()  # [w, x, y, z]

        state_entry = {
            "pos": pos,
            "rot": quat,
        }

        # Add joint positions if the object has joints
        joint_names = joint_orders.get(obj_name)
        if joint_names:
            joint_names = sorted(joint_names)
        if not joint_names and obj_name in obj_cfg_dict:
            obj_cfg = obj_cfg_dict[obj_name]
            if hasattr(obj_cfg, "actuators") and obj_cfg.actuators:
                joint_names = sorted(obj_cfg.actuators.keys())

        dof_pos = _build_dof_pos(joint_names, obj_state.joint_pos)
        if dof_pos:
            state_entry["dof_pos"] = dof_pos

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
        joint_names = joint_orders.get(robot_name)
        if joint_names:
            joint_names = sorted(joint_names)
        if not joint_names and robot_name in robot_cfg_dict:
            robot_cfg = robot_cfg_dict[robot_name]
            if robot_cfg.actuators is not None:
                joint_names = sorted(robot_cfg.actuators.keys())

        dof_pos = _build_dof_pos(joint_names, robot_state.joint_pos)
        if dof_pos:
            state_entry["dof_pos"] = dof_pos

        state_dict[robot_name] = state_entry

    return state_dict


def main():
    task_cls = get_task_class(args.task)
    # Create two cameras with different viewpoints
    camera1 = PinholeCameraCfg(name="camera_1", pos=(2.0, -2.0, 2.0), look_at=(0.0, 0.0, 0.0), width=512, height=512)
    camera2 = PinholeCameraCfg(name="camera_2", pos=(2.5, -1.2, 2.5), look_at=(0.0, 0.0, 0.0), width=512, height=512)
    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera1, camera2],
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
        decimation=2,
    )
    # HACK specific to isaacsim
    if args.sim == "isaacsim":
        scenario.update(decimation=2)
        if scenario.robots[0].name == "franka":
            # use smaller stiffness and damping for fingers for fine-grained control
            from metasim.scenario.robot import BaseActuatorCfg

            scenario.robots[0].actuators["panda_finger_joint1"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )
            scenario.robots[0].actuators["panda_finger_joint2"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )

    tic = time.time()
    device = torch.device("cpu")
    env = task_cls(scenario, device=device)

    # Optionally wrap with visualization
    if args.enable_viser or args.enable_rerun:
        from metasim.utils.viz_task_wrapper import TaskVizWrapper

        env = TaskVizWrapper(
            env,
            use_rerun=args.enable_rerun,
            use_viser=args.enable_viser,
            rerun_app_name="Teleop Keyboard",
            viser_port=args.viser_port,
            update_freq=1,
        )
        if args.enable_viser:
            log.info(f"Viser visualization enabled on port {args.viser_port}")
        if args.enable_rerun:
            log.info("Rerun visualization enabled")

    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")
    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    def get_joint_orders() -> dict[str, list[str]]:
        """Fetch current joint order mapping from handler (robots + articulated objs)."""
        handler = getattr(env, "handler", None)
        if handler is None:
            return {}
        joint_map = getattr(handler, "object_joint_order", None)
        return joint_map if joint_map is not None else {}

    # Initialize trajectory recording - support multiple episodes
    all_episodes = []  # List of completed trajectories
    current_episode_actions = []
    current_episode_states = [] if args.save_states else None
    current_episode_init_state = None
    episode_count = 1

    if args.save_traj:
        # Record initial state for first episode
        current_episode_init_state = extract_state_dict(obs, scenario, joint_orders=get_joint_orders())
        log.info(f"Episode {episode_count}: Initial state recorded")

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
        "r": False,  # Reset key (discard current episode)
        "b": False,  # Save checkpoint
        "n": False,  # Restore checkpoint
    }

    # Episode control flags
    reset_requested = False  # Reset and discard current episode (R key)
    complete_requested = False  # Complete current episode like timeout (C key)
    save_to_file_requested = False  # Save all episodes to file (S key)

    # Checkpoint system
    checkpoint = None  # Stores: {"state": state_dict, "actions": list, "states": list, "init_state": dict}

    def on_key_press(key):
        """Handle key press events"""
        nonlocal running, space_pressed, reset_requested, complete_requested, save_to_file_requested, checkpoint
        try:
            key_name = key.char.lower() if hasattr(key, "char") and key.char else str(key).split(".")[-1]
            if key_name in key_states:
                key_states[key_name] = True
            if key_name == "space":
                space_pressed = True
            if key_name == "r":
                reset_requested = True
                log.info("Reset requested (discard current episode)")
            if key_name == "v":
                complete_requested = True
                log.info("Complete requested (save current episode and reset)")
            if key_name == "b":
                # Save checkpoint - will be handled in main loop
                log.info("Checkpoint save requested (B key)")
            if key_name == "n":
                # Restore checkpoint - will be handled in main loop
                log.info("Checkpoint restore requested (N key)")
        except AttributeError:
            # Handle special keys (ESC, etc.)
            if str(key) == "Key.esc":
                log.debug("ESC pressed, exiting simulation...")
                running = False

    def on_key_release(key):
        """Handle key release events"""
        nonlocal space_pressed, reset_requested, complete_requested, save_to_file_requested
        try:
            key_name = key.char.lower() if hasattr(key, "char") and key.char else str(key).split(".")[-1]
            if key_name in key_states:
                key_states[key_name] = False
            if key_name == "space":
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
        nonlocal episode_count

        current_episode_actions = []
        current_episode_states = [] if args.save_states else None
        current_episode_init_state = None
        episode_count += 1

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
    log.info("  Movement: Arrow keys (↑↓←→), E/D (up/down)")
    log.info("  Rotation: Q/W (roll), A/S (pitch), Z/X (yaw)")
    log.info("  Gripper: SPACE (close/open)")
    log.info("  Episode: V (complete & save), R (reset & discard)")
    log.info("  Checkpoint: B (save checkpoint), N (restore checkpoint)")
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
            d_pos, d_rot_local, close_gripper = process_kb_input(keyboard_client, dpos=0.02, drot=0.03)
        else:
            # Handle keyboard input using pynput key states
            d_pos = [0.0, 0.0, 0.0]
            d_rot_local = [0.0, 0.0, 0.0]
            close_gripper = 0

            # Movement controls (pynput key mapping)
            if key_states["up"]:
                d_pos[0] += 0.008  # Move +X
            if key_states["down"]:
                d_pos[0] -= 0.008  # Move -X
            if key_states["left"]:
                d_pos[1] += 0.008  # Move +Y
            if key_states["right"]:
                d_pos[1] -= 0.008  # Move -Y
            if key_states["e"]:
                d_pos[2] += 0.008  # Move +Z
            if key_states["d"]:
                d_pos[2] -= 0.008  # Move -Z

            # Rotation controls (pynput key mapping)
            if key_states["q"]:
                d_rot_local[0] += 0.03  # Roll +
            if key_states["w"]:
                d_rot_local[0] -= 0.03  # Roll -
            if key_states["a"]:
                d_rot_local[1] += 0.03  # Pitch +
            if key_states["s"]:
                d_rot_local[1] -= 0.03  # Pitch -
            if key_states["z"]:
                d_rot_local[2] += 0.03  # Yaw +
            if key_states["x"]:
                d_rot_local[2] -= 0.03  # Yaw -

            # Gripper controls (space_pressed tracks key state)
            # Note: space_pressed=True means close gripper, False means open
            close_gripper = 1 if space_pressed else 0

        # Check if there is any control command
        has_control_command = (
            any(abs(x) > 1e-6 for x in d_pos) or any(abs(x) > 1e-6 for x in d_rot_local) or close_gripper != 0
        )

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
            torch.tensor(float(not close_gripper), dtype=torch.float32, device=device), scenario.robots[0], device
        )

        # Compose joint action
        actions = ik_solver.compose_joint_action(q_solution, gripper_widths, current_q=curr_robot_q, return_dict=True)

        obs, reward, success, time_out, extras = env.step(actions)

        # Record trajectory data (with downsampling) - only if there was a control command
        if args.save_traj and has_control_command and (step % args.save_every_n_steps == 0):
            # Extract robot action from action list
            # actions is a list of dicts: [{robot_name: {dof_pos_target: {...}}}]
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
                current_state = extract_state_dict(obs, scenario, joint_orders=get_joint_orders())
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

        # Handle checkpoint save (B key)
        if key_states.get("b", False):
            # Save checkpoint with current trajectory state
            # Note: Current step's action/state may or may not be recorded yet depending on save_every_n_steps
            # The checkpoint saves the current environment state, which corresponds to the state AFTER
            # executing all actions up to current step
            checkpoint = {
                "state": extract_state_dict(obs, scenario, joint_orders=get_joint_orders()),  # Current env state
                "actions": current_episode_actions.copy(),  # All recorded actions up to now
                "states": current_episode_states.copy()
                if current_episode_states is not None
                else None,  # All recorded states
                "init_state": current_episode_init_state.copy() if current_episode_init_state is not None else None,
                "step": step,  # Current step number
            }
            log.info(
                f"Checkpoint saved at step {step}: {len(current_episode_actions)} actions, "
                f"{len(current_episode_states) if current_episode_states else 0} states. "
                f"Trajectory can be resumed from here."
            )
            # Reset key state to avoid repeated saves
            key_states["b"] = False

        # Handle checkpoint restore (N key)
        if key_states.get("n", False) and checkpoint is not None:
            log.info(f"Restoring checkpoint from step {checkpoint['step']}...")
            # Restore trajectory data
            current_episode_actions = checkpoint["actions"].copy()
            current_episode_states = checkpoint["states"].copy() if checkpoint["states"] is not None else None
            current_episode_init_state = (
                checkpoint["init_state"].copy() if checkpoint["init_state"] is not None else None
            )

            # Restore environment state
            checkpoint_state = checkpoint["state"]
            joint_orders = get_joint_orders()
            # Convert state dict to format expected by env.reset
            # Format: {"objects": {...}, "robots": {...}}
            reset_state = {"objects": {}, "robots": {}}

            # Separate objects and robots
            obj_cfg_dict = {obj.name: obj for obj in scenario.objects}
            robot_cfg_dict = {robot.name: robot for robot in scenario.robots}

            for name, state_data in checkpoint_state.items():
                state_entry = {
                    "pos": torch.tensor(state_data["pos"], dtype=torch.float32),
                    "rot": torch.tensor(state_data["rot"], dtype=torch.float32),
                }
                if "dof_pos" in state_data:
                    ordered_items = sorted(state_data["dof_pos"].items())
                    state_entry["dof_pos"] = {
                        k: torch.tensor(v, dtype=torch.float32) if isinstance(v, (int, float)) else v
                        for k, v in ordered_items
                    }
                elif name in joint_orders:
                    ordered_joint_names = sorted(joint_orders[name])
                    state_entry["dof_pos"] = {
                        joint_name: torch.tensor(0.0, dtype=torch.float32) for joint_name in ordered_joint_names
                    }
                    log.warning(
                        f"Checkpoint missing dof_pos for articulated entity '{name}'. "
                        "Using zeros as fallback; consider resaving checkpoint."
                    )

                # Categorize as object or robot
                if name in obj_cfg_dict:
                    reset_state["objects"][name] = state_entry
                elif name in robot_cfg_dict:
                    reset_state["robots"][name] = state_entry

            # Reset environment to checkpoint state
            obs, extras = env.reset(states=[reset_state], env_ids=[0])
            step = checkpoint["step"]

            # Verify trajectory continuity
            # Expected number of recorded actions: steps that are multiples of save_every_n_steps
            # For step N, we record at steps 0, save_every_n_steps, 2*save_every_n_steps, ..., up to floor(N/save_every_n_steps)*save_every_n_steps
            if args.save_traj:
                expected_actions = (step // args.save_every_n_steps) + 1
                actual_actions = len(current_episode_actions)
                if actual_actions != expected_actions:
                    log.warning(
                        f"Trajectory length check: step {step}, expected {expected_actions} actions "
                        f"(assuming recording at steps 0, {args.save_every_n_steps}, "
                        f"{2 * args.save_every_n_steps}, ...), but found {actual_actions} actions. "
                        f"This is normal if checkpoint was saved at a non-recording step."
                    )
                else:
                    log.debug(f"Trajectory continuity verified: {actual_actions} actions match expected count.")

            log.info(
                f"Checkpoint restored! Resumed from step {step} with {len(current_episode_actions)} actions, "
                f"{len(current_episode_states) if current_episode_states else 0} states. "
                f"Next step will be {step + 1}, trajectory will continue appending from here."
            )
            # Reset key state
            key_states["n"] = False
            # Continue to next iteration - step will be incremented at end of loop
            # Trajectory recording will continue from next step, appending to existing trajectory
            continue  # Skip the rest of the loop to avoid processing actions this step
        elif key_states.get("n", False) and checkpoint is None:
            log.warning("No checkpoint available to restore! Press B first to save a checkpoint.")
            key_states["n"] = False

        # Reset environment if episode is done
        if episode_done:
            reset_episode()
            obs, extras = env.reset()
            if args.save_traj:
                current_episode_init_state = extract_state_dict(obs, scenario, joint_orders=get_joint_orders())
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
