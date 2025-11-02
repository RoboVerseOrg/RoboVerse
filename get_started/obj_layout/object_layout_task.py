"""Keyboard object control with Task-based architecture - Real-time object manipulation"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import cv2
import pygame
import rootutils
import torch
import tyro
from huggingface_hub import snapshot_download

rootutils.setup_root(__file__, pythonpath=True)

from loguru import logger as log
from rich.logging import RichHandler

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.render import RenderCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.math import matrix_from_euler, quat_from_matrix, quat_mul
from metasim.utils.obs_utils import display_obs

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def save_poses_to_file(obs, scenario, filename="saved_poses.py"):
    """Save current poses of all objects and robots to a Python file from observation

    Args:
        obs: TensorState observation from task
        scenario: Scenario configuration
        filename: Output filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename.replace(".py", f"_{timestamp}.py")

    with open(filename, "w") as f:
        f.write('"""Saved poses from keyboard control"""\n\n')
        f.write("import torch\n\n")
        f.write("# Saved at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("poses = {\n")
        f.write('    "objects": {\n')

        # Save objects from observation
        for obj_name, obj_state in obs.objects.items():
            pos = obj_state.root_state[0, :3].cpu()
            quat = obj_state.root_state[0, 3:7].cpu()

            f.write(f'        "{obj_name}": {{\n')
            f.write(f'            "pos": torch.tensor([{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]),\n')
            f.write(f'            "rot": torch.tensor([{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]),\n')

            # Add joint positions if available
            if obj_state.joint_pos is not None:
                obj_cfg = next((obj for obj in scenario.objects if obj.name == obj_name), None)
                if obj_cfg and hasattr(obj_cfg, "actuators") and obj_cfg.actuators:
                    joint_names = sorted(obj_cfg.actuators.keys())
                    joint_positions = obj_state.joint_pos[0].cpu().numpy()
                    f.write('            "dof_pos": {\n')
                    for name, pos in zip(joint_names, joint_positions):
                        f.write(f'                "{name}": {pos:.6f},\n')
                    f.write("            },\n")

            f.write("        },\n")

        f.write("    },\n")
        f.write('    "robots": {\n')

        # Save robots from observation
        for robot_name, robot_state in obs.robots.items():
            pos = robot_state.root_state[0, :3].cpu()
            quat = robot_state.root_state[0, 3:7].cpu()

            f.write(f'        "{robot_name}": {{\n')
            f.write(f'            "pos": torch.tensor([{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]),\n')
            f.write(f'            "rot": torch.tensor([{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]),\n')

            # Add joint positions
            robot_cfg = next((robot for robot in scenario.robots if robot.name == robot_name), None)
            if robot_cfg and robot_cfg.actuators:
                joint_names = sorted(robot_cfg.actuators.keys())
                joint_positions = robot_state.joint_pos[0].cpu().numpy()
                f.write('            "dof_pos": {\n')
                for name, pos in zip(joint_names, joint_positions):
                    f.write(f'                "{name}": {pos:.6f},\n')
                f.write("            },\n")

            f.write("        },\n")

        f.write("    },\n")
        f.write("}\n")

    return filename


class ObjectKeyboardClient:
    """Keyboard client for object and robot control"""

    def __init__(
        self,
        entity_names: list[str],
        entity_types: list[str],
        entity_joints: dict[str, list[str]],
        width: int = 550,
        height: int = 650,
        title: str = "Object/Robot Control",
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        self.entity_names = entity_names
        self.entity_types = entity_types
        self.entity_joints = entity_joints
        self.selected_idx = 0

        # Joint control mode
        self.joint_mode = False
        self.selected_joint_idx = 0

        self.instructions = [
            "=== Object/Robot Control ===",
            "",
            "   Movement:",
            "     UP    - Move +X",
            "     DOWN  - Move -X",
            "     LEFT  - Move +Y",
            "     RIGHT - Move -Y",
            "     e     - Move +Z",
            "     d     - Move -Z",
            "",
            "   Rotation:",
            "     q     - Roll +",
            "     w     - Roll -",
            "     a     - Pitch +",
            "     s     - Pitch -",
            "     z     - Yaw +",
            "     x     - Yaw -",
            "",
            "   Joint Control:",
            "     UP    - Increase angle",
            "     DOWN  - Decrease angle",
            "     LEFT  - Previous joint",
            "     RIGHT - Next joint",
            "",
            "   Control:",
            "     TAB   - Switch entity",
            "     j     - Toggle joint mode",
            "     c     - Save poses",
            "     ESC   - Quit",
        ]

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def get_selected_entity(self) -> tuple[str, str]:
        """Returns (entity_name, entity_type)"""
        return self.entity_names[self.selected_idx], self.entity_types[self.selected_idx]

    def switch_entity(self):
        """Switch to next entity (TAB key)"""
        self.selected_idx = (self.selected_idx + 1) % len(self.entity_names)
        name, etype = self.get_selected_entity()
        self.selected_joint_idx = 0

        joints = self.entity_joints.get(name, [])
        if self.joint_mode and not joints:
            self.joint_mode = False
            log.info(f"Auto-exited joint mode: {name} has no controllable joints")

        etype_tag = "[R]" if etype == "robot" else "[O]"
        joint_count = len(joints)

        if joint_count > 0:
            log.info(
                f"Selected: {etype_tag} {name} ({self.selected_idx + 1}/{len(self.entity_names)}) - {joint_count} joints"
            )
        else:
            log.info(f"Selected: {etype_tag} {name} ({self.selected_idx + 1}/{len(self.entity_names)}) - No joints")

    def toggle_joint_mode(self):
        """Toggle joint control mode (J key)"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])

        if not joints:
            log.warning(f"{name} has no controllable joints")
            return

        self.joint_mode = not self.joint_mode
        self.selected_joint_idx = 0

        if self.joint_mode:
            log.info(f"Joint mode ENABLED for {name} ({len(joints)} joints)")
        else:
            log.info("Joint mode DISABLED")

    def get_selected_joint(self) -> str | None:
        """Get currently selected joint name"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])
        if joints and 0 <= self.selected_joint_idx < len(joints):
            return joints[self.selected_joint_idx]
        return None

    def next_joint(self):
        """Select next joint"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])
        if joints:
            self.selected_joint_idx = (self.selected_joint_idx + 1) % len(joints)
            log.info(f"Selected joint: {joints[self.selected_joint_idx]} ({self.selected_joint_idx + 1}/{len(joints)})")

    def prev_joint(self):
        """Select previous joint"""
        name, _ = self.get_selected_entity()
        joints = self.entity_joints.get(name, [])
        if joints:
            self.selected_joint_idx = (self.selected_joint_idx - 1) % len(joints)
            log.info(f"Selected joint: {joints[self.selected_joint_idx]} ({self.selected_joint_idx + 1}/{len(joints)})")

    def draw_instructions(self):
        self.screen.fill((30, 30, 30))
        y = 25

        for instruction in self.instructions:
            if not instruction:
                y += 12
                continue
            color = (100, 200, 255) if "===" in instruction else (200, 200, 200)
            font = self.font if "===" in instruction else self.small_font
            text = font.render(instruction, True, color)
            self.screen.blit(text, (25, y))
            y += 30 if "===" in instruction else 24

        y += 20

        # Show joint info if in joint mode
        if self.joint_mode:
            y += 35
            joint_name = self.get_selected_joint()
            name, _ = self.get_selected_entity()
            joints = self.entity_joints.get(name, [])

            if joint_name:
                joint_text = f"Joint: {joint_name} ({self.selected_joint_idx + 1}/{len(joints)})"
                text = self.small_font.render(joint_text, True, (255, 255, 100))
                self.screen.blit(text, (25, y))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


def process_input(dpos: float = 0.01, drot: float = 0.05, dangle: float = 0.02, joint_mode: bool = False):
    """Process keyboard input - returns delta position, delta rotation, joint angle delta, and control flags"""
    keys = pygame.key.get_pressed()

    if not hasattr(process_input, "tab_pressed"):
        process_input.tab_pressed = False
        process_input.c_pressed = False
        process_input.j_pressed = False
        process_input.left_pressed = False
        process_input.right_pressed = False

    # In joint mode, arrow keys control joints
    if joint_mode:
        dangle_val = dangle * (keys[pygame.K_UP] - keys[pygame.K_DOWN])

        prev_joint = False
        if keys[pygame.K_LEFT] and not process_input.left_pressed:
            prev_joint = True
            process_input.left_pressed = True
        elif not keys[pygame.K_LEFT]:
            process_input.left_pressed = False

        next_joint = False
        if keys[pygame.K_RIGHT] and not process_input.right_pressed:
            next_joint = True
            process_input.right_pressed = True
        elif not keys[pygame.K_RIGHT]:
            process_input.right_pressed = False

        return None, None, dangle_val, False, False, True, prev_joint, next_joint

    else:
        # Normal mode: position and rotation control
        dx = dpos * (keys[pygame.K_UP] - keys[pygame.K_DOWN])
        dy = dpos * (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT])
        dz = dpos * (keys[pygame.K_e] - keys[pygame.K_d])

        droll = drot * (keys[pygame.K_q] - keys[pygame.K_w])
        dpitch = drot * (keys[pygame.K_a] - keys[pygame.K_s])
        dyaw = drot * (keys[pygame.K_z] - keys[pygame.K_x])

        return [dx, dy, dz], [droll, dpitch, dyaw], 0.0, False, False, False, False, False


def process_common_input():
    """Process common input (TAB, J, C, ESC)"""
    keys = pygame.key.get_pressed()

    if not hasattr(process_common_input, "tab_pressed"):
        process_common_input.tab_pressed = False
        process_common_input.c_pressed = False
        process_common_input.j_pressed = False

    switch = False
    if keys[pygame.K_TAB] and not process_common_input.tab_pressed:
        switch = True
        process_common_input.tab_pressed = True
    elif not keys[pygame.K_TAB]:
        process_common_input.tab_pressed = False

    save_poses = False
    if keys[pygame.K_c] and not process_common_input.c_pressed:
        save_poses = True
        process_common_input.c_pressed = True
    elif not keys[pygame.K_c]:
        process_common_input.c_pressed = False

    toggle_joint = False
    if keys[pygame.K_j] and not process_common_input.j_pressed:
        toggle_joint = True
        process_common_input.j_pressed = True
    elif not keys[pygame.K_j]:
        process_common_input.j_pressed = False

    return switch, save_poses, toggle_joint


def obs_to_state_dict(obs, scenario):
    """Convert observation to state dictionary format for handler.set_states()"""
    state_dict = {"objects": {}, "robots": {}}

    # Extract object states
    for obj_name, obj_state in obs.objects.items():
        pos = obj_state.root_state[0, :3]
        quat = obj_state.root_state[0, 3:7]

        obj_entry = {
            "pos": pos,
            "rot": quat,
        }

        # Add joint positions if available
        if obj_state.joint_pos is not None:
            obj_cfg = next((obj for obj in scenario.objects if obj.name == obj_name), None)
            if obj_cfg and hasattr(obj_cfg, "actuators") and obj_cfg.actuators:
                joint_names = sorted(obj_cfg.actuators.keys())
                joint_positions = obj_state.joint_pos[0]
                obj_entry["dof_pos"] = {name: pos.item() for name, pos in zip(joint_names, joint_positions)}

        state_dict["objects"][obj_name] = obj_entry

    # Extract robot states
    for robot_name, robot_state in obs.robots.items():
        pos = robot_state.root_state[0, :3]
        quat = robot_state.root_state[0, 3:7]

        robot_entry = {
            "pos": pos,
            "rot": quat,
        }

        # Add joint positions
        robot_cfg = next((robot for robot in scenario.robots if robot.name == robot_name), None)
        if robot_cfg and robot_cfg.actuators:
            joint_names = sorted(robot_cfg.actuators.keys())
            joint_positions = robot_state.joint_pos[0]
            robot_entry["dof_pos"] = {name: pos.item() for name, pos in zip(joint_names, joint_positions)}

        state_dict["robots"][robot_name] = robot_entry

    return state_dict


@configclass
class Args:
    task: str = "put_banana"
    robot: str = "franka"
    scene: str | None = None
    render: RenderCfg = RenderCfg()

    ## Simulator
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None

    num_envs: int = 1
    headless: bool = True

    ## Control parameters
    step_size: float = 0.01
    rot_size: float = 0.05
    angle_size: float = 0.02

    ## Physics settings
    enable_gravity: bool = False  # Whether to enable gravity for objects and robots

    ## Viser visualization
    enable_viser: bool = True
    viser_port: int = 8080

    ## Display
    display_camera: bool = True  # Whether to display camera view in real-time
    display_width: int = 1200  # Display window width (adjusted for dual camera split-screen)
    display_height: int = 600  # Display window height

    ## Step timing
    min_step_time: float = 0.001

    def __post_init__(self):
        log.info(f"Args: {self}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Download EmbodiedGen assets from huggingface dataset
    data_dir = "roboverse_data/assets/EmbodiedGenData"
    snapshot_download(
        repo_id="HorizonRobotics/EmbodiedGenData",
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns="demo_assets/*",
        local_dir_use_symlinks=False,
    )

    # Get task class and create scenario
    task_cls = get_task_class(args.task)

    # Add cameras for visualization
    camera1 = PinholeCameraCfg(name="camera_1", pos=(2.0, -2.0, 2.0), look_at=(0.0, 0.0, 0.0))
    camera2 = PinholeCameraCfg(name="camera_2", pos=(2.5, -1.2, 2.5), look_at=(0.0, 0.0, 0.0))

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

    # Configure gravity for all objects and robots based on args
    if not args.enable_gravity:
        for obj in scenario.objects:
            obj.enabled_gravity = False
        for robot in scenario.robots:
            robot.enabled_gravity = False
        log.info("Gravity disabled for all objects and robots")
    else:
        log.info("Gravity enabled for all objects and robots")

    log.info(f"Using simulator: {args.sim}")

    # Create task environment
    tic = time.time()
    device = torch.device("cuda")
    env = task_cls(scenario, device=device)

    # Optionally wrap with Viser visualization
    if args.enable_viser:
        from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

        env = TaskViserWrapper(env, port=args.viser_port)
        log.info(f"Viser visualization enabled on port {args.viser_port}")

    toc = time.time()
    log.info(f"Time to launch: {toc - tic:.2f}s")

    # Reset environment
    tic = time.time()
    obs, extras = env.reset()
    toc = time.time()
    log.info(f"Time to reset: {toc - tic:.2f}s")

    # Build entity list: objects first, then robots
    entity_names = [obj.name for obj in scenario.objects] + [robot.name for robot in scenario.robots]
    entity_types = ["object"] * len(scenario.objects) + ["robot"] * len(scenario.robots)

    # Get joint information for each entity
    entity_joints = {}

    # Get joints from objects
    for obj in scenario.objects:
        if hasattr(obj, "actuators") and obj.actuators:
            entity_joints[obj.name] = list(obj.actuators.keys())
        else:
            entity_joints[obj.name] = []

    # Get joints from robots
    for robot in scenario.robots:
        if hasattr(robot, "actuators") and robot.actuators:
            entity_joints[robot.name] = list(robot.actuators.keys())
        else:
            entity_joints[robot.name] = []

    keyboard_client = ObjectKeyboardClient(entity_names, entity_types, entity_joints)

    log.info("=" * 60)
    log.info("Keyboard Object/Robot Control (Task-based)")
    log.info("Use Arrow keys + e/d for position, q/w/a/s/z/x for rotation")
    log.info("Press J to enter joint control mode")
    log.info("TAB to switch entity, C to save poses, ESC to quit")
    log.info("=" * 60)

    os.makedirs("get_started/output", exist_ok=True)

    step = 0
    running = True

    while running:
        step_start_time = time.time()

        running = keyboard_client.update()
        if not running or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            break

        # Process common input (TAB, J, C)
        switch, save_poses_flag, toggle_joint = process_common_input()

        if switch:
            keyboard_client.switch_entity()

        if toggle_joint:
            keyboard_client.toggle_joint_mode()

        # Save poses when C is pressed
        if save_poses_flag:
            saved_file = save_poses_to_file(obs, scenario, filename="get_started/output/saved_poses.py")
            log.info(f"Poses saved to: {saved_file}")

        # Process mode-specific input
        delta_pos, delta_rot, delta_angle, _, _, in_joint_mode, prev_joint, next_joint = process_input(
            dpos=args.step_size, drot=args.rot_size, dangle=args.angle_size, joint_mode=keyboard_client.joint_mode
        )

        # Handle joint selection
        if prev_joint:
            keyboard_client.prev_joint()
        if next_joint:
            keyboard_client.next_joint()

        # Get current selection
        selected_name, selected_type = keyboard_client.get_selected_entity()

        keyboard_client.draw_instructions()

        # Convert observation to state dict for manipulation
        state_dict = obs_to_state_dict(obs, scenario)

        # Handle joint mode updates
        if keyboard_client.joint_mode and abs(delta_angle) > 1e-6:
            joint_name = keyboard_client.get_selected_joint()
            if joint_name:
                if selected_type == "object" and selected_name in state_dict["objects"]:
                    if "dof_pos" in state_dict["objects"][selected_name]:
                        if joint_name in state_dict["objects"][selected_name]["dof_pos"]:
                            state_dict["objects"][selected_name]["dof_pos"][joint_name] += delta_angle

                elif selected_type == "robot" and selected_name in state_dict["robots"]:
                    if "dof_pos" in state_dict["robots"][selected_name]:
                        if joint_name in state_dict["robots"][selected_name]["dof_pos"]:
                            state_dict["robots"][selected_name]["dof_pos"][joint_name] += delta_angle

                # Set the updated state
                env.handler.set_states([state_dict] * scenario.num_envs)

        # Handle normal mode updates (position/rotation)
        elif not keyboard_client.joint_mode and delta_pos is not None and delta_rot is not None:
            delta_pos_tensor = torch.tensor(delta_pos, device=device)
            delta_rot_tensor = torch.tensor(delta_rot, device=device)

            has_input = delta_pos_tensor.abs().sum() > 0 or delta_rot_tensor.abs().sum() > 0

            if has_input:
                # Check if it's an object or robot
                if selected_type == "object" and selected_name in state_dict["objects"]:
                    # Update position
                    if delta_pos_tensor.abs().sum() > 0:
                        pos = state_dict["objects"][selected_name]["pos"]
                        new_pos = pos + delta_pos_tensor
                        state_dict["objects"][selected_name]["pos"] = new_pos

                    # Update rotation
                    if delta_rot_tensor.abs().sum() > 0:
                        current_rot = state_dict["objects"][selected_name]["rot"]
                        delta_rot_mat = matrix_from_euler(delta_rot_tensor.unsqueeze(0), "XYZ")
                        delta_quat = quat_from_matrix(delta_rot_mat)[0]
                        new_rot = quat_mul(current_rot.unsqueeze(0), delta_quat.unsqueeze(0))[0]
                        state_dict["objects"][selected_name]["rot"] = new_rot

                elif selected_type == "robot" and selected_name in state_dict["robots"]:
                    # Update position
                    if delta_pos_tensor.abs().sum() > 0:
                        pos = state_dict["robots"][selected_name]["pos"]
                        new_pos = pos + delta_pos_tensor
                        state_dict["robots"][selected_name]["pos"] = new_pos

                    # Update rotation
                    if delta_rot_tensor.abs().sum() > 0:
                        current_rot = state_dict["robots"][selected_name]["rot"]
                        delta_rot_mat = matrix_from_euler(delta_rot_tensor.unsqueeze(0), "XYZ")
                        delta_quat = quat_from_matrix(delta_rot_mat)[0]
                        new_rot = quat_mul(current_rot.unsqueeze(0), delta_quat.unsqueeze(0))[0]
                        state_dict["robots"][selected_name]["rot"] = new_rot

                # Set the updated state
                env.handler.set_states([state_dict] * scenario.num_envs)

        # Create dummy action (hold current joint positions)
        # For object layout, we don't need to control robot actions
        actions = []
        for _ in range(scenario.num_envs):
            env_action = {}
            for robot in scenario.robots:
                robot_state = obs.robots[robot.name]
                joint_names = sorted(robot.actuators.keys())
                joint_positions = robot_state.joint_pos[0]

                env_action[robot.name] = {
                    "dof_pos_target": {name: pos.item() for name, pos in zip(joint_names, joint_positions)}
                }
            actions.append(env_action)

        # Step the environment
        obs, reward, success, time_out, extras = env.step(actions)

        # Display camera observation if requested
        if args.display_camera:
            running = display_obs(obs, args.display_width, args.display_height)
            if not running:
                break

        # Enforce minimum step time
        step_elapsed_time = time.time() - step_start_time
        if step_elapsed_time < args.min_step_time:
            time.sleep(args.min_step_time - step_elapsed_time)

        step += 1

    # Close OpenCV camera display window if it exists
    if args.display_camera:
        cv2.destroyAllWindows()
        log.info("OpenCV camera display window closed")

    keyboard_client.close()
    env.close()

    log.info(f"Done (steps: {step})")
