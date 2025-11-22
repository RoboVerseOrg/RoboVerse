"""Viser visualization utilities for RoboVerse demos.

This module provides a unified ViserVisualizer class for interactive 3D visualization
of robots, objects, and trajectories using the viser library.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import MISSING
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser

from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import RobotCfg as BaseRobotCfg

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Constants
DEFAULT_GRIPPER_OPEN_WIDTH = 0.04  # Default gripper width when open
DEFAULT_IK_TARGET_POS = [0.3, 0.0, 0.6]  # Default IK target position
DEFAULT_IK_TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]  # Default IK target quaternion (w,x,y,z)
QUAT_NORMALIZE_EPSILON = 1e-6  # Epsilon for quaternion normalization


def normalize_quaternion(quat: list[float]) -> list[float]:
    """Normalize a quaternion to unit length.

    Args:
        quat: Quaternion in [w, x, y, z] format

    Returns:
        Normalized quaternion in [w, x, y, z] format
    """
    quat_norm = math.sqrt(sum(q * q for q in quat))
    if quat_norm > QUAT_NORMALIZE_EPSILON:
        return [q / quat_norm for q in quat]
    else:
        return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion


def patch_urdf_mesh_paths_to_absolute(urdf_path: str) -> str:
    """Convert relative mesh paths in URDF to absolute paths.

    Also removes non-mesh visual geometries (box, sphere, cylinder) from visual elements
    when a mesh visual exists, to avoid rendering collision approximations as visuals.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        Path to the patched URDF file (temporary file if modifications were made)
    """
    urdf_path_obj = Path(urdf_path)
    tree = ET.parse(urdf_path_obj)
    root = tree.getroot()
    changed = False

    # Convert mesh paths to absolute paths first
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        # pass when existing an absolute path
        if os.path.isabs(filename) and os.path.exists(filename):
            continue
        # find candidate paths
        candidates = []
        # package://xxx/
        if filename.startswith("package://"):
            rel_path = filename.split("package://", 1)[1]
            candidates.append(urdf_path_obj.parent / rel_path)
        # relative paths
        candidates += [
            urdf_path_obj.parent / filename,
            urdf_path_obj.parent / "meshes" / filename,
            urdf_path_obj.parent / "visual" / filename,
            urdf_path_obj.parent / "collision" / filename,
            urdf_path_obj.parent.parent / "meshes" / filename,
        ]
        abs_path = None
        for cand in candidates:
            if os.path.exists(cand):
                abs_path = str(cand.resolve())
                break
        if abs_path:
            mesh.set("filename", abs_path)
            changed = True
        else:
            logger.warning(f"[URDF-patch] mesh file not found for {filename}")

    # Remove non-mesh visual geometries ONLY if a mesh visual exists in the same link
    # This avoids rendering collision approximations while keeping them as fallback
    for link in root.findall(".//link"):
        link_name = link.get("name", "unnamed")
        # Check if this link has any mesh visual
        has_mesh_visual = False
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is not None and geometry.find("mesh") is not None:
                has_mesh_visual = True
                break

        # Only remove non-mesh visuals if we have at least one mesh visual
        if has_mesh_visual:
            visuals_to_remove = []
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is not None:
                    # Check if geometry contains box, sphere, or cylinder (not mesh)
                    if (
                        geometry.find("box") is not None
                        or geometry.find("sphere") is not None
                        or geometry.find("cylinder") is not None
                    ):
                        visuals_to_remove.append(visual)

            # Remove the identified visual elements
            if visuals_to_remove:
                logger.info(
                    f"[URDF-patch] Link '{link_name}': Has mesh visual, removing {len(visuals_to_remove)} non-mesh visual(s)"
                )
                for visual in visuals_to_remove:
                    link.remove(visual)
                    changed = True

    if changed:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf")
        tree.write(tmp.name)
        return tmp.name
    else:
        return str(urdf_path_obj)


def load_urdf_with_patch(urdf_path: str) -> tuple[Any, str]:
    """Load URDF file with automatic mesh path patching.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        Tuple of (URDF object, patched URDF path)
    """
    import yourdfpy

    patched_urdf_path = patch_urdf_mesh_paths_to_absolute(urdf_path)
    try:
        urdf_obj = yourdfpy.URDF.load(patched_urdf_path)
        return urdf_obj, patched_urdf_path
    except Exception as e:
        logger.error(f"Error loading URDF {patched_urdf_path}: {e}")
        logger.error("This might be due to empty or malformed collision/geometry elements")
        logger.error("Please check that all collision elements have valid geometry tags")
        raise e


class ViserVisualizer:
    """Interactive 3D visualizer for robots and objects using viser.

    This class provides comprehensive visualization and control capabilities including:
    - Loading and displaying URDF models (robots and objects)
    - Primitive shape visualization (cubes, spheres, cylinders)
    - Interactive joint control via GUI sliders
    - IK (Inverse Kinematics) control with visual targets
    - Trajectory playback and recording
    - Camera controls and preset views

    Args:
        port: Port number for the viser server (default: 8080)
    """

    def __init__(self, port: int = 8080) -> None:
        self.server = viser.ViserServer(port=port)
        self._urdf_handles = {}
        self._mesh_handles = {}  # primitive mesh handle
        self._frame_handles = {}  # frame handle
        self._camera_update_callback = None  # camera view

        # trajectory
        self._trajectory_data = None
        self._trajectory_file_path = None
        self._current_trajectory = None
        self._trajectory_playing = False
        self._trajectory_paused = False
        self._current_frame = 0
        self._trajectory_timer = None
        self._trajectory_start_time = None

        # joint control
        self._joint_sliders = {}  # robot_name -> list of sliders
        self._joint_limits = {}  # robot_name -> joint limits dict
        self._initial_configs = {}  # robot_name -> initial joint config
        self._joint_folders = {}  # robot_name -> GUI folder handle
        self._current_joint_positions = {}  # robot_name -> current joint positions (persistent)

        # IK control
        self._robot_ik_solvers = {}  # robot_name -> IKSolver instance
        self._robot_configs = {}  # robot_name -> robot config
        self._ik_target_positions = {}  # robot_name -> [x, y, z] target position
        self._ik_target_orientations = {}  # robot_name -> [w, x, y, z] target orientation
        self._ik_sliders = {}  # robot_name -> dict of IK control sliders
        self._ik_folders = {}  # robot_name -> IK GUI folder handle
        self._ik_target_markers = {}  # robot_name -> target visualization marker
        self._ik_orientation_frames = {}  # robot_name -> orientation visualization frame
        self._env_handler = None  # reference to environment handler for getting current states

    def add_urdf(self, name: str, urdf_path: str, scale: float = 1.0, root_node_name: str | None = None) -> Any | None:
        """Add a URDF model to the scene.

        Args:
            name: Name identifier for the URDF model
            urdf_path: Path to the URDF file
            scale: Scale factor for the model (default: 1.0)
            root_node_name: Root node name in the scene graph (default: "/{name}")

        Returns:
            URDF handle if successful, None otherwise
        """
        try:
            from viser.extras import ViserUrdf

            urdf_obj, patched_urdf_path = load_urdf_with_patch(urdf_path)

            if root_node_name is None:
                root_node_name = f"/{name}"
            urdf_handle = ViserUrdf(
                self.server,
                Path(patched_urdf_path),
                scale=scale,
                root_node_name=root_node_name,
                load_meshes=True,
                load_collision_meshes=False,
            )
            logger.info(f"Successfully loaded URDF {name} with {len(urdf_handle.get_actuated_joint_names())} joints")
            return urdf_handle
        except Exception as e:
            logger.error(f"Error loading URDF {urdf_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def add_frame(self, name: str, show_axes: bool = True) -> Any:
        """Add a coordinate frame to the scene.

        Args:
            name: Name of the frame
            show_axes: Whether to show the axes (default: True)

        Returns:
            Frame handle
        """
        handle = self.server.scene.add_frame(name, show_axes=show_axes)
        self._frame_handles[name] = handle
        return handle

    def add_grid(self) -> Any:
        """Add a ground grid to the scene.

        Returns:
            Grid handle
        """
        return self.server.scene.add_grid("/grid")

    def visualize_scenario_items(self, items: list | dict, item_states: dict | None = None) -> None:
        """Visualize a collection of scenario items (objects or robots).

        Args:
            items: List or dict of item configurations to visualize
            item_states: Optional dict mapping item names to their states (pos, rot, etc.)
        """
        if item_states is None:
            item_states = {}
        if isinstance(items, list):
            for item_cfg in items:
                item_name = item_cfg.name
                item_state = item_states.get(item_name, {})
                self.visualize_item(item_cfg, item_name, item_state)
        elif isinstance(items, dict):
            for item_name, item_cfg in items.items():
                item_state = item_states.get(item_name, {})
                self.visualize_item(item_cfg, item_name, item_state)
        else:
            logger.warning(f"Unsupported objects type {type(items)}")

    def visualize_item(
        self,
        cfg: PrimitiveCubeCfg
        | PrimitiveSphereCfg
        | PrimitiveCylinderCfg
        | RigidObjCfg
        | ArticulationObjCfg
        | BaseRobotCfg,
        name: str,
        state: dict | None = None,
        base_offset_debug: bool = True,
    ):
        """Visualize a single item (robot or object) in the 3D scene.

        Args:
            cfg: Configuration object for the item (cube, sphere, cylinder, rigid object, articulated object, or robot)
            name: Name identifier for the item
            state: Optional state dictionary containing position, rotation, and joint positions
            base_offset_debug: Whether to apply base offset for debugging
        """
        # Debug: print out the states for debugging
        logger.debug(f"[Viser] {name} state: {state}")

        # initialize the item position
        position = None

        if state and "pos" in state and state["pos"] is not None:
            position = state["pos"]

        # set to default position if needed
        if position is None:
            position = cfg.default_position
        elif isinstance(position, (list, tuple)):
            if len(position) >= 3:
                position = (float(position[0]), float(position[1]), float(position[2]))
            else:
                position = cfg.default_position
        else:
            # if position is not a list/turple/None, set to default
            position = cfg.default_position

        # initialize the item rotation
        rotation = None
        if state and "rot" in state and state["rot"] is not None:
            rotation = state["rot"]

        if rotation is None:
            rotation = cfg.default_orientation
        elif isinstance(rotation, (list, tuple)):
            if len(rotation) >= 4:
                rotation = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
            else:
                rotation = cfg.default_orientation
        else:
            rotation = cfg.default_orientation

        # set color for primitive objects
        color = None
        # current item cfg inherits from class _PrimitiveMixin or not
        if hasattr(cfg, "__class__") and hasattr(cfg.__class__, "__mro__"):
            is_primitive_mixin = any("_PrimitiveMixin" in str(base) for base in cfg.__class__.__mro__)
        else:
            is_primitive_mixin = False

        if isinstance(cfg, (PrimitiveCubeCfg, PrimitiveSphereCfg, PrimitiveCylinderCfg)) or is_primitive_mixin:
            if hasattr(cfg, "color") and cfg.color != MISSING:
                if len(cfg.color) >= 3:
                    color = (float(cfg.color[0]), float(cfg.color[1]), float(cfg.color[2]))

        # URDF visualization
        if isinstance(cfg, (RigidObjCfg, ArticulationObjCfg, BaseRobotCfg)) and getattr(cfg, "urdf_path", None):
            scale = cfg.scale
            urdf_obj, patched_urdf_path = load_urdf_with_patch(cfg.urdf_path)

            # base link offset
            base_offset = np.zeros(3)
            try:
                base_link_name = urdf_obj.base_link
                if isinstance(base_link_name, str):
                    base_link = urdf_obj.link_map[base_link_name]
                else:
                    base_link = base_link_name
                if hasattr(base_link, "visuals") and base_link.visuals:
                    for visual in base_link.visuals:
                        if hasattr(visual, "origin") and visual.origin is not None:
                            base_offset = np.array(visual.origin[:3, 3])
                            break
            except Exception as e:
                logger.warning(f"Error parsing base link for {name}: {e}")
                base_offset = np.zeros(3)

            # standard frame and base offset
            if base_offset_debug and np.linalg.norm(base_offset) > 1e-6:
                logger.debug(f"{name} has base offset: {base_offset}")
                if position is not None:
                    adjusted_position = (
                        position[0] - base_offset[0] * scale[0],
                        position[1] - base_offset[1] * scale[1],
                        position[2] - base_offset[2] * scale[2],
                    )
                else:
                    adjusted_position = (
                        -base_offset[0] * scale[0],
                        -base_offset[1] * scale[1],
                        -base_offset[2] * scale[2],
                    )
                logger.debug(f"add_frame: /{name}_frame")
                parent_frame = self.add_frame(f"/{name}_frame", show_axes=False)
                logger.debug(f"self._frame_handles now: {list(self._frame_handles.keys())}")
                if rotation is not None:
                    parent_frame.wxyz = rotation
                urdf_handle = self.add_urdf(name, patched_urdf_path, scale=scale, root_node_name=f"/{name}_frame")
            else:
                if base_offset_debug:
                    logger.debug(f"{name} has no base offset")
                parent_frame = self.add_frame(f"/{name}_frame", show_axes=False)
                if position is not None:
                    parent_frame.position = position
                if rotation is not None:
                    parent_frame.wxyz = rotation
                urdf_handle = self.add_urdf(name, patched_urdf_path, scale=scale, root_node_name=f"/{name}_frame")
            self._urdf_handles[name] = urdf_handle

            # ====== standard dof pos ======
            if urdf_handle and state and "dof_pos" in state and state["dof_pos"] is not None:
                dof_pos = state["dof_pos"]
                joint_names = urdf_handle.get_actuated_joint_names()
                if isinstance(dof_pos, dict):
                    dof_pos_list = [dof_pos.get(jn, 0.0) for jn in joint_names]
                else:
                    dof_pos_list = list(dof_pos)
                if len(dof_pos_list) < len(joint_names):
                    dof_pos_list += [0.0] * (len(joint_names) - len(dof_pos_list))
                elif len(dof_pos_list) > len(joint_names):
                    dof_pos_list = dof_pos_list[: len(joint_names)]
                logger.debug(f"[Viser] {name} joint order: {joint_names}")
                logger.debug(f"[Viser] {name} dof_pos: {dof_pos_list}")
                urdf_handle.update_cfg(np.array(dof_pos_list, dtype=np.float32))

                # Store initial configuration for robots (for reset functionality)
                if isinstance(cfg, BaseRobotCfg):
                    if isinstance(dof_pos, dict):
                        self._initial_configs[name] = dof_pos.copy()
                    else:
                        # Convert list to dict using joint names
                        self._initial_configs[name] = {jn: val for jn, val in zip(joint_names, dof_pos_list)}
                    logger.info(f"Stored initial config for robot {name}")

        # primitive item visualization
        elif isinstance(cfg, PrimitiveCubeCfg):
            # viser add_box
            half_size = cfg.half_size
            dimensions = (half_size[0] * 2, half_size[1] * 2, half_size[2] * 2)
            handle = self.server.scene.add_box(
                name=f"/{name}",
                position=position,
                dimensions=dimensions,
                color=color,
                wxyz=rotation,
            )
            self._mesh_handles[f"/{name}"] = handle
        elif isinstance(cfg, PrimitiveSphereCfg):
            # viser add_icosphere
            handle = self.server.scene.add_icosphere(
                name=f"/{name}",
                radius=cfg.radius,
                color=color,
                position=position,
                wxyz=rotation,
            )
            self._mesh_handles[f"/{name}"] = handle
        elif isinstance(cfg, PrimitiveCylinderCfg):
            import trimesh

            radius = cfg.radius
            height = cfg.height

            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)

            handle = self.server.scene.add_mesh_trimesh(
                name=f"/{name}",
                mesh=mesh,
                position=position,
                wxyz=rotation,
            )
            self._mesh_handles[f"/{name}"] = handle
        else:
            logger.warning(f"Unsupported object type {type(cfg)} for {name}")

    def update_item_pose(self, name: str, state: dict) -> None:
        """Update the pose of a visualized item.

        Supports both URDF models and primitive objects rendered via visualize_item.

        Args:
            name: Name of the item to update
            state: State dict containing 'pos', 'rot', and optionally 'dof_pos' for articulated objects
        """
        import numpy as np

        # 1. URDF/robots
        if name in self._urdf_handles:
            urdf_handle = self._urdf_handles[name]
            # update base frame
            frame_name = f"/{name}_frame"

            if frame_name in self._frame_handles:
                frame = self._frame_handles[frame_name]
                if "pos" in state and state["pos"] is not None:
                    frame.position = np.array(state["pos"], dtype=np.float32)
                if "rot" in state and state["rot"] is not None:
                    frame.wxyz = np.array(state["rot"], dtype=np.float32)

            # update dof_pos
            if "dof_pos" in state and state["dof_pos"] is not None:
                joint_names = urdf_handle.get_actuated_joint_names()
                dof_pos = state["dof_pos"]
                if isinstance(dof_pos, dict):
                    dof_pos_list = [dof_pos.get(jn, 0.0) for jn in joint_names]
                else:
                    dof_pos_list = list(dof_pos)
                if len(dof_pos_list) < len(joint_names):
                    dof_pos_list += [0.0] * (len(joint_names) - len(dof_pos_list))
                elif len(dof_pos_list) > len(joint_names):
                    dof_pos_list = dof_pos_list[: len(joint_names)]
                urdf_handle.update_cfg(np.array(dof_pos_list, dtype=np.float32))
            return

        # 2. Primitive mesh (cube/sphere/etc.）
        mesh_name = f"/{name}"
        if mesh_name in self._mesh_handles:
            mesh = self._mesh_handles[mesh_name]
            if "pos" in state and state["pos"] is not None:
                mesh.position = np.array(state["pos"], dtype=np.float32)
            if "rot" in state and state["rot"] is not None:
                mesh.wxyz = np.array(state["rot"], dtype=np.float32)
            return

    def refresh_camera_view(self):
        """Refresh camera view."""
        if self._camera_update_callback is not None:
            import threading

            threading.Thread(target=self._camera_update_callback, daemon=True).start()
        else:
            logger.warning("Camera view callback not available. Make sure enable_camera_controls() has been called.")

    def load_trajectory(self, trajectory_path: str) -> bool:
        """Load a trajectory file for playback.

        Supports .pkl, .pkl.gz, .json, and .yaml formats.

        Args:
            trajectory_path: Path to the trajectory file
        """
        try:
            from metasim.utils.demo_util.loader import load_traj_file

            self._trajectory_data = load_traj_file(trajectory_path)
            self._trajectory_file_path = trajectory_path  # Store the path for GUI update
            logger.info(f"Loaded trajectory from {trajectory_path}")

            # Log available robots and trajectories
            if isinstance(self._trajectory_data, dict):
                for robot_name, demos in self._trajectory_data.items():
                    logger.info(f"  Robot: {robot_name}, Demos: {len(demos)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            return False

    def get_available_trajectories(self):
        """Get list of available robot trajectories.

        Returns:
            List of (robot_name, num_demos) tuples
        """
        if self._trajectory_data is None:
            return []

        trajectories = []
        for robot_name, demos in self._trajectory_data.items():
            trajectories.append((robot_name, len(demos)))
        return trajectories

    def set_current_trajectory(self, robot_name: str, demo_index: int = 0):
        """Set the current trajectory to play.

        Args:
            robot_name: Name of the robot
            demo_index: Index of the demo to play (default: 0)
        """
        if self._trajectory_data is None:
            logger.error("No trajectory data loaded")
            return False

        if robot_name not in self._trajectory_data:
            logger.error(f"Robot {robot_name} not found in trajectory data")
            return False

        demos = self._trajectory_data[robot_name]
        if demo_index >= len(demos):
            logger.error(f"Demo index {demo_index} out of range for robot {robot_name}")
            return False

        self._current_trajectory = demos[demo_index]
        self._current_frame = 0
        logger.info(
            f"Set current trajectory: {robot_name}, demo {demo_index}, {len(self._current_trajectory['actions'])} frames"
        )
        return True

    def load_and_set_trajectory(self, trajectory_path: str, robot_name: str | None = None, demo_index: int = 0):
        """Load trajectory file and optionally set current trajectory.

        Args:
            trajectory_path: Path to trajectory file
            robot_name: Robot name to set (if None, will not set trajectory)
            demo_index: Demo index to set (default: 0)
        """
        success = self.load_trajectory(trajectory_path)
        if success and robot_name is not None:
            return self.set_current_trajectory(robot_name, demo_index)
        return success

    def get_robot_joint_info(self, robot_name: str):
        """Get joint information for a robot.

        Args:
            robot_name: Name of the robot

        Returns:
            Tuple of (urdf_handle, joint_limits, joint_names) or None if robot not found
        """
        if robot_name not in self._urdf_handles:
            logger.error(f"Robot {robot_name} not found in URDF handles")
            return None

        urdf_handle = self._urdf_handles[robot_name]
        if urdf_handle is None:
            logger.error(f"URDF handle for {robot_name} is None")
            return None

        try:
            joint_limits = urdf_handle.get_actuated_joint_limits()
            joint_names = list(joint_limits.keys())
            return urdf_handle, joint_limits, joint_names
        except Exception as e:
            logger.error(f"Failed to get joint info for {robot_name}: {e}")
            return None

    def create_joint_sliders(self, robot_name: str, client_gui):
        """Create joint control sliders for a robot.

        Args:
            robot_name: Name of the robot
            client_gui: Viser client GUI object

        Returns:
            List of slider handles
        """
        joint_info = self.get_robot_joint_info(robot_name)
        if joint_info is None:
            return []

        urdf_handle, joint_limits, joint_names = joint_info

        slider_handles = []
        initial_config = []

        # Check if we have saved positions for this robot
        has_saved_positions = robot_name in self._current_joint_positions and len(
            self._current_joint_positions[robot_name]
        ) == len(joint_limits)

        # Check if we have initial config from demo (should preserve this!)
        has_demo_initial_config = robot_name in self._initial_configs

        def make_update_callback(robot_name, slider_list):
            def update_callback(_):
                if self._urdf_handles.get(robot_name):
                    joint_values = np.array([s.value for s in slider_list])
                    self._urdf_handles[robot_name].update_cfg(joint_values)
                    # Save current joint positions for persistence
                    self._current_joint_positions[robot_name] = joint_values.tolist()
                    # Refresh camera view if enabled
                    self.refresh_camera_view()

            return update_callback

        for i, (joint_name, (lower, upper)) in enumerate(joint_limits.items()):
            # Set reasonable defaults for joint limits
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi

            # Priority: saved positions > demo initial config > computed default
            if has_saved_positions:
                # Use previously saved position
                initial_pos = self._current_joint_positions[robot_name][i]
                # Clamp to current joint limits in case they changed
                initial_pos = max(lower, min(upper, initial_pos))
            elif has_demo_initial_config:
                # Use initial config from demo file (stored when robot was visualized)
                demo_config = self._initial_configs[robot_name]
                if isinstance(demo_config, dict) and joint_name in demo_config:
                    initial_pos = demo_config[joint_name]
                elif isinstance(demo_config, list) and i < len(demo_config):
                    initial_pos = demo_config[i]
                else:
                    # Fallback to computed default
                    initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
            else:
                # Set initial position to middle of range, or 0 if range includes 0
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0

            slider = client_gui.add_slider(
                label=joint_name,
                min=float(lower),
                max=float(upper),
                step=1e-3,
                initial_value=float(initial_pos),
            )

            slider_handles.append(slider)
            initial_config.append(initial_pos)

        # Set up update callbacks after all sliders are created
        update_callback = make_update_callback(robot_name, slider_handles)
        for slider in slider_handles:
            slider.on_update(update_callback)

        # Store slider information
        self._joint_sliders[robot_name] = slider_handles
        self._joint_limits[robot_name] = joint_limits
        # Note: Preserve demo initial config if it already exists
        if not has_demo_initial_config:
            # Convert list to dict for consistency
            self._initial_configs[robot_name] = {jn: val for jn, val in zip(joint_names, initial_config)}

        # Set initial configuration and save current positions
        if urdf_handle:
            urdf_handle.update_cfg(np.array(initial_config))

        # Save current joint positions for persistence (as list for backward compatibility)
        # Note: We don't save this as dict to avoid confusion with _initial_configs
        self._current_joint_positions[robot_name] = initial_config

        status_msg = f"Created {len(slider_handles)} joint sliders for {robot_name}"
        if has_saved_positions:
            status_msg += " (restored previous positions)"
        elif has_demo_initial_config:
            status_msg += " (using demo initial positions)"
        else:
            status_msg += " (computed default positions)"

        logger.info(status_msg)
        return slider_handles

    def reset_robot_joints(self, robot_name: str):
        """Reset robot joints to initial configuration from demo.

        Args:
            robot_name: Name of the robot
        """
        if robot_name not in self._joint_sliders:
            logger.warning(f"No joint sliders found for robot {robot_name}")
            return

        if robot_name not in self._initial_configs:
            logger.warning(f"No initial config found for robot {robot_name}")
            return

        sliders = self._joint_sliders[robot_name]
        initial_config = self._initial_configs[robot_name]  # This is a dict from demo
        joint_limits = self._joint_limits[robot_name]
        joint_names = list(joint_limits.keys())

        # Update slider values - initial_config is a dict, need to extract values in order
        if isinstance(initial_config, dict):
            for slider, joint_name in zip(sliders, joint_names):
                if joint_name in initial_config:
                    slider.value = initial_config[joint_name]
                    logger.debug(f"Reset {joint_name} to {initial_config[joint_name]}")
                else:
                    logger.warning(f"Joint {joint_name} not found in initial config")
        else:
            # If it's a list (legacy support)
            for slider, init_value in zip(sliders, initial_config):
                slider.value = init_value

        # Clear saved positions so they don't override demo initial config on next setup
        if robot_name in self._current_joint_positions:
            del self._current_joint_positions[robot_name]
            logger.info(f"Cleared current joint positions for {robot_name}")

        logger.info(f"Reset joints for robot {robot_name} to demo initial configuration")

    def update_robot_joint_config(self, robot_name: str, joint_config: dict):
        """Update robot joint configuration programmatically.

        Args:
            robot_name: Name of the robot
            joint_config: Dictionary of joint_name -> value
        """
        if robot_name not in self._joint_sliders:
            logger.warning(f"No joint sliders found for robot {robot_name}")
            return

        if robot_name not in self._joint_limits:
            logger.warning(f"No joint limits found for robot {robot_name}")
            return

        sliders = self._joint_sliders[robot_name]
        joint_limits = self._joint_limits[robot_name]
        joint_names = list(joint_limits.keys())

        # Update sliders based on joint config
        for i, joint_name in enumerate(joint_names):
            if joint_name in joint_config and i < len(sliders):
                value = joint_config[joint_name]
                # Clamp value to joint limits
                lower, upper = joint_limits[joint_name]
                lower = lower if lower is not None else -np.pi
                upper = upper if upper is not None else np.pi
                value = max(lower, min(upper, value))
                sliders[i].value = float(value)

    def update_robot_joint_config_direct(self, robot_name: str, joint_config: dict):
        """Update robot joint configuration directly without sliders.

        This is useful for IK control or programmatic updates when joint control GUI is not active.

        Args:
            robot_name: Name of the robot
            joint_config: Dictionary of joint_name -> value
        """
        if robot_name not in self._urdf_handles:
            logger.warning(f"Robot {robot_name} not found in URDF handles")
            return False

        urdf_handle = self._urdf_handles[robot_name]
        if urdf_handle is None:
            logger.warning(f"URDF handle for {robot_name} is None")
            return False

        try:
            # Get joint names from URDF
            joint_names = urdf_handle.get_actuated_joint_names()

            # Convert dict config to list
            joint_values = []
            for joint_name in joint_names:
                if joint_name in joint_config:
                    joint_values.append(joint_config[joint_name])
                else:
                    # If joint not in config, use 0.0 as default
                    joint_values.append(0.0)

            # Update URDF directly
            urdf_handle.update_cfg(np.array(joint_values, dtype=np.float32))
            logger.info(f"Updated robot {robot_name} joints directly: {len(joint_values)} joints")
            return True

        except Exception as e:
            logger.error(f"Error updating robot joints directly: {e}")
            import traceback

            traceback.print_exc()
            return False

    def clear_joint_control(self, robot_name: str | None = None):
        """Clear joint control sliders for a specific robot or all robots.

        This only clears GUI elements but preserves joint positions and demo initial configs.

        Args:
            robot_name: Name of the robot to clear. If None, clear all robots.
        """
        robots_to_clear = [robot_name] if robot_name else list(self._joint_sliders.keys())

        for robot in robots_to_clear:
            if robot in self._joint_sliders:
                del self._joint_sliders[robot]
            if robot in self._joint_limits:
                del self._joint_limits[robot]
            # Note: _initial_configs stores the demo's initial pose and should persist
            if robot in self._joint_folders:
                del self._joint_folders[robot]
            # Keep self._current_joint_positions[robot] for persistence across setup/clear cycles

        if robots_to_clear:
            logger.info(f"Cleared joint control GUI for: {', '.join(robots_to_clear)} (positions preserved)")
        else:
            logger.info("No joint controls to clear")

    def apply_trajectory_frame(self, frame_index: int):
        """Apply a specific frame from the current trajectory.

        Args:
            frame_index: The frame index to apply
        """
        if self._current_trajectory is None:
            logger.error("No trajectory set")
            return False

        actions = self._current_trajectory.get("actions", [])
        if frame_index >= len(actions):
            logger.error(f"Frame index {frame_index} out of range")
            return False

        action = actions[frame_index]
        self._current_frame = frame_index

        # Extract robot state from action
        if "dof_pos_target" in action:
            # Find robot name from trajectory data
            for robot_name, demos in self._trajectory_data.items():
                if self._current_trajectory in demos:
                    robot_state = {"dof_pos": action["dof_pos_target"]}

                    # Apply initial position and rotation if at frame 0
                    if frame_index == 0 and "init_state" in self._current_trajectory:
                        init_state = self._current_trajectory["init_state"].get(robot_name, {})
                        if "pos" in init_state:
                            robot_state["pos"] = init_state["pos"]
                        if "rot" in init_state:
                            robot_state["rot"] = init_state["rot"]

                    # Update robot pose
                    self.update_item_pose(robot_name, robot_state)
                    break

        return True

    def play_trajectory(self, fps: float = 10.0):
        """Start playing the current trajectory.

        Args:
            fps: Playback frame rate
        """
        if self._current_trajectory is None:
            logger.error("No trajectory set")
            return False

        if self._trajectory_playing:
            logger.warning("Trajectory already playing")
            return False

        logger.info(f"Starting trajectory playback at {fps} FPS")
        self._trajectory_playing = True
        self._trajectory_paused = False
        self._trajectory_start_time = time.time()

        def play_frame():
            if not self._trajectory_playing or self._trajectory_paused:
                logger.debug("play_frame: stopped or paused")
                return

            actions = self._current_trajectory.get("actions", [])
            if self._current_frame < len(actions):
                logger.debug(f"Playing frame {self._current_frame}/{len(actions)}")
                self.apply_trajectory_frame(self._current_frame)
                self._current_frame += 1

                # Refresh camera view if enabled
                self.refresh_camera_view()

                # Schedule next frame
                self._trajectory_timer = threading.Timer(1.0 / fps, play_frame)
                self._trajectory_timer.start()
            else:
                # Trajectory finished
                self._trajectory_playing = False
                logger.info("Trajectory playback finished")

        play_frame()
        return True

    def pause_trajectory(self):
        """Pause trajectory playback."""
        if self._trajectory_playing:
            self._trajectory_paused = True
            if self._trajectory_timer:
                self._trajectory_timer.cancel()
            logger.info("Trajectory playback paused")

    def resume_trajectory(self, fps: float = 10.0):
        """Resume trajectory playback."""
        if self._trajectory_playing and self._trajectory_paused:
            self._trajectory_paused = False
            logger.info(f"Resuming trajectory playback at {fps} FPS from frame {self._current_frame}")

            # Resume playback logic (similar to play_trajectory but without state checks)
            def play_frame():
                if not self._trajectory_playing or self._trajectory_paused:
                    logger.debug("play_frame: stopped or paused")
                    return

                actions = self._current_trajectory.get("actions", [])
                if self._current_frame < len(actions):
                    logger.debug(f"Playing frame {self._current_frame}/{len(actions)}")
                    self.apply_trajectory_frame(self._current_frame)
                    self._current_frame += 1

                    # Refresh camera view if enabled
                    self.refresh_camera_view()

                    # Schedule next frame
                    self._trajectory_timer = threading.Timer(1.0 / fps, play_frame)
                    self._trajectory_timer.start()
                else:
                    # Trajectory finished
                    self._trajectory_playing = False
                    logger.info("Trajectory playback finished")

            play_frame()
            return True
        else:
            logger.warning("Cannot resume: trajectory not in paused state")
            return False

    def stop_trajectory(self):
        """Stop trajectory playback."""
        self._trajectory_playing = False
        self._trajectory_paused = False
        if self._trajectory_timer:
            self._trajectory_timer.cancel()
            self._trajectory_timer = None
        self._current_frame = 0
        logger.info("Trajectory playback stopped")

    def seek_trajectory(self, frame_index: int):
        """Seek to a specific frame in the trajectory.

        Args:
            frame_index: Target frame index
        """
        if self._current_trajectory is None:
            return False

        actions = self._current_trajectory.get("actions", [])
        if 0 <= frame_index < len(actions):
            was_playing = self._trajectory_playing
            if was_playing:
                self.stop_trajectory()

            self.apply_trajectory_frame(frame_index)
            self.refresh_camera_view()

            return True
        return False

    def enable_camera_controls(
        self, initial_position=None, render_width=256, render_height=256, look_at_position=None, initial_fov=45.0
    ):
        """Enable camera controls and recording for all connected clients.

        Args:
            initial_position: Initial camera position [x, y, z]. Default: [3.0, -3.0, 2.0]
            render_width: Width of rendered camera view. Default: 256
            render_height: Height of rendered camera view. Default: 256
            look_at_position: Initial look-at target [x, y, z]. Default: [0.0, 0.0, 0.0]
            initial_fov: Initial field of view in degrees. Default: 45.0
        """
        # Set default values if not provided
        if initial_position is None:
            initial_position = [3.0, -3.0, 2.0]
        if look_at_position is None:
            look_at_position = [0.0, 0.0, 0.0]

        # Create a single camera frustum for all clients
        camera_frustum = self.server.scene.add_camera_frustum(
            name="/camera_visualization",
            fov=np.radians(initial_fov),  # FOV in radians
            aspect=render_width / render_height,  # Aspect ratio based on render dimensions
            scale=0.3,
            color=(255, 255, 0),  # Yellow
            position=np.array(initial_position),
        )

        @self.server.on_client_connect
        def setup_client_camera(client):
            """Setup camera controls and recording for each connected client."""
            try:
                # Set main view camera using provided parameters
                client.camera.position = np.array([
                    initial_position[0] * 2,
                    initial_position[1] * 2,
                    initial_position[2] * 2,
                ])  # Offset main view
                client.camera.look_at = np.array(look_at_position)
                client.camera.fov = 45.0  # Keep main view FOV fixed

                # Create camera control GUI elements
                with client.gui.add_folder("Camera Controls"):
                    pos_x = client.gui.add_slider(
                        "Camera X", min=-10.0, max=10.0, step=0.2, initial_value=initial_position[0]
                    )
                    pos_y = client.gui.add_slider(
                        "Camera Y", min=-10.0, max=10.0, step=0.2, initial_value=initial_position[1]
                    )
                    pos_z = client.gui.add_slider(
                        "Camera Z", min=0.1, max=10.0, step=0.2, initial_value=initial_position[2]
                    )

                    # Add incremental camera rotation controls using buttons (compact layout)
                    with client.gui.add_folder("Camera Rotation"):
                        # Yaw controls (left/right turn) - compact pair
                        yaw_left_btn = client.gui.add_button("◄ Yaw")
                        yaw_right_btn = client.gui.add_button("► Yaw")

                        # Pitch controls (up/down tilt) - compact pair
                        pitch_up_btn = client.gui.add_button("▲ Pitch")
                        pitch_down_btn = client.gui.add_button("▼ Pitch")

                        # Roll controls (left/right roll) - compact pair
                        roll_left_btn = client.gui.add_button("↺ Roll")
                        roll_right_btn = client.gui.add_button("↻ Roll")

                    fov_slider = client.gui.add_slider(
                        "Camera FOV", min=20.0, max=90.0, step=1.0, initial_value=initial_fov
                    )

                    # Add button for one-time look-at center
                    lookat_center_btn = client.gui.add_button("Look At Center")

                    camera_info = client.gui.add_text("Camera Info", initial_value="Camera position and settings")
                    reset_btn = client.gui.add_button("Reset Camera")

                with client.gui.add_folder("Camera Presets"):
                    top_view_btn = client.gui.add_button("Top View")
                    side_view_btn = client.gui.add_button("Side View")
                    front_view_btn = client.gui.add_button("Front View")

                with client.gui.add_folder("Camera View"):
                    camera_view_info = client.gui.add_text("Camera View", initial_value="Camera view information")
                    screenshot_btn = client.gui.add_button("Take Camera Screenshot")

                    # Add recording controls
                    with client.gui.add_folder("Recording"):
                        recording_status = client.gui.add_text("Recording Status", initial_value="Not recording")
                        start_recording_btn = client.gui.add_button("Start Recording")
                        stop_recording_btn = client.gui.add_button("Stop Recording")
                        recording_fps = client.gui.add_slider("Recording FPS", min=5, max=30, step=1, initial_value=10)

                    # Add camera view image display in GUI
                    camera_image_gui = client.gui.add_image(
                        np.zeros((256, 256, 3), dtype=np.uint8),  # Initial empty image
                        label="Live Camera View",
                    )

                # Flag to prevent infinite loop when updating sliders programmatically
                updating_sliders = False

                # Recording state variables
                is_recording = False
                recording_frames = []
                recording_timer = None
                recording_start_time = None

                # Now define all functions that will be used
                def update_camera_view():
                    """Update the camera view display in GUI."""
                    try:
                        # Small delay to ensure scene updates are applied
                        time.sleep(0.01)

                        # Get camera settings from frustum
                        camera_pos = camera_frustum.position
                        camera_fov_radians = camera_frustum.fov
                        render_wxyz = camera_frustum.wxyz

                        # Calculate direction vector for debugging
                        look_at = np.array(look_at_position)
                        direction = look_at - camera_pos
                        direction = direction / np.linalg.norm(direction)

                        logger.info(f"Camera position: {camera_pos}")
                        logger.info(f"Look at: {look_at}")
                        logger.info(f"Direction: {direction}")
                        logger.info(f"Render wxyz: {render_wxyz}")
                        logger.info(f"FOV radians: {camera_fov_radians}")

                        # Try slightly offset position to avoid being exactly AT the frustum center
                        # Move slightly back along the negative direction vector
                        offset_distance = 0.01  # Small offset
                        offset_pos = camera_pos - direction * offset_distance

                        logger.info(f"Offset position: {offset_pos}")

                        # Temporarily hide camera frustum from render
                        # camera_frustum.visible = False

                        try:
                            # Render image from camera viewpoint (without seeing the frustum itself)
                            image = client.get_render(
                                height=render_height,
                                width=render_width,
                                wxyz=render_wxyz,
                                position=offset_pos,  # Use offset position
                                fov=camera_fov_radians,
                            )
                        finally:
                            # Always restore camera frustum visibility for 3D scene
                            camera_frustum.visible = True

                        logger.info(
                            f"Rendered image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
                        )

                        # Check if image is completely black or has content
                        non_zero_pixels = np.count_nonzero(image)
                        total_pixels = image.size
                        logger.info(
                            f"Non-zero pixels: {non_zero_pixels}/{total_pixels} ({100 * non_zero_pixels / total_pixels:.1f}%)"
                        )

                        # Save original for debugging
                        original_image = image.copy()

                        # Update GUI image display
                        camera_image_gui.image = original_image
                    except Exception as e:
                        logger.error(f"Failed to update camera view: {e}")
                        import traceback

                        traceback.print_exc()

                self._camera_update_callback = update_camera_view

                def take_camera_screenshot(_):
                    """Take high-resolution camera screenshot."""
                    try:
                        import imageio

                        # Show feedback
                        camera_view_info.value = "Taking screenshot..."

                        # Get camera settings from frustum (same as GUI display)
                        camera_pos = camera_frustum.position
                        camera_fov_radians = camera_frustum.fov
                        render_wxyz = camera_frustum.wxyz

                        # Use same offset calculation as Live Camera View
                        look_at = np.array(look_at_position)
                        direction = look_at - camera_pos
                        direction = direction / np.linalg.norm(direction)
                        offset_distance = 0.01
                        offset_pos = camera_pos - direction * offset_distance

                        # Temporarily hide camera frustum from render
                        # camera_frustum.visible = False

                        try:
                            # Render high-resolution image (without seeing the frustum itself)
                            image = client.get_render(
                                height=render_height * 2,  # 2x resolution for screenshots
                                width=render_width * 2,
                                wxyz=render_wxyz,
                                position=offset_pos,  # Use offset position
                                fov=camera_fov_radians,
                            )
                        finally:
                            # Always restore camera frustum visibility for 3D scene
                            camera_frustum.visible = True

                            # Save image
                            filename = f"camera_view_{client.client_id}_{int(time.time())}.png"
                            imageio.imwrite(filename, image)

                        logger.info(f"Camera screenshot saved as {filename}")
                        camera_view_info.value = f"Screenshot saved: {filename}"

                    except Exception as e:
                        logger.error(f"Failed to take screenshot: {e}")
                        camera_view_info.value = f"Screenshot failed: {e}"

                def capture_recording_frame():
                    """Capture a frame for recording."""
                    nonlocal recording_frames
                    try:
                        # Get camera settings from frustum
                        camera_pos = camera_frustum.position
                        camera_fov_radians = camera_frustum.fov
                        render_wxyz = camera_frustum.wxyz

                        # Calculate offset position (same as screenshot)
                        look_at = np.array(look_at_position)
                        direction = look_at - camera_pos
                        direction = direction / np.linalg.norm(direction)
                        offset_distance = 0.01
                        offset_pos = camera_pos - direction * offset_distance

                        # Temporarily hide camera frustum from render
                        # camera_frustum.visible = False

                        try:
                            # Render frame for recording
                            image = client.get_render(
                                height=render_height,
                                width=render_width,
                                wxyz=render_wxyz,
                                position=offset_pos,
                                fov=camera_fov_radians,
                            )
                            recording_frames.append(image.copy())
                        finally:
                            # Always restore camera frustum visibility
                            camera_frustum.visible = True

                    except Exception as e:
                        logger.error(f"Failed to capture recording frame: {e}")

                def start_recording():
                    """Start recording camera view."""
                    nonlocal is_recording, recording_frames, recording_timer, recording_start_time

                    if is_recording:
                        return

                    is_recording = True
                    recording_frames = []
                    recording_start_time = time.time()

                    recording_status.value = "Recording..."
                    start_recording_btn.disabled = True
                    stop_recording_btn.disabled = False

                    # Start periodic frame capture
                    def capture_frames():
                        nonlocal recording_timer, recording_frames, recording_start_time
                        if is_recording:
                            capture_recording_frame()
                            frame_count = len(recording_frames)
                            elapsed_time = time.time() - recording_start_time
                            recording_status.value = f"Recording... {frame_count} frames ({elapsed_time:.1f}s)"

                            # Schedule next frame
                            recording_timer = threading.Timer(1.0 / recording_fps.value, capture_frames)
                            recording_timer.start()

                    capture_frames()
                    logger.info("Started camera recording")

                def stop_recording():
                    """Stop recording and save video."""
                    nonlocal is_recording, recording_timer, recording_frames

                    if not is_recording:
                        return

                    is_recording = False
                    if recording_timer:
                        recording_timer.cancel()
                        recording_timer = None

                    recording_status.value = "Saving video..."
                    start_recording_btn.disabled = False
                    stop_recording_btn.disabled = True

                    try:
                        import imageio

                        if len(recording_frames) > 0:
                            # Save video
                            filename = f"camera_recording_{client.client_id}_{int(time.time())}.mp4"
                            with imageio.get_writer(filename, fps=recording_fps.value) as writer:
                                for frame in recording_frames:
                                    writer.append_data(frame)

                            frame_count = len(recording_frames)
                            duration = frame_count / recording_fps.value
                            recording_status.value = f"Video saved: {filename} ({frame_count} frames, {duration:.1f}s)"
                            logger.info(f"Camera recording saved as {filename}")
                        else:
                            recording_status.value = "No frames recorded"
                            logger.warning("No frames were recorded")

                    except Exception as e:
                        logger.error(f"Failed to save recording: {e}")
                        recording_status.value = f"Save failed: {e}"
                    finally:
                        recording_frames = []

                # Store current camera rotation (starts as identity)
                current_camera_rotation = np.eye(3)

                def orthogonalize_matrix(R):
                    """Ensure rotation matrix remains orthogonal using Gram-Schmidt process."""
                    # Use SVD to get the closest orthogonal matrix
                    U, _, Vt = np.linalg.svd(R)
                    R_ortho = U @ Vt

                    # Ensure determinant is +1 (proper rotation, not reflection)
                    if np.linalg.det(R_ortho) < 0:
                        U[:, -1] *= -1
                        R_ortho = U @ Vt

                    return R_ortho

                def apply_incremental_rotation(axis, angle_degrees):
                    """Apply incremental rotation around specified axis."""
                    nonlocal current_camera_rotation

                    if abs(angle_degrees) < 1e-6:
                        return

                    angle_rad = np.radians(angle_degrees)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

                    # Create rotation matrix for the specified axis
                    if axis == "roll":
                        R_delta = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                    elif axis == "pitch":  # Around current X-axis
                        R_delta = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
                    else:
                        R_delta = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

                    # Apply incremental rotation to current state
                    current_camera_rotation = current_camera_rotation @ R_delta

                    # Ensure the rotation matrix remains orthogonal to prevent deformation
                    current_camera_rotation = orthogonalize_matrix(current_camera_rotation)

                    # Update camera with new rotation
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    # Update camera info display with new rotation
                    current_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    camera_info.value = f"Position: ({current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {fov_slider.value:.1f}°"

                    # Update camera view
                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                def update_camera():
                    """Update camera position and settings."""
                    nonlocal updating_sliders

                    # Prevent infinite loop when we update sliders programmatically
                    if updating_sliders:
                        return

                    new_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                    new_fov_degrees = fov_slider.value
                    new_fov_radians = np.radians(new_fov_degrees)

                    # Update camera frustum position and FOV
                    camera_frustum.position = new_position
                    camera_frustum.fov = new_fov_radians

                    # Extract Euler angles from current rotation matrix for display
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                    # Update camera info display
                    camera_info.value = f"Position: ({new_position[0]:.1f}, {new_position[1]:.1f}, {new_position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {new_fov_degrees:.1f}°"

                    # Update camera view
                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                def look_at_center():
                    """Make camera look at specified target from current position."""
                    nonlocal current_camera_rotation

                    current_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                    target = np.array(look_at_position)

                    # Calculate direction vector (from camera to target)
                    direction = target - current_position
                    distance = np.linalg.norm(direction)

                    # Check if we're too close to the center
                    if distance < 1e-6:
                        logger.warning("Camera is too close to the center, cannot look at itself")
                        return

                    # Normalize direction vector (this will be our +Z axis in camera space)
                    forward = direction / distance

                    # Choose world up vector (use Y-up convention, common in 3D graphics)
                    world_up = np.array([0.0, 0.0, 1.0])

                    # Calculate right vector (X axis in camera space)
                    right = np.cross(forward, world_up)
                    right_length = np.linalg.norm(right)

                    # Handle gimbal lock case (camera looking straight up or down)
                    if right_length < 1e-6:
                        # When looking straight up/down, use X-axis as reference
                        right = np.array([1.0, 0.0, 0.0])
                        if forward[1] < 0:  # looking down
                            right = np.array([-1.0, 0.0, 0.0])
                    else:
                        right = right / right_length

                    # Calculate up vector (Y axis in camera space) using cross product
                    # This ensures our coordinate system is orthogonal
                    up = np.cross(right, forward)
                    up = up / np.linalg.norm(up)  # Normalize to ensure orthogonality

                    # Construct rotation matrix (each column is an axis)
                    # OpenCV/Viser convention: +X right, +Y down, +Z forward
                    # So we need: [right, -up, forward] to match OpenCV convention
                    current_camera_rotation = np.column_stack([right, -up, forward])

                    # Ensure perfect orthogonality to prevent any deformation
                    current_camera_rotation = orthogonalize_matrix(current_camera_rotation)

                    # Verify orthogonality (debugging)
                    det = np.linalg.det(current_camera_rotation)
                    if abs(det - 1.0) > 1e-6:
                        logger.warning(f"Rotation matrix determinant is {det}, should be 1.0")

                    # Apply to camera frustum
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    # Update camera info display with new rotation
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    camera_info.value = f"Position: ({current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {fov_slider.value:.1f}°"

                    # Update camera view
                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                # Helper function for quaternion conversion
                def matrix_to_quaternion(R):
                    trace = R[0, 0] + R[1, 1] + R[2, 2]
                    if trace > 0:
                        s = np.sqrt(trace + 1.0) * 2
                        return np.array([
                            0.25 * s,
                            (R[2, 1] - R[1, 2]) / s,
                            (R[0, 2] - R[2, 0]) / s,
                            (R[1, 0] - R[0, 1]) / s,
                        ])
                    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                        return np.array([
                            (R[2, 1] - R[1, 2]) / s,
                            0.25 * s,
                            (R[0, 1] + R[1, 0]) / s,
                            (R[0, 2] + R[2, 0]) / s,
                        ])
                    elif R[1, 1] > R[2, 2]:
                        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                        return np.array([
                            (R[0, 2] - R[2, 0]) / s,
                            (R[0, 1] + R[1, 0]) / s,
                            0.25 * s,
                            (R[1, 2] + R[2, 1]) / s,
                        ])
                    else:
                        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                        return np.array([
                            (R[1, 0] - R[0, 1]) / s,
                            (R[0, 2] + R[2, 0]) / s,
                            (R[1, 2] + R[2, 1]) / s,
                            0.25 * s,
                        ])

                def reset_camera():
                    """Reset camera to initial position and orientation."""
                    nonlocal updating_sliders, current_camera_rotation
                    updating_sliders = True

                    # Reset position to initial values
                    pos_x.value = initial_position[0]
                    pos_y.value = initial_position[1]
                    pos_z.value = initial_position[2]

                    fov_slider.value = initial_fov

                    # Reset camera rotation to identity (default orientation)
                    current_camera_rotation = np.eye(3)
                    current_camera_rotation = orthogonalize_matrix(
                        current_camera_rotation
                    )  # Ensure perfect orthogonality
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    updating_sliders = False
                    update_camera()

                def set_preset_view(position, rotation_matrix=None):
                    """Set camera to preset position and orientation."""
                    nonlocal updating_sliders, current_camera_rotation
                    updating_sliders = True

                    # Set position
                    pos_x.value = position[0]
                    pos_y.value = position[1]
                    pos_z.value = position[2]

                    # Set rotation if provided
                    if rotation_matrix is not None:
                        current_camera_rotation = rotation_matrix.copy()
                        current_camera_rotation = orthogonalize_matrix(
                            current_camera_rotation
                        )  # Ensure perfect orthogonality

                    updating_sliders = False

                    # Update camera with new preset
                    camera_frustum.position = np.array(position)
                    camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    # Update camera info display
                    R = current_camera_rotation
                    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    pitch = np.degrees(np.arcsin(-R[2, 0]))
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    camera_info.value = f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})\nRotation: R={roll:.1f}° P={pitch:.1f}° Y={yaw:.1f}°\nFOV: {fov_slider.value:.1f}°"

                    import threading

                    threading.Thread(target=update_camera_view, daemon=True).start()

                # Connect all GUI controls
                pos_x.on_update(lambda _: update_camera())
                pos_y.on_update(lambda _: update_camera())
                pos_z.on_update(lambda _: update_camera())
                fov_slider.on_update(lambda _: update_camera())
                lookat_center_btn.on_click(lambda _: look_at_center())
                reset_btn.on_click(lambda _: reset_camera())
                screenshot_btn.on_click(lambda _: take_camera_screenshot(_))

                # Connect recording buttons
                start_recording_btn.on_click(lambda _: start_recording())
                stop_recording_btn.on_click(lambda _: stop_recording())

                # Set initial recording button states
                stop_recording_btn.disabled = True

                # Connect rotation buttons (10 degree increments)
                step_angle = 10.0
                yaw_left_btn.on_click(lambda _: apply_incremental_rotation("yaw", -step_angle))
                yaw_right_btn.on_click(lambda _: apply_incremental_rotation("yaw", step_angle))
                pitch_up_btn.on_click(lambda _: apply_incremental_rotation("pitch", step_angle))
                pitch_down_btn.on_click(lambda _: apply_incremental_rotation("pitch", -step_angle))
                roll_left_btn.on_click(lambda _: apply_incremental_rotation("roll", -step_angle))
                roll_right_btn.on_click(lambda _: apply_incremental_rotation("roll", step_angle))

                # Connect preset buttons with appropriate rotation matrices
                @top_view_btn.on_click
                def top_view(_):
                    # Top view: looking straight down (rotate -90° around X-axis)
                    R_top = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                    set_preset_view([0.0, 0.0, 3.0], R_top)

                @side_view_btn.on_click
                def side_view(_):
                    # Side view: looking from +X towards origin (rotate 90° around Y-axis)
                    R_side = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
                    set_preset_view([3.0, 0.0, 1.0], R_side)

                @front_view_btn.on_click
                def front_view(_):
                    # Front view: looking from -Y towards origin (default orientation)
                    R_front = np.eye(3)
                    set_preset_view([0.0, -3.0, 1.0], R_front)

                # Initial setup
                update_camera()

                # If look_at_position is not the default origin, set initial look-at
                if look_at_position != [0.0, 0.0, 0.0]:
                    # Modify look_at_center function to use custom target
                    def initial_look_at():
                        nonlocal current_camera_rotation
                        current_position = np.array([pos_x.value, pos_y.value, pos_z.value])
                        target = np.array(look_at_position)

                        direction = target - current_position
                        distance = np.linalg.norm(direction)

                        if distance > 1e-6:
                            forward = direction / distance
                            world_up = np.array([0.0, 1.0, 0.0])
                            right = np.cross(forward, world_up)
                            right_length = np.linalg.norm(right)

                            if right_length < 1e-6:
                                right = np.array([1.0, 0.0, 0.0])
                                if forward[1] < 0:
                                    right = np.array([-1.0, 0.0, 0.0])
                            else:
                                right = right / right_length

                            up = np.cross(right, forward)
                            up = up / np.linalg.norm(up)

                            current_camera_rotation = np.column_stack([right, -up, forward])
                            current_camera_rotation = orthogonalize_matrix(current_camera_rotation)
                            camera_frustum.wxyz = matrix_to_quaternion(current_camera_rotation)

                    initial_look_at()

                # Delay camera view update to ensure frustum is properly set
                def delayed_camera_view_update():
                    time.sleep(0.01)  # Small delay
                    update_camera_view()

                import threading

                threading.Thread(target=delayed_camera_view_update, daemon=True).start()

                logger.info(f"Camera controls and recording enabled for client {client.client_id}")

            except Exception as e:
                logger.error(f"Failed to setup camera controls: {e}")
                import traceback

                traceback.print_exc()

    def enable_trajectory_playback(self) -> None:
        """Enable trajectory playback controls in the GUI.

        Provides controls for loading, playing, pausing, and seeking through trajectories
        for all connected clients.
        """

        @self.server.on_client_connect
        def setup_trajectory_controls(client):
            """Setup trajectory playback controls for each connected client."""
            try:
                # Add trajectory playback controls
                with client.gui.add_folder("Trajectory Playback"):
                    trajectory_status = client.gui.add_text(
                        "Trajectory Status", initial_value="Use: visualizer.load_trajectory('path')"
                    )
                    trajectory_file_path = client.gui.add_text("Trajectory File", initial_value="No file loaded")
                    update_robot_btn = client.gui.add_button("Update Robot List")

                    # Robot and demo selection
                    robot_select = client.gui.add_dropdown(
                        "Robot", options=["No trajectories loaded"], initial_value="No trajectories loaded"
                    )
                    demo_select = client.gui.add_number("Demo Index", initial_value=0, min=0, step=1)
                    set_trajectory_btn = client.gui.add_button("Set Current Trajectory")

                    # Playback controls
                    play_btn = client.gui.add_button("Play")
                    pause_btn = client.gui.add_button("Pause")
                    stop_btn = client.gui.add_button("Stop")
                    playback_fps = client.gui.add_slider("Playback FPS", min=1, max=60, step=1, initial_value=10)

                    # Timeline control
                    timeline_slider = client.gui.add_slider("Timeline", min=0, max=100, step=1, initial_value=0)
                    frame_info = client.gui.add_text("Frame Info", initial_value="Frame: 0 / 0")

                # Trajectory playback state variables
                current_robot_name = None
                current_demo_index = 0
                updating_timeline = False  # Flag to prevent timeline update conflicts

                # Trajectory playback functions
                def update_robot_list():
                    """Update robot dropdown after loading trajectory."""
                    update_robot_dropdown()

                def update_robot_dropdown():
                    """Update robot dropdown with available robots from loaded trajectory."""
                    trajectories = self.get_available_trajectories()
                    if trajectories:
                        robot_options = [f"{robot_name} ({num_demos} demos)" for robot_name, num_demos in trajectories]
                        robot_names = [robot_name for robot_name, _ in trajectories]

                        # Clear and update dropdown options
                        robot_select.options = robot_options
                        if robot_names:
                            robot_select.value = robot_options[0]

                        # Update trajectory file path display
                        if hasattr(self, "_trajectory_file_path") and self._trajectory_file_path:
                            trajectory_file_path.value = self._trajectory_file_path

                        trajectory_status.value = f"Loaded {len(trajectories)} robot(s)"
                    else:
                        robot_select.options = ["No trajectories loaded"]
                        robot_select.value = "No trajectories loaded"
                        trajectory_file_path.value = "No trajectory file loaded"
                        trajectory_status.value = "No trajectories available"

                def set_current_trajectory_gui():
                    """Set current trajectory based on GUI selection."""
                    nonlocal current_robot_name, current_demo_index

                    if not robot_select.value or robot_select.value == "No trajectories loaded":
                        trajectory_status.value = "No robot selected or no trajectories loaded"
                        return

                    # Extract robot name from dropdown selection
                    robot_name = robot_select.value.split(" (")[0]
                    demo_index = int(demo_select.value)

                    if self.set_current_trajectory(robot_name, demo_index):
                        current_robot_name = robot_name
                        current_demo_index = demo_index

                        # Update timeline slider
                        if self._current_trajectory:
                            num_frames = len(self._current_trajectory.get("actions", []))
                            updating_timeline = True
                            timeline_slider.max = max(1, num_frames - 1)
                            timeline_slider.value = 0
                            frame_info.value = f"Frame: 0 / {num_frames}"
                            updating_timeline = False
                            trajectory_status.value = f"Set trajectory: {robot_name}, demo {demo_index}"
                        else:
                            trajectory_status.value = "Failed to set trajectory"
                    else:
                        trajectory_status.value = f"Failed to set trajectory: {robot_name}, demo {demo_index}"

                def play_trajectory_gui():
                    """Start or resume trajectory playback."""
                    if self._current_trajectory is None:
                        trajectory_status.value = "No trajectory set"
                        return

                    fps = playback_fps.value

                    # Check if we need to resume instead of start
                    if self._trajectory_playing and self._trajectory_paused:
                        # Resume paused trajectory
                        if self.resume_trajectory(fps):
                            play_btn.disabled = True
                            pause_btn.disabled = False
                            stop_btn.disabled = False
                            trajectory_status.value = f"Resumed at {fps} FPS"
                        else:
                            trajectory_status.value = "Failed to resume playback"
                    else:
                        # Start new playback
                        if self.play_trajectory(fps):
                            play_btn.disabled = True
                            pause_btn.disabled = False
                            stop_btn.disabled = False
                            trajectory_status.value = f"Playing at {fps} FPS"
                        else:
                            trajectory_status.value = "Failed to start playback"

                    # Start timeline update (for both start and resume)
                    if play_btn.disabled:  # Only if playback started successfully

                        def update_timeline():
                            nonlocal updating_timeline

                            if self._trajectory_playing and not self._trajectory_paused:
                                if self._current_trajectory:
                                    num_frames = len(self._current_trajectory.get("actions", []))

                                    # Set flag to prevent seek_timeline from interfering
                                    updating_timeline = True
                                    timeline_slider.value = self._current_frame
                                    frame_info.value = f"Frame: {self._current_frame} / {num_frames}"
                                    updating_timeline = False

                                # Schedule next update
                                threading.Timer(0.1, update_timeline).start()
                            else:
                                # Playback stopped or paused, reset flag
                                updating_timeline = False

                        update_timeline()

                def pause_trajectory_gui():
                    """Pause trajectory playback."""
                    if self._trajectory_playing and not self._trajectory_paused:
                        self.pause_trajectory()

                        # Re-enable play button so user can resume with play
                        play_btn.disabled = False
                        pause_btn.disabled = True
                        stop_btn.disabled = False

                        trajectory_status.value = "Paused - Click Play to resume"
                    elif self._trajectory_playing and self._trajectory_paused:
                        # This branch may not be needed now since we use play button to resume
                        fps = playback_fps.value
                        self.resume_trajectory(fps)

                        play_btn.disabled = True
                        pause_btn.disabled = False
                        stop_btn.disabled = False

                        trajectory_status.value = f"Resumed at {fps} FPS"

                def stop_trajectory_gui():
                    """Stop trajectory playback."""
                    nonlocal updating_timeline

                    self.stop_trajectory()
                    play_btn.disabled = False
                    pause_btn.disabled = True
                    pause_btn.value = "Pause"
                    stop_btn.disabled = True

                    # Reset timeline
                    updating_timeline = True
                    timeline_slider.value = 0
                    if self._current_trajectory:
                        num_frames = len(self._current_trajectory.get("actions", []))
                        frame_info.value = f"Frame: 0 / {num_frames}"
                    updating_timeline = False

                    trajectory_status.value = "Stopped"

                def seek_timeline():
                    """Seek to specific frame using timeline slider."""
                    nonlocal updating_timeline

                    # Skip if we're in the middle of updating timeline programmatically
                    if updating_timeline:
                        return

                    if self._current_trajectory is None:
                        return

                    frame_index = int(timeline_slider.value)
                    if self.seek_trajectory(frame_index):
                        num_frames = len(self._current_trajectory.get("actions", []))
                        frame_info.value = f"Frame: {frame_index} / {num_frames}"

                # Connect trajectory playback buttons
                update_robot_btn.on_click(lambda _: update_robot_list())
                set_trajectory_btn.on_click(lambda _: set_current_trajectory_gui())
                play_btn.on_click(lambda _: play_trajectory_gui())
                pause_btn.on_click(lambda _: pause_trajectory_gui())
                stop_btn.on_click(lambda _: stop_trajectory_gui())
                timeline_slider.on_update(lambda _: seek_timeline())

                # Set initial trajectory button states
                pause_btn.disabled = True
                stop_btn.disabled = True

                logger.info(f"Trajectory playback controls enabled for client {client.client_id}")

            except Exception as e:
                logger.error(f"Failed to setup trajectory controls: {e}")
                import traceback

                traceback.print_exc()

    def enable_joint_control(self) -> None:
        """Enable joint control GUI with sliders for each robot joint.

        Allows direct control of robot joint positions via interactive sliders
        for all connected clients.
        """

        @self.server.on_client_connect
        def setup_joint_controls(client):
            """Setup joint control for each connected client."""
            try:
                # Add joint control for robots
                with client.gui.add_folder("Joint Control"):
                    joint_status = client.gui.add_text("Joint Status", initial_value="No robot joints available")
                    robot_select_joint = client.gui.add_dropdown(
                        "Control Robot", options=["No robots loaded"], initial_value="No robots loaded"
                    )
                    setup_joint_control_btn = client.gui.add_button("Setup Joint Control")
                    clear_joint_control_btn = client.gui.add_button("Clear Joint Control")
                    reset_joints_btn = client.gui.add_button("Reset Joints")

                # Joint control state variables
                current_joint_robot = None

                # Joint control functions
                def update_robot_list_for_joints():
                    """Update robot dropdown for joint control."""
                    available_robots = []
                    for robot_name, urdf_handle in self._urdf_handles.items():
                        if urdf_handle is not None:
                            available_robots.append(robot_name)

                    if available_robots:
                        robot_select_joint.options = available_robots
                        robot_select_joint.value = available_robots[0]
                        joint_status.value = f"Found {len(available_robots)} robot(s) available for control"
                    else:
                        robot_select_joint.options = ["No robots loaded"]
                        robot_select_joint.value = "No robots loaded"
                        joint_status.value = "No robots available for joint control"

                def setup_joint_control_gui():
                    """Setup joint control sliders for selected robot."""
                    nonlocal current_joint_robot

                    if not robot_select_joint.value or robot_select_joint.value == "No robots loaded":
                        joint_status.value = "No robot selected"
                        return

                    selected_robot = robot_select_joint.value

                    # Check if joint control already exists for this robot
                    if selected_robot in self._joint_sliders and len(self._joint_sliders[selected_robot]) > 0:
                        joint_status.value = f"Joint control already exists for {selected_robot} ({len(self._joint_sliders[selected_robot])} joints)"
                        current_joint_robot = selected_robot
                        reset_joints_btn.disabled = False
                        return

                    # Create joint sliders for the selected robot (allow multiple robots)
                    try:
                        joint_folder = client.gui.add_folder(f"Joints - {selected_robot}")
                        self._joint_folders[selected_robot] = joint_folder  # Store folder reference

                        with joint_folder:
                            sliders = self.create_joint_sliders(selected_robot, client.gui)
                            if sliders:
                                current_joint_robot = selected_robot
                                joint_status.value = f"Setup {len(sliders)} joint controls for {selected_robot}"
                                logger.info(f"Created joint control panel for {selected_robot}")
                            else:
                                joint_status.value = f"Failed to setup joint control for {selected_robot}"
                                # Remove the empty folder if creation failed
                                if selected_robot in self._joint_folders:
                                    del self._joint_folders[selected_robot]
                    except Exception as e:
                        logger.error(f"Failed to create joint control folder: {e}")
                        joint_status.value = f"Error creating joint control: {e}"

                def clear_joint_control_gui():
                    """Clear all joint control panels but preserve joint positions."""
                    nonlocal current_joint_robot

                    cleared_robots = []
                    # Save current joint positions before clearing GUI
                    current_positions = {}
                    for robot_name in list(self._joint_folders.keys()):
                        if robot_name in self._joint_sliders:
                            # Save current slider values
                            current_positions[robot_name] = [slider.value for slider in self._joint_sliders[robot_name]]

                    # Try to remove all joint folders
                    for robot_name, folder in list(self._joint_folders.items()):
                        try:
                            if hasattr(folder, "remove"):
                                folder.remove()
                            cleared_robots.append(robot_name)
                        except Exception as e:
                            logger.warning(f"Could not remove joint folder for {robot_name}: {e}")

                    # Clear GUI references but preserve limits and configs for persistence
                    for robot_name in cleared_robots:
                        if robot_name in self._joint_folders:
                            del self._joint_folders[robot_name]
                        if robot_name in self._joint_sliders:
                            del self._joint_sliders[robot_name]
                        # Keep joint limits and initial configs for future setup
                        # Keep current position in robot's actual joints (they remain as-is)

                    current_joint_robot = None

                    if cleared_robots:
                        joint_status.value = (
                            f"Cleared joint control panels for: {', '.join(cleared_robots)} (positions preserved)"
                        )
                        logger.info(
                            f"Cleared GUI panels but preserved joint positions for: {', '.join(cleared_robots)}"
                        )
                    else:
                        joint_status.value = "No joint control panels to clear"

                def reset_joints_gui():
                    """Reset joints for currently selected robot only."""
                    if not robot_select_joint.value or robot_select_joint.value == "No robots loaded":
                        joint_status.value = "No robot selected for reset"
                        return

                    selected_robot = robot_select_joint.value

                    # Try to reset using joint sliders first
                    if selected_robot in self._joint_sliders:
                        self.reset_robot_joints(selected_robot)
                        joint_status.value = f"Reset joints for {selected_robot} to initial position"
                        logger.info(f"Reset joints for {selected_robot}")
                    else:
                        # If no joint control exists (cleared), try to reset robot directly
                        if self._urdf_handles.get(selected_robot):
                            try:
                                # Priority: Use demo initial config if available
                                if selected_robot in self._initial_configs:
                                    # Use stored initial config from demo
                                    initial_config = self._initial_configs[selected_robot]
                                    success = self.update_robot_joint_config_direct(selected_robot, initial_config)
                                    if success:
                                        # Clear saved positions
                                        if selected_robot in self._current_joint_positions:
                                            del self._current_joint_positions[selected_robot]
                                        joint_status.value = f"Reset {selected_robot} to demo initial pose"
                                        logger.info(f"Reset {selected_robot} to demo initial pose (no control panel)")
                                    else:
                                        joint_status.value = f"Failed to reset {selected_robot}"
                                else:
                                    # Fallback: compute default positions from joint limits
                                    joint_info = self.get_robot_joint_info(selected_robot)
                                    if joint_info:
                                        _, joint_limits, _ = joint_info
                                        # Compute default positions
                                        default_config = []
                                        for joint_name, (lower, upper) in joint_limits.items():
                                            lower = lower if lower is not None else -np.pi
                                            upper = upper if upper is not None else np.pi
                                            init_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
                                            default_config.append(init_pos)

                                        # Apply default configuration
                                        self._urdf_handles[selected_robot].update_cfg(np.array(default_config))

                                        # Clear saved positions
                                        if selected_robot in self._current_joint_positions:
                                            del self._current_joint_positions[selected_robot]

                                        joint_status.value = f"Reset {selected_robot} to computed default pose"
                                        logger.info(f"Reset {selected_robot} to computed default pose")
                                    else:
                                        joint_status.value = f"Cannot reset {selected_robot} - no joint info available"
                            except Exception as e:
                                logger.error(f"Failed to reset {selected_robot}: {e}")
                                joint_status.value = f"Failed to reset {selected_robot}: {e}"
                        else:
                            joint_status.value = f"Robot {selected_robot} not found or not loaded"

                # Connect joint control buttons
                setup_joint_control_btn.on_click(lambda _: setup_joint_control_gui())
                clear_joint_control_btn.on_click(lambda _: clear_joint_control_gui())
                reset_joints_btn.on_click(lambda _: reset_joints_gui())

                # Set initial joint control button states
                reset_joints_btn.disabled = False  # Always allow reset if robots are available

                # Auto-update robot list for joint control when GUI loads
                update_robot_list_for_joints()

                logger.info(f"Joint control enabled for client {client.client_id}")

            except Exception as e:
                logger.error(f"Failed to setup joint controls: {e}")
                import traceback

                traceback.print_exc()

    def setup_ik_solver(self, robot_name: str, robot_config, env_handler=None, solver="pyroki"):
        """Setup IK solver for a specific robot.

        Args:
            robot_name: Name of the robot
            robot_config: Robot configuration object
            env_handler: Environment handler for getting current states
            solver: IK solver backend ("curobo" or "pyroki")
        """
        try:
            from metasim.utils.ik_solver import IKSolver

            # Create unified IK solver
            ik_solver = IKSolver(robot_config, solver=solver, no_gnd=False)

            # Store IK solver and config
            self._robot_ik_solvers[robot_name] = ik_solver
            self._robot_configs[robot_name] = robot_config
            self._env_handler = env_handler

            # Initialize default target position (current end-effector position if possible)
            self._ik_target_positions[robot_name] = DEFAULT_IK_TARGET_POS.copy()
            self._ik_target_orientations[robot_name] = DEFAULT_IK_TARGET_QUAT.copy()

            logger.info(f"IK solver ({solver}) setup for robot {robot_name} with {ik_solver.n_dof_ik} DOF")
            return True

        except Exception as e:
            logger.error(f"Failed to setup IK solver for {robot_name}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def solve_ik_for_target(
        self,
        robot_name: str,
        target_pos: list[float],
        target_quat: list[float] | None = None,
    ):
        """Solve IK for target end-effector position and orientation.

        Args:
            robot_name: Name of the robot
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z] (optional)

        Returns:
            Joint configuration if successful, None otherwise
        """
        if robot_name not in self._robot_ik_solvers:
            logger.error(f"No IK solver found for robot {robot_name}")
            return None

        try:
            ik_solver = self._robot_ik_solvers[robot_name]
            robot_config = self._robot_configs[robot_name]

            # Use default orientation if not provided
            if target_quat is None:
                target_quat = self._ik_target_orientations[robot_name]

            # Convert to torch tensors with proper device handling
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            ee_pos_target = torch.tensor([target_pos], dtype=torch.float32, device=device)
            ee_quat_target = torch.tensor([target_quat], dtype=torch.float32, device=device)

            # Get current joint configuration as seed
            seed_q = None
            if self._env_handler is not None:
                try:
                    states = self._env_handler.get_states(mode="tensor")
                    if hasattr(states.robots[robot_name], "joint_pos"):
                        seed_q = states.robots[robot_name].joint_pos.to(device)
                except Exception as e:
                    logger.warning(f"Could not get current joint config for seed: {e}")

            # Create a default seed if none available
            if seed_q is None:
                seed_q = torch.zeros((1, ik_solver.n_robot_dof), device=device)

            # Solve IK using unified IKSolver
            q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, seed_q)

            if ik_succ[0]:
                # Compose full joint command with gripper
                gripper_widths = torch.tensor([[DEFAULT_GRIPPER_OPEN_WIDTH] * ik_solver.ee_n_dof], device=device)
                action_dicts = ik_solver.compose_joint_action(
                    q_solution, gripper_widths, current_q=seed_q, return_dict=True
                )

                # Extract joint config from action dict
                joint_config = action_dicts[0][robot_name]["dof_pos_target"]

                logger.info(f"IK solved successfully for {robot_name}")
                return joint_config
            else:
                logger.warning(f"IK failed to find solution for {robot_name}")
                return None

        except Exception as e:
            logger.error(f"Error solving IK for {robot_name}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def update_robot_from_ik(
        self,
        robot_name: str,
        target_pos: list[float],
        target_quat: list[float] | None = None,
    ):
        """Update robot configuration using IK solution.

        Args:
            robot_name: Name of the robot
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z] (optional)
        """
        joint_config = self.solve_ik_for_target(robot_name, target_pos, target_quat)
        if joint_config is not None:
            # Update robot using joint configuration
            if robot_name in self._urdf_handles:
                urdf_handle = self._urdf_handles[robot_name]
                joint_names = urdf_handle.get_actuated_joint_names()
                dof_pos_list = [joint_config.get(jn, 0.0) for jn in joint_names]
                urdf_handle.update_cfg(np.array(dof_pos_list, dtype=np.float32))

                # Update joint sliders if they exist
                if robot_name in self._joint_sliders:
                    for i, joint_name in enumerate(joint_names):
                        if i < len(self._joint_sliders[robot_name]) and joint_name in joint_config:
                            self._joint_sliders[robot_name][i].value = float(joint_config[joint_name])

                # Store current positions
                self._current_joint_positions[robot_name] = dof_pos_list

                # Refresh camera view
                self.refresh_camera_view()

                logger.debug(f"Updated robot {robot_name} with IK solution")
                return True
            else:
                logger.warning(f"Robot {robot_name} not found in URDF handles")
                return False
        return False

    def add_ik_target_marker(
        self,
        robot_name: str,
        position: list[float],
        orientation: list[float] | None = None,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ):
        """Add or update a visual marker for IK target position and orientation.

        Args:
            robot_name: Name of the robot
            position: Target position [x, y, z]
            orientation: Target orientation [w, x, y, z] (optional)
            color: RGB color (default: red)
        """
        try:
            marker_name = f"/ik_target_{robot_name}"
            frame_name = f"/ik_frame_{robot_name}"

            # Remove existing markers if they exist
            if robot_name in self._ik_target_markers:
                try:
                    self._ik_target_markers[robot_name].remove()
                except:
                    pass

            if robot_name in self._ik_orientation_frames:
                try:
                    self._ik_orientation_frames[robot_name].remove()
                except:
                    pass

            # Create new marker (a small sphere)
            marker = self.server.scene.add_icosphere(
                name=marker_name,
                radius=0.03,
                color=color,
                position=np.array(position, dtype=np.float32),
            )

            # Create orientation frame to show rotation
            if orientation is None:
                orientation = self._ik_target_orientations.get(robot_name, [0.0, 1.0, 0.0, 0.0])

            frame = self.server.scene.add_frame(
                name=frame_name,
                show_axes=True,
                axes_length=0.1,
                axes_radius=0.005,
            )
            frame.position = np.array(position, dtype=np.float32)
            frame.wxyz = np.array(orientation, dtype=np.float32)

            self._ik_target_markers[robot_name] = marker
            self._ik_orientation_frames[robot_name] = frame
            logger.debug(f"Added IK target marker and orientation frame for {robot_name} at {position}")

        except Exception as e:
            logger.error(f"Failed to add IK target marker: {e}")

    def update_ik_target_marker(
        self,
        robot_name: str,
        position: list[float],
        orientation: list[float] | None = None,
    ):
        """Update the position and orientation of IK target marker.

        Args:
            robot_name: Name of the robot
            position: New target position [x, y, z]
            orientation: New target orientation [w, x, y, z] (optional)
        """
        try:
            # Update position
            if robot_name in self._ik_target_markers:
                self._ik_target_markers[robot_name].position = np.array(position, dtype=np.float32)

                # Update orientation frame position
                if robot_name in self._ik_orientation_frames:
                    self._ik_orientation_frames[robot_name].position = np.array(position, dtype=np.float32)

                    # Update orientation if provided
                    if orientation is not None:
                        self._ik_orientation_frames[robot_name].wxyz = np.array(orientation, dtype=np.float32)

                logger.debug(f"Updated IK target marker for {robot_name} to {position}")
            else:
                # Create marker if it doesn't exist
                self.add_ik_target_marker(robot_name, position, orientation)
        except Exception as e:
            logger.error(f"Failed to update IK target marker: {e}")

    def remove_ik_target_marker(self, robot_name: str):
        """Remove IK target marker and orientation frame for a robot.

        Args:
            robot_name: Name of the robot
        """
        try:
            if robot_name in self._ik_target_markers:
                self._ik_target_markers[robot_name].remove()
                del self._ik_target_markers[robot_name]

            if robot_name in self._ik_orientation_frames:
                self._ik_orientation_frames[robot_name].remove()
                del self._ik_orientation_frames[robot_name]

            logger.debug(f"Removed IK target marker and orientation frame for {robot_name}")
        except Exception as e:
            logger.error(f"Failed to remove IK target marker: {e}")

    def enable_ik_control(self) -> None:
        """Enable IK control GUI for target-based robot control.

        Provides controls for setting target end-effector positions and orientations,
        with visual markers showing the target pose, for all connected clients.
        """

        @self.server.on_client_connect
        def setup_ik_controls(client):
            """Setup IK control for each connected client."""
            try:
                # Add IK control for robots
                with client.gui.add_folder("IK Control"):
                    ik_status = client.gui.add_text("IK Status", initial_value="No robot IK available")
                    robot_select_ik = client.gui.add_dropdown(
                        "Control Robot", options=["No robots loaded"], initial_value="No robots loaded"
                    )
                    setup_ik_control_btn = client.gui.add_button("Setup IK Control")
                    clear_ik_control_btn = client.gui.add_button("Clear IK Control")

                # IK control state variables
                current_ik_robot = None

                # IK control functions
                def update_robot_list_for_ik():
                    """Update robot dropdown for IK control."""
                    available_robots = []
                    for robot_name, robot_config in self._robot_configs.items():
                        if robot_name in self._urdf_handles and self._urdf_handles[robot_name] is not None:
                            available_robots.append(robot_name)

                    if available_robots:
                        robot_select_ik.options = available_robots
                        robot_select_ik.value = available_robots[0]
                        ik_status.value = f"Found {len(available_robots)} robot(s) available for IK control"
                    else:
                        robot_select_ik.options = ["No robots loaded"]
                        robot_select_ik.value = "No robots loaded"
                        ik_status.value = "No robots available for IK control"

                def setup_ik_control_gui():
                    """Setup IK control sliders for selected robot."""
                    nonlocal current_ik_robot

                    if not robot_select_ik.value or robot_select_ik.value == "No robots loaded":
                        ik_status.value = "No robot selected"
                        return

                    selected_robot = robot_select_ik.value

                    # Check if IK control already exists for this robot
                    if selected_robot in self._ik_sliders and len(self._ik_sliders[selected_robot]) > 0:
                        ik_status.value = f"IK control already exists for {selected_robot}"
                        current_ik_robot = selected_robot
                        return

                    # Check if robot has IK solver
                    if selected_robot not in self._robot_ik_solvers:
                        ik_status.value = f"No IK solver found for {selected_robot}. Please setup IK solver first."
                        return

                    # Create IK control sliders for the selected robot
                    try:
                        ik_folder = client.gui.add_folder(f"IK Control - {selected_robot}")
                        self._ik_folders[selected_robot] = ik_folder

                        # Add initial target marker with orientation
                        initial_target_pos = self._ik_target_positions[selected_robot]
                        initial_target_orient = self._ik_target_orientations[selected_robot]
                        self.add_ik_target_marker(
                            selected_robot, initial_target_pos, initial_target_orient, color=(1.0, 0.0, 0.0)
                        )

                        with ik_folder:
                            # Target position controls
                            with client.gui.add_folder("Target Position"):
                                target_pos = self._ik_target_positions[selected_robot]
                                pos_x_slider = client.gui.add_slider(
                                    "Target X", min=-1.0, max=1.0, step=0.01, initial_value=target_pos[0]
                                )
                                pos_y_slider = client.gui.add_slider(
                                    "Target Y", min=-1.0, max=1.0, step=0.01, initial_value=target_pos[1]
                                )
                                pos_z_slider = client.gui.add_slider(
                                    "Target Z", min=0.1, max=1.5, step=0.01, initial_value=target_pos[2]
                                )

                            # Target orientation controls
                            with client.gui.add_folder("Target Orientation"):
                                target_quat = self._ik_target_orientations[selected_robot]
                                quat_w_slider = client.gui.add_slider(
                                    "Quat W", min=-1.0, max=1.0, step=0.01, initial_value=target_quat[0]
                                )
                                quat_x_slider = client.gui.add_slider(
                                    "Quat X", min=-1.0, max=1.0, step=0.01, initial_value=target_quat[1]
                                )
                                quat_y_slider = client.gui.add_slider(
                                    "Quat Y", min=-1.0, max=1.0, step=0.01, initial_value=target_quat[2]
                                )
                                quat_z_slider = client.gui.add_slider(
                                    "Quat Z", min=-1.0, max=1.0, step=0.01, initial_value=target_quat[3]
                                )

                            # Control buttons
                            solve_ik_btn = client.gui.add_button("Solve & Apply IK")
                            reset_target_btn = client.gui.add_button("Reset Target")
                            reset_robot_btn = client.gui.add_button("Reset Robot Joints")

                            # Status display
                            ik_solve_status = client.gui.add_text("IK Solve Status", initial_value="Ready")

                            # Store slider references
                            self._ik_sliders[selected_robot] = {
                                "pos_x": pos_x_slider,
                                "pos_y": pos_y_slider,
                                "pos_z": pos_z_slider,
                                "quat_w": quat_w_slider,
                                "quat_x": quat_x_slider,
                                "quat_y": quat_y_slider,
                                "quat_z": quat_z_slider,
                                "status": ik_solve_status,
                            }

                            # Setup callbacks
                            def update_target_marker():
                                """Update target marker position and orientation when sliders change."""
                                target_pos = [pos_x_slider.value, pos_y_slider.value, pos_z_slider.value]
                                target_quat = [
                                    quat_w_slider.value,
                                    quat_x_slider.value,
                                    quat_y_slider.value,
                                    quat_z_slider.value,
                                ]

                                # Normalize quaternion
                                target_quat = normalize_quaternion(target_quat)
                                self.update_ik_target_marker(selected_robot, target_pos, target_quat)

                            def solve_and_apply_ik():
                                """Solve IK and apply to robot."""
                                try:
                                    target_pos = [pos_x_slider.value, pos_y_slider.value, pos_z_slider.value]
                                    target_quat = [
                                        quat_w_slider.value,
                                        quat_x_slider.value,
                                        quat_y_slider.value,
                                        quat_z_slider.value,
                                    ]

                                    # Normalize quaternion
                                    target_quat = normalize_quaternion(target_quat)

                                    ik_solve_status.value = "Solving IK..."
                                    logger.info(
                                        f"Starting IK solve for {selected_robot}, target: {target_pos}, quat: {target_quat}"
                                    )

                                    success = self.update_robot_from_ik(selected_robot, target_pos, target_quat)

                                    if success:
                                        ik_solve_status.value = f"IK solved! Target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
                                        # Update stored target
                                        self._ik_target_positions[selected_robot] = target_pos
                                        self._ik_target_orientations[selected_robot] = target_quat
                                        logger.info(f"IK solve succeeded for {selected_robot}")
                                    else:
                                        ik_solve_status.value = "IK failed to find solution"
                                        logger.warning(f"IK solve failed for {selected_robot}")

                                except Exception as e:
                                    ik_solve_status.value = f"Error: {e}"
                                    logger.error(f"Error in solve_and_apply_ik: {e}")
                                    import traceback

                                    traceback.print_exc()

                            def reset_target():
                                """Reset target to default values."""
                                default_pos = DEFAULT_IK_TARGET_POS.copy()
                                default_quat = DEFAULT_IK_TARGET_QUAT.copy()

                                pos_x_slider.value = default_pos[0]
                                pos_y_slider.value = default_pos[1]
                                pos_z_slider.value = default_pos[2]
                                quat_w_slider.value = default_quat[0]
                                quat_x_slider.value = default_quat[1]
                                quat_y_slider.value = default_quat[2]
                                quat_z_slider.value = default_quat[3]

                                # Update marker to default position and orientation
                                self.update_ik_target_marker(selected_robot, default_pos, default_quat)

                                ik_solve_status.value = "Target reset to default"

                            def reset_robot_joints():
                                """Reset robot joints to initial configuration."""
                                try:
                                    if selected_robot in self._initial_configs:
                                        initial_config = self._initial_configs[selected_robot]
                                        # Use direct update method that doesn't require sliders
                                        success = self.update_robot_joint_config_direct(selected_robot, initial_config)
                                        if success:
                                            ik_solve_status.value = "Robot joints reset to initial pose"
                                            logger.info(f"Reset robot {selected_robot} to initial joint configuration")
                                        else:
                                            ik_solve_status.value = "Failed to reset robot joints"
                                    else:
                                        ik_solve_status.value = "No initial config found for robot"
                                        logger.warning(f"No initial config stored for robot {selected_robot}")
                                except Exception as e:
                                    ik_solve_status.value = f"Error resetting robot: {e}"
                                    logger.error(f"Error in reset_robot_joints: {e}")

                            # Connect callbacks
                            solve_ik_btn.on_click(lambda _: solve_and_apply_ik())
                            reset_target_btn.on_click(lambda _: reset_target())
                            reset_robot_btn.on_click(lambda _: reset_robot_joints())

                            # Connect position and orientation sliders to update marker in real-time
                            pos_x_slider.on_update(lambda _: update_target_marker())
                            pos_y_slider.on_update(lambda _: update_target_marker())
                            pos_z_slider.on_update(lambda _: update_target_marker())
                            quat_w_slider.on_update(lambda _: update_target_marker())
                            quat_x_slider.on_update(lambda _: update_target_marker())
                            quat_y_slider.on_update(lambda _: update_target_marker())
                            quat_z_slider.on_update(lambda _: update_target_marker())

                        current_ik_robot = selected_robot
                        ik_status.value = f"Setup IK control for {selected_robot}"
                        logger.info(f"Created IK control panel for {selected_robot}")

                    except Exception as e:
                        logger.error(f"Failed to create IK control folder: {e}")
                        ik_status.value = f"Error creating IK control: {e}"

                def clear_ik_control_gui():
                    """Clear all IK control panels."""
                    nonlocal current_ik_robot

                    cleared_robots = []
                    # Try to remove all IK folders
                    for robot_name, folder in list(self._ik_folders.items()):
                        try:
                            if hasattr(folder, "remove"):
                                folder.remove()
                            cleared_robots.append(robot_name)
                        except Exception as e:
                            logger.warning(f"Could not remove IK folder for {robot_name}: {e}")

                    # Clear GUI references and target markers
                    for robot_name in cleared_robots:
                        if robot_name in self._ik_folders:
                            del self._ik_folders[robot_name]
                        if robot_name in self._ik_sliders:
                            del self._ik_sliders[robot_name]
                        # Remove target marker
                        self.remove_ik_target_marker(robot_name)

                    current_ik_robot = None

                    if cleared_robots:
                        ik_status.value = f"Cleared IK control panels for: {', '.join(cleared_robots)}"
                        logger.info(f"Cleared IK control panels for: {', '.join(cleared_robots)}")
                    else:
                        ik_status.value = "No IK control panels to clear"

                # Connect IK control buttons
                setup_ik_control_btn.on_click(lambda _: setup_ik_control_gui())
                clear_ik_control_btn.on_click(lambda _: clear_ik_control_gui())

                # Auto-update robot list for IK control when GUI loads
                update_robot_list_for_ik()

                logger.info(f"IK control enabled for client {client.client_id}")

            except Exception as e:
                logger.error(f"Failed to setup IK controls: {e}")
                import traceback

                traceback.print_exc()
