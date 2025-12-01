"""Rerun visualization utilities for RoboVerse demos.

This module provides a unified RerunVisualizer class for interactive 3D visualization
of robots, objects, and trajectories using the Rerun SDK.

Rerun is an open-source SDK for logging, storing, querying, and visualizing
multimodal data. It supports URDF, OBJ meshes, and various 3D primitives.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    rr = None

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

try:
    import yourdfpy

    YOURDFPY_AVAILABLE = True
except ImportError:
    YOURDFPY_AVAILABLE = False
    yourdfpy = None

from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import RobotCfg as BaseRobotCfg

# Configure logging - use loguru if available, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Constants
QUAT_NORMALIZE_EPSILON = 1e-6


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


def resolve_mesh_path(filename: str, urdf_dir: Path) -> str | None:
    """Resolve mesh path from URDF to absolute path.
    
    Args:
        filename: Mesh filename from URDF (may be relative or package://)
        urdf_dir: Directory containing the URDF file
        
    Returns:
        Absolute path to mesh file, or None if not found
    """
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    candidates = []
    
    # Handle package:// URIs - package root is usually the URDF directory
    if filename.startswith("package://"):
        rel_path = filename.split("package://", 1)[1]
        candidates = [
            urdf_dir / rel_path,  # Most common: package:// relative to URDF dir
            urdf_dir.parent / rel_path,
            urdf_dir.parent.parent / rel_path,
        ]
    else:
        # Build list of candidate paths for relative filenames
        base_name = Path(filename).name
        candidates = [
            urdf_dir / filename,
            urdf_dir / "meshes" / filename,
            urdf_dir / "meshes" / base_name,
            urdf_dir / "meshes" / "visual" / filename,
            urdf_dir / "meshes" / "visual" / base_name,
            urdf_dir / "meshes" / "collision" / filename,
            urdf_dir / "visual" / filename,
            urdf_dir / "visual" / base_name,
            urdf_dir / "collision" / filename,
            urdf_dir.parent / "meshes" / filename,
            urdf_dir.parent / "meshes" / base_name,
            urdf_dir.parent / "meshes" / "visual" / filename,
            urdf_dir.parent / "meshes" / "visual" / base_name,
        ]
        
        # Also try resolving relative paths like ../meshes/...
        try:
            resolved = (urdf_dir / filename).resolve()
            if resolved not in candidates:
                candidates.append(resolved)
        except Exception:
            pass
    
    for cand in candidates:
        try:
            resolved = cand.resolve() if not cand.is_absolute() else cand
            if resolved.exists():
                logger.debug(f"Resolved mesh: {filename} -> {resolved}")
                return str(resolved)
        except Exception:
            continue
    
    logger.warning(f"Could not resolve mesh path: {filename} from {urdf_dir}")
    logger.debug(f"Tried candidates: {[str(c) for c in candidates[:5]]}")
    return None


def parse_urdf_full(urdf_path: str) -> dict:
    """Parse URDF file extracting links, joints, and kinematic tree.
    
    Args:
        urdf_path: Path to URDF file
        
    Returns:
        Dictionary with:
        - 'links': dict of link_name -> list of visual geometries
        - 'joints': dict of joint_name -> joint info (parent, child, origin, type, axis)
        - 'parent_map': dict of child_link -> parent_link
        - 'root_link': name of the root link
    """
    urdf_dir = Path(urdf_path).parent
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links = {}
    joints = {}
    parent_map = {}  # child_link -> parent_link
    child_links = set()
    all_links = set()
    
    # Parse all links
    for link in root.findall(".//link"):
        link_name = link.get("name", "unnamed")
        all_links.add(link_name)
        visuals = []
        
        for visual in link.findall("visual"):
            visual_info = {"origin": np.eye(4)}
            
            # Parse origin
            origin = visual.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0").split()
                rpy = origin.get("rpy", "0 0 0").split()
                visual_info["origin"] = _make_transform(
                    [float(x) for x in xyz],
                    [float(r) for r in rpy]
                )
            
            # Parse geometry
            geometry = visual.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                box = geometry.find("box")
                sphere = geometry.find("sphere")
                cylinder = geometry.find("cylinder")
                
                if mesh is not None:
                    filename = mesh.get("filename", "")
                    scale = mesh.get("scale", "1 1 1").split()
                    resolved_path = resolve_mesh_path(filename, urdf_dir)
                    if resolved_path:
                        visual_info["type"] = "mesh"
                        visual_info["filename"] = resolved_path
                        visual_info["scale"] = [float(s) for s in scale]
                elif box is not None:
                    size = box.get("size", "1 1 1").split()
                    visual_info["type"] = "box"
                    visual_info["size"] = [float(s) for s in size]
                elif sphere is not None:
                    radius = float(sphere.get("radius", "1"))
                    visual_info["type"] = "sphere"
                    visual_info["radius"] = radius
                elif cylinder is not None:
                    radius = float(cylinder.get("radius", "1"))
                    length = float(cylinder.get("length", "1"))
                    visual_info["type"] = "cylinder"
                    visual_info["radius"] = radius
                    visual_info["length"] = length
            
            # Parse material color
            material = visual.find("material")
            if material is not None:
                color_elem = material.find("color")
                if color_elem is not None:
                    rgba = color_elem.get("rgba", "0.8 0.8 0.8 1").split()
                    visual_info["color"] = [float(c) for c in rgba]
                else:
                    visual_info["color"] = [0.8, 0.8, 0.8, 1.0]
            else:
                visual_info["color"] = [0.8, 0.8, 0.8, 1.0]
            
            if "type" in visual_info:
                visuals.append(visual_info)
        
        links[link_name] = visuals
    
    # Parse all joints
    for joint in root.findall(".//joint"):
        joint_name = joint.get("name", "unnamed")
        joint_type = joint.get("type", "fixed")
        
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        
        if parent_elem is None or child_elem is None:
            continue
            
        parent_link = parent_elem.get("link")
        child_link = child_elem.get("link")
        
        # Parse joint origin
        origin = joint.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            rpy = origin.get("rpy", "0 0 0").split()
            joint_origin = _make_transform(
                [float(x) for x in xyz],
                [float(r) for r in rpy]
            )
        else:
            joint_origin = np.eye(4)
        
        # Parse joint axis
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            axis = [float(x) for x in axis_elem.get("xyz", "0 0 1").split()]
        else:
            axis = [0, 0, 1]
        
        joints[joint_name] = {
            "parent": parent_link,
            "child": child_link,
            "origin": joint_origin,
            "type": joint_type,
            "axis": axis,
        }
        
        parent_map[child_link] = (parent_link, joint_name)
        child_links.add(child_link)
    
    # Find root link (links that are not children of any joint)
    root_links = all_links - child_links
    root_link = list(root_links)[0] if root_links else (list(all_links)[0] if all_links else None)
    
    return {
        "links": links,
        "joints": joints,
        "parent_map": parent_map,
        "root_link": root_link,
    }


def compute_link_transforms(urdf_data: dict, dof_pos: dict | None = None) -> dict:
    """Compute world transforms for each link using forward kinematics.
    
    Args:
        urdf_data: Output from parse_urdf_full()
        dof_pos: Optional dict mapping joint_name -> joint position (radians)
        
    Returns:
        Dictionary mapping link_name -> 4x4 world transform matrix
    """
    if dof_pos is None:
        dof_pos = {}
    
    links = urdf_data["links"]
    joints = urdf_data["joints"]
    parent_map = urdf_data["parent_map"]
    root_link = urdf_data["root_link"]
    
    link_transforms = {}
    
    def compute_link_transform(link_name: str) -> np.ndarray:
        """Recursively compute the world transform for a link."""
        if link_name in link_transforms:
            return link_transforms[link_name]
        
        if link_name == root_link or link_name not in parent_map:
            # Root link is at identity
            link_transforms[link_name] = np.eye(4)
            return link_transforms[link_name]
        
        parent_link, joint_name = parent_map[link_name]
        parent_transform = compute_link_transform(parent_link)
        
        joint_info = joints[joint_name]
        joint_origin = joint_info["origin"]
        joint_type = joint_info["type"]
        axis = joint_info["axis"]
        
        # Get joint position
        joint_pos = dof_pos.get(joint_name, 0.0)
        
        # Compute joint transform based on type
        if joint_type == "revolute" or joint_type == "continuous":
            # Rotation about axis
            joint_transform = _axis_angle_to_transform(axis, joint_pos)
        elif joint_type == "prismatic":
            # Translation along axis
            joint_transform = np.eye(4)
            joint_transform[:3, 3] = np.array(axis) * joint_pos
        else:
            # Fixed joint
            joint_transform = np.eye(4)
        
        # Link transform = parent * joint_origin * joint_rotation
        link_transforms[link_name] = parent_transform @ joint_origin @ joint_transform
        return link_transforms[link_name]
    
    # Compute transform for all links
    for link_name in links:
        compute_link_transform(link_name)
    
    return link_transforms


def _axis_angle_to_transform(axis: list, angle: float) -> np.ndarray:
    """Convert axis-angle to 4x4 transformation matrix (rotation only).
    
    Uses Rodrigues' rotation formula.
    """
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    T = np.eye(4)
    T[:3, :3] = R
    return T


def parse_urdf_simple(urdf_path: str) -> dict:
    """Simple URDF parser that extracts link visual geometries.
    
    This is a fallback when yourdfpy is not available.
    
    Args:
        urdf_path: Path to URDF file
        
    Returns:
        Dictionary with link names as keys and list of visual geometries as values
    """
    urdf_data = parse_urdf_full(urdf_path)
    return urdf_data["links"]


def _make_transform(xyz: list, rpy: list) -> np.ndarray:
    """Create 4x4 transformation matrix from xyz and rpy."""
    T = np.eye(4)
    T[:3, 3] = xyz
    
    # Roll, pitch, yaw to rotation matrix
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    T[0, 0] = cy * cp
    T[0, 1] = cy * sp * sr - sy * cr
    T[0, 2] = cy * sp * cr + sy * sr
    T[1, 0] = sy * cp
    T[1, 1] = sy * sp * sr + cy * cr
    T[1, 2] = sy * sp * cr - cy * sr
    T[2, 0] = -sp
    T[2, 1] = cp * sr
    T[2, 2] = cp * cr
    
    return T


class RerunVisualizer:
    """Interactive 3D visualizer for robots and objects using Rerun.

    This class provides visualization capabilities including:
    - Loading and displaying URDF models (robots and objects)
    - Primitive shape visualization (cubes, spheres, cylinders)
    - OBJ mesh visualization
    - Dynamic pose updates during simulation
    - Trajectory playback

    Args:
        app_name: Application name for Rerun (default: "RoboVerse")
        spawn: Whether to spawn the Rerun viewer automatically (default: True)
        connect: Whether to connect to an existing Rerun viewer (default: False)
        save_path: Optional path to save .rrd recording file
        use_normals: Whether to compute and include vertex normals for meshes (default: True)
    """

    def __init__(
        self,
        app_name: str = "RoboVerse",
        spawn: bool = True,
        connect: bool = False,
        save_path: str | None = None,
        use_normals: bool = True,
    ) -> None:
        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk is required. Install with: pip install rerun-sdk")

        self.app_name = app_name
        self.use_normals = use_normals
        self._urdf_models = {}  # name -> urdf object
        self._item_states = {}  # name -> current state dict
        self._item_configs = {}  # name -> config object
        self._initial_configs = {}  # name -> initial joint config
        self._joint_names_cache = {}  # name -> list of joint names
        self._time_step = 0

        # Initialize Rerun
        rr.init(app_name)

        if save_path:
            rr.save(save_path)
            logger.info(f"Recording to {save_path}")

        if connect:
            rr.connect()
            logger.info("Connected to existing Rerun viewer")
        elif spawn:
            rr.spawn()
            logger.info("Spawned Rerun viewer")

        # Set up 3D view with ground plane
        self._setup_scene()

    def _setup_scene(self) -> None:
        """Set up the initial 3D scene with coordinate system and ground plane."""
        # Log world coordinate frame
        rr.log(
            "world",
            rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )

        # Add ground plane as a grid of lines
        self._add_ground_grid()

    def _add_ground_grid(self, size: float = 5.0, divisions: int = 20) -> None:
        """Add a ground plane to the scene.

        Args:
            size: Half-size of the ground plane
            divisions: Number of divisions (unused, kept for API compatibility)
        """
        # Add solid ground plane only (no grid lines)
        if TRIMESH_AVAILABLE:
            try:
                # Create a flat box for the ground (very thin)
                ground = trimesh.creation.box(extents=[size * 2, size * 2, 0.01])
                # Move it slightly below z=0 so objects sit on top
                ground.vertices[:, 2] -= 0.005
                vertices = np.array(ground.vertices, dtype=np.float32)
                faces = np.array(ground.faces, dtype=np.uint32)
                ground_color = [220, 220, 220]  # Light gray
                rr.log(
                    "world/ground_plane",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=[ground_color] * len(vertices),
                    ),
                    static=True,
                )
            except Exception as e:
                logger.debug(f"Failed to create ground plane mesh: {e}")

    def add_frame(self, name: str) -> None:
        """Add a coordinate frame to the scene.

        Args:
            name: Name of the frame
        """
        # Log axes as arrows
        origin = [0, 0, 0]
        axis_length = 0.1

        rr.log(
            f"{name}/x_axis",
            rr.Arrows3D(
                origins=[origin],
                vectors=[[axis_length, 0, 0]],
                colors=[[255, 0, 0]],  # Red
            ),
            static=True,
        )
        rr.log(
            f"{name}/y_axis",
            rr.Arrows3D(
                origins=[origin],
                vectors=[[0, axis_length, 0]],
                colors=[[0, 255, 0]],  # Green
            ),
            static=True,
        )
        rr.log(
            f"{name}/z_axis",
            rr.Arrows3D(
                origins=[origin],
                vectors=[[0, 0, axis_length]],
                colors=[[0, 0, 255]],  # Blue
            ),
            static=True,
        )

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
            logger.warning(f"Unsupported items type {type(items)}")

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
    ) -> None:
        """Visualize a single item (robot or object) in the 3D scene.

        Args:
            cfg: Configuration object for the item
            name: Name identifier for the item
            state: Optional state dictionary containing position, rotation, and joint positions
        """
        logger.debug(f"[Rerun] Visualizing {name} with state: {state}")

        # Store config and state
        self._item_configs[name] = cfg
        if state:
            self._item_states[name] = state

        # Get position and rotation
        position = self._get_position(cfg, state)
        rotation = self._get_rotation(cfg, state)

        # Handle different item types
        if isinstance(cfg, (RigidObjCfg, ArticulationObjCfg, BaseRobotCfg)) and getattr(cfg, "urdf_path", None):
            self._visualize_urdf(cfg, name, position, rotation, state)
        elif isinstance(cfg, PrimitiveCubeCfg):
            self._visualize_cube(cfg, name, position, rotation)
        elif isinstance(cfg, PrimitiveSphereCfg):
            self._visualize_sphere(cfg, name, position, rotation)
        elif isinstance(cfg, PrimitiveCylinderCfg):
            self._visualize_cylinder(cfg, name, position, rotation)
        else:
            logger.warning(f"Unsupported object type {type(cfg)} for {name}")

    def _get_position(self, cfg, state: dict | None) -> tuple[float, float, float]:
        """Extract position from state or config defaults."""
        if state and "pos" in state and state["pos"] is not None:
            pos = state["pos"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
        return cfg.default_position

    def _get_rotation(self, cfg, state: dict | None) -> tuple[float, float, float, float]:
        """Extract rotation (wxyz quaternion) from state or config defaults."""
        if state and "rot" in state and state["rot"] is not None:
            rot = state["rot"]
            if isinstance(rot, (list, tuple)) and len(rot) >= 4:
                return (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        return cfg.default_orientation

    def _visualize_urdf(
        self,
        cfg,
        name: str,
        position: tuple,
        rotation: tuple,
        state: dict | None,
    ) -> None:
        """Visualize a URDF model with proper kinematic chain.

        Args:
            cfg: Configuration object with urdf_path
            name: Name identifier
            position: (x, y, z) position
            rotation: (w, x, y, z) quaternion
            state: Optional state with dof_pos
        """
        urdf_path = cfg.urdf_path
        
        # Try to resolve the URDF path
        urdf_path_resolved = Path(urdf_path)
        if not urdf_path_resolved.exists():
            # Try common base directories
            for base in [Path.cwd(), Path(__file__).parent.parent.parent.parent.parent]:
                candidate = base / urdf_path
                if candidate.exists():
                    urdf_path_resolved = candidate
                    break
        
        if not urdf_path_resolved.exists():
            logger.warning(f"URDF file not found: {urdf_path} (resolved: {urdf_path_resolved})")
            # Create a simple placeholder box for missing URDFs
            self._visualize_placeholder(name, position, rotation)
            return

        # Get scale - convert to tuple of floats
        scale = cfg.scale if hasattr(cfg, "scale") else (1.0, 1.0, 1.0)
        if isinstance(scale, (int, float)):
            scale = (float(scale), float(scale), float(scale))
        else:
            scale = tuple(float(s) for s in scale)

        try:
            # Parse URDF with full kinematic chain
            logger.info(f"Loading URDF for {name}: {urdf_path_resolved}")
            urdf_data = parse_urdf_full(str(urdf_path_resolved))
            link_visuals = urdf_data["links"]
            
            if not link_visuals:
                logger.warning(f"No visual geometries found in URDF: {urdf_path_resolved}")
                self._visualize_placeholder(name, position, rotation)
                return

            # Count total visuals for logging
            total_visuals = sum(len(v) for v in link_visuals.values())
            logger.info(f"Found {len(link_visuals)} links with {total_visuals} visual geometries")

            # Get joint positions from state
            dof_pos = {}
            if state and "dof_pos" in state and state["dof_pos"]:
                dof_pos = state["dof_pos"]
                # Convert tensor values to floats if needed
                for k, v in dof_pos.items():
                    if hasattr(v, "item"):
                        dof_pos[k] = v.item()
                    elif hasattr(v, "cpu"):
                        dof_pos[k] = v.cpu().numpy().item()
            
            # Compute forward kinematics to get link world transforms
            link_transforms = compute_link_transforms(urdf_data, dof_pos)

            # Build root transform matrix from position and rotation
            root_transform = np.eye(4)
            root_transform[:3, 3] = position
            # Convert quaternion (w,x,y,z) to rotation matrix
            root_transform[:3, :3] = self._quaternion_to_rotation_matrix(rotation)

            # Log the root entity with robot base transform
            rr.log(
                f"world/{name}",
                rr.Transform3D(
                    translation=position,
                    rotation=rr.Quaternion(xyzw=[rotation[1], rotation[2], rotation[3], rotation[0]]),
                ),
            )

            # Log each link's visual geometries with transforms relative to robot root
            # Since world/{name} already has the root transform, we only apply link_local_transform
            visuals_logged = 0
            for link_name, visuals in link_visuals.items():
                if not visuals:
                    continue
                    
                # Get link's local transform in URDF frame (relative to robot root)
                link_local_transform = link_transforms.get(link_name, np.eye(4))
                
                # Extract position and rotation for the link (relative to robot root, not world)
                link_pos = link_local_transform[:3, 3].tolist()
                link_rot = self._rotation_matrix_to_quaternion(link_local_transform[:3, :3])
                
                # Log link transform relative to robot root
                rr.log(
                    f"world/{name}/{link_name}",
                    rr.Transform3D(
                        translation=link_pos,
                        rotation=rr.Quaternion(xyzw=[link_rot[1], link_rot[2], link_rot[3], link_rot[0]]),
                    ),
                )
                
                for i, visual_info in enumerate(visuals):
                    self._log_visual_with_scale(
                        f"world/{name}/{link_name}/visual_{i}",
                        visual_info,
                        scale,
                    )
                    visuals_logged += 1

            logger.info(f"Successfully visualized URDF {name}: {visuals_logged} visuals logged")

            # Store URDF data for later updates
            self._urdf_models[name] = {
                "urdf_data": urdf_data,
                "scale": scale,
                "root_transform": root_transform,
            }

        except Exception as e:
            logger.error(f"Error visualizing URDF {name}: {e}")
            import traceback
            traceback.print_exc()
            # Create placeholder on error
            self._visualize_placeholder(name, position, rotation)

    def _quaternion_to_rotation_matrix(self, quat: tuple) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        w, x, y, z = quat
        
        # Normalize
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 1e-10:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return R

    def _visualize_placeholder(self, name: str, position: tuple, rotation: tuple) -> None:
        """Create a placeholder visualization for missing/invalid URDFs."""
        logger.info(f"Creating placeholder visualization for {name}")
        rr.log(
            f"world/{name}",
            rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=[rotation[1], rotation[2], rotation[3], rotation[0]]),
            ),
        )
        # Purple box as placeholder
        rr.log(
            f"world/{name}/placeholder",
            rr.Boxes3D(
                half_sizes=[[0.05, 0.05, 0.05]],
                colors=[[128, 0, 128]],  # Purple
            ),
        )

    def _log_visual_with_scale(
        self,
        entity_path: str,
        visual_info: dict,
        global_scale: tuple,
    ) -> None:
        """Log a visual geometry with global scale applied to vertices/sizes.
        
        Args:
            entity_path: Rerun entity path
            visual_info: Dictionary with type, origin, color, and geometry params
            global_scale: (sx, sy, sz) global scale to apply
        """
        origin = visual_info.get("origin", np.eye(4))
        # Don't scale the visual's local position - only mesh vertices are scaled
        pos = origin[:3, 3].tolist()
        rot_matrix = origin[:3, :3]
        quat = self._rotation_matrix_to_quaternion(rot_matrix)
        
        # Get color (RGBA 0-1 range)
        rgba = visual_info.get("color", [0.8, 0.8, 0.8, 1.0])
        color = [int(c * 255) for c in rgba[:3]]
        
        visual_type = visual_info.get("type")
        
        # Log transform for this visual
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]]),
            ),
        )
        
        if visual_type == "mesh":
            mesh_path = visual_info.get("filename")
            mesh_scale = visual_info.get("scale", [1, 1, 1])
            
            if not mesh_path:
                logger.debug(f"No mesh path for {entity_path}")
                return
                
            if not os.path.exists(mesh_path):
                logger.debug(f"Mesh file not found: {mesh_path}")
                rr.log(
                    entity_path,
                    rr.Boxes3D(
                        half_sizes=[[0.01, 0.01, 0.01]],
                        colors=[[255, 0, 255]],
                    ),
                )
                return
            
            ext = Path(mesh_path).suffix.lower()
            mesh_loaded = False
            
            if TRIMESH_AVAILABLE:
                try:
                    mesh = None
                    texture_info_data = None  # Store texture info for later
                    
                    # First, try to load with textures for OBJ files
                    if ext == ".obj":
                        try:
                            # Load without force="mesh" to get textures
                            mesh_with_tex = trimesh.load(mesh_path)
                            if isinstance(mesh_with_tex, trimesh.Scene):
                                # Extract texture info before concatenating
                                for geom_name, geom in mesh_with_tex.geometry.items():
                                    if hasattr(geom, 'visual') and geom.visual is not None:
                                        vis = geom.visual
                                        if hasattr(vis, 'uv') and vis.uv is not None and hasattr(vis, 'material'):
                                            mat = vis.material
                                            if hasattr(mat, 'image') and mat.image is not None:
                                                texture_info_data = {
                                                    'uv': np.array(vis.uv, dtype=np.float32),
                                                    'image': np.array(mat.image)
                                                }
                                                break
                                mesh = mesh_with_tex.dump(concatenate=True)
                            elif hasattr(mesh_with_tex, 'visual'):
                                vis = mesh_with_tex.visual
                                if hasattr(vis, 'uv') and vis.uv is not None and hasattr(vis, 'material'):
                                    mat = vis.material
                                    if hasattr(mat, 'image') and mat.image is not None:
                                        texture_info_data = {
                                            'uv': np.array(vis.uv, dtype=np.float32),
                                            'image': np.array(mat.image)
                                        }
                                mesh = mesh_with_tex
                        except Exception:
                            mesh = None
                    
                    # Fall back to force="mesh" for reliable geometry loading
                    if mesh is None or not hasattr(mesh, 'vertices'):
                        if ext == ".dae":
                            mesh = trimesh.load(mesh_path, force="mesh", resolver=None)
                        else:
                            mesh = trimesh.load(mesh_path, force="mesh")
                    
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    
                    if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                        vertices = np.array(mesh.vertices, dtype=np.float32)
                        
                        # Apply URDF mesh scale first
                        if mesh_scale != [1, 1, 1]:
                            vertices = vertices * np.array(mesh_scale, dtype=np.float32)
                        
                        # Apply global scale to vertices
                        vertices = vertices * np.array(global_scale, dtype=np.float32)
                        
                        faces = np.array(mesh.faces, dtype=np.uint32)
                        
                        if len(vertices) > 0 and len(faces) > 0:
                            mesh3d_kwargs = {
                                "vertex_positions": vertices,
                                "triangle_indices": faces,
                            }
                            
                            # Compute vertex normals for better lighting
                            if self.use_normals:
                                try:
                                    mesh.fix_normals()
                                    if hasattr(mesh, '_cache'):
                                        mesh._cache.clear()
                                    normals = mesh.vertex_normals.astype(np.float32)
                                    norms = np.linalg.norm(normals, axis=1, keepdims=True)
                                    norms[norms == 0] = 1.0
                                    normals = normals / norms
                                    mesh3d_kwargs["vertex_normals"] = normals
                                except Exception as e:
                                    logger.debug(f"Failed to compute normals for {mesh_path}: {e}")
                            
                            has_texture = False
                            
                            # Try to use pre-extracted texture info
                            if texture_info_data is not None:
                                uv = texture_info_data['uv']
                                texture_image = texture_info_data['image']
                                if len(uv) == len(vertices) and texture_image.size > 0:
                                    mesh3d_kwargs["vertex_texcoords"] = uv
                                    if len(texture_image.shape) == 2:
                                        texture_image = np.stack([texture_image] * 3, axis=-1)
                                    mesh3d_kwargs["albedo_texture"] = texture_image
                                    has_texture = True
                                    logger.debug(f"Loaded texture for {mesh_path}: {texture_image.shape}")
                            
                            # Fall back to vertex colors
                            if not has_texture:
                                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                                    vc = mesh.visual.vertex_colors
                                    if vc is not None and len(vc) == len(vertices):
                                        mesh3d_kwargs["vertex_colors"] = vc[:, :3].tolist()
                                    else:
                                        mesh3d_kwargs["vertex_colors"] = [color] * len(vertices)
                                else:
                                    mesh3d_kwargs["vertex_colors"] = [color] * len(vertices)
                            
                            rr.log(entity_path, rr.Mesh3D(**mesh3d_kwargs))
                            mesh_loaded = True
                            texture_info = " with texture" if has_texture else ""
                            normals_info = " with normals" if self.use_normals and "vertex_normals" in mesh3d_kwargs else ""
                            logger.debug(f"Loaded mesh: {mesh_path} ({len(vertices)} verts, scale={global_scale}){texture_info}{normals_info}")
                except Exception as e:
                    logger.warning(f"trimesh failed for {mesh_path}: {e}")
            
            if not mesh_loaded and ext in [".obj", ".glb", ".gltf", ".stl", ".dae"]:
                try:
                    # Asset3D can load textures automatically for supported formats
                    if global_scale != (1.0, 1.0, 1.0):
                        logger.warning(f"Cannot apply scale to Asset3D: {mesh_path}. Install trimesh for proper scaling.")
                    rr.log(
                        entity_path,
                        rr.Asset3D(path=mesh_path),
                    )
                    mesh_loaded = True
                    logger.debug(f"Loaded mesh as Asset3D: {mesh_path}")
                except Exception as e:
                    logger.debug(f"Asset3D failed for {mesh_path}: {e}")
            
            if not mesh_loaded:
                logger.warning(f"Could not load mesh: {mesh_path} (format: {ext})")
                rr.log(
                    entity_path,
                    rr.Boxes3D(
                        half_sizes=[[0.02, 0.02, 0.02]],
                        colors=[[255, 128, 0]],
                    ),
                )
            return
        
        elif visual_type == "box":
            size = visual_info.get("size", [1, 1, 1])
            # Apply global scale to box size
            scaled_size = [s * gs for s, gs in zip(size, global_scale)]
            
            # Use Mesh3D for solid rendering
            if TRIMESH_AVAILABLE:
                try:
                    box = trimesh.creation.box(extents=scaled_size)
                    vertices = np.array(box.vertices, dtype=np.float32)
                    faces = np.array(box.faces, dtype=np.uint32)
                    rr.log(
                        entity_path,
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            vertex_colors=[color] * len(vertices),
                        ),
                    )
                    return
                except Exception:
                    pass
            
            # Fallback
            half_sizes = [s / 2 for s in scaled_size]
            rr.log(
                entity_path,
                rr.Boxes3D(
                    half_sizes=[half_sizes],
                    colors=[color],
                ),
            )
        
        elif visual_type == "sphere":
            radius = visual_info.get("radius", 1.0)
            # Apply average scale to sphere radius
            avg_scale = sum(global_scale) / 3.0
            scaled_radius = radius * avg_scale
            
            # Use Mesh3D for solid rendering
            if TRIMESH_AVAILABLE:
                try:
                    sphere = trimesh.creation.icosphere(subdivisions=3, radius=scaled_radius)
                    vertices = np.array(sphere.vertices, dtype=np.float32)
                    faces = np.array(sphere.faces, dtype=np.uint32)
                    rr.log(
                        entity_path,
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            vertex_colors=[color] * len(vertices),
                        ),
                    )
                    return
                except Exception:
                    pass
            
            # Fallback
            rr.log(
                entity_path,
                rr.Ellipsoids3D(
                    half_sizes=[[scaled_radius, scaled_radius, scaled_radius]],
                    colors=[color],
                ),
            )
        
        elif visual_type == "cylinder":
            radius = visual_info.get("radius", 1.0)
            length = visual_info.get("length", 1.0)
            
            # Apply scale (xy for radius, z for length)
            scaled_radius = radius * (global_scale[0] + global_scale[1]) / 2.0
            scaled_length = length * global_scale[2]
            
            if TRIMESH_AVAILABLE:
                try:
                    cylinder = trimesh.creation.cylinder(radius=scaled_radius, height=scaled_length)
                    vertices = np.array(cylinder.vertices)
                    faces = np.array(cylinder.faces)
                    rr.log(
                        entity_path,
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            vertex_colors=[color] * len(vertices),
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to create cylinder mesh: {e}")
            else:
                rr.log(
                    entity_path,
                    rr.Ellipsoids3D(
                        half_sizes=[[scaled_radius, scaled_radius, scaled_length / 2]],
                        colors=[color],
                    ),
                )

    def _log_visual_simple(
        self,
        entity_path: str,
        visual_info: dict,
    ) -> None:
        """Log a visual geometry using the simple URDF parser output (no global scale).
        
        Args:
            entity_path: Rerun entity path
            visual_info: Dictionary with type, origin, color, and geometry params
        """
        # Delegate to scaled version with identity scale
        self._log_visual_with_scale(entity_path, visual_info, (1.0, 1.0, 1.0))

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> tuple:
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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

        # Normalize
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        return (w / norm, x / norm, y / norm, z / norm)

    def _visualize_cube(
        self,
        cfg: PrimitiveCubeCfg,
        name: str,
        position: tuple,
        rotation: tuple,
    ) -> None:
        """Visualize a cube primitive."""
        # cfg.size is the full edge length
        size = list(cfg.size)

        color = [200, 200, 200]
        if hasattr(cfg, "color") and cfg.color:
            color = [int(c * 255) if c <= 1.0 else int(c) for c in cfg.color[:3]]

        # Log transform to parent entity
        rr.log(
            f"world/{name}",
            rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=[rotation[1], rotation[2], rotation[3], rotation[0]]),
            ),
        )
        
        # Use Mesh3D for solid surface rendering instead of Boxes3D (which renders as wireframe)
        # Log geometry to child entity so it inherits parent transform
        if TRIMESH_AVAILABLE:
            try:
                # Create box mesh with trimesh - extents is the full size
                box = trimesh.creation.box(extents=size)
                vertices = np.array(box.vertices, dtype=np.float32)
                faces = np.array(box.faces, dtype=np.uint32)
                rr.log(
                    f"world/{name}/visual",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=[color] * len(vertices),
                    ),
                )
                return
            except Exception as e:
                logger.debug(f"Failed to create cube mesh: {e}")
        
        # Fallback to Boxes3D - uses half_size
        half_size = [s / 2 for s in size]
        rr.log(
            f"world/{name}/visual",
            rr.Boxes3D(
                half_sizes=[half_size],
                colors=[color],
                fill_mode=rr.components.FillMode.Solid,
            ),
        )

    def _visualize_sphere(
        self,
        cfg: PrimitiveSphereCfg,
        name: str,
        position: tuple,
        rotation: tuple,
    ) -> None:
        """Visualize a sphere primitive."""
        # cfg.radius is the radius (half the diameter)
        radius = cfg.radius

        color = [200, 200, 200]
        if hasattr(cfg, "color") and cfg.color:
            color = [int(c * 255) if c <= 1.0 else int(c) for c in cfg.color[:3]]

        # Log transform to parent entity
        rr.log(
            f"world/{name}",
            rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=[rotation[1], rotation[2], rotation[3], rotation[0]]),
            ),
        )
        
        # Use Mesh3D for solid surface rendering instead of Ellipsoids3D (which renders as wireframe)
        # Log geometry to child entity so it inherits parent transform
        if TRIMESH_AVAILABLE:
            try:
                # Create sphere mesh with trimesh (subdivisions=3 gives good quality)
                sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
                vertices = np.array(sphere.vertices, dtype=np.float32)
                faces = np.array(sphere.faces, dtype=np.uint32)
                rr.log(
                    f"world/{name}/visual",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=faces,
                        vertex_colors=[color] * len(vertices),
                    ),
                )
                return
            except Exception as e:
                logger.debug(f"Failed to create sphere mesh: {e}")
        
        # Fallback to Ellipsoids3D
        rr.log(
            f"world/{name}/visual",
            rr.Ellipsoids3D(
                half_sizes=[[radius, radius, radius]],
                colors=[color],
                fill_mode=rr.components.FillMode.Solid,
            ),
        )

    def _visualize_cylinder(
        self,
        cfg: PrimitiveCylinderCfg,
        name: str,
        position: tuple,
        rotation: tuple,
    ) -> None:
        """Visualize a cylinder primitive."""
        radius = cfg.radius
        height = cfg.height

        color = [200, 200, 200]
        if hasattr(cfg, "color") and cfg.color:
            color = [int(c * 255) if c <= 1.0 else int(c) for c in cfg.color[:3]]

        # Log transform to parent entity
        rr.log(
            f"world/{name}",
            rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=[rotation[1], rotation[2], rotation[3], rotation[0]]),
            ),
        )

        # Log geometry to child entity so it inherits parent transform
        if TRIMESH_AVAILABLE:
            # Create cylinder mesh
            cylinder = trimesh.creation.cylinder(radius=radius, height=height)
            vertices = np.array(cylinder.vertices)
            faces = np.array(cylinder.faces)

            rr.log(
                f"world/{name}/visual",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_colors=[color] * len(vertices),
                ),
            )
        else:
            # Fallback: use ellipsoid approximation
            rr.log(
                f"world/{name}/visual",
                rr.Ellipsoids3D(
                    half_sizes=[[radius, radius, height / 2]],
                    colors=[color],
                ),
            )

    def update_item_pose(self, name: str, state: dict) -> None:
        """Update the pose of a visualized item.

        Args:
            name: Name of the item to update
            state: State dict containing 'pos', 'rot', and optionally 'dof_pos'
        """
        self._item_states[name] = state

        position = None
        rotation = None

        if "pos" in state and state["pos"] is not None:
            pos = state["pos"]
            if hasattr(pos, "cpu"):
                pos = pos.cpu().numpy()
            position = list(pos)

        if "rot" in state and state["rot"] is not None:
            rot = state["rot"]
            if hasattr(rot, "cpu"):
                rot = rot.cpu().numpy()
            rotation = list(rot)

        # Update base transform
        if position is not None or rotation is not None:
            transform_args = {}
            if position is not None:
                transform_args["translation"] = position
            if rotation is not None:
                # rotation is [w, x, y, z], rerun wants xyzw
                transform_args["rotation"] = rr.Quaternion(
                    xyzw=[rotation[1], rotation[2], rotation[3], rotation[0]]
                )

            rr.log(f"world/{name}", rr.Transform3D(**transform_args))

        # Update articulated joints if we have URDF data and dof_pos
        if name in self._urdf_models and "dof_pos" in state and state["dof_pos"]:
            self._update_urdf_joints(name, state)

    def _update_urdf_joints(self, name: str, state: dict) -> None:
        """Update joint positions for an articulated URDF model.
        
        Args:
            name: Name of the item
            state: State dict with 'pos', 'rot', 'dof_pos'
        """
        urdf_info = self._urdf_models.get(name)
        if not urdf_info:
            return
        
        urdf_data = urdf_info["urdf_data"]
        
        # Get joint positions
        dof_pos = state.get("dof_pos", {})
        if not dof_pos:
            return
        
        # Convert tensor values to floats
        dof_pos_float = {}
        for k, v in dof_pos.items():
            if hasattr(v, "item"):
                dof_pos_float[k] = v.item()
            elif hasattr(v, "cpu"):
                dof_pos_float[k] = v.cpu().numpy().item()
            else:
                dof_pos_float[k] = float(v)
        
        # Compute new link transforms using forward kinematics
        link_transforms = compute_link_transforms(urdf_data, dof_pos_float)
        
        # Update each link's transform (relative to robot root, since world/{name} has root transform)
        for link_name in urdf_data["links"]:
            link_local_transform = link_transforms.get(link_name, np.eye(4))
            
            # Use local transform relative to robot root (not world transform)
            link_pos = link_local_transform[:3, 3].tolist()
            link_rot = self._rotation_matrix_to_quaternion(link_local_transform[:3, :3])
            
            rr.log(
                f"world/{name}/{link_name}",
                rr.Transform3D(
                    translation=link_pos,
                    rotation=rr.Quaternion(xyzw=[link_rot[1], link_rot[2], link_rot[3], link_rot[0]]),
                ),
            )


    def set_time(self, time_step: int) -> None:
        """Set the current time step for timeline-based visualization.

        Args:
            time_step: Current simulation step
        """
        self._time_step = time_step
        rr.set_time_sequence("step", time_step)

    def log_trajectory_point(
        self,
        name: str,
        position: list | np.ndarray,
        color: list | None = None,
    ) -> None:
        """Log a single trajectory point.

        Args:
            name: Name of the trajectory entity
            position: 3D position [x, y, z]
            color: Optional RGB color [r, g, b]
        """
        if color is None:
            color = [255, 165, 0]  # Orange

        rr.log(
            f"world/trajectories/{name}",
            rr.Points3D(
                positions=[position],
                colors=[color],
                radii=[0.01],
            ),
        )

    def log_trajectory(
        self,
        name: str,
        positions: list | np.ndarray,
        color: list | None = None,
    ) -> None:
        """Log a complete trajectory as a line strip.

        Args:
            name: Name of the trajectory entity
            positions: List of 3D positions [[x, y, z], ...]
            color: Optional RGB color [r, g, b]
        """
        if color is None:
            color = [255, 165, 0]  # Orange

        rr.log(
            f"world/trajectories/{name}",
            rr.LineStrips3D(
                [positions],
                colors=[color],
            ),
            static=True,
        )

    def log_camera_image(
        self,
        camera_name: str,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
    ) -> None:
        """Log camera images.

        Args:
            camera_name: Name of the camera
            rgb: RGB image array (H, W, 3)
            depth: Optional depth image array (H, W)
        """
        rr.log(f"cameras/{camera_name}/rgb", rr.Image(rgb))

        if depth is not None:
            rr.log(f"cameras/{camera_name}/depth", rr.DepthImage(depth))

    def clear(self) -> None:
        """Clear all logged data."""
        # Clear internal caches
        self._urdf_models.clear()
        self._item_states.clear()
        self._item_configs.clear()
        self._initial_configs.clear()
        self._joint_names_cache.clear()
        self._time_step = 0

    def close(self) -> None:
        """Close the visualizer and clean up resources."""
        self.clear()
        logger.info("Rerun visualizer closed")

