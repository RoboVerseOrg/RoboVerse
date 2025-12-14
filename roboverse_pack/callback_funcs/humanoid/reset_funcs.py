from __future__ import annotations

import numpy as np
import torch

from metasim.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform
from roboverse_pack.tasks.humanoid.base.types import EnvTypes


def random_root_state(
    env: EnvTypes,
    env_ids: torch.Tensor | list,
    pose_range: list[list] | None = None,
    velocity_range: list[list] | None = None,
) -> torch.Tensor:
    """Randomize root pose and velocity for selected environments."""
    if len(env_ids) == 0:
        return

    root_states = env.default_env_states.robots[env.name].root_state[env_ids].clone()
    pose_range = pose_range or [[0] * 6, [0] * 6]
    velocity_range = velocity_range or [[0] * 6, [0] * 6]

    # poses
    pose_range = torch.tensor(pose_range, device=env.device)
    rand_samples = sample_uniform(pose_range[0], pose_range[1], (len(env_ids), 6), device=env.device)
    positions = root_states[:, 0:3] + rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    velocity_range = torch.tensor(velocity_range, device=env.device)
    rand_samples = sample_uniform(velocity_range[0], velocity_range[1], (len(env_ids), 6), device=env.device)

    velocities = root_states[:, 7:13] + rand_samples

    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 0:3] = positions
    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 3:7] = orientations
    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 7:13] = velocities
    # # set into the physics simulation
    # env.write_robot_root_state(torch.cat([positions, orientations, velocities], dim=-1), env_ids=env_ids)


def reset_joints_by_scale(
    env: EnvTypes,
    env_ids: torch.Tensor | list,
    position_range: list | tuple = (1.0, 1.0),
    velocity_range: list | tuple = (1.0, 1.0),
) -> torch.Tensor:
    """Scale default joint states by random factors for a batch of environments."""
    if len(env_ids) == 0:
        return

    # get default joint state
    joint_pos = env.default_env_states.robots[env.name].joint_pos[env_ids].clone()
    joint_vel = env.default_env_states.robots[env.name].joint_vel[env_ids].clone()

    # scale these values randomly
    joint_pos *= sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = env.soft_dof_pos_limits
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = env.soft_dof_vel_limits
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    env.setup_initial_env_states.robots[env.name].joint_pos[env_ids] = joint_pos
    env.setup_initial_env_states.robots[env.name].joint_vel[env_ids] = joint_vel
    # # set into the physics simulation
    # asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def get_terrain_height_at_position(env: EnvTypes, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Query the terrain height at given (x, y) positions.

    All simulators now center the terrain at world origin (0,0).
    The terrain center is at heightfield ((rows-1)/2, (cols-1)/2) which maps to world (0,0).

    Args:
        env: The environment
        x: X coordinates in world frame (shape: [num_envs])
        y: Y coordinates in world frame (shape: [num_envs])

    Returns:
        Height values at those positions (shape: [num_envs])
    """
    # Access the terrain data from the simulator
    handler = env.handler
    if not hasattr(handler, "_height_mat") or handler._height_mat is None:
        # No terrain, return zero height
        return torch.zeros_like(x)

    height_mat = handler._height_mat  # numpy array
    vertices = handler._ground_mesh_vertices
    width = vertices[:, 0].max() - vertices[:, 0].min()
    height = vertices[:, 1].max() - vertices[:, 1].min()

    # Convert world coordinates to heightfield indices
    # Terrain is centered: world (0,0) = heightfield center
    # heightfield (i,j) → world ((i - (rows-1)/2) * h_scale, (j - (cols-1)/2) * h_scale)
    # Inverse: world (x,y) → heightfield (x/h_scale + (rows-1)/2, y/h_scale + (cols-1)/2)
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    rows, cols = height_mat.shape
    half_width = (rows - 1) / 2.0
    half_height = (cols - 1) / 2.0

    x_idx = x_np / width * (rows - 1) + half_width
    y_idx = y_np / height * (cols - 1) + half_height

    # Clamp to valid range
    x_idx = np.clip(x_idx, 0, rows - 1)
    y_idx = np.clip(y_idx, 0, cols - 1)

    # Use bilinear interpolation for smooth height queries
    from scipy.interpolate import RegularGridInterpolator

    x_coords = np.arange(rows)
    y_coords = np.arange(cols)
    interpolator = RegularGridInterpolator(
        (x_coords, y_coords), height_mat, method="linear", bounds_error=False, fill_value=0.0
    )

    points = np.column_stack([x_idx, y_idx])
    heights = interpolator(points)

    return torch.from_numpy(heights).to(x.device, dtype=x.dtype)


def random_root_state_terrain_aware(
    env: EnvTypes,
    env_ids: torch.Tensor | list,
    pose_range: list[list] | None = None,
    velocity_range: list[list] | None = None,
    base_height_offset: float | None = None,
) -> torch.Tensor:
    """Reset robot root state with terrain-aware height adjustment.

    The robot will be spawned at base_height_offset above the terrain surface
    at the randomly sampled (x, y) position.

    Args:
        env: The environment
        env_ids: Environment IDs to reset
        pose_range: Range for random pose sampling [min, max] for [x, y, z, roll, pitch, yaw]
                   Note: z here represents additional offset on top of terrain + base_height_offset
        velocity_range: Range for random velocity sampling
        base_height_offset: Height above terrain to spawn the robot. If None, uses the robot's
                           default z-position from the configuration.
    """
    if len(env_ids) == 0:
        return

    root_states = env.default_env_states.robots[env.name].root_state[env_ids].clone()
    pose_range = pose_range or [[0] * 6, [0] * 6]
    velocity_range = velocity_range or [[0] * 6, [0] * 6]

    # Get per-env world-frame origins (if available) so we can convert
    # local robot positions to world coordinates for terrain queries.
    env_ids_idx = (
        env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    )
    env_origins_xy = torch.zeros((len(env_ids_idx), 2), device=env.device, dtype=root_states.dtype)
    if hasattr(env.handler, "scene") and hasattr(env.handler.scene, "env_origins"):
        # IsaacSim exposes per-env origins via scene.env_origins
        env_origins_xy = env.handler.scene.env_origins[env_ids_idx, :2]
    elif hasattr(env.handler, "_env_origin") and len(getattr(env.handler, "_env_origin", [])) > 0:
        # IsaacGym: env origins queried from gym.get_env_origin and cached on the handler
        env_origins = torch.as_tensor(env.handler._env_origin, device=env.device, dtype=root_states.dtype)
        env_origins_xy = env_origins[env_ids_idx, :2]

    # If base_height_offset not provided, use the robot's default z position
    if base_height_offset is None:
        base_height_offset = root_states[0, 2].item()

    # Sample random poses
    pose_range = torch.tensor(pose_range, device=env.device)
    rand_samples = sample_uniform(pose_range[0], pose_range[1], (len(env_ids), 6), device=env.device)

    # Calculate local x, y positions (env frame)
    x_positions = root_states[:, 0] + rand_samples[:, 0]
    y_positions = root_states[:, 1] + rand_samples[:, 1]

    # Convert to world-frame coordinates for terrain height query
    x_world = x_positions + env_origins_xy[:, 0]
    y_world = y_positions + env_origins_xy[:, 1]

    # Query terrain height at these x, y positions
    terrain_heights = get_terrain_height_at_position(env, x_world, y_world)

    # Set z position = terrain height + base offset + random z offset
    z_positions = terrain_heights + base_height_offset

    positions = torch.stack([x_positions, y_positions, z_positions], dim=1)

    # Handle orientations
    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # Handle velocities
    velocity_range = torch.tensor(velocity_range, device=env.device)
    rand_samples = sample_uniform(velocity_range[0], velocity_range[1], (len(env_ids), 6), device=env.device)
    velocities = root_states[:, 7:13] + rand_samples

    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 0:3] = positions
    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 3:7] = orientations
    env.setup_initial_env_states.robots[env.name].root_state[env_ids, 7:13] = velocities
