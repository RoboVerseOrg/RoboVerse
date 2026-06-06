"""Bit-faithful ports of ManiSkill tabletop ``evaluate()`` success conditions.

Each function reproduces the corresponding ManiSkill success predicate exactly, taking the geometric
quantities + contact/static predicates the task already computes (object/goal positions, robot/object
velocities via the ``is_*`` flags, ``is_grasped`` from :mod:`grasp`). Verified by boolean agreement
against native ManiSkill over a rollout. numpy-only; clone-deletable.
"""

from __future__ import annotations

import numpy as np


def _xy(p):
    return np.asarray(p, dtype=np.float64)[:2]


def _d(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)))


def pick_cube(*, cube_pos, goal_pos, is_robot_static, goal_thresh=0.025) -> bool:
    return bool(_d(cube_pos, goal_pos) <= goal_thresh and is_robot_static)


def push_cube(*, cube_pos, goal_pos, cube_half_size=0.02, goal_radius=0.1) -> bool:
    placed = np.linalg.norm(_xy(cube_pos) - _xy(goal_pos)) < goal_radius
    on_surface = float(cube_pos[2]) < cube_half_size + 5e-3
    return bool(placed and on_surface)


def pull_cube(*, cube_pos, goal_pos, goal_radius=0.1) -> bool:
    return bool(np.linalg.norm(_xy(cube_pos) - _xy(goal_pos)) < goal_radius)


def poke_cube(*, cube_pos, goal_pos, is_robot_static, goal_radius=0.05) -> bool:
    placed = np.linalg.norm(_xy(cube_pos) - _xy(goal_pos)) < goal_radius
    return bool(placed and is_robot_static)


def roll_ball(*, ball_pos, goal_pos, goal_radius=0.1) -> bool:
    return bool(np.linalg.norm(_xy(ball_pos) - _xy(goal_pos)) < goal_radius)


def stack_cube(*, cubeA_pos, cubeB_pos, cube_half_size, is_cubeA_static, is_cubeA_grasped) -> bool:
    offset = np.asarray(cubeA_pos, np.float64) - np.asarray(cubeB_pos, np.float64)
    xy_flag = np.linalg.norm(offset[:2]) <= np.linalg.norm(np.asarray(cube_half_size)[:2]) + 0.005
    z_flag = abs(offset[2] - cube_half_size[2] * 2) <= 0.005
    return bool(xy_flag and z_flag and is_cubeA_static and not is_cubeA_grasped)


def place_sphere(*, obj_pos, bin_pos, radius, block_half_size0, is_obj_static, is_obj_grasped) -> bool:
    offset = np.asarray(obj_pos, np.float64) - np.asarray(bin_pos, np.float64)
    xy_flag = np.linalg.norm(offset[:2]) <= 0.005
    z_flag = abs(offset[2] - radius - block_half_size0) <= 0.005
    return bool(xy_flag and z_flag and is_obj_static and not is_obj_grasped)


def lift_peg_upright(*, peg_euler_z, peg_z, peg_half_length) -> bool:
    upright = abs(abs(peg_euler_z) - np.pi / 2) < 0.08
    close = abs(peg_z - peg_half_length) < 0.005
    return bool(upright and close)


def stack_pyramid(*, posA, posB, posC, cube_half_size, statics, grasps) -> bool:
    """A next-to B, C on B, C on A. ``statics``/``grasps`` keyed by 'A'/'B'/'C'."""
    hs = np.asarray(cube_half_size, np.float64)
    xy_tol = np.linalg.norm(2 * hs[:2]) + 0.005

    def ok(a, b, mode, key):
        off = np.asarray(a, np.float64) - np.asarray(b, np.float64)
        xy = np.linalg.norm(off[:2]) <= xy_tol
        flag = (xy and abs(off[2]) > 0.02) if mode == "top" else xy
        return bool(flag and statics[key] and not grasps[key])

    return bool(ok(posA, posB, "next_to", "A") and ok(posC, posB, "top", "C") and ok(posC, posA, "top", "C"))


def pull_cube_tool(*, cube_pos, robot_base_pos) -> bool:
    return bool(np.linalg.norm(_xy(cube_pos) - _xy(robot_base_pos)) < 0.6)
