"""Bit-faithful ports of ManiSkill tabletop dense rewards.

Each function reproduces the corresponding ManiSkill ``compute_dense_reward`` exactly (same tanh
shaping, same term order, same success override), taking the geometric quantities + contact predicate
the task already computes (object/tcp/goal positions, robot qvel, ``is_grasped`` from
:mod:`grasp`). Verified bitwise against native ManiSkill on identical inputs. numpy-only / float32 to
match the torch path; ``normalized`` divides by the task's max reward.
"""

from __future__ import annotations

import numpy as np


def _tanh(x):
    return np.tanh(np.float32(x))


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def pick_cube(
    *, cube_pos, tcp_pos, goal_pos, qvel_arm, is_grasped, goal_thresh=0.025, is_robot_static, normalized=False
) -> float:
    """ManiSkill PickCube-v1 ``compute_dense_reward`` (max 5).

    ``qvel_arm`` = robot qvel with the two finger DOFs dropped (panda). ``is_robot_static`` is the
    agent.is_static(0.2) flag. Success = placed AND static.
    """
    tcp_to_obj = _dist(cube_pos, tcp_pos)
    reward = 1.0 - _tanh(5.0 * tcp_to_obj)  # reaching
    reward += 1.0 if is_grasped else 0.0
    obj_to_goal = _dist(goal_pos, cube_pos)
    place_reward = 1.0 - _tanh(5.0 * obj_to_goal)
    reward += place_reward * (1.0 if is_grasped else 0.0)
    is_obj_placed = obj_to_goal <= goal_thresh
    static_reward = 1.0 - _tanh(5.0 * float(np.linalg.norm(np.asarray(qvel_arm, dtype=np.float32))))
    reward += static_reward * (1.0 if is_obj_placed else 0.0)
    success = is_obj_placed and bool(is_robot_static)
    if success:
        reward = 5.0
    return float(reward / 5.0) if normalized else float(reward)


def push_cube(*, cube_pos, tcp_pos, goal_pos, cube_half_size=0.02, normalized=False) -> float:
    """ManiSkill PushCube-v1 ``compute_dense_reward`` (max 3)."""
    push_pose = np.asarray(cube_pos, dtype=np.float32) + np.array([-cube_half_size - 0.005, 0, 0], np.float32)
    tcp_to_push = float(np.linalg.norm(push_pose - np.asarray(tcp_pos, dtype=np.float32)))
    reward = 1.0 - _tanh(5 * tcp_to_push)  # reaching
    reached = tcp_to_push < 0.01
    obj_to_goal = float(np.linalg.norm(np.asarray(cube_pos)[:2] - np.asarray(goal_pos)[:2]))
    place_reward = 1.0 - _tanh(5 * obj_to_goal)
    reward += place_reward * (1.0 if reached else 0.0)
    z_dev = abs(float(cube_pos[2]) - cube_half_size)
    z_reward = 1.0 - _tanh(5 * z_dev)
    reward += place_reward * z_reward * (1.0 if reached else 0.0)
    return float(reward / 3.0) if normalized else float(reward)


def pull_cube(*, cube_pos, tcp_pos, goal_pos, cube_half_size=0.02, normalized=False) -> float:
    """ManiSkill PullCube-v1 ``compute_dense_reward`` (max 3)."""
    pull_pose = np.asarray(cube_pos, dtype=np.float32) + np.array([cube_half_size + 0.01, 0, 0], np.float32)
    tcp_to_pull = float(np.linalg.norm(pull_pose - np.asarray(tcp_pos, dtype=np.float32)))
    reward = 1.0 - _tanh(5 * tcp_to_pull)
    reached = tcp_to_pull < 0.01
    obj_to_goal = float(np.linalg.norm(np.asarray(cube_pos)[:2] - np.asarray(goal_pos)[:2]))
    place_reward = 1.0 - _tanh(5 * obj_to_goal)
    reward += place_reward * (1.0 if reached else 0.0)
    return float(reward / 3.0) if normalized else float(reward)


def lift_peg_upright(*, peg_rot_mat, peg_pos, tcp_pos, is_grasped, peg_half_length, normalized=False) -> float:
    """ManiSkill LiftPegUpright-v1 ``compute_dense_reward`` (max ~2.2).

    ``peg_rot_mat`` is the peg pose's 3x3 rotation. rot reward = |(R @ x_hat) · z_hat|.
    """
    R = np.asarray(peg_rot_mat, dtype=np.float32)
    rot_vec = R @ np.array([1.0, 0, 0], np.float32)
    reward = abs(float(rot_vec @ np.array([0, 0, 1.0], np.float32)))
    z_dist = abs(float(peg_pos[2]) - peg_half_length)
    reward += 1.0 - _tanh(5 * z_dist)
    to_grip = float(np.linalg.norm(np.asarray(peg_pos, dtype=np.float32) - np.asarray(tcp_pos, dtype=np.float32)))
    reaching = 1.0 if is_grasped else (1.0 - _tanh(5 * to_grip))
    reward += reaching / 5.0
    return float(reward / 2.2) if normalized else float(reward)


def stack_cube(
    *,
    tcp_pos,
    cubeA_pos,
    cubeB_pos,
    cube_half_size_z,
    finger_qpos,
    gripper_width,
    cubeA_linvel,
    cubeA_angvel,
    is_cubeA_grasped,
    is_cubeA_on_cubeB,
    success,
    normalized=False,
) -> float:
    """ManiSkill StackCube-v1 ``compute_dense_reward`` (max 8)."""
    cubeA_to_tcp = float(np.linalg.norm(np.asarray(tcp_pos, np.float32) - np.asarray(cubeA_pos, np.float32)))
    reward = 2.0 * (1.0 - _tanh(5 * cubeA_to_tcp))
    goal_xyz = np.array([cubeB_pos[0], cubeB_pos[1], cubeB_pos[2] + cube_half_size_z * 2], np.float32)
    place_reward = 1.0 - _tanh(5.0 * float(np.linalg.norm(goal_xyz - np.asarray(cubeA_pos, np.float32))))
    if is_cubeA_grasped:
        reward = 4.0 + place_reward
    ungrasp_reward = float(np.sum(np.asarray(finger_qpos))) / gripper_width
    if not is_cubeA_grasped:
        ungrasp_reward = 1.0
    static_reward = 1.0 - _tanh(float(np.linalg.norm(cubeA_linvel)) * 10 + float(np.linalg.norm(cubeA_angvel)))
    if is_cubeA_on_cubeB:
        reward = 6.0 + (ungrasp_reward + static_reward) / 2.0
    if success:
        reward = 8.0
    return float(reward / 8.0) if normalized else float(reward)


def poke_cube(
    *,
    tcp_pos,
    peg_pos,
    angle_diff,
    head_to_cube_dist,
    is_peg_grasped,
    goal_pos,
    cube_pos,
    is_peg_cube_fit,
    is_cube_placed,
    qvel_arm,
    success,
    normalized=False,
) -> float:
    """ManiSkill PokeCube-v1 ``compute_dense_reward`` (max 10)."""
    tcp_to_peg = float(np.linalg.norm(np.asarray(tcp_pos, np.float32) - np.asarray(peg_pos, np.float32)))
    reached = tcp_to_peg < 0.01
    reward = 2.0 * (1.0 - _tanh(5.0 * tcp_to_peg))
    align = 1.0 - _tanh(5.0 * float(angle_diff))
    close = 1.0 - _tanh(5.0 * float(head_to_cube_dist))
    peg_grasped = bool(is_peg_grasped) and reached
    if peg_grasped:
        reward = 4.0 + close + align
    place = 1.0 - _tanh(5 * float(np.linalg.norm(np.asarray(goal_pos, np.float32) - np.asarray(cube_pos, np.float32))))
    peg_cube_fit = bool(is_peg_cube_fit) and peg_grasped
    if peg_cube_fit:
        reward = 7.0 + place
    static = 1.0 - _tanh(5 * float(np.linalg.norm(np.asarray(qvel_arm, np.float32))))
    if is_cube_placed:
        reward += static
    if success:
        reward = 10.0
    return float(reward / 10.0) if normalized else float(reward)


def place_sphere(
    *,
    tcp_pos,
    obj_pos,
    bin_pos,
    block_half_size0,
    radius,
    finger_qpos,
    gripper_width,
    obj_linvel,
    obj_angvel,
    robot_is_static,
    is_obj_grasped,
    is_obj_on_bin,
    success,
    normalized=False,
) -> float:
    """ManiSkill PlaceSphere-v1 ``compute_dense_reward`` (max 13)."""
    obj_to_tcp = float(np.linalg.norm(np.asarray(tcp_pos, np.float32) - np.asarray(obj_pos, np.float32)))
    reward = 2.0 * (1.0 - _tanh(5 * obj_to_tcp))
    bin_top = np.asarray(bin_pos, np.float32).copy()
    bin_top[2] = bin_top[2] + block_half_size0 + radius
    place_reward = 1.0 - _tanh(5.0 * float(np.linalg.norm(bin_top - np.asarray(obj_pos, np.float32))))
    if is_obj_grasped:
        reward = 4.0 + place_reward
    ungrasp_reward = float(np.sum(np.asarray(finger_qpos))) / gripper_width
    if not is_obj_grasped:
        ungrasp_reward = 16.0
    static_reward = 1.0 - _tanh(float(np.linalg.norm(obj_linvel)) * 10 + float(np.linalg.norm(obj_angvel)))
    if is_obj_on_bin:
        reward = 6.0 + (ungrasp_reward + static_reward + float(robot_is_static)) / 3.0
    if success:
        reward = 13.0
    return float(reward / 13.0) if normalized else float(reward)


def roll_ball(*, tcp_pos, ball_pos, goal_pos, ball_radius, reached_status, success, normalized=False):
    """ManiSkill RollBall-v1 ``compute_dense_reward`` (max 30). Stateful: pass ``reached_status`` in.

    Returns ``(reward, reached_status)``; ``reached_status`` is sticky (latches to 1.0).
    """
    ball = np.asarray(ball_pos, np.float32)
    goal = np.asarray(goal_pos, np.float32)
    unit = ball - goal
    unit = unit / np.linalg.norm(unit)
    tcp_hit = ball + unit * (ball_radius + 0.05)
    tcp_to_hit = float(np.linalg.norm(tcp_hit - np.asarray(tcp_pos, np.float32)))
    if tcp_to_hit < 0.04:
        reached_status = 1.0
    reaching = 1.0 - _tanh(2 * tcp_to_hit)
    obj_to_goal = float(np.linalg.norm(ball[:2] - goal[:2]))
    reached_reward = 1.0 - _tanh(obj_to_goal)
    reward = 20 * reached_reward * reached_status + reaching * (1 - reached_status) + reached_status
    if success:
        reward = 30.0
    return (float(reward / 30.0) if normalized else float(reward)), reached_status


def pull_cube_tool(
    *,
    tcp_pos,
    cube_pos,
    tool_pos,
    robot_base_pos,
    is_grasping,
    success,
    hook_length=0.05,
    cube_half_size=0.02,
    arm_reach=0.35,
    cube_size=0.02,
    normalized=False,
) -> float:
    """ManiSkill PullCubeTool-v1 ``compute_dense_reward`` (staged tool-use)."""
    tool_grasp = np.asarray(tool_pos, np.float64) + np.array([0.02, 0, 0])
    tcp_to_tool = float(np.linalg.norm(np.asarray(tcp_pos, np.float64) - tool_grasp))
    reaching = 2.0 * (1.0 - _tanh(5.0 * tcp_to_tool))
    grasping_reward = 2.0 * (1.0 if is_grasping else 0.0)
    ideal_hook = np.asarray(cube_pos, np.float64) + np.array([-(hook_length + cube_half_size), -0.067, 0])
    tool_pos_dist = float(np.linalg.norm(np.asarray(tool_pos, np.float64) - ideal_hook))
    positioning = 1.5 * (1.0 - _tanh(3.0 * tool_pos_dist))
    positioned = tool_pos_dist < 0.05
    workspace_target = np.asarray(robot_base_pos, np.float64) + np.array([0.05, 0, 0])
    cube_to_ws = float(np.linalg.norm(np.asarray(cube_pos, np.float64) - workspace_target))
    initial_dist = float(np.linalg.norm(np.array([arm_reach + 0.1, 0, cube_size / 2]) - workspace_target))
    pulling_progress = (initial_dist - cube_to_ws) / initial_dist
    pulling = 3.0 * pulling_progress * (1.0 if positioned else 0.0)
    g = 1.0 if is_grasping else 0.0
    reward = reaching + grasping_reward + positioning * g + pulling * g
    if float(cube_pos[0]) > (arm_reach + 0.15):
        reward -= 2.0
    if success:
        reward += 5.0
    return float(reward)
