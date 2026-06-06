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


def pick_cube(*, cube_pos, tcp_pos, goal_pos, qvel_arm, is_grasped, goal_thresh=0.025,
              is_robot_static, normalized=False) -> float:
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
