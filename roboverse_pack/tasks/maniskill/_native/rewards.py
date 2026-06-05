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
