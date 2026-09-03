# Copyright (c) ManiSkill2-real2sim contributors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from ManiSkill2-real2sim (https://github.com/simpler-env/ManiSkill2_real2sim).
# Changes: helpers collected from `mani_skill2_real2sim.utils.{common,sapien_utils,geometry}` into one module so
#   the vendored controllers import nothing from upstream; function bodies unchanged (one docstring re-wrapped; the
#   `sapien.Pose` annotation on `vectorize_pose` dropped along with the sapien import).
# Full license: roboverse_pack/tasks/simpler_env/_native/control/LICENSE
"""Vendored helper functions for the SimplerEnv native controller stack.

Copied verbatim (byte-identical bodies) from ``mani_skill2_real2sim.utils.common`` /
``mani_skill2_real2sim.utils.sapien_utils`` / ``mani_skill2_real2sim.utils.geometry`` so the
native SimplerEnv port has NO runtime dependency on the upstream ``mani_skill2_real2sim``
package (the whole point: be deletable). Pure numpy / sapien / gymnasium.

Provenance: simpler-env/ManiSkill2_real2sim @ ef7a4d4 (ManiSkill2 0.5.3 fork).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    low, high = np.asarray(low), np.asarray(high)
    action = np.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def normalize_action_space(action_space: spaces.Box):
    assert isinstance(action_space, spaces.Box), type(action_space)
    return spaces.Box(-1, 1, shape=action_space.shape, dtype=action_space.dtype)


def vectorize_pose(pose):
    return np.hstack([pose.p, pose.q])


def get_entity_by_name(entities, name: str, is_unique=True):
    """Get a Sapien.Entity given the name.

    Args:
        entities (List[sapien.Entity]): entities (link, joint, ...) to query.
        name (str): name for query.
        is_unique (bool, optional): whether the name should be unique. Defaults to True.

    Returns:
        sapien.Entity or List[sapien.Entity]: matched entity or entities. None if no matches.
    """
    matched_entities = [x for x in entities if x.get_name() == name]
    if len(matched_entities) > 1:
        if not is_unique:
            return matched_entities
        else:
            raise RuntimeError(f"Multiple entities with the same name {name}.")
    elif len(matched_entities) == 1:
        return matched_entities[0]
    else:
        return None


def rotate_2d_vec_by_angle(vec, theta):
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rot_mat @ vec


def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()
