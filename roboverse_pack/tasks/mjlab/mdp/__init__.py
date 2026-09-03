# RoboVerse-original: re-export shim only; contains no mjlab code.
"""Mjlab MDP function library ported to RoboVerse.

These functions are the term implementations used by mjlab task configs
(``mjlab/src/mjlab/tasks/{cartpole,velocity,tracking,manipulation}/``).
They are adapted to RoboVerse's ``ManagerBasedRVEnv`` calling convention:

  - mjlab signature:  ``func(env, **params)``   (reads ``env.scene["robot"].data``)
  - port signature:   ``func(env, env_states, **params)``  (reads ``env_states.robots[...]``)

The ``SceneEntityCfg`` adapter is a minimal port — entity name plus
joint/body name selectors, resolved against the simulator handler's
sorted joint/body ordering at first call.

Phase 2a scope (cartpole minimum):
  observations.joint_pos_rel, .joint_vel_rel, .pole_angle_cos_sin, .last_action
  rewards.cartpole_smooth_reward, ._gaussian_tolerance, ._quadratic_tolerance
  terminations.time_out
  events.reset_joints_by_offset

Future phases will add velocity / tracking / manipulation terms here as
each task is ported.
"""

from __future__ import annotations

from . import actions, commands, curriculums, events, events_dr, observations, rewards, sensors, terminations, terrain
from .scene_entity import SceneEntityCfg, resolve_body_ids, resolve_joint_ids

__all__ = [
    "SceneEntityCfg",
    "actions",
    "commands",
    "curriculums",
    "events",
    "events_dr",
    "observations",
    "resolve_body_ids",
    "resolve_joint_ids",
    "rewards",
    "sensors",
    "terminations",
    "terrain",
]
