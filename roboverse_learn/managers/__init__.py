"""Canonical manager-based RL env pattern for RoboVerse.

Generalized from the proto-implementation in
``roboverse_pack/tasks/beyondmimic/metasim/`` (see ``LeggedRobotTask``)
so non-humanoid tasks (mjlab port, manipulation, etc.) can share the same
``ObsTerm/RewTerm/DoneTerm`` declarative config style without inheriting
the legged-robot-specific machinery.

The existing ``LeggedRobotTask`` is intentionally not modified; it will
optionally migrate to inherit ``ManagerBasedRVEnv`` in a later pass once
this base is exercised by the mjlab task ports.

Public surface
--------------
``ObsTerm``, ``RewTerm``, ``DoneTerm``, ``CurriculumTerm``, ``EventTerm``
    Declarative term configs. Same shape as IsaacLab / mjlab and as the
    existing beyondmimic definitions.

``ManagerBasedRVEnvCfg``
    Base config dataclass. Subclasses declare ``observations / rewards /
    terminations / events / curriculum`` groups using the Term classes
    above plus ``decimation`` and ``max_episode_length_s``.

``ManagerBasedRVEnv``
    Sim-agnostic RL env. Composes the manager loop (compute reward and
    termination per step, dispatch events, reset failed envs, log
    per-term episode statistics). Subclasses implement only:
    - ``_apply_action(action)`` — raw → handler-friendly action
    - ``_get_initial_states()`` — per-env reset state
    - optionally override ``_observation_group(group_name)`` for non-
      standard observation aggregation.
"""

from __future__ import annotations

from .term_cfgs import (
    CfgTerm,
    CurriculumTerm,
    DoneTerm,
    EventTerm,
    ObsTerm,
    RewTerm,
)
from .manager_based_rv_env import (
    ManagerBasedRVEnv,
    ManagerBasedRVEnvCfg,
)

__all__ = [
    "CfgTerm",
    "ObsTerm",
    "RewTerm",
    "DoneTerm",
    "CurriculumTerm",
    "EventTerm",
    "ManagerBasedRVEnv",
    "ManagerBasedRVEnvCfg",
]
