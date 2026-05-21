"""Term config dataclasses shared by all manager-based RL envs.

Mirrors the IsaacLab / mjlab ``*TermCfg`` pattern. The shapes match the
existing beyondmimic definitions in
``roboverse_pack/tasks/beyondmimic/metasim/configs/tracking_g1.py:31-57``;
that file keeps its private copies for the moment so the motion-tracking
task is not perturbed.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Callable

from metasim.utils import configclass


@configclass
class CfgTerm:
    """Base for manager term configs.

    ``func`` receives ``(env, env_states, **params)`` and returns a tensor
    of shape ``(num_envs,)`` (rewards / terminations) or ``(num_envs, D)``
    (observations).
    """

    func: Callable = MISSING
    params: dict[str, Any] = {}


@configclass
class ObsTerm(CfgTerm):
    """Configuration for an observation term.

    ``noise_range`` adds uniform noise in ``[lo, hi]`` to the term output
    after computation. ``None`` disables noise.
    """

    noise_range: tuple[float, float] | None = None


@configclass
class RewTerm(CfgTerm):
    """Configuration for a reward term.

    The manager loop computes ``weight * func(env, env_states, **params) *
    step_dt`` and sums across terms. Per-term episode sums are recorded
    under ``extras["episode"]["Episode_Reward/<term_name>"]``.
    """

    weight: float = 1.0


@configclass
class DoneTerm(CfgTerm):
    """Configuration for a termination term.

    ``time_out=True`` routes the term's signal into the time-out (truncation)
    bucket instead of regular termination — required for PPO bootstrap-value
    accounting on infinite-horizon tasks.
    """

    time_out: bool = False


@configclass
class EventTerm(CfgTerm):
    """Configuration for an event term.

    ``mode`` is one of ``"setup" | "reset" | "pre_step" | "post_step" |
    "terminate"``. Event functions are dispatched at the corresponding
    point in the env lifecycle and mutate env state in-place. Backwards-
    compatible with the existing beyondmimic callback dict format.
    """

    mode: str = "reset"


@configclass
class CurriculumTerm(CfgTerm):
    """Configuration for a curriculum term.

    Called from ``_reset_idx`` for the freshly reset env subset. Return
    value (typically a scalar or dict) is logged under
    ``extras["episode"]["Curriculum/<term_name>"]``.
    """
