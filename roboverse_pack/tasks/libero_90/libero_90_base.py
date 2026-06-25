"""Configuration for the Libero kitchen open drawer put bowl task."""

from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class Libero90BaseTask(BaseTaskEnv):
    """Base class for LIBERO 90 tasks.

    This base class handles the common functionality for LIBERO 90 tasks,
    which are more complex than LIBERO Object tasks.
    """

    scenario = None
    max_episode_steps = 300  # LIBERO 90 tasks are more complex
    task_desc = None
    checker = None
    traj_filepath = None
    decimation = 30
    # Most libero_90 leaves intentionally skip the per-reset checker reset. Setting
    # this True on a leaf reproduces the old ``reset`` override (which jumped to
    # ``BaseTaskEnv.reset`` to bypass the checker reset) without overriding ``reset``
    # — so the seed-aware base ``reset`` signature is inherited unchanged.
    skip_checker_reset = False

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super()._init_ because handler init
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when task conditions are met."""
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None, seed=None):
        """Reset the env; reset the checker unless ``skip_checker_reset`` is set.

        ``seed`` is forwarded to ``super().reset``. Leaves that set
        ``skip_checker_reset = True`` get the old "skip checker reset" behaviour
        without overriding ``reset`` (so the seed contract is preserved).
        """
        states = super().reset(states, env_ids, seed)
        if self.checker is not None and not self.skip_checker_reset:
            self.checker.reset(self.handler, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Return per-env initial states sampled from the demo trajectory.

        Returns None when the trajectory is missing or empty, letting the
        handler fall back to its own defaults. Mirrors the hardened
        ``LiberoBaseTask._get_initial_states``: the previous version indexed
        ``self.scenario.robots[0]`` unconditionally and called ``get_traj``
        with no guard, so a robotless scenario or a missing/empty traj file
        raised (``IndexError``/``FileNotFoundError``/``KeyError``) instead of
        degrading gracefully like its libero sibling.
        """
        if not self.traj_filepath:
            return None
        # scenario.robots may legitimately be empty (perception-only tasks).
        robot_ref = self.scenario.robots[0] if self.scenario.robots else None
        try:
            initial_states, _, _ = get_traj(self.traj_filepath, robot_ref, self.handler)
        except (FileNotFoundError, KeyError, ValueError):
            return None

        if not initial_states:
            return None

        # Duplicate / trim list so that its length matches num_envs (n > 0 here).
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
