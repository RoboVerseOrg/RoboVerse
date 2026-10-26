"""Configuration for the Libero kitchen open drawer put bowl task."""

from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.site_util import SitePose, site_pose
from metasim.utils.state import TensorState

from ._goal import GoalTerm, check_goal, goal_holds


class Libero90BaseTask(BaseTaskEnv):
    """Base class for LIBERO 90 tasks.

    This base class handles the common functionality for LIBERO 90 tasks,
    which are more complex than LIBERO Object tasks.

    Like ``LiberoBaseTask``, these are **config ports**: a MetaSim scenario that runs on any backend, not a
    reproduction of LIBERO's robosuite physics, so the official demos are not expected to replay to success
    (on MuJoCo they do not: there is no table, and objects settle ~2 cm below where the demos hold them).
    For 1:1 replay use the native ids (``native_libero.py``).

    A task's success is its BDDL goal, declared as data (``goal``, see ``_goal``) and evaluated with LIBERO's own
    ``In`` / ``On`` / ``Open`` / ``Close`` / ``Turnon`` predicates. The site regions they name (a basket's
    ``contain_region``, a drawer's ``top_region``) are read with :meth:`_site_pose`, which composes the site's
    fixed offset from the object's MJCF with the pose in the state, so a goal reads the same frame on every
    backend. Writing LIBERO's demos into MuJoCo, the goal holds at the last state of every demo on 58 of 59
    tasks (the other misses a wine-rack region by 0.2 mm in one demo); the hand-written checkers it replaced
    held on 51.
    """

    scenario = None
    max_episode_steps = 300  # LIBERO 90 tasks are more complex
    task_desc = None
    checker = None
    goal: tuple[GoalTerm, ...] | None = None
    """The BDDL goal's terms (see ``_goal``); when set, success is that goal and ``checker`` is unused."""
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
        if self.goal is not None:
            check_goal(self.goal, {obj.name for obj in scenario.objects})
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when the task's goal (or, for a task without one, its checker) holds."""
        if self.goal is not None:
            return goal_holds(self, states, self.goal)
        return self.checker.check(self.handler, states)

    def _site_pose(self, states: TensorState, obj_name: str, site: str) -> SitePose:
        """World pose of ``site`` on the scenario object ``obj_name``, from ``states``."""
        for obj in self.scenario.objects:
            if obj.name == obj_name:
                return site_pose(states, obj, site)
        raise KeyError(f"{type(self).__name__}: no object {obj_name!r} in the scenario")

    def _on_reset(self, env_ids: list[int]) -> None:
        """Reset the checker for ``env_ids`` unless ``skip_checker_reset`` is set."""
        if self.checker is not None and not self.skip_checker_reset:
            self.checker.reset(self.handler, env_ids=env_ids)

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
