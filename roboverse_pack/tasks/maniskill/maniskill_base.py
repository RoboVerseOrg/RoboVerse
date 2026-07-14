from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class ManiskillBaseTask(BaseTaskEnv):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    scenario = None
    max_episode_steps = 250
    checker = None
    traj_filepath = None
    # Some leaves (draw_svg/draw_triangle/lift_peg_upright) intentionally skip the
    # per-reset checker reset. Set this True on a leaf instead of overriding reset,
    # so the seed-aware base reset() is inherited unchanged.
    skip_checker_reset = False

    default_robots: list[str] = ["franka"]
    """Robot(s) to use when a leaf's ``scenario`` does not name one.

    Leaves declare ``scenario = ScenarioCfg(objects=[...])``, which *replaces* this class's
    scenario wholesale; ``ScenarioCfg.robots`` then falls back to its ``[]`` default. Every
    ManiSkill task ported into this pack is a Franka/Panda tabletop task replaying a
    franka-keyed demo (``roboverse_data/trajs/maniskill/**/trajectory-franka-*``), so the arm
    is a property of the *family*, not of each of the ~2.6k object variants. Keep it here as
    the single source of truth; a leaf that needs a different arm names it explicitly in its
    own ``scenario.robots`` (which always wins) or overrides ``default_robots``.
    """

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        # update objects and robots defined by task, must before super().__init__ because handler init
        scenario = self._apply_default_robots(scenario)
        check_and_download_single(self.traj_filepath)
        super().__init__(scenario, device)

    @classmethod
    def _apply_default_robots(cls, scenario: ScenarioCfg | None = None) -> ScenarioCfg | None:
        """Fill in ``default_robots`` when the scenario names no robot, and return it."""
        if scenario is None:
            scenario = cls.scenario
        if isinstance(scenario, ScenarioCfg) and not scenario.robots and cls.default_robots:
            # update() re-runs __post_init__, which resolves the names into RobotCfg instances.
            scenario.update(robots=list(cls.default_robots))
        return scenario

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success when the cube has been lifted sufficiently along z from its reset height."""
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None, seed=None):
        """Reset the env; reset the checker unless ``skip_checker_reset`` is set.

        ``seed`` is forwarded to ``super().reset``.
        """
        states = super().reset(states, env_ids, seed)
        if self.checker is not None and not self.skip_checker_reset:
            self.checker.reset(self.handler, env_ids=env_ids)
        return states

    def _get_initial_states(self) -> list[dict] | None:
        """Give the inital states from traj file."""
        # Keep it simple and leave robot states to defaults; just seed cube pose.
        # If the handler handles None gracefully, this can be set to None.
        if not self.scenario.robots:
            # Previously an IndexError from `robots[0]` two lines down, which said nothing
            # about which task was misconfigured or why a robot is required.
            raise ValueError(
                f"{type(self).__name__} has no robot in its scenario, but ManiSkill tasks replay a"
                f" robot-keyed demo trajectory ({self.traj_filepath}) and cannot be manipulated"
                f" without an arm. Set robots=[...] on the task's ScenarioCfg, or leave"
                f" {type(self).__name__}.default_robots at its default."
            )
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
