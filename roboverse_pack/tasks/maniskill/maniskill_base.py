from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class ManiskillBaseTask(BaseTaskEnv):
    """Base class for the ManiSkill **config ports** (``maniskill.<task>``).

    ManiSkill is integrated on **two tiers** — pick the one that matches your goal:

    * **Config port (this class, e.g. ``maniskill.pick_cube``,
      ``maniskill.peg_insertion_side_*``)** — a MetaSim ``ScenarioCfg`` +
      RoboVerse-format ``roboverse_data`` trajectory, runnable across any backend, for
      cross-simulator training/data. **Not** bit-exact with ManiSkill's SAPIEN physics
      — do **not** expect 1:1 official-demo replay from these ids.
    * **Native 1:1 port (``maniskill.<task>_native`` — see ``_native/`` +
      ``native_tasks.py``)** — SAPIEN3 with ManiSkill's PhysX/controller recipe,
      replays ManiSkill's *official* ``.h5`` demos and is bitwise pose-parity vs
      upstream (``tools.maniskill_integration.parity_native --all`` → 15/15). **Use
      these ids for 1:1 replay / the documented parity videos.**

    See ``release/benchmark/CROSS_BENCHMARK_1TO1.md`` and
    ``docs/source/dataset_benchmark/integrations/maniskill.md``.
    """

    scenario = None
    max_episode_steps = 250
    checker = None
    traj_filepath = None
    # Some leaves (draw_svg/draw_triangle/lift_peg_upright) intentionally skip the
    # per-reset checker reset. Set this True on a leaf instead of overriding reset,
    # so the seed-aware base reset() is inherited unchanged.
    skip_checker_reset = False

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super().__init__ because handler init
        super().__init__(scenario, device)

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
        initial_states, _, _ = get_traj(self.traj_filepath, self.scenario.robots[0], self.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
