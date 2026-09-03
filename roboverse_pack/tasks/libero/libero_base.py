from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.utils.demo_util import get_traj
from metasim.utils.hf_util import check_and_download_single
from metasim.utils.state import TensorState


class LiberoBaseTask(BaseTaskEnv):
    """Base class for the LIBERO **config ports** (``libero.<task>``).

    LIBERO is integrated into RoboVerse on **two tiers** — pick the one that matches
    your goal:

    * **Config port (this class, ``libero.pick_*``)** — a MetaSim ``ScenarioCfg`` +
      RoboVerse-format ``roboverse_data`` trajectory, runnable across *any* backend
      (MuJoCo/SAPIEN/Isaac/…). Meant for cross-simulator training and data; it is
      **not** a bit-exact reproduction of LIBERO's original robosuite physics — do
      **not** expect 1:1 official-demo replay from these ids.
    * **Native 1:1 port (``libero_native.*`` — see ``native_libero.py``)** — loads
      LIBERO's *own* compiled MJCF onto MetaSim's MuJoCo handler and is **bitwise 1:1**
      with LIBERO (engine + state-replay ``max|Δ|=0``, recorded demo success
      reproduced), runnable with ``libero`` / ``robosuite`` uninstalled. **Use these
      ids for 1:1 replay / the documented parity videos.** A transparent
      ``_passthrough.py`` (needs ``libero`` installed) forwards to upstream verbatim
      and is used only for native-vs-upstream comparison.

    See ``release/benchmark/CROSS_BENCHMARK_1TO1.md`` and
    ``docs/source/dataset_benchmark/integrations/libero.md``.

    This task is transferred from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_object/pick_up_the_butter_and_place_it_in_the_basket.bddl
    """

    scenario = None
    max_episode_steps = 250
    task_desc = None
    checker = None
    traj_filepath = None

    def __init__(self, scenario: ScenarioCfg, device: str | torch.device | None = None) -> None:
        check_and_download_single(self.traj_filepath)
        # update objects and robots defined by task, must before super()._init_ because handler init
        super().__init__(scenario, device)

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Success when cube is detected in the bbox above base."""
        return self.checker.check(self.handler, states)

    def reset(self, states=None, env_ids=None, seed=None):
        """Reset the checker, then the environment.

        The checker is reset *before* ``super().reset()`` — super().reset()
        may invoke reset-callbacks that read termination state (computed
        via the checker), which would otherwise see stale checker state.

        ``seed`` is forwarded to ``super().reset`` so the ``env.reset(seed=)``
        contract reaches the handler (see ``BaseTaskEnv.reset``).
        """
        if self.checker is not None:
            self.checker.reset(self.handler, env_ids=env_ids)
        return super().reset(states, env_ids, seed)

    def _get_initial_states(self) -> list[dict] | None:
        """Return per-env initial states sampled from the demo trajectory.

        Returns None when the trajectory is missing or empty, letting the
        handler fall back to its own defaults.
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

        # Replicate to match num_envs (n is guaranteed > 0 here).
        n = len(initial_states)
        if n < self.num_envs:
            k, rem = divmod(self.num_envs, n)
            initial_states = initial_states * k + initial_states[:rem]
        self._initial_states = initial_states[: self.num_envs]
        return self._initial_states
