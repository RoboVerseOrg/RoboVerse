from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.state import TensorState

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.draw_triangle", "draw_triangle")
class DrawTriangleTask(ManiskillBaseTask):
    """Pick up the red cube with a Panda robot and lift it by 0.1 m."""

    scenario = ScenarioCfg(
        robots=["franka"],
    )

    traj_filepath = "roboverse_data/trajs/maniskill/draw_triangle/v2/franka_v2.pkl.gz"

    max_episode_steps = 250

    # rewrite terminate
    def _terminated(self, states: TensorState) -> torch.Tensor:
        """No terminate condition yet. Will terminate when time is up."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # rewrite checker
    skip_checker_reset = True  # inherit BaseTaskEnv.reset(seed=); skip per-reset checker reset
