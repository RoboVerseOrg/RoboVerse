from __future__ import annotations

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.state import TensorState

from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.draw_svg", "draw_svg")
class DrawSvgTask(ManiskillBaseTask):
    """Draw an SVG trajectory with a Panda robot using a red cube as marker."""

    scenario = ScenarioCfg(
        robots=["franka"],
    )

    traj_filepath = "roboverse_data/trajs/maniskill/draw_svg/v2/franka_v2.pkl.gz"

    max_episode_steps = 500

    # rewrite terminate
    def _terminated(self, states: TensorState) -> torch.Tensor:
        """No terminate condition yet. Will terminate when time is up."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # rewrite checker
    skip_checker_reset = True  # inherit BaseTaskEnv.reset(seed=); skip per-reset checker reset
