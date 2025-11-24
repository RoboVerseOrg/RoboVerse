from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.math import matrix_from_quat
from metasim.utils.state import TensorState

from .maniskill_base import ManiskillBaseTask

peg_length = 0.24
peg_widdth = 0.05


@register_task("maniskill.lift_peg_upright", "lift_peg_upright")
class LiftPegUprightTask(ManiskillBaseTask):
    """Lift peg upright task from ManiSkill."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="peg",
                size=(peg_length, peg_widdth, peg_widdth),
                mass=0.04,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            )
        ],
        robots=["franka"],
    )

    traj_filepath = "roboverse_data/trajs/maniskill/lift_peg_upright/v2/franka_v2.pkl.gz"

    max_episode_steps = 800

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Task success when the peg is upright enough (z-axis)."""
        # get peg pos and quat
        peg_pos = self.handler.get_states(mode="tensor").objects["peg"].root_state[:, :3]
        peg_quat = self.handler.get_states(mode="tensor").objects["peg"].root_state[:, 3:7]

        # check pos
        pos_condition = torch.abs(peg_pos[:, 2] - peg_length / 2) < 0.05

        # check quat
        peg_x = matrix_from_quat(peg_quat)[:, :, 0]  # peg local x_axis(N, 3)
        quat_condition = peg_x[:, 2] > torch.cos(torch.tensor(10 * torch.pi / 180))  # within 10 degree
        return pos_condition & quat_condition

    # rewrite checker
    def reset(self, states=None, env_ids=None):
        """Skip checker reset."""
        states = super(ManiskillBaseTask, self).reset(states, env_ids)
        return states
