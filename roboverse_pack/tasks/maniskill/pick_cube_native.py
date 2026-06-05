"""``maniskill.pick_cube_native`` — a PickCube task that reproduces native ManiSkill 1:1.

Unlike the demo-trajectory-backed ``maniskill.pick_cube`` / ``pick_cube_v1`` tasks, this one runs on
the ManiSkill-faithful PhysX recipe (via the opt-in sapien3 handler knobs) and consumes ManiSkill's
native ``pd_joint_delta_pos`` action contract, so its dynamics track the native ``PickCube-v1``
``physx_cpu`` rollout to PhysX float32 roundoff (object pose ~5e-6 over dozens of aggressive steps).

Scene = ManiSkill panda (panda_v2.urdf + its drive gains) + the kinematic table-workspace box + a
4 cm red cube. Success is a simple cube-lift check; the full goal-site + robot-static success and
dense reward (which needs the contact-force ``is_grasped`` query) are follow-ups.
"""

from __future__ import annotations

import torch

from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.state import TensorState

from ._native.base import ManiSkillNativeTask
from ._native.recipe import maniskill_panda_cfg, table_workspace_cfg

# ManiSkill PickCube-v1: 4 cm cube, mass 0.064 kg (density 1000), default friction.
_CUBE = PrimitiveCubeCfg(
    name="cube",
    size=[0.04, 0.04, 0.04],
    mass=0.064,
    color=[1.0, 0.0, 0.0],
    default_position=(0.0, 0.0, 0.02),  # resting on the table top (z=0)
)

_LIFT_HEIGHT = 0.1  # cube counts as picked once lifted ~10 cm off the table


@register_task("maniskill.pick_cube_native", "pick_cube_native")
class PickCubeNativeTask(ManiSkillNativeTask):
    """PickCube on the ManiSkill-faithful recipe (1:1 dynamics through the sapien3 handler)."""

    scenario = ScenarioCfg(
        robots=[maniskill_panda_cfg()],
        objects=[table_workspace_cfg(), _CUBE],
    )
    max_episode_steps = 50

    def _terminated(self, states: TensorState) -> torch.Tensor:
        """Cube lifted ~10 cm off the table (simple, contact-free success)."""
        cube_z = float(self.handler.object_ids["cube"].get_pose().p[2])
        picked = cube_z > 0.02 + _LIFT_HEIGHT
        return torch.tensor([picked] * self.num_envs, dtype=torch.bool)
