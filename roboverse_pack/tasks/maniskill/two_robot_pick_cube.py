"""``maniskill.two_robot_pick_cube_native`` — multi-robot ManiSkill task, native 1:1.

Two PandaWristCam arms (ManiSkill ``panda_v3.urdf``) at the table edges hand a cube between them. Runs
on the ManiSkill-faithful PhysX recipe through the standard sapien3 handler; the 16-dim action
(2 x 8-dim ``pd_joint_delta_pos``) is split per robot by :class:`ManiSkillMultiRobotTask`, matching
ManiSkill's MultiAgent contract 1:1 (object pose Δ=0 vs native; see ``parity_multi_agent``).
"""

from __future__ import annotations

import torch

from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.state import TensorState

from ._native.multi_robot import ManiSkillMultiRobotTask
from ._native.recipe import maniskill_panda_wristcam_cfg, table_workspace_cfg

# Panda rest keyframe (ManiSkill TwoRobotPickCube init, pre-noise).
_REST = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.401,
    "panda_joint3": 0.0,
    "panda_joint4": -1.919,
    "panda_joint5": 0.0,
    "panda_joint6": 2.337,
    "panda_joint7": 0.804,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
_QPOS_L = (0.7071068, 0.0, 0.0, 0.7071068)  # +90deg z (left robot faces +y)
_QPOS_R = (0.7071068, 0.0, 0.0, -0.7071068)  # -90deg z (right robot faces -y)

_CUBE = PrimitiveCubeCfg(
    name="cube",
    size=[0.04, 0.04, 0.04],
    mass=0.064,
    color=[1.0, 0.0, 0.0],
    default_position=(-0.0004, -0.1768, 0.02),
)


@register_task("maniskill.two_robot_pick_cube_native", "two_robot_pick_cube_native")
class TwoRobotPickCubeNativeTask(ManiSkillMultiRobotTask):
    """TwoRobotPickCube on the ManiSkill recipe (2 x panda_wristcam, 16-dim action)."""

    robot_names = ["panda-0", "panda-1"]
    per_robot_action_dim = 8

    scenario = ScenarioCfg(
        robots=[
            maniskill_panda_wristcam_cfg("panda-0", (0.0, -0.75, 0.0), _QPOS_L, _REST),
            maniskill_panda_wristcam_cfg("panda-1", (0.0, 0.75, 0.0), _QPOS_R, _REST),
        ],
        objects=[table_workspace_cfg(), _CUBE],
    )
    max_episode_steps = 100

    def _terminated(self, states: TensorState) -> torch.Tensor:
        # Proxy success: cube handed up off the table (full ManiSkill success uses a goal site +
        # per-arm reach conditions — a follow-up).
        cube_z = float(self.handler.object_ids["cube"].get_pose().p[2])
        return torch.tensor([cube_z > 0.1] * self.num_envs, dtype=torch.bool)
