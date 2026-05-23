from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.math import matrix_from_quat
from metasim.utils.state import TensorState

from ._dense_rl import DenseRLResetMixin
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


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 LiftPegUpright-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/lift_peg_upright.py).
# ---------------------------------------------------------------------------

_LP_HALF_LENGTH = 0.12  # peg_length / 2 — target z when upright
_LP_HALF_WIDTH = 0.025  # peg_widdth / 2 — rest height lying flat
_LP_TCP_OFFSET = (0.0, 0.0, 0.10312)
_LP_UPRIGHT_COS = float(torch.cos(torch.tensor(0.08)))  # ManiSkill 0.08 rad tol


@register_task("maniskill.lift_peg_upright_dense", "lift_peg_upright_dense")
class LiftPegUprightDenseTask(DenseRLResetMixin, LiftPegUprightTask):
    """LiftPegUpright with ManiSkill3's dense reward ported 1:1.

    Reward (un-normalized, max 3)::

        rot_rew  = |peg_x_axis · ẑ|                       # upright alignment
        reward   = rot_rew
        reward  += 1 - tanh(5·|peg.z - half_length|)      # lift to upright height
        reaching = 1 - tanh(5·||peg - tcp||); =1 if grasping; /5
        reward  += reaching
        reward[success] = 3

    success = upright (|peg_x·ẑ| > cos 0.08) & |peg.z-half_length|<0.005.
    ``is_grasping`` approximated (gripper closure + peg proximity). Randomized
    resets (peg lies flat at z=half_width), no demo data.
    """

    cube_z = _LP_HALF_WIDTH  # peg rests on its side
    goal_z_range = (_LP_HALF_WIDTH, _LP_HALF_WIDTH)  # no goal object, unused

    def _reward(self, env_states):
        from metasim.utils.math import matrix_from_quat, quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        peg = objs["peg"].root_state[:, :3]
        peg_quat = objs["peg"].root_state[:, 3:7]
        peg_x = matrix_from_quat(peg_quat)[:, :, 0]  # peg local x-axis in world

        rot_rew = peg_x[:, 2].abs()
        reward = rot_rew

        z_dist = torch.abs(peg[:, 2] - _LP_HALF_LENGTH)
        reward = reward + (1.0 - torch.tanh(5.0 * z_dist))

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_LP_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        to_grip = torch.linalg.norm(peg - tcp, dim=-1)
        reaching = 1.0 - torch.tanh(5.0 * to_grip)
        gripper_width = rs.joint_pos[:, 7:9].sum(dim=-1)
        is_grasping = (to_grip < 0.04) & (gripper_width < 0.07)
        reaching = torch.where(is_grasping, torch.ones_like(reaching), reaching)
        reaching = reaching / 5.0
        reward = reward + reaching

        success = (rot_rew > _LP_UPRIGHT_COS) & (z_dist < 0.005)
        reward = torch.where(success, torch.full_like(reward, 3.0), reward)
        return reward
