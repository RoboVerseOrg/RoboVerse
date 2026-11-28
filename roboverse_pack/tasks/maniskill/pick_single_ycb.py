"""The base class and derived classes for the pick up single YCB object task from ManiSkill."""

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.checkers.checkers import PositionShiftChecker
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask


class _PickSingleYcbBaseTask(ManiskillBaseTask):
    """The pick up single YCB object task from ManiSkill.

    The robot is tasked to pick up an EGAD object.
    Note that the checker is not same as the original one (checking if the cube is near the target position).
    The current one checks if the cube is lifted up 7.5 cm.
    This class should be derived to specify the exact configuration (asset path and demo path) of the task.
    """

    max_episode_steps = 250
    checker = PositionShiftChecker(
        obj_name="obj",
        distance=0.075,
        axis="z",
    )


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PickSingleYCB-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/pick_single_ycb.py).
# Distinct from PickCube: adds an ``is_obj_placed·is_grasped`` bonus, max 6.
# Uses a primitive object substitute (YCB meshes are usd-only / asset-heavy;
# the reward uses only the object's *position*, so the formula stays 1:1).
# ---------------------------------------------------------------------------

_YCB_GOAL_THRESH = 0.025
_YCB_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.pick_single_ycb_dense", "pick_single_ycb_dense")
class PickSingleYcbDenseTask(DenseRLResetMixin, _PickSingleYcbBaseTask):
    """PickSingleYCB with ManiSkill3's dense reward ported 1:1 (max 6).

    Reward::

        reaching = 1 - tanh(5·||tcp - obj||)
        reward   = reaching + is_grasped
        place    = 1 - tanh(5·||goal - obj||)
        reward  += place·is_grasped
        reward  += is_obj_placed·is_grasped            # PickSingleYCB-specific
        static   = 1 - tanh(5·||arm_qvel||)
        reward  += static·is_obj_placed·is_grasped
        reward[success] = 6

    is_obj_placed = ||obj-goal|| <= goal_thresh (0.025).
    success = placed & robot_static & grasped. Primitive object + goal sphere,
    randomized resets, no demo data. ``is_grasped`` approximated.
    """

    cube_z = 0.02
    goal_z_range = (0.10, 0.30)
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="obj",
                size=(0.04, 0.04, 0.04),
                mass=0.064,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.5, 0.0),
            ),
            PrimitiveSphereCfg(name="goal_site", radius=0.025, physics=PhysicStateType.XFORM, color=(0.0, 1.0, 0.0)),
        ],
        robots=["franka"],
    )
    # inherits base PositionShiftChecker(obj_name="obj") for termination — "obj"
    # exists in this scenario, so reset()/_terminated work; dense reward is separate.

    def _reward(self, env_states):
        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        obj = objs["obj"].root_state[:, :3]
        goal = objs["goal_site"].root_state[:, :3]

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_YCB_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        tcp_to_obj = torch.linalg.norm(obj - tcp, dim=-1)
        reaching = 1.0 - torch.tanh(5.0 * tcp_to_obj)
        gripper_w = rs.joint_pos[:, 0:2].sum(dim=-1)
        is_grasped = ((tcp_to_obj < 0.03) & (gripper_w < 0.07)).float()
        reward = reaching + is_grasped

        obj_to_goal = torch.linalg.norm(goal - obj, dim=-1)
        place = 1.0 - torch.tanh(5.0 * obj_to_goal)
        reward = reward + place * is_grasped

        is_obj_placed = (obj_to_goal <= _YCB_GOAL_THRESH).float()
        reward = reward + is_obj_placed * is_grasped

        arm_qvel = rs.joint_vel[:, 2:9]
        static = 1.0 - torch.tanh(5.0 * torch.linalg.norm(arm_qvel, dim=-1))
        reward = reward + static * is_obj_placed * is_grasped

        is_static = (arm_qvel.abs().amax(dim=-1) < 0.2).float()
        success = (is_obj_placed * is_static * is_grasped).bool()
        reward = torch.where(success, torch.full_like(reward, 6.0), reward)
        return reward
