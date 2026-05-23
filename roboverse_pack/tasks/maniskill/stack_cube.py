from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import RelativeBboxDetector
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask


# @register_task("maniskill.stack_cube", "stack_cube")
# registed in metasim/example
class StackCubeTask(ManiskillBaseTask):
    """Stack a red cube on top of a blue base cube and release it."""

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(1.0, 0.0, 0.0),
            ),
            PrimitiveCubeCfg(
                name="base",
                size=(0.04, 0.04, 0.04),
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=(0.0, 0.0, 1.0),
            ),
        ],
        robots=["franka"],
    )

    checker = DetectedChecker(
        obj_name="cube",
        detector=RelativeBboxDetector(
            base_obj_name="base",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
        ),
    )

    max_episode_steps = 250

    traj_filepath = "roboverse_data/trajs/maniskill/stack_cube/v2/franka_v2.pkl.gz"


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 StackCube-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/stack_cube.py).
# cubeA = "cube" (red, to stack), cubeB = "base" (blue, target).
# ---------------------------------------------------------------------------

_SC_HALF = 0.02
_SC_GRIPPER_WIDTH = 0.08  # ManiSkill: panda finger upper-limit (0.04) * 2
_SC_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.stack_cube_dense", "stack_cube_dense")
class StackCubeDenseTask(DenseRLResetMixin, StackCubeTask):
    """StackCube with ManiSkill3's dense reward ported 1:1.

    Staged reward (un-normalized, max 8)::

        reward = 2·(1 - tanh(5·||tcp - cubeA||))                    # reach cubeA
        place  = 1 - tanh(5·||goal - cubeA||); goal = cubeB.xy + (cubeB.z+2·half)
        reward[grasped]      = 4 + place
        ungrasp = Σ finger_q / gripper_width  (=1 if not grasped)
        static  = 1 - tanh(10·|v| + |ω|)        # cubeA lin/ang velocity
        reward[on_cubeB]     = 6 + (ungrasp + static)/2
        reward[success]      = 8

    is_cubeA_on_cubeB: xy within ||half[:2]||+0.005 & |Δz - 2·half|<0.005.
    success = on & cubeA-static & ~grasped. ``is_grasped`` approximated.
    Randomized resets (both cubes on table), no demo data.
    """

    cube_z = _SC_HALF
    goal_z_range = (_SC_HALF, _SC_HALF)  # no goal object; both cubes on table

    def _reward(self, env_states):
        import math

        import torch

        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        cubeA = objs["cube"].root_state[:, :3]
        cubeB = objs["base"].root_state[:, :3]
        cubeA_vel = objs["cube"].root_state[:, 7:10]
        cubeA_ang = objs["cube"].root_state[:, 10:13]

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_SC_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        a_to_tcp = torch.linalg.norm(tcp - cubeA, dim=-1)
        reward = 2.0 * (1.0 - torch.tanh(5.0 * a_to_tcp))

        goal_xyz = cubeB.clone()
        goal_xyz[:, 2] = cubeB[:, 2] + _SC_HALF * 2
        a_to_goal = torch.linalg.norm(goal_xyz - cubeA, dim=-1)
        place = 1.0 - torch.tanh(5.0 * a_to_goal)

        gripper_w = rs.joint_pos[:, 7:9].sum(dim=-1)
        is_grasped = (a_to_tcp < 0.03) & (gripper_w < 0.07)
        reward = torch.where(is_grasped, 4.0 + place, reward)

        ungrasp = rs.joint_pos[:, 7:9].sum(dim=-1) / _SC_GRIPPER_WIDTH
        ungrasp = torch.where(is_grasped, ungrasp, torch.ones_like(ungrasp))
        static = 1.0 - torch.tanh(torch.linalg.norm(cubeA_vel, dim=-1) * 10.0 + torch.linalg.norm(cubeA_ang, dim=-1))

        offset_ab = cubeA - cubeB
        xy_flag = torch.linalg.norm(offset_ab[:, :2], dim=-1) <= (math.hypot(_SC_HALF, _SC_HALF) + 0.005)
        z_flag = torch.abs(offset_ab[:, 2] - _SC_HALF * 2) <= 0.005
        is_on = xy_flag & z_flag
        reward = torch.where(is_on, 6.0 + (ungrasp + static) / 2.0, reward)

        is_static = (torch.linalg.norm(cubeA_vel, dim=-1) < 1e-2) & (torch.linalg.norm(cubeA_ang, dim=-1) < 0.5)
        success = is_on & is_static & (~is_grasped)
        reward = torch.where(success, torch.full_like(reward, 8.0), reward)
        return reward
