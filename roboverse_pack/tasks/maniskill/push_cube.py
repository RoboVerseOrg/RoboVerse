from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import Relative2DSphereDetector
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask

goal_region_size = 0.15  # The size of the goal region in meters
goal_region_visible_size = 0.05  # The visible size of the goal region in meters


@register_task("maniskill.push_cube", "push_cube")
class PushCubeCfg(ManiskillBaseTask):
    """The PushCube task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    PushCube
    ### group:
    Maniskill
    ### description:
    A simple task where the objective is to push and move a cube to a goal region in front of it
    ### randomizations:
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    ### success:
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]
    ### badges:
    - dense
    - sparse
    - demo
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#pushcube-v1
    ### platforms:
    - mujoco
    - isaaclab
    ### video_url:
    push_cube.mp4
    ### notes:
    """  # noqa: D205, D415

    episode_length = 500

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCylinderCfg(
                name="goal_region",
                radius=goal_region_visible_size,
                height=0.0001,
                color=[0.0, 0.0, 1.0],
                collision_enabled=False,
                physics=PhysicStateType.XFORM,
                fix_base_link=True,
            ),
            PrimitiveCubeCfg(
                name="cube",
                size=[0.04, 0.04, 0.04],
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=[1.0, 0.0, 0.0],
            ),
        ]
    )

    detector = Relative2DSphereDetector(
        base_obj_name="goal_region",
        relative_pos=(0.0, 0.0, 0.0),
        radius=goal_region_size,
        axis=(0, 1),
    )
    checker = DetectedChecker(
        detector=detector,
        obj_name="cube",
    )

    traj_filepath = "roboverse_data/trajs/maniskill/push_cube/v2"


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PushCube-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/push_cube.py).
# ---------------------------------------------------------------------------

_PC_CUBE_HALF = 0.02
_PC_GOAL_RADIUS = 0.1
_PC_TCP_OFFSET = (0.0, 0.0, 0.10312)  # franka panda_hand → tcp


@register_task("maniskill.push_cube_dense", "push_cube_dense")
class PushCubeDenseTask(DenseRLResetMixin, PushCubeCfg):
    """PushCube with ManiSkill3's dense reward ported 1:1.

    Randomized resets (cube + goal region on the table) — no demo data needed.

    Reward (un-normalized, max 4)::

        push_pose   = cube.p + [-cube_half_size-0.005, 0, 0]   # push from behind
        reaching    = 1 - tanh(5·||push_pose - tcp||)
        reward      = reaching
        reached     = ||push_pose - tcp|| < 0.01
        obj_to_goal = ||cube.xy - goal.xy||
        place       = 1 - tanh(5·obj_to_goal)
        reward     += place · reached
        z_reward    = 1 - tanh(5·|cube.z - cube_half_size|)
        reward     += place · z_reward · reached
        reward[success] = 4

    success = cube xy within goal_radius (0.1) of the goal region.
    TCP / qvel access identical to pick_cube_v1_dense.
    """

    cube_z = _PC_CUBE_HALF
    goal_z_range = (0.0001, 0.0001)  # flat goal region on the table

    # The replay-based PushCubeCfg gets its franka from the demo trajectory;
    # a standalone RL task must declare the robot in the scenario.
    scenario = PushCubeCfg.scenario.update(robots=["franka"])

    def _reward(self, env_states):
        import torch
        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        cube = objs["cube"].root_state[:, :3]
        goal = objs["goal_region"].root_state[:, :3]

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_PC_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        push_pose = cube.clone()
        push_pose[:, 0] = push_pose[:, 0] - _PC_CUBE_HALF - 0.005
        tcp_to_push = torch.linalg.norm(push_pose - tcp, dim=-1)
        reaching = 1.0 - torch.tanh(5.0 * tcp_to_push)
        reward = reaching

        reached = (tcp_to_push < 0.01).float()
        obj_to_goal = torch.linalg.norm(cube[:, :2] - goal[:, :2], dim=-1)
        place = 1.0 - torch.tanh(5.0 * obj_to_goal)
        reward = reward + place * reached

        z_dev = torch.abs(cube[:, 2] - _PC_CUBE_HALF)
        z_reward = 1.0 - torch.tanh(5.0 * z_dev)
        reward = reward + place * z_reward * reached

        success = (obj_to_goal < _PC_GOAL_RADIUS).bool()
        reward = torch.where(success, torch.full_like(reward, 4.0), reward)
        return reward
