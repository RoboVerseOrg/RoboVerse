from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import Relative2DSphereDetector
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask

goal_region_size = 0.1  # The size of the goal region in meters
goal_region_visible_size = 0.05  # The visible size of the goal region in meters


@register_task("maniskill.pull_cube", "pull_cube")
class PullCubeCfg(ManiskillBaseTask):
    """The PullCube task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    PullCube
    ### group:
    Maniskill
    ### description:
    A simple task where the objective is to pull a cube onto a target.
    ### randomizations:
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1].
    - the target goal region is marked by a red and white target. The position of the target is fixed to be the cube's xy position - [0.1 + goal_radius, 0]
    ### success:
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance..
    ### badges:
    - dense
    - sparse
    - demo
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#pullcube-v1
    ### platforms:
    - mujoco
    - isaaclab
    ### video_url:
    pull_cube.mp4
    ### notes:
    """  # noqa: D205, D415

    episode_length = 250
    scenario = ScenarioCfg(
        objects=[
            PrimitiveCylinderCfg(
                name="goal_region",
                radius=goal_region_visible_size,
                height=0.0001,
                color=[0.0, 0.0, 1.0],
                collision_enabled=False,
                physics=PhysicStateType.XFORM,
                fix_base_link=True,  # Fix the goal region to the pullground
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

    traj_filepath = "roboverse_data/trajs/maniskill/pull_cube/v2"


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PullCube-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/pull_cube.py).
# ---------------------------------------------------------------------------

_PLC_CUBE_HALF = 0.02
_PLC_GOAL_RADIUS = 0.1
_PLC_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.pull_cube_dense", "pull_cube_dense")
class PullCubeDenseTask(DenseRLResetMixin, PullCubeCfg):
    """PullCube with ManiSkill3's dense reward ported 1:1.

    Reward (un-normalized, max 3)::

        pull_pose   = cube.p + [cube_half + 2·0.005, 0, 0]   # pull from in front
        reaching    = 1 - tanh(5·||pull_pose - tcp||)
        reward      = reaching
        reached     = ||pull_pose - tcp|| < 0.01
        obj_to_goal = ||cube.xy - goal.xy||
        place       = 1 - tanh(5·obj_to_goal)
        reward     += place · reached
        reward[success] = 3

    success = cube xy within goal_radius (0.1). Randomized resets, no demo data.
    """

    cube_z = _PLC_CUBE_HALF
    goal_z_range = (0.0001, 0.0001)
    scenario = PullCubeCfg.scenario.update(robots=["franka"])

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
        offset = torch.tensor(_PLC_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        pull_pose = cube.clone()
        pull_pose[:, 0] = pull_pose[:, 0] + _PLC_CUBE_HALF + 2 * 0.005
        tcp_to_pull = torch.linalg.norm(pull_pose - tcp, dim=-1)
        reaching = 1.0 - torch.tanh(5.0 * tcp_to_pull)
        reward = reaching

        reached = (tcp_to_pull < 0.01).float()
        obj_to_goal = torch.linalg.norm(cube[:, :2] - goal[:, :2], dim=-1)
        place = 1.0 - torch.tanh(5.0 * obj_to_goal)
        reward = reward + place * reached

        success = (obj_to_goal < _PLC_GOAL_RADIUS).bool()
        reward = torch.where(success, torch.full_like(reward, 3.0), reward)
        return reward
