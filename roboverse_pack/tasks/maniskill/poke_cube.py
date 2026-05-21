from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import Relative2DSphereDetector
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

import torch

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask

goal_region_size = 0.05  # The size of the goal region in meters
goal_region_visible_size = 0.05  # The visible size of the goal region in meters


@register_task("maniskill.poke_cube", "poke_cube")
class PokeCubeCfg(ManiskillBaseTask):
    """The PokeCube task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    PokeCube
    ### group:
    Maniskill
    ### description:
    A simple task where the objective is to poke a red cube with a peg and push it to a target goal position.
    ### randomizations:
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table
    - the cube's x-coordinate is fixed to peg's x-coordinate + peg half-length (0.12) + 0.1 and y-coordinate is randomized in range [-0.1, 0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized in range [- pi/6, pi/6]
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.05 + goal_radius, 0]
    ### success:
    - the cube's xy position is within goal_radius (default 0.05) of the target's xy position by euclidean distance
    - the robot is static
    ### badges:
    - dense
    - sparse
    - demo
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#pokecube-v1
    ### platforms:
    - mujoco
    - isaaclab
    ### video_url:
    poke_cube.mp4
    ### notes:
    """  # noqa: D205, D415

    scenario = ScenarioCfg(
        objects=[
            PrimitiveCylinderCfg(
                name="goal_region",
                radius=goal_region_visible_size,
                height=0.0001,
                color=[0.0, 0.0, 1.0],
                collision_enabled=False,
                physics=PhysicStateType.XFORM,
                fix_base_link=True,  # Fix the goal region to the ground
            ),
            PrimitiveCubeCfg(
                name="cube",
                size=[0.04, 0.04, 0.04],
                mass=0.02,
                physics=PhysicStateType.RIGIDBODY,
                color=[1.0, 0.0, 0.0],
            ),
            RigidObjCfg(
                name="peg",
                usd_path="roboverse_data/assets/maniskill/PokeCube/peg/usd/stick.usd",
                mjcf_path="roboverse_data/assets/maniskill/PokeCube/peg/mjcf/stick.xml",
                urdf_path="roboverse_data/assets/maniskill/PokeCube/peg/urdf/stick.urdf",
                physics=PhysicStateType.RIGIDBODY,
                fix_base_link=False,
            ),
        ]
    )
    episode_length = 250

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
    traj_filepath = "roboverse_data/trajs/maniskill/poke_cube/v2/franka_v2.pkl.gz"


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PokeCube-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/poke_cube.py).
# ---------------------------------------------------------------------------

_PK_CUBE_HALF = 0.02
_PK_PEG_HALF_LEN = 0.12
_PK_GOAL_RADIUS = 0.05
_PK_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.poke_cube_dense", "poke_cube_dense")
class PokeCubeDenseTask(DenseRLResetMixin, PokeCubeCfg):
    """PokeCube with ManiSkill3's dense reward ported 1:1.

    Staged reward (un-normalized, max 10)::

        reaching = 2·(1 - tanh(5·||tcp - peg||));  reached = ||tcp-peg||<0.01
        reward   = reaching
        peg_head = peg.p + [peg_half_len, 0, 0]    # ManiSkill world-frame offset
        angle    = |yaw(peg) - yaw(cube)|;  align = 1 - tanh(5·angle)
        h2c      = ||peg_head.xy - cube.xy||;  close = 1 - tanh(5·h2c)
        grasped  = is_grasping(peg) & reached
        reward[grasped]  = 4 + close + align
        place    = 1 - tanh(5·||goal - cube||)
        fit      = (angle<0.05) & (h2c <= cube_half+0.005) & grasped
        reward[fit]      = 7 + place
        static   = 1 - tanh(5·||arm_qvel||)
        reward[cube_placed] += static
        reward[success]  = 10  (cube_placed & robot_static)

    Uses a primitive box peg (file-based stick asset avoided; reward uses only
    peg pose). yaw via ``euler_xyz_from_quat[2]`` (matches ManiSkill euler-XYZ[2]).
    ``is_grasping`` approximated. Randomized resets, no demo data.
    """

    cube_z = _PK_CUBE_HALF
    goal_z_range = (0.0001, 0.0001)

    from metasim.scenario.objects import PrimitiveCubeCfg as _Cube
    from metasim.scenario.objects import PrimitiveCylinderCfg as _Cyl

    scenario = ScenarioCfg(
        objects=[
            _Cyl(name="goal_region", radius=0.05, height=0.0001, color=[0.0, 0.0, 1.0],
                 collision_enabled=False, physics=PhysicStateType.XFORM, fix_base_link=True),
            _Cube(name="cube", size=[0.04, 0.04, 0.04], mass=0.02,
                  physics=PhysicStateType.RIGIDBODY, color=[1.0, 0.0, 0.0]),
            _Cube(name="peg", size=[0.24, 0.05, 0.05], mass=0.05,
                  physics=PhysicStateType.RIGIDBODY, color=[0.5, 0.5, 0.5]),
        ],
        robots=["franka"],
    )

    def _reward(self, env_states):
        from metasim.utils.math import euler_xyz_from_quat, quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        peg = objs["peg"].root_state[:, :3]
        peg_q = objs["peg"].root_state[:, 3:7]
        cube = objs["cube"].root_state[:, :3]
        cube_q = objs["cube"].root_state[:, 3:7]
        goal = objs["goal_region"].root_state[:, :3]

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_PK_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        tcp_to_peg = torch.linalg.norm(tcp - peg, dim=-1)
        reached = tcp_to_peg < 0.01
        reward = 2.0 * (1.0 - torch.tanh(5.0 * tcp_to_peg))

        peg_head = peg.clone()
        peg_head[:, 0] = peg_head[:, 0] + _PK_PEG_HALF_LEN
        peg_yaw = euler_xyz_from_quat(peg_q)[2]
        cube_yaw = euler_xyz_from_quat(cube_q)[2]
        angle = torch.abs(peg_yaw - cube_yaw)
        align = 1.0 - torch.tanh(5.0 * angle)
        h2c = torch.linalg.norm(peg_head[:, :2] - cube[:, :2], dim=-1)
        close = 1.0 - torch.tanh(5.0 * h2c)

        gripper_w = rs.joint_pos[:, 7:9].sum(dim=-1)
        grasped = (tcp_to_peg < 0.04) & (gripper_w < 0.07) & reached
        reward = torch.where(grasped, 4.0 + close + align, reward)

        place = 1.0 - torch.tanh(5.0 * torch.linalg.norm(goal - cube, dim=-1))
        fit = (angle < 0.05) & (h2c <= _PK_CUBE_HALF + 0.005) & grasped
        reward = torch.where(fit, 7.0 + place, reward)

        arm_qvel = rs.joint_vel[:, :7]
        static = 1.0 - torch.tanh(5.0 * torch.linalg.norm(arm_qvel, dim=-1))
        cube_placed = torch.linalg.norm(cube[:, :2] - goal[:, :2], dim=-1) < _PK_GOAL_RADIUS
        reward = torch.where(cube_placed, reward + static, reward)

        robot_static = arm_qvel.abs().amax(dim=-1) < 0.2
        success = cube_placed & robot_static
        reward = torch.where(success, torch.full_like(reward, 10.0), reward)
        return reward
