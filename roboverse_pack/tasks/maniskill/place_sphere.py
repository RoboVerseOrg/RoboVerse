from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import RelativeBboxDetector
from metasim.scenario.objects import PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

import torch

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask


@register_task("maniskill.place_sphere", "place_sphere")
class PlaceSphereTask(ManiskillBaseTask):
    """The place-sphere task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    PlaceSphere
    ### group:
    Maniskill
    ### description:
    Place the sphere into the shallow bin.
    ### randomizations:
    - The position of the bin and the sphere are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1], and the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]
    ### success:
    - The sphere is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    ### badges:
    - dense
    - sparse
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#placesphere-v1
    ### poster_url:
    (none).
    ---
    ### video_url:
    place_sphere.mp4
    ### platforms:
    - mujuco
    - isaaclab
    ### notes:
    (none)
    """  # noqa: D205

    scenario = ScenarioCfg(
        objects=[
            RigidObjCfg(
                name="bin",
                usd_path="roboverse_data/assets/maniskill/PlaceSphere/bin/usd/bin.usd",
                mjcf_path="roboverse_data/assets/maniskill/PlaceSphere/bin/mjcf/PlaceSphere_bin.xml",
                urdf_path="roboverse_data/assets/maniskill/PlaceSphere/bin/urdf/PlaceSphere_bin.urdf",
                physics=PhysicStateType.RIGIDBODY,
            ),
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.03,
                mass=0.001,
                physics=PhysicStateType.RIGIDBODY,
                color=[1.0, 0.0, 0.0],
            ),
        ]
    )

    episode_length = 250
    checker = DetectedChecker(
        obj_name="sphere",
        detector=RelativeBboxDetector(
            base_obj_name="bin",
            relative_pos=(0.0, 0.0, 0.04),
            relative_quat=(1.0, 0.0, 0.0, 0.0),
            checker_lower=(-0.02, -0.02, -0.02),
            checker_upper=(0.02, 0.02, 0.02),
            ignore_base_ori=True,
        ),
    )
    traj_filepath = "roboverse_data/trajs/maniskill/place_sphere/v2/franka_v2.pkl.gz"


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PlaceSphere-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/place_sphere.py).
# ---------------------------------------------------------------------------

_PS_RADIUS = 0.03
_PS_BLOCK_HALF0 = 0.0025          # short_side_half_size
_PS_BIN_TOP_DZ = _PS_BLOCK_HALF0 + _PS_RADIUS   # 0.0325
_PS_GRIPPER_WIDTH = 0.08
_PS_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.place_sphere_dense", "place_sphere_dense")
class PlaceSphereDenseTask(DenseRLResetMixin, PlaceSphereTask):
    """PlaceSphere with ManiSkill3's dense reward ported 1:1.

    Staged reward (un-normalized, max 13)::

        reward = 2·(1 - tanh(5·||tcp - sphere||))                 # reach
        place  = 1 - tanh(5·||bin_top - sphere||); bin_top=bin+(0,0,block_half0+r)
        reward[grasped]    = 4 + place
        ungrasp = Σ finger_q / 0.08  (=16 if not grasped)
        static  = 1 - tanh(10·|v| + |ω|)                          # sphere vel
        robot_static = (max|arm_qvel| < 0.2)
        reward[on_bin]     = 6 + (ungrasp + static + robot_static)/3
        reward[success]    = 13

    on_bin: ||Δxy||<0.005 & |Δz - (r+block_half0)|<0.005.
    success = on_bin & sphere-static & ~grasped. ``is_grasped`` approximated.
    Randomized resets, no demo data; bin asset auto-fetched from HF.
    """

    cube_z = _PS_RADIUS
    goal_z_range = (_PS_RADIUS, _PS_RADIUS)
    # The file-based bin asset's mjcf has a broken mesh path (a machine-specific
    # prefix leaked into bin.stl's reference → HF 404). The dense reward only
    # uses the bin's *position*, so substitute a primitive box bin: the reward
    # formula stays 1:1; only the bin's visual mesh differs (cosmetic for reward).
    from metasim.scenario.objects import PrimitiveCubeCfg as _Cube

    scenario = ScenarioCfg(
        objects=[
            _Cube(name="bin", size=(0.05, 0.05, 0.05), color=(0.4, 0.3, 0.2),
                  physics=PhysicStateType.GEOM),
            PrimitiveSphereCfg(name="sphere", radius=_PS_RADIUS, mass=0.001,
                               physics=PhysicStateType.RIGIDBODY, color=(1.0, 0.0, 0.0)),
        ],
        robots=["franka"],
    )

    def _reward(self, env_states):
        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        obj = objs["sphere"].root_state[:, :3]
        obj_v = objs["sphere"].root_state[:, 7:10]
        obj_w = objs["sphere"].root_state[:, 10:13]
        bin_top = objs["bin"].root_state[:, :3].clone()
        bin_top[:, 2] = bin_top[:, 2] + _PS_BIN_TOP_DZ

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_PS_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        o_to_tcp = torch.linalg.norm(tcp - obj, dim=-1)
        reward = 2.0 * (1.0 - torch.tanh(5.0 * o_to_tcp))

        o_to_bin = torch.linalg.norm(bin_top - obj, dim=-1)
        place = 1.0 - torch.tanh(5.0 * o_to_bin)
        gripper_w = rs.joint_pos[:, 7:9].sum(dim=-1)
        is_grasped = (o_to_tcp < 0.03) & (gripper_w < 0.07)
        reward = torch.where(is_grasped, 4.0 + place, reward)

        ungrasp = rs.joint_pos[:, 7:9].sum(dim=-1) / _PS_GRIPPER_WIDTH
        ungrasp = torch.where(is_grasped, ungrasp, torch.full_like(ungrasp, 16.0))
        static = 1.0 - torch.tanh(torch.linalg.norm(obj_v, dim=-1) * 10.0 + torch.linalg.norm(obj_w, dim=-1))
        arm_qvel = rs.joint_vel[:, :7]
        robot_static = (arm_qvel.abs().amax(dim=-1) < 0.2).float()

        off = obj - objs["bin"].root_state[:, :3]
        on_bin = (torch.linalg.norm(off[:, :2], dim=-1) < 0.005) & (torch.abs(off[:, 2] - _PS_BIN_TOP_DZ) < 0.005)
        reward = torch.where(on_bin, 6.0 + (ungrasp + static + robot_static) / 3.0, reward)

        obj_static = (torch.linalg.norm(obj_v, dim=-1) < 1e-2) & (torch.linalg.norm(obj_w, dim=-1) < 0.5)
        success = on_bin & obj_static & (~is_grasped)
        reward = torch.where(success, torch.full_like(reward, 13.0), reward)
        return reward
