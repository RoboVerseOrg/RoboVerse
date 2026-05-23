"""Spec-compliant port of ManiSkill3's `PickCube-v1` task.

The existing `maniskill.pick_cube` (in `pick_cube.py`) is a
RoboVerse-flavored cube-lift task (4 cm cube, simple Δz checker, 250
max steps). It predates this integration push and we keep it as-is for
backward compatibility.

This module registers `maniskill.pick_cube_v1` — a closer match to the
canonical ManiSkill3 PickCube-v1 spec:

- 4 cm-overall cube (half-size 0.02) with `size=(0.04, 0.04, 0.04)`
- Goal: a 2.5 cm-radius sphere region above the cube
- Success: cube center within 2.5 cm of goal site
- Max episode steps: 50

What is *still* gapped vs canonical PickCube-v1 (out of scope for this
phase):

- Robot URDF is RoboVerse's stock Franka, not ManiSkill's
  `panda_v2.urdf` with finger friction 2.0
- Controller is per-joint position target, not `pd_joint_delta_pos`
- Reward is sparse (`is_detected`), not the dm_control-style
  `reaching + grasp + place·grasp + static·placed` formula
- Robot-static success criterion (`max|qvel[:-2]| < 0.2`) is not yet
  evaluated

These are tracked items 3/4/5/7/8 of the parity recipe in the
integration report:
<http://localhost:8000/#roboverse/maniskill_integration>.
"""

from __future__ import annotations

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import Relative3DSphereDetector
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask

# Match ManiSkill: half-size 0.02 → 4 cm cube; default density picks up
# ~0.064 kg with sapien's 1000 kg/m³ default — close to ManiSkill's
# implicit choice (they don't set mass either).
_CUBE = PrimitiveCubeCfg(
    name="cube",
    size=(0.04, 0.04, 0.04),
    physics=PhysicStateType.RIGIDBODY,
    color=(1.0, 0.0, 0.0),
)

# Goal site: kinematic sphere, no collision contribution to the
# physics step. PhysicStateType.XFORM = pure visual / pose-only.
_GOAL = PrimitiveSphereCfg(
    name="goal_site",
    radius=0.025,
    physics=PhysicStateType.XFORM,
    color=(0.0, 1.0, 0.0),
)

_DETECTOR = Relative3DSphereDetector(
    base_obj_name="goal_site",
    relative_pos=(0.0, 0.0, 0.0),
    radius=0.025,
)


@register_task("maniskill.pick_cube_v1", "pick_cube_v1")
class PickCubeV1Task(ManiskillBaseTask):
    """ManiSkill3 PickCube-v1 geometry + sparse success checker."""

    scenario = ScenarioCfg(
        objects=[_CUBE, _GOAL],
        robots=["franka"],
    )

    # We reuse the v2 PickCube demo data the existing task already
    # references — the demo trajectories ship a 4 cm cube too, so the
    # geometry change here is observation-compatible.
    traj_filepath = "roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz"

    max_episode_steps = 50  # canonical ManiSkill PickCube-v1 cap

    checker = DetectedChecker(
        obj_name="cube",
        detector=_DETECTOR,
    )


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 PickCube-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/pick_cube.py).
# ---------------------------------------------------------------------------

_GOAL_THRESH = 0.025  # ManiSkill PickCube goal_thresh
_TCP_OFFSET = (0.0, 0.0, 0.10312)  # franka panda_hand → tcp (curobo_tcp_rel_pos)


@register_task("maniskill.pick_cube_v1_dense", "pick_cube_v1_dense")
class PickCubeV1DenseTask(DenseRLResetMixin, PickCubeV1Task):
    """PickCube-v1 with ManiSkill3's dense reward ported 1:1.

    Uses randomized resets (cube on table, goal floating above) so it needs
    no demo-trajectory data — a self-sufficient RL task.

    Reward (un-normalized, max 5, matching ``compute_dense_reward``)::

        tcp_to_obj   = ||cube - tcp||
        reaching     = 1 - tanh(5·tcp_to_obj)
        reward       = reaching + is_grasped
        obj_to_goal  = ||goal - cube||
        place        = 1 - tanh(5·obj_to_goal)
        reward      += place · is_grasped
        static       = 1 - tanh(5·||arm_qvel||)
        reward      += static · is_obj_placed
        reward[success] = 5

    ``is_grasped`` is contact-based in ManiSkill (``agent.is_grasping``);
    here it is approximated by gripper closure + tcp-cube proximity (the
    cross-sim contact-grasp semantics differ — the reward *shape* is 1:1,
    the grasp indicator is the documented residual). Divide by 5 for the
    normalized variant.
    """

    cube_z = 0.02
    goal_z_range = (0.10, 0.30)  # PickCube goal floats above the table

    def _reward(self, env_states):
        import torch

        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]

        cube = objs["cube"].root_state[:, :3]
        goal = objs["goal_site"].root_state[:, :3]

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]  # wxyz
        offset = torch.tensor(_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        tcp_to_obj = torch.linalg.norm(cube - tcp, dim=-1)
        reaching = 1.0 - torch.tanh(5.0 * tcp_to_obj)
        reward = reaching

        finger_q = rs.joint_pos[:, 7:9]
        gripper_width = finger_q.sum(dim=-1)
        is_grasped = ((tcp_to_obj < 0.03) & (gripper_width < 0.07)).float()
        reward = reward + is_grasped

        obj_to_goal = torch.linalg.norm(goal - cube, dim=-1)
        place = 1.0 - torch.tanh(5.0 * obj_to_goal)
        reward = reward + place * is_grasped

        arm_qvel = rs.joint_vel[:, :7]
        static = 1.0 - torch.tanh(5.0 * torch.linalg.norm(arm_qvel, dim=-1))
        is_obj_placed = (obj_to_goal <= _GOAL_THRESH).float()
        reward = reward + static * is_obj_placed

        is_static = (arm_qvel.abs().amax(dim=-1) < 0.2).float()
        success = (is_obj_placed * is_static).bool()
        reward = torch.where(success, torch.full_like(reward, 5.0), reward)
        return reward
