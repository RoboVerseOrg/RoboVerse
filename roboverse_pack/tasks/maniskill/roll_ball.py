from __future__ import annotations

import torch

from metasim.constants import PhysicStateType
from metasim.example.example_pack.tasks.checkers.checkers import DetectedChecker
from metasim.example.example_pack.tasks.checkers.detectors import Relative2DSphereDetector
from metasim.scenario.objects import PrimitiveCylinderCfg, PrimitiveSphereCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task

from ._dense_rl import DenseRLResetMixin
from .maniskill_base import ManiskillBaseTask

goal_region_size = 0.2  # The size of the goal region in meters
goal_region_visible_size = 0.05  # The visible size of the goal region in meters


@register_task("maniskill.roll_ball", "roll_ball")
class RollBallCfg(ManiskillBaseTask):
    """The RollBall task from ManiSkill.
    .. Description:
    ### 📦 Source Metadata (from ManiSkill)
    ### title:
    RollBall
    ### group:
    Maniskill
    ### description:
    A simple task where the objective is to push and roll a ball to a goal region at the other end of the table
    ### randomizations:
    - The ball's xy position is randomized on top of a table in the region [0.2, 0.5] x [-0.4, 0.7]. It is placed flat on the table
    - The target goal region is marked by a red/white circular target. The position of the target is randomized on top of a table in the region [-0.4, -0.7] x [0.2, -0.9]
    ### success:
    - The ball's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
    ### badges:
    - dense
    - sparse
    - demo
    ### official_url:
    https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html#rollball-v1
    ### platforms:
    - mujoco
    - isaaclab
    ### video_url:
    roll_ball.mp4
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
                fix_base_link=True,  # Fix the goal region to the ground
            ),
            PrimitiveSphereCfg(
                name="ball",
                radius=0.05,
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
        obj_name="ball",
    )

    traj_filepath = "roboverse_data/trajs/maniskill/roll_ball/v2"


# ---------------------------------------------------------------------------
# Dense-reward variant — 1:1 port of ManiSkill3 RollBall-v1
# ``compute_dense_reward`` (mani_skill/envs/tasks/tabletop/roll_ball.py).
# Stateful: ``reached_status`` latches per-env once the gripper reaches the
# behind-ball "hit" pose, and persists until episode reset.
# ---------------------------------------------------------------------------

_RB_BALL_RADIUS = 0.05
_RB_GOAL_RADIUS = 0.1
_RB_TCP_OFFSET = (0.0, 0.0, 0.10312)


@register_task("maniskill.roll_ball_dense", "roll_ball_dense")
class RollBallDenseTask(DenseRLResetMixin, RollBallCfg):
    """RollBall with ManiSkill3's (stateful) dense reward ported 1:1.

    Reward (un-normalized, max 30)::

        unit_vec    = (ball - goal) / ||ball - goal||
        hit_pose    = ball + unit_vec·(ball_radius + 0.05)   # behind the ball
        reached_status[||hit_pose - tcp|| < 0.04] = 1   # latches, persists
        reaching    = 1 - tanh(2·||hit_pose - tcp||)
        reached_rew = 1 - tanh(||ball.xy - goal.xy||)
        reward = 20·reached_rew·status + reaching·(1-status) + status
        reward[success] = 30

    success = ball xy within goal_radius (0.1). Randomized resets, no demo data.
    """

    cube_z = _RB_BALL_RADIUS
    goal_z_range = (0.0001, 0.0001)
    scenario = RollBallCfg.scenario.update(robots=["franka"])

    def _ensure_status(self, n: int, device) -> None:
        if getattr(self, "_reached_status", None) is None or self._reached_status.shape[0] != n:
            self._reached_status = torch.zeros(n, device=device)

    def reset(self, states=None, env_ids=None):
        out = super().reset(states, env_ids)
        if getattr(self, "_reached_status", None) is not None:
            if env_ids is None:
                self._reached_status.zero_()
            else:
                self._reached_status[torch.as_tensor(env_ids, device=self._reached_status.device)] = 0.0
        return out

    def _reward(self, env_states):
        from metasim.utils.math import quat_rotate

        rs = env_states.robots["franka"]
        objs = env_states.objects
        device = rs.joint_pos.device
        n = rs.joint_pos.shape[0]
        self._ensure_status(n, device)

        ball = objs["ball"].root_state[:, :3]
        goal = objs["goal_region"].root_state[:, :3]

        diff = ball - goal
        unit = diff / torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-8)
        hit_pose = ball + unit * (_RB_BALL_RADIUS + 0.05)

        names = rs.body_names
        hidx = next((i for i, nm in enumerate(names) if nm.endswith("panda_hand")), None)
        if hidx is None:
            return torch.zeros(n, device=device)
        hand_pos = rs.body_state[:, hidx, :3]
        hand_quat = rs.body_state[:, hidx, 3:7]
        offset = torch.tensor(_RB_TCP_OFFSET, device=device, dtype=hand_pos.dtype).expand(n, 3)
        tcp = hand_pos + quat_rotate(hand_quat, offset)

        tcp_to_hit = torch.linalg.norm(hit_pose - tcp, dim=-1)
        self._reached_status[tcp_to_hit < 0.04] = 1.0
        status = self._reached_status
        reaching = 1.0 - torch.tanh(2.0 * tcp_to_hit)

        obj_to_goal = torch.linalg.norm(ball[:, :2] - goal[:, :2], dim=-1)
        reached_rew = 1.0 - torch.tanh(obj_to_goal)

        reward = 20.0 * reached_rew * status + reaching * (1.0 - status) + status
        success = obj_to_goal < _RB_GOAL_RADIUS
        reward = torch.where(success, torch.full_like(reward, 30.0), reward)
        return reward
