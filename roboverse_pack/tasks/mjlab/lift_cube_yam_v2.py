"""Minimal YAM lift_cube scaffold on ManagerBasedRVEnv (Phase 4 / 12-task).

Mirrors the cartpole / go1 / g1 scaffolds for the i2rt YAM 6-DOF
fixed-base arm. Scope is intentionally minimal — proves the scene-MJCF
asset path works for a fixed-base manipulator. Full mjlab manipulation
parity (cube + reach + grasp + lift rewards + vision obs + curriculum)
is deferred to per-task follow-up.

Registered tasks (all share the same YAM scaffold; per-task differences
live in the reward / obs set which will be ported individually):
  mjlab.lift_cube_yam_v2
  mjlab.lift_cube_yam_depth_v2
  mjlab.lift_cube_yam_rgb_v2
  mjlab.multi_cube_seg_yam_v2

Currently all four point at the same joint-hold scaffold so the registry
exposes all 12 mjlab task names and the discovery pipeline + ManagerBasedRVEnv
contract are validated on a fixed-base arm.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.task.registry import register_task
from metasim.utils import configclass
from roboverse_learn.managers import (
    CurriculumTerm,
    DoneTerm,
    EventTerm,
    ManagerBasedRVEnv,
    ManagerBasedRVEnvCfg,
    ObsTerm,
    RewTerm,
)

from ._locator import lazy_scenario, mjlab_asset
from ._mjcf_patch import YAM_KP, patch_mjcf_add_cube_and_table, patch_mjcf_with_pd_actuators
from .mdp import (
    SceneEntityCfg,
)
from .mdp import (
    observations as obs,
)
from .mdp import (
    rewards as rew,
)
from .mdp import (
    terminations as term,
)
from .mdp.commands import (
    LiftingCommandCfg,
    LiftingCommandManager,
    ObjectPoseRange,
    TargetPositionRange,
)
from .mdp.curriculums import reward_curriculum


def _yam_with_pd_and_cube(yam_xml_path: str) -> str:
    """Apply both PD actuator injection AND cube/table injection to YAM."""
    pd_xml = patch_mjcf_with_pd_actuators(yam_xml_path, YAM_KP)
    return patch_mjcf_add_cube_and_table(pd_xml)


def _yam_newton_objects() -> list:
    """Cube + static table as scene objects for the Newton (RobotCfg) path.

    Mirrors ``patch_mjcf_add_cube_and_table`` geometry: a 5cm dynamic cube
    on top of a 0.4x0.4x0.04 m static table at (0.3, 0, 0). The mujoco path
    bakes these into the MJCF; Newton adds them as MetaSim scene objects.
    """
    from metasim.constants import PhysicStateType
    from metasim.scenario.objects import PrimitiveCubeCfg

    table = PrimitiveCubeCfg(
        name="table",
        size=[0.4, 0.4, 0.04],
        color=[0.6, 0.4, 0.2],
        default_position=(0.3, 0.0, 0.02),
        physics=PhysicStateType.GEOM,
    )
    cube = PrimitiveCubeCfg(
        name="cube",
        size=[0.05, 0.05, 0.05],
        color=[1.0, 0.3, 0.1],
        default_position=(0.3, 0.0, 0.065),
        physics=PhysicStateType.RIGIDBODY,
    )
    return [table, cube]


_YAM_XML = "asset_zoo/robots/i2rt_yam/xmls/yam.xml"
_YAM_JOINTS_NAMES: tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
_YAM_JOINTS = SceneEntityCfg("yam", joint_names=_YAM_JOINTS_NAMES)


# ---------------------------------------------------------------------------
# scaffold rewards / events
# ---------------------------------------------------------------------------


def _get_cube_pos(env) -> np.ndarray:
    """World-frame position of injected cube body."""
    m = env.handler.physics.model
    mp = m.ptr if hasattr(m, "ptr") else m
    cube_id = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return np.asarray(env.handler.physics.data.xpos[cube_id], dtype=np.float32)


def _get_ee_pos(env) -> np.ndarray:
    """World-frame position of YAM end-effector via tcp_site."""
    m = env.handler.physics.model
    mp = m.ptr if hasattr(m, "ptr") else m
    site_id = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    return np.asarray(env.handler.physics.data.site_xpos[site_id], dtype=np.float32)


def reward_ee_to_cube(env, env_states, *, std: float = 0.3) -> torch.Tensor:
    """Reach reward exp(-||ee - cube|| / std).

    Gentler than squared (std=0.1 earlier gave virtually 0 gradient at start
    distance 0.22m). Use std=0.3 so even from 0.3m there's meaningful gradient
    signal.
    """
    cube_p = _get_cube_pos(env)
    ee_p = _get_ee_pos(env)
    dist = float(np.linalg.norm(cube_p - ee_p))
    return torch.full((env.num_envs,), float(np.exp(-dist / std)), device=env.device)


def reward_cube_lifted(env, env_states, *, target_height: float = 0.15) -> torch.Tensor:
    """Lift reward: 1 if cube_z > target_height (above table), 0 else."""
    cube_p = _get_cube_pos(env)
    return torch.full((env.num_envs,), float(cube_p[2] > target_height), device=env.device)


def reward_hold_pose(env, env_states, *, target_norm: float = 0.0) -> torch.Tensor:
    """Reward for keeping joint positions near target_norm (action regularizer)."""
    qpos = np.asarray(env.handler.physics.data.qpos[:6], dtype=np.float32)
    err = float(np.sum((qpos - target_norm) ** 2))
    return torch.full((env.num_envs,), float(np.exp(-err / 1.0)), device=env.device)


def reset_yam_with_cube(env, env_ids: torch.Tensor, *, joint_noise: float = 0.05) -> None:
    """Reset YAM joints (6 hinge + 2 finger) + cube freejoint to random pose.

    qpos layout: [joint1..joint6 (6), left_finger (1), right_finger (1), cube_pos(3), cube_quat_wxyz(4)] = 15
    """
    if not hasattr(env.handler, "physics"):
        return  # Newton path: handler default init
    physics = env.handler.physics
    rng = np.random.default_rng()
    with physics.reset_context():
        physics.data.qpos[:6] = rng.uniform(-joint_noise, joint_noise, size=6)
        physics.data.qpos[6] = 0.0  # left_finger neutral
        physics.data.qpos[7] = 0.0  # right_finger neutral
        # Cube freejoint qpos[8:15] = pos(3) + quat_wxyz(4)
        physics.data.qpos[8] = rng.uniform(0.25, 0.40)  # cube x
        physics.data.qpos[9] = rng.uniform(-0.15, 0.15)  # cube y
        physics.data.qpos[10] = 0.045  # cube z (just above table top)
        physics.data.qpos[11:15] = (1, 0, 0, 0)  # identity quat
        physics.data.qvel[:] = 0.0


# Backward-compat alias for the old smoke
reset_yam_default_pose = reset_yam_with_cube


# ---------------------------------------------------------------------------
# manager configs
# ---------------------------------------------------------------------------


@configclass
class _YamObsCfg:
    @configclass
    class ActorCfg:
        joint_pos = ObsTerm(func=obs.joint_pos_rel, params={"asset_cfg": _YAM_JOINTS})
        joint_vel = ObsTerm(func=obs.joint_vel_rel, params={"asset_cfg": _YAM_JOINTS})
        # mjlab lift_cube obs order: joint_pos, joint_vel, ee_to_cube,
        # cube_to_goal, actions. Without the cube/goal terms the policy could
        # not see the object or the target (a real 1:1 gap vs mjlab).
        ee_to_cube = ObsTerm(func=obs.ee_to_object_distance, params={"object_name": "cube", "site_name": "tcp_site"})
        cube_to_goal = ObsTerm(
            func=obs.object_to_goal_distance, params={"object_name": "cube", "command_name": "lift_height"}
        )
        last_action = ObsTerm(func=obs.last_action)

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


@configclass
class _YamRewardsCfg:
    """mjlab ``manipulation.lift_cube_env_cfg`` reward composition.

    Uses ported mjlab MDP rewards:
      lift            → ``staged_position_reward``
      lift_precise    → ``bring_object_reward``
      action_rate_l2  → ``rewards.action_rate_l2``
      joint_pos_limits→ ``rewards.joint_pos_limits``
      joint_vel_hinge → ``joint_velocity_hinge_penalty``

    All read the ``lift_height`` LiftingCommand on
    ``env.command_managers["lift_height"]``. Weights mirror mjlab defaults
    (``mjlab/tasks/manipulation/lift_cube_env_cfg.py:154-187``).

    Legacy scaffold rewards (``reward_ee_to_cube`` / ``reward_cube_lifted``)
    remain available for fallback / smoke-test but are not in the cfg.
    """

    lift = RewTerm(
        func=rew.staged_position_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "object_name": "cube",
            "reaching_std": 0.2,
            "bringing_std": 0.3,
            "site_name": "tcp_site",
        },
    )
    lift_precise = RewTerm(
        func=rew.bring_object_reward,
        weight=1.0,
        params={
            "command_name": "lift_height",
            "object_name": "cube",
            "std": 0.05,
        },
    )
    action_rate_l2 = RewTerm(func=rew.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(
        func=rew.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": _YAM_JOINTS},
    )
    joint_vel_hinge = RewTerm(
        func=rew.joint_velocity_hinge_penalty,
        weight=-0.01,
        params={"max_vel": 0.5, "asset_cfg": _YAM_JOINTS},
    )


@configclass
class _YamTerminationsCfg:
    time_out = DoneTerm(func=term.time_out, time_out=True)


@configclass
class _YamEventsCfg:
    reset_state = EventTerm(func=reset_yam_with_cube, mode="reset")


@configclass
class _YamCurriculumCfg:
    """mjlab ``manipulation.lift_cube_env_cfg.curriculum`` port.

    Mirrors the upstream ``joint_vel_hinge_weight`` schedule:
    ``-0.01 → -0.1 → -1.0`` at iters 0 / 500 / 1000 (each iter ≈ 24 steps).
    """

    joint_vel_hinge_weight = CurriculumTerm(
        func=reward_curriculum,
        params={
            "reward_name": "joint_vel_hinge",
            "stages": [
                {"step": 0, "weight": -0.01},
                {"step": 500 * 24, "weight": -0.1},
                {"step": 1000 * 24, "weight": -1.0},
            ],
        },
    )


def _yam_scenario() -> ScenarioCfg:
    # Inject PD position actuators (kp=50 hip joints, 30 wrist, 20 end-effector)
    # AND cube + table. Stable PD + reach reward gradient should allow real
    # manipulation learning.
    patched_xml = _yam_with_pd_and_cube(mjlab_asset(_YAM_XML))
    return ScenarioCfg(
        scene=SceneCfg(mjcf_path=patched_xml),
        robots=[],
        sim_params=SimParamCfg(dt=0.002),  # smaller dt for PD stability
        decimation=10,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )


@configclass
class LiftCubeYamEnvCfg(ManagerBasedRVEnvCfg):
    """Manager-based env config for the YAM arm lift-cube task."""

    decimation = 10  # 10 x 2ms = 20ms control_dt = 50Hz
    max_episode_length_s = 5.0
    is_finite_horizon = False
    observation_group_names = ("actor", "critic")
    observations = _YamObsCfg()
    rewards = _YamRewardsCfg()
    terminations = _YamTerminationsCfg()
    events = _YamEventsCfg()
    curriculum = _YamCurriculumCfg()


class _YamTaskBase(ManagerBasedRVEnv):
    supported_simulators = ("mujoco", "newton")
    scenario = lazy_scenario(_yam_scenario)
    _env_cfg_cls: type = LiftCubeYamEnvCfg  # subclasses override to swap cfg

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None) -> None:
        cfg = self._env_cfg_cls()
        sim = getattr(scenario, "simulator", None) if scenario else None
        if scenario is not None and sim == "mujoco":
            scenario.robots = []
        elif scenario is not None and sim != "mujoco" and not getattr(scenario, "objects", None):
            # Newton / non-mujoco RobotCfg path: the cube + table that the
            # mujoco scene-MJCF path injects via ``patch_mjcf_add_cube_and_table``
            # don't exist as MJCF bodies here, so add them as scene objects so
            # the lift rewards + LiftingCommand have a cube to track.
            scenario.objects = _yam_newton_objects()
        self.num_actions = len(_YAM_JOINTS_NAMES)
        self._actuated_qvel_ids: np.ndarray | None = None
        super().__init__(scenario=scenario or self.scenario, cfg=cfg, device=device)

        # Register the LiftingCommand for staged_position_reward + bring_object_reward.
        # mjlab uses ``commands.lift_height = LiftingCommandCfg(...)`` in
        # ``manipulation.lift_cube_env_cfg``. Sampler ranges mirror upstream:
        # object xy reset inside the table footprint, target above table.
        self.command_managers["lift_height"] = LiftingCommandManager(
            self,
            LiftingCommandCfg(
                entity_name="cube",
                resampling_time_range=(8.0, 12.0),
                difficulty="dynamic",
                success_threshold=0.05,
                object_pose_range=ObjectPoseRange(
                    x=(0.2, 0.4),
                    y=(-0.2, 0.2),
                    z=(0.04, 0.05),  # just above table top (table_size_z = 0.02)
                    yaw=(-3.14, 3.14),
                ),
                target_position_range=TargetPositionRange(
                    x=(0.2, 0.4),
                    y=(-0.2, 0.2),
                    z=(0.15, 0.35),  # lift target 15-35cm above table
                ),
            ),
        )

    def _get_initial_states(self) -> list[dict]:
        return [{"objects": {}, "robots": {}} for _ in range(self.scenario.num_envs)]

    def _resolve_actuated_indices(self) -> np.ndarray:
        if self._actuated_qvel_ids is not None:
            return self._actuated_qvel_ids
        m = self.handler.physics.model
        mp = m.ptr if hasattr(m, "ptr") else m
        ids = []
        for jname in _YAM_JOINTS_NAMES:
            jid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_JOINT, jname)
            ids.append(mp.jnt_dofadr[jid])
        self._actuated_qvel_ids = np.asarray(ids, dtype=np.int64)
        return self._actuated_qvel_ids

    _ACTION_SCALE: float = 0.3

    def _apply_action(self, processed_action: torch.Tensor) -> None:
        """PD joint-position target. Sim-aware mujoco vs newton."""
        if not hasattr(self.handler, "physics"):
            clipped = torch.clamp(processed_action, -3.0, 3.0)
            # Newton yam loads with 8 joints (6 arm + 2 gripper); pad 0 for gripper.
            joint_pos = self.handler.get_states(mode="tensor").robots["yam"].joint_pos
            full_width = joint_pos.shape[-1]
            n_envs = processed_action.shape[0]
            targets = torch.zeros(n_envs, full_width, device=processed_action.device, dtype=processed_action.dtype)
            targets[:, :6] = self._ACTION_SCALE * clipped
            self.handler.set_dof_targets(targets)
            return
        action_np = processed_action.detach().cpu().numpy().reshape(-1)
        action_np = np.clip(action_np, -3.0, 3.0)
        target = self._ACTION_SCALE * action_np
        self.handler.physics.data.ctrl[:6] = target


@register_task("mjlab.lift_cube_yam_v2")
class LiftCubeYamTask(_YamTaskBase):
    """YAM 6-DOF arm lift-cube task. State-only obs (mjlab parity, no camera).

    Reward = ``staged_position_reward + bring_object_reward - action_rate_l2
              - joint_pos_limits - joint_velocity_hinge_penalty``
    matching mjlab manipulation/lift_cube_env_cfg.py.
    """


# ---------------------------------------------------------------------------
# camera-augmented variants — depth / rgb / instance_seg
# ---------------------------------------------------------------------------


from metasim.scenario.cameras import PinholeCameraCfg


def _yam_wrist_camera(name: str, data_types: list[str]) -> PinholeCameraCfg:
    """Table-overhead camera for the YAM lift_cube scene.

    The cube scene loads as scene-MJCF (no registered robot), so MetaSim's
    ``mount_to`` body-mount path is unavailable here. Use a world-frame
    camera positioned above the table looking at the cube spawn area
    (~0.3 m forward of the arm base). For full wrist-mounted parity
    once a real YAM RobotCfg is wired, switch back to
    ``mount_to="yam"`` + ``mount_link="tcp_site"``.

    64x64 default -> small enough for fast smoke + flat obs concat;
    raise via PinholeCameraCfg overrides when training real policies.
    """
    # NB: do NOT use straight-down look-at — the MetaSim mujoco backend
    # computes the camera basis via cross(direction, world_up); if
    # ``direction`` is parallel to (0, 0, 1) the cross product is zero
    # and the resulting camera matrix has NaN values. Tilt the view ~30°
    # off-vertical by placing the camera up + forward of the cube.
    return PinholeCameraCfg(
        name=name,
        data_types=data_types,
        width=64,
        height=64,
        pos=(0.0, -0.4, 0.6),  # behind + above the arm base
        look_at=(0.3, 0.0, 0.05),  # cube spawn area
    )


@configclass
class _YamDepthObsCfg(_YamObsCfg):
    @configclass
    class ActorCfg(_YamObsCfg.ActorCfg):
        depth = ObsTerm(func=obs.camera_depth, params={"camera_name": "wrist_depth"})

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


@configclass
class LiftCubeYamDepthEnvCfg(LiftCubeYamEnvCfg):
    """Lift-cube env config with an added depth-camera observation."""

    observations = _YamDepthObsCfg()


@configclass
class _YamRgbObsCfg(_YamObsCfg):
    @configclass
    class ActorCfg(_YamObsCfg.ActorCfg):
        rgb = ObsTerm(func=obs.camera_rgb, params={"camera_name": "wrist_rgb"})

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


@configclass
class LiftCubeYamRgbEnvCfg(LiftCubeYamEnvCfg):
    """Lift-cube env config with an added RGB-camera observation."""

    observations = _YamRgbObsCfg()


@configclass
class _YamSegObsCfg(_YamObsCfg):
    @configclass
    class ActorCfg(_YamObsCfg.ActorCfg):
        seg = ObsTerm(func=obs.camera_instance_seg, params={"camera_name": "wrist_seg"})

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


@configclass
class MultiCubeSegYamEnvCfg(LiftCubeYamEnvCfg):
    """Lift-cube env config with an added instance-segmentation observation."""

    observations = _YamSegObsCfg()


@register_task("mjlab.lift_cube_yam_depth_v2")
class LiftCubeYamDepthTask(_YamTaskBase):
    """YAM lift-cube with overhead depth camera.

    Mjlab parity: mjlab.lift_cube_yam_depth — same state rewards as
    lift_cube_yam, obs adds a flattened depth tensor. Camera positioned
    above + behind the arm base looking at the cube spawn area; resolution
    64x64 (raise via PinholeCameraCfg overrides).

    NB: requires ``MUJOCO_GL=egl`` (or ``osmesa``) env var for headless
    rendering when no display is attached.
    """

    _env_cfg_cls = LiftCubeYamDepthEnvCfg

    def __init__(self, scenario: ScenarioCfg | None = None, device=None):
        scenario = scenario or _yam_scenario()
        scenario.cameras = list(getattr(scenario, "cameras", []) or []) + [
            _yam_wrist_camera("wrist_depth", ["depth"]),
        ]
        super().__init__(scenario=scenario, device=device)


@register_task("mjlab.lift_cube_yam_rgb_v2")
class LiftCubeYamRgbTask(_YamTaskBase):
    """YAM lift-cube with overhead RGB camera.

    Mjlab parity: mjlab.lift_cube_yam_rgb. Obs adds flattened RGB tensor
    (uint8 → float, /255).
    """

    _env_cfg_cls = LiftCubeYamRgbEnvCfg

    def __init__(self, scenario: ScenarioCfg | None = None, device=None):
        scenario = scenario or _yam_scenario()
        scenario.cameras = list(getattr(scenario, "cameras", []) or []) + [
            _yam_wrist_camera("wrist_rgb", ["rgb"]),
        ]
        super().__init__(scenario=scenario, device=device)


@register_task("mjlab.multi_cube_seg_yam_v2")
class MultiCubeSegYamTask(_YamTaskBase):
    """YAM lift-cube with instance-segmentation obs.

    Mjlab parity: mjlab.multi_cube_seg_yam. Obs adds flattened
    instance_seg tensor (int32 → float).
    """

    _env_cfg_cls = MultiCubeSegYamEnvCfg

    def __init__(self, scenario: ScenarioCfg | None = None, device=None):
        scenario = scenario or _yam_scenario()
        scenario.cameras = list(getattr(scenario, "cameras", []) or []) + [
            _yam_wrist_camera("wrist_seg", ["instance_seg"]),
        ]
        super().__init__(scenario=scenario, device=device)
