"""Minimal G1 walking task scaffold on ManagerBasedRVEnv (Phase 4 / 12-task).

Mirrors ``velocity_go1_v2.py``'s template but for the Unitree G1 humanoid
(29 hinge joints + floating base). Like go1_v2, this is a *scaffold* —
purpose is to prove the manager-based pattern + scene-MJCF asset path
scales to a 29-DOF bipedal humanoid through the canonical RoboVerse RL
trainer. Full mjlab parity (PPO convergence to walking) is deferred to
the per-task tuning phase.

Registered tasks:
  mjlab.velocity_flat_g1_v2
  mjlab.velocity_rough_g1_v2  (same scaffold; rough terrain wiring deferred)
"""

from __future__ import annotations

import math

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

from ._locator import mjlab_asset
from ._mjcf_patch import G1_KP, patch_mjcf_with_pd_actuators
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
    MotionCommandCfg,
    MotionCommandManager,
    VelocityCommandCfg,
    VelocityCommandManager,
    VelocityCommandRanges,
)
from .mdp.curriculums import commands_vel
from .mdp.sensors import (
    BuiltinSensor,
    BuiltinSensorCfg,
    ContactSensor,
    ContactSensorCfg,
    TerrainHeightSensor,
    TerrainHeightSensorCfg,
)

# mjlab KNEES_BENT_KEYFRAME — the actual training-time init pose from mjlab
# (g1_constants.py). NOT the HOME pose. The HOME pose (hip_pitch=-0.1, knee=0.3)
# is dynamically unstable under PD and was the root cause of G1 falling every
# episode during 9h Newton training. Order matches _G1_ALL_JOINT_NAMES below.
_G1_DEFAULT_POSE_NP = np.array(
    [
        # left leg: hip_pitch=-0.312, hip_roll=0, hip_yaw=0, knee=0.669, ankle_pitch=-0.363, ankle_roll=0
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        # right leg — symmetric
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        # waist (3) — all zero
        0.0,
        0.0,
        0.0,
        # left arm — shoulder_pitch=0.2, shoulder_roll=0.2, elbow=0.6
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        # right arm — shoulder_pitch=0.2, shoulder_roll=-0.2, elbow=0.6
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

_G1_XML = "asset_zoo/robots/unitree_g1/xmls/g1.xml"

# All 29 actuated joints (excludes the floating-base freejoint).
_G1_ALL_JOINT_NAMES: tuple[str, ...] = (
    # left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_G1_JOINTS = SceneEntityCfg("g1", joint_names=_G1_ALL_JOINT_NAMES)
_G1_TRUNK = SceneEntityCfg("g1", body_names=("pelvis",))


def reward_forward_velocity_g1(env, env_states, *, target: float = 0.5) -> torch.Tensor:
    """Reward for matching commanded forward (x) base velocity."""
    qvel = np.asarray(env.handler.physics.data.qvel[0], dtype=np.float32)
    err_sq = float((qvel - target) ** 2)
    return torch.full((env.num_envs,), np.exp(-err_sq / 0.25), device=env.device)


def reset_g1_default_pose(
    env,
    env_ids: torch.Tensor,
    *,
    base_height: float = 0.76,  # mjlab KNEES_BENT_KEYFRAME pelvis z
    joint_noise: float = 0.05,
) -> None:
    """Drop the G1 in a near-default standing pose."""
    if not hasattr(env.handler, "physics"):
        return  # Newton path: handler default init state
    physics = env.handler.physics
    rng = np.random.default_rng()
    with physics.reset_context():
        physics.data.qpos[0:3] = (0.0, 0.0, base_height)
        physics.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        n_joints = len(_G1_ALL_JOINT_NAMES)
        physics.data.qpos[7 : 7 + n_joints] = _G1_DEFAULT_POSE_NP + rng.uniform(
            -joint_noise, joint_noise, size=n_joints
        )
        physics.data.qvel[:] = 0.0


@configclass
class _G1ObsCfg:
    """mjlab velocity actor obs (G1), 99D in mjlab ``velocity_env_cfg`` term order.

    Terms: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) +
    joint_pos(29) + joint_vel(29) + actions(29) + command(3) = 99D.
    (Tracking tasks inherit this velocity obs; their motion-command obs is a
    separate 1:1 follow-up.)
    """

    @configclass
    class ActorCfg:
        base_lin_vel = ObsTerm(func=obs.base_lin_vel, params={"asset_cfg": _G1_TRUNK})
        base_ang_vel = ObsTerm(func=obs.base_ang_vel, params={"asset_cfg": _G1_TRUNK})
        projected_gravity = ObsTerm(func=obs.projected_gravity, params={"asset_cfg": _G1_TRUNK})
        joint_pos = ObsTerm(
            func=obs.joint_pos_rel,
            params={"asset_cfg": _G1_JOINTS, "default": _G1_DEFAULT_POSE_NP.tolist()},
        )
        joint_vel = ObsTerm(func=obs.joint_vel_rel, params={"asset_cfg": _G1_JOINTS})
        last_action = ObsTerm(func=obs.last_action)
        command = ObsTerm(func=obs.generated_commands, params={"command_name": "twist"})

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


_G1_DEFAULT_POSE_T = torch.from_numpy(_G1_DEFAULT_POSE_NP.astype(np.float32)).unsqueeze(0)


@configclass
class _G1RewardsCfg:
    """mjlab ``velocity_env_cfg`` 14-term composition for G1.

    Sensor-dep terms (``angular_momentum``, ``air_time``,
    ``foot_clearance``, ``foot_swing_height``, ``foot_slip``,
    ``soft_landing``) are wired in at mjlab default weights but stubbed
    to return zero until sensor wiring lands; one-shot warnings ensure
    silent degradation is visible.

    Plus a non-mjlab ``alive`` survival bonus to compensate for early
    ``fell_over`` resets while PPO learns balance — removed once full
    sensor wiring lets the foot rewards carry the signal.
    """

    track_linear_velocity = RewTerm(
        func=rew.track_linear_velocity,
        weight=2.0,
        params={"asset_cfg": _G1_TRUNK, "command_name": "twist", "std": math.sqrt(0.25)},
    )
    track_angular_velocity = RewTerm(
        func=rew.track_angular_velocity,
        weight=2.0,
        params={"asset_cfg": _G1_TRUNK, "command_name": "twist", "std": math.sqrt(0.5)},
    )
    upright = RewTerm(
        func=rew.upright,
        weight=1.0,
        params={"asset_cfg": _G1_TRUNK, "std": math.sqrt(0.2)},
    )
    # alive: non-mjlab survival bonus until sensor wiring carries the signal.
    alive = RewTerm(
        func=lambda env, env_states: torch.ones(env.num_envs, device=env.device),
        weight=2.0,
    )
    pose = RewTerm(
        func=rew.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": _G1_JOINTS,
            "command_name": "twist",
            # mjlab G1 stance: legs tight, arms looser to allow swing.
            "std_standing": {
                ".*hip.*": 0.10,
                ".*knee.*": 0.15,
                ".*ankle.*": 0.20,
                ".*waist.*": 0.15,
                ".*shoulder.*": 0.20,
                ".*elbow.*": 0.25,
                ".*wrist.*": 0.30,
            },
            "std_walking": {
                ".*hip.*": 0.20,
                ".*knee.*": 0.30,
                ".*ankle.*": 0.30,
                ".*waist.*": 0.25,
                ".*shoulder.*": 0.40,
                ".*elbow.*": 0.50,
                ".*wrist.*": 0.60,
            },
            "std_running": {
                ".*hip.*": 0.30,
                ".*knee.*": 0.45,
                ".*ankle.*": 0.45,
                ".*waist.*": 0.40,
                ".*shoulder.*": 0.60,
                ".*elbow.*": 0.70,
                ".*wrist.*": 0.80,
            },
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
            "default_pose": _G1_DEFAULT_POSE_T,
        },
    )
    body_ang_vel = RewTerm(
        func=rew.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": _G1_TRUNK},
    )
    dof_pos_limits = RewTerm(
        func=rew.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": _G1_JOINTS},
    )
    action_rate_l2 = RewTerm(func=rew.action_rate_l2, weight=-0.1)

    # Sensor-dep — NOW FIRING with real sensors registered.
    # mjlab G1 weights from mjlab/tasks/velocity/config/g1/env_cfgs.py:
    #   angular_momentum: -0.02, air_time: 1.0
    angular_momentum = RewTerm(
        func=rew.angular_momentum_penalty,
        weight=-0.02,
        params={"sensor_name": "robot/root_angmom"},
    )
    air_time = RewTerm(
        func=rew.feet_air_time,
        weight=1.0,  # mjlab G1 default
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.05,
            "threshold_max": 0.5,
            "command_name": "twist",
            "command_threshold": 0.5,
        },
    )
    foot_clearance = RewTerm(
        func=rew.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    foot_swing_height = RewTerm(
        func=rew.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    foot_slip = RewTerm(
        func=rew.feet_slip,
        weight=-0.1,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.05},
    )
    soft_landing = RewTerm(
        func=rew.soft_landing,
        weight=-1e-5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.05},
    )


@configclass
class _G1TerminationsCfg:
    """mjlab parity terminations."""

    time_out = DoneTerm(func=term.time_out, time_out=True)
    fell_over = DoneTerm(
        func=term.bad_orientation,
        params={"asset_cfg": _G1_TRUNK, "limit_angle": 1.222},  # math.radians(70)
    )


@configclass
class _G1EventsCfg:
    reset_state = EventTerm(func=reset_g1_default_pose, mode="reset")


@configclass
class _G1CurriculumCfg:
    """mjlab ``velocity_env_cfg.curriculum`` port — command-vel stages."""

    command_vel = CurriculumTerm(
        func=commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
                {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
            ],
        },
    )


def _g1_scenario() -> ScenarioCfg:
    # Smaller dt + larger decimation for 29-DOF humanoid stability under explicit Euler
    patched_xml = patch_mjcf_with_pd_actuators(mjlab_asset(_G1_XML), G1_KP)
    return ScenarioCfg(
        scene=SceneCfg(mjcf_path=patched_xml),
        robots=[],
        # mjlab parity:
        #   dt=0.005, decimation=4 (matches mjlab native; 5ms substep × 4 = 20ms control = 50Hz)
        #   newton_mujoco_iterations=10, ls_iterations=20 — mjlab's SolverMuJoCo config for
        #   stable stiff PD on 29-DOF humanoid. Without these, mujoco_warp default fewer
        #   iters → G1 pendulum-swings past 70° limit_angle in ~25 steps.
        sim_params=SimParamCfg(
            dt=0.005,
            newton_mujoco_iterations=10,
            newton_mujoco_ls_iterations=20,
        ),
        decimation=4,
        simulator="mujoco",
        num_envs=1,
        headless=True,
        add_default_ground=True,
    )


@configclass
class VelocityFlatG1EnvCfg(ManagerBasedRVEnvCfg):
    """Manager-based env config for G1 velocity tracking on flat ground."""

    decimation = 4  # matches scenario.decimation (mjlab parity: dt=0.005 x 4 = 20ms = 50Hz)
    max_episode_length_s = 20.0
    is_finite_horizon = False
    observation_group_names = ("actor", "critic")
    observations = _G1ObsCfg()
    rewards = _G1RewardsCfg()
    terminations = _G1TerminationsCfg()
    events = _G1EventsCfg()
    curriculum = _G1CurriculumCfg()


class _G1TaskBase(ManagerBasedRVEnv):
    """Shared scaffold for all G1 velocity variants (flat / rough)."""

    scenario = _g1_scenario()

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None) -> None:
        cfg = VelocityFlatG1EnvCfg()
        sim = getattr(scenario, "simulator", None) if scenario else None
        if scenario is not None and sim == "mujoco":
            scenario.robots = []
        self.num_actions = len(_G1_ALL_JOINT_NAMES)
        self._actuated_qvel_ids: np.ndarray | None = None
        super().__init__(scenario=scenario or self.scenario, cfg=cfg, device=device)
        # mjlab parity: register velocity command manager "twist". track_*
        # rewards reference it via params["command_name"]="twist".
        self.command_managers["twist"] = VelocityCommandManager(
            self,
            VelocityCommandCfg(
                resampling_time_range=(3.0, 8.0),
                rel_standing_envs=0.1,
                rel_heading_envs=0.3,
                heading_command=True,
                heading_control_stiffness=0.5,
                ranges=VelocityCommandRanges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-0.5, 0.5),
                ),
            ),
        )

        # G1 sensors. ContactSensor works on both backends (Newton per-env
        # contact reader); TerrainHeightSensor + BuiltinSensor stay MuJoCo-only.
        _G1_FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
        self._mjlab_sensors["feet_ground_contact"] = ContactSensor(
            self,
            ContactSensorCfg(
                name="feet_ground_contact",
                primary_bodies=_G1_FEET_BODIES,
                secondary_body=None,
                fields=("found", "force"),
                track_air_time=True,
                history_length=4,
            ),
        )
        self._mjlab_sensors["foot_height_scan"] = TerrainHeightSensor(
            self,
            TerrainHeightSensorCfg(
                name="foot_height_scan",
                primary_bodies=_G1_FEET_BODIES,
                max_distance=1.0,
                geom_groups=(0,),
                target_height=0.0,
            ),
        )
        # BuiltinSensor (subtree_angmom) works on both backends (Newton reads
        # mujoco_warp's batched subtree_angmom). Completes 6/6 sensor rewards.
        self._mjlab_sensors["robot/root_angmom"] = BuiltinSensor(
            self,
            BuiltinSensorCfg(
                name="robot/root_angmom",
                field="subtree_angmom",
                body_name="pelvis",
            ),
        )

    def _get_initial_states(self) -> list[dict]:
        if not self.scenario.robots:
            return [{"objects": {}, "robots": {}} for _ in range(self.scenario.num_envs)]
        robot = self.scenario.robots[0]
        pos_t = torch.tensor(list(robot.default_pos), dtype=torch.float32)
        rot_t = torch.tensor(list(robot.default_rot), dtype=torch.float32)
        defaults = dict(robot.default_joint_positions)
        return [
            {
                "objects": {},
                "robots": {
                    robot.name: {
                        "pos": pos_t.clone(),
                        "rot": rot_t.clone(),
                        "dof_pos": dict(defaults),
                        "dof_vel": {k: 0.0 for k in defaults},
                    }
                },
            }
            for _ in range(self.scenario.num_envs)
        ]

    def _resolve_actuated_indices(self) -> np.ndarray:
        if self._actuated_qvel_ids is not None:
            return self._actuated_qvel_ids
        m = self.handler.physics.model
        mp = m.ptr if hasattr(m, "ptr") else m
        ids = []
        for jname in _G1_ALL_JOINT_NAMES:
            jid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_JOINT, jname)
            ids.append(mp.jnt_dofadr[jid])
        self._actuated_qvel_ids = np.asarray(ids, dtype=np.int64)
        return self._actuated_qvel_ids

    _ACTION_CLIP: float = 100.0  # generous, action scale handled per-joint below

    @property
    def _action_scale_t(self) -> torch.Tensor:
        """Per-joint action scale = 0.25 * effort_limit / stiffness (mjlab convention).

        Cached as float32 tensor on env.device after first call. Pulled from
        ``roboverse_pack.robots.mjlab_g1_cfg.G1_ACTION_SCALE``.
        """
        if not hasattr(self, "_action_scale_cache"):
            from roboverse_pack.robots.mjlab_g1_cfg import G1_ACTION_SCALE

            scales = [G1_ACTION_SCALE[j] for j in _G1_ALL_JOINT_NAMES]
            self._action_scale_cache = torch.tensor(scales, dtype=torch.float32, device=self.device)
        return self._action_scale_cache

    def _apply_action(self, processed_action: torch.Tensor) -> None:
        """PD joint-position target. Sim-aware (mujoco scene-MJCF vs newton GPU).

        Uses per-joint action scale matching mjlab (0.25 * effort / stiffness):
          - hip_pitch / hip_yaw / waist_yaw: 0.55
          - hip_roll / knee:                 0.35
          - ankle / waist_pitch_roll:        0.44
          - shoulder / elbow / wrist_roll:   0.44
          - wrist_pitch / wrist_yaw:         0.07
        """
        scale = self._action_scale_t  # (29,)
        if not hasattr(self.handler, "physics"):
            default = torch.from_numpy(_G1_DEFAULT_POSE_NP).to(processed_action.device, processed_action.dtype)
            clipped = torch.clamp(processed_action, -self._ACTION_CLIP, self._ACTION_CLIP)
            # Per-joint scale broadcast: (1, 29) * (N, 29)
            targets = default.unsqueeze(0) + scale.unsqueeze(0) * clipped
            self.handler.set_dof_targets(targets)
            return
        action_np = processed_action.detach().cpu().numpy().reshape(-1)
        action_np = np.clip(action_np, -self._ACTION_CLIP, self._ACTION_CLIP)
        scale_np = scale.detach().cpu().numpy()
        target = _G1_DEFAULT_POSE_NP + scale_np * action_np
        self.handler.physics.data.ctrl[:29] = target


@register_task("mjlab.velocity_flat_g1_v2")
class VelocityFlatG1Task(_G1TaskBase):
    """Bipedal G1 forward-velocity walking scaffold (flat ground)."""


@register_task("mjlab.velocity_rough_g1_v2")
class VelocityRoughG1Task(_G1TaskBase):
    """Same scaffold as flat. Rough terrain wiring deferred (needs height-field MJCF)."""


# Bodies tracked by mjlab tracking_flat_g1 — the G1 humanoid's principal
# kinematic chain. Matches mjlab/tasks/tracking/config/g1/tracking_flat.py
# body_names. Used by MotionCommand for relative-body anchor errors.
_G1_TRACKING_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_elbow_link",
)


@configclass
class _G1TrackingRewardsCfg(_G1RewardsCfg):
    """Tracking-flat-G1 reward set.

    Base G1 velocity rewards + 6 motion-tracking anchor/body error terms
    (mjlab tracking_env_cfg.rewards).
    """

    track_anchor_pos = RewTerm(
        func=rew.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    track_anchor_orient = RewTerm(
        func=rew.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    track_body_pos = RewTerm(
        func=rew.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    track_body_orient = RewTerm(
        func=rew.motion_relative_body_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    track_body_lin_vel = RewTerm(
        func=rew.motion_global_body_linear_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0},
    )
    track_body_ang_vel = RewTerm(
        func=rew.motion_global_body_angular_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0},
    )


@configclass
class TrackingFlatG1EnvCfg(VelocityFlatG1EnvCfg):
    """Tracking-flat-G1 cfg = velocity cfg with rewards swapped to tracking."""

    rewards = _G1TrackingRewardsCfg()


class _G1TrackingTaskBase(_G1TaskBase):
    """G1 motion-tracking scaffold — same dynamics base, swaps rewards + commands.

    Mirrors ``mjlab/tasks/tracking/config/g1/tracking_flat.py``:
      - registers a ``motion`` MotionCommand (npz motion file path comes
        from cfg; falls back to identity-tracking degenerate mode if
        ``motion_file=None``)
      - registers all 6 motion_*_error_exp rewards
      - keeps the velocity command + 14-term base reward composition (mjlab
        tracking_flat_g1 actually drops several velocity rewards; future
        refinement)

    With ``motion_file=None`` the tracking rewards saturate at 1.0 (target
    = robot state); supply ``motion_file=/path/to/clip.npz`` to actually
    track a reference clip.
    """

    motion_file: str | None = None  # subclass / cfg can override
    cfg: TrackingFlatG1EnvCfg

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None) -> None:
        super().__init__(scenario=scenario, device=device)
        # Swap to tracking cfg so the manager loop picks up the new reward terms.
        self.cfg = TrackingFlatG1EnvCfg()
        # Reinitialize episode_reward buffers for the new term names.
        self._init_buffers()

        # Register MotionCommand. body_names map to robot's body name list;
        # mjlab tracking_flat_g1 uses pelvis as anchor.
        self.command_managers["motion"] = MotionCommandManager(
            self,
            MotionCommandCfg(
                entity_name="g1",
                anchor_body_name="pelvis",
                body_names=_G1_TRACKING_BODY_NAMES,
                motion_file=self.motion_file,
            ),
        )


import os as _os


@register_task("mjlab.tracking_flat_g1_v2")
class TrackingFlatG1Task(_G1TrackingTaskBase):
    """G1 motion-tracking on flat ground, using MotionCommand + 6 motion_* rewards.

    Motion file is selected by env var ``MJLAB_G1_MOTION_FILE`` (path to
    npz with mjlab MotionLoader schema). If the env var is unset and
    ``/tmp/g1_synth_motion.npz`` exists, that's used (regenerate via
    ``python /tmp/generate_g1_motion.py``). If neither is available the
    command falls back to identity-tracking degenerate mode.
    """

    @property
    def motion_file(self) -> str | None:  # type: ignore[override]
        """Resolve the motion npz path from env var, then the /tmp fallback, else None."""
        env_path = _os.environ.get("MJLAB_G1_MOTION_FILE")
        if env_path and _os.path.exists(env_path):
            return env_path
        default = "/tmp/g1_synth_motion.npz"
        if _os.path.exists(default):
            return default
        return None


@register_task("mjlab.tracking_flat_g1_no_state_est_v2")
class TrackingFlatG1NoStateEstTask(_G1TrackingTaskBase):
    """``tracking_flat_g1_v2`` no-state-est variant.

    Drops base lin/ang velocity from actor obs (matches mjlab's asymmetric
    observation setup). Currently inherits the symmetric obs from the base -
    this matches the existing observation_group_names structure; mjlab's
    actor-side noise on state-est is the next refinement.
    """
