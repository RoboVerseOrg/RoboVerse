"""Minimal Go1 walking task scaffold on ManagerBasedRVEnv (Phase 3b smoke).

Purpose: demonstrate that the manager-based pattern scales beyond cartpole
to a 12-DOF quadruped with a floating base. Loads mjlab's go1.xml as
scene-MJCF (same trick as cartpole_v2.py) so no go1 RobotCfg wiring is
required for the smoke.

Scope (intentionally minimal):
  - Action: 12-D direct torque (via qfrc_applied — bypasses actuators since
    the raw mjlab MJCF doesn't define any; mjlab adds them at runtime via
    XmlActuatorCfg).
  - Observation (mjlab velocity 1:1, 48D): base_lin_vel(3) + base_ang_vel(3)
    + projected_gravity(3) + joint_pos(12) + joint_vel(12) + actions(12)
    + command(3). See ``_Go1ObsCfg``.
  - Reward: mjlab velocity_env_cfg reward set (track lin/ang vel, upright,
    posture, sensor-dependent feet terms).
  - Termination: time_out + bad_orientation (fell over).
  - Reset: drop the robot from default pose with small noise.

Full mjlab velocity_flat_go1 parity (14 reward terms / curriculum / DR /
14-reward composition / Newton GPU batching) is deferred — those depend
on a proper go1 RobotCfg (joint defaults, actuator gains, soft joint
limits) which is a separate task. See M-Plan task #62.
"""

from __future__ import annotations

import math
import os

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
from ._mjcf_patch import patch_mjcf_with_exact_spec
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
from .mdp.actions import JointPositionActionCfg, JointPositionActionManager
from .mdp.commands import (
    VelocityCommandCfg,
    VelocityCommandManager,
    VelocityCommandRanges,
)
from .mdp.curriculums import commands_vel, terrain_levels_vel
from .mdp.events_dr import (
    body_com_offset,
    encoder_bias,
    geom_friction,
    push_by_setting_velocity,
)
from .mdp.sensors import (
    BuiltinSensor,
    BuiltinSensorCfg,
    ContactSensor,
    ContactSensorCfg,
    HeightScanSensor,
    HeightScanSensorCfg,
    TerrainHeightSensor,
    TerrainHeightSensorCfg,
)
from .mdp.terrain import TerrainLevelsCfg, TerrainLevelsManager

_GO1_XML = "asset_zoo/robots/unitree_go1/xmls/go1.xml"

# Selector for all 12 actuated joints (regex). Floating-base freejoint is
# excluded — it's not control-able.
_GO1_JOINTS = SceneEntityCfg(
    "go1",
    joint_names=(
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ),
)
_GO1_TRUNK = SceneEntityCfg("go1", body_names=("trunk",))

# Go1 default standing pose — must match mjlab go1 ``default_joint_pos`` exactly
# (verified against mjlab native): hips splayed ±0.1 (FR/RR +0.1, FL/RL -0.1),
# thigh 0.9, calf -1.8. This default feeds the joint_pos obs (default-relative),
# the reset pose, AND the action PD offset, so it must be 1:1.
import torch as _torch

_GO1_DEFAULT_POSE = _torch.tensor([
    0.1,
    0.9,
    -1.8,  # FR
    -0.1,
    0.9,
    -1.8,  # FL
    0.1,
    0.9,
    -1.8,  # RR
    -0.1,
    0.9,
    -1.8,  # RL
])
# Numpy mirror used in the PD action-application path
import numpy as _np

_GO1_DEFAULT_POSE_NP = _np.array(
    [
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
        0.1,
        0.9,
        -1.8,
        -0.1,
        0.9,
        -1.8,
    ],
    dtype=_np.float64,
)

# Per-joint action scale = 0.25 * effort_limit / stiffness (mjlab go1_constants.py
# GO1_ACTION_SCALE). hip+thigh share the 6:1-gear actuator (effort 23.7, kp ≈ 15.9
# -> 0.3728); calf is the 9:1-gear actuator (effort 35.55, kp ≈ 35.8 -> 0.2485).
# These exact values are verified against mjlab's resolved scale dict.
_GO1_ACTION_SCALE: dict[str, float] = {
    "FR_hip_joint": 0.372753,
    "FR_thigh_joint": 0.372753,
    "FR_calf_joint": 0.248502,
    "FL_hip_joint": 0.372753,
    "FL_thigh_joint": 0.372753,
    "FL_calf_joint": 0.248502,
    "RR_hip_joint": 0.372753,
    "RR_thigh_joint": 0.372753,
    "RR_calf_joint": 0.248502,
    "RL_hip_joint": 0.372753,
    "RL_thigh_joint": 0.372753,
    "RL_calf_joint": 0.248502,
}


# ---------------------------------------------------------------------------
# task-local term funcs (rewards / events specific to this minimal scaffold)
# ---------------------------------------------------------------------------


def reward_forward_velocity(env, env_states, *, target: float = 1.0) -> torch.Tensor:
    """Reward for matching commanded forward (x) base velocity.

    Reads ``data.qvel[0]`` (world-frame x velocity of the free base) and
    returns ``exp(-(v - target)^2 / 0.5^2)``. Shape: ``(num_envs,)``.
    """
    qvel = np.asarray(env.handler.physics.data.qvel[0], dtype=np.float32)
    err_sq = float((qvel - target) ** 2)
    return torch.full((env.num_envs,), np.exp(-err_sq / 0.25), device=env.device)


def reset_go1_default_pose(
    env,
    env_ids: torch.Tensor,
    *,
    base_height: float = 0.278,  # mjlab INIT_STATE pelvis z (go1_constants.py)
    joint_noise: float = 0.05,
) -> None:
    """Drop the robot at a quadruped-friendly stance with small joint noise."""
    if not hasattr(env.handler, "physics"):
        # Newton path: rely on handler default init state. Match-bit per-env
        # randomization needs batched qpos write — deferred.
        return
    physics = env.handler.physics
    rng = np.random.default_rng()
    with physics.reset_context():
        # Base pose: pos (3) + quat xyzw (4)
        physics.data.qpos[0:3] = (0.0, 0.0, base_height)
        physics.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # identity quat (wxyz in MuJoCo)
        # Joint defaults — canonical mjlab go1 stance (hips ±0.1, thigh 0.9, calf -1.8).
        defaults = _GO1_DEFAULT_POSE_NP.copy()
        defaults += rng.uniform(-joint_noise, joint_noise, size=12)
        physics.data.qpos[7:19] = defaults
        physics.data.qvel[:] = 0.0


# ---------------------------------------------------------------------------
# manager configs
# ---------------------------------------------------------------------------


@configclass
class _Go1ObsCfg:
    """mjlab ``velocity_env_cfg`` actor obs, term-for-term in the same order.

    base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(12)
    + joint_vel(12) + actions(12) + command(3) = 48D. (height_scan is rough-only
    and wired via the rough subclass.) Previously this was 36D (joint state +
    action only), which omitted the command + base velocity the policy needs to
    track velocity - see [[mjlab-dual-path-newton]] "Go1 only learned to stand".
    """

    @configclass
    class ActorCfg:
        base_lin_vel = ObsTerm(func=obs.base_lin_vel, params={"asset_cfg": _GO1_TRUNK})
        base_ang_vel = ObsTerm(func=obs.base_ang_vel, params={"asset_cfg": _GO1_TRUNK})
        projected_gravity = ObsTerm(func=obs.projected_gravity, params={"asset_cfg": _GO1_TRUNK})
        joint_pos = ObsTerm(
            func=obs.joint_pos_rel,
            params={"asset_cfg": _GO1_JOINTS, "default": _GO1_DEFAULT_POSE_NP.tolist()},
        )
        joint_vel = ObsTerm(func=obs.joint_vel_rel, params={"asset_cfg": _GO1_JOINTS})
        last_action = ObsTerm(func=obs.last_action)
        command = ObsTerm(func=obs.generated_commands, params={"command_name": "twist"})

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


# mjlab terrain_scan: GridPatternCfg(size=(1.6, 1.0), resolution=0.1), yaw-aligned,
# max_distance=5.0 (velocity_env_cfg.py:44-54). height_scan obs scale =
# 1/max_distance = 0.2 (velocity_env_cfg.py:204).
_GO1_HEIGHT_SCAN_SCALE = 1.0 / 5.0


@configclass
class _Go1RoughObsCfg:
    """mjlab ROUGH actor obs = flat 48-D proprio + ``height_scan(187)`` appended.

    Matches the native rough velocity cfg, whose ROUGH obs group adds a trailing
    ``height_scan`` term reading the ``terrain_scan`` grid raycast sensor
    (``velocity_env_cfg.py:200-205``). The 187-D grid is 17 (x: -0.8..0.8) x 11
    (y: -0.5..0.5) downward rays, yaw-aligned to the Go1 trunk; per-ray value is
    ``(frame_z - hit_z) * (1/max_distance)``.
    """

    @configclass
    class ActorCfg(_Go1ObsCfg.ActorCfg):
        height_scan = ObsTerm(
            func=obs.height_scan,
            params={"sensor_name": "terrain_scan", "scale": _GO1_HEIGHT_SCAN_SCALE},
        )

    @configclass
    class CriticCfg(ActorCfg):
        pass

    actor = ActorCfg()
    critic = CriticCfg()


@configclass
class _Go1RewardsCfg:
    """mjlab ``velocity_env_cfg.rewards`` — 14-term composition.

    Sensor-dependent terms (``angular_momentum``, ``air_time``,
    ``foot_clearance``, ``foot_swing_height``, ``foot_slip``,
    ``soft_landing``) are registered using stub functions in
    ``rewards.py`` that emit a one-shot warning + return zeros until
    sensor wiring is complete. The weights mirror mjlab native
    (``mjlab/tasks/velocity/velocity_env_cfg.py:275-371``); when sensors
    are wired the term function bodies activate automatically.

    Reads the ``twist`` velocity command from
    ``env.command_managers["twist"]`` (registered in the task ``__init__``).
    """

    # Tracking (positive-weight signals).
    track_linear_velocity = RewTerm(
        func=rew.track_linear_velocity,
        weight=2.0,
        params={"asset_cfg": _GO1_TRUNK, "command_name": "twist", "std": math.sqrt(0.25)},
    )
    track_angular_velocity = RewTerm(
        func=rew.track_angular_velocity,
        weight=2.0,
        params={"asset_cfg": _GO1_TRUNK, "command_name": "twist", "std": math.sqrt(0.5)},
    )

    # Stability / posture.
    upright = RewTerm(
        func=rew.upright,
        weight=1.0,
        params={"asset_cfg": _GO1_TRUNK, "std": math.sqrt(0.2)},
    )
    pose = RewTerm(
        func=rew.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": _GO1_JOINTS,
            "command_name": "twist",
            # mjlab Go1 stance (config/go1/env_cfgs.py:186-197): hip/thigh strict,
            # calf looser. EXACT match to native std maps.
            "std_standing": {
                r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
            },
            "std_walking": {
                r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
            },
            "std_running": {
                r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
            },
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
            "default_pose": _GO1_DEFAULT_POSE.unsqueeze(0),
        },
    )

    # Penalties — generic. mjlab go1 leaves body_ang_vel at weight 0.0
    # (config/go1/env_cfgs.py:206); only g1 (humanoid) penalizes it.
    body_ang_vel = RewTerm(
        func=rew.body_angular_velocity_penalty,
        weight=0.0,
        params={"asset_cfg": _GO1_TRUNK},
    )
    dof_pos_limits = RewTerm(
        func=rew.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": _GO1_JOINTS},
    )
    action_rate_l2 = RewTerm(func=rew.action_rate_l2, weight=-0.1)

    # Sensor-dependent — now FIRING (replaced stubs with real fn bodies).
    # Mjlab quadruped overrides for go1 (config/go1/env_cfgs.py:206-208):
    #   angular_momentum=0.0, body_ang_vel=0.0, air_time=0.0
    #   (mjlab leaves these at 0 for the quadruped; only g1 humanoid uses them)
    angular_momentum = RewTerm(
        func=rew.angular_momentum_penalty,
        weight=0.0,
        params={"sensor_name": "robot/root_angmom"},
    )
    air_time = RewTerm(
        func=rew.feet_air_time,
        weight=0.0,  # mjlab go1 sets air_time weight 0.0 (config/go1/env_cfgs.py:208).
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
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    soft_landing = RewTerm(
        func=rew.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )


@configclass
class _Go1TerminationsCfg:
    time_out = DoneTerm(func=term.time_out, time_out=True)
    fell_over = DoneTerm(func=term.bad_orientation, params={"asset_cfg": _GO1_TRUNK, "limit_angle": 1.0})


def _go1_foot_friction_event(env, **_):
    """Mjlab Go1 foot friction DR (mjlab/tasks/velocity/velocity_env_cfg.py:238).

    ``foot_friction``: tangential μ uniform in [0.3, 1.2], shared across all
    foot collision geoms. Bound to a startup event so it fires once at
    env construction.
    """
    from .mdp.scene_entity import SceneEntityCfg as _SE

    foot_geoms_cfg = _SE("go1")
    foot_geoms_cfg.geom_names = ("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision")
    geom_friction(
        env,
        asset_cfg=foot_geoms_cfg,
        ranges=(0.3, 1.2),
        operation="abs",
        axes=(0,),
        shared_random=True,
    )


def _go1_encoder_bias_event(env, **_):
    encoder_bias(env, asset_cfg=_GO1_JOINTS, bias_range=(-0.015, 0.015))


def _go1_base_com_event(env, **_):
    """Perturb trunk COM offset to model uneven robot mass distribution."""
    from .mdp.scene_entity import SceneEntityCfg as _SE

    cfg = _SE("go1")
    cfg.body_names = ("trunk",)
    body_com_offset(
        env,
        asset_cfg=cfg,
        operation="add",
        ranges={0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
    )


@configclass
class _Go1EventsCfg:
    reset_state = EventTerm(func=reset_go1_default_pose, mode="reset")
    # mjlab parity DR — startup events.
    foot_friction = EventTerm(func=_go1_foot_friction_event, mode="setup")
    encoder_bias_dr = EventTerm(func=_go1_encoder_bias_event, mode="setup")
    base_com = EventTerm(func=_go1_base_com_event, mode="setup")
    # Interval push (mjlab parity — push_by_setting_velocity in
    # velocity_env_cfg.events).
    push_robot = EventTerm(
        func=push_by_setting_velocity,
        mode="post_step",
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


@configclass
class _Go1CurriculumCfg:
    """mjlab ``velocity_env_cfg.curriculum`` port — both curricula now active."""

    terrain_levels = CurriculumTerm(
        func=terrain_levels_vel,
        params={"command_name": "twist", "asset_name": "go1"},
    )
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


# ---------------------------------------------------------------------------
# env config + task class
# ---------------------------------------------------------------------------


def _go1_scenario() -> ScenarioCfg:
    # Apply mjlab's EXACT compiled go1 actuator/dynamics spec (per-joint kp/kd/
    # armature/forcerange + integrator=implicitfast, dt=0.005, iters=10/ls=20).
    # implicitfast + the per-joint armature keep the stiff PD stable at dt=0.005,
    # matching mjlab native (no need for the old dt=0.002 explicit-Euler hack).
    patched_xml = patch_mjcf_with_exact_spec(mjlab_asset(_GO1_XML), "go1")
    return ScenarioCfg(
        scene=SceneCfg(mjcf_path=patched_xml),
        robots=[],
        # mjlab parity: dt=0.005, decimation=4 (5ms × 4 = 20ms = 50Hz). Solver
        # options come from the patched MJCF's <option> block.
        sim_params=SimParamCfg(dt=0.005),
        decimation=4,
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )


@configclass
class VelocityFlatGo1EnvCfg(ManagerBasedRVEnvCfg):
    """Manager-based env config for Go1 velocity tracking on flat ground."""

    decimation = 4  # matches scenario.decimation; step_dt = sim_dt x decimation = 0.02s = 50Hz
    max_episode_length_s = 20.0
    is_finite_horizon = False
    observation_group_names = ("actor", "critic")
    observations = _Go1ObsCfg()
    rewards = _Go1RewardsCfg()
    terminations = _Go1TerminationsCfg()
    events = _Go1EventsCfg()
    curriculum = _Go1CurriculumCfg()


@configclass
class VelocityRoughGo1EnvCfg(VelocityFlatGo1EnvCfg):
    """Rough-terrain Go1 env cfg — flat proprio obs + trailing ``height_scan(187)``."""

    observations = _Go1RoughObsCfg()


class _Go1TaskBase(ManagerBasedRVEnv):
    """Shared scaffold for all go1 velocity variants (flat / rough)."""

    scenario = _go1_scenario()
    env_cfg_cls = VelocityFlatGo1EnvCfg
    use_height_scan = False

    def __init__(self, scenario: ScenarioCfg | None = None, device: str | torch.device | None = None) -> None:
        cfg = self.env_cfg_cls()
        # Two-path scenario handling (mirror cartpole_v2):
        #   mujoco: scene-MJCF self-contained — drop trainer-injected robots
        #   newton: needs RobotCfg list — keep trainer-injected mjlab_go1
        sim = getattr(scenario, "simulator", None) if scenario else None
        if scenario is not None and sim == "mujoco":
            scenario.robots = []
        elif scenario is not None and sim == "newton":
            # CRITICAL: go1.xml has NO <actuator> elements (mjlab adds them at
            # runtime). Without actuators (mj nu=0) the Newton SolverMuJoCo
            # applies ZERO position-control torque -> go1 free-falls. The mujoco
            # scene path patches PD actuators in; the Newton RobotCfg path must
            # do the same. Point the go1 robot at an exact-spec-patched MJCF so
            # nu=12 and mjlab's per-joint PD actually holds the stance.
            for r in scenario.robots:
                mjcf = getattr(r, "mjcf_path", None)
                if getattr(r, "name", None) == "go1" and mjcf:
                    src = mjcf if os.path.isabs(mjcf) else os.path.abspath(mjcf)
                    try:
                        r.mjcf_path = patch_mjcf_with_exact_spec(src, "go1")
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"go1 Newton actuator patch failed ({e}); go1 will not stand",
                            RuntimeWarning,
                            stacklevel=2,
                        )
        self.num_actions = 12
        # Cache joint qpos/qvel indices for fast write in _apply_action
        # (lazy — set on first call, since handler isn't constructed yet)
        self._actuated_qpos_ids: np.ndarray | None = None
        super().__init__(scenario=scenario or self.scenario, cfg=cfg, device=device)

        # Register the ``twist`` VelocityCommand. Ranges + heading-command
        # mirror mjlab's ``velocity_env_cfg.commands["twist"]``.
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

        # Mjlab sensors. Go1 has no foot body; ``*_calf`` owns the foot
        # collision geoms. ContactSensor's primary_body walks the subtree, so a
        # contact on ``FR_foot_collision`` correctly attributes to ``FR_calf``.
        _GO1_FEET_BODIES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
        # Foot site offset in the calf body frame (go1.xml: <site pos="0 0 -0.213">).
        # mjlab's foot_height_scan + feet_slip read the foot SITE, not the calf
        # body origin; on Newton we reconstruct the site world pose from this.
        _GO1_FOOT_SITE_OFFSET = (0.0, 0.0, -0.213)
        _GO1_FEET_SITE_OFFSETS = tuple(_GO1_FOOT_SITE_OFFSET for _ in _GO1_FEET_BODIES)
        # ContactSensor works on BOTH backends now (Newton reads per-env contact
        # via mujoco_warp); register it unconditionally so feet_air_time /
        # soft_landing / feet_slip fire on the GPU path too.
        self._mjlab_sensors["feet_ground_contact"] = ContactSensor(
            self,
            ContactSensorCfg(
                name="feet_ground_contact",
                primary_bodies=_GO1_FEET_BODIES,
                secondary_body=None,
                fields=("found", "force"),
                track_air_time=True,
                history_length=4,
                site_offsets=_GO1_FEET_SITE_OFFSETS,
            ),
        )
        # TerrainHeightSensor works on both backends now (Newton uses foot body
        # z on flat terrain). BuiltinSensor (subtree_angmom) remains MuJoCo-only.
        self._mjlab_sensors["foot_height_scan"] = TerrainHeightSensor(
            self,
            TerrainHeightSensorCfg(
                name="foot_height_scan",
                primary_bodies=_GO1_FEET_BODIES,
                max_distance=1.0,
                geom_groups=(0,),
                target_height=0.0,
                site_offsets=_GO1_FEET_SITE_OFFSETS,
            ),
        )
        # BuiltinSensor (subtree_angmom) works on both backends now (Newton reads
        # mujoco_warp's batched subtree_angmom). Completes 6/6 sensor rewards on GPU.
        self._mjlab_sensors["robot/root_angmom"] = BuiltinSensor(
            self,
            BuiltinSensorCfg(
                name="robot/root_angmom",
                field="subtree_angmom",
                body_name="trunk",
            ),
        )

        # Rough-terrain only: the terrain_scan grid raycast feeding height_scan
        # obs. mjlab sets the frame to the Go1 trunk (config/go1/env_cfgs.py:50).
        if self.use_height_scan:
            self._mjlab_sensors["terrain_scan"] = HeightScanSensor(
                self,
                HeightScanSensorCfg(
                    name="terrain_scan",
                    frame_body="trunk",
                    size=(1.6, 1.0),
                    resolution=0.1,
                    ray_alignment="yaw",
                    max_distance=5.0,
                    geom_groups=(0,),
                ),
            )

        # mjlab parity: ActionManager (declarative scale/offset for joint pos action).
        # mjlab go1 (config/go1/env_cfgs.py) overrides the velocity-env default
        # scale=0.5 with the PER-JOINT GO1_ACTION_SCALE (0.25*effort/stiffness):
        # hip/thigh=0.3728, calf=0.2485. use_default_offset=True -> offset is the
        # go1 default standing pose (INIT_STATE keyframe): thigh=0.9, calf=-1.8,
        # FR/RR hip=+0.1, FL/RL hip=-0.1.
        _go1_scale = {n: float(_GO1_ACTION_SCALE[n]) for n in _GO1_JOINTS.joint_names}
        _go1_defaults = {n: float(v) for n, v in zip(_GO1_JOINTS.joint_names, _GO1_DEFAULT_POSE_NP.tolist())}
        self.action_manager = JointPositionActionManager(
            self,
            JointPositionActionCfg(
                entity_name="go1",
                actuator_names=_GO1_JOINTS.joint_names,
                scale=_go1_scale,
                use_default_offset=True,
            ),
            joint_names=_GO1_JOINTS.joint_names,
            defaults=_go1_defaults,
        )

        # Terrain manager — mjlab parity. Per-env terrain_levels start at 0
        # and progress through ``terrain_levels_vel`` curriculum hook.
        self.terrain_manager = TerrainLevelsManager(
            self,
            TerrainLevelsCfg(max_level=5, row_size=2.0, fail_factor=0.5),
        )

    def _get_initial_states(self) -> list[dict]:
        # mujoco scene-MJCF path: empty (reset_event writes physics.data directly).
        # newton RobotCfg path: explicit default base pose + joint defaults so the
        # robot doesn't spawn intersecting the ground or in midair.
        if not self.scenario.robots:
            return [{"objects": {}, "robots": {}} for _ in range(self.scenario.num_envs)]
        robot = self.scenario.robots[0]
        rname = robot.name  # 'go1'
        defaults = dict(robot.default_joint_positions)
        base_pos = list(robot.default_pos)
        base_rot = list(robot.default_rot)
        pos_t = torch.tensor(base_pos, dtype=torch.float32)
        rot_t = torch.tensor(base_rot, dtype=torch.float32)
        return [
            {
                "objects": {},
                "robots": {
                    rname: {
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
        """Return MuJoCo qvel indices for the 12 actuated joints (skip floating base)."""
        if self._actuated_qpos_ids is not None:
            return self._actuated_qpos_ids
        m = self.handler.physics.model
        mp = m.ptr if hasattr(m, "ptr") else m
        actuated = []
        for jname in _GO1_JOINTS.joint_names:
            jid = mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qvel_addr = mp.jnt_dofadr[jid]  # offset into qvel
            actuated.append(qvel_addr)
        self._actuated_qpos_ids = np.asarray(actuated, dtype=np.int64)
        return self._actuated_qpos_ids

    _ctrl_ids_for_joints: np.ndarray | None = None

    def _resolve_ctrl_indices(self) -> np.ndarray:
        """MuJoCo ctrl (actuator) index for each joint in ``_GO1_JOINTS``.

        The exact-spec patch emits actuators grouped by actuator type (hip/thigh
        first, then calf), NOT joint order, so ``data.ctrl`` is permuted relative
        to the policy's joint order. Map each target to its actuator slot.
        """
        if self._ctrl_ids_for_joints is not None:
            return self._ctrl_ids_for_joints
        m = self.handler.physics.model
        mp = m.ptr if hasattr(m, "ptr") else m
        ids = [mujoco.mj_name2id(mp, mujoco.mjtObj.mjOBJ_ACTUATOR, jn) for jn in _GO1_JOINTS.joint_names]
        assert all(i >= 0 for i in ids), "go1 exact-spec patch missing a per-joint actuator"
        self._ctrl_ids_for_joints = np.asarray(ids, dtype=np.int64)
        return self._ctrl_ids_for_joints

    def _apply_action(self, processed_action: torch.Tensor) -> None:
        """Write action via mjlab-parity ActionManager processing.

        Now ``self.action_manager`` (JointPositionActionManager) handles
        scale + offset + clip + encoder_bias subtraction; this method
        just routes the processed target to the simulator handler.
        """
        target = self.action_manager.process(processed_action)
        if not hasattr(self.handler, "physics"):
            # The action_manager produces targets in cfg (_GO1_JOINTS) order, but
            # the Newton handler's tensor set_dof_targets applies them in
            # get_joint_names(sort=True) (alphabetical) order. Without reordering,
            # PD targets land on the WRONG joints -> go1 collapses. Build the
            # cfg->sorted permutation once and reorder.
            if getattr(self, "_cfg_to_sorted", None) is None:
                sorted_bare = [n.split("/")[-1] for n in self.handler.get_joint_names("go1", sort=True)]
                cfg_order = list(_GO1_JOINTS.joint_names)
                self._cfg_to_sorted = torch.tensor(
                    [cfg_order.index(b) for b in sorted_bare], device=target.device, dtype=torch.long
                )
            self.handler.set_dof_targets(target[..., self._cfg_to_sorted])
            return
        target_np = target.detach().cpu().numpy().reshape(-1)
        # target_np[i] is the setpoint for joint _GO1_JOINTS[i]; route each to
        # its actuator's ctrl slot (actuator order != joint order under the
        # exact-spec patch).
        self.handler.physics.data.ctrl[self._resolve_ctrl_indices()] = target_np


@register_task("mjlab.velocity_flat_go1_v2")
class VelocityFlatGo1Task(_Go1TaskBase):
    """Go1 quadruped forward-velocity scaffold (flat ground)."""


@register_task("mjlab.velocity_rough_go1_v2")
class VelocityRoughGo1Task(_Go1TaskBase):
    """Go1 rough-terrain velocity task: flat proprio obs + height_scan(187).

    Actor obs = ``_Go1ObsCfg`` 48-D proprio + trailing ``height_scan`` term that
    reads the yaw-aligned 17x11 ``terrain_scan`` grid raycast (mjlab parity).
    """

    env_cfg_cls = VelocityRoughGo1EnvCfg
    use_height_scan = True
