from __future__ import annotations

import os
from dataclasses import MISSING
from typing import Any

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass
from roboverse_pack.tasks.beyondmimic.metasim.utils.string import resolve_matching_names_values

ASSET_DIR = "roboverse_data"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


@configclass
class ActuatorCfg:
    joint_names_expr: list[str] = MISSING
    effort_limit_sim: dict[str, float] | float = MISSING
    velocity_limit_sim: dict[str, float] | float = MISSING
    stiffness: dict[str, float] | float = MISSING
    damping: dict[str, float] | float = MISSING
    armature: dict[str, float] | float = MISSING


@configclass
class G1TrackingCfg(RobotCfg):
    name: str = "g1_tracking"
    num_joints: int = 29

    # NOTE this path should be absolute because the converted USD file may contain references to other files (e.g., @configuration/g1_sensor.usd@) which will be incorrectly resolved if the USD path is relative
    usd_path: str = os.path.abspath(f"{ASSET_DIR}/unitree_description/usd/g1/g1.usd")
    xml_path: str = f"{ASSET_DIR}/unitree_description/mjcf/g1.xml"
    urdf_path: str = f"{ASSET_DIR}/unitree_description/urdf/g1/main.urdf"
    mjcf_path = xml_path
    enabled_gravity: bool = True
    enabled_self_collisions: bool = True

    max_depenetration_velocity: float = 1.0
    fix_base_link: bool | None = None

    # to override the default collision properties of USD file config in Isaac Sim handler
    collision_props: Any | None = None

    # NOTE initial state is defined in `BaseEnvCfg.InitialStates`

    default_pos = (0.0, 0.0, 0.76)
    default_joint_positions = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
    default_joint_velocities = {".*": 0.0}
    default_rot = (1.0, 0.0, 0.0, 0.0)
    soft_joint_pos_limit_factor = 0.9

    # NOTE joint position limits obtained through `Articulation.root_physx_view.get_dof_limits()` in Isaac Lab
    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_pitch_joint": (-2.5306997299194336, 2.8797998428344727),
        "right_hip_pitch_joint": (-2.5306997299194336, 2.8797998428344727),
        "waist_yaw_joint": (-2.618000030517578, 2.618000030517578),
        "left_hip_roll_joint": (-0.5235999226570129, 2.967099666595459),
        "right_hip_roll_joint": (-2.967099666595459, 0.5235999226570129),
        "waist_roll_joint": (-0.5199999213218689, 0.5199999213218689),
        "left_hip_yaw_joint": (-2.7576000690460205, 2.7576000690460205),
        "right_hip_yaw_joint": (-2.7576000690460205, 2.7576000690460205),
        "waist_pitch_joint": (-0.5199999213218689, 0.5199999213218689),
        "left_knee_joint": (-0.08726699650287628, 2.8797998428344727),
        "right_knee_joint": (-0.08726699650287628, 2.8797998428344727),
        "left_shoulder_pitch_joint": (-3.0891997814178467, 2.6703999042510986),
        "right_shoulder_pitch_joint": (-3.0891997814178467, 2.6703999042510986),
        "left_ankle_pitch_joint": (-0.8726699352264404, 0.5235999226570129),
        "right_ankle_pitch_joint": (-0.8726699352264404, 0.5235999226570129),
        "left_shoulder_roll_joint": (-1.5881999731063843, 2.251499652862549),
        "right_shoulder_roll_joint": (-2.251499652862549, 1.5881999731063843),
        "left_ankle_roll_joint": (-0.26179996132850647, 0.26179996132850647),
        "right_ankle_roll_joint": (-0.26179996132850647, 0.26179996132850647),
        "left_shoulder_yaw_joint": (-2.618000030517578, 2.618000030517578),
        "right_shoulder_yaw_joint": (-2.618000030517578, 2.618000030517578),
        "left_elbow_joint": (-1.0471998453140259, 2.0943996906280518),
        "right_elbow_joint": (-1.0471998453140259, 2.0943996906280518),
        "left_wrist_roll_joint": (-1.972221851348877, 1.972221851348877),
        "right_wrist_roll_joint": (-1.972221851348877, 1.972221851348877),
        "left_wrist_pitch_joint": (-1.6144295930862427, 1.6144295930862427),
        "right_wrist_pitch_joint": (-1.6144295930862427, 1.6144295930862427),
        "left_wrist_yaw_joint": (-1.6144295930862427, 1.6144295930862427),
        "right_wrist_yaw_joint": (-1.6144295930862427, 1.6144295930862427),
    }

    actuators_cfg = {
        "legs": ActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    }
    actuators: dict[str, BaseActuatorCfg] = dict()
    action_scale: dict[str, float] = dict()
    action_clip: float | None = None
    action_offset: bool = True  # offset actions by `default_dof_pos_original` specified in the task class

    def __post_init__(self):
        super().__post_init__()
        actuators = {}
        action_scale = {}
        for cfg in self.actuators_cfg.values():
            for name in cfg.joint_names_expr:
                effort_limit = (
                    cfg.effort_limit_sim[name] if isinstance(cfg.effort_limit_sim, dict) else cfg.effort_limit_sim
                )
                vel_limit = (
                    cfg.velocity_limit_sim[name] if isinstance(cfg.velocity_limit_sim, dict) else cfg.velocity_limit_sim
                )
                stiffness = cfg.stiffness[name] if isinstance(cfg.stiffness, dict) else cfg.stiffness
                damping = cfg.damping[name] if isinstance(cfg.damping, dict) else cfg.damping
                armature = cfg.armature[name] if isinstance(cfg.armature, dict) else cfg.armature

                # align actuators with RoboVerse API
                actuators[name] = BaseActuatorCfg(
                    effort_limit_sim=effort_limit,
                    velocity_limit_sim=vel_limit,
                    stiffness=stiffness,
                    damping=damping,
                    armature=armature,
                )
                # compute action scales
                action_scale[name] = 0.25 * effort_limit / stiffness

        # resolve regex to avoid compatibility issues with RoboVerse APIs
        joint_names = list(self.joint_limits.keys())
        _, name_list, value_list = resolve_matching_names_values(actuators, joint_names)
        self.actuators = {k: v for k, v in zip(name_list, value_list)}
        _, name_list, value_list = resolve_matching_names_values(action_scale, joint_names)
        self.action_scale = {k: v for k, v in zip(name_list, value_list)}

        # Default to position control for motion-tracking across simulators.
        # This is required by some backends (e.g., IsaacGym) to configure DOF drive modes.
        self.control_type = {name: "position" for name in joint_names}

        # Resolve regex-based default joint state dictionaries for simulators that expect explicit names (e.g. MuJoCo).
        if self.default_joint_positions is not None:
            _, name_list, value_list = resolve_matching_names_values(self.default_joint_positions, joint_names)
            self.default_joint_positions = {k: v for k, v in zip(name_list, value_list)}
        if self.default_joint_velocities is not None:
            _, name_list, value_list = resolve_matching_names_values(self.default_joint_velocities, joint_names)
            self.default_joint_velocities = {k: v for k, v in zip(name_list, value_list)}
