from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from metasim.utils import configclass
from roboverse_pack.tasks.beyondmimic.metasim.configs.cfg_randomizers import MassRandomizer, MaterialRandomizer
from roboverse_pack.tasks.beyondmimic.metasim.mdp import (
    events,
    observations,
    rewards,
    terminations,
)
from roboverse_pack.tasks.beyondmimic.metasim.mdp.commands import MotionCommandCfg

from .cfg_base import BaseEnvCfg
from .cfg_queries import ContactForces

VELOCITY_RANGE = {
    # linear velocity
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    # angular velocity
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class CfgTerm:
    """Configuration for terminal functions."""

    func: Callable = MISSING
    params: dict[str, any] = {}


@configclass
class ObsTerm(CfgTerm):
    """Configuration for observation functions."""

    noise_range: tuple[float, float] | None = None


@configclass
class RewTerm(CfgTerm):
    """Configuration for reward functions."""

    weight: float = 1.0


@configclass
class DoneTerm(CfgTerm):
    """Configuration for termination functions."""

    time_out: bool = False


@configclass
class ObservationsCfg:
    """Configuration for observations."""

    @configclass
    class PolicyCfg:
        """Configuration for policy observations."""

        command = ObsTerm(func=observations.generated_commands)
        motion_anchor_pos_b = ObsTerm(func=observations.motion_anchor_pos_b, noise_range=(-0.25, 0.25))
        motion_anchor_ori_b = ObsTerm(func=observations.motion_anchor_ori_b, noise_range=(-0.05, 0.05))
        base_lin_vel = ObsTerm(func=observations.base_lin_vel, noise_range=(-0.5, 0.5))
        base_ang_vel = ObsTerm(func=observations.base_ang_vel, noise_range=(-0.2, 0.2))
        joint_pos = ObsTerm(func=observations.joint_pos_rel, noise_range=(-0.01, 0.01))
        joint_vel = ObsTerm(func=observations.joint_vel_rel, noise_range=(-0.5, 0.5))
        actions = ObsTerm(func=observations.last_action)

    @configclass
    class PrivilegedCfg:
        """Configuration for privileged observations."""

        command = ObsTerm(func=observations.generated_commands)
        motion_anchor_pos_b = ObsTerm(func=observations.motion_anchor_pos_b)
        motion_anchor_ori_b = ObsTerm(func=observations.motion_anchor_ori_b)
        body_pos = ObsTerm(func=observations.robot_body_pos_b)
        body_ori = ObsTerm(func=observations.robot_body_ori_b)
        base_lin_vel = ObsTerm(func=observations.base_lin_vel)
        base_ang_vel = ObsTerm(func=observations.base_ang_vel)
        joint_pos = ObsTerm(func=observations.joint_pos_rel)
        joint_vel = ObsTerm(func=observations.joint_vel_rel)
        actions = ObsTerm(func=observations.last_action)

    # observation groups
    policy = PolicyCfg()
    critic = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Configuration for rewards."""

    motion_global_anchor_pos = RewTerm(
        func=rewards.motion_global_anchor_position_error_exp, weight=0.5, params={"std": 0.3}
    )
    motion_global_anchor_ori = RewTerm(
        func=rewards.motion_global_anchor_orientation_error_exp, weight=0.5, params={"std": 0.4}
    )
    motion_body_pos = RewTerm(func=rewards.motion_relative_body_position_error_exp, weight=1.0, params={"std": 0.3})
    motion_body_ori = RewTerm(func=rewards.motion_relative_body_orientation_error_exp, weight=1.0, params={"std": 0.4})
    motion_body_lin_vel = RewTerm(
        func=rewards.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"std": 1.0}
    )
    motion_body_ang_vel = RewTerm(
        func=rewards.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"std": 3.14}
    )
    action_rate_l2 = RewTerm(func=rewards.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(func=rewards.joint_pos_limits, weight=-10.0)
    undesired_contacts = RewTerm(
        func=rewards.undesired_contacts,
        weight=-0.1,
        params={
            "threshold": 1.0,
            "body_names": r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$",
        },
    )


@configclass
class TerminationsCfg:
    """Configuration for terminations."""

    time_out = DoneTerm(func=terminations.time_out, time_out=True)
    anchor_pos = DoneTerm(func=terminations.bad_anchor_pos_z_only, params={"threshold": 0.25})
    anchor_ori = DoneTerm(func=terminations.bad_anchor_ori, params={"threshold": 0.8})
    ee_body_pos = DoneTerm(
        func=terminations.bad_motion_body_pos_z_only,
        params={
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class TrackingG1EnvCfg(BaseEnvCfg):
    """Environment configuration for humanoid motion tracking task."""

    commands = MotionCommandCfg(
        anchor_body_name="torso_link",
        body_names=[  # for indexing motion body links
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    # NOTE extra obs will be included in `env_states.extras["contact_forces"]`
    callbacks_query = {"contact_forces": ContactForces(history_length=3)}

    # TODO fully align domain randomization with BeyondMimic
    callbacks_setup = {
        "material_randomizer": MaterialRandomizer(
            obj_name="g1_tracking",
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            num_buckets=64,
        ),
        # TODO change `MassRandomizer` to `randomize_rigid_body_com()` from BeyondMimic
        "mass_randomizer": MassRandomizer(
            obj_name="g1_tracking",
            body_names="torso_link",
            mass_distribution_params=(-1.0, 3.0),  # TODO change this
            operation="add",
        ),
        # NOTE `env` will be passed to the functions inside `LeggedRobotTask._bind_callbacks()`
        "add_joint_default_pos": (
            events.randomize_joint_default_pos,
            {
                "pos_distribution_params": (-0.01, 0.01),
                "operation": "add",
            },
        ),
    }
    callbacks_post_step = {
        # NOTE perhaps slightly different from how it's triggered in BeyondMimic
        "push_robot": (
            events.push_by_setting_velocity,
            {
                "interval_range_s": (1.0, 3.0),
                "velocity_range": VELOCITY_RANGE,
            },
        )
    }
