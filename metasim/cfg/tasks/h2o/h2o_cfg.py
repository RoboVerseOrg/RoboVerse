# h2o_task_cfg.py
from __future__ import annotations

from dataclasses import MISSING
from typing import Callable, Dict, List

import torch

from metasim.cfg.checkers.base_checker import BaseChecker
from metasim.cfg.control import ControlCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg
from metasim.sim import BaseSimHandler
from metasim.utils import configclass
from metasim.utils.humanoid_robot_util import contact_forces_tensor


# ───────────────── domain randomisation ───────────────────────
@configclass
class H2ODomainRandCfg:
    class Push:  # external hits
        enabled = True
        max_push_vel_xy = 0.5
        push_interval_s = 5.0
    push = Push()

    friction_enabled = True
    friction_range = [-0.6, 1.2]

    randomize_base_com = True
    base_com_range = {"x": [-0.1, 0.1], "y": [-0.1, 0.1], "z": [-0.1, 0.1]}

    randomize_link_mass = True
    link_mass_range = [0.7, 1.3]
    link_names = [
        "pelvis",
        "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
        "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
        "torso_link",
    ]

    randomize_pd_gain = True
    kp_range = [0.75, 1.25]
    kd_range = [0.75, 1.25]

    randomize_torque_rfi = True
    rfi_lim = 0.1
    rfi_lim_range = [0.5, 1.5]

    randomize_ctrl_delay = True
    ctrl_delay_step_range = [1, 3]

    randomize_motion_ref_xyz = True
    motion_ref_xyz_range = [[-0.02, 0.02], [-0.02, 0.02], [-0.05, 0.05]]


# ───────────────── observation noise ──────────────────────────
@configclass
class H2ONoiseCfg:
    add_noise = True
    noise_level = 1.0

    class scales:
        base_z = 0.05; dof_pos = 0.01; dof_vel = 0.10
        lin_vel = 0.20; ang_vel = 0.50; gravity = 0.10
        height_measurements = 0.05; body_pos = 0.01
        body_lin_vel = 0.01; body_rot = 0.001
        delta_base_pos = 0.05; delta_heading = 0.10
        ref_body_pos = 0.10; ref_body_rot = 0.01; ref_body_vel = 0.01
        ref_lin_vel = 0.01; ref_ang_vel = 0.01
        ref_dof_pos = 0.01; ref_dof_vel = 0.01; ref_gravity = 0.01


# ───────────────── contact-based reset checker ────────────────
@configclass
class H2OContactChecker(BaseChecker):
    def check(self, handler: BaseSimHandler):
        cf = contact_forces_tensor(handler.get_states(), handler.robot.name)
        idx = handler.task.termination_contact_indices
        return torch.any(torch.norm(cf[:, idx, :], dim=-1) > 1.0, dim=1)


# ───────────────── main task cfg ──────────────────────────────
@configclass
class H2OTaskCfg(BaseRLTaskCfg):
    # rewards
    class RewardCfg:
        feet_air_time_teleop = 800.0
        teleop_body_position_extend = 40.0
        teleop_body_rotation = 20.0
        teleop_body_vel = 8.0
        teleop_body_ang_vel = 8.0
        teleop_selected_joint_position = 32.0
        teleop_selected_joint_vel = 16.0
        torques = -9e-5 * 1.25
        torque_limits = -2e-1 * 1.25
        dof_acc = -8.4e-6 * 1.25
        dof_vel = -0.003 * 1.25
        lower_action_rate = -0.9 * 1.25
        upper_action_rate = -0.05 * 1.25
        termination = -200.0 * 1.25
        stumble = -1000.0 * 1.25
        soft_torque_limit = 0.85
        max_contact_force = 500.0
        only_positive_rewards = False
        tracking_sigma = 0.25
    reward_cfg: RewardCfg = RewardCfg()
    reward_functions: List[Callable] = MISSING
    reward_weights: Dict[str, float] = MISSING

    # command generator
    class Commands:
        curriculum = False
        max_curriculum = 0.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = False
    commands: Commands = Commands()

    class CommandRanges:
        lin_vel_x = [0.0, 0.0]
        lin_vel_y = [0.0, 0.0]
        ang_vel_yaw = [0.0, 0.0]
        heading = [0.0, 0.0]
    command_ranges: CommandRanges = CommandRanges()

    # observation & action sizes
    decimation = 4
    num_obs = 138
    num_privileged_obs = 214
    num_actions = 19

    # normalisation
    class Normalization:
        class obs_scales:
            lin_vel = ang_vel = dof_pos = dof_vel = height_measurements = 1.0
            body_pos = body_lin_vel = body_rot = delta_base_pos = delta_heading = 1.0
        clip_observations = 100.0
        clip_actions = 100.0
    normalization: Normalization = Normalization()

    # physics
    sim_params = SimParamCfg(dt=0.005, num_threads=10)
    dt = decimation * sim_params.dt

    env_spacing = 2.0
    send_timeouts = True
    episode_length_s = 20.0

    # control
    control: ControlCfg = ControlCfg(
        control_type="P",
        action_scale=0.25,
        decimation=decimation,
        torque_limit_scale=0.85,
        stiffness={
            "hip_yaw": 200, "hip_roll": 200, "hip_pitch": 200,
            "knee": 300, "ankle": 40, "torso": 300,
            "shoulder": 100, "elbow": 100,
        },
        damping={
            "hip_yaw": 5, "hip_roll": 5, "hip_pitch": 5,
            "knee": 6, "ankle": 2, "torso": 6,
            "shoulder": 2, "elbow": 2,
        },
        action_filt=False,
        action_cutfreq=4.0,
    )

    # robot asset
    class Asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h2o/urdf/h2o.urdf"
        name = "h2o"
        foot_name = "ankle"
        terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
        self_collisions = 1
        replace_cylinder_with_capsule = True
        density = 0.001
        set_dof_properties = True
        default_dof_prop_damping = [5] * 19
        default_dof_prop_stiffness = [0] * 19
        default_dof_prop_friction = [0] * 19
        terminate_by_ref_motion_distance = True
        terminate_by_1time_motion = True

        class termination_scales:
            base_height = 0.3
            base_vel = 10.0
            base_ang_vel = 5.0
            gravity_x = gravity_y = 0.7
            max_ref_motion_distance = 0.5
    asset: Asset = Asset()

    # terrain
    class Terrain:
        mesh_type = "trimesh"
        border_size = 25
        curriculum = False
        static_friction = dynamic_friction = 1.0
        restitution = 0.0
        slope_treshold = 0.75
    terrain: Terrain = Terrain()

    # reference motion
    class Motion:
        teleop = True
        num_markers = 19
        motion_file = "{LEGGED_GYM_ROOT_DIR}/resources/motions/h1/amass_phc_filtered.pkl"
        skeleton_file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/xml/h1.xml"
        marker_file = "{LEGGED_GYM_ROOT_DIR}/resources/objects/Marker/traj_marker.urdf"
        teleop_obs_version = "v-teleop-extend-max"
        recycle_motion = True
        resample_motions_for_envs = True
        resample_motions_for_envs_interval_s = 1000.0
        extend_head = False
        extend_hand = True
        teleop_selected_keypoints_names = [
            "left_ankle_link", "right_ankle_link",
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",
            "left_elbow_link", "right_elbow_link",
        ]
    motion: Motion = Motion()

    # viewer
    class Viewer:
        debug_viz = False
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11, 5, 3]
    viewer: Viewer = Viewer()

    # plugs
    random: H2ODomainRandCfg = H2ODomainRandCfg()
    noise: H2ONoiseCfg = H2ONoiseCfg()
    checker = H2OContactChecker()

    # indices filled by wrapper
    feet_indices: torch.Tensor = MISSING
    knee_indices: torch.Tensor = MISSING
    elbow_indices: torch.Tensor = MISSING
    wrist_indices: torch.Tensor = MISSING
    torso_indices: torch.Tensor = MISSING
    penalised_contact_indices: torch.Tensor = MISSING
    termination_contact_indices: torch.Tensor = MISSING

    # initial pose
    init_states = [
        {
            "objects": {},
            "robots": {
                "h2o": {
                    "pos": torch.tensor([0.0, 0.0, 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": -0.4,
                        "left_knee": 0.8, "left_ankle": -0.4,
                        "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -0.4,
                        "right_knee": 0.8, "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0, "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0, "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0, "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0, "right_elbow": 0.0,
                    },
                },
            },
        }
    ]
