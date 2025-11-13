from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1Dof12Cfg(RobotCfg):
    name: str = "g1_dof12"
    num_joints: int = 12
    usd_path: str = "roboverse_data/robots/g1/usd/g1_12dof.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_12dof.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_12dof.urdf"
    mjcf_path = xml_path
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = True
    # isaacgym_read_mjcf = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = False  # True

    actuators: dict[str, BaseActuatorCfg] = {
        # N7520-14.3: hip_pitch, hip_yaw (stiffness 100, damping 2, torque 88, vel 32)
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88, velocity_limit=32.0),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88, velocity_limit=32.0),
        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88, velocity_limit=32.0),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=88, velocity_limit=32.0),
        # N7520-22.5: hip_roll, knee (hip_roll stiffness 100/damping 2; knee stiffness 150/damping 4; torque 139; vel 20)
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=139, velocity_limit=20.0),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=100, damping=2, torque_limit=139, velocity_limit=20.0),
        "left_knee_joint": BaseActuatorCfg(stiffness=150, damping=4, torque_limit=139, velocity_limit=20.0),
        "right_knee_joint": BaseActuatorCfg(stiffness=150, damping=4, torque_limit=139, velocity_limit=20.0),
        # N5020-16: ankles (stiffness 40, damping 2, torque 25, vel 37)
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=25, velocity_limit=37.0),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=25, velocity_limit=37.0),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=25, velocity_limit=37.0),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=40, damping=2, torque_limit=25, velocity_limit=37.0),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Hips & legs
        "left_hip_pitch_joint": (-2.5307, 2.8798),
        "left_hip_roll_joint": (-0.5236, 2.9671),
        "left_hip_yaw_joint": (-2.7576, 2.7576),
        "left_knee_joint": (-0.087267, 2.8798),
        "left_ankle_pitch_joint": (-0.87267, 0.5236),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_hip_pitch_joint": (-2.5307, 2.8798),
        "right_hip_roll_joint": (-2.9671, 0.5236),
        "right_hip_yaw_joint": (-2.7576, 2.7576),
        "right_knee_joint": (-0.087267, 2.8798),
        "right_ankle_pitch_joint": (-0.87267, 0.5236),
        "right_ankle_roll_joint": (-0.2618, 0.2618),
    }

    # torque_limits: dict[str, float] = {
    #     # Hips & legs
    #     "left_hip_pitch_joint": 88,
    #     "left_hip_roll_joint": 139,
    #     "left_hip_yaw_joint": 88,
    #     "left_knee_joint": 139,
    #     "left_ankle_pitch_joint": 25,
    #     "left_ankle_roll_joint": 25,
    #     "right_hip_pitch_joint": 88,
    #     "right_hip_roll_joint": 139,
    #     "right_hip_yaw_joint": 88,
    #     "right_knee_joint": 139,
    #     "right_ankle_pitch_joint": 25,
    #     "right_ankle_roll_joint": 25,
    # }

    default_joint_positions: dict[str, float] = {
        # Hips & legs
        "left_hip_pitch_joint": -0.1,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.1,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # Hips & legs
        "left_hip_pitch_joint": "effort",
        "left_hip_roll_joint": "effort",
        "left_hip_yaw_joint": "effort",
        "left_knee_joint": "effort",
        "left_ankle_pitch_joint": "effort",
        "left_ankle_roll_joint": "effort",
        "right_hip_pitch_joint": "effort",
        "right_hip_roll_joint": "effort",
        "right_hip_yaw_joint": "effort",
        "right_knee_joint": "effort",
        "right_ankle_pitch_joint": "effort",
        "right_ankle_roll_joint": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    torso_links: list[str] = ["torso_link"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["rubber_hand"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee", "shoulder", "elbow", "wrist"]

    # joint substrings, to find indices of joints.
    left_hip_yaw_roll_joints: list[str] = ["left_hip_yaw_joint", "left_hip_roll_joint"]
    right_hip_yaw_roll_joints: list[str] = ["right_hip_yaw_joint", "right_hip_roll_joint"]

    soft_joint_pos_limit_factor = 0.9
    # From default joint armature in XML
    armature: float = 0.01


@configclass
class G1Dof23Cfg(G1Dof12Cfg):
    name: str = "g1_dof23"
    num_joints: int = 23
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str = MISSING
    mjcf_path = xml_path

    actuators = {
        **G1Dof12Cfg().actuators,
        # N7520-14.3: waist_yaw (stiffness 200, damping 5, torque 88, vel 32)
        "waist_yaw_joint": BaseActuatorCfg(stiffness=200, damping=5, torque_limit=88, velocity_limit=32.0),
        # N5020-16: shoulders, elbows, wrist_roll (stiffness 40, damping 1, torque 25, vel 37)
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "left_elbow_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=25, velocity_limit=37.0),
    }

    joint_limits = {
        **G1Dof12Cfg().joint_limits,
        # Waist
        "waist_yaw_joint": (-2.618, 2.618),
        # Shoulders & arms
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_roll_joint": (-1.972222, 1.972222),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "right_wrist_roll_joint": (-1.972222, 1.972222),
    }

    # torque_limits = {
    #     **G1Dof12Cfg().torque_limits,
    #     # Waist
    #     "waist_yaw_joint": 88,
    #     # Shoulders & arms
    #     "left_shoulder_pitch_joint": 25,
    #     "left_shoulder_roll_joint": 25,
    #     "left_shoulder_yaw_joint": 25,
    #     "left_elbow_joint": 25,
    #     "left_wrist_roll_joint": 25,
    #     "right_shoulder_pitch_joint": 25,
    #     "right_shoulder_roll_joint": 25,
    #     "right_shoulder_yaw_joint": 25,
    #     "right_elbow_joint": 25,
    #     "right_wrist_roll_joint": 25,
    # }

    default_joint_positions = {
        **G1Dof12Cfg().default_joint_positions,
        # Waist
        "waist_yaw_joint": 0.0,
        # Shoulders & arms
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
    }

    control_type = {
        **G1Dof12Cfg().control_type,
        # Waist
        "waist_yaw_joint": "effort",
        # Shoulders & arms
        "left_shoulder_pitch_joint": "effort",
        "left_shoulder_roll_joint": "effort",
        "left_shoulder_yaw_joint": "effort",
        "left_elbow_joint": "effort",
        "left_wrist_roll_joint": "effort",
        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
        "right_wrist_roll_joint": "effort",
    }


@configclass
class G1Dof27Cfg(G1Dof23Cfg):
    name: str = "g1_dof27"
    num_joints: int = 27
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str = MISSING
    mjcf_path = xml_path

    actuators = {
        **G1Dof23Cfg().actuators,
        # W4010-25: wrist_pitch/yaw (stiffness 40, damping 1, torque 5, vel 22)
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=5, velocity_limit=22.0),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=5, velocity_limit=22.0),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=5, velocity_limit=22.0),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=1, torque_limit=5, velocity_limit=22.0),
    }

    joint_limits = {
        **G1Dof23Cfg().joint_limits,
        "left_wrist_pitch_joint": (-1.61443, 1.61443),
        "left_wrist_yaw_joint": (-1.61443, 1.61443),
        "right_wrist_pitch_joint": (-1.61443, 1.61443),
        "right_wrist_yaw_joint": (-1.61443, 1.61443),
    }

    # torque_limits = {
    #     **G1Dof23Cfg().torque_limits,
    #     "left_wrist_pitch_joint": 5,
    #     "left_wrist_yaw_joint": 5,
    #     "right_wrist_pitch_joint": 5,
    #     "right_wrist_yaw_joint": 5,
    # }

    default_joint_positions = {
        **G1Dof23Cfg().default_joint_positions,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    control_type = {
        **G1Dof23Cfg().control_type,
        "left_wrist_pitch_joint": "effort",
        "left_wrist_yaw_joint": "effort",
        "right_wrist_pitch_joint": "effort",
        "right_wrist_yaw_joint": "effort",
    }


@configclass
class G1Dof29Cfg(G1Dof27Cfg):
    name: str = "g1_dof29"
    num_joints: int = 29
    usd_path: str = "roboverse_data/robots/g1/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
    xml_path: str = "roboverse_data/robots/g1/mjcf/g1_29dof_rev_1_0.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_rev_1_0.urdf"
    mjcf_path = xml_path

    actuators = {
        **G1Dof27Cfg().actuators,
        # N5020-16: waist roll/pitch (stiffness 40, damping 5, torque 25, vel 37)
        "waist_roll_joint": BaseActuatorCfg(stiffness=40, damping=5, torque_limit=25, velocity_limit=37.0),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=5, torque_limit=25, velocity_limit=37.0),
    }

    joint_limits = {
        **G1Dof27Cfg().joint_limits,
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
    }

    # torque_limits = {
    #     **G1Dof27Cfg().torque_limits,
    #     "waist_roll_joint": 25,
    #     "waist_pitch_joint": 25,
    # }

    default_joint_positions = {
        **G1Dof27Cfg().default_joint_positions,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
    }

    control_type = {
        **G1Dof27Cfg().control_type,
        "waist_roll_joint": "effort",
        "waist_pitch_joint": "effort",
    }

    # def __post_init__(self):
    #     self.cameras: list = [
    #         PinholeCameraCfg(
    #             name="front_cam",
    #             data_types=["rgb"],
    #             height=480,
    #             width=640,
    #             focal_length=7.6,
    #             focus_distance=400.0,
    #             horizontal_aperture=20.0,
    #             clipping_range=(0.1, 1.0e5),
    #             mount_to=self.name,
    #             mount_link="d435_link",
    #             mount_pos=(0, 0.0, 0),
    #             # mount_quat=(0.5, -0.5, 0.5, -0.5), # ros convention
    #             mount_quat=(1, 0, 0, 0),  # world convention
    #             # update_period: float = 0.02,
    #         )
    #     ]
    #     return super().__post_init__()


@configclass
class G1Dof29Dex3Cfg(G1Dof29Cfg):
    name: str = "g1_dof29_dex3"
    num_joints: int = 43
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str = MISSING
    mjcf_path = xml_path

    actuators = {
        **G1Dof29Cfg().actuators,
        "left_hand_thumb_0_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=2.45),
        "left_hand_thumb_1_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "left_hand_thumb_2_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "left_hand_middle_0_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "left_hand_middle_1_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "left_hand_index_0_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "left_hand_index_1_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "right_hand_thumb_0_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=2.45),
        "right_hand_thumb_1_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "right_hand_thumb_2_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "right_hand_middle_0_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "right_hand_middle_1_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "right_hand_index_0_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
        "right_hand_index_1_joint": BaseActuatorCfg(stiffness=5, damping=1, torque_limit=1.4),
    }

    joint_limits = {
        **G1Dof29Cfg().joint_limits,
        # Hands
        "left_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "left_hand_thumb_1_joint": (-0.61086523, 1.04719755),
        "left_hand_thumb_2_joint": (0.0, 1.74532925),
        "left_hand_middle_0_joint": (-1.57079632, 0.0),
        "left_hand_middle_1_joint": (-1.74532925, 0.0),
        "left_hand_index_0_joint": (-1.57079632, 0.0),
        "left_hand_index_1_joint": (-1.74532925, 0.0),
        "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "right_hand_thumb_1_joint": (-1.04719755, 0.61086523),
        "right_hand_thumb_2_joint": (-1.74532925, 0.0),
        "right_hand_middle_0_joint": (0.0, 1.57079632),
        "right_hand_middle_1_joint": (0.0, 1.74532925),
        "right_hand_index_0_joint": (0.0, 1.57079632),
        "right_hand_index_1_joint": (0.0, 1.74532925),
    }

    # torque_limits = {
    #     **G1Dof29Cfg().torque_limits,
    #     # Hands
    #     "left_hand_thumb_0_joint": 2.45,
    #     "left_hand_thumb_1_joint": 1.4,
    #     "left_hand_thumb_2_joint": 1.4,
    #     "left_hand_middle_0_joint": 1.4,
    #     "left_hand_middle_1_joint": 1.4,
    #     "left_hand_index_0_joint": 1.4,
    #     "left_hand_index_1_joint": 1.4,
    #     "right_hand_thumb_0_joint": 2.45,
    #     "right_hand_thumb_1_joint": 1.4,
    #     "right_hand_thumb_2_joint": 1.4,
    #     "right_hand_middle_0_joint": 1.4,
    #     "right_hand_middle_1_joint": 1.4,
    #     "right_hand_index_0_joint": 1.4,
    #     "right_hand_index_1_joint": 1.4,
    # }

    default_joint_positions = {
        **G1Dof29Cfg().default_joint_positions,
        # Hands
        "left_hand_thumb_0_joint": 0.0,
        "left_hand_thumb_1_joint": 0.0,
        "left_hand_thumb_2_joint": 0.0,
        "left_hand_middle_0_joint": 0.0,
        "left_hand_middle_1_joint": 0.0,
        "left_hand_index_0_joint": 0.0,
        "left_hand_index_1_joint": 0.0,
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": 0.0,
        "right_hand_middle_0_joint": 0.0,
        "right_hand_middle_1_joint": 0.0,
        "right_hand_index_0_joint": 0.0,
        "right_hand_index_1_joint": 0.0,
    }

    control_type = {
        **G1Dof29Cfg().control_type,
        # Hands
        "left_hand_thumb_0_joint": "effort",
        "left_hand_thumb_1_joint": "effort",
        "left_hand_thumb_2_joint": "effort",
        "left_hand_middle_0_joint": "effort",
        "left_hand_middle_1_joint": "effort",
        "left_hand_index_0_joint": "effort",
        "left_hand_index_1_joint": "effort",
        "right_hand_thumb_0_joint": "effort",
        "right_hand_thumb_1_joint": "effort",
        "right_hand_thumb_2_joint": "effort",
        "right_hand_middle_0_joint": "effort",
        "right_hand_middle_1_joint": "effort",
        "right_hand_index_0_joint": "effort",
        "right_hand_index_1_joint": "effort",
    }

    def __post_init__(self):
        self.cameras: list = [
            PinholeCameraCfg(
                name="front_cam",
                data_types=["rgb"],
                height=480,
                width=640,
                focal_length=7.6,
                focus_distance=400.0,
                horizontal_aperture=20.0,
                clipping_range=(0.1, 1.0e5),
                mount_to=self.name,
                mount_link="d435_link",
                mount_pos=(0, 0.0, 0),
                # mount_quat=(0.5, -0.5, 0.5, -0.5), # ros convention
                mount_quat=(1, 0, 0, 0),  # world convention
                # update_period: float = 0.02,
            )
        ]
        return super().__post_init__()
