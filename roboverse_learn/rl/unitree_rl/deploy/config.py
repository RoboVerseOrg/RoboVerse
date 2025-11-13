import copy

import numpy as np
import yaml


class G1Config:
    def __init__(self, file_path) -> None:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.kps = np.array(config["kps"])
            self.kds = np.array(config["kds"])
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            self.obs_len_history = config["obs_len_history"]

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.enable_actions_reindex = config["enable_actions_reindex"]

        # the default joint order of g1 robot
        self.default_body_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # default kps, kds, angles for the body joints
        self.default_body_kps = np.array([
            200,
            150,
            150,
            200,
            20,
            20,
            200,
            150,
            150,
            200,
            20,
            20,
            200,
            200,
            200,
            40,
            40,
            40,
            40,
            20,
            20,
            20,
            40,
            40,
            40,
            40,
            20,
            20,
            20,
        ])
        self.default_body_kds = np.array([
            5,
            5,
            5,
            5,
            4,
            4,
            5,
            5,
            5,
            5,
            4,
            4,
            5,
            5,
            5,
            10,
            10,
            10,
            10,
            4,
            4,
            4,
            10,
            10,
            10,
            10,
            4,
            4,
            4,
        ])
        self.default_body_angles = np.array([
            -0.4,
            0.0,
            0.0,
            0.8,
            -0.4,
            0.0,
            -0.4,
            0.0,
            0.0,
            0.8,
            -0.4,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])

        # default kps, kds, angles for both hand joints
        self.default_left_hand_kps = self.default_right_hand_kps = np.array([5, 5, 5, 5, 5, 5, 5])
        self.default_left_hand_kds = self.default_right_hand_kds = np.array([1, 1, 1, 1, 1, 1, 1])
        self.default_left_hand_angles = self.default_right_hand_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.default_left_hand_joint_names = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        ]

        self.default_right_hand_joint_names = [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
        ]
        joint_names = config["joint_names"]
        if self.enable_actions_reindex:
            sorted_joint_names = copy.deepcopy(joint_names)
            sorted_joint_names.sort()
            config_default2sorted_idx = [joint_names.index(joint_name) for joint_name in sorted_joint_names]
            self.kps = self.kps[config_default2sorted_idx]
            self.kds = self.kds[config_default2sorted_idx]
            self.default_angles = self.default_angles[config_default2sorted_idx]
            self.init_joint_settings_from_config(sorted_joint_names)
        else:
            self.init_joint_settings_from_config(joint_names)

    def init_joint_settings_from_config(self, joint_names):
        body_default_sorted_idx_tuples = []
        left_hand_default_sorted_idx_tuples = []
        right_hand_default_sorted_idx_tuples = []
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.default_body_joint_names:
                default_idx = self.default_body_joint_names.index(joint_name)
                body_default_sorted_idx_tuples.append([default_idx, i])
                self.default_body_kps[default_idx] = self.kps[i]
                self.default_body_kds[default_idx] = self.kds[i]
                self.default_body_angles[default_idx] = self.default_angles[i]
            elif joint_name in self.default_left_hand_joint_names:
                default_idx = self.default_left_hand_joint_names.index(joint_name)
                left_hand_default_sorted_idx_tuples.append([default_idx, i])
                self.default_left_hand_kps[default_idx] = self.kps[i]
                self.default_left_hand_kds[default_idx] = self.kds[i]
                self.default_left_hand_angles[default_idx] = self.default_angles[i]
            elif joint_name in self.default_right_hand_joint_names:
                default_idx = self.default_right_hand_joint_names.index(joint_name)
                right_hand_default_sorted_idx_tuples.append([default_idx, i])
                self.default_right_hand_kps[default_idx] = self.kps[i]
                self.default_right_hand_kds[default_idx] = self.kds[i]
                self.default_right_hand_angles[default_idx] = self.default_angles[i]
            else:
                raise ValueError(f"Invalid joint name: {joint_name}")

        self.policy_joint_names = joint_names
        self.body_default_sorted_idx_tuples = np.array(body_default_sorted_idx_tuples, dtype=np.int32)
        self.left_hand_default_sorted_idx_tuples = np.array(left_hand_default_sorted_idx_tuples, dtype=np.int32)
        self.right_hand_default_sorted_idx_tuples = np.array(right_hand_default_sorted_idx_tuples, dtype=np.int32)
