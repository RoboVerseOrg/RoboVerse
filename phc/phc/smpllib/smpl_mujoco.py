import threading
import math
import glfw
import cv2
from scipy.spatial.transform import Rotation as sRot
import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from uhc.utils.image_utils import write_frames_to_video
from uhc.khrylib.utils import get_body_qposaddr, get_body_qveladdr
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES
from uhc.smpllib.smpl_parser import SMPL_Parser

from uhc.utils.torch_geometry_transforms import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from uhc.utils.transform_utils import (
    convert_aa_to_orth6d,
    convert_orth_6d_to_aa,
    vertizalize_smpl_root,
    rotation_matrix_to_angle_axis,
    convert_orth_6d_to_mat,
)
import uhc.utils.rotation_conversions as tR


class SMPLConverter:
    def __init__(self, model, new_model, smpl_model="smpl"):
        if smpl_model == "smpl":
            self.body_ws = {
                "Pelvis": 1.0,
                "L_Hip": 1.0,
                "L_Knee": 1.0,
                "L_Ankle": 1.0,
                "L_Toe": 0.0,
                "R_Hip": 1.0,
                "R_Knee": 1.0,
                "R_Ankle": 1.0,
                "R_Toe": 0.0,
                "Torso": 1.0,
                "Spine": 1.0,
                "Chest": 1.0,
                "Neck": 1.0,
                "Head": 1.0,
                "L_Thorax": 1.0,
                "L_Shoulder": 1.0,
                "L_Elbow": 1.0,
                "L_Wrist": 1.0,
                "L_Hand": 0.0,
                "R_Thorax": 1.0,
                "R_Shoulder": 1.0,
                "R_Elbow": 1.0,
                "R_Wrist": 1.0,
                "R_Hand": 0.0,
            }

            self.body_params = {
                "L_Hip": [500, 50, 1, 500],
                "L_Knee": [500, 50, 1, 500],
                "L_Ankle": [400, 40, 1, 500],
                "L_Toe": [200, 20, 1, 500],
                "R_Hip": [500, 50, 1, 500],
                "R_Knee": [500, 50, 1, 500],
                "R_Ankle": [400, 40, 1, 500],
                "R_Toe": [200, 20, 1, 500],
                "Torso": [1000, 100, 1, 500],
                "Spine": [1000, 100, 1, 500],
                "Chest": [1000, 100, 1, 500],
                "Neck": [100, 10, 1, 250],
                "Head": [100, 10, 1, 250],
                "L_Thorax": [400, 40, 1, 500],
                "L_Shoulder": [400, 40, 1, 500],
                "L_Elbow": [300, 30, 1, 150],
                "L_Wrist": [100, 10, 1, 150],
                "L_Hand": [100, 10, 1, 150],
                "R_Thorax": [400, 40, 1, 150],
                "R_Shoulder": [400, 40, 1, 250],
                "R_Elbow": [300, 30, 1, 150],
                "R_Wrist": [100, 10, 1, 150],
                "R_Hand": [100, 10, 1, 150],
            }
        elif smpl_model == "smplh" or smpl_model == "smplx":
            self.body_ws = {
                "Pelvis": 1.0,
                "L_Hip": 1.0,
                "L_Knee": 1.0,
                "L_Ankle": 1.0,
                "L_Toe": 0.0,
                "R_Hip": 1.0,
                "R_Knee": 1.0,
                "R_Ankle": 1.0,
                "R_Toe": 0.0,
                "Torso": 1.0,
                "Spine": 1.0,
                "Chest": 1.0,
                "Neck": 1.0,
                "Head": 1.0,
                "L_Thorax": 1.0,
                "L_Shoulder": 1.0,
                "L_Elbow": 1.0,
                "L_Wrist": 1.0,
                "R_Thorax": 1.0,
                "R_Shoulder": 1.0,
                "R_Elbow": 1.0,
                "R_Wrist": 1.0,
                "L_Index1": 0.3,
                "L_Index2": 0.3,
                "L_Index3": 0.3,
                "L_Middle1": 0.3,
                "L_Middle2": 0.3,
                "L_Middle3": 0.3,
                "L_Pinky1": 0.3,
                "L_Pinky2": 0.3,
                "L_Pinky3": 0.3,
                "L_Ring1": 0.3,
                "L_Ring2": 0.3,
                "L_Ring3": 0.3,
                "L_Thumb1": 0.3,
                "L_Thumb2": 0.3,
                "L_Thumb3": 0.3,
                "R_Index1": 0.3,
                "R_Index2": 0.3,
                "R_Index3": 0.3,
                "R_Middle1": 0.3,
                "R_Middle2": 0.3,
                "R_Middle3": 0.3,
                "R_pinky1": 0.3,
                "R_pinky2": 0.3,
                "R_pinky3": 0.3,
                "R_Ring1": 0.3,
                "R_Ring2": 0.3,
                "R_Ring3": 0.3,
                "R_Thumb1": 0.3,
                "R_Thumb2": 0.3,
                "R_Thumb3": 0.3,
            }

            self.body_params = {
                "L_Hip": [500, 50, 1, 500],
                "L_Knee": [500, 50, 1, 500],
                "L_Ankle": [400, 40, 1, 500],
                "L_Toe": [200, 20, 1, 500],
                "R_Hip": [500, 50, 1, 500],
                "R_Knee": [500, 50, 1, 500],
                "R_Ankle": [400, 40, 1, 500],
                "R_Toe": [200, 20, 1, 500],
                "Torso": [1000, 100, 1, 500],
                "Spine": [1000, 100, 1, 500],
                "Chest": [1000, 100, 1, 500],
                "Neck": [100, 10, 1, 250],
                "Head": [100, 10, 1, 250],
                "L_Thorax": [400, 40, 1, 500],
                "L_Shoulder": [400, 40, 1, 500],
                "L_Elbow": [300, 30, 1, 150],
                "L_Wrist": [100, 10, 1, 150],
                "R_Thorax": [400, 40, 1, 150],
                "R_Shoulder": [400, 40, 1, 250],
                "R_Elbow": [300, 30, 1, 150],
                "R_Wrist": [100, 10, 1, 150],
                "L_Index1": [100, 10, 1, 100],
                "L_Index2": [100, 10, 1, 100],
                "L_Index3": [100, 10, 1, 100],
                "L_Middle1": [100, 10, 1, 100],
                "L_Middle2": [100, 10, 1, 100],
                "L_Middle3": [100, 10, 1, 100],
                "L_Pinky1": [100, 10, 1, 100],
                "L_Pinky2": [100, 10, 1, 100],
                "L_Pinky3": [100, 10, 1, 100],
                "L_Ring1": [100, 10, 1, 100],
                "L_Ring2": [100, 10, 1, 100],
                "L_Ring3": [100, 10, 1, 100],
                "L_Thumb1": [100, 10, 1, 100],
                "L_Thumb2": [100, 10, 1, 100],
                "L_Thumb3": [100, 10, 1, 100],
                "R_Index1": [100, 10, 1, 100],
                "R_Index2": [100, 10, 1, 100],
                "R_Index3": [100, 10, 1, 100],
                "R_Middle1": [100, 10, 1, 100],
                "R_Middle2": [100, 10, 1, 100],
                "R_Middle3": [100, 10, 1, 100],
                "R_pinky1": [100, 10, 1, 100],
                "R_pinky2": [100, 10, 1, 100],
                "R_pinky3": [100, 10, 1, 100],
                "R_Ring1": [100, 10, 1, 100],
                "R_Ring2": [100, 10, 1, 100],
                "R_Ring3": [100, 10, 1, 100],
                "R_Thumb1": [100, 10, 1, 100],
                "R_Thumb2": [100, 10, 1, 100],
                "R_Thumb3": [100, 10, 1, 100],
            }

        self.model = model
        self.new_model = new_model

        self.smpl_qpos_addr = get_body_qposaddr(model)
        self.smpl_qvel_addr = get_body_qveladdr(model)
        self.new_qpos_addr = get_body_qposaddr(new_model)
        self.new_qvel_addr = get_body_qveladdr(new_model)

        self.smpl_joint_names = list(self.smpl_qpos_addr.keys())
        self.new_joint_names = list(self.new_qpos_addr.keys())
        self.smpl_nq = model.nq
        self.new_nq = new_model.nq

    def qpos_smpl_2_new(self, qpos):
        smpl_qpos_addr = self.smpl_qpos_addr
        new_qpos_addr = self.new_qpos_addr
        if len(qpos.shape) == 2:
            b_size = qpos.shape[0]
            jt_array = [
                qpos[:, smpl_qpos_addr[k][0]:smpl_qpos_addr[k][1]]
                if k in smpl_qpos_addr else np.zeros((b_size, 3))
                for k in new_qpos_addr.keys()
            ]
        else:
            jt_array = [
                qpos[smpl_qpos_addr[k][0]:smpl_qpos_addr[k][1]]
                if k in smpl_qpos_addr else np.zeros((3))
                for k in new_qpos_addr.keys()
            ]
        return np.hstack(jt_array)

    def qvel_smpl_2_new(self, qpvel):
        smpl_qvel_addr = self.smpl_qvel_addr
        new_qvel_addr = self.new_qvel_addr
        if len(qpvel.shape) == 2:
            b_size = qpvel.shape[0]
            jt_array = [
                qpvel[:, smpl_qvel_addr[k][0]:smpl_qvel_addr[k][1]]
                if k in smpl_qvel_addr else np.zeros((b_size, 3))
                for k in new_qvel_addr.keys()
            ]
        else:
            jt_array = [
                qpvel[smpl_qvel_addr[k][0]:smpl_qvel_addr[k][1]]
                if k in smpl_qvel_addr else np.zeros((3))
                for k in new_qvel_addr.keys()
            ]
        return np.hstack(jt_array)

    def qpos_new_2_smpl(self, qpos):
        new_qpos_addr = self.new_qpos_addr
        subset = np.concatenate([
            np.arange(new_qpos_addr[jt][0], new_qpos_addr[jt][1])
            for jt in self.smpl_joint_names
        ])
        if len(qpos.shape) == 2:
            return qpos[:, subset]
        else:
            return qpos[subset]

    def qvel_new_2_smpl(self, qvel):
        new_qvel_addr = self.new_qvel_addr
        subset = np.concatenate([
            np.arange(new_qvel_addr[jt][0], new_qvel_addr[jt][1])
            for jt in self.smpl_joint_names
        ])
        if len(qvel.shape) == 2:
            return qvel[:, subset]
        else:
            return qvel[subset]

    def jpos_new_2_smpl(self, jpos):
        new_qpos_names = list(self.new_qpos_addr.keys())
        subset = np.array(
            [new_qpos_names.index(jt) for jt in self.smpl_joint_names])
        if len(jpos.shape) == 1 or (len(jpos.shape) == 2
                                    and jpos.shape[1] == 3):
            return jpos.reshape(-1, 3)[subset, :]
        elif (len(jpos.shape) == 2) or len(jpos.shape) == 3:
            return jpos.reshape(jpos.shape[0], -1, 3)[:, subset, :]

    def get_new_qpos_lim(self):
        return np.max(
            self.new_model.jnt_qposadr
        ) + self.new_model.jnt_qposadr[-1] - self.new_model.jnt_qposadr[-2]

    def get_new_qvel_lim(self):
        return np.max(
            self.new_model.jnt_dofadr
        ) + self.new_model.jnt_dofadr[-1] - self.new_model.jnt_dofadr[-2]

    def get_new_body_lim(self):
        return len(self.new_model.body_names)

    def get_new_diff_weight(self):
        return np.array([
            self.body_ws[n] if n in self.body_ws else 0
            for n in self.new_joint_names
        ])

    def get_new_jkp(self):
        return np.concatenate([[self.body_params[n][0]] *
                               3 if n in self.body_ws else [50] * 3
                               for n in self.new_joint_names[1:]])

    def get_new_jkd(self):
        return np.concatenate([[self.body_params[n][1]] *
                               3 if n in self.body_ws else [5] * 3
                               for n in self.new_joint_names[1:]])

    def get_new_a_scale(self):
        return np.concatenate([[self.body_params[n][2]] *
                               3 if n in self.body_ws else [1] * 3
                               for n in self.new_joint_names[1:]])

    def get_new_torque_limit(self):
        return np.concatenate([[self.body_params[n][3]] *
                               3 if n in self.body_ws else [200] * 3
                               for n in self.new_joint_names[1:]])

 
def smplh_to_smpl(pose):
    batch_size = pose.shape[0]
    return torch.cat([pose[:, :66], torch.zeros((batch_size, 6))], dim=1)


def smpl_to_smplh(pose):
    batch_size = pose.shape[0]
    return torch.cat([pose[:, :66], torch.zeros((batch_size, 90))], dim=1)


def smpl_to_qpose(
    pose,
    mj_model,
    trans=None,
    normalize=False,
    random_root=False,
    count_offset=True,
    use_quat=False,
    euler_order="ZYX",
    model="smpl",
):
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)

    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES
        if pose.shape[-1] == 156:
            pose = smplh_to_smpl(pose)
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)
    elif model == "smplx":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)

    num_joints = len(joint_names)
    num_angles = num_joints * 3
    smpl_2_mujoco = [
        joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
        if q in joint_names
    ]

    pose = pose.reshape(-1, num_angles)

    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(
        pose.shape[0], -1, 4, 4)

    curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(
        -1, 3, 3).numpy())
    if use_quat:
        curr_spose = curr_spose.as_quat()[:, [3, 0, 1, 2]].reshape(
            curr_pose_mat.shape[0], -1)
        num_angles = num_joints * (4 if use_quat else 3)
    else:
        curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
            curr_pose_mat.shape[0], -1)

    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
    if use_quat:
        curr_qpos = np.concatenate([trans, curr_spose], axis=1)
    else:
        root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
        curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                                   axis=1)

    if count_offset:

        curr_qpos[:, :3] = trans + mj_model.body_pos[1]

    return curr_qpos


def smpl_to_qpose_multi(
    pose,
    offset,
    mujoco_body_order,
    num_people=1,
    trans=None,
    normalize=False,
    random_root=False,
    count_offset=True,
    use_quat=False,
    euler_order="ZYX",
    model="smpl",
):
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    if trans is None:
        trans = np.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)

    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES
        if pose.shape[-1] == 156:
            pose = smplh_to_smpl(pose)
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)
    elif model == "smplx":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)

    num_joints = len(joint_names)
    num_angles = num_joints * 3
    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_body_order if q in joint_names
    ]

    pose = pose.reshape(-1, num_angles)

    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(
        pose.shape[0], -1, 4, 4)

    curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(
        -1, 3, 3).numpy())
    if use_quat:
        curr_spose = curr_spose.as_quat()[:, [3, 0, 1, 2]].reshape(
            curr_pose_mat.shape[0], -1)
        num_angles = num_joints * (4 if use_quat else 3)
    else:
        curr_spose = curr_spose.as_euler(euler_order, degrees=False).reshape(
            curr_pose_mat.shape[0], -1)

    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)
    if use_quat:
        curr_qpos = np.concatenate([trans, curr_spose], axis=1)
    else:

        root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
        curr_qpos = np.concatenate((trans, root_quat, curr_spose[:, 3:]),
                                   axis=1)

    if count_offset:
        curr_qpos[:, :3] = trans + offset

    return curr_qpos


def smpl_to_qpose_torch(
    pose,
    mj_model,
    trans=None,
    normalize=False,
    random_root=False,
    count_offset=True,
    use_quat=False,
    euler_order="ZYX",
    model="smpl",
):
    """
    Expect pose to be batch_size x 72
    trans to be batch_size x 3
    differentiable 
    """
    if trans is None:
        trans = torch.zeros((pose.shape[0], 3))
        trans[:, 2] = 0.91437225
    if normalize:
        pose, trans = normalize_smpl_pose(pose, trans, random_root=random_root)

    if not torch.is_tensor(pose):
        pose = torch.tensor(pose)

    if model == "smpl":
        joint_names = SMPL_BONE_ORDER_NAMES
        if pose.shape[-1] == 156:
            pose = smplh_to_smpl(pose)
    elif model == "smplh":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)
    elif model == "smplx":
        joint_names = SMPLH_BONE_ORDER_NAMES
        if pose.shape[-1] == 72:
            pose = smpl_to_smplh(pose)

    num_joints = len(joint_names)
    num_angles = num_joints * 3
    smpl_2_mujoco = [
        joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
        if q in joint_names
    ]

    pose = pose.reshape(-1, num_angles)

    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(
        pose.shape[0], -1, 4, 4)

    curr_spose = tR.matrix_to_euler_angles(curr_pose_mat[:, :, :3, :3],
                                           convention=euler_order)
    curr_spose = curr_spose.reshape(
        -1, num_joints,
        4 if use_quat else 3)[:, smpl_2_mujoco, :].reshape(-1, num_angles)

    root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])
    curr_qpos = torch.cat((trans, root_quat, curr_spose[:, 3:]), axis=1)

    if count_offset:
        curr_qpos[:, :3] = trans + torch.from_numpy(
            mj_model.body_pos[1]).to(root_quat)

    return curr_qpos


def qpos_to_smpl(qpos, mj_model, smpl_model="smpl"):
    body_qposaddr = get_body_qposaddr(mj_model)
    batch_size = qpos.shape[0]
    trans = qpos[:, :3] - mj_model.body_pos[1]
    smpl_bones_to_use = (SMPL_BONE_ORDER_NAMES
                         if smpl_model == "smpl" else SMPLH_BONE_ORDER_NAMES)
    pose = np.zeros([batch_size, len(smpl_bones_to_use), 3])
    for ind1, bone_name in enumerate(smpl_bones_to_use):
        ind2 = body_qposaddr[bone_name]
        if ind1 == 0:
            quat = qpos[:, 3:7]
            pose[:, ind1, :] = sRot.from_quat(quat[:,
                                                   [1, 2, 3, 0]]).as_rotvec()
        else:
            pose[:,
                 ind1, :] = sRot.from_euler("ZYX",
                                            qpos[:,
                                                 ind2[0]:ind2[1]]).as_rotvec()

    return pose, trans


def qpos_to_smpl_torch(qpos, mj_model, smpl_model="smpl"):
    body_qposaddr = get_body_qposaddr(mj_model)
    batch_size = qpos.shape[0]
    trans = qpos[:, :3] - torch.from_numpy(mj_model.body_pos[1]).to(qpos)
    smpl_bones_to_use = (SMPL_BONE_ORDER_NAMES
                         if smpl_model == "smpl" else SMPLH_BONE_ORDER_NAMES)

    pose = torch.zeros([batch_size, len(smpl_bones_to_use), 3]).to(qpos)

    for ind1, bone_name in enumerate(smpl_bones_to_use):
        ind2 = body_qposaddr[bone_name]
        if ind1 == 0:
            quat = qpos[:, 3:7]
            import ipdb
            ipdb.set_trace()
            pose[:, ind1, :] = sRot.from_quat(quat[:,
                                                   [1, 2, 3, 0]]).as_rotvec()
        else:
            pose[:,
                 ind1, :] = sRot.from_euler("ZYX",
                                            qpos[:,
                                                 ind2[0]:ind2[1]]).as_rotvec()

    return pose, trans


def smpl_6d_to_qpose(full_pose, model, normalize=False):
    pose_aa = convert_orth_6d_to_aa(torch.tensor(full_pose[:, 3:]))
    trans = full_pose[:, :3]
    curr_qpose = smpl_to_qpose(pose_aa, model, trans, normalize=normalize)
    return curr_qpose


def normalize_smpl_pose(pose_aa, trans=None, random_root=False):
    root_aa = pose_aa[0, :3]
    root_rot = sRot.from_rotvec(np.array(root_aa))
    root_euler = np.array(root_rot.as_euler("xyz", degrees=False))
    target_root_euler = (
        root_euler.copy()
    )  # take away Z axis rotation so the human is always facing same direction
    if random_root:
        target_root_euler[2] = np.random.random(1) * np.pi * 2
    else:
        target_root_euler[2] = -1.57
    target_root_rot = sRot.from_euler("xyz", target_root_euler, degrees=False)
    target_root_aa = target_root_rot.as_rotvec()

    target_root_mat = target_root_rot.as_matrix()
    root_mat = root_rot.as_matrix()
    apply_mat = np.matmul(target_root_mat, np.linalg.inv(root_mat))

    if torch.is_tensor(pose_aa):
        pose_aa = vertizalize_smpl_root(pose_aa, root_vec=target_root_aa)
    else:
        pose_aa = vertizalize_smpl_root(torch.from_numpy(pose_aa),
                                        root_vec=target_root_aa)

    if not trans is None:
        trans[:, [0, 1]] -= trans[0, [0, 1]]
        trans[:, [2]] = trans[:, [2]] - trans[0, [2]] + 0.91437225
        trans = np.matmul(apply_mat, trans.T).T
    return pose_aa, trans
