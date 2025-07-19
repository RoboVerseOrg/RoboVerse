

import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

from phc.utils import torch_utils
import joblib
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import copy
import gc
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode
from uhc.utils.torch_ext import to_torch

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy




class MotionLibSMPL(MotionLibBase):

    def __init__(self, motion_file, device, fix_height=FixHeightMode.full_fix, masterfoot_conifg=None, min_length=-1, im_eval=False, multi_thread=True):
        super().__init__(motion_file=motion_file, device=device, fix_height=fix_height, masterfoot_conifg=masterfoot_conifg, min_length=min_length, im_eval=im_eval, multi_thread=multi_thread)
        
        data_dir = "data/smpl"
        smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
        smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
        smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
        self.mesh_parsers = {0: smpl_parser_n, 1: smpl_parser_m, 2: smpl_parser_f}
    
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[gender.item()]
            height_tolorance = 0.0
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
            offset = joints_curr[:, 0] - trans[:frame_check] # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".
            
            if fix_height_mode == FixHeightMode.ankle_fix:
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (((assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int() + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int() 
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int() + + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()) == 4).nonzero().squeeze()
                diff_fix = ((vertices_curr[:, pick] - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:
                
                diff_fix = ((vertices_curr - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            
            
            
            trans[..., -1] -= diff_fix
            return trans, diff_fix

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, masterfoot_config, max_len, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        np.random.seed(np.random.randint(5000)* pid)
        res = {}
        assert (len(ids) == len(motion_data_list))
        for f in range(len(motion_data_list)):
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = gender_betas[f]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = curr_file['root_trans_offset'].clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end])
            pose_quat_global = curr_file['pose_quat_global'][start:end]

            B, J, N = pose_quat_global.shape

            ##### ZL: randomize the heading ######
            if (not flags.im_eval) and (not flags.test):
                # if True:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
                pose_quat_global = (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
                trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T))
            ##### ZL: randomize the heading ######

            trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)

            if not masterfoot_config is None:
                num_bodies = len(masterfoot_config['body_names'])
                pose_quat_holder = np.zeros([B, num_bodies, N])
                pose_quat_holder[..., -1] = 1
                pose_quat_holder[...,masterfoot_config['body_to_orig_without_toe'], :] \
                    = pose_quat_global[..., masterfoot_config['orig_to_orig_without_toe'], :]

                pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["L_Toe", "L_Toe_1", "L_Toe_1_1", "L_Toe_2"]], :] = pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["L_Ankle"]], :]
                pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["R_Toe", "R_Toe_1", "R_Toe_1_1", "R_Toe_2"]], :] = pose_quat_holder[..., [masterfoot_config['body_names'].index(name) for name in ["R_Ankle"]], :]

                pose_quat_global = pose_quat_holder

            pose_quat_global = to_torch(pose_quat_global)
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans, is_local=False)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            
            if flags.real_traj:
                quest_sensor_data = to_torch(curr_file['quest_sensor_data'])
                quest_trans = quest_sensor_data[..., :3]
                quest_rot = quest_sensor_data[..., 3:]
                
                quest_trans[..., -1] -= trans_fix # Fix trans
                
                global_angular_vel = SkeletonMotion._compute_angular_velocity(quest_rot, time_delta=1 / curr_file['fps'])
                linear_vel = SkeletonMotion._compute_velocity(quest_trans, time_delta=1 / curr_file['fps'])
                quest_motion = {"global_angular_vel": global_angular_vel, "linear_vel": linear_vel, "quest_trans": quest_trans, "quest_rot": quest_rot}
                curr_motion.quest_motion = quest_motion

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)
            
            

        if not queue is None:
            queue.put(res)
        else:
            return res


    
    