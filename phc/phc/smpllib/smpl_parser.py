# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import torch
import numpy as np
import os.path as osp
from smplx import SMPL as _SMPL
from smplx import SMPLH as _SMPLH
from smplx import SMPLX as _SMPLX

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

SMPLH_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

SMPLX_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "Jaw",
    "L_Eye_Smplhf",
    "R_Eye_Smplhf",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
    "Nose",
    "R_Eye",
    "L_Eye",
    "R_Ear",
    "R_Ear",
    "L_Big_Toe",
    "L_Small_Toe",
    "L_Heel",
    "R_Big_Toe",
    "R_Small_Toe",
    "R_heel",
    "L_thumb",
    "L_index",
    "L_middle",
    "L_ring",
    "L_Pinky",
    "R_thumb",
    "R_index",
    "R_middle",
    "R_ring",
    "R_Pinky",
    "R_Eye_Bow1",
    "R_Eye_Bow2",
    "R_Eye_Bow3",
    "R_Eye_Bow4",
    "R_Eye_Bow5",
    "L_Eye_Bow5",
    "L_Eye_Bow4",
    "L_Eye_Bow3",
    "L_Eye_Bow2",
    "L_Eye_Bow1",
    "Nose1",
    "Nose2",
    "Nose3",
    "Nose4",
    "R_Nose_2",
    "R_Nose_1",
    "Nose_middle",
    "L_Nose_1",
    "L_Nose_2",
    "R_eye1",
    "R_eye2",
    "R_eye3",
    "R_eye4",
    "R_eye5",
    "R_eye6",
    "L_eye4",
    "L_eye3",
    "L_eye2",
    "L_eye1",
    "L_eye6",
    "L_eye5",
    "R_Mouth_1",
    "R_Mouth_2",
    "R_Mouth_3",
    "mouth_top",
    "L_Mouth_3",
    "L_Mouth_2",
    "L_Mouth_1",
    "L_Mouth_5",  # 59 in OpenPose output
    "L_Mouth_4",  # 58 in OpenPose output
    "Mouth_Bottom",
    "R_Mouth_4",
    "R_Mouth_5",
    "R_lip_1",
    "R_lip_2",
    "Lip_Top",
    "L_lip_2",
    "L_lip_1",
    "L_lip_3",
    "Lip_Bottom",
    "R_lip_3",
]

SMPL_EE_NAMES = ["L_Ankle", "R_Ankle", "L_Wrist", "R_Wrist", "Head"]

# SMPL_BONE_ORDER_NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', "LeftLeg", "RightLeg", "Spine1", "LeftFoot", \
#         "RightFoot", "Spine2", "LeftToe", "RightToe", "Neck", "LeftChest", "RightChest", "Mouth", "LeftShoulder", \
#          "RightShoulder", "LeftArm", "RightArm", "LeftWrist", "RightWrist", "LeftHand", "RightHand"
#         ]

JOINST_TO_USE = np.array([
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    37,
])


class SMPL_Parser(_SMPL):

    def __init__(self, create_transl=False, *args, **kwargs):
        """SMPL model constructor
        Parameters
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        data_struct: Strct
            A struct object. If given, then the parameters of the model are
            read from the object. Otherwise, the model tries to read the
            parameters from the given `model_path`. (default = None)
        create_global_orient: bool, optional
            Flag for creating a member variable for the global orientation
            of the body. (default = True)
        global_orient: torch.tensor, optional, Bx3
            The default value for the global orientation variable.
            (default = None)
        create_body_pose: bool, optional
            Flag for creating a member variable for the pose of the body.
            (default = True)
        body_pose: torch.tensor, optional, Bx(Body Joints * 3)
            The default value for the body pose variable.
            (default = None)
        create_betas: bool, optional
            Flag for creating a member variable for the shape space
            (default = True).
        betas: torch.tensor, optional, Bx10
            The default value for the shape member variable.
            (default = None)
        create_transl: bool, optional
            Flag for creating a member variable for the translation
            of the body. (default = True)
        transl: torch.tensor, optional, Bx3
            The default value for the transl variable.
            (default = None)
        dtype: torch.dtype, optional
            The data type for the created variables
        batch_size: int, optional
            The batch size used for creating the member variables
        joint_mapper: object, optional
            An object that re-maps the joints. Useful if one wants to
            re-order the SMPL joints to some other convention (e.g. MSCOCO)
            (default = None)
        gender: str, optional
            Which gender to load
        vertex_ids: dict, optional
            A dictionary containing the indices of the extra vertices that
            will be selected
        """
        super(SMPL_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPL_BONE_ORDER_NAMES

        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["x", "y", "z"] for x in self.joint_names}
        self.joint_range = {x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi]) for x in self.joint_names}
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4
        self.joint_range["L_Shoulder"] *= 4
        self.joint_range["R_Shoulder"] *= 4

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}

        # self.contype = {
        #     3: ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee','R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Neck', 'Head','L_Thorax',  'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax',  'R_Elbow', 'R_Wrist', 'R_Hand'],
        #     1: ['Chest', "L_Shoulder", "R_Shoulder"]
        #     }

        # self.conaffinity = {
        #     1: ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee','R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Neck', 'Head','L_Thorax',  'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax',  'R_Elbow', 'R_Wrist', 'R_Hand'],
        #     3: ['Chest', "L_Shoulder", "R_Shoulder"]
        # }

        self.zero_pose = torch.zeros(1, 72).float()

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 72
        """
        if pose.shape[1] != 72:
            pose = pose.reshape(-1, 72)

        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

            if th_betas.shape[-1] == 16:
                th_betas = th_betas[:, :10]

        batch_size = pose.shape[0]

        smpl_output = self.forward(
            betas=th_betas,
            transl=th_trans,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints[:, :24]
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        return vertices, joints

    def get_offsets(self, zero_pose=None, betas=torch.zeros(1, 10).float()):
        with torch.no_grad():
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            jts_np = Jtr.detach().cpu().numpy()
            parents = self.parents.cpu().numpy()
            offsets_smpl = [np.array([0, 0, 0])]
            for i in range(1, len(parents)):
                p_id = parents[i]
                p3d = jts_np[0, p_id]
                curr_3d = jts_np[0, i]
                offset_curr = curr_3d - p3d
                offsets_smpl.append(offset_curr)
            offsets_smpl = np.array(offsets_smpl)
            joint_names = self.joint_names
            joint_pos = Jtr[0].numpy()
            smpl_joint_parents = self.parents.cpu().numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in enumerate(smpl_joint_parents)}
            parents_dict = {joint_names[i]: joint_names[parents[i]] for i in range(len(joint_names))}
            channels = ["z", "y", "x"]
            skin_weights = self.lbs_weights.numpy()
            return (verts[0], jts_np[0], skin_weights, self.joint_names, joint_offsets, parents_dict, channels, self.joint_range)

    def get_mesh_offsets(self, zero_pose=None, betas=torch.zeros(1, 10), flatfoot=False):
        with torch.no_grad():
            joint_names = self.joint_names
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)

            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr[0].numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            # skin_weights = smpl_layer.th_weights.numpy()
            skin_weights = self.lbs_weights.numpy()
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )

    def get_mesh_offsets_batch(self, betas=torch.zeros(1, 10), flatfoot=False):
        with torch.no_grad():
            joint_names = self.joint_names
            verts, Jtr = self.get_joints_verts(self.zero_pose.repeat(betas.shape[0], 1), th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()

            joint_pos = Jtr
            joint_offsets = {joint_names[c]: (joint_pos[:, c] - joint_pos[:, p]) if c > 0 else joint_pos[:, c] for c, p in enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            skin_weights = self.lbs_weights
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )


class SMPLH_Parser(_SMPLH):

    def __init__(self, *args, **kwargs):
        super(SMPLH_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPLH_BONE_ORDER_NAMES
        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["z", "y", "x"] for x in self.joint_names}
        self.joint_range = {x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi]) for x in self.joint_names}
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4
        # import ipdb
        # ipdb.set_trace()

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}
        self.zero_pose = torch.zeros(1, 156).float()

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPLH_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 156
        """

        if pose.shape[1] != 156:
            pose = pose.reshape(-1, 156)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

        batch_size = pose.shape[0]
        smpl_output = self.forward(
            body_pose=pose[:, 3:66],
            global_orient=pose[:, :3],
            L_hand_pose=pose[:, 66:111],
            R_hand_pose=pose[:, 111:156],
            betas=th_betas,
            transl=th_trans,
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints
        # joints = smpl_output.joints[:,JOINST_TO_USE]
        return vertices, joints

    def get_offsets(self, betas=torch.zeros(1, 16).float()):
        with torch.no_grad():
            verts, jts = self.get_joints_verts(self.zero_pose, th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            jts_np = jts.detach().cpu().numpy()

            parents = self.parents.cpu().numpy()
            offsets_smpl = [np.array([0, 0, 0])]
            for i in range(1, len(parents)):
                p_id = parents[i]
                p3d = jts_np[0, p_id]
                curr_3d = jts_np[0, i]
                offset_curr = curr_3d - p3d
                offsets_smpl.append(offset_curr)
            offsets_smpl = np.array(offsets_smpl)
            names_smpl = self.joint_names
            offset_smpl_dict = {names_smpl[i]: offsets_smpl[i] for i in range(len(names_smpl))}
            parents_dict = {names_smpl[i]: names_smpl[parents[i]] for i in range(len(names_smpl))}
            parents_dict["Hips"] = "None"
            channels = ["z", "y", "x"]

            return offset_smpl_dict, parents_dict, channels

    def get_mesh_offsets(self, betas=torch.zeros(1, 16), flatfoot=False):
        with torch.no_grad():
            joint_names = self.joint_names
            verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)

            verts_np = verts.detach().cpu().numpy()
            verts = verts_np[0]

            if flatfoot:
                feet_subset = verts[:, 1] < np.min(verts[:, 1]) + 0.01
                verts[feet_subset, 1] = np.mean(verts[feet_subset][:, 1])

            smpl_joint_parents = self.parents.cpu().numpy()
            joint_pos = Jtr[0].numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in enumerate(smpl_joint_parents)}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents)}

            # skin_weights = smpl_layer.th_weights.numpy()
            skin_weights = self.lbs_weights.numpy()
            return (
                verts,
                joint_pos,
                skin_weights,
                joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )


class SMPLX_Parser(_SMPLX):

    def __init__(self, *args, **kwargs):
        super(SMPLX_Parser, self).__init__(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.joint_names = SMPLH_BONE_ORDER_NAMES
        self.joint_axes = {x: np.identity(3) for x in self.joint_names}
        self.joint_dofs = {x: ["z", "y", "x"] for x in self.joint_names}
        self.joint_range = {x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi]) for x in self.joint_names}
        self.joint_range["L_Elbow"] *= 4
        self.joint_range["R_Elbow"] *= 4
        # import ipdb
        # ipdb.set_trace()

        self.contype = {1: self.joint_names}
        self.conaffinity = {1: self.joint_names}
        self.zero_pose = torch.zeros(1, 156).float()
        self.joint_to_use = [SMPLX_BONE_ORDER_NAMES.index(i) for i in SMPLH_BONE_ORDER_NAMES]
        self.parents_to_use = np.concatenate([np.arange(0, 22), np.arange(25, 55)])

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPLX_Parser, self).forward(*args, **kwargs)
        return smpl_output

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        """
        Pose should be batch_size x 156
        """

        if pose.shape[1] != 156:
            pose = pose.reshape(-1, 156)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()

        batch_size = pose.shape[0]
        smpl_output = self.forward(
            body_pose=pose[:, 3:66],
            global_orient=pose[:, :3],
            left_hand_pose=pose[:, 66:111],
            right_hand_pose=pose[:, 111:156],
            betas=th_betas,
            transl=th_trans,
        )
        vertices = smpl_output.vertices
        joints = smpl_output.joints
        #         return vertices, joints
        return vertices, joints
    
     

    def get_offsets(self, v_template=None, zero_pose=None, betas=torch.zeros(1, 26).float()):
        if not v_template is None:
            self.v_template = v_template
        with torch.no_grad():
            if zero_pose is None:
                verts, Jtr = self.get_joints_verts(self.zero_pose, th_betas=betas)
            else:
                verts, Jtr = self.get_joints_verts(zero_pose, th_betas=betas)
            verts_np = verts.detach().cpu().numpy()
            jts_np = Jtr.detach().cpu().numpy()
            parents = self.parents.cpu().numpy()
            offsets_smpl = [np.array([0, 0, 0])]
            for i in range(1, len(parents)):
                p_id = parents[i]
                p3d = jts_np[0, p_id]
                curr_3d = jts_np[0, i]
                offset_curr = curr_3d - p3d
                offsets_smpl.append(offset_curr)
            offsets_smpl = np.array(offsets_smpl)
            joint_names = self.joint_names
            joint_pos = Jtr[0].numpy()
            smpl_joint_parents = self.parents.cpu().numpy()
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in enumerate(smpl_joint_parents)}
            parents_dict = {joint_names[i]: joint_names[parents[i]] for i in range(len(joint_names))}
            channels = ["z", "y", "x"]
            skin_weights = self.lbs_weights.numpy()
            return (verts[0], jts_np[0], skin_weights, self.joint_names, joint_offsets, parents_dict, channels, self.joint_range)

    def get_mesh_offsets(self, v_template=None):
        if not v_template is None:
            self.v_template = v_template
        with torch.no_grad():
            #             joint_names = self.joint_names
            joint_names = SMPLX_BONE_ORDER_NAMES
            verts, Jtr = self.get_joints_verts(self.zero_pose)

            smpl_joint_parents = self.parents.cpu().numpy()
            joint_pos = Jtr[0].numpy()
            # print(
            #     joint_pos.shape,
            #     smpl_joint_parents.shape,
            #     len(self.parents_to_use),
            #     self.parents.cpu().numpy().shape,
            # )
            joint_offsets = {joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c] for c, p in enumerate(smpl_joint_parents) if joint_names[c] in self.joint_names}
            joint_parents = {x: joint_names[i] if i >= 0 else None for x, i in zip(joint_names, smpl_joint_parents) if joint_names[i] in self.joint_names}

            verts = verts[0].numpy()
            # skin_weights = smpl_layer.th_weights.numpy()
            skin_weights = self.lbs_weights.numpy()[:, self.parents_to_use]
            return (
                verts,
                joint_pos,
                skin_weights,
                self.joint_names,
                joint_offsets,
                joint_parents,
                self.joint_axes,
                self.joint_dofs,
                self.joint_range,
                self.contype,
                self.conaffinity,
            )


if __name__ == "__main__":
    smpl_p = SMPLH_Parser("data/smpl", gender="neutral")
