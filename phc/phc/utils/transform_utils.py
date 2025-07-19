import cv2
import numpy as np
import torch

from copycat.utils.torch_geometry_transforms import *
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as sRot
from scipy.ndimage import gaussian_filter1d


def smpl_mat_to_aa(poses):
    poses_aa = []
    for pose_frame in poses:
        pose_frames = []
        for joint in pose_frame:
            pose_frames.append(cv2.Rodrigues(joint)[0].flatten())
        pose_frames = np.array(pose_frames)
        poses_aa.append(pose_frames)
    poses_aa = np.array(poses_aa)
    return poses_aa


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  #batch*3
    y_raw = ortho6d[:, 3:6]  #batch*3

    x = normalize_vector(x_raw)  #batch*3
    z = cross_product(x, y_raw)  #batch*3
    z = normalize_vector(z)  #batch*3
    y = cross_product(z, x)  #batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    zeros = torch.zeros(z.shape, dtype=z.dtype).to(ortho6d.device)
    matrix = torch.cat((x, y, z, zeros), 2)  #batch*3*3
    return matrix


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  #batch*3

    return out


def compute_orth6d_from_rotation_matrix(rot_mats):
    rot_mats = rot_mats[:, :, :2].transpose(1, 2).reshape(-1, 6)
    return rot_mats


def convert_mat_to_6d(poses):
    if torch.is_tensor(poses):
        curr_pose = poses.to(poses.device).float().reshape(-1, 3, 3)
    else:
        curr_pose = torch.tensor(poses).to(poses.device).float().reshape(-1, 3, 3)

    orth6d = curr_pose[:, :, :2].transpose(1, 2).reshape(-1, 6)
    orth6d = orth6d.view(poses.shape[0], -1, 6)
    return orth6d


def convert_aa_to_orth6d(poses):
    if torch.is_tensor(poses):
        curr_pose = poses.to(poses.device).float().reshape(-1, 3)
    else:
        curr_pose = torch.from_numpy(poses).to(poses.device).float().reshape(-1, 3)
    rot_mats = angle_axis_to_rotation_matrix(curr_pose)
    rot_mats = rot_mats[:, :3, :]
    orth6d = compute_orth6d_from_rotation_matrix(rot_mats)
    orth6d = orth6d.view(poses.shape[0], -1, 6)
    return orth6d


def convert_orth_6d_to_aa(orth6d):
    orth6d_flat = orth6d.reshape(-1, 6)
    rot_mat6d = compute_rotation_matrix_from_ortho6d(orth6d_flat)
    pose_aa = rotation_matrix_to_angle_axis(rot_mat6d)

    shape_curr = list(orth6d.shape)
    shape_curr[-1] /= 2
    shape_curr = tuple([int(i) for i in shape_curr])
    pose_aa = pose_aa.reshape(shape_curr)
    return pose_aa


def convert_orth_6d_to_mat(orth6d):
    num_joints = int(orth6d.shape[-1] / 6)
    orth6d_flat = orth6d.reshape(-1, 6)

    rot_mat6d = compute_rotation_matrix_from_ortho6d(orth6d_flat)[:, :, :3]

    shape_curr = list(orth6d.shape)
    shape_curr[-1] = num_joints
    shape_curr += [3, 3]
    shape_curr = tuple([int(i) for i in shape_curr])
    rot_mat = rot_mat6d.reshape(shape_curr)
    return rot_mat


def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.tensor([1e-8], dtype=v_mag.dtype).to(v.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def vertizalize_smpl_root(poses, root_vec=[np.pi / 2, 0, 0]):
    device = poses.device
    target_mat = angle_axis_to_rotation_matrix(torch.tensor([root_vec], dtype=poses.dtype).to(device))[:, :3, :3].to(device)
    org_mats = angle_axis_to_rotation_matrix(poses[:, :3])[:, :3, :3].to(device)
    org_mat_inv = torch.inverse(org_mats[0]).to(device)
    apply_mat = torch.matmul(target_mat, org_mat_inv)
    res_root_mat = torch.matmul(apply_mat, org_mats)
    zeros = torch.zeros((res_root_mat.shape[0], res_root_mat.shape[1], 1), dtype=res_root_mat.dtype).to(device)
    res_root_mats_4 = torch.cat((res_root_mat, zeros), 2)  #batch*3*4
    res_root_aa = rotation_matrix_to_angle_axis(res_root_mats_4)

    poses[:, :3] = res_root_aa
    # print(res_root_aa)
    return poses


def vertizalize_smpl_root_and_trans(poses, trans, root_vec=[np.pi / 2, 0, 0]):
    device = poses.device
    target_mat = angle_axis_to_rotation_matrix(torch.tensor([root_vec], dtype=poses.dtype).to(device))[:, :3, :3].to(device)
    org_mats = angle_axis_to_rotation_matrix(poses[:, :3])[:, :3, :3].to(device)
    org_mat_inv = torch.inverse(org_mats[0]).to(device)
    apply_mat = torch.matmul(target_mat, org_mat_inv)
    res_root_mat = torch.matmul(apply_mat, org_mats)
    zeros = torch.zeros((res_root_mat.shape[0], res_root_mat.shape[1], 1), dtype=res_root_mat.dtype).to(device)
    res_root_mats_4 = torch.cat((res_root_mat, zeros), 2)  #batch*3*4
    res_root_aa = rotation_matrix_to_angle_axis(res_root_mats_4)

    trans = torch.matmul(apply_mat, trans[:, :, None])

    poses[:, :3] = res_root_aa
    # print(res_root_aa)
    return poses, trans.squeeze()


def rotate_smpl_root_and_trans(poses, trans, root_vec=[np.pi / 2, 0, 0]):
    device = poses.device
    org_mats = angle_axis_to_rotation_matrix(poses[:, :3])[:, :3, :3].to(device)
    apply_mat = angle_axis_to_rotation_matrix(torch.tensor([root_vec], dtype=poses.dtype).to(device))[:, :3, :3].to(device)
    res_root_mat = torch.matmul(apply_mat, org_mats)
    zeros = torch.zeros((res_root_mat.shape[0], res_root_mat.shape[1], 1), dtype=res_root_mat.dtype).to(device)
    res_root_mats_4 = torch.cat((res_root_mat, zeros), 2)  #batch*3*4
    res_root_aa = rotation_matrix_to_angle_axis(res_root_mats_4)

    trans = torch.matmul(apply_mat, trans[:, :, None])

    poses[:, :3] = res_root_aa
    # print(res_root_aa)
    return poses, trans.squeeze()


def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


def perspective_projection_cam(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints, rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device), translation=pred_cam_t, focal_length=5000., camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q - 1] - quat[q], axis=0) > np.linalg.norm(quat[q - 1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat


def quat_correct_two_batch(quats_prev, quats_next):
    selects = np.linalg.norm(quats_prev - quats_next, axis=1) > np.linalg.norm(quats_prev + quats_next, axis=1)
    quats_next[selects] = -quats_next[selects]
    return quats_next


def quat_smooth_window(quats, sigma=5):
    quats = quat_correct(quats)
    quats = gaussian_filter1d(quats, sigma, axis=0)
    quats /= np.linalg.norm(quats, axis=1)[:, None]
    return quats


def smooth_smpl_quat_window(pose_aa, sigma=5):
    batch = pose_aa.shape[0]
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch, -1, 4)
    pose_quat = pose_quat[:, :, [1, 2, 3, 0]].copy()

    quats_all = []
    for i in range(pose_quat.shape[1]):
        quats = pose_quat[:, i, :].copy()
        quats_all.append(quat_smooth_window(quats, sigma))

    pose_quat_smooth = np.stack(quats_all, axis=1)[:, :, [3, 0, 1, 2]]

    pose_rot_vec = (sRot.from_quat(pose_quat_smooth.reshape(-1, 4)).as_rotvec().reshape(batch, -1, 3))
    return pose_rot_vec