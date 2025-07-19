import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np

import torch
import numpy as np
import pickle as pk
from tqdm import tqdm
from collections import defaultdict
import random
import argparse

import copy

def compute_metrics_lite(pred_pos_all, gt_pos_all, root_idx = 0, use_tqdm = True, concatenate = True):
    metrics = defaultdict(list)
    if use_tqdm:
        pbar = tqdm(range(len(pred_pos_all)))
    else:
        pbar = range(len(pred_pos_all))
        
    for idx in pbar:
        jpos_pred = pred_pos_all[idx].copy()
        jpos_gt = gt_pos_all[idx].copy()
        mpjpe_g = np.linalg.norm(jpos_gt - jpos_pred, axis=2)  * 1000
        

        vel_dist = (compute_error_vel(jpos_pred, jpos_gt)) * 1000
        accel_dist = (compute_error_accel(jpos_pred, jpos_gt)) * 1000

        jpos_pred = jpos_pred - jpos_pred[:, [root_idx]]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[:, [root_idx]]

        pa_mpjpe = p_mpjpe(jpos_pred, jpos_gt) * 1000
        mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2)* 1000
        
        metrics["mpjpe_g"].append(mpjpe_g)
        metrics["mpjpe_l"].append(mpjpe)
        metrics["mpjpe_pa"].append(pa_mpjpe)
        metrics["accel_dist"].append(accel_dist)
        metrics["vel_dist"].append(vel_dist)
    
    if concatenate:
        metrics = {k:np.concatenate(v) for k, v in metrics.items()}
    return metrics

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)

def compute_metrics(res, converter=None):
    res = copy.deepcopy(res)
    res_dict = {}

    jpos_pred = (
        converter.jpos_new_2_smpl(res["pred_jpos"])
        if converter is not None
        else res["pred_jpos"]
    )
    jpos_gt = (
        converter.jpos_new_2_smpl(res["gt_jpos"])
        if converter is not None
        else res["gt_jpos"]
    )

    traj_pred = res["pred"]
    traj_gt = res["gt"]
    batch_size = traj_pred.shape[0]
    jpos_pred = jpos_pred.reshape(batch_size, -1, 3)
    jpos_gt = jpos_gt.reshape(batch_size, -1, 3)

    root_mat_pred = get_root_matrix(traj_pred)
    root_mat_gt = get_root_matrix(traj_gt)
    root_dist = get_frobenious_norm(root_mat_pred, root_mat_gt)

    vel_dist = np.mean(compute_error_vel(jpos_pred, jpos_gt)) * 1000
    accel_dist = np.mean(compute_error_accel(jpos_pred, jpos_gt)) * 1000

    mpjpe_g = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

    if jpos_pred.shape[-2] == 24:
        jpos_pred = jpos_pred - jpos_pred[:, 0:1]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[:, 0:1]
    elif jpos_pred.shape[-2] == 14:
        jpos_pred = jpos_pred - jpos_pred[..., 7:8, :]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[..., 7:8, :]
    elif jpos_pred.shape[-2] == 12:
        jpos_pred = jpos_pred - jpos_pred[..., 7:8, :]  # zero out root
        jpos_gt = jpos_gt - jpos_gt[..., 7:8, :]

    pa_mpjpe = p_mpjpe(jpos_pred, jpos_gt) * 1000
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000
    succ = not res["fail_safe"] and res["percent"] == 1

    info = {}
    info["floor_z"] = 0
    res_dict["root_dist"] = root_dist
    res_dict["pa_mpjpe"] = pa_mpjpe
    res_dict["mpjpe"] = mpjpe
    res_dict["mpjpe_g"] = mpjpe_g
    res_dict["accel_dist"] = accel_dist
    res_dict["vel_dist"] = vel_dist
    res_dict["succ"] = succ

    if "pred_vertices" in res:
        pent = np.mean(compute_penetration(res["pred_vertices"], info))
        skate = np.mean(compute_skate(res["pred_vertices"], info))
        res_dict["pentration"] = pent
        res_dict["skate"] = skate
        del res["pred_vertices"]
        del res["gt_vertices"]
        del res["gt_joints"]
        del res["pred_joints"]
    return res_dict


def compute_penetration(vert, info):
    pen = []
    for vert_i in vert:
        vert_z = vert_i[:, 2] - info["floor_z"]
        pind = vert_z < 0
        if torch.any(pind):
            pen_i = -vert_z[pind].mean().item() * 1000
        else:
            pen_i = 0.0
        pen.append(pen_i)
    return pen


def compute_skate(vert, info):
    skate = []
    for t in range(vert.shape[0] - 1):
        cind = (vert[t, :, 2] <= info["floor_z"]) & (
            vert[t + 1, :, 2] <= info["floor_z"]
        )
        if torch.any(cind):
            offset = vert[t + 1, cind, :2] - vert[t, cind, :2]
            skate_i = torch.norm(offset, dim=1).mean().item() * 1000
        else:
            skate_i = 0.0
        skate.append(skate_i)
    return skate


def get_root_matrix(poses):
    matrices = []
    for pose in poses:
        mat = np.identity(4)
        root_pos = pose[:3]
        root_quat = pose[3:7]
        mat = quaternion_matrix(root_quat)
        mat[:3, 3] = root_pos
        matrices.append(mat)
    return matrices


def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i + 1], dt, "heading")
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels


def get_frobenious_norm(x, y):
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(4)
        error += np.linalg.norm(ident_mat - error_mat, "fro")
    return error / len(x)


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_mean_abs(x):
    return np.abs(x).mean()


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return np.mean(velocity_normed, axis=1)


def compute_error_vel(joints_gt, joints_pred, vis=None):
    vel_gt = joints_gt[1:] - joints_gt[:-1]
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)
