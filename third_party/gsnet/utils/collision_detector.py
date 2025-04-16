""" Collision detection to remove collided grasp pose predictions.
Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d

class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [numpy.ndarray, (N,3), numpy.float32]
                    the scene points to detect
                voxel_size: [float]
                    used for downsample

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.005)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """
    def __init__(self, scene_points, voxel_size=0.005):
        # self.finger_width = 0.01
        # self.finger_length = 0.06
        self.finger_width = 0.02
        self.finger_length = 0.07
        self.voxel_size = voxel_size
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
        scene_cloud = scene_cloud.voxel_down_sample(voxel_size)
        self.scene_points = np.array(scene_cloud.points)

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False, empty_thresh=0.01, return_ious=False):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        approach_dist = max(approach_dist, self.finger_width)
        T = grasp_group.translations
        R = grasp_group.rotation_matrices
        heights = grasp_group.heights[:,np.newaxis]
        depths = grasp_group.depths[:,np.newaxis]
        widths = grasp_group.widths[:,np.newaxis]
        targets = self.scene_points[np.newaxis,:,:] - T[:,np.newaxis,:]
        targets = np.matmul(targets, R)

        ## collision detection
        # height mask
        mask1 = ((targets[:,:,2] > -heights/2) & (targets[:,:,2] < heights/2))
        # left finger mask
        mask2 = ((targets[:,:,0] > depths - self.finger_length) & (targets[:,:,0] < depths))
        mask3 = (targets[:,:,1] > -(widths/2 + self.finger_width))
        mask4 = (targets[:,:,1] < -widths/2)
        # right finger mask
        mask5 = (targets[:,:,1] < (widths/2 + self.finger_width))
        mask6 = (targets[:,:,1] > widths/2)
        # bottom mask
        mask7 = ((targets[:,:,0] <= depths - self.finger_length)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width))
        # shifting mask
        mask8 = ((targets[:,:,0] <= depths - self.finger_length - self.finger_width)\
                & (targets[:,:,0] > depths - self.finger_length - self.finger_width - approach_dist))

        # get collision mask of each point
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        shifting_mask = (mask1 & mask3 & mask5 & mask8)
        global_mask = (left_mask | right_mask | bottom_mask | shifting_mask)

        # calculate equivalant volume of each part
        left_right_volume = (heights * self.finger_length * self.finger_width / (self.voxel_size**3)).reshape(-1)
        bottom_volume = (heights * (widths+2*self.finger_width) * self.finger_width / (self.voxel_size**3)).reshape(-1)
        shifting_volume = (heights * (widths+2*self.finger_width) * approach_dist / (self.voxel_size**3)).reshape(-1)
        volume = left_right_volume*2 + bottom_volume + shifting_volume

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume+1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask

        ret_value = [collision_mask,]
        if return_empty_grasp:
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            inner_volume = (heights * self.finger_length * widths / (self.voxel_size**3)).reshape(-1)
            empty_mask = (inner_mask.sum(axis=-1)/inner_volume < empty_thresh)
            ret_value.append(empty_mask)
        if return_ious:
            left_iou = left_mask.sum(axis=1) / (left_right_volume+1e-6)
            right_iou = right_mask.sum(axis=1) / (left_right_volume+1e-6)
            bottom_iou = bottom_mask.sum(axis=1) / (bottom_volume+1e-6)
            shifting_iou = shifting_mask.sum(axis=1) / (shifting_volume+1e-6)
            ret_value.append([global_iou, left_iou, right_iou, bottom_iou, shifting_iou])
        return ret_value

class FrankaCollisionDetector(ModelFreeCollisionDetector):
    def __init__(self, scene_points, voxel_size=0.005):
        super(FrankaCollisionDetector, self).__init__(scene_points, voxel_size)
        self.finger_width1 = 0.025
        self.finger_length1 = 0.04
        self.finger_height1 = 0.024

        self.finger_width2 = 0.025+0.011
        self.finger_length2 = 0.03
        self.finger_height2 = 0.024

        self.bottom_width = 0.205
        self.bottom_length = 0.075
        self.bottom_height = 0.063

    def detect(self, grasp_group, approach_dist=0.03, collision_thresh=0.05, return_empty_grasp=False,
               empty_thresh=0.01, return_ious=False):

        T = grasp_group.translations
        R = grasp_group.rotation_matrices

        heights1 = np.ones_like(grasp_group.heights)[:, np.newaxis] * self.finger_height1
        heights2 = np.ones_like(grasp_group.heights)[:, np.newaxis] * self.finger_height2
        heights3 = np.ones_like(grasp_group.heights)[:, np.newaxis] * self.bottom_height

        depths = grasp_group.depths[:, np.newaxis]
        widths = grasp_group.widths[:, np.newaxis]-0.005

        targets = self.scene_points[np.newaxis, :, :] - T[:, np.newaxis, :]
        targets = np.matmul(targets, R)

        # height 1
        mask_h1 = ((targets[:, :, 2] > -heights1 / 2) & (targets[:, :, 2] < heights1 / 2))  #z
        mask_h2 = ((targets[:, :, 2] > -heights2 / 2) & (targets[:, :, 2] < heights2 / 2))
        mask_h3 = ((targets[:, :, 2] > -heights3 / 2) & (targets[:, :, 2] < heights3 / 2))

        # left finger1
        mask_11 = ((targets[:, :, 0] > depths - self.finger_length1) & (targets[:, :, 0] < depths))  #x
        mask_12 = (targets[:, :, 1] > -(widths / 2 + self.finger_width1))                            #y
        mask_13 = (targets[:, :, 1] < -widths / 2)                                                      #y

        # left finger2
        mask_21 = ((targets[:, :, 0] > depths - self.finger_length1 - self.finger_length2) & (
                targets[:, :, 0] < depths - self.finger_length1))
        mask_22 = (targets[:, :, 1] > -(widths / 2 + self.finger_width2))
        mask_23 = (targets[:, :, 1] < -widths / 2)

        # right finger mask1
        mask_32 = (targets[:, :, 1] < (widths / 2 + self.finger_width1))
        mask_33 = (targets[:, :, 1] > widths / 2)

        # right finger mask2
        mask_42 = (targets[:, :, 1] < (widths / 2 + self.finger_width2))
        mask_43 = (targets[:, :, 1] > widths / 2)

        # bottom mask
        mask_51 = ((targets[:, :, 0] <= depths - self.finger_length1 - self.finger_length2) \
                 & (targets[:, :, 0] > depths - self.finger_length1 - self.finger_length2 - self.bottom_length))
        mask_52 = (targets[:, :, 1] < self.bottom_width / 2) & (targets[:, :, 1] >- self.bottom_width / 2)
        # shifting mask
        mask_61 = ((targets[:, :, 0] <= depths - self.finger_length1 - self.finger_length2 - self.bottom_length) \
                 & (targets[:, :, 0] > depths - self.finger_length1 - self.finger_length2 - self.bottom_length - approach_dist))

        # get collision mask of each point
        left_mask1 = (mask_11 & mask_12 & mask_13 & mask_h1)
        left_mask2 = (mask_21 & mask_22 & mask_23 & mask_h2)

        right_mask1 = (mask_11 & mask_32 & mask_33 & mask_h1)
        right_mask2 = (mask_21 & mask_42 & mask_43 & mask_h2)
        bottom_mask = (mask_51 & mask_52 & mask_h3)
        shifting_mask = (mask_61 & mask_52 & mask_h3)
        global_mask = (left_mask1 |left_mask2 | right_mask1| right_mask2 | bottom_mask | shifting_mask)

        # # get collision mask of each point
        left_right_volume1 = (heights1 * self.finger_length1 * self.finger_width1 / (self.voxel_size ** 3)).reshape(-1)
        left_right_volume2 = (heights2 * self.finger_length2 * self.finger_width2 / (self.voxel_size ** 3)).reshape(-1)
        left_right_volume3 = (heights3 * self.bottom_length * self.bottom_width / (self.voxel_size ** 3)).reshape(-1)
        left_right_volume4 = (heights3 * approach_dist * self.bottom_width / (self.voxel_size ** 3)).reshape(-1)
        volume = (left_right_volume1 + left_right_volume2 + left_right_volume3 + left_right_volume4) * 2

        # get collision iou of each part
        global_iou = global_mask.sum(axis=1) / (volume + 1e-6)

        # get collison mask
        collision_mask = (global_iou > collision_thresh)

        if not (return_empty_grasp or return_ious):
            return collision_mask, global_iou
