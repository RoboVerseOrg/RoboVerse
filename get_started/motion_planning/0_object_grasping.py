"""This script is used to grasp an object from a point cloud."""

import os
import time

import numpy as np
import open3d as o3d
import rootutils
import torch

# from graspnetAPI.graspnet_eval import GraspGroup
from graspnetAPI import Grasp, GraspGroup

rootutils.setup_root(__file__, pythonpath=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from third_party.gsnet.dataset.graspnet_dataset import minkowski_collate_fn
from third_party.gsnet.models.graspnet import GraspNet, pred_decode
from third_party.gsnet.utils.collision_detector import ModelFreeCollisionDetector


class GSNet:
    """This class is used to grasp an object from a point cloud."""

    def __init__(self):
        """This function is used to initialize the configuration."""
        dir = os.path.dirname(os.path.abspath(__file__))

        class Config:
            pass

        self.cfgs = Config()
        self.cfgs.dataset_root = f"{dir}/data/datasets/graspnet"
        self.cfgs.checkpoint_path = "third_party/gsnet/assets/minkuresunet_realsense_tune_epoch20.tar"
        self.cfgs.dump_dir = "logs"
        self.cfgs.seed_feat_dim = 512
        self.cfgs.camera = "realsense"
        self.cfgs.num_point = 15000
        self.cfgs.batch_size = 1
        self.cfgs.voxel_size = 0.005
        self.cfgs.collision_thresh = 0.01
        self.cfgs.voxel_size_cd = 0.01
        self.cfgs.infer = False
        self.cfgs.vis = False
        self.cfgs.scene = "0188"
        self.cfgs.index = "0000"

    def inference(self, cloud_masked, max_grasps=200):
        """This function is used to infer the grasp from the point cloud."""
        # sample points random
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
            # print("sampled point cloud idxs:", idxs.shape)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        data_dict = {
            "point_clouds": cloud_sampled.astype(np.float32),
            "coors": cloud_sampled.astype(np.float32) / self.cfgs.voxel_size,
            "feats": np.ones_like(cloud_sampled).astype(np.float32),
        }

        batch_data = minkowski_collate_fn([data_dict])
        net = GraspNet(seed_feat_dim=self.cfgs.seed_feat_dim, is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Load checkpoint
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        # print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        net.eval()
        tic = time.time()

        for key in batch_data:
            if "list" in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            if end_points is None:
                return None
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # collision detection
        if self.cfgs.collision_thresh > 0:
            cloud = data_dict["point_clouds"]

            # Model-free collision detector
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            collision_mask_mfc = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            gg = gg[~collision_mask_mfc]

            # # Franka collision detector
            # fcdetector = FrankaCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            # collision_mask_fc, global_iou_fc = fcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            # gg = gg[~collision_mask_fc]

        gg = gg.nms()
        gg = gg.sort_by_score()

        if gg.__len__() > max_grasps:
            gg = gg[:max_grasps]

        return gg

    def visualize(self, cloud, gg: GraspGroup = None, g: Grasp = None):
        """This function is used to visualize the grasp group or grasp."""
        pcd = cloud
        if gg is not None:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([pcd, *grippers])
        elif g is not None:
            gripper = g.to_open3d_geometry()
            o3d.visualization.draw_geometries([pcd, gripper])
        else:
            o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    import open3d as o3d

    # try:
    cloud = o3d.io.read_point_cloud("third_party/gsnet/assets/test.ply")

    gsnet = GSNet()
    gg = gsnet.inference(np.array(cloud.points))
    gsnet.visualize(cloud, gg)
