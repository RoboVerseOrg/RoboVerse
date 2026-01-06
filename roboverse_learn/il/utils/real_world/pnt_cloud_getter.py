import sys

import numpy as np
import pytorch3d.ops as torch3d_ops
import torch
from termcolor import cprint

sys.path.append(".")
from roboverse_learn.il.utils.real_world.pnt_cloud_generator import PointCloudGenerator

DEBUG = False

ENV_POINT_CLOUD_CONFIG = {
    "CloseBox": {
        "min_bound": [
            -0.71,
            -1.75,
            0.0,
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [0.6, 0.5, 100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 4096,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
    },
    "StackCube": {
        "min_bound": [
            -0.71,
            -1.75,
            0.027,  # 0.00
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [0.6, 0.5, 100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 4096,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
    },
    "Libero": {
        "min_bound": [
            -0.71,
            -1.75,
            0.017,  # 0.00
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [0.57, 0.5, 100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 4096,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
    },
    "RealworldLibero": {
        "min_bound": [
            -0.04,
            -0.48,
            0.061 # 0.033
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [
            0.98,#100,
            0.47,
            100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 1024,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
        "additional_cropping_box": {
            "bbox_1":{
                "min_bound": [-100, 0.18, 0.0],
                "max_bound": [0.20, 100, 100],
            },
            "bbox_2":{
                "min_bound": [-100, -100, 0.0],
                "max_bound": [0.20, -0.10, 100],
            }
        }
    },
    "RealworldPour": {
        "min_bound": [
            -0.04,
            -0.40,
            0.06 # 0.033
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [
            0.98,#100,
            0.40,
            100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 1024,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
        "additional_cropping_box": {
            "bbox_1":{
                "min_bound": [-100, 0.18, 0.0],
                "max_bound": [0.20, 100, 100],
            },
            "bbox_2":{
                "min_bound": [-100, -100, 0.0],
                "max_bound": [0.20, -0.10, 100],
            },
            "bbox_3":{
                "min_bound": [0.65, -100, -0.05],
                "max_bound": [0.80, 100, 0.08],
            },
            "bbox_4":{
                "min_bound": [0.10, -100, -0.05],
                "max_bound": [0.25, 100, 0.08],
            },
        }
    },
    "RealworldTest": {
        "min_bound": [
            -0.04,
            -0.40,
            0.06 # 0.033
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [
            0.98,#100,
            0.40,
            100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 1024,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
        "additional_cropping_box": {
            "bbox_1":{
                "min_bound": [-100, 0.18, 0.0],
                "max_bound": [0.20, 100, 100],
            },
            "bbox_2":{
                "min_bound": [-100, -100, 0.0],
                "max_bound": [0.20, -0.10, 100],
            },
            "bbox_3":{
                "min_bound": [0.15, -100, -0.05],
                "max_bound": [100, 0.05, 0.20],
            },
            "bbox_4":{
                "min_bound": [0.13, 0.05, -0.05],
                "max_bound": [100, 100, 0.30],
            },
        }
    },
    "RealworldPickBottle": {
        "min_bound": [
            -0.04,
            -0.48,
            0.061 # 0.033
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [
            0.98,#100,
            0.47,
            100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 1024,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
        "additional_cropping_box": {
            "bbox_1":{
                "min_bound": [-100, 0.18, 0.0],
                "max_bound": [0.20, 100, 100],
            },
            "bbox_2":{
                "min_bound": [-100, -100, 0.0],
                "max_bound": [0.20, -0.10, 100],
            }
        }
    },
    "RealworldManiskillLiftPegUpright": {
        "min_bound": [
            -0.04,
            -0.48,
            0.061 # 0.033
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [
            0.98,#100,
            0.45,
            100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 1024,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
        "additional_cropping_box": {
            "bbox_1":{
                "min_bound": [-100, 0.18, 0.0],
                "max_bound": [0.24, 100, 100],
            },
            "bbox_2":{
                "min_bound": [-100, -100, 0.0],
                "max_bound": [0.20, -0.10, 100],
            },
            "bbox_3":{
                "min_bound": [-100, -100, 0.30],
                "max_bound": [100, -0.40, 100],
            }
        }
    },
    "RealworldInsertCable": {
        "min_bound": [
            -0.034,
            -0.50,
            -0.05 # 0.033
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [
            0.80,#100,
            0.44,
            100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 1024,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),
        "additional_cropping_box": {
            "bbox_1":{
                "min_bound": [-100, -100, -100],
                "max_bound": [0.25, -0.20, 100],
            },
            "bbox_2":{
                "min_bound": [-100, 0.32, -100],
                "max_bound": [0.25, 100, 100],
            },
            "bbox_3":{
                "min_bound": [-100, -100, 0.25],
                "max_bound": [100, -0.40, 100],
            }
        }
    },
    "NoBox":{
        "min_bound": [
            -100,
            -100,
            -100,  # 0.00
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [100, 100, 100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 4096,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),

    },
    "RealworldMoveCamBottle":{
        "min_bound": [
            -100,
            -100,
            -100,  # 0.00
        ],  # gt approxiamately [-4.2, -2.5, -0.74] #0.0025
        "max_bound": [100, 100, 100],  # gt approxiamately [0.75, 2.45, 0.98]
        "num_points": 4096,
        "point_sampling_method": "fps",
        "cam_names": ["top"],
        "transform": None,
        "scale": np.array([1, 1, 1]),
        "offset": np.array([0, 0, 0]),

    }
}

BBOX_OFFSET_DIC = {1: [0.0, 0.0, 0.0], 25: [8.0, -8.01, 0.0], 50: [14.3, -12.01, 0.0]}


def point_cloud_sampling(point_cloud: np.ndarray, num_points: int, method: str = "fps"):
    """
    support different point cloud sampling methods
    point_cloud: (N, 6), xyz+rgb or (N, 3), xyz
    """
    if num_points == "all":  # use all points
        return point_cloud

    if point_cloud.shape[0] <= num_points:
        # cprint(f"warning: point cloud has {point_cloud.shape[0]} points, but we want to sample {num_points} points", 'yellow')
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate(
            [
                point_cloud,
                np.zeros((num_points - point_cloud.shape[0], point_cloud_dim)),
            ],
            axis=0,
        )
        return point_cloud

    if method == "uniform":
        # uniform sampling
        sampled_indices = np.random.choice(
            point_cloud.shape[0], num_points, replace=False
        )
        point_cloud = point_cloud[sampled_indices]
    elif method == "fps":
        # fast point cloud sampling using torch3d
        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(
            points=point_cloud[..., :3], K=num_points
        )
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        raise NotImplementedError(
            f"point cloud sampling method {method} not implemented"
        )

    return point_cloud


class PntCloudGetter:
    """
    fetch point cloud from mujoco and add it to obs
    """

    def __init__(self, task_name, num_envs=1, use_point_crop=True):
        if num_envs not in BBOX_OFFSET_DIC.keys():
            raise NotImplementedError(
                "The point cloud getter relies on bounding box, whose x,y boundaries will translate as num_envs change. You can either set your num_envs to one of [1,25,50], or you can verify the specific offset for your num_envs by setting DEBUG=True in pnt_cloud_getter.py and run eval.py"
            )
        self.env_cfg = ENV_POINT_CLOUD_CONFIG
        print(f"Given task name: {task_name}")
        task_name = self._get_task_name(task_name)
        print(f"Using cropping for task: {task_name}")
        # task_name = "CloseBox"
        # print(f"task_name not found, using default bounding box for task: {task_name}")
        # point cloud cropping
        self.min_bound = self.env_cfg[task_name].get("min_bound", None)
        self.max_bound = self.env_cfg[task_name].get("max_bound", None)
        self.additional_cropping_box = self.env_cfg[task_name].get(
            "additional_cropping_box", None
        )
        if self.min_bound is not None:
            self.min_bound = np.array(self.min_bound) + np.array(
                BBOX_OFFSET_DIC[num_envs]
            )
        if self.max_bound is not None:
            self.max_bound = np.array(self.max_bound) + np.array(
                BBOX_OFFSET_DIC[num_envs]
            )
        if self.additional_cropping_box is not None:
            temp_additional_cropping_box = {}
            for bbox_name, bbox in self.additional_cropping_box.items():
                bbox["min_bound"] = np.array(bbox["min_bound"]) + np.array(
                    BBOX_OFFSET_DIC[num_envs]
                )
                bbox["max_bound"] = np.array(bbox["max_bound"]) + np.array(
                    BBOX_OFFSET_DIC[num_envs]
                )
                temp_additional_cropping_box[bbox_name] = bbox
            self.additional_cropping_box = temp_additional_cropping_box
        self.use_point_crop = use_point_crop
        cprint(
            f"[MujocoPointcloudWrapper] use_point_crop: {self.use_point_crop}", "green"
        )

        # point cloud sampling
        self.num_points = self.env_cfg[task_name].get("num_points", 512)
        self.point_sampling_method = self.env_cfg[task_name].get(
            "point_sampling_method", "uniform"
        )
        cprint(
            f"[MujocoPointcloudWrapper] sampling {self.num_points} points from point cloud using {self.point_sampling_method}",
            "green",
        )
        assert self.point_sampling_method in ["uniform", "fps"], (
            f"point_sampling_method should be one of ['uniform', 'fps'], but got {self.point_sampling_method}"
        )

        # point cloud generator
        self.pc_generator = PointCloudGenerator(
            cam_names=self.env_cfg[task_name]["cam_names"]
        )
        self.pc_transform = self.env_cfg[task_name].get("transform", None)
        self.pc_scale = self.env_cfg[task_name].get("scale", None)
        self.pc_offset = self.env_cfg[task_name].get("offset", None)

    def get_point_cloud(self, rgb, depth, cam_intr, cam_extr, use_RGB=True):
        # set save_img_dir to save images for debugging
        # save_img_dir = "/home/yanjieze/projects/diffusion-for-dex/imgs"
        save_img_dir = None
        if len(rgb.shape) == 3:
            point_cloud, depth = self.pc_generator.generateCroppedPointCloud(
                rgb, depth, cam_intr, cam_extr, save_img_dir=save_img_dir, debug=DEBUG
            )  # (N, 6), xyz+rgb
            # if DEBUG:
            #    print(
            #        f"[({min(point_cloud[:, 0])}, {min(point_cloud[:, 1])}, {min(point_cloud[:, 2])}), ({max(point_cloud[:, 0])}, {max(point_cloud[:, 1])}, {max(point_cloud[:, 2])})]"
            #    )
            #    print(f"[{self.min_bound}, {self.max_bound}]")
            # do transform, scale, offset, and crop
            if self.pc_transform is not None:
                point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
            if self.pc_scale is not None:
                point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale

            if self.pc_offset is not None:
                point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

            if self.use_point_crop:
                if self.min_bound is not None:
                    mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                    point_cloud = point_cloud[mask]
                if self.max_bound is not None:
                    mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                    point_cloud = point_cloud[mask]
                if self.additional_cropping_box is not None:
                    mask_out = np.zeros(point_cloud.shape[0], dtype=bool)
                    for bbox in self.additional_cropping_box.values():
                        in_box = (
                            np.all(point_cloud[:, :3] >= bbox["min_bound"], axis=1)
                            & np.all(point_cloud[:, :3] <= bbox["max_bound"], axis=1)
                        )
                        mask_out |= in_box
                    point_cloud = point_cloud[~mask_out]
            # sampling to fixed number of points
            point_cloud = point_cloud_sampling(
                point_cloud=point_cloud,
                num_points=self.num_points,
                method=self.point_sampling_method,
            )

            if not use_RGB:
                point_cloud = point_cloud[:, :3]

            device = getattr(rgb, "device", torch.device("cpu"))
            return torch.from_numpy(point_cloud).to(device).float()

        elif len(rgb.shape) == 4:  # [N_env,C,H,W]
            N_env = rgb.shape[0]
            pointcloud_batch = []
            for env in range(N_env):
                single_rgb = rgb[env]
                single_depth = np.ascontiguousarray(
                    depth[env].cpu().numpy().astype(np.float32)
                )
                single_cam_intr = cam_intr[env]
                single_cam_extr = cam_extr[env]
                point_cloud = self.get_point_cloud(
                    single_rgb,
                    single_depth,
                    single_cam_intr,
                    single_cam_extr,
                    use_RGB=use_RGB,
                )
                pointcloud_batch.append(point_cloud.cpu())
            pointcloud_batch = np.stack(pointcloud_batch, axis=0)
            return pointcloud_batch

    def _get_task_name(self, task_name):
        if task_name in self.env_cfg:
            return task_name
        matches = [key for key in self.env_cfg.keys() if key in task_name]
        if matches:
            return max(matches, key=len)
        raise NotImplementedError(
            f"task_name {task_name} not in self.env_cfg, only support: {list(self.env_cfg.keys())}"
        )
