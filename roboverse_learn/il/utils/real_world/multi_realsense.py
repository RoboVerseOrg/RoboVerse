import os
import random
from collections import defaultdict
import sys
sys.path.append("./")
from roboverse_learn.il.utils.real_world.pnt_cloud_getter import ENV_POINT_CLOUD_CONFIG
import numpy as np
import torch
import cv2
import time
import pyrealsense2 as rs
import pytorch3d.ops as torch3d_ops
import sys
sys.path.append("./")
from roboverse_learn.il.utils.real_world.attr_dict import AttrDict
from roboverse_learn.il.utils.pytorch_util import dict_apply
from PIL import Image
from termcolor import cprint
import numpy as np
import re

# cam2base = {
#     "L515": np.array(  [[-7.37889046e-03,  7.60062505e-01, -6.49808137e-01,  1.2273651e+00],
#                     [ 9.99848507e-01, -4.63702699e-03, -1.67775763e-02,  0.000000001e-01],
#                     [-1.57651857e-02, -6.49833410e-01, -7.59913091e-01,  1.12513354e+00],
#                     [-3.71780196e-09,  6.84743026e-09, -4.15145010e-08,  1.00000008e+00]], dtype=np.float64)
# }

cam2base = {
    "L515": np.array([[-0.74375083,  0.05602508, -0.666105,    1.13072166],
                    [ 0.66804933,  0.0275028,  -0.74360855,  0.72382752],
                    [-0.02334101, -0.99805049, -0.05788277,  0.23470142],
                    [ 0.0,          0.0,          0.0,          1.0        ]], dtype=np.float64)
}
CALIBRATION_MODE = False


# def raw_data_to_extrinsics(cam_raw_data):
#     rvec = np.array(cam_raw_data["rvec"], dtype=np.float32)
#     tvec = np.array(cam_raw_data["tvec"], dtype=np.float32)
#     rmat, jcob = cv2.Rodrigues(rvec)
#     c2w = np.eye(4, dtype=np.float32)
#     c2w[:3, :3] = rmat
#     c2w[:3, 3] = tvec
#     w2c = np.linalg.inv(c2w)
#     return torch.tensor(w2c, dtype=torch.float32)


# def compose_extrinsics(T_cam_A: np.ndarray,
#                        R_A2B: np.ndarray,
#                        t_A2B: np.ndarray) -> np.ndarray:
#     assert T_cam_A.shape == (4,4)
#     assert R_A2B.shape == (3,3) and t_A2B.shape == (3,)
#     if not isinstance(T_cam_A, np.ndarray):
#         T_cam_A = np.array(T_cam_A, dtype=np.float32)
#     if not isinstance(R_A2B, np.ndarray):
#         R_A2B = np.array(R_A2B, dtype=np.float32)
#     if not isinstance(t_A2B, np.ndarray):
#         t_A2B = np.array(t_A2B, dtype=np.float32)
#     R_cam_A = T_cam_A[:3, :3]
#     t_cam_A = T_cam_A[:3,  3]

#     R_cam_B = R_A2B @ R_cam_A
#     t_cam_B = R_A2B @ t_cam_A + t_A2B

#     # 组装成 4x4 同态矩阵
#     T_cam_B = np.eye(4, dtype=T_cam_A.dtype)
#     T_cam_B[:3, :3] = R_cam_B
#     T_cam_B[:3,  3] = t_cam_B

#     return T_cam_B


def _detect_rs_model(device):
    name = device.get_info(rs.camera_info.name)               # 如 "Intel RealSense D455"
    if "l515" in name.lower():
        return "L515"
    elif "d455" in name.lower():
        return "D455"
    elif "d435" in name.lower():
        return "D435"
    else:
        print(f"Could not detect model from device name: {name}")
        import pdb; pdb.set_trace()
        raise ValueError("Unsupported RealSense model.")



def gather_realsense_cameras(task_name, enable_rgb = True, enable_depth = True, use_rs_pcd = False, restore_depth = False, set_auto_exposure = True, exposure_time = None, use_post_process = False, gain = None, num_points=1024, **stream_kwargs):
    """
    Gather all available Realsense cameras.
    :param enable_rgb: Whether to enable RGB stream.
    :param enable_depth: Whether to enable Depth stream.
    :param use_rs_pcd: Whether to use Realsense point cloud.
    :param stream_kwargs: Additional arguments for the streams.
    :return: List of RealsenseCamera objects.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    cameras = []
    if len(devices) == 0:
        print("未检测到任何 RealSense 设备。")
    else:
        for device in devices:
            serial_number = device.get_info(rs.camera_info.serial_number)
            name = _detect_rs_model(device)
            print(f"Detected RealSense Camera: {name}, S/N: {serial_number}")
            camera = RealsenseCamera(serial_number, name, enable_rgb=enable_rgb, enable_depth=enable_depth, enable_pcd=use_rs_pcd, restore_depth=restore_depth, set_auto_exposure=set_auto_exposure, exposure_time=exposure_time, gain=gain, task_name = task_name, use_post_process= use_post_process, num_points=num_points, **stream_kwargs)
            cameras.append(camera)
            print(f"--------------------------------------------")
            print(f"Camera 名称: {name}, S/N: {serial_number}")
            sensors = device.query_sensors()

            for sensor in sensors:
                print(f"\n传感器名称: {sensor.get_info(rs.camera_info.name)}")
                for profile in sensor.get_stream_profiles():
                    vprofile = profile.as_video_stream_profile()
                    fmt = vprofile.format()
                    fmt_name = str(fmt).split('.')[-1]
                    width = vprofile.width()
                    height = vprofile.height()
                    fps = vprofile.fps()
                    stream_type = vprofile.stream_type()
                    print(f"流类型: {stream_type}, 分辨率: {width}x{height}, 格式: {fmt_name}, 帧率: {fps}fps")



    return cameras


class RealsenseCamera:
    def __init__(self, serial, name, task_name="RealworldLiberoPickButter", enable_rgb=True, enable_depth=True, enable_pcd=False, restore_depth = False, fps=30, set_auto_exposure=True, exposure_time=None, gain=None, use_post_process = False, num_points=1024):
        self.serial_number = serial
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pcd = enable_pcd
        self.restore_depth = restore_depth
        self.name = name
        self.use_post_process = use_post_process
        self._init_bounding_box(task_name)
        self.set_auto_exposure = set_auto_exposure
        self.exposure_time = exposure_time
        self.gain = gain
        self.num_points = num_points
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)

        if self.enable_rgb:
            width, height = (1280, 720)#(960, 540) if self.name in ["L515", "D435"] else (640, 480)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)#1280, 800, rs.format.rgb8, fps) # D455
        if self.enable_depth:
            #self.config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, fps) # L515
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)#1280, 720, rs.format.z16, fps) # D455
        if self.enable_pcd:
            self.pcd_mapper = rs.pointcloud()
            print(f"[INFO] Num of points to sample from pcd: {self.num_points}")


        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self._started = False
        cprint(f"[Warning]: Using camera: {self.name}, please check if you are using the correct camera!", "red")

    def _init_bounding_box(self, task_name):
        task_name = self._get_task_name(task_name)
        print(f"Using cropping for task: {task_name}")
        self.min_bound = ENV_POINT_CLOUD_CONFIG[task_name].get("min_bound", None)
        self.max_bound = ENV_POINT_CLOUD_CONFIG[task_name].get("max_bound", None)
        self.additional_cropping_box = ENV_POINT_CLOUD_CONFIG[task_name].get(
            "additional_cropping_box", None
        )
        if self.min_bound is not None:
            self.min_bound = np.array(self.min_bound)
        if self.max_bound is not None:
            self.max_bound = np.array(self.max_bound)
        if self.additional_cropping_box is not None:
            temp_additional_cropping_box = {}
            for bbox_name, bbox in self.additional_cropping_box.items():
                bbox["min_bound"] = np.array(bbox["min_bound"])
                bbox["max_bound"] = np.array(bbox["max_bound"])
                temp_additional_cropping_box[bbox_name] = bbox
            self.additional_cropping_box = temp_additional_cropping_box

    def open(self):
        if not self._started:
            self.profile = self.pipeline.start(self.config)
            self._started = True
            device = self.profile.get_device()
            color_sensor = device.query_sensors()[1] #rgb
            if not self.set_auto_exposure:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, self.exposure_time)
                color_sensor.set_option(rs.option.gain, self.gain)
            else:
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            ds = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            cs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = cs.get_intrinsics()
            fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
            self.intrinsics = torch.tensor([[fx, 0, cx],
                                             [0, fy, cy],
                                             [0, 0, 1]], dtype=torch.float32)
            distCoeffs = np.array(intrinsics.coeffs)  # e.g. [k1, k2, p1, p2, k3]
            self.distCoeffs = torch.tensor(distCoeffs, dtype=torch.float32)
            self.extrinsics = self.get_extrinsics() if not CALIBRATION_MODE else None
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

    def close(self):
        if self._started:
            self.pipeline.stop()
            self._started = False

    def read_camera(self):
        frames = self.pipeline.wait_for_frames()
        raw_depth = frames.get_depth_frame() if self.enable_depth else None
        aligned_frames = self.align.process(frames)
        timestamp = aligned_frames.get_timestamp()
        depth = aligned_frames.get_depth_frame() if self.enable_depth else None
        color = aligned_frames.get_color_frame() if self.enable_rgb else None
        if self.enable_pcd:
            # start_time = time.time()
            self.pcd_mapper.map_to(color)
            #pcd = self.pcd_mapper.calculate(depth)
            pcd = self.pcd_mapper.calculate(raw_depth)

            xyz = np.asanyarray(pcd.get_vertices()).view(np.float32).reshape(-1, 3)    # (N,3) 米
            uv  = np.asanyarray(pcd.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # (N,2) [0,1]

            color_img = np.asanyarray(color.get_data())   # (H,W,3) RGB uint8
            H, W = color_img.shape[:2]
            u = np.clip(uv[:, 0], 0, 1 - 1e-6) * (W - 1)
            v = np.clip(uv[:, 1], 0, 1 - 1e-6) * (H - 1)
            ui = u.astype(np.int32)
            vi = v.astype(np.int32)
            rgb = color_img[vi, ui, :]                    # (N,3) uint8
            rgb = rgb.astype(np.float32)
            xyzrgb = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
            if self.use_post_process:
                # start_time = time.time()
                xyzrgb = preprocess_point_cloud(
                    xyzrgb, num_points=self.num_points, extrinsics=self.extrinsics,
                    min_bound=self.min_bound, max_bound=self.max_bound,
                    additional_cropping_box=self.additional_cropping_box,
                    use_cuda=True
                )
                # end_time = time.time()
                # print(f"Time taken for post-processing point cloud: {end_time - start_time:.4f} seconds")
            # mid_time = time.time()
            # xyzrgb = self.preprocess_point_cloud(xyzrgb, num_points=1024, use_cuda=True)
            # end_time = time.time()
            # print(f"Time taken for getting raw pcd: {mid_time - start_time:.4f} seconds")
            # print(f"Time taken for preprocessing pcd: {end_time - mid_time:.4f} seconds")
        depth = np.asanyarray(depth.get_data()).copy() if depth is not None else None
        color = np.asanyarray(color.get_data()).copy() if color is not None else None
        if depth is not None:
            depth = depth * self.depth_scale
        depth_meter = depth.copy() if depth is not None else None

        depth_img = (depth-depth.min()) / (depth.max() - depth.min()) if depth is not None else None

        #color = self._center_crop_and_resize(Image.fromarray(color), self.target_width, self.target_height) if color is not None else None
        #depth_img = self._center_crop_and_resize(Image.fromarray(depth_img), self.target_width, self.target_height) if depth_img is not None else None

        cam_intr = self.intrinsics
        cam_extr = self.extrinsics
        disCoeffs = self.distCoeffs

        time_stamp_dict = {
            self.serial_number: timestamp
        }
        obs_dict = {
            "rgb": color.copy(),
            "depth": depth_meter.copy(),
            #"depth_meter": depth_meter,
            #"timestamp": timestamp,
            "intrinsics": cam_intr,
            "extrinsics": cam_extr,
            #"depth_min": depth_min,
            #"depth_max": depth_max,
            #"distCoeffs": disCoeffs,
        }
        if self.enable_pcd:
            obs_dict["xyzrgb"] = xyzrgb.copy()
            #obs_dict["depth_scale"] = self.depth_scale
        if self.restore_depth and self.enable_depth:
            depth_restored = restore_depth(depth, color, method='inpaint')
            obs_dict["depth"] = depth_restored.copy()
            obs_dict["depth_raw"] = depth.copy()
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert to AttrDict for easier access
        return obs_dict, timestamp

    def is_running(self):
        return self._started

    def set_trajectory_mode(self):
        """
        Set the camera to trajectory mode.
        This is a placeholder for any specific settings needed for trajectory mode.
        """
        # In this case, we assume trajectory mode is just the default mode.
        if not self._started:
            self.open()

    def start_recording(self, filepath):
        self.config.enable_record_to_file(filepath)
        if self._started:
            self.pipeline.stop()
            self._started = False

        self.open()

    def stop_recording(self):
        self.close()

    def get_intrinsics(self):
        return {self.serial_number: {"cameraMatrix": self.intrinsics}  }

    def get_extrinsics(self):
        """
        Get the extrinsics matrix for the camera.
        :param cam_pos: Camera position in world coordinates.
        :param cam_look_at: Point in world coordinates that the camera is looking at.
        :return: Extrinsics matrix as a 4x4 tensor.
        """
        # cam_pos = torch.tensor(cam_pos, device="cuda")
        # cam_look_at = torch.tensor(cam_look_at, device="cuda")
        # c2w = torch.zeros((4, 4), device="cuda")
        # c2w[:3, 3] = cam_pos  # Set camera position
        # z = cam_look_at - cam_pos  # Camera forward vector
        # z = z / torch.norm(z)  # Normalize
        # x = torch.tensor([0.0, 1.0, 0.0], device="cuda")  # Up vector
        # y = torch.cross(z, x)
        # y = y / torch.norm(y)  # Normalize
        # c2w[:3, 0] = x  # Set camera right vector
        # c2w[:3, 1] = y  # Set camera up vector
        # c2w[:3, 2] = z  # Set camera forward vector
        # c2w[3, 3] = 1.0  # Set homogeneous
        # w2c = torch.linalg.inv(c2w.cpu())  # Inverse to get world to camera
        if "d435" in self.name.lower():
            return torch.zeros((4, 4), device="cpu")
        c2w = cam2base.get(self.name, None)
        if c2w is None:
            raise ValueError(f"Camera {self.name} not found in cam2base.")
        w2c = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32))
        return w2c

    def _get_task_name(self, task_name):
        if task_name in ENV_POINT_CLOUD_CONFIG:
            return task_name

        matches = [key for key in ENV_POINT_CLOUD_CONFIG.keys() if key in task_name]
        if matches:
            return max(matches, key=len)

        raise NotImplementedError(
            f"task_name {task_name} not in ENV_POINT_CLOUD_CONFIG, only support: {list(ENV_POINT_CLOUD_CONFIG.keys())}"
        )

def _farthest_point_sampling(points, num_points=4096, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices


def restore_depth(depth: np.ndarray, rgb: np.ndarray = None, method: str = 'inpaint') -> np.ndarray:
        """
        对深度图中深度值为0的区域进行恢复。
        - method='inpaint'：基于 Navier‑Stokes 的 inpainting。
        - method='guided'：基于 ximgproc.guidedFilter 的深度补全。

        Args:
            depth: np.float32，深度图（米或同摄像机单位），无效值为0。
            rgb:  np.uint8，BGR 彩色图，仅 guided 时需要。
            method: 'inpaint' 或 'guided'。
        Returns:
            np.float32，恢复后的深度图。
        """
        mask = (depth == 0).astype(np.uint8)

        if method == 'inpaint':
            valid = depth[depth > 0]
            if valid.size == 0:
                return depth.copy()
            d_min, d_max = valid.min(), valid.max()
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            inpainted = cv2.inpaint(depth_norm, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
            restored = inpainted.astype(np.float32) / 255 * (d_max - d_min) + d_min

        elif method == 'guided':
            if rgb is None:
                raise ValueError("使用 'guided' 方法时必须传入对齐的 rgb 图像")
            depth_f32 = depth.astype(np.float32)

            # 调用 guidedFilter
            try:
                guided = cv2.ximgproc.guidedFilter(guide=rgb,
                                                src=depth_f32,
                                                radius=8,
                                                eps=0.1)  # eps 根据噪声水平调整
            except AttributeError:
                raise RuntimeError("cv2.ximgproc.guidedFilter 不可用，请确认已安装 opencv-contrib-python")

            # 保留原有有效深度
            restored = guided
            restored[depth > 0] = depth_f32[depth > 0]

        else:
            raise ValueError("method 必须是 'inpaint' 或 'guided'")

        return restored


def preprocess_point_cloud(points, num_points, extrinsics, min_bound, max_bound, additional_cropping_box, use_cuda=True):
    extrinsics_matrix = extrinsics.cpu().numpy() if isinstance(extrinsics, torch.Tensor) else extrinsics
    c2w = np.linalg.inv(extrinsics_matrix)  # Convert to camera to world
    # scale
    point_xyz = points[..., :3]
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, c2w.T)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    # crop
    if min_bound is not None:
        mask = np.all(points[:, :3] > min_bound, axis=1)
        points = points[mask]
    if max_bound is not None:
        mask = np.all(points[:, :3] < max_bound, axis=1)
        points = points[mask]
    if additional_cropping_box is not None:
        mask_out = np.zeros(points.shape[0], dtype=bool)
        for bbox in additional_cropping_box.values():
            in_box = (
                np.all(points[:, :3] >= bbox["min_bound"], axis=1)
                & np.all(points[:, :3] <= bbox["max_bound"], axis=1)
            )
            mask_out |= in_box
        points = points[~mask_out]
    points_xyz = points[..., :3]
    # start_time = time.time()
    points_xyz, sample_indices = _farthest_point_sampling(points_xyz, num_points, use_cuda)
    # end_time = time.time()
    # print(f"Time taken for farthest point sampling: {end_time - start_time:.4f} seconds")
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points



class MultiRealsenseWrapper:
    def __init__(self, set_auto_exposure = True, exposure_time = None, gain = None, use_rs_pcd = False, use_post_process = False, restore_depth = False, task_name = "RealworldLiberoPickButter",num_points = 1024, camera_kwargs={}):
        # Open Cameras #
        rs_cameras = gather_realsense_cameras(set_auto_exposure = set_auto_exposure, exposure_time = exposure_time, gain = gain, use_rs_pcd = use_rs_pcd, restore_depth=restore_depth, task_name = task_name, use_post_process=use_post_process, num_points=num_points, **camera_kwargs)
        self.camera_dict = {cam.name: cam for cam in rs_cameras}
        print(f"Camera dict:{self.camera_dict.keys()}")
        # Launch Camera #
        self.set_trajectory_mode()

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        pass

    def disable_advanced_calibration(self):
        pass

    def set_calibration_mode(self, cam_id):
        pass

    def set_trajectory_mode(self):
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath):
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def read_cameras(self):
        full_obs_dict = {}
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        all_cam_ids.sort(reverse=True)
        for idx, cam_id in enumerate(all_cam_ids):
            # print(f"trying to read cam: {cam_id}")
            # if not self.camera_dict[cam_id].is_running():
            #     print(f"cam: {cam_id} not running!")
            #     continue
            data_dict, timestamp = self.camera_dict[cam_id].read_camera()
            #recursive_print_dic(data_dict, 0)
            #print("\n\n\n\n\n")
            data_dict = dict_apply(data_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
            full_obs_dict[f"camera{idx}"] = data_dict
        return full_obs_dict, timestamp #full_timestamp_dict

    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.close()

    def __call__(self):
        obs_dict, _ = self.read_cameras()
        return obs_dict

def recursive_print_dic(dic, intendent = 0):
    for key, value in dic.items():
        pref = "\t" * intendent
        if isinstance(value, dict):
            print(f"{pref}{key}:")
            recursive_print_dic(value, intendent + 1)
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"{pref}{key}: {value.shape}")
        elif isinstance(value, (int, float, str)):
            print(f"{pref}{key}: {value}")
        elif hasattr(value, "__len__"):
            try:
                print(f"{pref}{key}: len = {len(value)}")
            except Exception:
                print(f"{pref}{key}: {value}")
        else:
            print(f"{pref}{key}: {value}")


def main():
    # ============ 初始化 ============
    multi_realsense = MultiRealsenseWrapper(
        use_rs_pcd=False,      # 先关掉点云，避免拖慢
        restore_depth=False
    )

    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            obs_dict, _ = multi_realsense.read_cameras()

            if len(obs_dict) == 0:
                print("[WARN] No camera data received.")
                continue

            # 取任意一个相机（例如 camera0）
            cam_key = sorted(obs_dict.keys())[0]
            cam_obs = obs_dict[cam_key]

            # -------- RGB --------
            if "rgb" in cam_obs and cam_obs["rgb"] is not None:
                rgb = cam_obs["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()

                # RealSense 给的是 RGB，OpenCV 用 BGR
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"{cam_key} - RGB", rgb_bgr)

            # -------- Depth（可选）--------
            # if "depth" in cam_obs and cam_obs["depth"] is not None:
            #     depth = cam_obs["depth"]
            #     if isinstance(depth, torch.Tensor):
            #         depth = depth.cpu().numpy()

            #     # 可视化用 normalize
            #     depth_vis = cv2.normalize(
            #         depth, None, 0, 255, cv2.NORM_MINMAX
            #     ).astype(np.uint8)
            #     depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            #     cv2.imshow(f"{cam_key} - Depth", depth_vis)

            # -------- 键盘监听 --------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        print("[INFO] Shutting down cameras.")
        multi_realsense.disable_cameras()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
