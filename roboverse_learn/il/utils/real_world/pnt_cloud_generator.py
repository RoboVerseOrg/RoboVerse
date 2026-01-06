from __future__ import annotations

import numpy as np


class PointCloudGenerator:
    """从单目 RGB-D + (K, w2c) 反投影生成 xyzrgb 点云（世界系）。

    该实现用于兼容旧版 `pnt_cloud_getter.py` 中的调用：
    `generateCroppedPointCloud(rgb, depth, cam_intr, cam_extr, ...)`

    约定：
    - `cam_intr`: 3x3 相机内参矩阵 K
    - `cam_extr`: 4x4 外参矩阵 w2c（world->camera），需要取逆得到 c2w
    - `depth`: HxW (meter) 或 HxWx1
    - `rgb`: HxWx3，uint8 或 float，RGB 顺序
    """

    def __init__(self, cam_names: list[str] | None = None) -> None:
        self.cam_names = cam_names or ["top"]

    def generateCroppedPointCloud(
        self,
        rgb,
        depth,
        cam_intr,
        cam_extr,
        save_img_dir=None,
        debug: bool = False,
    ):
        # --- normalize inputs to numpy ---
        if hasattr(rgb, "detach"):
            rgb_np = rgb.detach().cpu().numpy()
        else:
            rgb_np = np.asarray(rgb)

        if hasattr(depth, "detach"):
            depth_np = depth.detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth)

        if hasattr(cam_intr, "detach"):
            K = cam_intr.detach().cpu().numpy()
        else:
            K = np.asarray(cam_intr)

        if hasattr(cam_extr, "detach"):
            w2c = cam_extr.detach().cpu().numpy()
        else:
            w2c = np.asarray(cam_extr)

        # depth: (H,W,1) -> (H,W)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]

        if rgb_np.ndim == 3 and rgb_np.shape[-1] != 3:
            raise ValueError(f"Expected rgb shape (H,W,3), got {rgb_np.shape}")
        if depth_np.ndim != 2:
            raise ValueError(f"Expected depth shape (H,W) or (H,W,1), got {depth_np.shape}")

        H, W = depth_np.shape
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        # pixel grid
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        z = depth_np.astype(np.float32)
        valid = z > 0
        if not np.any(valid):
            # 返回空点云，保持接口一致
            return np.zeros((0, 6), dtype=np.float32), depth_np

        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z

        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)[valid]  # (N,4)

        # world transform: w2c -> c2w
        c2w = np.linalg.inv(w2c)
        pts_world = (c2w @ pts_cam.T).T[:, :3].astype(np.float32)  # (N,3)

        colors = rgb_np[valid].astype(np.float32)  # (N,3)
        xyzrgb = np.concatenate([pts_world, colors], axis=1).astype(np.float32)  # (N,6)

        return xyzrgb, depth_np


