from __future__ import annotations

import datetime
import random
import torch
import os
import sys
import time
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Optional
import dill
import rootutils

# Important: Add project root to pythonpath first, then we can import `roboverse_learn.*`
rootutils.setup_root(__file__, pythonpath=True)

from roboverse_learn.il.utils.real_world.real_world_env_ros import RealWorldEnv

import isaacgym  # type: ignore # noqa: F401

import select
import cv2
import imageio.v2 as iio
import numpy as np
import tyro
from typing import Union
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from tqdm import tqdm
from PIL import Image
from termcolor import cprint

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner
from roboverse_learn.il.runners.default_runner import DefaultRunner

input_action_seq = [
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

desired_action_seq = [
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

@dataclass
class Args:
    task: str
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaaclab"
    """Simulator backend"""
    max_demo: Optional[int] = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    level: int = 0
    """Randomization level for naming only (real-world eval does not apply DR here)"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    checkpoint_path: str = "/home/priosin/murphy/wyz/franka_sim/RoboVerse/il_outputs/ddpm_dit/track_il/checkpoints/200.ckpt"
    """Path to the checkpoint"""
    algo: str = "diffusion_policy"
    """Algorithm to use"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 2000
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""
    wrapper_class: Optional[str] = None
    """Env wrapper to use"""
    use_touch: bool = False
    """Use touch sensor"""
    use_server_robot: bool = True
    """Use server robot, if False, use local robot"""
    use_rs_pntcloud: bool = False
    """Use realsense point cloud"""
    use_ee_control: bool = False
    """Use end effector control"""
    no_wait: bool = False
    """Do not block on ENTER between demos (useful for non-interactive runs)"""

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)

DEBUG_RGB = False
DEBUG_RAND_STATE = False
DEBUG_PCD = False
DEBUG_RGBD = False


def main():
    num_envs: int = args.num_envs
    log.info(f"Using GPU device: {args.gpu_id}")
    if not args.checkpoint_path or (not os.path.exists(args.checkpoint_path)):
        raise FileNotFoundError(f"checkpoint_path does not exist: {args.checkpoint_path}")
    if num_envs != 1:
        raise ValueError("Real-world evaluation only supports num_envs=1 (RealWorldEnv currently only supports single environment actions).")

    realworld_task_name = "Realworld" + args.task.split("_")[0]
    env = RealWorldEnv(
        use_server_robot=args.use_server_robot,
        use_rs_pntcloud=args.use_rs_pntcloud,
        task_name=realworld_task_name,
    )

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = args.checkpoint_path.split("/")[-1] + "_" + time_str
    ckpt_name = f"{args.task}/{args.algo}/{args.robot}/L{args.level}/{ckpt_name}"

    # THIS IS ONLY FOR ALIGNING WITH INTERFACE, DO NOT USE IN REAL WORLD EVAL
    camera = PinholeCameraCfg(pos=(1.5, 0, 1.5), look_at=(0.0, 0.0, 0.0))
    task_cls = get_task_class(args.task)
    scenario = task_cls.scenario.copy().update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        cameras=[camera],
    )

    # Compatible with BaseEvalRunner: it reads scenario.episode_length
    scenario.episode_length = args.action_set_steps * args.max_step
    # WARNING ENDS

    # Construct workspace runner (DefaultEvalRunner will load policy from checkpoint)
    payload = torch.load(open(args.checkpoint_path, "rb"), pickle_module=dill)
    workspace = DefaultRunner(payload["cfg"], output_dir=os.getcwd())
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policyRunner = DefaultEvalRunner(
        workspace,
        scenario=scenario,
        num_envs=num_envs,
        checkpoint_path=args.checkpoint_path,
        device=f"cuda:{args.gpu_id}",
        task_name=realworld_task_name,
        subset=args.subset,
    )
    action_set_steps = 2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1

    if use_pcd(policyRunner.yaml_cfg):
        if not args.use_rs_pntcloud:
            from roboverse_learn.il.utils.real_world.pnt_cloud_getter import PntCloudGetter
            pnt_cloud_getter = PntCloudGetter(realworld_task_name, use_point_crop=True)


    total_success = 0
    total_completed = 0
    if args.max_demo is None:
        max_demos = args.task_id_range_high - args.task_id_range_low
    else:
        max_demos = args.max_demo

    print(f"Waiting for 5 seconds for camera exposure...")
    hz = 30
    sec = 5
    start_time = time.time()
    for i in tqdm(range(hz * sec)):
        vis, _ = env.camera.read_cameras()
        #time.sleep(1.0 / hz)
    end_time = time.time()
    print(f"Warm up done, took {(end_time - start_time)/(hz*sec):.2f} second for each read")


    for demo_start_idx in range(args.task_id_range_low, args.task_id_range_low + max_demos, num_envs):
        demo_end_idx = min(demo_start_idx + num_envs, max_demos)
        ## Reset before first step
        tic = time.time()
        obs = env.reset()

        policyRunner.reset()
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")
        log.info(f"ckpt to eval: {args.checkpoint_path.split('/')[-3]}")
        step = 0
        MaxStep = args.max_step
        SuccessOnce = [False] * num_envs
        TimeOut = [False] * num_envs
        images_list = []
        while step < MaxStep:
            #log.debug(f"Step {step}")
            start_time = time.time()
            new_obs = {
                "rgb": obs.cameras["camera0"].rgb,
                "joint_qpos": obs.agent_pos.squeeze(1),
            }
            if use_rgbd(policyRunner.yaml_cfg):
                new_obs["depth"] = obs.cameras["camera0"].depth  # (50, 256, 256, 1)
                assert new_obs["depth"].shape[3] == 1, f"Depth should be 1 channels, but got {new_obs['depth'].shape}"
            if use_pcd(policyRunner.yaml_cfg):
                if args.use_rs_pntcloud:
                    print("Using Realsense point cloud directly")
                    pnt_cloud = obs.cameras["camera0"].xyzrgb
                else:
                    depth = obs.cameras["camera0"].depth
                    cam_intr = obs.cameras["camera0"].intrinsics
                    cam_extr = obs.cameras["camera0"].extrinsics
                    pnt_cloud = pnt_cloud_getter.get_point_cloud(
                        new_obs["rgb"],
                        depth,
                        cam_intr.cpu(),
                        cam_extr.cpu(),
                    )
                new_obs["point_cloud"] = pnt_cloud
                if not use_spUnet_pcd(policyRunner.yaml_cfg):
                    feat_dim = get_pnt_cloud_feat_dim(policyRunner.yaml_cfg)
                    new_obs["point_cloud"] = new_obs["point_cloud"][..., :feat_dim]

            if use_sensor(policyRunner.yaml_cfg):
                new_obs["sensors"] = obs.sensors  # {sensor_name: {"force": Tensor[N_env, 3]}}
            # new_obs["rgb"] = _center_crop_and_resize(
            #     new_obs["rgb"].to(torch.float32), 256, 256)
            new_obs["rgb"] = _side_crop_and_resize(
                new_obs["rgb"].to(torch.float32),
                left_up=(235, 0),
                right_down=(775, 540),
                target_width=256,
                target_height=256,
            )
            if new_obs.get("depth", None) is not None:
                # new_obs["depth"] = _center_crop_and_resize(
                #     new_obs["depth"].to(torch.float32), 256, 256)
                new_obs["depth"] = _side_crop_and_resize(
                    new_obs["depth"].to(torch.float32),
                    left_up=(235, 0),
                    right_down=(775, 540),
                    target_width=256,
                    target_height=256,
                )
            if DEBUG_RGB:
                save_dir = f"./tmp/visualize/{args.task}L{args.level}"
                for i in range(num_envs):
                    # Get RGB image for env i (shape: (H, W, 3))
                    img = np.array(new_obs["rgb"][i].cpu())
                    #depth = np.array(new_obs["depth"][i].cpu())
                    demo_idx = demo_start_idx + i
                    demo_dir = save_dir
                    os.makedirs(demo_dir, exist_ok=True)
                    print(f"Img shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
                    img = img.astype(np.uint8)  # Ensure image is uint8 type
                    file_path = os.path.join(demo_dir, f"demo_{demo_idx:04d}.png")
                    iio.imwrite(file_path, img)
                    #depth_file_path = os.path.join(demo_dir, f"demo_{demo_idx:04d}_depth.png")

                    # Assume depth is numpy array, dtype e.g. float32 or uint16
                    # depth_min, depth_max = depth.min(), depth.max()
                    # if depth_max > depth_min:
                    #     depth_norm = (depth - depth_min) / (depth_max - depth_min)
                    # else:
                    #     # All zeros or constant image
                    #     depth_norm = np.zeros_like(depth)
                    # # Normalize to 0-255, then convert to uint8
                    # depth_uint8 = (depth_norm * 255).astype(np.uint8)
                    # depth_uint8 = np.squeeze(depth_uint8)

                    # depth_img = Image.fromarray(depth_uint8, mode="L")
                    # depth_img.save(depth_file_path)
                # env.close()
                # raise NotImplementedError()

            if DEBUG_PCD:
                save_folder = f"./tmp/visualize/{args.task}L{args.level}"
                os.makedirs(save_folder, exist_ok=True)
                for idx, single_pcd in enumerate(pnt_cloud):
                    pcd_filename = os.path.join(save_folder, f"demo_{idx:04d}_step_{step}.npy")
                    np.save(pcd_filename, single_pcd.cpu().numpy())
                # env.close()
                # raise NotImplementedError("DEBUG")

            if DEBUG_RGBD:
                save_folder = f"./tmp/visualize/{args.task}L{args.level}"
                os.makedirs(save_folder, exist_ok=True)
                rgbd = new_obs["depth"][0].cpu().numpy()
                rgbd = np.repeat(rgbd, 3, axis=-1)
                min_depth = rgbd.min()
                max_depth = rgbd.max()
                rgbd = (rgbd-min_depth) / (max_depth-min_depth) * 255.0
                rgbd = rgbd.astype(np.uint8)
                file_path = os.path.join(save_folder, f"demo_step_{step}_depth.png")
                iio.imwrite(file_path, rgbd)
                return

            images_list.append(np.array(new_obs["rgb"].cpu()))
            # for key, value in new_obs.items():
            #    print(f"Key: {key}, Value shape: {value.shape}")
            action = policyRunner.get_action(new_obs)
            # NOTE: `policyRunner.get_action` already returns action dict matching robot joint names,
            # no longer need old joint order reordering logic.
            for round_i in range(action_set_steps):
                obs = env.step(action, use_ee_control=args.use_ee_control)
                # print("Press ENTER if success", end="", flush=True)
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    line = sys.stdin.readline().strip().lower()
                    if line == "q":
                        success = [False]
                    else:
                        success = [True]
                else:
                    success = [False]
                time_out = [step >= MaxStep - 1] * num_envs

            # eval
            SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
            TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
            end_time = time.time()
            log.debug(f"Step {step} took {end_time - start_time:.2f}s")
            step += 1
            if all(SuccessOnce):
                break

        SuccessEnd = success.tolist() if isinstance(success, torch.Tensor) else success
        total_success += SuccessOnce.count(True)
        total_completed += len(SuccessOnce)
        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
            demo_idx_str = str(demo_idx).zfill(4)
            if i % args.save_video_freq == 0:
                iio.mimwrite(
                    f"tmp/{ckpt_name}/{demo_idx}.mp4",
                    [images[i] for images in images_list],
                )
            with open(f"tmp/{ckpt_name}/{demo_idx_str}.txt", "w") as f:
                f.write(f"Demo Index: {demo_idx}\n")
                f.write(f"Num Envs: {num_envs}\n")
                f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                f.write(f"TimeOut: {TimeOut[i]}\n")
                f.write(f"Cumulative Average Success Rate: {total_success / total_completed}\n")
                f.write(f"Evaling checkpoint: {args.checkpoint_path}")
        log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
        log.info("Num Envs: ", num_envs)
        log.info(f"SuccessOnce: {SuccessOnce}")
        log.info(f"SuccessEnd: {SuccessEnd}")
        log.info(f"TimeOut: {TimeOut}")
        log.info(f"Finished evaling checkpoint: {'/'.join(args.checkpoint_path.split('/')[-4:-2])}")
        log.info(f"Results saved to tmp/{ckpt_name}")
        if not args.no_wait:
            input("Waiting for reset environment, press ENTER to start the next demo")

    log.info(f"FINAL RESULTS: {total_success / total_completed}")
    with open(f"tmp/{ckpt_name}/final_stats.txt", "w") as f:
        f.write(f"Total Success: {total_success}\n")
        f.write(f"Total Completed: {total_completed}\n")
        f.write(f"Average Success Rate: {total_success / total_completed}\n")
        f.write(f"ckpt: {args.checkpoint_path}\n")
        f.write(f"random level: {args.level}\n")
    env.close()


def use_rgbd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "head_cam" in cfg.task.shape_meta.obs.keys() and (
            cfg.task.shape_meta.obs.head_cam.type == "rgbd" or cfg.task.shape_meta.obs.head_cam.type == "rgbd_resnet"
        )
    else:
        keys = cfg.dataset.obs_keys.keys()
        return "head_camera_depth" in keys


def use_pcd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "point_cloud" in cfg.task.shape_meta.obs.keys() or "pcds" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return "point_cloud" in keys or "pcds" in keys or "head_camera_pnt_cloud" in keys


def use_dp3_pcd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "pcds" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return (
            "head_camera_pnt_cloud" in keys
            and not cfg.dataset.obs_keys.head_camera_pnt_cloud.get("type", None) == "spUnet"
        )


def use_spUnet_pcd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "point_cloud" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return (
            "head_camera_pnt_cloud" in keys and cfg.dataset.obs_keys.head_camera_pnt_cloud.get("type", None) == "spUnet"
        )


def use_sensor(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "franka_panda_leftfinger_touch_sensor_pred" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return "sensors" in keys


def get_pnt_cloud_feat_dim(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return task.shape_meta.obs.point_cloud.shape[-1]
    else:
        return cfg.dataset.obs_keys.head_camera_pnt_cloud.shape[-1]


def _center_crop_and_resize(
    img: torch.Tensor,
    target_width: int,
    target_height: int
) -> torch.Tensor:
    """
    Args:
        img (torch.Tensor): Input image tensor of shape (N, H, W, C), range [0, 255], dtype uint8.
        target_width (int): Target width.
        target_height (int): Target height.
    Returns:
        torch.Tensor: Resized image tensor of shape (N, target_height, target_width, C),
                      range [0, 255], dtype uint8.
    """
    type = img.dtype
    #print(f"Input image shape: {img.shape}, dtype: {type}, min: {img.min()}, max: {img.max()}")
    N, H, W, C = img.shape
    target_ratio = target_width / target_height
    orig_ratio = W / H

    # determine crop size
    if orig_ratio > target_ratio:
        # input is wider → crop width
        new_h = H
        new_w = int(target_ratio * H)
    else:
        # input is taller → crop height
        new_w = W
        new_h = int(W / target_ratio)

    # compute crop coordinates
    left = (W - new_w) // 2
    top = (H - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    # center crop
    img_cropped = img[:, top:bottom, left:right, :]  # (N, new_h, new_w, C)

    # prepare for interpolation: to NCHW, float
    img_nchw = img_cropped.permute(0, 3, 1, 2).to(torch.float32)

    # resize with antialiasing for better quality
    img_resized = F.interpolate(
        img_nchw,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False,
        antialias=True
    )

    # back to original shape and type
    img_out = img_resized.permute(0, 2, 3, 1).to(type)  # (N, target_height, target_width, C)

    return img_out



def _side_crop_and_resize(
    img: Union[torch.Tensor, np.ndarray],
    left_up: tuple,
    right_down: tuple,
    target_width: int,
    target_height: int,
    dryrun: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Args:
        img: (N, H, W, C) or (H, W, C), range [0,255], np.ndarray or torch.Tensor
        left_up: (x_left, y_top)
        right_down: (x_right, y_bottom)
        target_width: Output width
        target_height: Output height
        dryrun: If True, pops up GUI to adjust crop box

    Returns:
        Image matching input type/dimensions:
        - Input (H, W, C) -> (target_height, target_width, C)
        - Input (N, H, W, C) -> (N, target_height, target_width, C)
    """
    is_np = isinstance(img, np.ndarray)
    need_squeeze = False

    # ---- Handle numpy input & negative stride issues ----
    if is_np:
        # Avoid negative stride / non-contiguous memory
        if (not img.flags["C_CONTIGUOUS"]) or any(s < 0 for s in img.strides):
            img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

    img_dtype = img.dtype

    # Add batch dimension if missing
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # (1, H, W, C)
        need_squeeze = True

    N, H, W, C = img.shape

    # ================= dryrun: Pop up GUI to adjust crop box =================
    if dryrun:
        try:
            import cv2
        except ImportError:
            raise ImportError("dryrun=True requires opencv-python: pip install opencv-python")

        # Use first image for visualization
        img0 = img[0]
        if img0.is_floating_point():
            img0_vis = img0.detach().cpu().clamp(0, 255).numpy().astype(np.uint8)
        else:
            img0_vis = img0.detach().cpu().numpy()

        # img0_vis is already RGB, OpenCV uses BGR for display
        img0_vis_bgr = img0_vis[..., ::-1].copy()  # .copy() to avoid negative stride

        x1_init, y1_init = left_up
        x2_init, y2_init = right_down

        # Clamp initial box to valid range
        x1_init = max(0, min(x1_init, W - 2))
        y1_init = max(0, min(y1_init, H - 2))
        x2_init = max(x1_init + 1, min(x2_init, W))
        y2_init = max(y1_init + 1, min(y2_init, H))

        window_name = "Crop dryrun (press q / Enter / Esc to confirm)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        state = {
            "left": x1_init,
            "top": y1_init,
            "right": x2_init,
            "bottom": y2_init,
        }

        def _update(_=None):
            l = cv2.getTrackbarPos("left", window_name)
            t = cv2.getTrackbarPos("top", window_name)
            r = cv2.getTrackbarPos("right", window_name)
            b = cv2.getTrackbarPos("bottom", window_name)

            # Ensure at least 1 pixel width/height
            l = max(0, min(l, W - 2))
            t = max(0, min(t, H - 2))
            r = max(l + 1, min(r, W))
            b = max(t + 1, min(b, H))

            state["left"], state["top"], state["right"], state["bottom"] = l, t, r, b

            canvas = img0_vis_bgr.copy()
            cv2.rectangle(canvas, (l, t), (r, b), (0, 255, 0), 2)

            crop_w = r - l
            crop_h = b - t
            text = f"crop_w={crop_w}, crop_h={crop_h}"
            cv2.putText(
                canvas,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, canvas)

        # Four sliders
        cv2.createTrackbar("left", window_name, x1_init, W - 1, _update)
        cv2.createTrackbar("top", window_name, y1_init, H - 1, _update)
        cv2.createTrackbar("right", window_name, x2_init, W, _update)
        cv2.createTrackbar("bottom", window_name, y2_init, H, _update)

        _update()

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 13, 27):  # q / Enter / Esc
                break

        cv2.destroyWindow(window_name)

        left_up = (state["left"], state["top"])
        right_down = (state["right"], state["bottom"])

        final_w = right_down[0] - left_up[0]
        final_h = right_down[1] - left_up[1]
        print(
            f"[dryrun] Final crop box: left={left_up[0]}, top={left_up[1]}, "
            f"right={right_down[0]}, bottom={right_down[1]}, "
            f"crop_w={final_w}, crop_h={final_h}"
        )

    # ================= Actual crop + resize =================
    x1, y1 = left_up
    x2, y2 = right_down

    x1 = max(0, min(x1, W - 2))
    y1 = max(0, min(y1, H - 2))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))

    img_cropped = img[:, y1:y2, x1:x2, :]  # (N, crop_h, crop_w, C)

    img_nchw = img_cropped.permute(0, 3, 1, 2).to(torch.float32)

    img_resized = F.interpolate(
        img_nchw,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    img_out = img_resized.permute(0, 2, 3, 1).to(img_dtype)

    if is_np:
        img_out = img_out.cpu().numpy()
    if need_squeeze:
        img_out = img_out.squeeze(0)

    return img_out


def restore_depth(depth: np.ndarray, rgb: np.ndarray = None, method: str = 'inpaint') -> np.ndarray:
    """
    Restore regions in depth map where depth value is 0.
    - method='inpaint': Navier-Stokes based inpainting.
    - method='guided': Depth completion based on ximgproc.guidedFilter.

    Args:
        depth: np.float32, depth map (in meters or same camera units), invalid values are 0.
        rgb:  np.uint8, BGR color image, only needed for 'guided' method.
        method: 'inpaint' or 'guided'.
    Returns:
        np.float32, restored depth map.
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
            raise ValueError("Must provide aligned rgb image when using 'guided' method")
        depth_f32 = depth.astype(np.float32)

        # Call guidedFilter
        try:
            guided = cv2.ximgproc.guidedFilter(guide=rgb,
                                               src=depth_f32,
                                               radius=8,
                                               eps=0.1)  # eps adjusted based on noise level
        except AttributeError:
            raise RuntimeError("cv2.ximgproc.guidedFilter not available, please install opencv-contrib-python")

        # Preserve original valid depth
        restored = guided
        restored[depth > 0] = depth_f32[depth > 0]

    else:
        raise ValueError("method must be 'inpaint' or 'guided'")

    return restored



if __name__ == "__main__":
    main()
