from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import imageio as iio
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from numpy.typing import NDArray
from rich.logging import RichHandler
from torchvision.utils import make_grid, save_image

from metasim.scenario.cameras import PinholeCameraCfg

# from metasim.scenario.randomization import RandomizationCfg
from metasim.scenario.render import RenderCfg
from metasim.scenario.robot import RobotCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.save_util import save_demo
from metasim.utils.state import TensorState, state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu

rootutils.setup_root(__file__, pythonpath=True)

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


@configclass
class Args:
    task: str = "put_banana"
    robot: str = "vega"
    scene: str | None = None
    render: RenderCfg = RenderCfg(mode="raytracing")
    # random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = (
        "isaacsim"
    )
    renderer: (
        Literal["isaaclab", "isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None
    ) = None

    ## Others
    num_envs: int = 1
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    ## Only in args
    save_image_dir: str | None = "test_output/tmp"
    save_video_path: str | None = "test_output/test_replay.mp4"
    stop_on_runout: bool = False
    traj_filepath: str | None = (
        "/home/balen/murphy/isaaclab_rv/2/RoboVerse/eval_trajs/trackgrasphandrelative_vega_eval_20251126_133029_v2.pkl"
    )
    """Path to trajectory file. If None, uses env.traj_filepath"""
    save_demo_dir: str | None = "test_output/demos"
    """Directory to save demo format states. If None, don't save demo format"""

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)


###########################################################
## Utils
###########################################################
def get_actions(all_actions, episode_idx: int, action_idx: int, num_envs: int, robot: RobotCfg):
    """Get actions for a specific episode and step."""
    episode_actions = all_actions[episode_idx] if episode_idx < len(all_actions) else all_actions[-1]
    envs_actions = [episode_actions] * num_envs  # Use same episode for all envs
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_states(all_states, episode_idx: int, action_idx: int, num_envs: int):
    """Get states for a specific episode and step."""
    episode_states = all_states[episode_idx] if episode_idx < len(all_states) else all_states[-1]
    envs_states = [episode_states] * num_envs  # Use same episode for all envs
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def get_runout(all_actions, action_idx: int):
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout


def _suffix_path(p: str | None, suffix: str) -> str | None:
    """Add suffix to file path before extension."""
    if p is None:
        return None
    base, ext = os.path.splitext(p)
    if ext:
        return f"{base}_{suffix}{ext}"
    return f"{p}_{suffix}"


class ObsSaver:
    """Save the observations to images or videos."""

    def __init__(self, image_dir: str | None = None, video_path: str | None = None):
        """Initialize the ObsSaver."""
        self.image_dir = image_dir
        self.video_path = video_path
        self.images: list[NDArray] = []

        self.image_idx = 0

    def add(self, state: TensorState):
        """Add the observation to the list."""
        if self.image_dir is None and self.video_path is None:
            return

        try:
            rgb_data = next(iter(state.cameras.values())).rgb
            image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        except Exception as e:
            log.error(f"Error adding observation: {e}")
            return

        if self.image_dir is not None:
            os.makedirs(self.image_dir, exist_ok=True)
            save_image(image, os.path.join(self.image_dir, f"rgb_{self.image_idx:04d}.png"))
            self.image_idx += 1

        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        self.images.append(image)

    def save(self):
        """Save the images or videos."""
        if self.video_path is not None and self.images:
            log.info(f"Saving video of {len(self.images)} frames to {self.video_path}")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
            iio.mimsave(self.video_path, self.images, fps=30)

    def clear(self):
        """Clear images for next episode."""
        self.images = []
        self.image_idx = 0


###########################################################
## Main
###########################################################
def main():
    task_cls = get_task_class(args.task)
    
    from scipy.spatial.transform import Rotation as R
    
    fx = 365.5782165527344
    fy = 365.5782165527344
    cx = 494.15985107421875
    cy = 301.70770263671875
    img_width = 960
    img_height = 600
    focal_length_mm = 2.2112011909484863
    focal_length_cm = focal_length_mm / 10.0
    horizontal_aperture_cm = img_width * focal_length_cm / fx
    
    quat_xyzw = R.from_euler("xyz", [0, 0, 0], degrees=True).as_quat()
    quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  # convert to wxyz
    translation_from_torso_l3 = (0.01742, 0.0302, 0.75528)  # z increased by 0.3m to raise camera view

    camera = PinholeCameraCfg(
        name="zed_rgb_camera",
        width=img_width,
        height=img_height,
        data_types=["rgb"],
        focal_length=focal_length_cm,
        horizontal_aperture=horizontal_aperture_cm,
        mount_to=args.robot,
        mount_link="torso_l3",
        mount_pos=translation_from_torso_l3,
        mount_quat=quat,
    )

    scene_cfg = task_cls.scenario.scene if task_cls.scenario.scene is not None else args.scene
    if scene_cfg is None:
        log.warning("Scene is not specified by task or args; proceeding with None.")

    if args.robot == "None":
        scenario = task_cls.scenario.update(
            # robots=[args.robot],
            scene=scene_cfg,
            cameras=[camera],
            # random=args.random,
            render=args.render,
            simulator=args.sim,
            renderer=args.renderer,
            num_envs=args.num_envs,
            headless=args.headless,
        )

    else:
        scenario = task_cls.scenario.update(
            robots=[args.robot],
            scene=scene_cfg,
            cameras=[camera],
            # random=args.random,
            render=args.render,
            simulator=args.sim,
            renderer=args.renderer,
            num_envs=args.num_envs,
            headless=args.headless,
        )

    num_envs: int = scenario.num_envs

    if args.sim == "isaacsim":
        scenario.update(decimation=2)
        if scenario.robots[0].name == "franka":
            # use smaller stiffness and damping for fingers for fine-grained control
            from metasim.scenario.robot import BaseActuatorCfg

            scenario.robots[0].actuators["panda_finger_joint1"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )
            scenario.robots[0].actuators["panda_finger_joint2"] = BaseActuatorCfg(
                stiffness=50, damping=15, velocity_limit=0.2, is_ee=True
            )

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")
    
    # Use specified trajectory filepath or fall back to env.traj_filepath
    traj_filepath = args.traj_filepath if args.traj_filepath is not None else env.traj_filepath
    log.info(f"Loading trajectory from: {traj_filepath}")
    
    ## Data
    tic = time.time()
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(
        traj_filepath, scenario.robots[0], env.handler
    )  # XXX: only support one robot
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")
    
    # Log trajectory info
    if all_actions is not None:
        num_episodes = len(all_actions)
        log.info(f"Loaded {num_episodes} episodes with actions")
        for ep_idx in range(min(3, num_episodes)):  # Show first 3 episodes
            log.info(f"  Episode {ep_idx} has {len(all_actions[ep_idx])} actions")
    if all_states is not None:
        num_episodes = len(all_states)
        log.info(f"Loaded {num_episodes} episodes with states")
        for ep_idx in range(min(3, num_episodes)):  # Show first 3 episodes
            log.info(f"  Episode {ep_idx} has {len(all_states[ep_idx])} states")

    # Determine number of episodes
    if args.object_states:
        if all_states is None:
            raise ValueError("All states are None, please check the trajectory file")
        num_episodes = len(all_states)
    else:
        if all_actions is None:
            raise ValueError("All actions are None, please check the trajectory file")
        num_episodes = len(all_actions)

    log.info(f"\n{'=' * 70}")
    log.info(f"REPLAYING {num_episodes} EPISODES")
    log.info(f"{'=' * 70}")

    os.makedirs("test_output", exist_ok=True)
    saved_videos = []

    ########################################################
    ## Main: Loop through all episodes
    ########################################################
    for episode_idx in range(num_episodes):
        log.info(f"\n{'=' * 70}")
        log.info(f"EPISODE {episode_idx + 1}/{num_episodes}")
        log.info(f"{'=' * 70}")

        # Setup output paths for this episode
        episode_suffix = f"episode_{episode_idx + 1:03d}"
        video_path = _suffix_path(args.save_video_path, episode_suffix)
        image_dir = _suffix_path(args.save_image_dir, episode_suffix) if args.save_image_dir else None
        
        # Setup RGB camera save directory
        camera_rgb_dir = os.path.join(os.path.dirname(video_path) if video_path else "test_output", 
                                      f"zed_rgb_camera_{episode_suffix}")
        os.makedirs(camera_rgb_dir, exist_ok=True)
        log.info(f"Zed RGB camera images will be saved to: {camera_rgb_dir}")

        obs_saver = ObsSaver(image_dir=image_dir, video_path=video_path)
        log.info(f"Video will be saved to: {video_path}")

        # Setup joint tracking log file
        joint_log_path = os.path.join(os.path.dirname(video_path) if video_path else "test_output", 
                                      f"joint_tracking_episode_{episode_idx + 1:03d}.log")
        joint_log_file = open(joint_log_path, "w")
        joint_log_file.write(f"Joint Tracking Log - Episode {episode_idx + 1}\n")
        joint_log_file.write("=" * 100 + "\n")
        log.info(f"Joint tracking log will be saved to: {joint_log_path}")

        # Get joint names for the robot
        robot_name = scenario.robots[0].name
        joint_names = env.handler.get_joint_names(robot_name)
        log.info(f"Tracking {len(joint_names)} joints for robot '{robot_name}'")

        # Collect demo data if saving demo format
        demo_data = []
        save_demo_format = args.save_demo_dir is not None

        # Determine max trajectory length for this episode (limit to 70 frames)
        max_frames = 70
        if args.object_states:
            max_traj_length = min(len(all_states[episode_idx]), max_frames)
        else:
            max_traj_length = min(len(all_actions[episode_idx]), max_frames)

        log.info(f"Episode {episode_idx + 1} trajectory length: {max_traj_length} steps (limited to {max_frames} frames)")

        ## Reset before first step (don't close, just reset)
        tic = time.time()
        obs, extras = env.reset()
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")
        obs_saver.add(obs)
        
        # Save initial zed_rgb_camera RGB image (frame -1 / reset)
        if "zed_rgb_camera" in obs.cameras:
            camera_state = obs.cameras["zed_rgb_camera"]
            if camera_state.rgb is not None:
                rgb_image = camera_state.rgb[0].cpu().numpy()  # (H, W, 3)
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)
                frame_path = os.path.join(camera_rgb_dir, "frame_reset.png")
                iio.imwrite(frame_path, rgb_image)
                log.debug(f"Saved zed_rgb_camera RGB reset frame to {frame_path}")

        # Log initial joint angles
        robot_state = obs.robots[robot_name]
        if robot_state.joint_pos is not None:
            joint_log_file.write(f"\nInitial State (Step -1 / Reset):\n")
            joint_log_file.write("-" * 100 + "\n")
            joint_log_file.write(f"{'Joint Name':<30} {'Target':<15} {'Actual':<15} {'Error':<15}\n")
            joint_log_file.write("-" * 100 + "\n")
            
            for i, joint_name in enumerate(joint_names):
                target_val = robot_state.joint_pos_target[0, i].item() if robot_state.joint_pos_target is not None else 0.0
                actual_val = robot_state.joint_pos[0, i].item()
                error = target_val - actual_val
                joint_log_file.write(f"{joint_name:<30} {target_val:>15.6f} {actual_val:>15.6f} {error:>15.6f}\n")

        # Collect initial state if saving demo format
        if save_demo_format:
            obs_nested = state_tensor_to_nested(env.handler, obs)
            demo_data.append(deepcopy(tensor_to_cpu(obs_nested[0])))

        ## Main loop for this episode
        step = 0
        success_logged = False
        while step < max_traj_length:
            log.debug(f"Episode {episode_idx + 1}, Step {step}/{max_traj_length - 1}")
            tic = time.time()
            if args.object_states:
                ## TODO: merge states replay into env.step function
                states = get_states(all_states, episode_idx, step, num_envs)
                env.handler.set_states(states)
                env.handler.refresh_render()
                obs = env.handler.get_states()

                ## XXX: hack
                success = env.checker.check(env.handler)
                if success.any() and not success_logged:
                    success_envs = success.nonzero().squeeze(-1).tolist()
                    if isinstance(success_envs, int):
                        success_envs = [success_envs]
                    log.info(f"[SUCCESS] Task completed at step {step} (envs: {success_envs})")
                    success_logged = True
                if success.all():
                    log.info("All environments succeeded, stopping early")
                    break

            else:
                actions = get_actions(all_actions, episode_idx, step, num_envs, scenario.robots[0])
                obs, reward, success, time_out, extras = env.step(actions)

                if success.any() and not success_logged:
                    success_envs = success.nonzero().squeeze(-1).tolist()
                    if isinstance(success_envs, int):
                        success_envs = [success_envs]
                    log.info(f"[SUCCESS] Task completed at step {step} (envs: {success_envs})")
                    success_logged = True

                if time_out.any():
                    log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

                if success.all() or time_out.all():
                    log.info("All environments succeeded or timed out, stopping early")
                    break

            toc = time.time()
            log.trace(f"Time to step: {toc - tic:.2f}s")

            tic = time.time()
            obs_saver.add(obs)
            
            # Save zed_rgb_camera RGB image for this frame
            if "zed_rgb_camera" in obs.cameras:
                camera_state = obs.cameras["zed_rgb_camera"]
                if camera_state.rgb is not None:
                    # RGB shape: (num_envs, H, W, 3), get first env
                    rgb_image = camera_state.rgb[0].cpu().numpy()  # (H, W, 3)
                    # Ensure values are in [0, 255] range
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)
                    # Save image
                    frame_path = os.path.join(camera_rgb_dir, f"frame_{step:04d}.png")
                    iio.imwrite(frame_path, rgb_image)
                    if step % 10 == 0:
                        log.debug(f"Saved zed_rgb_camera RGB frame {step} to {frame_path}")
            
            # Log joint angles (target vs actual)
            robot_state = obs.robots[robot_name]
            if robot_state.joint_pos is not None and robot_state.joint_pos_target is not None:
                joint_log_file.write(f"\nStep {step}:\n")
                joint_log_file.write("-" * 100 + "\n")
                joint_log_file.write(f"{'Joint Name':<30} {'Target':<15} {'Actual':<15} {'Error':<15}\n")
                joint_log_file.write("-" * 100 + "\n")
                
                for i, joint_name in enumerate(joint_names):
                    target_val = robot_state.joint_pos_target[0, i].item() if robot_state.joint_pos_target is not None else 0.0
                    actual_val = robot_state.joint_pos[0, i].item()
                    error = target_val - actual_val
                    joint_log_file.write(f"{joint_name:<30} {target_val:>15.6f} {actual_val:>15.6f} {error:>15.6f}\n")
                
                # Also print to console (first few joints or summary)
                if step % 10 == 0 or step < 5:  # Print every 10 steps or first 5 steps
                    log.info(f"[Step {step}] Joint tracking: {len(joint_names)} joints logged")
            
            # Collect demo data if saving demo format
            if save_demo_format:
                obs_nested = state_tensor_to_nested(env.handler, obs)
                demo_data.append(deepcopy(tensor_to_cpu(obs_nested[0])))
            
            toc = time.time()
            log.trace(f"Time to save obs: {toc - tic:.2f}s")
            step += 1

        if step >= max_traj_length:
            log.info(f"Reached trajectory length limit ({max_traj_length} steps)")

        # Close joint tracking log file
        joint_log_file.close()
        log.info(f"Joint tracking log saved to: {joint_log_path}")

        # Save video for this episode
        obs_saver.save()
        saved_videos.append(video_path)
        log.info(f"✓ Episode {episode_idx + 1} completed, video saved: {video_path}")
        
        # Count saved RGB images
        if os.path.exists(camera_rgb_dir):
            rgb_files = [f for f in os.listdir(camera_rgb_dir) if f.endswith('.png')]
            log.info(f"✓ Saved {len(rgb_files)} zed_rgb_camera RGB images to: {camera_rgb_dir}")

        # Save demo format if enabled
        if save_demo_format:
            # Determine success status
            success_status = "success" if success_logged else "failed"

            # Create save directory following collect_demo.py format
            # Format: {save_demo_dir}/{status}/demo_{demo_idx:04d}
            save_dir = os.path.join(args.save_demo_dir, success_status, f"demo_{episode_idx:04d}")

            os.makedirs(save_dir, exist_ok=True)
            log.info(f"Saving demo {episode_idx} to: {save_dir}")

            # Get robot config and task description
            robot_cfg = scenario.robots[0]
            task_desc = getattr(env, "task_desc", "")

            # Save demo
            save_demo(save_dir, demo_data, robot_cfg, task_desc)
            log.info(f"Demo {episode_idx} saved to: {save_dir}")

            # Also save status file
            status_path = os.path.join(save_dir, "status.txt")
            with open(status_path, "w") as f:
                f.write(success_status)
            log.info(f"Status ({success_status}) saved to: {status_path}")

        # Small delay between episodes
        if episode_idx < num_episodes - 1:
            log.info("\nPreparing next episode...")
            time.sleep(0.5)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("ALL EPISODES COMPLETED")
    log.info("=" * 70)
    log.info(f"Total episodes: {num_episodes}")
    log.info("\nGenerated videos:")
    for video in saved_videos:
        log.info(f"  - {video}")
    log.info("=" * 70)

    # Close handler only at the end
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()


if __name__ == "__main__":
    main()
