from __future__ import annotations

import argparse
import os
import time

import imageio
import numpy as np
import rootutils
import torch
from loguru import logger as log
from PIL import Image
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.cfg.scenario import ScenarioCfg
from metasim.utils.kinematics_utils import get_curobo_models


def images_to_video(images, video_path, frame_size=(1920, 1080), fps=30):
    if not images:
        print("No images found in the specified directory!")
        return

    writer = imageio.get_writer(video_path, fps=fps)

    for image in images:
        if image.shape[1] > frame_size[0] or image.shape[0] > frame_size[1]:
            print("Warning: frame size is smaller than the one of the images.")
            print("Images will be resized to match frame size.")
            image = np.array(Image.fromarray(image).resize(frame_size))

        writer.append_data(image)

    writer.close()
    print("Video created successfully!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--sim",
        type=str,
        default="isaaclab",
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="rdt",
        choices=["diffusion_policy", "openvla", "rdt", "act", "octo", "debug"],
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="openvla/openvla-7b",
    )
    parser.add_argument(
        "--headless",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_envs: int = args.num_envs

    from metasim.cfg.scenario import RandomizationCfg
    from metasim.cfg.sensors import PinholeCameraCfg
    from metasim.constants import SimType
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_sim_env_class

    camera = PinholeCameraCfg(
        name="camera",
        data_types=["rgb", "depth"],
        width=256,
        height=256,
        pos=(1.5, 0.0, 1.5),
        look_at=(0.0, 0.0, 0.0),
    )
    randomization = RandomizationCfg(camera=False, light=False, ground=False, reflection=False)
    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        cameras=[camera],
        random=randomization,
        sim=args.sim,
        num_envs=args.num_envs,
        try_add_table=True,
        headless=args.headless,
    )

    tic = time.time()
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    if args.algo == "rdt":
        from collections import deque

        import yaml

        from roboverse_learn.algorithms.rdt.roboverse_rdt_model import create_model

        config_path = "roboverse_learn/algorithms/rdt/configs/base.yaml"
        with open(config_path) as fp:
            config = yaml.safe_load(fp)
        pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
        pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
        pretrained_path = args.ckpt_path
        policy = create_model(
            args=config,
            dtype=torch.bfloat16,
            pretrained=pretrained_path,
            pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        )

        if os.path.exists(f"text_embed_{args.task}.pt"):
            text_embed = torch.load(f"text_embed_{args.task}.pt")
        else:
            text_embed = policy.encode_instruction("Close the box")  # TODO: Change this to your instruction
            torch.save(text_embed, f"text_embed_{args.task}.pt")

    ckpt_name = args.ckpt_path.split("/")[-1]

    ## Data
    tic = time.time()
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    # save_obs(obs, 0)

    ## cuRobo controller

    *_, robot_ik = get_curobo_models(scenario.robot)
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(scenario.robot.gripper_open_q)

    step = 0
    MaxStep = 800
    SuccessOnce = [False] * num_envs
    SuccessEnd = [False] * num_envs
    TimeOut = [False] * num_envs
    image_list = []

    if args.algo == "rdt":
        obs_window = deque(maxlen=2)
        policy.reset()
        obs_window.append(None)
        obs_window.append(np.array(obs["rgb"].squeeze(0)))
        proprio = np.array(obs["joint_qpos"]).squeeze(0)

    while step < MaxStep:
        log.debug(f"Step {step}")
        robot_joint_limits = scenario.robot.joint_limits

        import imageio

        os.makedirs("tmp/policy_test", exist_ok=True)
        imageio.imwrite(f"tmp/policy_test/obs_{step}.png", np.array(obs["rgb"])[0])

        # process obs input
        obs_np_dict = dict()

        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        imageio.imwrite(f"tmp/{ckpt_name}/obs_{step}.png", np.array(obs["rgb"])[0])
        image_list.append(np.array(obs["rgb"])[0])

        obs_np_dict["head_cam"] = np.array(obs["rgb"]).transpose(0, 3, 1, 2).squeeze(0)
        obs_np_dict["agent_pos"] = np.array(obs["joint_qpos"]).squeeze(0)

        # get action
        if args.algo == "rdt":
            image_arrs = []
            for window_img in obs_window:
                image_arrs.append(window_img)
                image_arrs.append(None)
                image_arrs.append(None)
            images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
            actions = actions[
                ::4, :
            ]  # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 14 steps)
        # Compute targets

        for idx in range(actions.shape[0]):
            action = actions[idx]

            log.debug(f"Action: {action}")

            q = None  # XXX: What is q???
            actions = [
                {"dof_pos_target": dict(zip(scenario.robot.joint_limits.keys(), q[i_env].tolist()))}
                for i_env in range(num_envs)
            ]
            obs, reward, success, time_out, extras = env.step(actions)
            env.handler.refresh_render()
            print(reward, success, time_out)

            # queue operation
            obs_window.append(np.array(obs["rgb"]).squeeze(0))
            proprio = np.array(obs["joint_qpos"]).squeeze(0)

            # eval
            SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
            TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
            for TimeOutIndex in range(num_envs):
                if TimeOut[TimeOutIndex]:
                    SuccessEnd[TimeOutIndex] = False
            if all(TimeOut):
                print("All time out")
                break

            step += 1

    images_to_video(image_list, f"tmp/{ckpt_name}/000.mp4")
    print("Num Envs: ", num_envs)
    print(f"SuccessOnce: {SuccessOnce}")
    print(f"SuccessEnd: {SuccessEnd}")
    print(f"TimeOut: {TimeOut}")
    env.close()


if __name__ == "__main__":
    main()
