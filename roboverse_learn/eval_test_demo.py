from __future__ import annotations

import argparse
import os
import time

import imageio
import numpy as np
import rootutils
from loguru import logger as log
from PIL import Image
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.cfg.scenario import ScenarioCfg


def images_to_video(images, video_path, frame_size=(1920, 1080), fps=30):
    if not images:
        log.info("No images found in the specified directory!")
        return

    writer = imageio.get_writer(video_path, fps=fps)

    for image in images:
        if image.shape[1] > frame_size[0] or image.shape[0] > frame_size[1]:
            log.info("Warning: frame size is smaller than the one of the images.")
            log.info("Images will be resized to match frame size.")
            image = np.array(Image.fromarray(image).resize(frame_size))

        writer.append_data(image)

    writer.close()
    log.info("Video created successfully!")


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
        "--render",
        type=str,
        choices=["isaaclab", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="diffusion_policy",
        choices=["diffusion_policy"],
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
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

    # specificly for isaacgym
    if args.sim == "isaacgym":
        pass

    ## Import put here to support isaacgym

    import numpy as np
    import torch

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
    scenario = ScenarioCfg(
        task=args.task,
        robot=args.robot,
        cameras=[camera],
        sim=args.sim,
        renderer=args.render,
        num_envs=args.num_envs,
        try_add_table=True,
        headless=args.headless,
    )

    tic = time.time()
    env_class = get_sim_env_class(SimType(args.sim))
    env = env_class(scenario)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    import datetime

    from roboverse_learn.algorithms.base_policy import get_algos

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = "test_demo" + "_" + time_str
    algo_class = get_algos(args.algo)
    policy = algo_class(
        checkpoint_path=args.checkpoint_path,
    )

    ## Data
    tic = time.time()
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    # meta data
    import json

    with open(
        os.path.join(
            "data_isaaclab/demo/CloseBox/robot-franka/demo_0000",
            "metadata.json",
        ),
        encoding="utf-8",
    ) as f:
        # log.info("metadata load dir:", demo_dir)
        metadata = json.load(f)
    ## Reset before first step
    tic = time.time()
    obs, extras = env.reset(states=init_states[:num_envs])
    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")
    # save_obs(obs, 0)

    step = 0
    MaxStep = 250
    SuccessOnce = [False] * num_envs
    SuccessEnd = [False] * num_envs
    TimeOut = [False] * num_envs
    image_list = []

    while step < MaxStep:
        log.debug(f"Step {step}")
        robot_joint_limits = scenario.robot.joint_limits
        # process obs input
        obs_np_dict = dict()

        import imageio

        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        imageio.imwrite(f"tmp/{ckpt_name}/obs_{step}.png", np.array(obs["rgb"])[0])
        image_list.append(np.array(obs["rgb"])[0])

        obs_np_dict["head_cam"] = np.array(obs["rgb"]).transpose(0, 3, 1, 2).squeeze(0)
        obs_np_dict["agent_pos"] = np.array(obs["joint_qpos"]).squeeze(0)
        # get action
        action = policy.get_action(obs_np_dict)

        # execute action
        # for action_index in range(len(action)):

        actions = [
            {
                "dof_pos_target": {
                    joint_name: torch.tensor(metadata["joint_qpos"][step][index], dtype=torch.float32, device="cuda")
                    for index, joint_name in enumerate(robot_joint_limits.keys())
                }
            }
            for _ in range(num_envs)
        ]
        # actions = [all_actions[0][step]]
        # log.info(actions)
        obs, reward, success, time_out, extras = env.step(actions)
        env.handler.refresh_render()
        log.info(reward, success, time_out)

        # eval
        SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
        TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
        for TimeOutIndex in range(num_envs):
            if TimeOut[TimeOutIndex]:
                SuccessEnd[TimeOutIndex] = False
        if all(TimeOut):
            break

        step += 1

        if all(TimeOut):
            break

    images_to_video(image_list, f"tmp/{ckpt_name}/000.mp4")
    log.info("Num Envs: ", num_envs)
    log.info(f"SuccessOnce: {SuccessOnce}")
    log.info(f"SuccessEnd: {SuccessEnd}")
    log.info(f"TimeOut: {TimeOut}")
    env.close()


if __name__ == "__main__":
    main()
