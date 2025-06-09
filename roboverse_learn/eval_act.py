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
        default="openvla",
        choices=["diffusion_policy", "openvla", "rdt", "act"],
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="openvla/openvla-7b",
    )
    parser.add_argument(
        "--temporal_agg",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--headless",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    num_envs: int = args.num_envs

    # specificly for isaacgym
    if args.sim == "isaacgym":
        pass

    ## Import put here to support isaacgym

    import numpy as np
    import torch

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
        robots=[args.robot],
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

    if args.algo == "act":
        state_dim = 14
        franka_state_dim = 9
        lr_backbone = 1e-5
        backbone = "resnet18"
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        camera_names = ["front"]
        kl_weight = 10
        chunk_size = 100
        hidden_dim = 512
        batch_size = 8
        dim_feedforward = 3200
        lr = 1e-5
        act_ckpt_name = "policy_best.ckpt"
        policy_config = {
            "lr": lr,
            "num_queries": chunk_size,
            "kl_weight": kl_weight,
            "hidden_dim": hidden_dim,
            "dim_feedforward": dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }

        import pickle

        from roboverse_learn.algorithms.act.policy import ACTPolicy

        ckpt_path = os.path.join(args.ckpt_path, act_ckpt_name)
        policy = ACTPolicy(policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        policy.cuda()
        policy.eval()
        print(f"Loaded: {ckpt_path}")
        stats_path = os.path.join(args.ckpt_path, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        def pre_process(s_qpos):
            return (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]

        def post_process(a):
            return a * stats["action_std"] + stats["action_mean"]

        query_frequency = policy_config["num_queries"]
        if args.temporal_agg:
            query_frequency = 1
            num_queries = policy_config["num_queries"]
        max_timesteps = scenario.task.episode_length
        max_timesteps = int(max_timesteps * 1)

    ckpt_name = args.ckpt_path.split("/")[-1]
    os.makedirs(f"tmp/{args.algo}/{args.task}/{ckpt_name}", exist_ok=True)

    ## Data
    tic = time.time()
    assert os.path.exists(scenario.task.traj_filepath), (
        f"Trajectory file: {scenario.task.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(scenario.task, scenario.robots[0], env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    ## cuRobo controller
    *_, robot_ik = get_curobo_models(scenario.robots[0])
    curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    ee_n_dof = len(scenario.robots[0].gripper_open_q)

    ## Reset before first step
    TotalSuccess = 0

    for i in range(100):
        tic = time.time()
        obs, extras = env.reset(states=[init_states[-(i + 1)]])
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")
        # save_obs(obs, 0)
        log.debug(f"Env: {i}")

        step = 0
        MaxStep = 800
        SuccessOnce = [False] * num_envs
        SuccessEnd = [False] * num_envs
        TimeOut = [False] * num_envs
        image_list = []

        # act specific
        if args.temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()

        with torch.no_grad():
            while step < MaxStep:
                log.debug(f"Step {step}")
                robot_joint_limits = scenario.robots[0].joint_limits

                image_list.append(np.array(obs["rgb"])[0])

                # act
                qpos_numpy = np.array(obs["joint_qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = np.concatenate([qpos, np.zeros((qpos.shape[0], 14 - qpos.shape[1]))], axis=1)
                qpos = torch.from_numpy(qpos).float().cuda()
                qpos_history[:, step] = qpos
                curr_image = np.array(obs["rgb"]).transpose(0, 3, 1, 2)
                # cur_image = np.stack([curr_image, curr_image], axis=0)
                curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
                # breakpoint()
                # Compute targets

                if step % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if args.temporal_agg:
                    all_time_actions[[step], step : step + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, step]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, step % query_frequency]

                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = action[:franka_state_dim]
                log.debug(f"Action: {action}")

                action = torch.tensor(action, dtype=torch.float32, device="cuda")
                actions = [{"dof_pos_target": dict(zip(scenario.robots[0].joint_limits.keys(), action))}]
                obs, reward, success, time_out, extras = env.step(actions)
                env.handler.refresh_render()
                # print(reward, success, time_out)

                # eval
                if success[0]:
                    TotalSuccess += 1
                    print(f"Env {i} Success")

                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                for TimeOutIndex in range(num_envs):
                    if TimeOut[TimeOutIndex]:
                        SuccessEnd[TimeOutIndex] = False
                if all(TimeOut):
                    # print("All time out")
                    break

                step += 1

            images_to_video(image_list, f"tmp/{args.algo}/{args.task}/{ckpt_name}/{i}.mp4")

    print("Success Rate: ", TotalSuccess / 100.0)
    env.close()


if __name__ == "__main__":
    main()
