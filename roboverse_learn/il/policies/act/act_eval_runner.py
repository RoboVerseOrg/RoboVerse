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

from metasim.utils.kinematics import get_curobo_models
from metasim.task.registry import get_task_class

# Try to import randomization components
try:
    from metasim.randomization import DomainRandomizationManager, DRConfig
    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Domain randomization not available: {e}")
    RANDOMIZATION_AVAILABLE = False



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


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot", type=str, default="franka")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument(
        "--sim",
        type=str,
        default="isaacsim",
        choices=["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"],
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="openvla",
        choices=["act"],
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
    parser.add_argument(
        "--num_eval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=400,
    )

    # Domain Randomization options
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"
    )
    parser.add_argument(
        "--scene_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"
    )
    parser.add_argument(
        "--randomization_seed",
        type=int,
        default=None,
        help="Seed for reproducible randomization. If None, uses random seed"
    )

    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    num_envs: int = args.num_envs

    import numpy as np
    import torch

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot

    task_cls = get_task_class(args.task)

    if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
        dp_camera = True
    else:
        dp_camera = args.task != "close_box"

    is_libero_dataset = "libero_90" in args.task

    if is_libero_dataset:
        dp_pos = (2.0, 0.0, 2)
    elif dp_camera:
        dp_pos = (1.0, 0.0, 0.75)
    else:
        dp_pos = (1.5, 0.0, 1.5)

    camera = PinholeCameraCfg(
        name="camera",
        data_types=["rgb", "depth"],
        width=256,
        height=256,
        pos=dp_pos,
        look_at=(0.0, 0.0, 0.0),
    )

    # Lighting setup (same logic as collect_demo.py)
    # Determine intensity based on render mode (if available)
    render_mode = getattr(args, 'render_mode', 'raytracing')
    if render_mode == "pathtracing":
        ceiling_main = 18000.0
        ceiling_corners = 8000.0
    else:
        ceiling_main = 12000.0
        ceiling_corners = 5000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
        ),
    ]

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        simulator=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
        lights=lights,
        cameras=[camera]
    )

    tic = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)
    robot = get_robot(args.robot)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    ## Data
    tic = time.time()
    assert os.path.exists(env.traj_filepath), (
        f"Trajectory file: {env.traj_filepath} does not exist."
    )
    init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")

    # Initialize Domain Randomization Manager
    if not RANDOMIZATION_AVAILABLE:
        log.warning("Randomization components not available!")
        raise ImportError("Domain Randomization not available. Please check installation.")

    # Determine render mode from args
    render_mode = getattr(args, 'render_mode', 'raytracing')

    # Create render config for DR
    from dataclasses import dataclass
    @dataclass
    class SimpleRenderCfg:
        mode: str = render_mode

    randomization_manager = DomainRandomizationManager(
        config=DRConfig(
            level=args.level,
            scene_mode=args.scene_mode,
            randomization_seed=args.randomization_seed,
        ),
        scenario=scenario,
        handler=env.handler,
        init_states=init_states,
        render_cfg=SimpleRenderCfg(mode=render_mode)
    )

    if args.algo == "act":
        state_dim = 9
        franka_state_dim = 9
        lr_backbone = 1e-5
        backbone = "resnet18"
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        camera_names = ["front"]
        kl_weight = 10
        # chunk_size = args.chunk_size
        hidden_dim = 512
        batch_size = 8
        dim_feedforward = 3200
        lr = 1e-5
        act_ckpt_name = "policy_best.ckpt"
        policy_config = {
            "lr": lr,
            "num_queries": args.chunk_size,
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

        from roboverse_learn.il.policies.act.policy import ACTPolicy

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
           # return (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
            return (s_qpos - stats["state_mean"]) / stats["state_std"]


        def post_process(a):
            return a * stats["action_std"] + stats["action_mean"]

        query_frequency = policy_config["num_queries"]
        if args.temporal_agg:
            query_frequency = 1
            num_queries = policy_config["num_queries"]
        max_timesteps = env.max_episode_steps
        max_timesteps = int(max_timesteps * 1)

    ckpt_name = args.ckpt_path.split("/")[-1]
    os.makedirs(f"il_outputs/{args.algo}/{args.task}/{ckpt_name}", exist_ok=True)

    ## cuRobo controller (commented out - not needed for ACT joint control)
    # *_, robot_ik = get_curobo_models(scenario.robots[0])
    # curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
    # ee_n_dof = len(scenario.robots[0].gripper_open_q)

    ## Reset before first step
    TotalSuccess = 0
    num_eval: int = args.num_eval

    for i in range(num_eval):
        demo_idx = i

        # Apply domain randomization before reset
        log.info(f"[ACT Eval] Episode {i}: Applying DR for demo_idx={demo_idx}")
        randomization_manager.apply_randomization(demo_idx=demo_idx, is_initial=(i == 0))
        randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=0)
        randomization_manager.update_camera_look_at(env_id=0)
        randomization_manager.apply_camera_randomization()

        tic = time.time()
        obs, extras = env.reset(states=[init_states[demo_idx]])
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")

        # Ensure environment stabilizes after reset
        ensure_clean_state(env.handler, expected_state=init_states[demo_idx])

        # Reset episode step counter after stabilization
        if hasattr(env, "_episode_steps"):
            env._episode_steps[0] = 0

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

                image_list.append(np.array(obs.cameras['camera'].rgb.cpu())[0])

                qpos_numpy = np.array(obs.robots['franka'].joint_pos.cpu())
                # qpos_numpy = np.array(obs["joint_qpos"])
                qpos = pre_process(qpos_numpy)
                # qpos = np.concatenate([qpos, np.zeros((qpos.shape[0], 14 - qpos.shape[1]))], axis=1)
                qpos = torch.from_numpy(qpos).float().cuda()
                qpos_history[:, step] = qpos
                curr_image = np.array(obs.cameras['camera'].rgb.cpu()).transpose(0, 3, 1, 2)
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
                action = torch.tensor(action, dtype=torch.float32, device="cpu")

                # IK solver expects original joint order, but state uses alphabetical order
                reorder_idx = env.handler.get_joint_reindex(args.robot)
                inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
                actions = action[inverse_reorder_idx]
                inner_actions = {"dof_pos_target": dict(zip(scenario.robots[0].joint_limits.keys(), actions))}
                # Format: actions[env_id][robot_name][action_type]
                actions = [{"franka": inner_actions}]
                #log.debug(f"Actions: {actions}")
                # log.debug(f"Action: {actions}")
                obs, reward, success, time_out, extras = env.step(actions)
                env.handler.refresh_render()
                # print(reward, success, time_out)

                # eval
                # if success[0]:
                #     TotalSuccess += 1
                #     print(f"Env {i} Success")
                if success[0] and not SuccessOnce[0]:
                    TotalSuccess += 1
                    SuccessOnce[0] = True
                    print(f"Env {i} Success")

                log.debug(f"TotalSuccess: {TotalSuccess}")
                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                for TimeOutIndex in range(num_envs):
                    if TimeOut[TimeOutIndex]:
                        SuccessEnd[TimeOutIndex] = False
                if all(TimeOut):
                    print("All time out")
                    break

                step += 1

            images_to_video(image_list, f"il_outputs/{args.algo}/{args.task}/{ckpt_name}/{i}.mp4")

    success_rate = TotalSuccess / num_eval
    print("Success Rate: ", success_rate)

    result_dir = f"il_outputs/{args.algo}/{args.task}/{ckpt_name}"
    result_file = os.path.join(result_dir, "success_rate.txt")
    with open(result_file, "w") as f:
        f.write(f"Success Rate: {success_rate:.4f}\n")

    env.close()


if __name__ == "__main__":
    main()
