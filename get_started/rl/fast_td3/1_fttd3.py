from __future__ import annotations

import os
import random
import sys
import time
from typing import Any

CONFIG: dict[str, Any] = {
    # -------------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------------
    "sim": "mjx",
    "robots": ["h1"],
    "task": "walk",
    "decimation": 10,
    "train_or_eval": "train",
    "headless": True,
    "use_viser": True,
    # -------------------------------------------------------------------------------
    # Seeds & Device
    # -------------------------------------------------------------------------------
    "seed": 1,
    "cuda": True,
    "torch_deterministic": True,
    "device_rank": 0,
    # -------------------------------------------------------------------------------
    # Rollout & Timesteps
    # -------------------------------------------------------------------------------
    "num_envs": 1024,
    "num_eval_envs": 1024,
    "total_timesteps": 1500,
    "learning_starts": 10,
    "num_steps": 1,
    # -------------------------------------------------------------------------------
    # Replay, Batching, Discounting
    # -------------------------------------------------------------------------------
    "buffer_size": 20480,
    "batch_size": 32768,
    "gamma": 0.99,
    "tau": 0.1,
    # -------------------------------------------------------------------------------
    # Update Schedule
    # -------------------------------------------------------------------------------
    "policy_frequency": 2,
    "num_updates": 12,
    # -------------------------------------------------------------------------------
    # Optimizer & Network
    # -------------------------------------------------------------------------------
    "critic_learning_rate": 0.0003,
    "actor_learning_rate": 0.0003,
    "weight_decay": 0.1,
    "critic_hidden_dim": 1024,
    "actor_hidden_dim": 512,
    "init_scale": 0.01,
    "num_atoms": 101,
    # -------------------------------------------------------------------------------
    # Value Distribution & Exploration
    # -------------------------------------------------------------------------------
    "v_min": -250.0,
    "v_max": 250.0,
    "policy_noise": 0.001,
    "std_min": 0.001,
    "std_max": 0.4,
    "noise_clip": 0.5,
    # -------------------------------------------------------------------------------
    # Algorithm Flags
    # -------------------------------------------------------------------------------
    "use_cdq": True,
    "compile": True,
    "obs_normalization": True,
    "max_grad_norm": 0.0,
    "amp": True,
    "amp_dtype": "fp16",
    "disable_bootstrap": False,
    "measure_burnin": 3,
    # -------------------------------------------------------------------------------
    # Logging & Checkpointing
    # -------------------------------------------------------------------------------
    "wandb_project": "get_started_fttd3",
    "exp_name": "get_started_fttd3",
    "use_wandb": False,
    "checkpoint_path": None,
    "eval_interval": 700,
    "save_interval": 700,
    "video_width": 1024,
    "video_height": 1024,
}
cfg = CONFIG.get

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure repository root is on sys.path for local package imports
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from loguru import logger as log
from tensordict import TensorDict
from torch import optim
from torch.amp import GradScaler, autocast

torch.set_float32_matmul_precision("high")

from get_started.rl.fast_td3.fttd3_module import Actor, Critic, EmpiricalNormalization, SimpleReplayBuffer
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper


def main() -> None:
    GAMMA = float(cfg("gamma"))
    USE_CDQ = bool(cfg("use_cdq"))
    MAX_GRAD_NORM = float(cfg("max_grad_norm"))
    DISABLE_BOOTSTRAP = bool(cfg("disable_bootstrap"))

    amp_enabled = cfg("amp") and cfg("cuda") and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if cfg("cuda") and torch.cuda.is_available()
        else "mps"
        if cfg("cuda") and torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if cfg("amp_dtype") == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if cfg("use_wandb") and cfg("train_or_eval") == "train":
        wandb.init(
            project=cfg("wandb_project", "fttd3_training"),
            save_code=True,
        )

    random.seed(cfg("seed"))
    np.random.seed(cfg("seed"))
    torch.manual_seed(cfg("seed"))
    torch.backends.cudnn.deterministic = cfg("torch_deterministic")

    if not cfg("cuda"):
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cfg('device_rank')}")
            torch.cuda.set_device(cfg("device_rank"))

        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{cfg('device_rank')}")
        else:
            raise ValueError("No GPU available")
    log.info(f"Using device: {device}")

    task_cls = get_task_class(cfg("task"))
    headless = cfg("headless")
    scenario = task_cls.scenario.update(
        robots=cfg("robots"), simulator=cfg("sim"), num_envs=cfg("num_envs"), headless=headless, cameras=[]
    )
    envs = task_cls(scenario, device=device)

    # Optionally wrap with Viser for real-time visualization
    if cfg("use_viser"):
        envs = TaskViserWrapper(envs, update_freq=1)
        eval_envs = envs
    else:
        eval_envs = envs

    # ---------------- derive shapes ------------------------------------
    n_act = envs.num_actions
    n_obs = envs.num_obs
    n_critic_obs = n_obs  # no privileged obs
    action_low, action_high = -1.0, 1.0

    # ---------------- normalisers -------------------------------------
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device)

    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=cfg("num_envs"),
        device=device,
        init_scale=cfg("init_scale"),
        hidden_dim=cfg("actor_hidden_dim"),
    )
    actor_detach = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=cfg("num_envs"),
        device=device,
        init_scale=cfg("init_scale"),
        hidden_dim=cfg("actor_hidden_dim"),
    )
    # Copy params to actor_detach without grad
    TensorDict.from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=cfg("num_atoms"),
        v_min=cfg("v_min"),
        v_max=cfg("v_max"),
        hidden_dim=cfg("critic_hidden_dim"),
        device=device,
    )
    qnet_target = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=cfg("num_atoms"),
        v_min=cfg("v_min"),
        v_max=cfg("v_max"),
        hidden_dim=cfg("critic_hidden_dim"),
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=cfg("critic_learning_rate"),
        weight_decay=cfg("weight_decay"),
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=cfg("actor_learning_rate"),
        weight_decay=cfg("weight_decay"),
    )

    rb = SimpleReplayBuffer(
        n_env=cfg("num_envs"),
        buffer_size=cfg("buffer_size"),
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        n_steps=cfg("num_steps"),
        gamma=cfg("gamma"),
        device=device,
    )

    policy_noise = cfg("policy_noise")
    noise_clip = cfg("noise_clip")

    def evaluate():
        obs_normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        obs, info = eval_envs.reset()

        # Run for a fixed number of steps
        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                obs = normalize_obs(obs)
                actions = actor(obs)
            next_obs, rewards, terminated, time_out, infos = eval_envs.step(actions.float())
            dones = terminated | time_out
            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        obs_normalizer.train()
        obs, info = eval_envs.reset()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout() -> list:
        """
        Collect a short rollout and return a list of RGB frames (H, W, 3, uint8).
        Works with FastTD3EnvWrapper: render_env.render() must return one frame.
        """
        video_path: str = cfg("video_path", "output/rollout.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        robots = cfg("robots")
        simulator = cfg("sim")
        num_envs = cfg("num_envs")
        headless = True
        cameras = [
            PinholeCameraCfg(
                width=cfg("video_width", 1024),
                height=cfg("video_height", 1024),
                pos=(3.0, -3.0, 3.0),  # adjust as needed
                look_at=(0.0, 0.0, 0.0),
            )
        ]
        scenario_render = scenario.update(
            robots=robots, simulator=simulator, num_envs=num_envs, headless=headless, cameras=cameras
        )
        env = task_cls(scenario_render, device=device)

        obs_normalizer.eval()
        obs, info = env.reset()

        # Initialize video recording
        os.makedirs("get_started/output/rl", exist_ok=True)
        obs_saver = ObsSaver(video_path=f"get_started/output/rl/1_fttd3_{simulator}.mp4")
        obs_orin = env.handler.get_states()
        obs_saver.add(obs_orin)

        for _ in range(env.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                actions = actor(obs_normalizer(obs))
            obs, _, terminated, time_out, _ = env.step(actions.float())
            done = terminated | time_out

            obs_orin = env.handler.get_states()
            obs_saver.add(obs_orin)
            if done.any():
                break

        obs_saver.save()
        env.close()
        obs_normalizer.train()

    def update_main(data, logs_dict):
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = observations
            next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if DISABLE_BOOTSTRAP:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(-noise_clip, noise_clip)

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(action_low, action_high)

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards,
                    bootstrap,
                    GAMMA,
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if USE_CDQ:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1) < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
            qf2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(),
            max_norm=MAX_GRAD_NORM if MAX_GRAD_NORM > 0 else float("inf"),
        )
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["buffer_rewards"] = rewards.mean()
        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            critic_observations = data["observations"]

            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if USE_CDQ:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=MAX_GRAD_NORM if MAX_GRAD_NORM > 0 else float("inf"),
        )
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    if cfg("compile"):
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=mode)
    else:
        normalize_obs = obs_normalizer.forward
    obs, info = envs.reset()

    if cfg("checkpoint_path"):
        # Load checkpoint if specified
        torch_checkpoint = torch.load(f"{cfg('checkpoint_path')}", map_location=device, weights_only=False)
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(torch_checkpoint["critic_obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=cfg("total_timesteps"), initial=global_step)
    start_time = None
    desc = ""

    while global_step < cfg("total_timesteps"):
        logs_dict = TensorDict()
        if start_time is None and global_step >= cfg("measure_burnin") + cfg("learning_starts"):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)
        next_obs, rewards, terminated, time_out, infos = envs.step(actions.float())
        dones = terminated | time_out

        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs)
        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                    "truncations": time_out.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        obs = next_obs

        rb.extend(transition)

        batch_size = cfg("batch_size") // cfg("num_envs")
        if global_step > cfg("learning_starts"):
            for i in range(cfg("num_updates")):
                data = rb.sample(batch_size)
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])
                logs_dict = update_main(data, logs_dict)
                if cfg("num_updates") > 1:
                    if i % cfg("policy_frequency") == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % cfg("policy_frequency") == 0:
                        logs_dict = update_pol(data, logs_dict)

                for param, target_param in zip(qnet.parameters(), qnet_target.parameters()):
                    target_param.data.copy_(cfg("tau") * param.data + (1 - cfg("tau")) * target_param.data)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "buffer_rewards": logs_dict["buffer_rewards"].mean(),
                        "env_rewards": rewards.mean(),
                    }

                    if cfg("eval_interval") > 0 and global_step % cfg("eval_interval") == 0:
                        log.info(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        obs, info = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length
                        log.info(f"avg_return={eval_avg_return:.4f}, avg_length={eval_avg_length:.4f}")

                if cfg("use_wandb"):
                    wandb.log(
                        {
                            "speed": speed,
                            "frame": global_step * cfg("num_envs"),
                            **logs,
                        },
                        step=global_step,
                    )

            if cfg("save_interval") > 0 and global_step > 0 and global_step % cfg("save_interval") == 0:
                log.info(f"Saving model at global step {global_step}")

        global_step += 1
        pbar.update(1)
        # Close environment and wandb
    envs.close()
    render_with_rollout()


if __name__ == "__main__":
    main()
