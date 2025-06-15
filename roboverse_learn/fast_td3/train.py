from __future__ import annotations

import os
import random
import sys
import time

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_DEFAULT_MATMUL_PRECISION"]          = "highest"

import numpy as np

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
import yaml
from fast_td3 import Actor, Critic
from fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer
from loguru import logger as log
from tensordict import TensorDict
from torch.amp import GradScaler, autocast
from wrapper import FastTD3EnvWrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision("high")


def load_config_from_yaml(config_name: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_name (str): Name of the YAML config file

    Returns:
        dict: The loaded config dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "configs", f"{config_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    if len(sys.argv) < 2:
        log.error("Please provide the config file path, e.g. python train_sb3.py configs/isaacgym.yaml")
        exit(1)
    config_name = sys.argv[1]
    config = load_config_from_yaml(config_name)
    cfg = config.get
    GAMMA = float(cfg("gamma"))
    USE_CDQ = bool(cfg("use_cdq"))
    MAX_GRAD_NORM = float(cfg("max_grad_norm"))
    DISABLE_BOOTSTRAP = bool(cfg("disable_bootstrap"))
    log.info(f"Load config: {config_name}")

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
            config=config,
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
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{cfg('device_rank')}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    from metasim.cfg.scenario import ScenarioCfg

    scenario = ScenarioCfg(
        task=cfg("task"),
        robots=cfg("robots"),
        try_add_table=cfg("add_table", False),
        sim=cfg("sim"),
        num_envs=cfg("num_envs", 1),
        headless=True if cfg("train_or_eval") == "train" else False,
        cameras=[],
    )

    # For different simulators, the decimation factor is different, so we need to set it here
    scenario.task.decimation = cfg("decimation", 1)

    envs = FastTD3EnvWrapper(scenario, device=device)
    eval_envs = envs  # reuse for evaluation
    render_env = envs  # reuse for optional rendering

    # ---------------- derive shapes ------------------------------------
    n_act = envs.num_actions
    n_obs = envs.num_obs
    n_critic_obs = n_obs  # no privileged obs
    action_low, action_high = -1.0, 1.0

    # ---------------- normalisers -------------------------------------

    if cfg("obs_normalization"):
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

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

        obs = eval_envs.reset()

        # Run for a fixed number of steps
        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                obs = normalize_obs(obs)
                actions = actor(obs)

            next_obs, rewards, dones, _ = eval_envs.step(actions.float())
            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        obs_normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout() -> list:
        """
        Collect a short rollout and return a list of RGB frames (H, W, 3, uint8).
        Works with FastTD3EnvWrapper: render_env.render() must return one frame.
        """
        obs_normalizer.eval()

        # first frame after reset
        obs = render_env.reset()
        frames = [render_env.render()]

        for _ in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                act = actor(obs_normalizer(obs))
            next_obs, _, done, _ = render_env.step(act.float())

            # store every second frame to keep GIF size reasonable
            if _ % 2 == 0:
                frames.append(render_env.render())

            if done.any():
                break
            obs = next_obs

        obs_normalizer.train()
        return frames

    def update_main(data, logs_dict):
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
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
            critic_observations = data["critic_observations"] if envs.asymmetric_obs else data["observations"]

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
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=mode)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()

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

        next_obs, rewards, dones, infos = envs.step(actions.float())

        truncations = infos["time_outs"]

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]

        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs)
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )
        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        rb.extend(transition)

        batch_size = cfg("batch_size") // cfg("num_envs")
        if global_step > cfg("learning_starts"):
            for i in range(cfg("num_updates")):
                data = rb.sample(batch_size)
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(data["critic_observations"])
                    data["next"]["critic_observations"] = normalize_critic_obs(data["next"]["critic_observations"])
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
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length
                        print(f"avg_return={eval_avg_return:.4f}, avg_length={eval_avg_length:.4f}")

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
                print(f"Saving model at global step {global_step}")

        global_step += 1
        pbar.update(1)


if __name__ == "__main__":
    main()
