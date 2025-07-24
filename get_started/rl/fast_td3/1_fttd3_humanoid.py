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
    "task": "humanoidbench:Walk",
    "decimation": 10,
    "train_or_eval": "train",
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
    "total_timesteps": 1200,
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
    "eval_interval": 1200,
    "save_interval": 1200,
    "video_width": 1024,
    "video_height": 1024,
}
cfg = CONFIG.get

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if cfg("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg("device_rank"))
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch

torch.set_float32_matmul_precision("high")

import numpy as np

try:
    import isaacgym  # noqa: F401 – optional, only if sim == "isaacgym"
except ImportError:
    pass

import torch
import torch.nn.functional as F
import tqdm
import wandb
from fast_td3 import Actor, Critic, EmpiricalNormalization, SimpleReplayBuffer
from loguru import logger as log
from tensordict import TensorDict
from torch import optim
from torch.amp import GradScaler, autocast
from torchvision.utils import make_grid

from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import list_state_to_tensor


class FastTD3EnvWrapper:
    def __init__(
        self,
        scenario: ScenarioCfg,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Build the underlying MetaSim environment
        EnvironmentClass = get_sim_env_class(SimType(scenario.sim))
        self.env = EnvironmentClass(scenario)

        self.num_envs = scenario.num_envs
        self.robot = scenario.robots[0]
        self.task = scenario.task
        # ----------- initial states --------------------------------------------------
        initial_states, _, _ = get_traj(self.task, self.robot, self.env.handler)
        # Duplicate / trim list so that its length matches num_envs
        if len(initial_states) < self.num_envs:
            k = self.num_envs // len(initial_states)
            initial_states = initial_states * k + initial_states[: self.num_envs % len(initial_states)]
        self._initial_states = initial_states[: self.num_envs]
        if scenario.sim == "mjx":
            self._initial_states = list_state_to_tensor(self.env.handler, self._initial_states)
        self.env.reset(states=self._initial_states)
        states = self.env.handler.get_states()
        first_obs = self.get_humanoid_observation(states)
        self.num_obs = first_obs.shape[-1]
        self._raw_observation_cache = first_obs.clone()

        limits = self.robot.joint_limits  # dict: {joint_name: (low, high)}
        self.joint_names = self.env.handler.get_joint_names(self.robot.name)

        self._action_low = torch.tensor(
            [limits[j][0] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self._action_high = torch.tensor(
            [limits[j][1] for j in self.joint_names], dtype=torch.float32, device=self.device
        )
        self.num_actions = self._action_low.shape[0]
        self.max_episode_steps = self.env.handler.task.episode_length
        self.asymmetric_obs = False  # privileged critic input not used (for now)

    def reset(self) -> torch.Tensor:
        self.env.reset(states=self._initial_states)
        states = self.env.handler.get_states()
        observation = self.get_humanoid_observation(states)
        observation = observation.to(self.device)
        self._raw_observation_cache.copy_(observation)
        return observation

    def step(self, actions: torch.Tensor):
        real_action = self._unnormalise_action(actions)
        states, _, terminated, truncated, _ = self.env.step_actions(real_action)

        obs_now = self.get_humanoid_observation(states).to(self.device)
        reward_now = self.get_humanoid_reward(states).to(self.device)

        done_flag = terminated.to(self.device, torch.bool)
        time_out_flag = truncated.to(self.device, torch.bool)

        info = {
            "time_outs": time_out_flag,
            "observations": {"raw": {"obs": self._raw_observation_cache.clone().to(self.device)}},
        }

        if (done_indices := (done_flag | time_out_flag).nonzero(as_tuple=False).squeeze(-1)).numel():
            self.env.reset(states=self._initial_states, env_ids=done_indices.tolist())
            reset_states = self.env.handler.get_states()
            reset_obs_full = self.get_humanoid_observation(reset_states).to(self.device)
            obs_now[done_indices] = reset_obs_full[done_indices]
            self._raw_observation_cache[done_indices] = reset_obs_full[done_indices]
        else:
            keep_mask = (~done_flag).unsqueeze(-1)
            self._raw_observation_cache = torch.where(keep_mask, self._raw_observation_cache, obs_now)

        return obs_now, reward_now, done_flag, info

    def render(self) -> None:
        state = self.env.handler.get_states()
        rgb_data = next(iter(state.cameras.values())).rgb
        image = make_grid(rgb_data.permute(0, 3, 1, 2) / 255, nrow=int(rgb_data.shape[0] ** 0.5))  # (C, H, W)
        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image * 255).astype(np.uint8)
        return image

    def close(self) -> None:
        self.env.close()

    def get_humanoid_observation(self, states) -> torch.Tensor:
        """Flatten humanoid states and move them onto the training device."""
        return self.task.humanoid_obs_flatten_func(states).to(self.device)

    def get_humanoid_reward(self, states) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, device=self.device)
        for reward_fn, weight in zip(self.task.reward_functions, self.task.reward_weights):
            total_reward += reward_fn(self.robot.name)(states).to(self.device) * weight
        return total_reward

    def _unnormalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map actions from [-1, 1] to the robot's joint-limit range."""
        return (action + 1) / 2 * (self._action_high - self._action_low) + self._action_low


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
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{cfg('device_rank')}")
        else:
            raise ValueError("No GPU available")
    log.info(f"Using device: {device}")

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
    scenario_render = ScenarioCfg(
        task=cfg("task"),
        robots=cfg("robots"),
        try_add_table=cfg("add_table", False),
        sim=cfg("sim"),
        num_envs=1,
        headless=True,
        cameras=[
            PinholeCameraCfg(
                width=cfg("video_width", 1024),
                height=cfg("video_height", 1024),
                pos=(4.0, -4.0, 4.0),  # adjust as needed
                look_at=(0.0, 0.0, 0.0),
            )
        ],
    )
    scenario_render.task.decimation = cfg("decimation", 1)

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
        import imageio.v2 as iio

        """
        Collect a short rollout and return a list of RGB frames (H, W, 3, uint8).
        Works with FastTD3EnvWrapper: render_env.render() must return one frame.
        """
        video_path: str = cfg("video_path", "output/rollout.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        env = FastTD3EnvWrapper(scenario_render, device=device)

        obs_normalizer.eval()
        obs = env.reset()
        frames = [env.render()]

        for _ in range(env.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                act = actor(obs_normalizer(obs))
            obs, _, done, _ = env.step(act.float())
            frames.append(env.render())
            if done.any():
                break

        env.close()
        obs_normalizer.train()

        iio.mimsave(video_path, frames, fps=30)
        return frames

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
        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs)
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
                        obs = envs.reset()
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
