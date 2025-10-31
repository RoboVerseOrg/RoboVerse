from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import copy
import os
import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from tensordict import TensorDict
from torch import Tensor

from roboverse_learn.dexbench_rvrl.algos.drqv2.module import Actor, Critic, Encoder, RandomShiftsAug, schedule, soft_update_params
from roboverse_learn.dexbench_rvrl.algos.drqv2.storage import ReplayBufferStorage, IterableReplayBuffer, make_replay_loader, Buffer
from roboverse_learn.dexbench_rvrl.algos.drqv2.env_wrapper import DrqDexEnv
from roboverse_learn.dexbench_rvrl.utils.reproducibility import enable_deterministic_run
from roboverse_learn.dexbench_rvrl.utils.timer import timer


class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards):
        if isinstance(rewards, torch.Tensor):
            self.deque.extend(rewards.detach().cpu().flatten().tolist())
        elif isinstance(rewards, float):
            self.deque.append(rewards)
        elif isinstance(rewards, int):
            self.deque.append(float(rewards))
        else:
            raise ValueError(f"Unsupported type for rewards: {type(rewards)}. Expected torch.Tensor, float, or int.")

    @property
    def len(self) -> int:
        return len(self.deque)

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


class DRQv2:    
    def __init__(
        self,
        env,
        train_cfg: dict,
        device="cpu",
        log_dir="run",
        model_dir=None,
        is_testing=False,
        print_log=True,
        wandb_run=None,
    ):
        self.device = device
        self.sim_device = env.sim_device if hasattr(env, "sim_device") else device

        self.obs_type = getattr(env, "obs_type", "state")
        ori_obs_shape = {k: v.shape for k, v in env.single_observation_space.spaces.items()}
        self.action_dim = np.prod(env.single_action_space.shape)
        self.max_episode_steps = env.max_episode_steps
        self.img_h = getattr(env, "img_h", None)
        self.img_w = getattr(env, "img_w", None)

        # learn cfg
        self.train_cfg = copy.deepcopy(train_cfg)
        
        if train_cfg.get("deterministic", False):
            enable_deterministic_run()
            
        learn_cfg = self.train_cfg["learn"]
        self.env = DrqDexEnv(obs_shape=ori_obs_shape, num_frames=learn_cfg.get("num_frames", 3), env=env, device=self.device)
        self.num_envs = self.env.num_envs
        self.obs_shape = {k: v.shape for k, v in self.env.single_observation_space.spaces.items()}
        print(f"Observation shape: {self.obs_shape}")
        print(f"Action dimension: {self.action_dim}")
        self.actor_lr = learn_cfg.get("actor_lr", 1e-4)
        self.critic_lr = learn_cfg.get("critic_lr", 1e-4)
        self.encoder_lr = learn_cfg.get("encoder_lr", 1e-4)
        
        self.gamma = learn_cfg.get("gamma", 0.99)
        self.tau = learn_cfg.get("tau", 0.005)
        self.std_clip = learn_cfg.get("std_clip", 0.3)
        self.std_schedule = learn_cfg.get("std_schedule", "linear(1.0,0.1,500000)")
        
        self.batch_size = learn_cfg["batch_size"]
        self.buffer_size = learn_cfg.get("buffer_size", 1000000)
        self.nstep = learn_cfg.get("nstep", 3)
        self.max_iterations = learn_cfg.get("max_iterations", 100000)
        self.env_step = learn_cfg.get("env_step", 1)
        self.log_interval = learn_cfg.get("log_interval", 1)
        self.print_interval = learn_cfg.get("print_interval", 10)
        self.prefill = learn_cfg.get("prefill", 200)
        self.max_grad_norm = learn_cfg.get("max_grad_norm", None)
        

        self.model_cfg = self.train_cfg.get("policy", None)

        ## Replay buffer
        self.replay_storage = ReplayBufferStorage(self.obs_shape, self.action_dim, log_dir + '/buffer', self.num_envs, self.max_episode_steps)

        self.replay_loader = make_replay_loader(
            obs_shape=self.obs_shape,
            action_size=self.action_dim,
            replay_dir=log_dir + '/buffer', 
            max_size=self.buffer_size,
            batch_size=self.batch_size, 
            num_workers=learn_cfg.get("replay_buffer_num_workers", 4),
            save_snapshot=learn_cfg.get("save_snapshot", False), 
            nstep=learn_cfg.get("nstep", 3), 
            discount=self.gamma,
        )
        self._replay_iter = None

        ## Drqv2 components
        self.encoder = Encoder(self.obs_type, self.obs_shape, self.model_cfg, self.img_h, self.img_w).to(device)
        self.actor = Actor(
            self.obs_type,
            self.obs_shape,
            self.action_dim,
            self.model_cfg,
        ).to(device)
        
        self.critic = Critic(
            self.obs_type,
            self.obs_shape,
            self.action_dim,
            self.model_cfg,
        ).to(device)
        
        self.critic_target = Critic(
            self.obs_type,
            self.obs_shape,
            self.action_dim,
            self.model_cfg,
        ).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        ## Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.encoder_lr)
        
        self.aug = RandomShiftsAug(pad=6).to(device)

        ## Logging
        self.episode_rewards = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_lengths = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_rewards_step = RollingMeter(learn_cfg.get("window_size", 1000))
        self.actor_losses = RollingMeter(learn_cfg.get("window_size", 100))
        self.critic_losses = RollingMeter(learn_cfg.get("window_size", 100))
        self.q_values = RollingMeter(learn_cfg.get("window_size", 100))
        self.cur_rewards_sum = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.cur_episode_length = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.batch_reward = 0

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.print_log = print_log
        self.wandb_run = wandb_run
        self.is_testing = is_testing
        self.global_step = 0
        self.prev_global_step = 0
        self.current_learning_iteration = 0

    def test(self, path):
        actor_path = os.path.join(path, "actor.pt")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            log.info(f"Loaded model from {actor_path}")
        critic_path = os.path.join(path, "critic.pt")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            log.info(f"Loaded model from {critic_path}")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        log.info("Evaluation mode")

    def load(self, path):
        self.current_learning_iteration = int(path.split("_")[-1])
        actor_path = os.path.join(path, "actor.pt")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            log.info(f"Loaded model from {actor_path}")
        critic_path = os.path.join(path, "critic.pt")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            log.info(f"Loaded model from {critic_path}")
        critic_target_path = os.path.join(path, "critic_target.pt")
        if os.path.exists(critic_target_path):
            self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=self.device))
            log.info(f"Loaded model from {critic_target_path}")
            
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        log.info("Training mode")

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target.pt"))
        log.info(f"Saved model to {path}")
        
    def to_device(self, batch):
        batch_on_device = {}
        start_idx = 0
        batch_on_device["observation"] = {}
        batch_on_device["next_observation"] = {}
        for key in self.obs_shape.keys():
            batch_on_device["observation"][key] = torch.as_tensor(batch[start_idx], device=self.device)
            start_idx += 1
        for key in self.obs_shape.keys():
            batch_on_device["next_observation"][key] = torch.as_tensor(batch[start_idx], device=self.device)
            start_idx += 1
        batch_on_device["action"] = torch.as_tensor(batch[start_idx], device=self.device)
        start_idx += 1
        batch_on_device["reward"] = torch.as_tensor(batch[start_idx], device=self.device)
        start_idx += 1
        batch_on_device["discount"] =  torch.as_tensor(batch[start_idx], device=self.device)
        return batch_on_device

    def run(self):
        if self.is_testing:
            assert self.model_dir is not None, "model_dir must be specified in testing mode"
            self.test(self.model_dir)
        elif self.model_dir is not None:
            self.load(self.model_dir)
        rest_obs = self.env.reset()
        obs = {k: rest_obs[k].clone().to(self.device) for k in rest_obs.keys()}
        self.start_time = time.time()
        for param in self.critic_target.parameters():
            param.requires_grad = False
        if not self.is_testing:
            ep_infos = []
            for iteration in range(self.current_learning_iteration, self.max_iterations):
                with timer("time/iteration"):
                    with torch.inference_mode(), timer("time/roll_out"):
                        for _ in range(self.env_step):
                            if self.global_step < self.prefill:
                                action = torch.rand((self.num_envs, self.action_dim), device=self.sim_device) * 2 - 1
                            else:
                                feature = self.encoder(obs)
                                std = schedule(self.std_schedule, iteration)
                                dist = self.actor(feature, std)
                                action = dist.sample(clip=None).to(self.sim_device)

                            start_time = time.time()
                            next_obs, reward, terminated, truncated, info = self.env.step(action)
                            done = torch.logical_or(terminated, truncated)
                            self.replay_storage.add(obs, action, reward, next_obs, done)

                            for k in obs:
                                obs[k].copy_(next_obs[k])

                            ep_infos.append(info)
                            self.episode_rewards_step.update(reward)
                            self.cur_rewards_sum += reward.to(self.device)
                            self.cur_episode_length += 1
                            self.global_step += self.num_envs

                            if done.any():
                                self.episode_rewards.update(self.cur_rewards_sum[done])
                                self.episode_lengths.update(self.cur_episode_length[done])
                                self.cur_rewards_sum[done] = 0
                                self.cur_episode_length[done] = 0

                    mean_reward = 0 if self.episode_rewards_step.len == 0 else self.episode_rewards_step.mean
                    # update the model
                    if self.global_step >= self.prefill:
                        with timer("time/sample_data"):
                            data = self.to_device(next(self.replay_iter))
                            data = self.aug(data)
                            data["feature"] = self.encoder(data["observation"])
                            with torch.no_grad():
                                data["next_feature"] = self.encoder(data["next_observation"])
                            self.batch_reward = data["reward"].mean().item()
                        with timer("time/update_model"):
                            self.update_q(data)
                            self.update_actor(data)
                            soft_update_params(self.critic, self.critic_target, self.tau)

                ## log
                if (iteration + 1) % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    log_dir = os.path.join(self.log_dir, f"model_{iteration + 1}")
                    os.makedirs(log_dir, exist_ok=True)
                    self.save(log_dir)
                if (iteration + 1) % self.print_interval == 0 and self.print_log:
                    self.log(locals())
                    ep_infos.clear()
                torch.cuda.empty_cache()

    def update_q(self, data: dict[str, Tensor]) -> TensorDict:
        feature = data["feature"]
        next_feature = data["next_feature"]
        action = data["action"]
        reward = data["reward"]
        discount = data["discount"]
        
        with torch.no_grad():
            std = schedule(self.std_schedule, self.global_step)
            dist = self.actor(next_feature, std)
            next_action = dist.sample(clip=self.std_clip)
            target_Q1, target_Q2 = self.critic_target(next_feature, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(feature, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.encoder_optimizer.step()
        self.critic_optimizer.step()
        
        q_mean = 0.5 * (Q1.mean() + Q2.mean()).detach().item()
        self.q_values.update(q_mean)
        self.critic_losses.update(critic_loss.detach().item())

    def update_actor(self, data: dict[str, Tensor]) -> TensorDict:
        feature = data["feature"].detach()
        std = schedule(self.std_schedule, self.global_step)
        dist = self.actor(feature, std)
        action = dist.sample(clip=self.std_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(feature, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor_losses.update(actor_loss.detach().item())

    def log(self, locs, width=80, pad=35):
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""
                if key == "total_succ_rate":
                    locs["mean_succ_rate"] = value.item()

        str = f" \033[1m Learning iteration {locs['iteration']}/{self.max_iterations} \033[0m "
        log_string = (
            f"""{"#" * width}\n"""
            f"""{str.center(width, " ")}\n\n"""
        )
        log_data = {
            "Train/mean_succ_rate": locs["mean_succ_rate"] if "mean_succ_rate" in locs else 0,
            "timesteps_total": self.global_step,
            "iteration": locs["iteration"],
        }

        if not timer.disabled:
            time_ = timer.compute()
            fps = (self.global_step - self.prev_global_step) / time_["time/iteration"]
            if "time/update_model" not in time_:
                update_time = 0.0
            else:
                update_time = time_["time/update_model"]
            roll_out_time = time_["time/roll_out"]
            iteration_time = time_["time/iteration"]
            time_str = f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection {roll_out_time:.3f}s, learning {update_time:.3f}s)\n"""
            log_data["Train/fps"] = fps
            log_data["Time/roll_out"] = roll_out_time
            log_data["Time/update_model"] = update_time
            log_data["Time/iteration"] = iteration_time
            log_string += time_str
            timer.reset()
            self.prev_global_step = self.global_step

        if self.episode_lengths.len > 0:
            episode_length = self.episode_lengths.mean
            episode_reward = self.episode_rewards.mean
            log_string += f"""{"Mean episode length:":>{pad}} {episode_length:.1f}\n"""
            log_string += f"""{"Mean episode reward:":>{pad}} {episode_reward:.3f}\n"""
            log_data["Train/mean_episode_length"] = episode_length
            log_data["Train/mean_reward"] = episode_reward

        log_string += f"""{"Mean reward per step:":>{pad}} {locs["mean_reward"]:.3f}\n"""
        log_string += f"""{"Batch reward:":>{pad}} {self.batch_reward:.3f}\n"""
        log_data["Train/mean_reward_per_step"] = locs["mean_reward"]
        log_data["Train/batch_reward"] = self.batch_reward

        if self.actor_losses.len > 0:
            actor_loss = self.actor_losses.mean
            q_value = self.q_values.mean
            critic_loss = self.critic_losses.mean
            log_string += f"""{"Mean actor loss:":>{pad}} {actor_loss:.3f}\n"""
            log_string += f"""{"Mean critic loss:":>{pad}} {critic_loss:.3f}\n"""
            log_string += f"""{"Mean Q value:":>{pad}} {q_value:.3f}\n"""
            log_data["Loss/mean_actor_loss"] = actor_loss
            log_data["Loss/mean_critic_loss"] = critic_loss
            log_data["Loss/mean_q_value"] = q_value

        tot_time = time.time() - self.start_time
        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.global_step}\n"""
            f"""{"Total time:":>{pad}} {tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {tot_time / (locs["iteration"] + 1) * (self.max_iterations - locs["iteration"]):.1f}s\n"""
        )
        print(log_string)
        if self.wandb_run is not None:
            self.wandb_run.log(log_data)

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter