from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import copy
import os
import time
from collections import deque

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from roboverse_learn.dexbench_rvrl.algos.ppo.module import ActorCritic
from roboverse_learn.dexbench_rvrl.algos.ppo.storage import RolloutStorage


########################################################
## Standalone utils
########################################################
class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards: torch.Tensor):
        self.deque.extend(rewards.cpu().numpy().tolist())

    @property
    def len(self) -> int:
        return len(self.deque)

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


########################################################
## Networks
########################################################
class PPO:
    def __init__(
        self,
        env,
        train_cfg,
        device="cpu",
        log_dir="run",
        model_dir=None,
        is_testing=False,
        print_log=True,
        wandb_run=None,
    ):
        self.device = device

        ## env info
        self.action_dim = np.prod(env.single_action_space.shape)
        self.obs_shape = {k: v.shape for k, v in env.single_observation_space.spaces.items()}
        self.obs_type = getattr(env, "obs_type", "state")
        self.img_h = getattr(env, "img_h", None)
        self.img_w = getattr(env, "img_w", None)

        ## learn cfg
        self.train_cfg = copy.deepcopy(train_cfg)
        learn_cfg = self.train_cfg["learn"]
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.sampler = learn_cfg.get("sampler", "sequential")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.nsteps = learn_cfg["nsteps"]
        self.num_envs = env.num_envs
        self.num_learning_epochs = learn_cfg.get("num_learning_epochs", 8)
        self.max_iterations = learn_cfg.get("max_iterations", 100000)
        self.log_interval = learn_cfg.get("log_interval", 1)

        ## model cfg
        self.model_cfg = self.train_cfg["policy"]

        ## ppo components
        self.env = env
        print(f"Observation type: {self.obs_type}")
        self.actor_critic = ActorCritic(
            obs_type=self.obs_type,
            obs_shape=self.obs_shape,
            actions_shape=self.action_dim,
            initial_std=self.init_noise_std,
            model_cfg=self.model_cfg,
            img_h=self.img_h,
            img_w=self.img_w,
        )

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size)

        self.buffer = RolloutStorage(
            self.num_envs, self.nsteps, self.obs_shape, self.action_dim, device=self.device, sampler=self.sampler
        )

        self.episode_rewards = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_lengths = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_rewards_step = RollingMeter(learn_cfg.get("window_size", 2000))
        self.cur_rewards_sum = torch.zeros(self.num_envs, device=device)
        self.cur_episode_length = torch.zeros(self.num_envs, device=device)

        ## PPO params
        self.num_mini_batches = learn_cfg.get("num_minibatches", 32)
        self.clip_param = learn_cfg.get("clip_param", 0.2)
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg.get("entropy_coef", 0.0)
        self.gamma = learn_cfg.get("gamma", 0.99)
        self.lam = learn_cfg.get("lam", 0.95)
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)

        ## logging
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.print_log = print_log
        self.wandb_run = wandb_run
        self.is_testing = is_testing
        self.current_learning_epoch = 0
        self.current_learning_iteration = 0
        self.global_step = 0
        self.tot_time = 0

        self.actor_critic.to(self.device)

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self):
        if self.is_testing:
            assert self.model_dir is not None
            self.test(self.model_dir)
            print(f"Loaded model from {self.model_dir}")
        elif self.model_dir is not None:
            self.load(self.model_dir)
            print(f"Loaded model from {self.model_dir}")
        rest_obs = self.env.reset()
        obs = {k: rest_obs[k].clone() for k in rest_obs.keys()}
        if not self.is_testing:
            for iteration in range(self.current_learning_iteration, self.max_iterations):
                ## collect rollout
                start_time = time.time()
                ep_infos = []
                for t in range(self.nsteps):
                    self.global_step += self.num_envs

                    action, log_prob, value, mu, sigma = self.actor_critic.act(obs)
                    next_obs, reward, terminated, truncated, infos = self.env.step(action)
                    dones = torch.logical_or(terminated, truncated)
                    self.buffer.add_transitions(obs, action, reward, dones, value.view(-1), log_prob, mu, sigma)

                    for k in obs.keys():
                        obs[k].copy_(next_obs[k])

                    self.cur_rewards_sum += reward
                    self.cur_episode_length += 1
                    self.episode_rewards.update(self.cur_rewards_sum[dones])
                    self.episode_lengths.update(self.cur_episode_length[dones])
                    self.episode_rewards_step.update(reward)
                    self.cur_rewards_sum[dones] = 0
                    self.cur_episode_length[dones] = 0
                    ep_infos.append(infos)

                _, _, last_values, _, _ = self.actor_critic.act(obs)
                collection_time = time.time() - start_time
                self.tot_time += collection_time

                mean_episode_length = self.episode_lengths.mean if self.episode_lengths.len > 0 else 0
                mean_episode_reward = self.episode_rewards.mean if self.episode_rewards.len > 0 else 0
                mean_reward = self.episode_rewards_step.mean if self.episode_rewards_step.len > 0 else 0

                compute_start_time = time.time()
                self.buffer.compute_returns(last_values.view(-1), self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss, mean_kl = self.update()
                self.buffer.clear()
                learn_time = time.time() - compute_start_time
                self.tot_time += learn_time
                if self.print_log:
                    self.log(locals())
                if (iteration + 1) % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))
                ep_infos.clear()
        else:
            raise NotImplementedError

    def update(self):
        ## log
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        num_updates = 0
        kl_over_limit = False

        ## flatten and shuffle
        batch = self.buffer.mini_batch_generator(self.num_mini_batches)

        ## optimize
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = {
                    key: self.buffer.observations[key].view(-1, *self.obs_shape[key])[indices]
                    if "rgb" not in key
                    else self.buffer.observations[key].view(-1, *self.obs_shape[key])[indices].float() / 255.0
                    for key in self.obs_shape
                }
                actions_batch = self.buffer.actions.view(-1, self.action_dim)[indices]
                returns_batch = self.buffer.returns.view(-1)[indices]
                old_log_probs_batch = self.buffer.actions_log_prob.view(-1)[indices]
                advantages_batch = self.buffer.advantages.view(-1)[indices]
                old_mu_batch = self.buffer.mu.view(-1, self.action_dim)[indices]
                old_sigma_batch = self.buffer.sigma.view(-1, self.action_dim)[indices]

                new_log_probs_batch, entropy_batch, values_batch, new_mu_batch, new_sigma_batch = (
                    self.actor_critic.evaluate(obs_batch, actions_batch)
                )

                # KL
                if self.desired_kl is not None and (self.schedule == "adaptive" or self.schedule == "clip"):
                    kl = torch.sum(
                        new_sigma_batch
                        - old_sigma_batch
                        + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - new_mu_batch))
                        / (2.0 * torch.square(new_sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 1.5 and self.schedule == "clip":
                        kl_over_limit = True
                        break
                    elif self.schedule == "adaptive":
                        if kl_mean > self.desired_kl * 2.0:
                            self.step_size = max(1e-5, self.step_size / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.step_size = min(1e-3, self.step_size * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.step_size
                    mean_kl += kl_mean.item()

                # surrogate loss
                ratio = torch.exp(new_log_probs_batch - old_log_probs_batch)

                surr_loss1 = -advantages_batch * ratio
                surr_loss2 = -advantages_batch * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                surr_loss = torch.max(surr_loss1, surr_loss2).mean()

                # value loss
                value_loss = (returns_batch - values_batch.view(-1)).pow(2).mean()

                # total loss
                loss = surr_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surr_loss.item()
                num_updates += 1
            if kl_over_limit:
                break

        mean_value_loss = mean_value_loss / num_updates if num_updates > 0 else 0
        mean_surrogate_loss = mean_surrogate_loss / num_updates if num_updates > 0 else 0
        mean_kl = mean_kl / num_updates if num_updates > 0 else 0

        return mean_value_loss, mean_surrogate_loss, mean_kl

    def log(self, locs, width=80, pad=35):
        iteration_time = locs["collection_time"] + locs["learn_time"]
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
        mean_std = self.actor_critic.log_std.exp().mean()

        fps = int(self.nsteps * self.num_envs / iteration_time)
        str = f" \033[1m Learning iteration {locs['iteration']}/{self.max_iterations} \033[0m "

        if self.episode_rewards.len > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                f"""{"KL":>{pad}} {locs["mean_kl"]:.6f}\n"""
                f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
                f"""{"Mean reward:":>{pad}} {locs["mean_episode_reward"]:.2f}\n"""
                f"""{"Mean episode length:":>{pad}} {locs["mean_episode_length"]:.2f}\n"""
                f"""{"Mean reward/step:":>{pad}} {locs["mean_reward"]:.2f}\n"""
            )
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                f"""{"KL":>{pad}} {locs["mean_kl"]:.6f}\n"""
                f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
                f"""{"Mean reward/step:":>{pad}} {locs["mean_reward"]:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.global_step}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {self.tot_time / (locs["iteration"] + 1) * (self.max_iterations - locs["iteration"]):.1f}s\n"""
        )
        print(log_string)
        if self.wandb_run is not None:
            log_data = {
                "Loss/surrogate": locs["mean_surrogate_loss"],
                "Loss/value_function": locs["mean_value_loss"],
                "Policy/noise_std": mean_std.item(),
                "Train/mean_reward_per_step": locs["mean_reward"],
                "Train/mean_succ_rate": locs["mean_succ_rate"],
                "Train/fps": fps,
                "timesteps_total": self.global_step,
                "iteration_time": iteration_time,
                "iteration": locs["iteration"],
                "Train/mean_reward": locs["mean_episode_reward"],
                "Train/mean_episode_length": locs["mean_episode_length"],
                "Train/kl": locs["mean_kl"],
            }

            self.wandb_run.log(log_data)
