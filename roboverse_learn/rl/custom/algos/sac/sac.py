import copy
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from .experience import ReplayBuffer
from .models import Actor, Critic
from .running_mean_std import RunningMeanStd


class SAC:
    def __init__(self, env, output_dif, full_config):
        self.device = full_config["experiment"]["rl_device"]
        self.network_config = full_config["train"]["sac"]["network"]
        self.sac_config = full_config["train"]["sac"]
        # ---- build environment ----
        self.env = env
        self.num_actors = self.sac_config["num_actors"]
        # action_space = full_config["action_space"]
        self.actions_num = self.sac_config["action_num"]
        # self.observation_space = full_config["observation_space"]
        self.obs_shape = full_config["train"]["observation_space"]["shape"]
        # ---- Model ----
        actor_config = {
            "actor_units": self.network_config["mlp"]["units"],
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
        }
        critic_config = {
            "critic_units": self.network_config["mlp"]["units"],
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
        }
        target_critic_config = copy.deepcopy(critic_config)
        self.actor = Actor(actor_config)
        self.critic = Critic(critic_config)
        self.target_critic = Critic(target_critic_config)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, "stage1_nn")
        self.tb_dif = os.path.join(self.output_dir, "stage1_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- SAC Train Param ----
        self.gamma = self.sac_config["gamma"]
        self.normalize_input = self.sac_config.get("normalize_input", True)
        self.normalize_value = self.sac_config.get("normalize_value", True)
        self.alpha = self.sac_config.get("init_temperature", 0.1)
        self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
        self.alpha = torch.tensor(self.alpha, device=self.device)
        self.target_entropy = -torch.prod(torch.Tensor((self.actions_num,)).to(self.device)).item()
        self.soft_target_update_rate = self.sac_config.get("soft_target_update_rate", 0.005)
        self.use_entropy_reward = self.sac_config.get("use_entropy_reward", True)
        self.reward_shaper = self.sac_config.get("reward_shaper", 1)
        self.num_actors = self.sac_config["num_actors"]  # num_envs
        self.truncate_grads = self.sac_config.get("truncate_grads", False)
        self.grad_norm = self.sac_config.get("grad_norm", 1.0)
        self.gradient_steps = self.sac_config.get("gradient_steps", 1)
        self.num_frames_per_epoch = self.num_actors * self.gradient_steps
        # ---- Optim ----
        self.weight_decay = self.sac_config.get("weight_decay", 0)
        self.last_actor_lr = float(self.sac_config.get("actor_lr", 5e-4))
        self.last_critic_lr = float(self.sac_config.get("critic_lr", 5e-4))
        self.last_alpha_lr = float(self.sac_config.get("alpha_lr", 5e-3))
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.last_actor_lr, weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), self.last_critic_lr, weight_decay=self.weight_decay
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], self.last_alpha_lr, weight_decay=self.weight_decay
        )
        self.min_alpha = torch.tensor(self.sac_config.get("min_temperature", 0.05), device=self.device)
        self.c_loss = torch.nn.MSELoss()
        # ---- SAC Collect Param ----
        self.memory_size = self.sac_config.get("memory_size", 1000000)
        self.batch_size = self.sac_config.get("batch_size", 32768)
        self.random_steps = self.sac_config.get("random_steps", 20)
        self.start_step = self.sac_config.get("start_step", 5)
        # ---- scheduler ----
        self.kl_threshold = self.sac_config.get("kl_threshold", 0.008)
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.log_alpha_scheduler = None
        # ---- Snapshot
        self.save_freq = self.sac_config.get("save_frequency", 100)
        self.eval_freq = self.sac_config.get("eval_frequency", 10)
        self.save_best_after = self.sac_config.get("save_best_after", 50)
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        self.last_mean_rewards = 0
        self.storage = ReplayBuffer(self.obs_shape, (self.actions_num,), self.memory_size, self.device)
        current_rewards_shape = (self.num_actors, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros((self.num_actors, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.ones((self.num_actors,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.sac_config.get("max_agent_steps", 10000000)
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0
        self.max_env_steps = self.sac_config.get("max_env_steps", 1000)

    def write_stats(self, actor_loss, critic_loss, alpha_loss, log_prob, actor_lr, critic_lr, alpha_lr, alpha):
        self.writer.add_scalar(
            "performance/RLTrainFPS", self.agent_steps / (self.rl_train_time + 1e-9), self.agent_steps
        )
        self.writer.add_scalar(
            "performance/EnvStepFPS", self.agent_steps / (self.data_collect_time + 1e-9), self.agent_steps
        )

        self.writer.add_scalar("losses/actor_loss", actor_loss, self.agent_steps)
        self.writer.add_scalar("losses/critic_loss", critic_loss, self.agent_steps)
        self.writer.add_scalar("losses/alpha_loss", alpha_loss, self.agent_steps)
        self.writer.add_scalar("losses/alpha_loss", alpha_loss, self.agent_steps)
        self.writer.add_scalar("losses/log_prob", log_prob, self.agent_steps)

        self.writer.add_scalar("info/actor_lr", actor_lr, self.agent_steps)
        self.writer.add_scalar("info/critic_lr", critic_lr, self.agent_steps)
        self.writer.add_scalar("info/alpha_lr", alpha_lr, self.agent_steps)

        self.writer.add_scalar("debug/alpha", alpha, self.agent_steps)

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f"{k}", v, self.agent_steps)

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        if self.normalize_input:
            self.running_mean_std.eval()

    def set_train(self):
        self.actor.train()
        self.critic.eval()
        if self.normalize_input:
            self.running_mean_std.train()

    def train(self):
        total_time = 0
        self.obs = self.env.reset()

        while True:
            self.epoch_num += 1
            (
                step_time,
                play_time,
                update_time,
                epoch_total_time,
                actor_losses,
                entropies,
                alphas,
                alpha_losses,
                critic1_losses,
                critic2_losses,
            ) = self.train_epoch()

            total_time += epoch_total_time
            curr_frames = self.num_frames_per_epoch
            self.agent_steps += curr_frames

            fps_step = curr_frames / step_time
            fps_step_inference = curr_frames / play_time
            fps_total = curr_frames / epoch_total_time
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {fps_total:.1f} | "
                f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                f"Current Best: {self.best_rewards:.2f}"
            )
            print(info_string)

            self.writer.add_scalar("performance/step_inference_rl_update_fps", fps_total, self.agent_steps)
            self.writer.add_scalar("performance/step_inference_fps", fps_step_inference, self.agent_steps)
            self.writer.add_scalar("performance/step_fps", fps_step, self.agent_steps)
            self.writer.add_scalar("performance/rl_update_time", update_time, self.agent_steps)
            self.writer.add_scalar("performance/step_inference_time", play_time, self.agent_steps)
            self.writer.add_scalar("performance/step_time", step_time, self.agent_steps)

            if self.epoch_num >= self.random_steps:
                self.write_stats(
                    actor_loss=actor_losses[0].item(),
                    critic_loss=(critic1_losses[0].item() + critic2_losses[0].item()) / 2.0,
                    alpha_loss=alpha_losses[0].item(),
                    log_prob=entropies[0].item(),
                    actor_lr=self.last_actor_lr,
                    critic_lr=self.last_critic_lr,
                    alpha_lr=self.last_alpha_lr,
                    alpha=self.alpha[0].item(),
                )

            self.writer.add_scalar("info/epochs", self.epoch_num, self.agent_steps)

            if self.episode_rewards.current_size > 0:
                mean_rewards = self.episode_rewards.get_mean()
                mean_lengths = self.episode_lengths.get_mean()
                self.best_rewards = max(self.best_rewards, mean_rewards)

                self.writer.add_scalar("episode_rewards/step", mean_rewards, self.agent_steps)
                self.writer.add_scalar("episode_lengths/step", mean_lengths, self.agent_steps)
                checkpoint_name = (
                    f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}"
                )
                should_exit = False

                if self.save_freq > 0:
                    if self.epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, "last"))

                if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                    print("saving next best rewards: ", mean_rewards)
                    self.last_mean_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, "best"))

                update_time = 0

        print("max steps achieved")

    def extract_actor_stats(self, actor_losses, entropies, alphas, alpha_losses, actor_loss_info):
        actor_loss, entropy, alpha, alpha_loss = actor_loss_info

        actor_losses.append(actor_loss)
        entropies.append(entropy)
        if alpha_losses is not None:
            alphas.append(alpha)
            alpha_losses.append(alpha_loss)

    def train_epoch(self):
        random_exploration = self.epoch_num < self.random_steps
        return self.play_steps(random_exploration)

    def act(self, obs, action_dim, sample=False):
        obs = self.running_mean_std(obs)
        input_dict = {
            "obs": obs,
        }
        actions, _, _ = self.actor.get_action(input_dict)

        assert actions.ndim == 2

        return actions

    def play_steps(self, random_exploration=False):
        total_time_start = time.time()
        total_update_time = 0
        total_time = 0
        step_time = 0.0
        actor_losses = []
        entropies = []
        alphas = []
        alpha_losses = []
        critic1_losses = []
        critic2_losses = []

        obs = self.obs["obs"].clone()

        next_obs_processed = obs.clone()

        for s in range(self.gradient_steps):
            self.set_eval()
            if random_exploration:
                action = torch.rand((self.num_actors, self.actions_num)) * 2.0 - 1.0
            else:
                with torch.no_grad():
                    action = self.act(obs.float(), self.actions_num, sample=True)

            step_start = time.time()

            with torch.no_grad():
                next_obs, rewards, dones, timeout, infos = self.env.step(action)
            step_end = time.time()

            self.current_rewards += rewards
            self.current_lengths += 1

            total_time += step_end - step_start
            step_time += step_end - step_start

            done_indices = dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            not_dones = 1.0 - dones.float()
            no_timeouts = self.current_lengths != self.max_env_steps
            no_timeouts = no_timeouts.squeeze(1)
            dones = dones * no_timeouts

            not_dones = not_dones.unsqueeze(1)
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

            if isinstance(next_obs, dict):
                next_obs_processed = next_obs["obs"]

            self.obs = next_obs.copy()

            rewards = rewards * self.reward_shaper

            self.storage.add(obs, action, rewards, next_obs_processed, torch.unsqueeze(dones, 1))

            if isinstance(obs, dict):
                obs = self.obs["obs"]

            if not random_exploration:
                self.set_train()
                update_time_start = time.time()
                actor_loss_info, critic1_loss, critic2_loss = self.update(self.epoch_num)
                update_time_end = time.time()
                update_time = update_time_end - update_time_start

                self.extract_actor_stats(actor_losses, entropies, alphas, alpha_losses, actor_loss_info)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
            else:
                update_time = 0

            total_update_time += update_time

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        self.data_collect_time = total_time - total_update_time
        self.rl_train_time = total_update_time

        return (
            step_time,
            self.data_collect_time,
            total_update_time,
            total_time,
            actor_losses,
            entropies,
            alphas,
            alpha_losses,
            critic1_losses,
            critic2_losses,
        )

    def save(self, name):
        weights = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha,
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.log_alpha = checkpoint["log_alpha"]
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.log_alpha = checkpoint["log_alpha"]
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]),
            }
            mu = self.actor.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def update(self, step):
        obs, action, reward, next_obs, done = self.storage.sample(self.batch_size)
        not_done = ~done

        obs = self.running_mean_std(obs)
        next_obs = self.running_mean_std(next_obs)
        critic_loss, critic1_loss, critic2_loss = self.update_critic(obs, action, reward, next_obs, not_done, step)

        actor_loss, entropy, alpha, alpha_loss = self.update_actor_and_alpha(obs, step)

        actor_loss_info = actor_loss, entropy, alpha, alpha_loss
        self.soft_update_params(self.critic, self.target_critic, self.soft_target_update_rate)
        return actor_loss_info, critic1_loss, critic2_loss

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        with torch.no_grad():
            input_dict = {
                "obs": next_obs,
            }
            next_action, log_prob, _ = self.actor.get_action(input_dict)

            target_Q1, target_Q2 = self.target_critic(input_dict, next_action)

            target_V = torch.min(target_Q1, target_Q2) - self.alpha * log_prob

            target_Q = reward + (not_done * self.gamma * target_V)
            target_Q = target_Q.detach()

        input_dict = {
            "obs": obs,
        }
        current_Q1, current_Q2 = self.critic(input_dict, action)

        critic1_loss = self.c_loss(current_Q1, target_Q)
        critic2_loss = self.c_loss(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
        self.critic_optimizer.step()

        return critic_loss.detach(), critic1_loss.detach(), critic2_loss.detach()

    def update_actor_and_alpha(self, obs, step):
        for p in self.target_critic.parameters():
            p.requires_grad = False

        input_dict = {
            "obs": obs,
        }
        action, log_prob, _ = self.actor.get_action(input_dict)
        entropy = -log_prob.mean()
        actor_Q1, actor_Q2 = self.critic(input_dict, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = torch.max(self.alpha, self.min_alpha) * log_prob - actor_Q
        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        if self.use_entropy_reward:
            alpha_loss = (self.log_alpha.exp() * (-log_prob - self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        else:
            alpha_loss = None

        self.alpha = self.log_alpha.exp()
        return actor_loss.detach(), entropy.detach(), self.log_alpha.exp().detach(), alpha_loss

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler:
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class AverageScalarMeter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
