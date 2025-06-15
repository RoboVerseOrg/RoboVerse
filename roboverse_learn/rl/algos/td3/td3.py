import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tensorboardX import SummaryWriter

from .experience import ReplayBuffer
from .models import Actor, Critic
from .utils import AdaptiveScheduler, AverageScalarMeter, LinearScheduler, RunningMeanStd


class TD3:
    def __init__(self, env, output_dif, full_config):
        self.device = full_config["experiment"]["rl_device"]
        self.network_config = full_config["train"]["td3"]["network"]
        self.td3_config = full_config["train"]["td3"]

        self.env = env
        self.num_actors = self.td3_config["num_actors"]
        self.actions_num = self.td3_config["action_num"]
        self.obs_shape = full_config["train"]["observation_space"]["shape"]

        self.tau = self.td3_config.get("tau", 0.005)
        self.policy_noise = self.td3_config.get("policy_noise", 0.2)
        self.noise_clip = self.td3_config.get("noise_clip", 0.5)
        self.policy_delay = self.td3_config.get("policy_delay", 2)
        self.exploration_noise = self.td3_config.get("exploration_noise", 0.1)

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

        self.actor = Actor(actor_config)
        self.actor_target = Actor(actor_config)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(critic_config)
        self.critic_target = Critic(critic_config)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.normalize_input = self.td3_config.get("normalize_input", True)
        self.normalize_value = self.td3_config.get("normalize_value", True)
        self.normalize_reward = self.td3_config.get("normalize_reward", True)
        self.reward_scale_value = self.td3_config.get("reward_scale_value", 0.01)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        self.reward_mean_std = RunningMeanStd((1,)).to(self.device)

        self.actor_lr = float(self.td3_config["learning_rate"])
        self.critic_lr = float(self.td3_config.get("critic_learning_rate", self.actor_lr))
        self.last_lr = self.actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.lr_schedule = self.td3_config.get("lr_schedule", "adaptive")
        if self.lr_schedule == "adaptive":
            self.kl_threshold = self.td3_config.get("kl_threshold", 0.016)
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == "linear":
            self.scheduler = LinearScheduler(self.actor_lr, self.td3_config["max_agent_steps"])
        else:
            self.scheduler = None

        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, "nn")
        self.tb_dir = os.path.join(self.output_dir, "summaries")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        self.batch_size = self.td3_config["batch_size"]
        self.gamma = self.td3_config["gamma"]
        self.max_epochs = self.td3_config["max_epochs"]
        self.steps_num = self.td3_config["steps_num"]
        self.save_frequency = self.td3_config.get("save_frequency", 1000)
        self.save_best_after = self.td3_config.get("save_best_after", 50)
        self.truncate_grads = self.td3_config.get("truncate_grads", True)
        self.grad_norm = self.td3_config.get("grad_norm", 1.0)

        self.replay_buffer_size = self.td3_config["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.obs_shape, (self.actions_num,), self.replay_buffer_size, self.device)

        self.games_num = self.td3_config["games_per_epoch"]
        self.game_rewards = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.game_lengths = torch.zeros(self.num_actors, dtype=torch.int32, device=self.device)
        self.obs = None
        self.total_time = 0
        self.total_games = 0
        self.mean_rewards = 0
        self.last_mean_rewards = -100500
        self.best_rewards = -100500
        self.frame = 0
        self.epoch_num = 0
        self.update_time = 0
        self.play_time = 0
        self.update_count = 0
        self.agent_steps = 0

        self.data_collect_time = 0
        self.rl_train_time = 0

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)

        self.warmup_steps = self.td3_config.get("warmup_steps", 10000)
        self.learning_starts = self.td3_config.get("learning_starts", 10000)

        self.log_interval = self.td3_config.get("log_interval", 100)
        self.eval_interval = self.td3_config.get("eval_interval", 1000)

    def save(self, filename):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "epoch": self.epoch_num,
            "frame": self.frame,
            "agent_steps": self.agent_steps,
            "best_rewards": self.best_rewards,
        }
        if self.normalize_input:
            checkpoint["running_mean_std"] = self.running_mean_std.state_dict()
        if self.normalize_value:
            checkpoint["value_mean_std"] = self.value_mean_std.state_dict()
        if self.normalize_reward:
            checkpoint["reward_mean_std"] = self.reward_mean_std.state_dict()
        torch.save(checkpoint, f"{filename}.pth")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.epoch_num = checkpoint.get("epoch", 0)
        self.frame = checkpoint.get("frame", 0)
        self.agent_steps = checkpoint.get("agent_steps", 0)
        self.best_rewards = checkpoint.get("best_rewards", -100500)

        if self.normalize_input and "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_value and "value_mean_std" in checkpoint:
            self.value_mean_std.load_state_dict(checkpoint["value_mean_std"])
        if self.normalize_reward and "reward_mean_std" in checkpoint:
            self.reward_mean_std.load_state_dict(checkpoint["reward_mean_std"])

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def write_stats(self, actor_losses, critic_losses):
        log_dict = {
            "performance/RLTrainFPS": self.agent_steps / self.rl_train_time if self.rl_train_time > 0 else 0,
            "performance/EnvStepFPS": self.agent_steps / self.data_collect_time if self.data_collect_time > 0 else 0,
            "losses/actor_loss": np.mean(actor_losses) if actor_losses else 0,
            "losses/critic_loss": np.mean(critic_losses) if critic_losses else 0,
            "info/last_lr": self.last_lr,
            "info/replay_buffer_size": self.replay_buffer.idx
            if not self.replay_buffer.full
            else self.replay_buffer.capacity,
            "info/exploration_noise": self.exploration_noise,
            "info/updates": self.update_count,
        }

        wandb.log(log_dict, step=self.agent_steps)

        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def update_networks(self):
        if not hasattr(self.replay_buffer, "idx") or self.replay_buffer.idx < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        if self.normalize_input:
            obs = self.running_mean_std(obs)
            next_obs = self.running_mean_std(next_obs)

        if self.normalize_reward:
            rewards = self.reward_mean_std(rewards)
        else:
            rewards = rewards * self.reward_scale_value

        obs_dict = {"obs": obs}
        next_obs_dict = {"obs": next_obs}

        with torch.no_grad():
            next_actions = self.actor_target.act(
                next_obs_dict, target_noise=self.policy_noise, noise_clip=self.noise_clip
            )

            target_Q1, target_Q2 = self.critic_target(next_obs_dict, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones.float()) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(obs_dict, actions)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
        self.critic_optimizer.step()

        losses_dict = {"critic_loss": critic_loss.item()}

        if self.update_count % self.policy_delay == 0:
            actor_actions = self.actor(obs_dict)
            actor_loss = -self.critic.Q1_forward(obs_dict, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
            self.actor_optimizer.step()

            self.soft_update(self.critic_target, self.critic, self.tau)
            self.soft_update(self.actor_target, self.actor, self.tau)

            losses_dict["actor_loss"] = actor_loss.item()

        self.update_count += 1
        return losses_dict

    def play_steps(self):
        play_time_start = time.time()

        if self.obs is None:
            self.obs = self.env.reset()

        obs_tensor = self.obs["obs"] if isinstance(self.obs, dict) else self.obs

        if self.normalize_input:
            obs_normalized = self.running_mean_std(obs_tensor)
        else:
            obs_normalized = obs_tensor

        obs_dict = {"obs": obs_normalized}

        if self.frame < self.warmup_steps:
            actions = torch.rand((self.num_actors, self.actions_num), device=self.device) * 2 - 1
        else:
            with torch.no_grad():
                actions = self.actor.act(obs_dict, exploration_noise=self.exploration_noise)

        next_obs, rewards, dones, timeouts, terminations = self.env.step(actions)

        self.game_rewards += rewards.squeeze(-1)
        self.game_lengths += 1

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            for done_idx in done_indices:
                game_reward = self.game_rewards[done_idx].item()
                game_length = self.game_lengths[done_idx].item()

                self.episode_rewards.update(torch.tensor([game_reward]))
                self.episode_lengths.update(torch.tensor([game_length]))

                self.writer.add_scalar("episode_rewards/train", game_reward, self.frame)
                self.writer.add_scalar("episode_lengths/train", game_length, self.frame)

                self.game_rewards[done_idx] = 0
                self.game_lengths[done_idx] = 0
                self.total_games += 1

        obs_tensor = self.obs["obs"] if isinstance(self.obs, dict) else self.obs
        next_obs_tensor = next_obs["obs"] if isinstance(next_obs, dict) else next_obs

        self.replay_buffer.add(
            obs_tensor, actions, rewards.view(self.num_actors, 1), next_obs_tensor, dones.view(self.num_actors, 1)
        )

        self.obs = next_obs
        self.frame += self.num_actors
        self.agent_steps += self.num_actors
        self.data_collect_time += time.time() - play_time_start

        return len(done_indices)

    def train_epoch(self):
        epoch_start_time = time.time()

        games_played = 0
        actor_losses = []
        critic_losses = []

        for _ in range(self.steps_num):
            games_played += self.play_steps()

            if self.agent_steps >= self.learning_starts:
                update_time_start = time.time()
                losses = self.update_networks()
                self.rl_train_time += time.time() - update_time_start

                if losses:
                    if "critic_loss" in losses:
                        critic_losses.append(losses["critic_loss"])
                    if "actor_loss" in losses:
                        actor_losses.append(losses["actor_loss"])

        self.epoch_num += 1
        epoch_time = time.time() - epoch_start_time

        self.mean_rewards = self.episode_rewards.get_mean()
        self.mean_lengths = self.episode_lengths.get_mean()

        if self.scheduler is not None:
            if self.lr_schedule == "linear":
                self.last_lr = self.scheduler.update(self.agent_steps)

            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = self.last_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = self.last_lr

        self.write_stats(actor_losses, critic_losses)

        self.writer.add_scalar("metrics/episode_rewards_per_step", self.mean_rewards, self.agent_steps)
        self.writer.add_scalar("metrics/episode_lengths_per_step", self.mean_lengths, self.agent_steps)
        wandb.log(
            {
                "metrics/episode_rewards_per_step": self.mean_rewards,
                "metrics/episode_lengths_per_step": self.mean_lengths,
            },
            step=self.agent_steps,
        )

        checkpoint_name = f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{self.mean_rewards:.2f}"

        if self.save_frequency > 0:
            if (self.epoch_num % self.save_frequency == 0) and (self.mean_rewards <= self.best_rewards):
                self.save(os.path.join(self.nn_dir, checkpoint_name))
            self.save(os.path.join(self.nn_dir, "last"))

        if self.mean_rewards > self.best_rewards:
            print(f"save current best reward: {self.mean_rewards:.2f}")
            prev_best_ckpt = os.path.join(self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth")
            if os.path.exists(prev_best_ckpt):
                os.remove(prev_best_ckpt)
            self.best_rewards = self.mean_rewards
            self.save(os.path.join(self.nn_dir, f"best_reward_{self.mean_rewards:.2f}"))

        return self.mean_rewards

    def train(self):
        self.start_time = time.time()
        _t = time.time()
        _last_t = time.time()

        if self.obs is None:
            self.obs = self.env.reset()

        if self.normalize_input:
            self.running_mean_std.train()
            for _ in range(10):
                obs_tensor = self.obs["obs"] if isinstance(self.obs, dict) else self.obs
                _ = self.running_mean_std(obs_tensor)
            self.running_mean_std.eval()

        max_agent_steps = self.td3_config.get("max_agent_steps", 500000000)

        while self.agent_steps < max_agent_steps:
            self.epoch_num += 1
            mean_rewards = self.train_epoch()

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (self.num_actors * self.steps_num) / (time.time() - _last_t)
            _last_t = time.time()

            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | "
                f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                f"Current Best: {self.best_rewards:.2f}"
            )
            print(info_string)

            reward_threshold = self.td3_config.get("reward_threshold", None)
            if reward_threshold is not None and mean_rewards >= reward_threshold:
                print(f"Reached reward threshold {reward_threshold}")
                break

        print("max steps achieved")
