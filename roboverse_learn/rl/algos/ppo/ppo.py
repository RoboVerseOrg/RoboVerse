# Reference: https://github.com/ToruOwO/minimal-stable-PPO/tree/main
import math
import os
import time

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

import wandb

from .experience import ExperienceBuffer
from .models import ActorCritic
from .utils import AverageScalarMeter, RunningMeanStd


class PPO(object):
    def __init__(self, env, output_dif, full_config, verbose=False):
        # ---- MultiGPU ----
        self.multi_gpu = full_config["experiment"]["multi_gpu"]
        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
            print(f"current rank: {self.rank} and use device {self.device}")
        else:
            self.rank = -1
            self.device = full_config["experiment"]["rl_device"]
        self.network_config = full_config["train"]["ppo"]["network"]
        self.ppo_config = full_config["train"]["ppo"]
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        self.actions_num = self.ppo_config["action_num"]  # XXX: replace this with action space from robot config
        # self.observation_space = full_config[
        #     "observation_space"
        # ]  # XXX: replace this with observation space from robot config
        self.obs_shape = full_config["train"]["observation_space"]["shape"]
        # ---- Model ----
        net_config = {
            "actor_units": self.network_config["mlp"]["units"],
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "separate_value_mlp": self.network_config.get("separate_value_mlp", True),
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, "nn")
        self.tb_dif = os.path.join(self.output_dir, "tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config["learning_rate"])
        self.last_lr = float(self.ppo_config["learning_rate"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.init_lr, eps=1e-5)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config["e_clip"]
        self.clip_value = self.ppo_config["clip_value"]
        self.entropy_coef = self.ppo_config["entropy_coef"]
        self.critic_coef = self.ppo_config["critic_coef"]
        self.bounds_loss_coef = self.ppo_config["bounds_loss_coef"]
        self.gamma = self.ppo_config["gamma"]
        self.tau = self.ppo_config["tau"]
        self.truncate_grads = self.ppo_config["truncate_grads"]
        self.grad_norm = self.ppo_config["grad_norm"]
        self.value_bootstrap = self.ppo_config["value_bootstrap"]
        self.normalize_advantage = self.ppo_config["normalize_advantage"]
        self.normalize_input = self.ppo_config["normalize_input"]
        self.normalize_value = self.ppo_config["normalize_value"]
        self.reward_scale_value = self.ppo_config["reward_scale_value"]
        self.clip_value_loss = self.ppo_config["clip_value_loss"]
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config["horizon_length"]
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config["minibatch_size"]
        self.mini_epochs_num = self.ppo_config["mini_epochs"]
        assert self.batch_size % self.minibatch_size == 0
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config["lr_schedule"]
        if self.lr_schedule == "kl":
            self.kl_threshold = self.ppo_config["kl_threshold"]
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == "linear":
            self.scheduler = LinearScheduler(self.init_lr, self.ppo_config["max_agent_steps"])
        # ---- Snapshot
        self.save_freq = self.ppo_config["save_frequency"]
        self.save_best_after = self.ppo_config["save_best_after"]
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_shape[0],
            self.actions_num,
            self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config["max_agent_steps"]
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        log_dict = {
            "performance/RLTrainFPS": self.agent_steps / self.rl_train_time,
            "performance/EnvStepFPS": self.agent_steps / self.data_collect_time,
            "losses/actor_loss": torch.mean(torch.stack(a_losses)).item(),
            "losses/bounds_loss": torch.mean(torch.stack(b_losses)).item(),
            "losses/critic_loss": torch.mean(torch.stack(c_losses)).item(),
            "losses/entropy": torch.mean(torch.stack(entropies)).item(),
            "info/last_lr": self.last_lr,
            "info/e_clip": self.e_clip,
            "info/kl": torch.mean(torch.stack(kls)).item(),
        }
        for k, v in self.extra_info.items():
            log_dict[f"{k}"] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        # Check for NaN/inf in observations
        obs = obs_dict["obs"]
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"Warning: NaN/inf detected in observations before normalization. Shape: {obs.shape}")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            obs_dict["obs"] = obs

        processed_obs = self.running_mean_std(obs_dict["obs"])

        # Check again after normalization
        if torch.isnan(processed_obs).any() or torch.isinf(processed_obs).any():
            print(f"Warning: NaN/inf detected in observations after normalization. Shape: {processed_obs.shape}")
            processed_obs = torch.nan_to_num(processed_obs, nan=0.0, posinf=5.0, neginf=-5.0)

        input_dict = {
            "obs": processed_obs,
        }
        res_dict = self.model.act(input_dict)
        res_dict["values"] = self.value_mean_std(res_dict["values"], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        # Verify observation shape matches configuration
        actual_obs_shape = self.obs["obs"].shape[1:]  # Remove batch dimension
        expected_obs_shape = tuple(self.obs_shape)
        if actual_obs_shape != expected_obs_shape:
            print(f"Warning: Observation shape mismatch!")
            print(f"  Expected: {expected_obs_shape}")
            print(f"  Actual: {actual_obs_shape}")
            print(f"  First observation sample: {self.obs['obs'][0][:10]}...")

        # Warm up the observation normalizer with initial observations
        if self.normalize_input:
            print("Warming up observation normalizer...")
            self.running_mean_std.train()
            warmup_steps = 100 if self.num_actors == 1 else 10
            for _ in range(warmup_steps):
                random_actions = torch.rand((self.num_actors, self.actions_num), device=self.device) * 2 - 1
                next_obs, _, _, _, _ = self.env.step(random_actions)
                _ = self.running_mean_std(next_obs["obs"])

            self.obs = self.env.reset()

            _ = self.running_mean_std(self.obs["obs"])

            self.running_mean_std.eval()
            print(f"Normalizer warmup complete. Mean: {self.running_mean_std.running_mean[:5]}, Var: {self.running_mean_std.running_var[:5]}")

        self.agent_steps = self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            if self.lr_schedule == "linear":
                self.last_lr = self.scheduler.update(self.agent_steps)

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = (self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size) / (
                    time.time() - _last_t
                )
                _last_t = time.time()
                info_string = (
                    f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                    f"Last FPS: {last_fps:.1f} | "
                    f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                    f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                    f"Current Best: {self.best_rewards:.2f}"
                )
                print(info_string)

                self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

                mean_rewards = self.episode_rewards.get_mean()
                mean_lengths = self.episode_lengths.get_mean()
                self.writer.add_scalar("metrics/episode_rewards_per_step", mean_rewards, self.agent_steps)
                self.writer.add_scalar("metrics/episode_lengths_per_step", mean_lengths, self.agent_steps)
                wandb.log(
                    {
                        "metrics/episode_rewards_per_step": mean_rewards,
                        "metrics/episode_lengths_per_step": mean_lengths,
                    },
                    step=self.agent_steps,
                )
                checkpoint_name = (
                    f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}"
                )

                if self.save_freq > 0:
                    if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, f"last"))

                if mean_rewards > self.best_rewards:
                    print(f"save current best reward: {mean_rewards:.2f}")
                    # remove previous best file
                    prev_best_ckpt = os.path.join(self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth")
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}"))

        print("max steps achieved")

    def eval(self):
        self.set_eval()
        for _ in range(self.num_eval_steps):
            actions = self.model_act(self.obs)["actions"]
            self.obs, rewards, self.dones, timeout, infos = self.env.step(actions)

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights["value_mean_std"] = self.value_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def test(self, video_fn="eval"):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]),
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            info["reward"] = r

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        # update network
        _t = time.time()
        self.set_train()
        a_losses, c_losses, b_losses = [], [], []
        entropies, kls = [], []
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs = self.storage[i]

                obs = self.running_mean_std(obs)
                batch_dict = {
                    "prev_actions": actions,
                    "obs": obs,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                if self.clip_value_loss:
                    value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (values - returns) ** 2
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros_like(a_loss)
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                )

                self.optimizer.zero_grad()

                if torch.isnan(loss).any():
                    print(f"Warning: NaN detected in loss before backward. Loss components:")
                    print(f"  a_loss: {a_loss.item()}")
                    print(f"  c_loss: {c_loss.item()}")
                    print(f"  entropy: {entropy.item()}")
                    print(f"  b_loss: {b_loss.item()}")
                    # Skip this update
                    continue

                loss.backward(retain_graph=True)

                # Check for NaN in gradients
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        print(f"Warning: NaN/inf detected in gradients for {name}")
                        has_nan_grad = True

                if has_nan_grad:
                    print("Skipping optimizer step due to NaN gradients")
                    self.optimizer.zero_grad()
                    continue

                if self.multi_gpu:
                    # batch all_reduce ops https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                            )
                            offset += param.numel()

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            kls.append(av_kls)

            if self.lr_schedule == "kl":
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == "cos":
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)

            if self.multi_gpu:
                lr_tensor = torch.tensor([self.last_lr], device=self.device)
                dist.broadcast(lr_tensor, 0)
                lr = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.last_lr

        self.rl_train_time += time.time() - _t
        return a_losses, c_losses, b_losses, entropies, kls

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data("obses", n, self.obs["obs"])
            for k in ["actions", "neglogpacs", "values", "mus", "sigmas"]:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = res_dict["actions"]
            actions = actions.squeeze(0)
            self.obs, rewards, self.dones, timeout, infos = self.env.step(actions)
            if not self.env.headless:
                self.env.render()

            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                print(f"Warning: NaN/inf detected in rewards. Shape: {rewards.shape}")
                rewards = torch.nan_to_num(rewards, nan=0.0, posinf=100.0, neginf=-100.0)

            self.storage.update_data("dones", n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()

            shaped_rewards = torch.clamp(shaped_rewards, min=-1000.0, max=1000.0)

            if self.value_bootstrap and (timeout is not None):
                shaped_rewards += self.gamma * res_dict["values"].squeeze(0) * timeout
            self.storage.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = done_indices[:, 0]
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            not_dones = 1.0 - self.dones.float()
            not_dones_tp = not_dones.unsqueeze(0).transpose(0, 1)
            self.current_rewards = self.current_rewards * not_dones_tp
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict["values"]

        self.agent_steps = (
            (self.agent_steps + self.batch_size)
            if not self.multi_gpu
            else self.agent_steps + self.batch_size * self.rank_size
        )
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict["returns"]
        values = self.storage.data_dict["values"]
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict["values"] = values
        self.storage.data_dict["returns"] = returns

    def adjust_learning_rate_cos(self, epoch):
        lr = (
            self.init_lr
            * 0.5
            * (1.0 + math.cos(math.pi * (self.agent_steps + epoch / self.mini_epochs_num) / self.max_agent_steps))
        )
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


class AdaptiveScheduler(object):
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


class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
