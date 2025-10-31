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
import torch.nn as nn
from loguru import logger as log
from rich.logging import RichHandler
from torch import Tensor
from torch.distributions import (
    Independent,
    OneHotCategoricalStraightThrough,
    kl_divergence,
)
from torchmetrics import MeanMetric

from roboverse_learn.dexbench_rvrl.algos.dm3.module import (
    Actor,
    ContinueModel,
    Critic,
    Decoder,
    Encoder,
    Moments,
    MSEDistribution,
    RecurrentModel,
    RepresentationModel,
    RewardPredictor,
    SafeBernoulli,
    SymlogDist,
    TransitionModel,
    TwoHotEncodingDistribution,
)
from roboverse_learn.dexbench_rvrl.algos.dm3.storage import ReplayBuffer, ReplayBuffer_Pytorch
from roboverse_learn.dexbench_rvrl.utils.metrics import MetricAggregator
from roboverse_learn.dexbench_rvrl.utils.reproducibility import enable_deterministic_run
from roboverse_learn.dexbench_rvrl.utils.timer import timer
from roboverse_learn.dexbench_rvrl.utils.utils import Ratio

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


########################################################
## Standalone utils
########################################################
class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards):
        if isinstance(rewards, torch.Tensor):
            self.deque.extend(rewards.cpu().numpy().tolist())
        elif isinstance(rewards, float):
            self.deque.append(rewards)

    @property
    def len(self) -> int:
        return len(self.deque)

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


def compute_lambda_values(
    rewards: Tensor, values: Tensor, continues: Tensor, horizon: int, gae_lambda: float
) -> Tensor:
    """
    Compute lambda returns (λ-returns) for Generalized Advantage Estimation (GAE).

    The lambda return is computed recursively as:
    R_t^λ = r_t + γ * [(1 - λ) * V(s_{t+1}) + λ * R_{t+1}^λ]

    Args:
        rewards: (batch_size, time_step) - r_t is the immediate reward received after taking action at time t
        values: (batch_size, time_step) - V(s_t) is the value estimate of the state s_t
        continues: (batch_size, time_step) - c_t is the continue flag after taking action at time t. It is already multiplied by gamma (γ).
        horizon: int - T is the length of the planning horizon
        gae_lambda: float - lambda parameter for GAE (λ, typically 0.95)

    Returns:
        Tensor: (batch_size, horizon-1) - R_t^λ is the lambda return at time t = 0, ..., T-2.
    """
    # Given the following diagram, with horizon=4
    # Actions:            a'0      a'1      a'2
    #                     ^ \      ^ \      ^ \
    #                    /   \    /   \    /   \
    #                   /     \  /     \  /     \
    # States:         z0  ->  z'1  ->  z'2  ->  z'3
    # Values:         v'0    [v'1]    [v'2]    [v'3]      <-- input
    # Rewards:       [r'0]   [r'1]    [r'2]     r'3       <-- input
    # Continues:     [c'0]   [c'1]    [c'2]     c'3       <-- input
    # Lambda-values: [l'0]   [l'1]    [l'2]     l'3       <-- output

    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]

    # Compute the base term: r_t + γ * (1 - λ) * V(s_{t+1})
    inputs = rewards + continues * next_values * (1 - gae_lambda)

    # Compute lambda returns backward in time
    outputs = torch.zeros_like(values)
    outputs[:, -1] = next_values[:, -1]  # initialize with the last value
    for t in range(horizon - 2, -1, -1):  # t = T-2, ..., 0
        # R_t^λ = [r_t + γ * (1 - λ) * V(s_{t+1})] + γ * λ * R_{t+1}^λ
        outputs[:, t] = inputs[:, t] + continues[:, t] * gae_lambda * outputs[:, t + 1]

    return outputs[:, :-1]


class DreamerV3:
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
        self.cast_device = self.device.type

        self.action_dim = np.prod(env.single_action_space.shape)
        self.obs_shape = {k: v.shape for k, v in env.single_observation_space.spaces.items()}
        self.obs_type = getattr(env, "obs_type", "state")
        self.img_h = getattr(env, "img_h", None)
        self.img_w = getattr(env, "img_w", None)
        self.env = env
        self.num_envs = env.num_envs

        # learn cfg
        self.train_cfg = copy.deepcopy(train_cfg)
        if train_cfg.get("deterministic", False):
            enable_deterministic_run()
        learn_cfg = self.train_cfg["learn"]

        # training
        self.batch_size = learn_cfg.get("batch_size", 16)
        self.batch_length = learn_cfg.get("batch_length", 64)
        self.horizon = learn_cfg.get("horizon", 15)
        self.max_iterations = learn_cfg.get("max_iterations", 500000)
        self.prefill = learn_cfg.get("prefill", 200)
        self.amp = learn_cfg.get("amp", False)
        self.nstep = learn_cfg.get("nstep", 1)

        # all models
        self.bins = learn_cfg.get("bins", 255)
        self.symlog_input = learn_cfg.get("symlog_input", True)
        self.vmin = learn_cfg.get("vmin", -20)
        self.vmax = learn_cfg.get("vmax", 20)

        # world model
        self.model_lr = learn_cfg.get("model_lr", 1e-4)
        self.model_eps = learn_cfg.get("model_eps", 1e-8)
        self.model_clip = learn_cfg.get("model_clip", 1000.0)
        self.free_nats = learn_cfg.get("free_nats", 1.0)
        self.stochastic_length = learn_cfg.get("stochastic_length", 32)
        self.stochastic_classes = learn_cfg.get("stochastic_classes", 32)
        self.stochastic_size = self.stochastic_length * self.stochastic_classes
        self.deterministic_size = learn_cfg.get("deterministic_size", 512)

        # actor critic
        self.actor_grad = learn_cfg.get("actor_grad", "dynamics")
        self.actor_lr = learn_cfg.get("actor_lr", 8e-5)
        self.actor_eps = learn_cfg.get("actor_eps", 1e-5)
        self.actor_clip = learn_cfg.get("actor_clip", 100.0)
        self.actor_ent_coef = learn_cfg.get("actor_ent_coef", 0.0003)
        self.critic_lr = learn_cfg.get("critic_lr", 8e-5)
        self.critic_eps = learn_cfg.get("critic_eps", 1e-5)
        self.critic_clip = learn_cfg.get("critic_clip", 100.0)
        self.gae_lambda = learn_cfg.get("gae_lambda", 0.95)
        self.gamma = learn_cfg.get("gamma", 0.997)

        self.model_cfg = self.train_cfg.get("model", {})

        self.buffer = ReplayBuffer_Pytorch(
            self.obs_shape,
            self.action_dim,
            device,
            num_envs=self.num_envs,
            capacity=learn_cfg.get("buffer_size", 5000),
        )

        # dm3 components
        self.encoder = Encoder(
            self.model_cfg,
            self.obs_type,
            self.obs_shape,
            self.symlog_input,
            self.img_h,
            self.img_w,
        ).to(device)
        self.decoder = Decoder(
            self.obs_type,
            self.obs_shape,
            self.deterministic_size,
            self.stochastic_size,
            self.model_cfg,
            decode_shapes=self.encoder.conv_shapes,
        ).to(device)
        self.embedded_obs_dim = self.encoder.output_dim
        self.recurrent_model = RecurrentModel(
            self.action_dim, self.deterministic_size, self.stochastic_size, self.model_cfg
        ).to(device)
        self.transition_model = TransitionModel(
            self.deterministic_size,
            self.stochastic_size,
            self.stochastic_length,
            self.stochastic_classes,
            self.model_cfg,
        ).to(device)
        self.reward_predictor = RewardPredictor(
            self.deterministic_size, self.stochastic_size, self.bins, self.model_cfg
        ).to(device)
        self.representation_model = RepresentationModel(
            self.deterministic_size,
            self.stochastic_size,
            self.stochastic_length,
            self.stochastic_classes,
            self.embedded_obs_dim,
            self.model_cfg,
        ).to(device)
        self.continue_model = ContinueModel(self.deterministic_size, self.stochastic_size, self.model_cfg).to(device)
        self.actor = Actor(self.action_dim, self.deterministic_size, self.stochastic_size, self.model_cfg).to(device)
        self.critic = Critic(self.deterministic_size, self.stochastic_size, self.bins, self.model_cfg).to(device)
        self.moments = Moments().to(device)

        self.model_params = chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.recurrent_model.parameters(),
            self.transition_model.parameters(),
            self.representation_model.parameters(),
            self.reward_predictor.parameters(),
            self.continue_model.parameters(),
        )
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=self.model_lr, eps=self.model_eps)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=self.actor_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.critic_eps)
        self.model_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.actor_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.critic_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        ## logging
        self.global_step = 0
        self.prev_global_step = 0
        self.ratio = Ratio(ratio=0.5)
        self.aggregator = MetricAggregator({
            "Loss/reconstruction_loss": MeanMetric(sync_on_compute=False),
            "Loss/reward_loss": MeanMetric(sync_on_compute=False),
            "Loss/continue_loss": MeanMetric(sync_on_compute=False),
            "Loss/kl_loss": MeanMetric(sync_on_compute=False),
            "Loss/model_loss": MeanMetric(sync_on_compute=False),
            "Loss/actor_loss": MeanMetric(sync_on_compute=False),
            "Loss/value_loss": MeanMetric(sync_on_compute=False),
            "Train/kl": MeanMetric(sync_on_compute=False),
            "Train/prior_entropy": MeanMetric(sync_on_compute=False),
            "Train/posterior_entropy": MeanMetric(sync_on_compute=False),
            "Train/actor_entropy": MeanMetric(sync_on_compute=False),
            "Grad_norm/model": MeanMetric(sync_on_compute=False),
            "Grad_norm/actor": MeanMetric(sync_on_compute=False),
            "Grad_norm/critic": MeanMetric(sync_on_compute=False),
        })
        self.episode_rewards = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_lengths = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_rewards_step = RollingMeter(learn_cfg.get("window_size", 1000))
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.print_log = print_log
        self.wandb_run = wandb_run
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.log_interval = learn_cfg.get("log_interval", 1)
        self.print_interval = learn_cfg.get("print_interval", 1)

    def test(self, path):
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.recurrent_model.load_state_dict(state["recurrent_model"])
        self.transition_model.load_state_dict(state["transition_model"])
        self.representation_model.load_state_dict(state["representation_model"])
        self.reward_predictor.load_state_dict(state["reward_predictor"])
        self.continue_model.load_state_dict(state["continue_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        log.info(f"Loaded checkpoint from {path}")

    def load(self, path):
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.recurrent_model.load_state_dict(state["recurrent_model"])
        self.transition_model.load_state_dict(state["transition_model"])
        self.representation_model.load_state_dict(state["representation_model"])
        self.reward_predictor.load_state_dict(state["reward_predictor"])
        self.continue_model.load_state_dict(state["continue_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.moments.load_state_dict(state["moments"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.model_scaler.load_state_dict(state["model_scaler"])
        self.actor_scaler.load_state_dict(state["actor_scaler"])
        self.critic_scaler.load_state_dict(state["critic_scaler"])
        self.ratio.load_state_dict(state["ratio"])
        self.global_step = state["global_step"]
        log.info(f"Loaded checkpoint from {path}")

    def save(self, path):
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "recurrent_model": self.recurrent_model.state_dict(),
            "transition_model": self.transition_model.state_dict(),
            "representation_model": self.representation_model.state_dict(),
            "reward_predictor": self.reward_predictor.state_dict(),
            "continue_model": self.continue_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "moments": self.moments.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "model_scaler": self.model_scaler.state_dict(),
            "actor_scaler": self.actor_scaler.state_dict(),
            "critic_scaler": self.critic_scaler.state_dict(),
            "ratio": self.ratio.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(state, path)
        log.info(f"Saved checkpoint to {path}")

    def run(self):
        if self.is_testing:
            assert self.model_dir is not None, "model_dir must be specified in testing mode"
            self.test(self.model_dir)
        elif self.model_dir is not None:
            self.load(self.model_dir)
        reset_obs = self.env.reset()
        obs = {k: reset_obs[k].clone() for k in reset_obs.keys()}
        self.start_time = time.time()
        if not self.is_testing:
            cur_rewards_sum = torch.zeros(self.num_envs, device=self.device)
            cur_episode_length = torch.zeros(self.num_envs, device=self.device)
            posterior = torch.zeros(self.num_envs, self.stochastic_size, device=self.device)
            deterministic = torch.zeros(self.num_envs, self.deterministic_size, device=self.device)
            action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
            ep_infos = []

            for iteration in range(self.current_learning_iteration, self.max_iterations):
                ## Step the environment and add to buffer
                for _step in range(self.nstep):
                    with torch.inference_mode(), timer("time/iteration"):
                        embeded_obs = self.encoder(obs)
                        deterministic = self.recurrent_model(posterior, action, deterministic)
                        posterior_dist, _ = self.representation_model(
                            embeded_obs.view(self.num_envs, -1), deterministic
                        )
                        posterior = posterior_dist.sample().view(-1, self.stochastic_size)
                        if self.global_step < self.prefill:
                            action = torch.rand((self.num_envs, self.action_dim), device=self.device) * 2 - 1
                        else:
                            action = self.actor(posterior, deterministic).sample()
                        next_obs, reward, terminated, truncated, info = self.env.step(action)
                        done = torch.logical_or(terminated, truncated)
                        env_step = self.env.episode_lengths
                        self.buffer.add(obs, action, reward, next_obs, done, terminated, env_step)
                        for key in obs.keys():
                            obs[key].copy_(next_obs[key])

                        ep_infos.append(info)
                        cur_rewards_sum += reward
                        cur_episode_length += 1
                        self.global_step += self.num_envs
                        self.episode_rewards_step.update(reward)

                        if done.any():
                            self.episode_rewards.update(cur_rewards_sum[done])
                            self.episode_lengths.update(cur_episode_length[done])
                            cur_rewards_sum[done] = 0
                            cur_episode_length[done] = 0
                            posterior[done] = 0
                            deterministic[done] = 0
                            action[done] = 0

                mean_reward = self.episode_rewards_step.mean if self.episode_rewards_step.len > 0 else 0
                ## Update the model
                if self.global_step > self.prefill:
                    with timer("time/train"):
                        # gradient_steps = self.ratio(self.global_step - self.prefill)

                        # for _ in range(gradient_steps):
                        with timer("time/data_sample"):
                            data = self.buffer.sample(self.batch_size, self.batch_length)
                        with timer("time/dynamic_learning"):
                                posteriors, deterministics = self.dynamic_learning(data, save_recon=True)
                        with timer("time/behavior_learning"):
                            self.behavior_learning(posteriors, deterministics)

                ## log
                if (iteration + 1) % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    log_dir = os.path.join(self.log_dir, f"model_{iteration + 1}.pth")
                    self.save(log_dir)
                if (iteration + 1) % self.print_interval == 0 and self.print_log:
                    metrics_dict = self.aggregator.compute()
                    self.log(locals())
                    ep_infos.clear()
                    self.aggregator.reset()

    def dynamic_learning(self, data: dict[str, Tensor], save_recon=False) -> tuple[Tensor, Tensor]:
        # TODO: utilize "next_observation" to update the model
        # TODO: since the replay buffer may contain termination/truncation in the middle of a rollout, we need to handle this case by resetting posterior, deterministic, and action to initial state (zero)

        # Given the following diagram, with batch_length=4
        # Actions:           [a'0]    [a'1]    [a'2]    a'3  <-- input
        #                       \        \        \
        #                        \        \        \
        #                         \        \        \
        # States:          0  ->  z'1  ->  z'2  ->  z'3      <-- output
        # Observations:   o'0    [o'1]    [o'2]    [o'3]     <-- input
        # Rewards:                r'1      r'2      r'3      <-- output
        # Continues:              c'1      c'2      c'3      <-- output
        with torch.autocast(self.cast_device, enabled=self.amp):
            posterior = torch.zeros(self.batch_size, self.stochastic_size, device=self.device)
            deterministic = torch.zeros(self.batch_size, self.deterministic_size, device=self.device)
            flatten_obs = {k: v.flatten(0, 1) for k, v in data["observation"].items()}
            embeded_obs = self.encoder(flatten_obs).unflatten(0, (self.batch_size, self.batch_length))
            is_first = data["if_first"]
            is_first[:, 0] = True  # the first step of each trajectory must be True

            deterministics = []
            priors_logits = []
            posteriors = []
            posteriors_logits = []

            for t in range(0, self.batch_length):
                if t == 0:
                    action = torch.zeros((self.batch_size, self.action_dim), device=self.device)
                else:
                    action = data["action"][:, t - 1] * (~is_first[:, t])
                deterministic = deterministic * (~is_first[:, t])
                posterior = posterior * (~is_first[:, t])
                deterministic = self.recurrent_model(posterior, action, deterministic)
                prior_dist, prior_logits = self.transition_model(deterministic)
                posterior_dist, posterior_logits = self.representation_model(embeded_obs[:, t], deterministic)
                posterior = posterior_dist.rsample().view(-1, self.stochastic_size)

                deterministics.append(deterministic)
                priors_logits.append(prior_logits)
                posteriors.append(posterior)
                posteriors_logits.append(posterior_logits)

            deterministics = torch.stack(deterministics, dim=1).to(self.device)
            prior_logits = torch.stack(priors_logits, dim=1).to(self.device)
            posteriors = torch.stack(posteriors, dim=1).to(self.device)
            posteriors_logits = torch.stack(posteriors_logits, dim=1).to(self.device)

            reconstructed_obs = self.decoder(posteriors, deterministics)
            recon_loss = 0.0
            for key in self.obs_shape.keys():
                recon_obs = reconstructed_obs[key]
                true_obs = data["observation"][key].view(self.batch_size, self.batch_length, *self.obs_shape[key])
                if key == "rgb" and save_recon:
                    import cv2
                    import numpy as np

                    img = recon_obs[0, 0]
                    img0 = img.permute(1, 2, 0).cpu().detach().numpy()  # Get the first environment's camera image
                    img0_uint8 = (img0 * 255).astype(np.uint8)
                    img0_bgr = cv2.cvtColor(img0_uint8, cv2.COLOR_RGB2BGR)
                    img1 = true_obs[0, 0]
                    img1 = img1.permute(1, 2, 0).cpu().detach().numpy()  # Get the first environment's camera image
                    img1_uint8 = (img1 * 255).astype(np.uint8)
                    img1_bgr = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2BGR)
                    img_bgr = np.concatenate([img1_bgr, img0_bgr], axis=1)
                    cv2.imwrite("red_recon_img.png", img_bgr)
                recon_obs_dist = (
                    MSEDistribution(recon_obs, len(self.obs_shape[key])) if "rgb" in key else SymlogDist(recon_obs)
                )
                recon_loss += -recon_obs_dist.log_prob(true_obs).mean()
            reconstructed_obs_loss = recon_loss / len(self.obs_shape)

            predicted_reward_bins = self.reward_predictor(posteriors, deterministics)
            predicted_reward_dist = TwoHotEncodingDistribution(predicted_reward_bins, dims=1, low=self.vmin, high=self.vmax)
            reward_loss = -predicted_reward_dist.log_prob(data["reward"][:, :]).mean()

            predicted_continue = self.continue_model(posteriors, deterministics)
            predicted_continue_dist = SafeBernoulli(logits=predicted_continue)
            true_continue = 1 - data["terminated"][:, :]
            continue_loss = -predicted_continue_dist.log_prob(true_continue).mean()

            # KL balancing, Eq. 3 in the paper
            kl = dyn_loss = kl_divergence(
                Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
                Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1),
            )
            dyn_loss = torch.max(dyn_loss, torch.tensor(self.free_nats, device=self.device))
            rep_loss = kl_divergence(
                Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
                Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1),
            )
            rep_loss = torch.max(rep_loss, torch.tensor(self.free_nats, device=self.device))
            kl_loss = (0.5 * dyn_loss + 0.1 * rep_loss).mean()

            model_loss = reconstructed_obs_loss + reward_loss + continue_loss + kl_loss

        self.model_optimizer.zero_grad()
        self.model_scaler.scale(model_loss).backward()
        self.model_scaler.unscale_(self.model_optimizer)
        self.model_grad_norm = nn.utils.clip_grad_norm_(self.model_params, self.model_clip)
        self.model_scaler.step(self.model_optimizer)
        self.model_scaler.update()

        with torch.no_grad():
            self.aggregator.update("loss/reconstruction_loss", reconstructed_obs_loss.item())
            self.aggregator.update("loss/reward_loss", reward_loss.item())
            self.aggregator.update("loss/continue_loss", continue_loss.item())
            self.aggregator.update("loss/kl_loss", kl_loss.item())
            self.aggregator.update("loss/model_loss", model_loss.item())
            self.aggregator.update("state/kl", kl.mean().item())
            self.aggregator.update("state/prior_entropy", prior_dist.entropy().mean().item())
            self.aggregator.update("state/posterior_entropy", posterior_dist.entropy().mean().item())
            self.aggregator.update("grad_norm/model", self.model_grad_norm.mean().item())

        return posteriors, deterministics

    def behavior_learning(self, posteriors_: Tensor, deterministics_: Tensor):
        ## reuse the `posteriors` and `deterministics` from model learning, important to detach them!
        state = posteriors_.detach().view(-1, self.stochastic_size)
        deterministic = deterministics_.detach().view(-1, self.deterministic_size)

        # Given the following diagram, with horizon=4
        # Actions:            a'0      a'1      a'2       a'3
        #                    ^  \     ^  \     ^  \      ^  \
        #                   /    \   /    \   /    \    /    \
        #                  /      \ /      \ /      \  /      \
        # States:        z'0  ->  z'1  ->  z'2  ->  z'3  ->  z'4    <-- input is z'0, output is z'1~z'4
        # Rewards:                r'1      r'2      r'3      r'4    <-- output
        # Continues:              c'1      c'2      c'3      c'4    <-- output
        # Values:                 v'1      v'2      v'3      v'4    <-- output
        # Lambda-values:          l'1      l'2      l'3             <-- output

        with torch.autocast(self.cast_device, enabled=self.amp):
            actions = []
            # states = [state]
            # deterministics = [deterministic]
            states = []
            deterministics = []
            for t in range(self.horizon):
                action = self.actor(state.detach(), deterministic.detach()).rsample()  # detach help speed up about 10%
                deterministic = self.recurrent_model(state, action, deterministic)
                state_dist, _ = self.transition_model(deterministic)
                state = state_dist.rsample().view(-1, self.stochastic_size)
                actions.append(action)
                states.append(state)
                deterministics.append(deterministic)

            actions = torch.stack(actions, dim=1)
            states = torch.stack(states, dim=1)
            deterministics = torch.stack(deterministics, dim=1)

            predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(states, deterministics), dims=1, low=self.vmin, high=self.vmax).mean
            predicted_values = TwoHotEncodingDistribution(self.critic(states, deterministics), dims=1, low=self.vmin, high=self.vmax).mean

            continues_logits = self.continue_model(states, deterministics)
            continues = SafeBernoulli(logits=continues_logits).mode
            lambda_values = compute_lambda_values(
                predicted_rewards, predicted_values, continues * self.gamma, self.horizon, self.gae_lambda
            )

            ## Normalize return, Eq. 7 in the paper
            baselines = predicted_values[:, :-1]
            offset, invscale = self.moments(lambda_values)
            normalized_lambda_values = (lambda_values - offset) / invscale
            normalized_baselines = (baselines - offset) / invscale

            advantages = normalized_lambda_values - normalized_baselines

            # TODO: what would happen if we don't use discount factor?
            with torch.no_grad():
                discount = torch.cumprod(continues[:, :-1] * self.gamma, dim=1) / self.gamma

            actor_dist = self.actor(states[:, :-1], deterministics[:, :-1])
            actor_entropy = actor_dist.entropy().unsqueeze(-1)
            if self.actor_grad == "dynamics":
                # Below directly computes the gradient through dynamics.
                actor_target = advantages
            elif self.actor_grad == "reinforce":
                actor_target = advantages.detach() * actor_dist.log_prob(actions[:, 1:]).unsqueeze(-1)
            # For discount factor, see https://ai.stackexchange.com/q/7680
            actor_loss = -((actor_target + self.actor_ent_coef * actor_entropy) * discount).mean()
        self.actor_optimizer.zero_grad()
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.unscale_(self.actor_optimizer)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip)
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()

        # TODO: implement target critic
        with torch.autocast(self.cast_device, enabled=self.amp):
            predicted_value_bins = self.critic(states[:, :-1].detach(), deterministics[:, :-1].detach())
            predicted_value_dist = TwoHotEncodingDistribution(predicted_value_bins, dims=1, low=self.vmin, high=self.vmax)
            value_loss = -predicted_value_dist.log_prob(lambda_values.detach())
            value_loss = (value_loss * discount.squeeze(-1)).mean()
        self.critic_optimizer.zero_grad()
        self.critic_scaler.scale(value_loss).backward()
        self.critic_scaler.unscale_(self.critic_optimizer)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip)
        self.critic_scaler.step(self.critic_optimizer)
        self.critic_scaler.update()

        with torch.no_grad():
            self.aggregator.update("loss/actor_loss", actor_loss.item())
            self.aggregator.update("loss/value_loss", value_loss.item())
            self.aggregator.update("state/actor_entropy", actor_entropy.mean().item())
            self.aggregator.update("grad_norm/actor", actor_grad_norm.mean().item())
            self.aggregator.update("grad_norm/critic", critic_grad_norm.mean().item())

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
            if "time/train" not in time_:
                train_time = 0.0
            else:
                train_time = time_["time/train"]
            data_sample_time = time_["time/data_sample"] if "time/data_sample" in time_ else 0.0
            dyn_time = time_["time/dynamic_learning"] if "time/dynamic_learning" in time_ else 0.0
            behavior_time = time_["time/behavior_learning"] if "time/behavior_learning" in time_ else 0.0
            iteration_time = time_["time/iteration"]
            time_str = f"""{"Computation:":>{pad}} {fps:.0f} steps/s (iteration {iteration_time:.3f}s, learning {train_time:.3f}s)\n"""
            log_data["Train/fps"] = fps
            log_data["Time/train"] = train_time
            log_data["Time/iteration"] = iteration_time
            log_data["Time/data_sample"] = data_sample_time
            log_data["Time/dynamic_learning"] = dyn_time
            log_data["Time/behavior_learning"] = behavior_time
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
        log_data["Train/mean_reward_per_step"] = locs["mean_reward"]

        metrics_dict = locs["metrics_dict"]
        for key, value in metrics_dict.items():
            log_data[key] = value
            if "loss" in key:
                loss_name = key.split("/")[-1].replace("_", " ").title()
                log_string += f"""{f"Mean {loss_name}:":>{pad}} {value:.3f}\n"""

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
