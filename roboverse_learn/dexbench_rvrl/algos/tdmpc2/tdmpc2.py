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
from rich.logging import RichHandler
from tensordict import TensorDict
from torch import Tensor
from torchmetrics import MeanMetric

import roboverse_learn.dexbench_rvrl.algos.tdmpc2.math as math
from roboverse_learn.dexbench_rvrl.algos.tdmpc2.module import WorldModel, api_model_conversion
from roboverse_learn.dexbench_rvrl.algos.tdmpc2.scale import RunningScale
from roboverse_learn.dexbench_rvrl.algos.tdmpc2.storage import Buffer
from roboverse_learn.dexbench_rvrl.algos.tdmpc2.env_wrapper import Tdmpc2DexEnv
from roboverse_learn.dexbench_rvrl.utils.metrics import MetricAggregator
from roboverse_learn.dexbench_rvrl.utils.reproducibility import enable_deterministic_run
from roboverse_learn.dexbench_rvrl.utils.timer import timer


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


class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    currently only support single-task in DexBench.
    and supports both state and pixel observations.
    Reference: https://github.com/nicklashansen/tdmpc2
    """

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
        ori_obs_shape = {k: v.shape for k, v in env.single_observation_space.spaces.items()}
        self.obs_type = getattr(env, "obs_type", "state")
        self.img_h = getattr(env, "img_h", None)
        self.img_w = getattr(env, "img_w", None)
        self.num_envs = env.num_envs
        self.max_episode_length = env.max_episode_steps

        # learn cfg
        self.train_cfg = copy.deepcopy(train_cfg)
        if train_cfg.get("deterministic", False):
            enable_deterministic_run()
        learn_cfg = self.train_cfg["learn"]
        self.env = Tdmpc2DexEnv(
            obs_shape=ori_obs_shape,
            num_frames=learn_cfg.get("num_frames", 3),
            env=env,
            device=self.device,
        )
        self.obs_shape = {k: v.shape for k, v in self.env.single_observation_space.spaces.items()}

        # params
        self.lr = learn_cfg.get("lr", 3e-4)
        self.enc_lr_scale = learn_cfg.get("enc_lr_scale", 0.3)
        self.grad_clip_norm = learn_cfg.get("grad_clip_norm", 20)
        self.episodic = learn_cfg.get("episodic", True)
        self.mpc = learn_cfg.get("mpc", True)
        self.multitask = learn_cfg.get("multitask", False)
        self.tasks = learn_cfg.get("tasks", None)

        self.num_samples = learn_cfg.get("num_samples", 512)
        self.horizon = learn_cfg.get("horizon", 3)
        self.num_pi_trajs = learn_cfg.get("num_pi_trajs", 24)
        self.num_elites = learn_cfg.get("num_elites", 64)
        self.batch_size = learn_cfg.get("batch_size", 256)

        self.tau = learn_cfg.get("tau", 0.01)
        self.discount_max = learn_cfg.get("discount_max", 0.995)
        self.discount_min = learn_cfg.get("discount_min", 0.95)
        self.rho = learn_cfg.get("rho", 0.5)
        self.entropy_coef = learn_cfg.get("entropy_coef", 1.0e-4)
        self.max_std = learn_cfg.get("max_std", 1.0)
        self.min_std = learn_cfg.get("min_std", 0.01)
        self.temperature = learn_cfg.get("temperature", 0.5)
        self.discount_denom = learn_cfg.get("discount_denom", 5)

        self.consistency_coef = learn_cfg.get("consistency_coef", 20.0)
        self.reward_coef = learn_cfg.get("reward_coef", 0.1)
        self.termination_coef = learn_cfg.get("termination_coef", 1.0)
        self.value_coef = learn_cfg.get("value_coef", 0.1)
        self.recon_coef = learn_cfg.get("recon_coef", 0.0)

        self.max_iterations = learn_cfg.get("max_iterations", 500000)
        self.nstep = learn_cfg.get("nstep", 1)
        self.nupdate = learn_cfg.get("nupdate", 1)
        self.prefill = learn_cfg.get("prefill", 400)
        self.mpc_iterations = learn_cfg.get("mpc_iterations", 6)

        self.model_cfg = self.train_cfg.get("model", {})
        self.num_bins = self.model_cfg.get("num_bins", 51)
        self.vmin = self.model_cfg.get("vmin", -10)
        self.vmax = self.model_cfg.get("vmax", 10)
        self.reward_vmin = self.model_cfg.get("reward_vmin", -1)
        self.reward_vmax = self.model_cfg.get("reward_vmax", 1)
        self.bin_size = self.model_cfg.get("bin_size", (self.vmax - self.vmin) / (self.num_bins - 1))
        self.reward_bin_size = self.model_cfg.get(
            "reward_bin_size", (self.reward_vmax - self.reward_vmin) / (self.num_bins - 1)
        )
        self.latent_dim = self.model_cfg.get("latent_dim", 512)
        self.num_q = self.model_cfg.get("num_q", 5)
        self.decode = self.model_cfg.get("decode", False)

        self.buffer = Buffer(
            self.obs_shape,
            self.action_dim,
            self.model_cfg["task_dim"],
            self.device,
            self.num_envs,
            learn_cfg.get("buffer_size", 5000),
            batch_size=self.batch_size,
            horizon=self.horizon,
            max_length=self.max_episode_length,
        )

        # world model
        self.model = WorldModel(
            obs_shape=self.obs_shape,
            model_cfg=self.model_cfg,
            tau=self.tau,
            episodic=self.episodic,
            multitask=self.multitask,
            tasks=self.tasks,
            action_dims=self.action_dim,
            img_h=self.img_h,
            img_w=self.img_w,
            device=self.device,
        ).to(self.device)

        self.optim = torch.optim.Adam(
            [
                {"params": self.model._encoder.parameters(), "lr": self.lr * self.enc_lr_scale},
                {"params": self.model._decoder.parameters() if self.decode else [], "lr": self.lr * self.enc_lr_scale},
                # {"params": self.model._linear.parameters()},
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._termination.parameters() if self.episodic else []},
                {"params": self.model._Qs.parameters()},
                {"params": self.model._task_emb.parameters() if self.multitask else []},
            ],
            lr=self.lr,
            capturable=True,
        )
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(self.tau, self.device)
        self.mpc_iterations += 2 * int(self.action_dim >= 20)  # Heuristic for large action spaces
        self.discount = learn_cfg.get("discount", 0.99)
        print("Episode length:", self.max_episode_length)
        print("Discount factor:", self.discount)
        self._prev_mean = torch.zeros(self.num_envs, self.horizon, self.action_dim, device=self.device)

        self.global_step = 0
        self.prev_global_step = 0
        self.aggregator = MetricAggregator({
            "Loss/reconstruction_loss": MeanMetric(sync_on_compute=False),
            "Train/kl": MeanMetric(sync_on_compute=False),
            "Grad_norm/model": MeanMetric(sync_on_compute=False),
            "Grad_norm/actor": MeanMetric(sync_on_compute=False),
            "Grad_norm/critic": MeanMetric(sync_on_compute=False),
        })
        self.episode_rewards = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_lengths = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_rewards_step = RollingMeter(learn_cfg.get("window_size", 100))
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.save_image_path = os.path.join(self.log_dir, "tdmpc2_image.png")
        self.print_log = print_log
        self.wandb_run = wandb_run
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.log_interval = learn_cfg.get("log_interval", 1)
        self.print_interval = learn_cfg.get("print_interval", 1)

    @property
    def plan(self):
        return self._plan

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
                episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
                float: Discount factor for the task.
        """
        frac = episode_length / self.discount_denom
        return min(max((frac - 1) / (frac), self.discount_min), self.discount_max)

    def load(self, path):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
                fp (str or dict): Filepath or state dict to load.
        """
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        state_dict = torch.load(path, map_location=self.device)
        self.global_step = state_dict.get("global_step", 0)
        self.model.load_state_dict(state_dict["model"])
        log.info(f"Loaded model at step {self.global_step} from {path}")

    def save(self, path):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save(
            {
                "global_step": self.global_step,
                "model": self.model.state_dict(),
            },
            path,
        )
        log.info(f"Saved model at step {self.global_step} to {path}")

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
            is_first = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            cur_rewards_sum = torch.zeros(self.num_envs, device=self.device)
            cur_episode_length = torch.zeros(self.num_envs, device=self.device)
            ep_infos = []

            for iteration in range(self.current_learning_iteration, self.max_iterations):
                ## Step the environment and add to buffer
                for _step in range(self.nstep):
                    with torch.inference_mode(), timer("time/iteration"):
                        if self.global_step < self.prefill:
                            action = torch.rand((self.num_envs, self.action_dim), device=self.device) * 2 - 1
                        else:
                            action = self.act(obs, t0=is_first)
                        next_obs, reward, terminated, truncated, info = self.env.step(action, self.save_image_path)
                        done = torch.logical_or(terminated, truncated)
                        env_step = self.env.env.episode_lengths
                        is_first = (env_step == 0)
                        self.buffer.add(obs, self.env.tensor_obs, action, reward, done, terminated, None)
                        for key in obs.keys():
                            obs[key].copy_(next_obs[key])

                        ep_infos.append(info)
                        cur_rewards_sum += reward
                        cur_episode_length += 1
                        self.global_step += self.num_envs
                        self.episode_rewards_step.update(reward.mean().item())

                        if done.any():
                            self.episode_rewards.update(cur_rewards_sum[done])
                            self.episode_lengths.update(cur_episode_length[done])
                            cur_rewards_sum[done] = 0
                            cur_episode_length[done] = 0
                            action[done] = 0

                mean_reward = self.episode_rewards_step.mean
                ## Update the model
                if self.global_step > self.prefill:
                    with timer("time/train"):
                        # gradient_steps = self.ratio(self.global_step - self.prefill)

                        for _ in range(self.nupdate):
                            with timer("time/data_sample"):
                                data = self.buffer.sample(self.horizon)
                            with timer("time/learning"):
                                self.update(data)

                ## log
                if (iteration + 1) % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    log_dir = os.path.join(self.log_dir, f"model_{iteration + 1}.pth")
                    self.save(log_dir)
                if (iteration + 1) % self.print_interval == 0 and self.print_log:
                    metrics_dict = self.aggregator.compute()
                    self.log(locals())
                    ep_infos.clear()
                    self.aggregator.reset()

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        for key in obs.keys():
            obs[key] = obs[key].to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z, _ = self.model.encode(obs, task)
        if self.mpc:
            return self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            action, info = self.model.pi(z, task)
        if eval_mode:
            action = info["mean"]
        return action

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G = torch.zeros(self.num_envs, self.num_samples, 1, device=z.device)
        discount = torch.ones(self.num_envs, self.num_samples, 1, device=z.device)
        termination = torch.zeros(self.num_envs, self.num_samples, 1, dtype=torch.float32, device=z.device)
        for t in range(self.horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, actions[:, t], task), self.num_bins, self.reward_vmin, self.reward_vmax
            ) # (num_envs, num_samples, 1)
            z = self.model.next(z, actions[:, t], task)
            G = G + discount * (1 - termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.multitask else self.discount
            discount = discount * discount_update
            if self.episodic:
                termination = torch.clip(termination + self.model.termination(z, task), max=1.0)
        action, _ = self.model.pi(z, task)
        return G + discount * (1 - termination) * self.model.Q(z, action, task, return_type="avg")

    @torch.no_grad()
    def _plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
                z (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        if self.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.num_envs, self.horizon, self.num_pi_trajs, self.action_dim, device=self.device
            )
            _z = z.unsqueeze(1).repeat(1, self.num_pi_trajs, 1).view(self.num_envs * self.num_pi_trajs, -1)
            for t in range(self.horizon - 1):
                a, _ = self.model.pi(_z, task)
                pi_actions[:, t] = a.view(self.num_envs, self.num_pi_trajs, self.action_dim)
                _z = self.model.next(_z, a, task)
            a, _ = self.model.pi(_z, task)
            pi_actions[:, -1] = a.view(self.num_envs, self.num_pi_trajs, self.action_dim)

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, self.num_samples, 1)
        mean = torch.zeros(self.num_envs, self.horizon, self.action_dim, device=self.device)
        std = torch.full(
            (self.num_envs, self.horizon, self.action_dim), self.max_std, dtype=torch.float, device=self.device
        )
        mask = (t0 == 0)
        mean[mask, :-1] = self._prev_mean[mask, 1:]
        actions = torch.empty(self.num_envs, self.horizon, self.num_samples, self.action_dim, device=self.device)
        if self.num_pi_trajs > 0:
            actions[:, :, : self.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.mpc_iterations):
            # Sample actions
            r = torch.randn(
                self.num_envs, self.horizon, self.num_samples - self.num_pi_trajs, self.action_dim, device=std.device
            )
            actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, self.num_pi_trajs :] = actions_sample
            if self.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task)
            elite_idxs = torch.topk(value.squeeze(2), self.num_elites, dim=1).indices
            elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
            elite_actions = actions.gather(
                dim=2,
                index=elite_idxs[:, None, :, None].expand(-1, self.horizon, self.num_elites, self.action_dim)
            )

            # Update parameters
            score = torch.exp(self.temperature * (elite_value - elite_value.max(1, keepdim=True).values))
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)
            score_exp = score.unsqueeze(1)
            mean = (score_exp * elite_actions).sum(dim=2) / (score_exp.sum(dim=2) + 1e-9)
            std = ((score_exp * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2) /
                (score_exp.sum(dim=2) + 1e-9)).sqrt().clamp(self.min_std, self.max_std)
            if self.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        logits = torch.log(score.squeeze(2) + 1e-9)
        rand_idx = math.gumbel_softmax_sample(logits, temperature=self.temperature, dim=1)  # (num_envs,)
        selected_actions = elite_actions.gather(
            dim=2,
            index=rand_idx[:, None, None, None].expand(-1, self.horizon, 1, self.action_dim)
        ).squeeze(2)
        a, std = selected_actions[:, 0], std[:, 0]
        if not eval_mode:
            a = a + std * torch.randn_like(a)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        action, info = self.model.pi(zs, task) # (horizon, batch_size, action_dim)
        self.model.track_q_grad(False)
        qs = self.model.Q(zs, action, task, return_type="avg")
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        with torch.no_grad():
            self.aggregator.update("loss/pi_loss", pi_loss.item())
            self.aggregator.update("grad_norm/pi_grad_norm", pi_grad_norm.item())
            self.aggregator.update("train/pi_entropy", info["entropy"].mean().item())
            self.aggregator.update("train/pi_scaled_entropy", info["scaled_entropy"].mean().item())
            self.aggregator.update("train/pi_scale", self.scale.value.item())

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                reward (torch.Tensor): Reward at the current time step.
                terminated (torch.Tensor): Termination signal at the current time step.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: TD-target.
        """
        action, _ = self.model.pi(next_z, task)
        discount = self.discount[task].unsqueeze(-1) if self.multitask else self.discount
        return reward + discount * (1 - terminated) * self.model.Q(next_z, action, task, return_type="min", target=True)

    def _update(self, obs, action, reward, terminated, task=None):
        # Compute targets
        # obs: (horizon + 1, batch_size, obs_dim)
        # action: (horizon, batch_size, action_dim)
        # reward: (horizon, batch_size, 1)
        # terminated: (horizon, batch_size, 1)
        next_obs = {
            k: obs[k][1:] for k in obs.keys()
        }
        if not self.decode:
            with torch.no_grad():
                next_z, _ = self.model.encode(next_obs, task)
                td_targets = self._td_target(next_z, reward, terminated, task)
        else:
            next_z, next_feature = self.model.encode(next_obs, task)
            td_targets = self._td_target(next_z, reward, terminated, task).detach()
            next_z = next_z.detach()

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.horizon + 1, self.batch_size, self.latent_dim, device=self.device)
        first_obs = {
            k: obs[k][0] for k in obs.keys()
        }
        z, _ = self.model.encode(first_obs, task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss += F.mse_loss(z, _next_z, reduction="none").mean() * (self.rho**t)
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all") # (num_q, horizon, batch_size, num_bins)
        qs_value = math.two_hot_inv(qs, self.num_bins, self.vmin, self.vmax)
        reward_preds = self.model.reward(_zs, action, task)
        if self.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
            zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
        ):
            reward_loss = reward_loss + math.soft_ce(
                rew_pred_unbind, rew_unbind, self.num_bins, self.reward_vmin, self.reward_vmax, self.reward_bin_size
            ).mean() * self.rho**t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(
                    qs_unbind_unbind, td_targets_unbind, self.num_bins, self.vmin, self.vmax, self.bin_size
                ).mean() * self.rho**t
                
        if self.decode:
            recon_next_obs = self.model.decode(next_feature)
            reconstruction_loss = 0.0
            for k in recon_next_obs.keys():
                recon_obs = recon_next_obs[k]
                true_obs = next_obs[k]
                if k == "rgb":
                    import cv2
                    import numpy as np
                    
                    img = recon_obs[0,0]
                    img0 = img.permute(1,2,0).cpu().detach().numpy()[:, :, :3]
                    img0_uint8 = (img0 * 255).astype(np.uint8)
                    img0_bgr = cv2.cvtColor(img0_uint8, cv2.COLOR_RGB2BGR)
                    img1 = true_obs[0,0]
                    img1_uint8 = (img1.permute(1,2,0).cpu().detach().numpy()[:, :, :3] * 255).astype(np.uint8)
                    img1_bgr = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2BGR)
                    img_bgr = np.concatenate([img0_bgr, img1_bgr], axis=1)
                    cv2.imwrite("tdmpc2_recon.png", img_bgr)
                loss = F.mse_loss(recon_obs, true_obs, reduction='none')
                loss = loss.sum(dim=tuple(range(2, loss.ndim)))
                reconstruction_loss += loss.mean()
            reconstruction_loss = reconstruction_loss / len(recon_next_obs.keys())

        consistency_loss = consistency_loss / self.horizon
        reward_loss = reward_loss / self.horizon
        if self.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
        else:
            termination_loss = 0.0
        value_loss = value_loss / (self.horizon * self.num_q)
        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.termination_coef * termination_loss
            + self.value_coef * value_loss
        )
        if self.decode:
            total_loss = total_loss + self.recon_coef * reconstruction_loss

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optim.step()

        # Update policy
        self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        with torch.no_grad():
            self.aggregator.update("loss/consistency_loss", consistency_loss.item())
            self.aggregator.update("loss/reward_loss", reward_loss.item())
            if self.episodic:
                self.aggregator.update("loss/termination_loss", termination_loss.item())
            if self.decode:
                self.aggregator.update("loss/reconstruction_loss", reconstruction_loss.item())
            self.aggregator.update("loss/value_loss", value_loss.item())
            self.aggregator.update("loss/total_loss", total_loss.item())
            self.aggregator.update("grad_norm/model_grad_norm", grad_norm.item())
            self.aggregator.update("train/scale", self.scale.value.item())
            self.aggregator.update("train/pred_value", qs_value.mean().item())

    def update(self, data):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        obs = data["observation"]
        action = data["action"]
        reward = data["reward"]
        done = data["done"]
        task = data.get("task", None)
        return self._update(obs, action, reward, done, task)

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
            if "time/train" not in time_:
                train_time = 0.0
            else:
                train_time = time_["time/train"]
            collection_time = time_["time/iteration"]
            iteration_time = collection_time + train_time
            fps = (self.global_step - self.prev_global_step) / iteration_time
            data_sample_time = time_["time/data_sample"] if "time/data_sample" in time_ else 0.0
            learn_time = time_["time/learning"] if "time/learning" in time_ else 0.0
            time_str = f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection {collection_time:.3f}s, learning {train_time:.3f}s)\n"""
            log_data["Train/fps"] = fps
            log_data["Time/train"] = train_time
            log_data["Time/collection"] = collection_time
            log_data["Time/iteration"] = iteration_time
            log_data["Time/data_sample"] = data_sample_time
            log_data["Time/learning"] = learn_time
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
                log_string += f"""{f"Mean {loss_name}:":>{pad}} {value:.4f}\n"""
            if "pred_value" in key:
                log_string += f"""{"Mean Predicted Value:":>{pad}} {value:.4f}\n"""

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
