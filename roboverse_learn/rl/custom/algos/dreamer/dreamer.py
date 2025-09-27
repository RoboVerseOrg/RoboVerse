import os
import time

import numpy as np
import torch
import torch.distributed as dist
import wandb
from tensorboardX import SummaryWriter

from .experience import ExperienceBuffer
from .models import ImagBehavior, WorldModel
from .utils import AverageScalarMeter


class Dreamer:
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

        # ---- Config ----
        self.dreamer_config = full_config["train"]["dreamer"]

        # ---- build environment ----
        self.env = env
        self.num_actors = self.dreamer_config["num_actors"]
        self.actions_num = self.dreamer_config["action_num"]
        self.obs_shape = full_config["train"]["observation_space"]["shape"]
        self.obs_space = self._get_obs_space()

        self.act_space = self._get_act_space()

        # Initialize world model and behavior policy
        self._step = 0
        self._should_log = self._every(self.dreamer_config["log_every"])
        self._should_train = self._every(
            self.dreamer_config["batch_size"] * self.dreamer_config["batch_length"] / self.dreamer_config["train_ratio"]
        )

        self.world_model = WorldModel(
            self.obs_space, self.act_space, self._step, self.dreamer_config, device=self.device
        )
        self.behavior = ImagBehavior(self.dreamer_config, self.world_model, device=self.device)

        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, "nn")
        self.tb_dif = os.path.join(self.output_dir, "tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Replay Buffer ----
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.dreamer_config["horizon_length"],
            self.dreamer_config["batch_size"],
            self.dreamer_config["batch_length"],
            self.obs_space["obs"].shape,
            self.actions_num,
            self.device,
        )

        # ---- Tensorboard Logger ----
        self.writer = SummaryWriter(self.tb_dif)
        self._metrics = {}

        # ---- Training state ----
        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.obs = None
        self.state = None
        self.epoch_num = 0
        self.agent_steps = 0
        self.best_rewards = -10000
        self.max_agent_steps = self.dreamer_config["max_agent_steps"]

        # Current episode tracking
        batch_size = self.num_actors
        self.current_rewards = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)

        # ---- Timing stats ----
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

        # Add prefill parameter from config with a default value
        self.prefill = full_config.get("prefill", 2500)

    def _get_obs_space(self):
        """Create observation space dictionary for world model"""
        from gym import spaces

        # Check what observation the environment actually returns
        sample_obs = self.env.reset()

        # Create spaces based on the actual observation structure
        spaces_dict = {}

        # Assume observation contains "obs" at minimum
        if isinstance(sample_obs, dict) and "obs" in sample_obs:
            obs_shape = sample_obs["obs"].shape[1:]  # Remove batch dimension
            spaces_dict["obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

            # Add RGB space if images are included in observations
            if "rgb" in sample_obs:
                # Get shape from first RGB tensor, removing batch dimension
                rgb_shape = (
                    sample_obs["rgb"][0].shape if isinstance(sample_obs["rgb"], list) else sample_obs["rgb"].shape[1:]
                )
                spaces_dict["obs"] = spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8)
        else:
            # Fallback to using the configured observation shape
            spaces_dict["obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)

        # Add standard dreamer fields
        spaces_dict["is_first"] = spaces.Box(0, 1, (1,), dtype=np.bool_)
        spaces_dict["is_terminal"] = spaces.Box(0, 1, (1,), dtype=np.bool_)
        spaces_dict["action"] = spaces.Box(low=-1.0, high=1.0, shape=(self.actions_num,), dtype=np.float32)

        return spaces.Dict(spaces_dict)

    def _get_act_space(self):
        """Create action space for world model"""
        from gym import spaces

        # Check if the environment has a defined action space
        if hasattr(self.env, "action_space"):
            # Use the environment's action space
            return self.env.action_space
        else:
            # Create action space based on action dimension in config
            return spaces.Box(low=-1.0, high=1.0, shape=(self.actions_num,), dtype=np.float32)

    def _every(self, steps):
        """Create a function that returns true every n steps"""
        counter = 0

        def test():
            nonlocal counter
            if counter >= steps:
                counter = 0
                return True
            counter += 1
            return False

        return test

    def write_stats(self, metrics):
        """Write stats to tensorboard and wandb"""
        log_dict = {}
        for k, v in metrics.items():
            log_dict[k] = v.mean() if hasattr(v, "mean") else v

        # Add performance metrics
        log_dict.update({
            "performance/RLTrainFPS": self.agent_steps / self.rl_train_time if self.rl_train_time > 0 else 0,
            "performance/EnvStepFPS": self.agent_steps / self.data_collect_time if self.data_collect_time > 0 else 0,
        })

        # Log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # Log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def set_eval(self):
        """Set models to eval mode"""
        self.world_model.eval()
        self.behavior.eval()
        if hasattr(self, "running_mean_std"):
            self.running_mean_std.eval()

    def set_train(self):
        """Set models to train mode"""
        self.world_model.train()
        self.behavior.train()
        if hasattr(self, "running_mean_std"):
            self.running_mean_std.train()

    def preprocess_obs(self, obs_dict):
        """Preprocess observations for the world model"""
        # Create the expected input format for world model
        obs_dict["is_first"] = self.dones.unsqueeze(1).float()
        obs_dict["is_terminal"] = torch.zeros_like(self.dones).unsqueeze(1)
        obs_dict["action"] = torch.zeros((self.num_actors, self.actions_num), device=self.device, dtype=torch.float32)
        return obs_dict

    def model_act(self, obs_dict):
        """Get actions from the model"""
        processed_obs = self.preprocess_obs(obs_dict)
        policy_output, self.state = self._policy(processed_obs, self.state, training=True)
        return policy_output

    def _policy(self, obs, state, training=True):
        """Get actions from the policy"""
        embed = self.world_model.encoder(obs)

        # Initialize latent state if needed
        if state is None:
            latent = None
            action = torch.zeros((self.num_actors, self.actions_num), device=self.device, dtype=torch.float32)
        else:
            latent, action = state

        latent, _ = self.world_model.dynamics.obs_step(latent, action, embed, obs["is_first"])

        # Get features for actor
        feat = self.world_model.dynamics.get_feat(latent)

        # Get action from actor
        if not training:
            # Deterministic mode for evaluation
            actor = self.behavior.actor(feat)
            # Fix: Check if mode is a callable method or a tensor property
            if callable(getattr(actor, "mode", None)):
                action = actor.mode()
            else:
                # Fallback if mode is not callable - use mean or other property
                if hasattr(actor, "mean"):
                    action = actor.mean
                elif hasattr(actor, "loc"):
                    action = actor.loc
                else:
                    # Last resort: sample deterministically with zero noise
                    action = actor.sample()
        else:
            # Sampling mode for exploration
            actor = self.behavior.actor(feat)
            action = actor.sample()

        # Handle log prob calculation
        if hasattr(actor, "log_prob"):
            logprob = actor.log_prob(action)
        else:
            # Create a placeholder log probability if not available
            logprob = torch.zeros_like(action).sum(dim=-1)

        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def train(self):
        """Main training loop"""
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.obs = {"obs": self.obs["rgb"]}  # XXX: we don't need joint positions for dreamer observation
        self.agent_steps = self.num_actors * (1 if not self.multi_gpu else self.rank_size)

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            world_model_params = [self.world_model.state_dict()]
            behavior_params = [self.behavior.state_dict()]
            dist.broadcast_object_list(world_model_params, 0)
            dist.broadcast_object_list(behavior_params, 0)
            self.world_model.load_state_dict(world_model_params[0])
            self.behavior.load_state_dict(behavior_params[0])

        # Prefill replay buffer with random experiences
        print(f"Prefilling replay buffer with {self.prefill} transitions...")
        prefill_start_time = time.time()
        prefill_steps = 0

        while prefill_steps < self.prefill:
            # Take random actions during prefill
            action = (
                torch.rand((self.num_actors, self.actions_num), device=self.device) * 2.0 - 1.0
            )  # Scale from [0,1] to [-1,1]

            # Step environment
            next_obs, reward, done, timeout, info = self.env.step(action)

            # Add to replay buffer - use the new add_data method
            self.storage.add_data(self.obs["obs"], action, reward, done)

            # Update current observation
            self.obs = next_obs
            self.obs = {"obs": self.obs["rgb"]}  # XXX: we don't need joint positions for dreamer observation

            # Reset only the done environments
            if torch.any(done):
                # Get indices of done environments
                done_indices = done.nonzero(as_tuple=False).squeeze(-1)

                # Reset only the done environments
                reset_obs = self.env.reset_idx(done_indices)

                # Update only the observations for done environments
                for i, idx in enumerate(done_indices):
                    self.obs["obs"][idx] = reset_obs["rgb"][i]
            prefill_steps += self.num_actors
            self.agent_steps += self.num_actors

        prefill_time = time.time() - prefill_start_time
        print(f"Prefill complete! Collected {prefill_steps} steps in {prefill_time:.2f}s")

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1

            # Collect data and train on it
            metrics = self.train_epoch()

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = (self.num_actors * (1 if not self.multi_gpu else self.rank_size)) / (time.time() - _last_t)
                _last_t = time.time()
                info_string = (
                    f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                    f"Last FPS: {last_fps:.1f} | "
                    f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                    f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                    f"Current Best: {self.best_rewards:.2f}"
                )
                print(info_string)

                # Log metrics
                self.write_stats(metrics)

                # Log episode stats
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

                # Save checkpoints
                checkpoint_name = (
                    f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}"
                )

                if self.dreamer_config["save_frequency"] > 0:
                    if self.epoch_num % self.dreamer_config["save_frequency"] == 0:
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, "last"))

                if mean_rewards > self.best_rewards:
                    print(f"Save current best reward: {mean_rewards:.2f}")
                    # Remove previous best file
                    prev_best_ckpt = os.path.join(self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth")
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}"))

        print("max steps achieved")

    def save(self, name):
        """Save model checkpoint

        Args:
            name: Name/path of the checkpoint file (without extension)
        """
        weights = {
            "world_model": self.world_model.state_dict(),
            "behavior": self.behavior.state_dict(),
        }

        # Save running mean std if applicable
        if hasattr(self, "running_mean_std"):
            weights["running_mean_std"] = self.running_mean_std.state_dict()

        # Save additional info if needed
        weights["config"] = self.dreamer_config
        weights["step"] = self._step  # Fix: use _step instead of step

        torch.save(weights, f"{name}.pth")
        print(f"Saved checkpoint to {name}.pth")

    def restore_train(self, fn):
        """Restore model for training"""
        checkpoint = torch.load(fn + ".pth", map_location=self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.behavior.load_state_dict(checkpoint["behavior"])

    def train_epoch(self):
        """Train for one epoch"""
        # Collect data
        collect_time_start = time.time()
        data = self.collect_data()
        self.data_collect_time += time.time() - collect_time_start

        # Train models
        train_time_start = time.time()
        wm_metrics, behavior_metrics = self.update_models(data)
        self.rl_train_time += time.time() - train_time_start

        # Combine metrics
        metrics = {}
        metrics.update(wm_metrics)
        metrics.update(behavior_metrics)

        # Update step counter
        self.agent_steps += self.num_actors * self.dreamer_config["horizon_length"]

        return metrics

    def collect_data(self):
        """Collect experience data from environment"""
        # Set to evaluation mode when collecting data
        self.set_eval()

        # Initialize episode stats
        done_episodes_rewards = []
        done_episodes_lengths = []

        for t in range(self.dreamer_config["horizon_length"]):
            # Get actions from model
            with torch.no_grad():
                policy_output = self.model_act(self.obs)
                actions = policy_output["action"]

            # Step environment
            next_obs, rewards, dones, timeouts, infos = self.env.step(actions)

            # Store experience
            self.storage.update_data("obses", t, self.obs["obs"])
            self.storage.update_data("actions", t, actions)
            self.storage.update_data("rewards", t, rewards)
            self.storage.update_data("dones", t, dones)

            # Update current episode stats
            self.current_rewards += rewards
            self.current_lengths += 1

            # Handle episode termination
            done_indices = dones.nonzero(as_tuple=False)
            if done_indices.numel() > 0:
                done_indices = done_indices.squeeze(-1)
                done_episodes_rewards.extend(self.current_rewards[done_indices].squeeze(1).cpu().numpy().tolist())
                done_episodes_lengths.extend(self.current_lengths[done_indices].cpu().numpy().tolist())
                self.current_rewards[done_indices] = 0
                self.current_lengths[done_indices] = 0

            # Update observations and dones
            next_obs = {"obs": next_obs["rgb"]}  # XXX: we don't need joint positions for dreamer observation
            self.obs = next_obs
            self.dones = dones

        # Update episode stats
        for reward in done_episodes_rewards:
            self.episode_rewards.update(reward)
        for length in done_episodes_lengths:
            self.episode_lengths.update(length)

        # Prepare data for training
        data = self.storage.prepare_training()

        # Return to training mode
        self.set_train()

        return data

    def update_models(self, data):
        """Update world model and behavior policy"""
        # World model update
        post, context, wm_metrics = self.world_model.train_batch(data)

        # Behavior policy update
        behavior_metrics = self.behavior.train_batch(post)

        return wm_metrics, behavior_metrics

    def eval(self):
        """Evaluate the policy"""
        self.set_eval()
        # Reset environment and state
        self.obs = self.env.reset()
        self.state = None

        total_rewards = []
        episode_lengths = []

        current_rewards = torch.zeros((self.num_actors, 1), device=self.device)
        current_lengths = torch.zeros(self.num_actors, device=self.device)

        # Run evaluation episodes
        for _ in range(self.dreamer_config["eval_episodes"]):
            done = torch.zeros(self.num_actors, device=self.device).bool()

            while not done.all():
                with torch.no_grad():
                    policy_output, self.state = self._policy(self.preprocess_obs(self.obs), self.state, training=False)

                # Step environment using deterministic actions
                next_obs, rewards, dones, timeouts, _ = self.env.step(policy_output["action"])

                # Update episode stats
                current_rewards += rewards
                current_lengths += 1

                # Check for episode termination
                new_dones = (dones | timeouts) & ~done
                if new_dones.any():
                    done_indices = new_dones.nonzero(as_tuple=False).squeeze(-1)
                    total_rewards.extend(current_rewards[done_indices].cpu().numpy().tolist())
                    episode_lengths.extend(current_lengths[done_indices].cpu().numpy().tolist())
                    current_rewards[done_indices] = 0
                    current_lengths[done_indices] = 0

                # Update done flags
                done = done | dones | timeouts

                # Update observations
                self.obs = next_obs

        # Compute average metrics
        mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        mean_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0

        return {"eval/mean_reward": mean_reward, "eval/mean_length": mean_length}

    def evaluate(self):
        """Run evaluation episodes and log metrics and videos"""
        self.set_eval()
        eval_returns = []
        eval_lengths = []
        has_video_logged = False

        # Limit number of eval episodes for efficiency
        num_episodes = min(self.dreamer_config.get("eval_episode_num", 5), self.num_actors)

        # Reset evaluation environments and states
        eval_obs = self.env.reset()
        eval_obs = {"obs": eval_obs["rgb"]}  # Format for dreamer
        eval_dones = torch.zeros(self.num_actors, dtype=torch.bool, device=self.device)
        eval_state = None
        episode_returns = torch.zeros(self.num_actors, device=self.device)
        episode_lengths = torch.zeros(self.num_actors, device=self.device)

        # For video logging
        if self.dreamer_config.get("video_pred_log", True):
            frames_buffer = {i: [] for i in range(num_episodes)}
            imagined_frames_buffer = {i: [] for i in range(num_episodes)}

        # Collect evaluation episodes
        max_eval_steps = self.dreamer_config.get("time_limit", 1000)

        for step in range(max_eval_steps):
            # Process observation and get policy action
            with torch.no_grad():
                policy_output, eval_state = self._policy(eval_obs, eval_state, training=False)
                actions = policy_output["action"]

            # Store frames for video if enabled
            if self.dreamer_config.get("video_pred_log", True) and not has_video_logged:
                for i in range(num_episodes):
                    if not eval_dones[i]:
                        frames_buffer[i].append(eval_obs["obs"][i].detach().cpu().numpy())

                        # Generate imagined trajectory for visualization
                        if eval_state and step % 10 == 0:  # Only predict every 10 steps to save computation
                            imagined_obs = self._generate_imagined_trajectory(eval_state, i, horizon=8)
                            imagined_frames_buffer[i].append(imagined_obs)

            # Step environment
            eval_obs, rewards, new_dones, _ = self.env.step(actions)
            eval_obs = {"obs": eval_obs["rgb"]}  # Format for dreamer

            # Update episode returns and lengths
            episode_returns += rewards * ~eval_dones
            episode_lengths += (~eval_dones).float()

            # Check for episode terminations
            for i in range(self.num_actors):
                if new_dones[i] and not eval_dones[i]:
                    eval_dones[i] = True
                    eval_returns.append(episode_returns[i].item())
                    eval_lengths.append(episode_lengths[i].item())

                    # Reset episode stats for completed episodes
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

            # Stop if we've collected enough episodes
            if sum(eval_dones[:num_episodes].int()).item() >= num_episodes:
                break

        # Log evaluation metrics
        self.writer.add_scalar("eval/mean_return", np.mean(eval_returns), self._step)
        self.writer.add_scalar("eval/std_return", np.std(eval_returns), self._step)
        self.writer.add_scalar("eval/mean_length", np.mean(eval_lengths), self._step)

        # Log video frames if enabled
        if self.dreamer_config.get("video_pred_log", True) and not has_video_logged:
            self._log_videos(frames_buffer, imagined_frames_buffer, num_episodes)

            # Additionally log world model predictions (similar to original Dreamer)
            try:
                # Get a batch from experience buffer
                eval_batch = self.storage.sample_sequence()
                # Generate world model predictions video
                with torch.no_grad():
                    video_pred = self.world_model.video_pred(eval_batch)

                # Convert to numpy and log
                video_pred_np = video_pred.detach().cpu().numpy()

                # Use matplotlib to create a clean video visualization
                import io

                import matplotlib.pyplot as plt
                from PIL import Image

                # Create a figure with good dimensions for the video
                fig = plt.figure(figsize=(12, 9))
                plt.title("World Model Predictions")
                plt.imshow(np.concatenate([f for f in video_pred_np], axis=1))
                plt.axis("off")

                # Save figure to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)

                # Log to tensorboard
                self.writer.add_image("eval/world_model_predictions", Image.open(buf), self._step)
                plt.close(fig)

                # Also log animation if possible
                try:
                    import matplotlib.animation as animation

                    fig = plt.figure(figsize=(12, 9))
                    ims = []

                    # Create frames showing each sequence step
                    for i in range(video_pred_np.shape[1]):
                        im = plt.imshow(np.concatenate([seq[i] for seq in video_pred_np], axis=0), animated=True)
                        plt.axis("off")
                        plt.title(f"Step {i}")
                        ims.append([im])

                    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)

                    # Save animation
                    buf = io.BytesIO()
                    ani.save(buf, writer="pillow", fps=5)
                    buf.seek(0)

                    # Log to tensorboard
                    self.writer.add_image("eval/world_model_animation", Image.open(buf), self._step)
                    plt.close(fig)

                except Exception as e:
                    print(f"Error creating animation: {e}")

            except Exception as e:
                print(f"Error generating world model prediction video: {e}")

            has_video_logged = True

        self.set_train()
        return eval_returns

    def _generate_imagined_trajectory(self, state, batch_idx, horizon=8):
        """Generate an imagined trajectory for visualization"""
        with torch.no_grad():
            # Get current latent from state
            latent, _ = state

            # Select batch index
            batch_latent = {k: v[batch_idx : batch_idx + 1] for k, v in latent.items()}

            # Generate imagined trajectory
            imagined_latents = [batch_latent]
            actions = []

            for _ in range(horizon):
                # Get features for actor
                feat = self.world_model.dynamics.get_feat(imagined_latents[-1])

                # Sample action from actor
                actor = self.behavior.actor(feat)

                # Fix: Check if mode is a callable method or a tensor property
                if callable(getattr(actor, "mode", None)):
                    action = actor.mode()
                else:
                    # Fallback if mode is not callable - use mean or other property
                    if hasattr(actor, "mean"):
                        action = actor.mean
                    elif hasattr(actor, "loc"):
                        action = actor.loc
                    else:
                        # Last resort: sample deterministically
                        action = actor.sample()

                actions.append(action)

                # Predict next latent
                next_latent = self.world_model.dynamics.img_step(imagined_latents[-1], action)
                imagined_latents.append(next_latent)

            # Decode frames from latents
            decoded_frames = []
            for latent in imagined_latents:
                feat = self.world_model.dynamics.get_feat(latent)
                decoded = self.world_model.decoder(feat)
                if "image" in decoded:
                    decoded_frames.append(decoded["image"].cpu().numpy()[0])
                else:
                    # If no image in decoded output, create a placeholder
                    decoded_frames.append(np.zeros((64, 64, 3), dtype=np.uint8))

            return decoded_frames

    def _log_videos(self, frames_buffer, imagined_frames_buffer, num_episodes):
        """Log videos of actual gameplay and imagined trajectories"""
        try:
            import io

            import matplotlib.animation as animation
            import matplotlib.pyplot as plt
            from PIL import Image

            for ep_idx in range(num_episodes):
                # Real gameplay video
                if len(frames_buffer[ep_idx]) > 0:
                    fig = plt.figure(figsize=(4, 4))
                    ims = []

                    for i, frame in enumerate(frames_buffer[ep_idx]):
                        ax = plt.subplot(111)
                        if frame.shape[-1] == 3:  # RGB
                            im = ax.imshow(frame)
                        else:  # Grayscale
                            im = ax.imshow(frame.squeeze(), cmap="gray", animated=True)
                        ax.set_title(f"Step {i}")
                        ax.axis("off")
                        ims.append([im])

                    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
                    buf = io.BytesIO()
                    ani.save(buf, writer="pillow", fps=10)
                    buf.seek(0)
                    self.writer.add_image(f"eval/video_ep{ep_idx}", Image.open(buf), self._step)
                    plt.close(fig)

                # Imagined trajectory video (predictions)
                if imagined_frames_buffer[ep_idx]:
                    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                    axes = axes.flatten()

                    for i, prediction_sequence in enumerate(imagined_frames_buffer[ep_idx]):
                        if i % 10 == 0 and i < 40:  # Log only 4 prediction sequences
                            for j, pred_frame in enumerate(prediction_sequence[:8]):
                                if pred_frame.shape[-1] == 3:  # RGB
                                    axes[j].imshow(pred_frame)
                                else:  # Grayscale
                                    axes[j].imshow(pred_frame.squeeze(), cmap="gray")
                                axes[j].set_title(f"t+{j}")
                                axes[j].axis("off")

                            buf = io.BytesIO()
                            plt.tight_layout()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            self.writer.add_image(f"eval/predictions_ep{ep_idx}_step{i}", Image.open(buf), self._step)
                            plt.clf()

                    plt.close(fig)
        except Exception as e:
            print(f"Error logging videos: {e}")

    def restore_test(self, fn):
        """Load checkpoint weights for evaluation/testing

        Args:
            fn: Path to the checkpoint file
        """
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Checkpoint file not found: {fn}")

        print(f"Loading checkpoint from {fn}")
        checkpoint = torch.load(fn, map_location=self.device)

        # Load world model state
        if "world_model" in checkpoint:
            self.world_model.load_state_dict(checkpoint["world_model"])
        else:
            print("Warning: Checkpoint does not contain world_model weights")

        # Load behavior model state
        if "behavior" in checkpoint:
            self.behavior.load_state_dict(checkpoint["behavior"])
        else:
            print("Warning: Checkpoint does not contain behavior weights")

        # Load running mean std if applicable (for observation normalization)
        if hasattr(self, "running_mean_std") and "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        # Set models to evaluation mode
        self.set_eval()

        print("Checkpoint loaded successfully")

    def predict(self, obs):
        """Generate action prediction for given observation

        Args:
            obs: Observation tensor

        Returns:
            action: Action tensor
        """
        # Format observation for dreamer
        obs_dict = {"obs": obs}
        obs_dict = self.preprocess_obs(obs_dict)

        # Convert to correct device if needed
        if isinstance(obs, torch.Tensor) and obs.device != torch.device(self.device):
            obs_dict["obs"] = obs_dict["obs"].to(self.device)

        # Get policy action
        with torch.no_grad():
            policy_output, _ = self._policy(obs_dict, None, training=False)
            action = policy_output["action"]

        return action
