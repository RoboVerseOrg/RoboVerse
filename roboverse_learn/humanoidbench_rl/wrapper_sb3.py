from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from loguru import logger as log
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task


class Sb3EnvWrapper(VecEnv):
    """Wraps MetaSim environment to be compatible with Gymnasium API."""

    def __init__(self, scenario: ScenarioCfg):
        # Create the base environment
        self.sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if SimType(scenario.sim) == SimType.MUJOCO:
            self.sim_device = torch.device("cpu")
        scenario.headless = True
        self.num_envs = scenario.num_envs
        self.robot = scenario.robot
        self.task = scenario.task

        env_class = get_sim_env_class(SimType(scenario.sim))
        self.env = env_class(scenario)

        self.init_states, _, _ = get_traj(scenario.task, scenario.robot, self.env.handler)
        if len(self.init_states) < self.num_envs:
            self.init_states = (
                self.init_states * (self.num_envs // len(self.init_states))
                + self.init_states[: self.num_envs % len(self.init_states)]
            )
        else:
            self.init_states = self.init_states[: self.num_envs]

        # Set up action space based on robot joint limits
        joint_limits = self.robot.joint_limits
        action_low = []
        action_high = []
        for joint_name in joint_limits.keys():
            action_low.append(joint_limits[joint_name][0])
            action_high.append(joint_limits[joint_name][1])

        self._action_low = torch.tensor(action_low, dtype=torch.float32, device=self.sim_device)
        self._action_high = torch.tensor(action_high, dtype=torch.float32, device=self.sim_device)

        # action space is normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=self._action_low.shape, dtype=np.float32)

        # observation space
        # Create an observation space (51 dimensions) for a single environment, instead of the entire batch (2,51).
        initial_obs = self.reset()
        obs_shape = (initial_obs.shape[1],)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        log.info(f"Observation space: {self.observation_space}")
        log.info(f"Action space: {self.action_space}")

        self.max_episode_steps = self.env.handler.task.episode_length
        log.info(f"Max episode steps: {self.max_episode_steps}")

        # Get success_bar, and if it does not exist, set it to a relatively large value.
        try:
            self.success_bar = getattr(self.env.handler.task.reward_functions[0], "success_bar", 1000.0)
            log.info(f"Success reward threshold: {self.success_bar}")
        except (AttributeError, Exception) as e:
            log.warning(f"Could not get reward_success_bar: {e}. Using default value 1000.0")
            self.success_bar = 1000.0

        # Episode tracking variables for EpisodeLogCallback
        self.episode_rewards = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_success = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_success_subtasks = np.zeros(self.num_envs, dtype=np.int32)

        VecEnv.__init__(self, self.num_envs, self.observation_space, self.action_space)

    def normalize_action(self, action):
        return 2 * (action - self._action_low) / (self._action_high - self._action_low) - 1

    def unnormalize_action(self, action):
        return (action + 1) / 2 * (self._action_high - self._action_low) + self._action_low

    def reset(self, options=None):
        """Reset the environment."""
        _, _ = self.env.reset(states=self.init_states)

        humanoid_observation = self.get_humanoid_observation(self.env.handler.get_states())
        observations = humanoid_observation.cpu().numpy()

        # Reset episode tracking variables
        self.episode_rewards = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_success = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_success_subtasks = np.zeros(self.num_envs, dtype=np.int32)

        log.info("reset now")
        return observations

    def step_async(self, actions):
        # convert input to numpy array
        import torch

        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # convert to tensor
        self._async_actions = actions

    def step_wait(self):
        """
        Perform a single time step in the environment and wait for the result.
        This method follows the Stable Baselines3 VecEnv API.

        Returns:
            tuple: A tuple containing the following elements:
                - observations (np.ndarray, shape=(num_envs, obs_dim)): Observations from the environment.
                - rewards (np.ndarray, shape=(num_envs,)): Reward values for each environment.
                - dones (np.ndarray, shape=(num_envs,)): Flags indicating if the episode has ended for each environment (due to termination or truncation).
                - infos (list[dict]): List of additional information for each environment. Each dictionary contains the "TimeLimit.truncated" key,
                                      indicating if the episode was truncated due to timeout.
        """
        # Convert action format
        # joint_names (list[str]): List of joint names for the robot.
        joint_names = list(self.robot.joint_limits.keys())
        # unnormalized_actions (torch.Tensor, shape=(num_envs, action_dim)): Actions unnormalized to the robot's joint limits.
        unnormalized_actions = self.unnormalize_action(self._async_actions)

        # action_dict (list[Action]): List of action dictionaries for each environment.
        action_dict = [
            {
                "dof_pos_target": {
                    # joint_name (str): Name of the joint.
                    # pos (float): Target position for the joint.
                    joint_name: float(pos)
                    for joint_name, pos in zip(joint_names, unnormalized_actions[env_id])
                }
            }
            for env_id in range(self.num_envs)
        ]

        # Call the step method of the underlying MetaSim environment
        _, _, terminated_tensor, truncated_tensor, _ = self.env.step(action_dict)

        # Get formatted observations
        states = self.env.handler.get_states()
        humanoid_observation = self.get_humanoid_observation(states)
        humanoid_reward = self.get_humanoid_reward(states)

        # Convert tensors to NumPy arrays
        observations = humanoid_observation.cpu().numpy()
        rewards = humanoid_reward.cpu().numpy()
        terminateds = terminated_tensor.cpu().numpy()
        truncateds = truncated_tensor.cpu().numpy()

        # Calculate dones flags (episode end)
        # dones (np.ndarray, shape=(num_envs,)): Done flags for each environment (True if terminated or truncated).
        dones = terminateds | truncateds

        # Update episode tracking variables
        self.episode_rewards += rewards
        self.episode_lengths += 1

        # Determine whether the cumulative reward exceeds success_bar.
        success_mask = (self.episode_rewards >= self.success_bar).astype(np.int32)
        self.episode_success = success_mask

        # Construct infos list containing "TimeLimit.truncated" required by SB3
        infos = []
        for i in range(self.num_envs):
            env_info = {}
            env_info["TimeLimit.truncated"] = truncateds[i] and not terminateds[i]
            env_info["success"] = int(self.episode_success[i])
            if dones[i]:
                env_info["terminal_observation"] = observations[i]
                env_info["episode"] = {
                    "r": self.episode_rewards[i],
                    "l": self.episode_lengths[i],
                }
                # Reset episode tracking variables for this environment
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

            infos.append(env_info)

        # Auto-reset for done environments
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            _, _ = self.env.reset(
                self.init_states,  # Input all initial states
                done_indices,  # Reset only the done environments
            )

            reset_observations = self.get_humanoid_observation(self.env.handler.get_states())
            reset_observations_np = reset_observations.cpu().numpy()

            observations[done_indices] = reset_observations_np[done_indices]

        # Return in the format required by SB3 VecEnv API
        return observations, rewards, dones, infos

    def get_humanoid_observation(self, states) -> torch.Tensor:
        gym_observation = self.task.humanoid_obs_flatten_func(states)
        return gym_observation

    def get_humanoid_reward(self, states):
        # NOTE: For IsaacLab, metasim_reward is None, so calculate reward here
        final_reward = torch.zeros(self.num_envs, device=self.sim_device)
        for reward_func, reward_weight in zip(self.task.reward_functions, self.task.reward_weights):
            final_reward += reward_func(self.robot.name)(states) * reward_weight
        return final_reward

    def render(self):
        """Render the environment."""
        pass

    def close(self):
        """Clean up environment resources."""
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError("Checking if environment is wrapped is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def get_attr(self, attr_name, indices=None):
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        import torch

        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Setting attributes is not supported.")


def check_metasim_env(sim_type: SimType = SimType.ISAACGYM, num_envs: int = 2):
    """Create and check a MetaSim environment."""
    # Create test environment
    task = get_task("Slide")
    robot = get_robot("h1")
    scenario = ScenarioCfg(task=task, robot=robot)

    # Create wrapped environment
    env = Sb3EnvWrapper(scenario)

    # Run environment checker
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)

    return env


if __name__ == "__main__":
    from gymnasium.envs import register

    from metasim.cfg.scenario import ScenarioCfg
    from metasim.constants import SimType
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task

    task = get_task("humanoidbench:Walk")
    robot = get_robot("h1")
    scenario = ScenarioCfg(task=task, robot=robot)
    num_envs = 2
    sim_type = SimType.ISAACGYM
    register(
        id="metasim-hb-wrapper",
        entry_point="metasim.scripts.RL.sb3_wrapper_isaacgym:Sb3EnvWrapperIsaacgym",
        kwargs={"scenario": scenario, "sim_type": sim_type, "num_envs": num_envs},
    )

    env = gym.make("metasim-hb-wrapper", headless=False)
    ob, _ = env.reset()
    log.info(f"ob_space = {env.observation_space}, ob = {ob.shape}")
    log.info(f"ac_space = {env.action_space.shape}")
    for i in range(1000):
        env.render()
