import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from loguru import logger as log
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class MetaSimGymEnv(gym.Env):
    """Vectorized environment for MetaSim that supports parallel RL training"""

    def __init__(self, scenario: ScenarioCfg, num_envs: int = 1, sim_type: SimType = SimType.ISAACLAB):
        super().__init__()

        self.num_envs = num_envs
        env_class = get_sim_env_class(sim_type)
        env = env_class(scenario)
        self.env: EnvWrapper[BaseSimHandler] = env

        # Get joint limits for action space
        joint_limits = scenario.robots[0].joint_limits
        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits.values()]),
            high=np.array([lim[1] for lim in joint_limits.values()]),
            dtype=np.float32,
        )

        # Observation space: joint positions + end effector position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(joint_limits) + 3,),  # joints + XYZ
            dtype=np.float32,
        )

        self.max_episode_steps = 500
        self.current_step = 0

    def reset(self, seed=None):
        log.info("Resetting environments")
        """Reset all environments"""
        self.current_step = 0
        init_states = self._get_default_states()
        self.env.reset(states=init_states)
        return self._get_obs(), None

    def step(self, actions):
        """Step through all environments"""
        self.current_step += 1

        # Convert numpy actions to handler format
        ## TODO: action should support vectorized actions
        action_dicts = [{"dof_pos_target": dict(zip(self.env.handler.robot.joint_limits.keys(), actions))}]

        # Step the simulation
        obs, rewards, success, timeout, _ = self.env.step(action_dicts)
        return obs, rewards, success, timeout, {}

    def _get_obs(self):
        """Get current observations for all environments"""
        states = self.env.handler.get_states()
        states = [{**state["robots"], **state["objects"]} for state in states]  # XXX: compatible with old states format
        joint_pos = np.array([
            [state[self.env.handler.robot.name]["dof_pos"][j] for j in self.env.handler.robot.joint_limits.keys()]
            for state in states
        ])

        # Get end effector positions (assuming 'ee' is the end effector subpath)
        ee_pos = np.array([state["metasim_body_panda_hand"]["pos"] for state in states])

        return np.concatenate([joint_pos, ee_pos], axis=1)

    def _calculate_rewards(self):
        """Calculate rewards based on distance to origin"""
        states = self.env.handler.get_states()
        states = [{**state["robots"], **state["objects"]} for state in states]  # XXX: compatible with old states format
        ee_pos = np.array([state["metasim_body_panda_hand"]["pos"] for state in states])
        distances = np.linalg.norm(ee_pos, axis=1)
        return -distances  # Negative distance as reward

    def _get_default_states(self):
        """Generate default reset states"""
        return [
            {
                self.env.handler.robot.name: {
                    "dof_pos": {j: 0.0 for j in self.env.handler.robot.joint_limits.keys()},
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                }
            }
            for _ in range(self.num_envs)
        ]

    def close(self):
        self.env.close()


def train_ppo():
    """Training procedure for PPO"""
    # Environment setup
    scenario = ScenarioCfg(
        task=get_task("PickCube"),
        robot=get_robot("franka"),
    )

    env = MetaSimGymEnv(scenario, num_envs=1)
    env = DummyVecEnv([lambda: env])

    # PPO configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_reach")


if __name__ == "__main__":
    train_ppo()
