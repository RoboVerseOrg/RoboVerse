"""RSL-RL algorithm package for RoboVerse.

Provides PPO training using the rsl_rl library.
"""

from .env_wrapper import RslRlEnvWrapper

__all__ = ["RslRlEnvWrapper"]
