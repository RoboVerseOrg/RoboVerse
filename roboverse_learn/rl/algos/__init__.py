"""Algorithm registry for RoboVerse.

This module sets up a registry of available reinforcement learning algorithms
that can be easily selected via configuration files.
"""

from __future__ import annotations

from typing import Any

from roboverse_learn.rl.algos.dreamer.dreamer import Dreamer
from roboverse_learn.rl.algos.ppo.ppo import PPO
from roboverse_learn.rl.algos.sac.sac import SAC
from roboverse_learn.rl.algos.td3.td3 import TD3

# Export the dreamer_main function for direct use
__all__ = ["get_algorithm", "register_algorithm"]

# Algorithm registry - maps algorithm names to their class constructors
ALGORITHM_REGISTRY: dict[str, type[Any]] = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "dreamer": Dreamer,
}


def register_algorithm(name: str, algorithm_class: type[Any]) -> None:
    """Register a new algorithm in the registry.

    Args:
        name: The name of the algorithm (used in config files)
        algorithm_class: The class implementing the algorithm

    Raises:
        ValueError: If an algorithm with this name is already registered
    """
    name = name.lower()
    if name in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' is already registered")

    ALGORITHM_REGISTRY[name] = algorithm_class


def get_algorithm(algo_name: str, **kwargs) -> Any:
    """Factory function to get an instance of the specified algorithm.

    Args:
        algo_name: Name of the algorithm to instantiate
        **kwargs: Arguments to pass to the algorithm constructor

    Returns:
        An instance of the requested algorithm

    Raises:
        ValueError: If the algorithm name is not recognized
    """
    if algo_name.lower() not in ALGORITHM_REGISTRY:
        available_algos = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm '{algo_name}' not found in registry. Available algorithms: {available_algos}")

    # Instantiate the algorithm with the provided kwargs
    return ALGORITHM_REGISTRY[algo_name.lower()](**kwargs)
