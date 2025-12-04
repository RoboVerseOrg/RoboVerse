"""Scenario configuration test suite registration for shared handler utilities."""

from __future__ import annotations

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import _SUPPORTED_SIMS, register_shared_suite
from roboverse_pack.robots.franka_cfg import FrankaCfg


def get_1_robot_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for 1-robot tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for 1-robot tests")

    return ScenarioCfg(
        robots=[FrankaCfg()],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )


def get_2_robots_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for 2-robots tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for 2-robots tests")

    return ScenarioCfg(
        robots=[FrankaCfg(name="franka1"), FrankaCfg(name="franka2")],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )


# Register scenarios with file-specific prefixes
register_shared_suite("metasim.test.test_scenario_cfg.test_1_robot", get_1_robot_scenario)
register_shared_suite("metasim.test.test_scenario_cfg.test_2_robots", get_2_robots_scenario)
