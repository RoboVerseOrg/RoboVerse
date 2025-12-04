"""Robot configuration test suite registration for shared handler utilities."""

from __future__ import annotations

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import _SUPPORTED_SIMS, register_shared_suite
from roboverse_pack.robots.franka_cfg import FrankaCfg


def get_default_qpos_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for default qpos tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for default qpos tests")

    return ScenarioCfg(
        robots=[
            FrankaCfg(
                default_joint_positions={
                    "panda_joint1": 0.0 - 0.1,
                    "panda_joint2": -0.785398 - 0.1,
                    "panda_joint3": 0.0 - 0.1,
                    "panda_joint4": -2.356194 - 0.1,
                    "panda_joint5": 0.0 - 0.1,
                    "panda_joint6": 1.570796 + 0.1,
                    "panda_joint7": 0.785398 + 0.1,
                    "panda_finger_joint1": 0.0,
                    "panda_finger_joint2": 0.0,
                },
            )
        ],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )


def get_qpos_limit_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create scenario configuration for qpos limit tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for qpos limit tests")

    return ScenarioCfg(
        robots=[
            FrankaCfg(
                default_joint_positions={
                    "panda_joint1": 0.0 - 0.1,
                    "panda_joint2": -0.785398 - 0.1,
                    "panda_joint3": 0.0 - 0.1,
                    "panda_joint4": -2.356194 - 0.1,
                    "panda_joint5": 0.0 - 0.1,
                    "panda_joint6": 1.570796 + 0.1,
                    "panda_joint7": 0.785398 + 0.1,
                    "panda_finger_joint1": 0.0,
                    "panda_finger_joint2": 0.0,
                },
                default_position=(0, 0, 1.0),
            )
        ],
        headless=True,
        num_envs=num_envs,
        simulator=sim,
    )


# Register scenarios with file-specific prefixes
register_shared_suite("metasim.test.test_scenario_cfg.test_robot_cfg.test_default_qpos", get_default_qpos_scenario)
register_shared_suite("metasim.test.test_scenario_cfg.test_robot_cfg.test_qpos_limit", get_qpos_limit_scenario)
