"""Query-suite registration for shared handler utilities."""

from __future__ import annotations

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import _SUPPORTED_SIMS, register_shared_suite


def get_query_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create the standard scenario configuration for query tests."""
    if sim not in _SUPPORTED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for query tests")

    from metasim.scenario.lights import DomeLightCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from roboverse_pack.robots.g1_cfg import G1Dof29Cfg

    sim_params = SimParamCfg(
        dt=0.005,
        substeps=1,
        num_threads=10,
        solver_type=1,
        num_position_iterations=4,
        num_velocity_iterations=0,
        contact_offset=0.01,
        rest_offset=0.0,
        bounce_threshold_velocity=0.5,
        max_depenetration_velocity=1.0,
        default_buffer_size_multiplier=5,
        replace_cylinder_with_capsule=True,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
    )

    return ScenarioCfg(
        robots=[G1Dof29Cfg()],
        objects=[],
        cameras=[],
        num_envs=num_envs,
        simulator=sim,
        headless=True,
        env_spacing=2.5,
        decimation=1,
        sim_params=sim_params,
        lights=[
            DomeLightCfg(
                intensity=800.0,
                color=(0.85, 0.9, 1.0),
            )
        ],
    )


# Register this suite with the shared handler machinery in metasim/test/conftest.py
register_shared_suite("metasim.test.queries", get_query_scenario)
