"""Configuration for utils test suite.

Most tests in this suite are @pytest.mark.general (pure unit tests)
and don't require a simulator. A few integration tests for kinematics
may use the handler fixture.
"""

from metasim.example.example_pack.robots.g1_cfg import G1Dof29Cfg
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import register_shared_suite


def get_kinematics_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Build scenario for kinematics integration tests.

    Only used by tests that need a simulator (e.g., IK solver tests).
    Most utils tests are @pytest.mark.general and don't use this.
    """
    return ScenarioCfg(
        robots=[G1Dof29Cfg()],
        objects=[PrimitiveCubeCfg(name="test_cube", size=[0.1, 0.1, 0.1], color=[0.5, 0.5, 0.5])],
        num_envs=num_envs,
        simulator=sim,
        headless=True,
    )


# Register scenario only for kinematics integration tests
register_shared_suite("metasim.test.utils.test_kinematics", get_kinematics_scenario)
