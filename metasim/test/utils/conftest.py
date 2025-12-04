"""Configuration for utils test suite.

Most tests in this suite are @pytest.mark.general (pure unit tests)
and don't require a simulator. A few integration tests for kinematics
may use the handler fixture.
"""

from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import register_shared_suite

try:
    from roboverse_pack.robots.g1_cfg import G1Cfg
except ImportError:
    G1Cfg = None


def get_kinematics_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Build scenario for kinematics integration tests.

    Only used by tests that need a simulator (e.g., IK solver tests).
    Most utils tests are @pytest.mark.general and don't use this.
    """
    if G1Cfg is None:
        raise ImportError("G1Cfg not available for kinematics tests")

    return ScenarioCfg(
        robots=[G1Cfg()],
        objects=[PrimitiveCubeCfg(name="test_cube")],
        num_envs=num_envs,
        simulator=sim,
        headless=True,
    )


# Register scenario only for kinematics integration tests
register_shared_suite("metasim.test.utils.test_kinematics", get_kinematics_scenario)
