"""Register randomization suite with the shared handler utilities."""

from __future__ import annotations

from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import register_shared_suite

_ALLOWED_SIMS = {"isaacsim"}


def get_randomization_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Create a standard scenario configuration for randomization tests."""
    if sim not in _ALLOWED_SIMS:
        raise ValueError(f"Unsupported simulator '{sim}' for randomization tests")

    from metasim.constants import PhysicStateType
    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg
    from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveSphereCfg
    from roboverse_pack.robots.franka_cfg import FrankaCfg

    return ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveSphereCfg(
                name="sphere",
                radius=0.1,
                color=[0.0, 0.0, 1.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.4, -0.6, 0.05],
            ),
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0.5, 0.0, 0.5],
            ),
        ],
        lights=[
            DiskLightCfg(
                name="test_light",
                intensity=20000.0,
                color=(1.0, 1.0, 1.0),
                radius=1.2,
                pos=(0.0, 0.0, 4.5),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        robots=[FrankaCfg()],
        cameras=[
            PinholeCameraCfg(
                name="test_camera",
                width=1024,
                height=1024,
                pos=(2.0, -2.0, 2.0),
                look_at=(0.0, 0.0, 0.05),
            )
        ],
    )


# Register this suite with the shared handler machinery in metasim/test/conftest.py
register_shared_suite("metasim.test.randomization", get_randomization_scenario)
