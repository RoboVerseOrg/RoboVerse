"""SuperDex takes implicit-solver-sized steps without changing the simulated time per env step.

``dt`` (1 ms when None) and ``decimation`` define the env step on every backend; SuperDex covers it
with ``round(env_step / superdex_solver_dt)`` equal solver steps (5 ms by default), so 15 x 1 ms runs
as 3 x 5 ms and an explicit ``dt=0.005, decimation=15`` (75 ms) stays 15 x 5 ms.
"""

from __future__ import annotations

import pytest

pytest.importorskip("superdex.physics", reason="SuperDex physics wheels are not installed")
pytest.importorskip("superdex.robotics", reason="SuperDex robotics wheels are not installed")

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg

pytestmark = pytest.mark.superdex


def _scenario(**kwargs) -> ScenarioCfg:
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg

    return ScenarioCfg(
        robots=[FrankaCfg()],
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.05, 0.05, 0.05),
                color=[0.8, 0.1, 0.1],
                default_position=[0.4, 0.0, 0.3],
                physics=PhysicStateType.RIGIDBODY,
            )
        ],
        simulator="superdex",
        num_envs=1,
        headless=True,
        **kwargs,
    )


def _plan(**kwargs) -> tuple[int, float]:
    """(solver steps per env step, solver step in s) — derived in ``__init__``, no engine needed."""
    from metasim.sim.superdex.superdex import SuperdexHandler

    h = SuperdexHandler(_scenario(**kwargs))
    return h._substeps, h._dt


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (3, 0.005)),  # 15 x 1 ms env step -> 3 x 5 ms
        ({"sim_params": SimParamCfg(dt=0.005)}, (15, 0.005)),  # explicit dt: 75 ms env step, unchanged from before
        ({"sim_params": SimParamCfg(dt=0.001)}, (3, 0.005)),  # explicit 1 ms dt is still a 15 ms env step
        ({"sim_params": SimParamCfg(superdex_solver_dt=0.001)}, (15, 0.001)),  # the old 1 ms solver stepping
        ({"decimation": 4}, (1, 0.004)),  # 4 ms env step: one 4 ms solver step, never zero
        ({"decimation": 13}, (3, 0.013 / 3)),  # rounded to divide the env step exactly
    ],
    ids=["default", "explicit-dt-5ms", "explicit-dt-1ms", "solver-1ms", "decimation-4", "decimation-13"],
)
def test_solver_steps_cover_the_env_step_exactly(kwargs, expected):
    substeps, solver_dt = _plan(**kwargs)
    assert substeps == expected[0]
    assert solver_dt == pytest.approx(expected[1])
    physics_dt = kwargs.get("sim_params", SimParamCfg()).dt or 0.001
    assert substeps * solver_dt == pytest.approx(physics_dt * kwargs.get("decimation", 15))


def test_zero_env_step_is_rejected():
    from metasim.sim.superdex.superdex import SuperdexHandler

    with pytest.raises(ValueError, match="must be > 0"):
        SuperdexHandler(_scenario(sim_params=SimParamCfg(superdex_solver_dt=0.0)))


def test_simulate_advances_the_declared_env_step(monkeypatch, tmp_path):
    from metasim.sim.superdex.superdex import SuperdexHandler

    monkeypatch.setenv("METASIM_SUPERDEX_CACHE", str(tmp_path / "cache"))
    h = SuperdexHandler(_scenario())
    h.launch()
    try:
        t0 = h._sim_time
        h.simulate()
        assert h._sim_time - t0 == pytest.approx(15 * 0.001)
    finally:
        h.close()
