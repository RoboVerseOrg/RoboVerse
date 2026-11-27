"""Newton sizes its constraint buffer for contact-rich robots when ``njmax`` is left unset.

MuJoCo-Warp's own estimate comes from the initial state: 64 rows for a Sharpa Wave hand in free
space, while a closed fist needs 137, and rows past the buffer are dropped with only a printed
``nefc overflow``. Needs the newton engine (GPU env); the resolver itself touches no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

newton_handler = pytest.importorskip("metasim.sim.newton.newton")
NewtonHandler = newton_handler.NewtonHandler


def _stub(num_envs: int, dof_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        scenario=SimpleNamespace(num_envs=num_envs),
        _model=SimpleNamespace(joint_dof_count=dof_count),
        NJMAX_FLOOR=NewtonHandler.NJMAX_FLOOR,
        NJMAX_ROWS_PER_DOF=NewtonHandler.NJMAX_ROWS_PER_DOF,
    )


def test_user_njmax_is_passed_through():
    assert NewtonHandler._resolve_njmax(_stub(4, 88), 40) == 40


def test_unset_njmax_gets_the_floor_for_a_hand():
    # Sharpa Wave: 22 DoFs per world. The closed-fist workload measured 137 rows.
    assert NewtonHandler._resolve_njmax(_stub(4, 4 * 22), None) >= 137


def test_unset_njmax_scales_with_dofs_per_world_not_total():
    big = NewtonHandler._resolve_njmax(_stub(1, 64), None)
    assert big == NewtonHandler.NJMAX_ROWS_PER_DOF * 64
    assert NewtonHandler._resolve_njmax(_stub(1024, 1024 * 64), None) == big


def test_native_estimate_is_read_from_the_built_solver():
    import mujoco

    # a free body over a height field: MuJoCo-Warp asks for ~41 x nv + 53 rows, above the floor
    xml = """<mujoco><asset><hfield name="h" nrow="4" ncol="4" size="1 1 0.1 0.1"/></asset>
    <worldbody><geom type="hfield" hfield="h"/><body pos="0 0 1"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    stub = SimpleNamespace(_solver=SimpleNamespace(mj_model=model, mj_data=mujoco.MjData(model)))
    native = NewtonHandler._native_njmax_estimate(stub)
    assert native is not None and native > NewtonHandler.NJMAX_FLOOR
    assert NewtonHandler._native_njmax_estimate(SimpleNamespace(_solver=None)) is None
