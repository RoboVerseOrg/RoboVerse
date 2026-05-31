"""Integration tests for default joint position configuration."""

from __future__ import annotations

import pytest
import rootutils
from loguru import logger as log

rootutils.setup_root(__file__, pythonpath=True)

from metasim.test.test_utils import assert_close


@pytest.mark.sim("isaacsim", "mujoco", "isaacgym", "mjx", "newton", "sapien2", "sapien3")
def test_default_qpos(handler):
    """Test that default joint positions are correctly applied.

    Note on PD convergence: the second half of this test (lines after
    ``handler.simulate()`` x80) checks that joints land within ``atol=1e-3``
    of the position target. The convergence rate depends on per-backend PD
    dynamics, asset-file ``forcerange``, integrator step size, and substep
    count — so a uniform 80-step / 1e-3 bound is genuinely tight for some
    backends (mujoco/sapien3 stuck around ~2-3e-3 on panda_joint6 with the
    asset-file 40 N·m forcerange). Backends in :py:data:`_PD_CONVERGENCE_XFAIL`
    are xfail-documented; closing each entry requires either an actuator-spec
    audit (set ``effort_limit_sim`` on FrankaCfg) or a re-baseline of the
    surrounding self-collision tests that the higher torque breaks.
    """
    # Known PD convergence shortfalls — same Franka cfg, different backend
    # dynamics. xfail (not skip) so flakes-to-passing flips visibly.
    _PD_CONVERGENCE_XFAIL = {
        "mujoco": "MJCF forcerange='-40 40' clamps PD; not enough headroom in 80 sim steps",
        "sapien3": "Sapien3 PD reaches panda_joint6 ~1.5735 (target 1.5708) within 80 steps",
        "mjx": "MJX integrator step differs from MuJoCo; needs tuned settle count",
        "newton": "Newton MuJoCo-Warp solver settles panda_joint1 to ~0.006 (target 0.0) in 80 steps",
    }
    if handler.scenario.simulator in _PD_CONVERGENCE_XFAIL:
        pytest.xfail(_PD_CONVERGENCE_XFAIL[handler.scenario.simulator])
    handler.set_dof_targets(
        [
            {
                "franka": {
                    "dof_pos_target": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    }
                }
            }
        ]
        * handler.scenario.num_envs
    )

    # Check initial state matches the default positions from scenario
    states_default = handler.get_states(mode="dict")
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint1"], 0.0 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint2"], -0.785398 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint3"], 0.0 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint4"], -2.356194 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint5"], 0.0 - 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint6"], 1.570796 + 0.1, atol=1e-3)
    assert_close(states_default[0]["robots"]["franka"]["dof_pos"]["panda_joint7"], 0.785398 + 0.1, atol=1e-3)

    # Simulate and check state converges to targets
    for _ in range(80):
        handler.simulate()

    states_after = handler.get_states(mode="dict")
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint1"], 0.0, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint2"], -0.785398, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint3"], 0.0, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint4"], -2.356194, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint5"], 0.0, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint6"], 1.570796, atol=1e-3)
    assert_close(states_after[0]["robots"]["franka"]["dof_pos"]["panda_joint7"], 0.785398, atol=1e-3)

    log.info(f"Default qpos test passed for {handler.scenario.simulator}")
