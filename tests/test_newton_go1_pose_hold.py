"""Regression test: go1 holds its pose under zero action on the Newton backend.

Guards the closed-loop control fix in MetaSim's Newton handler
(``metasim/sim/newton/newton.py::_apply_actuator_settings``). The raw go1 MJCF
ships no ``<actuator>`` elements, so ``add_mjcf`` infers every DOF's
``joint_target_mode`` as ``NONE`` from the (zero) MJCF gains. ``SolverMuJoCo``
only synthesizes a MuJoCo position-PD actuator for a DOF whose target mode is
non-NONE, so the model was built with ``nu == 0``: the per-actuator stiffness/
damping from ``RobotCfg.actuators`` was written to ``joint_target_*`` but nothing
read it, and the quadruped collapsed under gravity (base z 0.31 -> ~0.05).

The fix sets ``joint_target_mode`` to POSITION for position-controlled joints
with non-zero stiffness so the solver installs the PD servo that reads
``joint_target_pos``. This test asserts both the mechanism (nu > 0) and the
behavior (base z stays well above the collapse threshold under zero action).

Newton is GPU-backed; the test self-skips when CUDA is unavailable.
"""

from __future__ import annotations

import pytest
import torch

import roboverse_pack  # noqa: F401  (registers mjlab tasks/robots)
from metasim.task.registry import get_task_class

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton backend requires a GPU")


def test_go1_holds_pose_zero_action_newton():
    """Zero-action go1 on Newton must synthesize PD actuators and not collapse."""
    task_cls = get_task_class("mjlab.velocity_flat_go1_v2")
    scenario = task_cls.scenario.update(
        robots=["mjlab_go1"], simulator="newton", num_envs=1, headless=True, cameras=[]
    )
    env = task_cls(scenario=scenario, device="cuda:0")
    env.reset()

    # Mechanism: the MuJoCo solver must have synthesized one position actuator
    # per actuated DOF (12 for go1). nu == 0 is the exact pre-fix failure.
    assert env.handler._solver.mjw_model.nu == 12, (
        f"expected 12 synthesized position actuators, got nu={env.handler._solver.mjw_model.nu}"
    )

    def base_z() -> float:
        state = env.handler.get_states(mode="tensor")
        robot = state.robots[next(iter(state.robots))]
        return float(robot.root_state[0, 2])

    z0 = base_z()
    z_min = z0
    for _ in range(200):
        env.step(torch.zeros(1, 12, device="cuda:0"))
        z_min = min(z_min, base_z())

    # Behavior: under pure PD (zero policy action) the legs sag to a stable
    # steady state but the robot must NOT collapse. Pre-fix it fell to ~0.05;
    # the post-fix stable stance settles around ~0.23 (mjlab-native zero-action
    # settles ~0.26). 0.15 is a generous floor that still catches a collapse.
    assert z_min > 0.15, f"go1 collapsed under zero action on Newton: min base z={z_min:.3f} (start {z0:.3f})"
