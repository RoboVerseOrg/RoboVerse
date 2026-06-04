"""Regression: handlers apply ``RobotCfg`` actuator armature to the model.

Armature (reflected motor inertia) set on a ``RobotCfg`` actuator must reach the
simulated model's joint-DOF armature, identically across backends. The MuJoCo
handler previously applied stiffness/damping/effort_limit from the cfg but
silently *ignored* armature — it kept the MJCF-authored value — while the Newton
handler applied it (``joint_armature``). The same ``RobotCfg`` therefore had a
different effective inertia on the two backends, breaking cross-backend parity
(and forcing downstream packs, e.g. the mjlab port, to hand-patch armature into
their MuJoCo MJCF). These tests pin the fix.
"""

from __future__ import annotations

import dataclasses

import pytest

# A distinctive value that differs from the Franka MJCF default (0.1) so a pass
# proves the cfg value flowed through rather than the asset value surviving.
_ARM = 0.0123
_JOINT = "panda_joint1"


def _franka_with_armature():
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg

    base = FrankaCfg()
    actuators = {name: dataclasses.replace(act, armature=_ARM) for name, act in base.actuators.items()}
    return dataclasses.replace(base, actuators=actuators)


@pytest.mark.mujoco
def test_mujoco_applies_robotcfg_armature():
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.sim_context import HandlerContext

    scenario = ScenarioCfg(simulator="mujoco", num_envs=1, headless=True, robots=[_franka_with_armature()])
    with HandlerContext(scenario) as handler:
        model = handler.physics.model
        jid = model.joint(f"{handler._mujoco_robot_names[0]}{_JOINT}").id
        dof_armature = float(model.dof_armature[model.jnt_dofadr[jid]])

    assert dof_armature == pytest.approx(_ARM, abs=1e-9), (
        f"MuJoCo handler did not apply RobotCfg armature: dof_armature={dof_armature}, expected {_ARM}"
    )


@pytest.mark.newton
def test_newton_applies_robotcfg_armature():
    pytest.importorskip("newton")
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.sim_context import HandlerContext

    scenario = ScenarioCfg(simulator="newton", num_envs=1, headless=True, robots=[_franka_with_armature()])
    with HandlerContext(scenario) as handler:
        model = handler._model
        # Newton stores per-DOF armature on the model; the robot is single-DOF
        # per actuated joint, so every controlled DOF should carry _ARM.
        joint_armature = model.joint_armature.numpy()

    applied = [a for a in joint_armature if abs(a - _ARM) < 1e-9]
    assert applied, f"Newton handler did not apply RobotCfg armature {_ARM}; armatures={sorted(set(joint_armature.tolist()))}"
