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

# Distinctive values that differ from the Franka MJCF defaults (armature 0.1,
# frictionloss 0.0) so a pass proves the cfg value flowed through rather than the
# asset value surviving.
_ARM = 0.0123
_FRICTION = 0.0456
_JOINT = "panda_joint1"


def _franka_with_armature():
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg

    base = FrankaCfg()
    actuators = {
        name: dataclasses.replace(act, armature=_ARM, frictionloss=_FRICTION) for name, act in base.actuators.items()
    }
    return dataclasses.replace(base, actuators=actuators)


@pytest.mark.mujoco
def test_mujoco_applies_robotcfg_armature():
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.sim_context import HandlerContext

    scenario = ScenarioCfg(simulator="mujoco", num_envs=1, headless=True, robots=[_franka_with_armature()])
    with HandlerContext(scenario) as handler:
        model = handler.physics.model
        dof_adr = model.jnt_dofadr[model.joint(f"{handler._mujoco_robot_names[0]}{_JOINT}").id]
        dof_armature = float(model.dof_armature[dof_adr])
        dof_frictionloss = float(model.dof_frictionloss[dof_adr])

    assert dof_armature == pytest.approx(_ARM, abs=1e-9), (
        f"MuJoCo handler did not apply RobotCfg armature: dof_armature={dof_armature}, expected {_ARM}"
    )
    assert dof_frictionloss == pytest.approx(_FRICTION, abs=1e-9), (
        f"MuJoCo handler did not apply RobotCfg frictionloss: dof_frictionloss={dof_frictionloss}, expected {_FRICTION}"
    )


@pytest.mark.newton
def test_newton_applies_robotcfg_armature():
    pytest.importorskip("newton")
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.sim.sim_context import HandlerContext

    scenario = ScenarioCfg(simulator="newton", num_envs=1, headless=True, robots=[_franka_with_armature()])
    with HandlerContext(scenario) as handler:
        model = handler._model
        # Newton stores per-DOF armature/friction on the model; the robot is
        # single-DOF per actuated joint, so controlled DOFs should carry _ARM/_FRICTION.
        joint_armature = model.joint_armature.numpy()
        joint_friction = model.joint_friction.numpy() if getattr(model, "joint_friction", None) is not None else None

    assert any(abs(a - _ARM) < 1e-9 for a in joint_armature), (
        f"Newton handler did not apply RobotCfg armature {_ARM}; armatures={sorted(set(joint_armature.tolist()))}"
    )
    assert joint_friction is not None and any(abs(f - _FRICTION) < 1e-9 for f in joint_friction), (
        f"Newton handler did not apply RobotCfg frictionloss {_FRICTION}; "
        f"frictions={sorted(set(joint_friction.tolist())) if joint_friction is not None else None}"
    )
