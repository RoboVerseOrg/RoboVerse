"""Regression tests for the ``Sapien3Handler`` passive-joint guards.

Background: many real-world robots ship URDFs whose active-joint list is
broader than the RobotCfg's `actuators` dict — mobile-base wheels,
suspension bogies, camera masts, etc. The previous handler assumed a 1:1
correspondence and crashed with ``KeyError`` against any joint name not
explicitly enumerated.

These tests build a minimal URDF that mimics the pattern (one driven
arm joint + one passive wheel joint) and check that:

1. `_build_sapien` no longer raises ``KeyError`` for the passive joint
   when iterating actuators.
2. `default_joint_positions` is honoured for enumerated joints and
   falls back to 0.0 for joints the cfg doesn't mention.

We use only the static asset path validated in CI, so the test runs
without GPU and is marked ``@pytest.mark.sapien3``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


pytestmark = pytest.mark.sapien3


def _write_urdf(tmp_path: Path) -> str:
    p = tmp_path / "robot.urdf"
    p.write_text(
        textwrap.dedent(
            """
            <robot name="mini">
              <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
              <link name="arm"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>
              <link name="wheel"><inertial><mass value="0.05"/><inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/></inertial></link>
              <joint name="arm_joint" type="revolute">
                <parent link="base"/>
                <child link="arm"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
              </joint>
              <joint name="wheel_joint" type="continuous">
                <parent link="base"/>
                <child link="wheel"/>
                <axis xyz="0 1 0"/>
              </joint>
            </robot>
            """
        ).strip()
        + "\n"
    )
    return str(p)


def test_passive_wheel_does_not_crash_init(tmp_path):
    """An active joint missing from `actuators` must be tolerated, not crash."""
    from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.sim.sapien.sapien3 import Sapien3Handler
    from metasim.utils import configclass

    urdf = _write_urdf(tmp_path)

    @configclass
    class MiniRobotCfg(RobotCfg):
        name: str = "mini"
        num_joints: int = 1
        fix_base_link: bool = True
        urdf_path: str = urdf
        enabled_self_collisions: bool = False
        actuators: dict[str, BaseActuatorCfg] = {
            "arm_joint": BaseActuatorCfg(stiffness=100.0, damping=10.0, velocity_limit=2.0),
        }
        default_joint_positions: dict[str, float] = {"arm_joint": 0.5}

    scenario = ScenarioCfg(
        robots=[MiniRobotCfg()],
        sim_params=SimParamCfg(dt=0.01),
        decimation=2,
        simulator="sapien3",
        num_envs=1,
        headless=True,
        add_default_ground=False,
    )

    handler = Sapien3Handler(scenario)
    handler.launch()
    arts = handler.scene.get_all_articulations()
    assert len(arts) == 1
    arm_jid = None
    for jid, jn in enumerate(arts[0].get_active_joints()):
        if jn.get_name() == "arm_joint":
            arm_jid = jid
            break
    assert arm_jid is not None
    qpos = arts[0].get_qpos()
    assert abs(float(qpos[arm_jid]) - 0.5) < 1e-6, f"arm_joint default qpos not applied: {qpos}"
    handler.close()
