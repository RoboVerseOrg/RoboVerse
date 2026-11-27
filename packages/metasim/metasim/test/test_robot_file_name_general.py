"""``RobotCfg.file_name`` names the file the backend loads: Newton loads a robot's MJCF when it has one."""

from __future__ import annotations

import pytest

from metasim.scenario.robot import RobotCfg

pytestmark = pytest.mark.general


def test_newton_prefers_the_mjcf_and_falls_back_to_the_urdf():
    both = RobotCfg(name="r", mjcf_path="r.xml", urdf_path="r.urdf")
    assert both.file_name("newton") == "r.xml"
    assert both.file_name("isaacgym") == "r.urdf"
    assert both.file_name("mujoco") == "r.xml"
    assert RobotCfg(name="r", urdf_path="r.urdf").file_name("newton") == "r.urdf"
    assert RobotCfg(name="r", mjcf_path="r.mjcf", urdf_path="r.urdf").file_name("newton") == "r.urdf"  # not .xml
