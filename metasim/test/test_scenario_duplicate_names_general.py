"""Static guard against duplicate entity names in ``ScenarioCfg``.

Background: every handler builds an internal
``object_dict = {obj.name: obj for obj in self.objects + self.robots}``
in its ``__init__``. Duplicate names silently collapse — the second
entity overrides the first, and set_states / get_states / set_dof_targets
silently reference the wrong entity. Different backends fail
differently downstream (MJCF composer errors on duplicate bodies,
Sapien3 picks the first, etc.), so the bug is genuinely cross-platform.

``ScenarioCfg.__post_init__`` now warns on construction. These tests
pin the behaviour without booting any simulator.
"""

from __future__ import annotations

import pytest
from loguru import logger as _loguru_logger


@pytest.fixture
def loguru_warnings():
    records: list[str] = []
    sink_id = _loguru_logger.add(lambda msg: records.append(str(msg)), level="WARNING")
    try:
        yield records
    finally:
        _loguru_logger.remove(sink_id)


def _make_robot(name: str):
    """Build a minimal RobotCfg-like object for the warning helper.

    We avoid importing real RobotCfg / instantiating the example pack
    because ``__post_init__`` only inspects ``.name`` on each list item;
    a duck-typed stand-in keeps the test independent of the example
    pack's optional asset downloads.
    """
    from metasim.scenario.scenario import ScenarioCfg

    class _Stub:
        def __init__(self, n):
            self.name = n

    _ = ScenarioCfg  # ensure import
    return _Stub(name)


def _make_obj(name: str):
    class _Stub:
        def __init__(self, n):
            self.name = n

    return _Stub(name)


@pytest.mark.general
def test_unique_robot_names_do_not_warn(loguru_warnings):
    from metasim.scenario.scenario import ScenarioCfg

    ScenarioCfg(robots=[_make_robot("arm_a"), _make_robot("arm_b")])
    out = "\n".join(loguru_warnings)
    assert "duplicate entity name" not in out


@pytest.mark.general
def test_duplicate_robot_names_warn(loguru_warnings):
    from metasim.scenario.scenario import ScenarioCfg

    ScenarioCfg(robots=[_make_robot("arm"), _make_robot("arm")])
    out = "\n".join(loguru_warnings)
    assert "duplicate entity name 'arm'" in out
    assert "set_states / set_dof_targets" in out


@pytest.mark.general
def test_robot_name_collides_with_object_name_warns(loguru_warnings):
    """The collision can be cross-kind — robots and objects share the
    same ``object_dict`` namespace inside handlers."""
    from metasim.scenario.scenario import ScenarioCfg

    ScenarioCfg(
        robots=[_make_robot("franka")],
        objects=[_make_obj("franka")],
    )
    out = "\n".join(loguru_warnings)
    assert "duplicate entity name 'franka'" in out


@pytest.mark.general
def test_duplicate_object_names_warn(loguru_warnings):
    from metasim.scenario.scenario import ScenarioCfg

    ScenarioCfg(objects=[_make_obj("cube"), _make_obj("cube"), _make_obj("cube")])
    out = "\n".join(loguru_warnings)
    assert "duplicate entity name 'cube'" in out
    # 3 entities sharing one name should be visible in the kinds list.
    assert "3 entities" in out
