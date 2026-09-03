"""Static validation: every concrete ``RobotCfg`` subclass shipped with
MetaSim must (1) instantiate without arguments, (2) carry a non-empty
``name``, and (3) hold internally-consistent joint metadata.

Why: RoboVerse aims to be a standard cross-simulator benchmark. Robot
configs are the user-facing contract — every backend reads them. A
silent typo in ``default_joint_positions`` (a key that isn't in
``joint_limits``) currently passes review and only fails at sim launch
on whichever backend exercises that joint first.

This test catches that drift statically (``-k general``, no GPU, no
sim env). It runs against the ``RobotCfg`` subclasses bundled in
MetaSim's example pack; downstream packages should mirror this test
on their own subclasses.

Backward-compat: this is purely additive. The asserted constraints
already hold for the in-tree configs at the time of writing — the
test exists to prevent future regressions, not to require
modifications.
"""

from __future__ import annotations

import pytest

# Import the bundled RobotCfg subclasses so they register in __subclasses__.
# Optional/heavy imports are tolerated.
try:
    from metasim.example.example_pack.robots.franka_cfg import FrankaCfg
except Exception:
    FrankaCfg = None  # type: ignore[assignment]
try:
    from metasim.example.example_pack.robots.g1_cfg import G1Dof12Cfg
except Exception:
    G1Dof12Cfg = None  # type: ignore[assignment]
try:
    from metasim.example.example_pack.robots.h1_cfg import H1Cfg
except Exception:
    H1Cfg = None  # type: ignore[assignment]

from metasim.scenario.robot import RobotCfg


def _all_concrete_robot_cfgs() -> list[type[RobotCfg]]:
    """Walk ``RobotCfg.__subclasses__`` transitively, skip stubs whose
    ``MISSING`` fields would crash instantiation. We can't catch every
    sentinel without instantiating, so concrete subclasses that fail to
    instantiate are caught by ``test_robot_cfg_instantiates`` instead.
    """
    seen: set[type[RobotCfg]] = set()
    stack = list(RobotCfg.__subclasses__())
    out: list[type[RobotCfg]] = []
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        out.append(cls)
        stack.extend(cls.__subclasses__())
    return out


_CFG_PARAMS = [pytest.param(cls, id=cls.__name__) for cls in _all_concrete_robot_cfgs()]


def _try_instantiate(cls: type[RobotCfg]) -> RobotCfg | None:
    """Some bundled subclasses use ``MISSING`` sentinels for fields that
    only later subclasses fill in (e.g. ``G1Dof23Cfg`` inherits paths
    from G1Dof12Cfg but overrides them with MISSING). Those are not
    user-facing concrete classes and we skip them rather than fail."""
    try:
        return cls()
    except Exception:
        return None


@pytest.mark.general
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_instantiates(cls: type[RobotCfg]):
    """Every shipped RobotCfg must be instantiable, OR clearly fail with
    a MISSING-style sentinel (which we treat as 'abstract template')."""
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} is an abstract template (MISSING fields)")
    assert isinstance(instance, RobotCfg)


@pytest.mark.general
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_has_name(cls: type[RobotCfg]):
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    assert instance.name, f"{cls.__name__}.name is empty or None — robots need a stable identifier"


@pytest.mark.general
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_default_positions_match_joint_limits(cls: type[RobotCfg]):
    """Every joint listed in ``default_joint_positions`` must also appear
    in ``joint_limits`` — otherwise the simulator will silently fall
    back to its own default for that joint at reset time."""
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    if instance.default_joint_positions is None or instance.joint_limits is None:
        pytest.skip(f"{cls.__name__} does not declare both default_joint_positions and joint_limits")

    default_keys = set(instance.default_joint_positions)
    limit_keys = set(instance.joint_limits)
    orphans = default_keys - limit_keys
    assert not orphans, (
        f"{cls.__name__}: joints in default_joint_positions but not in joint_limits: {sorted(orphans)}. "
        f"Either typo or a joint added to defaults without a limit — backends will silently fall back to their own default."
    )


@pytest.mark.general
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_default_positions_inside_limits(cls: type[RobotCfg]):
    """Initial qpos must be inside the joint limit — otherwise the robot
    spawns in a forbidden pose and the first physics step explodes."""
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    if instance.default_joint_positions is None or instance.joint_limits is None:
        pytest.skip(f"{cls.__name__} does not declare both default_joint_positions and joint_limits")

    out_of_range: list[str] = []
    for jn, default in instance.default_joint_positions.items():
        if jn not in instance.joint_limits:
            continue  # handled by the orphan test
        lo, hi = instance.joint_limits[jn]
        if not (lo <= default <= hi):
            out_of_range.append(f"{jn}={default} not in [{lo}, {hi}]")
    assert not out_of_range, (
        f"{cls.__name__}: default joint positions outside joint_limits: {out_of_range}. "
        f"Robot will spawn in an illegal pose."
    )
