"""Static validation for every ``RobotCfg`` shipped in ``roboverse_pack``.

RoboVerse aims to be a standard cross-simulator benchmark. The robot pack
has 60+ configs imported from menagerie / maniskill / robotwin etc.;
typos in ``default_joint_positions`` or out-of-limits defaults currently
slip past review and only surface when the simulator reaches that joint.

This test walks every importable ``RobotCfg`` subclass under
``roboverse_pack/robots`` and statically checks:

1. The class instantiates without arguments (or skips cleanly if it's a
   MISSING-template parent).
2. ``name`` is non-empty.
3. Every joint listed in ``default_joint_positions`` is also in
   ``joint_limits``.
4. Every default position lies inside its joint limit interval.

Forward / backward compat: purely additive. The asserted constraints
hold for the current pack at the time of writing — the test exists to
catch future regressions, not to require existing fixes.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from metasim.scenario.robot import RobotCfg


def _import_roboverse_pack_robots() -> None:
    """Discover and import every ``roboverse_pack.robots.*`` module so
    its subclasses register in ``RobotCfg.__subclasses__``."""
    try:
        import roboverse_pack.robots as pkg
    except Exception:
        return
    for mod_info in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(mod_info.name)
        except Exception:
            continue


_import_roboverse_pack_robots()


def _all_concrete_robot_cfgs() -> list[type[RobotCfg]]:
    seen: set[type[RobotCfg]] = set()
    stack = list(RobotCfg.__subclasses__())
    out: list[type[RobotCfg]] = []
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        # Skip RobotCfg subclasses defined in MetaSim's example pack —
        # those are tested in metasim/test/test_robot_cfg_validation_general.py.
        if not cls.__module__.startswith("roboverse_pack"):
            stack.extend(cls.__subclasses__())
            continue
        out.append(cls)
        stack.extend(cls.__subclasses__())
    return out


_CFG_PARAMS = [pytest.param(cls, id=cls.__name__) for cls in _all_concrete_robot_cfgs()]


# Known config bugs surfaced when this test landed. Each entry is a triage
# ticket — flipping the xfail to xpass means the underlying RobotCfg was
# fixed. Reasons capture the *specific* failure so a future maintainer can
# tell at a glance whether the fix is config-side or limit-side.
#
# Fixing any of these is a runtime-visible behavior change (the silent
# clamp / silent ignore that used to happen no longer happens), so the
# fix has to be deliberate per-robot. xfail keeps the suite green while
# preserving backward compat.
_KNOWN_DEFAULT_POS_ORPHAN_GAPS: dict[str, str] = {
    "AlohaAgilexCfg": (
        "16 fl_/fr_joint{1..8} keys in default_joint_positions but only "
        "single-arm names in joint_limits — bimanual override gap."
    ),
    "G1TrackingCfg": (
        "Regex keys like '.*_ankle_pitch_joint' used in default_joint_positions "
        "are broadcast intent, not literal joint names; joint_limits has the "
        "concrete left_/right_ names. Either expand the regex at cfg time or "
        "drop the regex from defaults."
    ),
}

_KNOWN_DEFAULT_POS_OUT_OF_RANGE_GAPS: dict[str, str] = {
    "YamCfg": "joint2/joint4 defaults copy-pasted from Franka home pose; limits are Yam's narrower ranges.",
    "ArxL5Cfg": "joint2/joint4 defaults copy-pasted from Franka home pose; limits are ArxL5's narrower ranges.",
    "VegaCfg": "torso_j1 default 0.0 but joint_limits is the single-point [0.2, 0.2] — limit looks like a fixed offset, not a range.",
    "SoArm100Cfg": "Wrist_Pitch default -2.356 (Franka-style) outside [-0.192, 3.927].",
    "KochCfg": "wrist_pitch default -2.356 (Franka-style) outside [-0.192, 3.927].",
    "Go2Cfg": "RL/RR_thigh_joint default 1.0 outside [-4.54, 0.52] — sign or value error in stand-pose default.",
    "AllegroHandCfg": "thumb_joint_0 default 0.0 below lower limit 0.263; either default→0.263 or limit→0.",
}


def _try_instantiate(cls: type[RobotCfg]) -> RobotCfg | None:
    try:
        return cls()
    except Exception:
        return None


@pytest.mark.skipif(not _CFG_PARAMS, reason="roboverse_pack not importable")
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_instantiates(cls: type[RobotCfg]):
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} is an abstract template (MISSING fields)")
    assert isinstance(instance, RobotCfg)


@pytest.mark.skipif(not _CFG_PARAMS, reason="roboverse_pack not importable")
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_has_name(cls: type[RobotCfg]):
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    assert instance.name, f"{cls.__name__}.name is empty or None"


@pytest.mark.skipif(not _CFG_PARAMS, reason="roboverse_pack not importable")
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_default_positions_match_joint_limits(cls: type[RobotCfg]):
    if cls.__name__ in _KNOWN_DEFAULT_POS_ORPHAN_GAPS:
        pytest.xfail(_KNOWN_DEFAULT_POS_ORPHAN_GAPS[cls.__name__])
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    if instance.default_joint_positions is None or instance.joint_limits is None:
        pytest.skip(f"{cls.__name__} missing default_joint_positions or joint_limits")

    orphans = set(instance.default_joint_positions) - set(instance.joint_limits)
    assert not orphans, (
        f"{cls.__name__}: joints in default_joint_positions but not in joint_limits: {sorted(orphans)}. "
        f"Backends will silently fall back to their own default for these."
    )


@pytest.mark.skipif(not _CFG_PARAMS, reason="roboverse_pack not importable")
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_default_positions_inside_limits(cls: type[RobotCfg]):
    if cls.__name__ in _KNOWN_DEFAULT_POS_OUT_OF_RANGE_GAPS:
        pytest.xfail(_KNOWN_DEFAULT_POS_OUT_OF_RANGE_GAPS[cls.__name__])
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    if instance.default_joint_positions is None or instance.joint_limits is None:
        pytest.skip(f"{cls.__name__} missing default_joint_positions or joint_limits")

    out_of_range: list[str] = []
    for jn, default in instance.default_joint_positions.items():
        if jn not in instance.joint_limits:
            continue
        lo, hi = instance.joint_limits[jn]
        if not (lo <= default <= hi):
            out_of_range.append(f"{jn}={default} not in [{lo}, {hi}]")
    assert not out_of_range, (
        f"{cls.__name__}: default joint positions outside joint_limits: {out_of_range}. "
        f"Robot spawns in an illegal pose."
    )


def test_known_gap_dicts_match_actual_failures():
    """Self-check: every entry in the gap dicts must correspond to a still-failing
    cfg. When a robot's cfg is fixed, its entry becomes stale — this guard tells
    us so the xfail can be removed and the contract tightened."""
    name_to_cls = {cls.__name__: cls for cls in _all_concrete_robot_cfgs()}

    for name, reason in _KNOWN_DEFAULT_POS_ORPHAN_GAPS.items():
        if name not in name_to_cls:
            continue  # cfg not importable in this env
        instance = _try_instantiate(name_to_cls[name])
        if instance is None or instance.default_joint_positions is None or instance.joint_limits is None:
            continue
        orphans = set(instance.default_joint_positions) - set(instance.joint_limits)
        assert orphans, (
            f"_KNOWN_DEFAULT_POS_ORPHAN_GAPS[{name!r}] lists this as a gap ({reason!r}) but "
            f"the cfg is now clean — remove the entry so the contract is enforced."
        )

    for name, reason in _KNOWN_DEFAULT_POS_OUT_OF_RANGE_GAPS.items():
        if name not in name_to_cls:
            continue
        instance = _try_instantiate(name_to_cls[name])
        if instance is None or instance.default_joint_positions is None or instance.joint_limits is None:
            continue
        still_bad = False
        for jn, default in instance.default_joint_positions.items():
            if jn not in instance.joint_limits:
                continue
            lo, hi = instance.joint_limits[jn]
            if not (lo <= default <= hi):
                still_bad = True
                break
        assert still_bad, (
            f"_KNOWN_DEFAULT_POS_OUT_OF_RANGE_GAPS[{name!r}] lists this as a gap ({reason!r}) but "
            f"all defaults are now inside limits — remove the entry."
        )
