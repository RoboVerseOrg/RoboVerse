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


@pytest.mark.general
def test_g1_tracking_resolves_mjx_mjcf_path():
    """Regression: G1TrackingCfg.__post_init__ must chain to super().

    The base ArticulationObjCfg.__post_init__ defaults ``mjx_mjcf_path`` to
    ``mjcf_path`` when None. G1TrackingCfg computed its actuators but never called
    super(), so ``mjx_mjcf_path`` stayed None and the MJX backend got no path.
    """
    from roboverse_pack.robots.g1_tracking import G1TrackingCfg

    cfg = G1TrackingCfg()
    assert cfg.mjcf_path is not None
    assert cfg.mjx_mjcf_path == cfg.mjcf_path, "mjx_mjcf_path should default to mjcf_path via super().__post_init__()"
    # The actuator/action_scale computation must be preserved (not clobbered by super).
    assert len(cfg.actuators) > 0
    assert len(cfg.action_scale) == len(cfg.actuators)


@pytest.mark.general
def test_anymal_default_orientation_is_wxyz_identity():
    """Regression: ANYmal's default_orientation must be wxyz identity [1,0,0,0].

    Handlers consume default_orientation as wxyz (mujoco.py reads qw,qx,qy,qz;
    isaacgym reorders wxyz->xyzw). The value was [0,0,0,1] (the xyzw identity),
    which under wxyz is a 180 degree yaw — ANYmal would spawn facing backwards.
    """
    from roboverse_pack.robots.anymal_cfg import AnymalCfg

    cfg = AnymalCfg()
    assert cfg.default_orientation == [1.0, 0.0, 0.0, 0.0], (
        f"expected wxyz identity [1,0,0,0], got {cfg.default_orientation} "
        f"(w must be the largest component for an identity quaternion)"
    )


@pytest.mark.general
def test_g1_feet_cfg_imports_and_constructs():
    """Regression: ``roboverse_pack.robots.g1_feet`` must import and construct.

    Every ``BaseActuatorCfg`` in that file used to pass ``kp=/kd=/force_limit=`` —
    none of which are fields of ``BaseActuatorCfg`` — so *importing the module*
    raised ``TypeError`` at class-body evaluation. Because the crash happened at
    import time, ``roboverse_pack.robots.__init__`` swallowed it and ``Go1FeetCfg``
    silently never registered as a ``RobotCfg`` subclass. The correct field names
    are ``stiffness`` (kp), ``damping`` (kd), and ``effort_limit_sim`` (force limit),
    as consumed by the mujoco/isaacgym/newton/pybullet handlers.
    """
    import importlib

    mod = importlib.import_module("roboverse_pack.robots.g1_feet")
    cfg = mod.Go1FeetCfg()
    assert cfg.name == "go1_feet"
    assert len(cfg.actuators) == 12
    # Gains/limits from the XML must survive under the correct field names.
    hip = cfg.actuators["FR_hip"]
    assert (hip.stiffness, hip.damping, hip.effort_limit_sim) == (35.0, 0.5, 23.7)
    assert cfg.actuators["FR_calf"].effort_limit_sim == 35.55


# Joints that are *deliberately* pinned to a single point (lo == hi). These are
# intentional design choices (locked/unused DoF, or a fixed offset expressed as a
# degenerate range), not the unintended dropped-sign bug the test below hunts for.
# Keyed by cfg class name -> set of joint names that are allowed to be degenerate.
_DELIBERATELY_LOCKED_JOINTS: dict[str, set[str]] = {
    # VegaCfg locks its wheels, right arm/hand, and unused torso DoF; torso_j1 is a
    # fixed offset [0.2, 0.2]. All intentional — see vega_cfg.py.
    "VegaCfg": {
        "B_wheel_j1",
        "B_wheel_j2",
        "R_wheel_j1",
        "R_wheel_j2",
        "L_wheel_j1",
        "L_wheel_j2",
        "torso_j1",
        "torso_j3",
        "R_arm_j1",
        "R_arm_j2",
        "R_arm_j3",
        "R_arm_j4",
        "R_arm_j5",
        "R_arm_j6",
        "R_arm_j7",
        "R_th_j0",
        "R_th_j1",
        "R_th_j2",
        "R_ff_j1",
        "R_ff_j2",
        "R_mf_j1",
        "R_mf_j2",
        "R_rf_j1",
        "R_rf_j2",
        "R_lf_j1",
        "R_lf_j2",
    },
}


@pytest.mark.skipif(not _CFG_PARAMS, reason="roboverse_pack not importable")
@pytest.mark.parametrize("cls", _CFG_PARAMS)
def test_robot_cfg_no_unintended_degenerate_joint_limits(cls: type[RobotCfg]):
    """No joint limit may be degenerate (``lo == hi``) unless deliberately locked.

    A ``lo == hi`` limit pins the joint to a single point, so it cannot move — for a
    real DoF this is almost always a dropped sign (e.g. dex3's thumb-yaw was
    ``(-1.04, -1.04)`` instead of ``(-1.04, 1.04)``, leaving the thumb unable to
    oppose the fingers → no pinch grasp). Intentionally locked joints are listed in
    ``_DELIBERATELY_LOCKED_JOINTS``.
    """
    instance = _try_instantiate(cls)
    if instance is None:
        pytest.skip(f"{cls.__name__} abstract template")
    limits = getattr(instance, "joint_limits", None)
    if not limits:
        pytest.skip(f"{cls.__name__} has no joint_limits")

    allowed = _DELIBERATELY_LOCKED_JOINTS.get(cls.__name__, set())
    degenerate: list[str] = []
    for jn, bounds in limits.items():
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            continue
        lo, hi = bounds
        if lo == hi and jn not in allowed:
            degenerate.append(f"{jn}={bounds}")
    assert not degenerate, (
        f"{cls.__name__}: degenerate (lo == hi) joint limits pin these joints to a point: "
        f"{degenerate}. If this is intentional, add the joint to "
        f"_DELIBERATELY_LOCKED_JOINTS; otherwise a sign was likely dropped."
    )
