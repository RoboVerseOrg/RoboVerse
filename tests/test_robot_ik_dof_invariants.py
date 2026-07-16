"""IK / gripper DOF-accounting invariants for the robots fixed in
``deep-fix/robot-ik-gripper-dof``.

``metasim.utils.ik_solver.IKSolver`` splits a robot's actuators into an IK block
and a gripper block purely by count::

    n_dof_ik  = len(actuators) - len(gripper_open_q)   # leading arm joints
    ee_n_dof  = len(gripper_open_q)                     # trailing gripper joints
    n_robot_dof = len(joint_limits)                     # joint ordering / DOF count

For this to be correct, the arm joints must come first in ``actuators``, all
gripper joints must be last and contiguous, ``len(gripper_open_q)`` must equal
the real gripper-joint count, and ``joint_limits`` must enumerate exactly the
actuated joints (``compose_joint_action`` writes the gripper widths to the
trailing ``ee_n_dof`` columns of an ``n_robot_dof``-wide vector and then zips
those columns against ``actuators.keys()`` -- an oversized ``joint_limits``
silently drops the gripper command).

A wrong ``gripper_open_q`` length makes IK solve the wrong number of arm joints
(swallowing an arm joint into the gripper block, or vice-versa) with no error.
This test pins the corrected accounting so the regression can't silently return.
"""

from __future__ import annotations

import importlib

import pytest

# name -> (module, class name)
_ROBOTS = {
    "Ur5E2F85Cfg": ("roboverse_pack.robots.ur5e_2f85_cfg", "Ur5E2F85Cfg"),
    "IiwaCfg": ("roboverse_pack.robots.iiwa_cfg", "IiwaCfg"),
    "KinovaGen3Cfg": ("roboverse_pack.robots.kinova_gen3_cfg", "KinovaGen3Cfg"),
    "KinovaGen3Robotiq2f85Cfg": ("roboverse_pack.robots.kinova_gen3_robotiq_2f85", "KinovaGen3Robotiq2f85Cfg"),
    "FetchCfg": ("roboverse_pack.robots.fetch_cfg", "FetchCfg"),
    "ArxL5Cfg": ("roboverse_pack.robots.arx_l5_mjcf_cfg", "ArxL5Cfg"),
    "YamCfg": ("roboverse_pack.robots.yam_mjcf_cfg", "YamCfg"),
    "KochCfg": ("roboverse_pack.robots.koch_mjcf_cfg", "KochCfg"),
}

# Per-robot expectations. ``ik_wired``:
#   True          -> gripper_open_q set; IK block must equal ``arm_dof``.
#   False         -> deliberately not IK-wired (no gripper, no curobo ref).
#   "incompatible"-> a valid gripper descriptor is kept but the arm is not the
#                    leading actuator block, so the arm-first IKSolver cannot be
#                    used; we only require that no wrong curobo config is left.
# ``limits_eq_actuators`` asserts len(joint_limits) == len(actuators) (n_robot_dof
# consistency). Left False for ArxL5Cfg, whose passive 2nd finger ``joint8`` is a
# documented residual (see report / cfg comment).
_EXPECT = {
    "Ur5E2F85Cfg": dict(ik_wired=False, arm_dof=6),
    "IiwaCfg": dict(ik_wired=True, arm_dof=7, gripper_len=2, limits_eq_actuators=True),
    "KinovaGen3Cfg": dict(ik_wired=False, arm_dof=7),
    "KinovaGen3Robotiq2f85Cfg": dict(ik_wired=True, arm_dof=7, gripper_len=3, limits_eq_actuators=True),
    "FetchCfg": dict(ik_wired="incompatible", arm_dof=7, gripper_len=2),
    "ArxL5Cfg": dict(ik_wired=True, arm_dof=6, gripper_len=1, limits_eq_actuators=False),
    "YamCfg": dict(ik_wired=True, arm_dof=6, gripper_len=1, limits_eq_actuators=True),
    "KochCfg": dict(ik_wired=True, arm_dof=5, gripper_len=1, limits_eq_actuators=True),
}


def _load(name: str):
    mod, cls = _ROBOTS[name]
    return getattr(importlib.import_module(mod), cls)()


@pytest.mark.general
@pytest.mark.parametrize("name", list(_ROBOTS))
def test_ik_dof_accounting(name: str):
    exp = _EXPECT[name]
    cfg = _load(name)
    n_act = len(cfg.actuators)

    if exp["ik_wired"] is False:
        # Not IK-wired: no gripper widths and no (wrong) curobo reference, so
        # IKSolver is never mis-constructed on a robot it can't handle.
        assert cfg.gripper_open_q is None, f"{name}: expected no gripper_open_q on a non-IK robot"
        assert cfg.gripper_close_q is None, f"{name}: expected no gripper_close_q on a non-IK robot"
        assert cfg.curobo_ref_cfg_name is None, f"{name}: stale curobo_ref_cfg_name should be removed"
        return

    if exp["ik_wired"] == "incompatible":
        # Gripper descriptor is valid, but the arm is not the leading block; only
        # require that no wrong curobo config is left pointing at it.
        assert cfg.curobo_ref_cfg_name is None, f"{name}: wrong curobo_ref_cfg_name should be removed"
        assert len(cfg.gripper_open_q) == exp["gripper_len"]
        assert len(cfg.gripper_close_q) == len(cfg.gripper_open_q)
        return

    # ik_wired is True: full DOF-accounting contract.
    assert cfg.joint_limits is not None, f"{name}: joint_limits is None (IKSolver would crash)"
    assert cfg.gripper_open_q is not None and cfg.gripper_close_q is not None
    assert len(cfg.gripper_open_q) == exp["gripper_len"], f"{name}: gripper_open_q length"
    assert len(cfg.gripper_close_q) == len(cfg.gripper_open_q), f"{name}: open/close length mismatch"

    n_dof_ik = n_act - len(cfg.gripper_open_q)
    assert n_dof_ik == exp["arm_dof"], (
        f"{name}: n_dof_ik = len(actuators)-len(gripper_open_q) = {n_act}-{len(cfg.gripper_open_q)} "
        f"= {n_dof_ik}, expected the true arm DOF {exp['arm_dof']}"
    )

    if exp.get("limits_eq_actuators"):
        assert len(cfg.joint_limits) == n_act, (
            f"{name}: len(joint_limits)={len(cfg.joint_limits)} != len(actuators)={n_act}; "
            f"compose_joint_action would drop the gripper command"
        )

    # The trailing ``ee_n_dof`` actuator keys are the gripper block; their
    # open/close targets must lie inside the joints' own limits.
    trailing = list(cfg.actuators.keys())[-len(cfg.gripper_open_q) :]
    for jn, q_open, q_close in zip(trailing, cfg.gripper_open_q, cfg.gripper_close_q):
        if jn not in cfg.joint_limits:
            continue
        lo, hi = cfg.joint_limits[jn]
        assert lo <= q_open <= hi, f"{name}: gripper_open_q {jn}={q_open} not in [{lo}, {hi}]"
        assert lo <= q_close <= hi, f"{name}: gripper_close_q {jn}={q_close} not in [{lo}, {hi}]"


@pytest.mark.general
@pytest.mark.parametrize("name", list(_ROBOTS))
def test_default_positions_inside_limits(name: str):
    """Every default joint position must lie inside its own limit (pins the
    Franka-home-pose clamps for arx/yam/koch)."""
    cfg = _load(name)
    if cfg.default_joint_positions is None or cfg.joint_limits is None:
        pytest.skip(f"{name}: no default_joint_positions / joint_limits")
    bad = []
    for jn, default in cfg.default_joint_positions.items():
        if jn not in cfg.joint_limits:
            continue
        lo, hi = cfg.joint_limits[jn]
        if not (lo <= default <= hi):
            bad.append(f"{jn}={default} not in [{lo}, {hi}]")
    assert not bad, f"{name}: defaults outside limits: {bad}"
