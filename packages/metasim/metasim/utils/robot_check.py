"""The robot self-check: what a robot config must survive before a task is built on it.

``python -m metasim robot check <name>`` runs it. Four checks, each reported as a row:

- **config**: joint limits and default positions exist for every actuator and the defaults lie inside the limits;
- **assets**: the asset files the config names for the chosen backend exist (locally or on the Hub);
- **kinematics**: the MJCF loads in MuJoCo's model, the end-effector body is found and forward kinematics of the
  default pose is finite (the skill library and the IL adapters depend on this);
- **limits**: the config's ``joint_limits`` agree with the ranges the MJCF declares (a config copied from another
  robot passes the config check and still commands poses the model cannot reach);
- **hold**: on the backend, commanding the default joint positions for ``steps`` steps leaves every joint within
  ``joint_tol`` radians of its target and a fixed base within ``base_tol`` metres (a robot that drifts or explodes
  at rest cannot be evaluated).

Returns the rows and an overall verdict; nothing here touches a task.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from metasim.constants import SimType


@dataclass
class CheckRow:
    """One check: its name, whether it passed, and the measurement or error behind the verdict."""

    check: str
    ok: bool
    detail: str


_OPEN_HANDLERS: list[Any] = []
"""Simulator handlers the hold check left open; :func:`close_handlers` releases them after the report is printed."""


def close_handlers() -> None:
    """Close the handlers the hold checks left open (last: on Isaac Sim this ends the process)."""
    while _OPEN_HANDLERS:
        handler = _OPEN_HANDLERS.pop()
        try:
            handler.close()
        except Exception:
            pass


@dataclass
class RobotReport:
    """Every check for one robot on one backend."""

    robot: str
    simulator: str
    rows: list[CheckRow] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Every check passed."""
        return all(r.ok for r in self.rows)

    def table(self) -> str:
        """The report as a Markdown table."""
        lines = [f"robot {self.robot} on {self.simulator}", "| check | result | detail |", "|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r.check} | {'ok' if r.ok else 'FAIL'} | {r.detail} |")
        lines.append(f"verdict: {'pass' if self.ok else 'FAIL'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"robot": self.robot, "simulator": self.simulator, "ok": self.ok, "rows": [asdict(r) for r in self.rows]}


_ASSET_FIELDS = {
    "mujoco": ("mjcf_path",),
    "mjx": ("mjx_mjcf_path", "mjcf_path"),
    "isaacsim": ("usd_path",),
    "isaacgym": ("urdf_path",),
    "genesis": ("urdf_path", "mjcf_path"),
    "newton": ("urdf_path", "mjcf_path", "usd_path"),
    "sapien3": ("urdf_path",),
    "pybullet": ("urdf_path",),
}
"""Which config fields name the asset each backend loads; any one of them present and existing passes."""
_ANY_ASSET = ("mjcf_path", "urdf_path", "usd_path")


def check_config(cfg) -> CheckRow:
    """Limits and defaults exist for every actuator and the defaults lie inside the limits."""
    joints = sorted(getattr(cfg, "actuators", {}) or {})
    limits = getattr(cfg, "joint_limits", None) or {}
    defaults = getattr(cfg, "default_joint_positions", None) or {}
    if not joints:
        return CheckRow("config", False, "no actuators declared")
    missing_limits = [j for j in joints if j not in limits]
    missing_defaults = [j for j in joints if j not in defaults]
    outside = [
        j
        for j in joints
        if j in limits and j in defaults and not (limits[j][0] - 1e-9 <= defaults[j] <= limits[j][1] + 1e-9)
    ]
    problems = []
    if missing_limits:
        problems.append(f"no joint_limits for {missing_limits}")
    if missing_defaults:
        problems.append(f"no default position for {missing_defaults}")
    if outside:
        problems.append(f"default outside limits for {outside}")
    return CheckRow(
        "config", not problems, "; ".join(problems) or f"{len(joints)} actuators, limits and defaults consistent"
    )


def check_assets(cfg, sim: str) -> CheckRow:
    """The asset files the config names for ``sim`` exist locally or can be fetched from the Hub."""
    import os

    fields = _ASSET_FIELDS.get(sim, _ANY_ASSET)
    declared = [
        (f, getattr(cfg, f, None)) for f in fields if isinstance(getattr(cfg, f, None), str) and getattr(cfg, f)
    ]
    if not declared:
        return CheckRow("assets", False, f"no {' / '.join(fields)} declared for {sim}")
    missing = [p for _, p in declared if not os.path.exists(p)]
    if missing:
        try:
            from metasim.utils.hf_util import LOCAL_DIR, _download_tree, check_and_download_single

            for p in list(missing):
                # the whole asset folder, not just the descriptor: an MJCF/URDF names meshes next to it
                rel_dir = os.path.relpath(os.path.realpath(os.path.dirname(p)), os.path.realpath(LOCAL_DIR))
                if not rel_dir.startswith(".."):
                    _download_tree(rel_dir.replace(os.sep, "/"))
                check_and_download_single(p)
                if os.path.exists(p):
                    missing.remove(p)
        except Exception as err:  # the Hub said no: report the path, not the traceback
            return CheckRow("assets", False, f"{missing} not local and not fetched from the Hub ({type(err).__name__})")
    if missing:
        return CheckRow("assets", False, f"missing files: {missing}")
    return CheckRow("assets", True, ", ".join(f"{f}={p}" for f, p in declared))


def check_kinematics(cfg) -> CheckRow:
    """The MJCF loads, the end-effector body exists and forward kinematics of the default pose is finite."""
    if not isinstance(getattr(cfg, "mjcf_path", None), str) or not cfg.mjcf_path:
        return CheckRow(
            "kinematics", False, "no mjcf_path: the kinematics check loads the robot's MJCF in MuJoCo's model"
        )
    if not getattr(cfg, "ee_body_name", None):
        return CheckRow("kinematics", True, "no ee_body_name (not an arm): tool kinematics not judged")
    try:
        from metasim.bench.skills import ArmModel

        arm = ArmModel(cfg)
        q = np.array([float((cfg.default_joint_positions or {}).get(j, 0.0)) for j in arm.arm_joints])
        pos, rot = arm.fk(q)
        if not (np.isfinite(pos).all() and np.isfinite(rot).all()):
            return CheckRow("kinematics", False, "forward kinematics of the default pose is not finite")
        return CheckRow(
            "kinematics",
            True,
            f"ee body {cfg.ee_body_name!r} found; {len(arm.arm_joints)} arm joints, {len(arm.gripper_joints)} gripper joints; "
            f"default tool pose ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m in the base frame",
        )
    except Exception as err:
        return CheckRow("kinematics", False, f"{type(err).__name__}: {str(err)[:160]}")


def check_limits(cfg, tol: float = 1e-3) -> CheckRow:
    """The config's joint limits agree with the MJCF's joint ranges (limited joints only)."""
    path = getattr(cfg, "mjcf_path", None)
    limits = getattr(cfg, "joint_limits", None) or {}
    if not isinstance(path, str) or not path or not limits:
        return CheckRow("limits", True, "no mjcf_path or no joint_limits: not judged")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(path)
    except Exception as err:
        return CheckRow("limits", False, f"MJCF did not load: {type(err).__name__}: {str(err)[:140]}")
    mismatches, compared = [], 0
    for joint, (lo, hi) in limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0 or not model.jnt_limited[jid]:
            continue
        compared += 1
        mlo, mhi = (float(x) for x in model.jnt_range[jid])
        if abs(mlo - float(lo)) > tol or abs(mhi - float(hi)) > tol:
            mismatches.append(f"{joint} cfg [{float(lo):.3f}, {float(hi):.3f}] vs mjcf [{mlo:.3f}, {mhi:.3f}]")
    if mismatches:
        return CheckRow(
            "limits",
            False,
            f"{len(mismatches)} of {compared} limited joints disagree with the MJCF: " + "; ".join(mismatches[:4]),
        )
    return CheckRow("limits", True, f"{compared} limited joints match the MJCF ranges")


def check_hold(cfg, sim: str, *, steps: int = 200, joint_tol: float = 0.05, base_tol: float = 0.01) -> CheckRow:
    """Hold the default pose on ``sim`` for ``steps`` steps; joints and a fixed base must stay put."""
    import torch

    from metasim.scenario.scenario import ScenarioCfg
    from metasim.utils.setup_util import get_sim_handler_class

    try:
        scenario = ScenarioCfg(robots=[cfg], simulator=sim, num_envs=1, headless=True)
        handler = get_sim_handler_class(SimType(sim))(scenario)
        handler.launch()
    except Exception as err:
        return CheckRow("hold", False, f"could not build the robot on {sim}: {type(err).__name__}: {str(err)[:160]}")
    try:
        joints = handler.get_joint_names(cfg.name, sort=True)
        defaults = {j: float((cfg.default_joint_positions or {}).get(j, 0.0)) for j in joints}
        start = handler.get_states(mode="tensor").robots[cfg.name]
        base0 = start.root_state[0, :3].detach().cpu().numpy()
        target = torch.tensor([defaults[j] for j in joints], dtype=torch.float32)
        worst_joint, worst_dev = "", 0.0
        for _ in range(int(steps)):
            handler.set_dof_targets([{cfg.name: {"dof_pos_target": dict(defaults)}}])  # one dict per env
            handler.simulate()
        state = handler.get_states(mode="tensor").robots[cfg.name]
        q = state.joint_pos[0].detach().cpu().float()
        dev = (q - target).abs()
        if not torch.isfinite(q).all():
            return CheckRow("hold", False, "joint positions are not finite after holding the default pose")
        i = int(torch.argmax(dev))
        worst_joint, worst_dev = joints[i], float(dev[i])
        base_drift = float(np.linalg.norm(state.root_state[0, :3].detach().cpu().numpy() - base0))
        fixed = bool(getattr(cfg, "fix_base_link", True))
        if not fixed:
            # a floating-base robot (legged, humanoid) at its default pose falls under gravity without a controller;
            # that is not a config defect, so only finiteness is judged and the drift is reported for the record
            return CheckRow(
                "hold",
                True,
                f"floating base: not judged; after {steps} steps max joint deviation {worst_dev:.4f} rad "
                f"({worst_joint}), base moved {base_drift:.4f} m",
            )
        ok = worst_dev <= joint_tol and base_drift <= base_tol
        return CheckRow(
            "hold",
            ok,
            f"after {steps} steps at the default pose: max joint deviation {worst_dev:.4f} rad ({worst_joint}), "
            f"base moved {base_drift:.4f} m",
        )
    except Exception as err:
        return CheckRow("hold", False, f"{type(err).__name__}: {str(err)[:160]}")
    finally:
        # not closed here: Isaac Sim 6 ends the process inside close(), which would take the report with it.
        # The caller closes it once the report is printed (close_handlers()).
        _OPEN_HANDLERS.append(handler)


def check_robot(name: str, *, sim: str = "mujoco", steps: int = 200, hold: bool = True) -> RobotReport:
    """Run every check for the robot config ``name`` (a registered robot) on ``sim``."""
    from metasim.utils.setup_util import get_robot

    cfg = get_robot(name)
    report = RobotReport(robot=cfg.name, simulator=sim)
    report.rows.append(check_config(cfg))
    report.rows.append(check_assets(cfg, sim))
    report.rows.append(check_kinematics(cfg))
    report.rows.append(check_limits(cfg))
    if hold:
        report.rows.append(check_hold(cfg, sim, steps=steps))
    return report
