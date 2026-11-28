"""The robot self-check: what a robot config must survive before a task is built on it.

``python -m metasim robot check <name>`` runs it. Four checks, each reported as a row:

- **config**: joint limits and default positions exist for every actuator and the defaults lie inside the limits;
- **assets**: the asset files the config names for the chosen backend exist (locally or on the Hub);
- **kinematics**: the MJCF loads in MuJoCo's model, the end-effector body is found and forward kinematics of the
  default pose is finite (the skill library and the IL adapters depend on this);
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


_ASSET_FIELDS = {"mujoco": ("mjcf_path",), "mjx": ("mjx_mjcf_path", "mjcf_path"), "isaacsim": ("usd_path",)}


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

    fields = _ASSET_FIELDS.get(sim, ("mjcf_path",))
    declared = [
        (f, getattr(cfg, f, None)) for f in fields if isinstance(getattr(cfg, f, None), str) and getattr(cfg, f)
    ]
    if not declared:
        return CheckRow("assets", False, f"no {' / '.join(fields)} declared for {sim}")
    missing = [p for _, p in declared if not os.path.exists(p)]
    if missing:
        try:
            from metasim.utils.hf_util import check_and_download_single

            for p in list(missing):
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


def check_hold(
    cfg, sim: str, *, steps: int = 200, joint_tol: float = 0.05, base_tol: float = 0.01, device: str = "cpu"
) -> CheckRow:
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
        ok = worst_dev <= joint_tol and (not fixed or base_drift <= base_tol)
        return CheckRow(
            "hold",
            ok,
            f"after {steps} steps at the default pose: max joint deviation {worst_dev:.4f} rad ({worst_joint}), "
            f"base moved {base_drift:.4f} m" + ("" if fixed else " (floating base, not judged)"),
        )
    except Exception as err:
        return CheckRow("hold", False, f"{type(err).__name__}: {str(err)[:160]}")
    finally:
        try:
            handler.close()
        except Exception:
            pass


def check_robot(
    name: str, *, sim: str = "mujoco", steps: int = 200, device: str = "cpu", hold: bool = True
) -> RobotReport:
    """Run every check for the robot config ``name`` (a registered robot) on ``sim``."""
    from metasim.utils.setup_util import get_robot

    cfg = get_robot(name)
    report = RobotReport(robot=cfg.name, simulator=sim)
    report.rows.append(check_config(cfg))
    report.rows.append(check_assets(cfg, sim))
    report.rows.append(check_kinematics(cfg))
    if hold:
        report.rows.append(check_hold(cfg, sim, steps=steps, device=device))
    return report
