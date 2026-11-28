"""The robot self-check: what a robot config must survive before a task is built on it.

``python -m metasim robot check <name>`` runs it. Six checks, each reported as a row:

- **config**: joint limits and default positions exist for every actuator and the defaults lie inside the limits;
- **assets**: the asset files the config names for the chosen backend exist (locally or on the Hub);
- **kinematics**: the MJCF loads in MuJoCo's model, the end-effector body is found and forward kinematics of the
  default pose is finite (the skill library and the IL adapters depend on this);
- **limits**: the config's ``joint_limits`` agree with the ranges the MJCF declares (a config copied from another
  robot passes the config check and still commands poses the model cannot reach);
- **actuators**: every MJCF actuator that drives a config joint is named after that joint (the MuJoCo handler finds
  an actuator by the joint's name and adds a second servo when it does not), and every joint the config drives
  exists in the MJCF;
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


def load_mjcf(cfg):
    """The config's MJCF as a ``mujoco.MjModel``, the exception that stopped it loading, or None without an MJCF."""
    path = getattr(cfg, "mjcf_path", None)
    if not isinstance(path, str) or not path:
        return None
    try:
        import mujoco

        return mujoco.MjModel.from_xml_path(path)
    except Exception as err:
        return err


def check_limits(cfg, tol: float = 1e-3, model=None) -> CheckRow:
    """The config's joint limits agree with the MJCF's joint ranges (limited joints only)."""
    path = getattr(cfg, "mjcf_path", None)
    limits = getattr(cfg, "joint_limits", None) or {}
    if not isinstance(path, str) or not path or not limits:
        return CheckRow("limits", True, "no mjcf_path or no joint_limits: not judged")
    import mujoco

    model = load_mjcf(cfg) if model is None else model
    if isinstance(model, Exception):
        return CheckRow("limits", False, f"MJCF did not load: {type(model).__name__}: {str(model)[:140]}")
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


def check_actuators(cfg, model=None) -> CheckRow:
    """MJCF actuators are named after the joints they drive; every config joint exists in the MJCF."""
    path = getattr(cfg, "mjcf_path", None)
    joints = sorted(getattr(cfg, "actuators", {}) or {})
    if not isinstance(path, str) or not path or not joints:
        return CheckRow("actuators", True, "no mjcf_path or no actuators: not judged")
    import mujoco

    model = load_mjcf(cfg) if model is None else model
    if isinstance(model, Exception):
        return CheckRow("actuators", False, f"MJCF did not load: {type(model).__name__}: {str(model)[:140]}")
    problems = []
    unknown = [j for j in joints if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) < 0]
    if unknown:
        problems.append(f"config drives joints the MJCF does not define: {unknown}")
    misnamed, driven = [], 0
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
        if joint not in joints:
            continue
        driven += 1
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if aname != joint:
            misnamed.append(f"{aname!r} drives {joint!r}")
    if misnamed:
        problems.append(
            f"{len(misnamed)} actuator(s) not named after their joint (the handler would add a second servo): "
            + "; ".join(misnamed[:4])
        )
    synthesized = len(joints) - len(unknown) - driven
    detail = "; ".join(problems) or (
        f"{driven} MJCF actuators named after their joints"
        + (f", {synthesized} joint(s) driven by a servo synthesized from the config" if synthesized else "")
    )
    return CheckRow("actuators", not problems, detail)


def check_hold(cfg, sim: str, *, steps: int = 200, joint_tol: float = 0.05, base_tol: float = 0.01) -> CheckRow:
    """Hold the default pose on ``sim`` for ``steps`` steps; joints and a fixed base must stay put."""
    import torch

    from metasim.scenario.scenario import ScenarioCfg
    from metasim.utils.setup_util import get_sim_handler_class

    try:
        scenario = ScenarioCfg(robots=[cfg], simulator=sim, num_envs=1, headless=True)
        handler = get_sim_handler_class(SimType(sim))(scenario)
        _OPEN_HANDLERS.append(handler)  # registered before launch: a launch that raises must still be closed
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
    # not closed here: Isaac Sim 6 ends the process inside close(), which would take the report with it.
    # The caller closes it once the report is printed (close_handlers()); the handler is registered above.


def check_robot_cfg(cfg, *, sim: str = "mujoco", steps: int = 200, hold: bool = True) -> RobotReport:
    """Run every check for the config object ``cfg`` on ``sim`` (``robot add`` checks the file it just wrote)."""
    report = RobotReport(robot=cfg.name, simulator=sim)
    report.rows.append(check_config(cfg))
    report.rows.append(check_assets(cfg, sim))
    report.rows.append(check_kinematics(cfg))
    model = load_mjcf(cfg)  # one parse for the two MJCF checks
    report.rows.append(check_limits(cfg, model=model))
    report.rows.append(check_actuators(cfg, model=model))
    if hold:
        report.rows.append(check_hold(cfg, sim, steps=steps))
    return report


def check_robot(name: str, *, sim: str = "mujoco", steps: int = 200, hold: bool = True) -> RobotReport:
    """Run every check for the robot config ``name`` (a registered robot) on ``sim``."""
    from metasim.utils.setup_util import get_robot

    return check_robot_cfg(get_robot(name), sim=sim, steps=steps, hold=hold)


# --------------------------------------------------------------------------- cross-engine parity


@dataclass
class TrackingTrace:
    """One backend tracking the canned joint trajectory: what was commanded and what was measured, per step."""

    simulator: str
    joints: list[str]
    commanded: np.ndarray
    """``(steps, n_joints)`` targets written each control step."""
    measured: np.ndarray
    """``(steps, n_joints)`` joint positions read after each control step."""
    control_dt: float | None = None

    @property
    def tracking_error(self) -> np.ndarray:
        """Per joint, the largest ``|measured - commanded|`` over the trace."""
        return np.abs(self.measured - self.commanded).max(axis=0)

    def lag_steps(self, max_lag: int = 60) -> np.ndarray:
        """Per joint, the delay (control steps) at which the measured trace best correlates with the command.

        A servo that follows the command 100 ms late shows a large ``tracking_error`` and a small lag; the two
        together say whether an error is slowness or a different path.
        """
        n, k = self.measured.shape
        lags = np.zeros(k, dtype=int)
        for j in range(k):
            c, m = self.commanded[:, j], self.measured[:, j]
            if np.ptp(c) < 1e-9 or np.ptp(m) < 1e-9 or n < 4:
                continue
            best, best_s = -np.inf, 0
            for s in range(0, min(max_lag, n - 3)):
                cc = c[: n - s] if s else c
                mm = m[s:]
                if np.std(cc) < 1e-12 or np.std(mm) < 1e-12:
                    continue
                r = float(np.corrcoef(cc, mm)[0, 1])
                if r > best:
                    best, best_s = r, s
            lags[j] = best_s
        return lags

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (arrays as lists)."""
        return {
            "simulator": self.simulator,
            "joints": list(self.joints),
            "control_dt": self.control_dt,
            "commanded": self.commanded.tolist(),
            "measured": self.measured.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingTrace:
        """Inverse of :meth:`to_dict`."""
        return cls(
            simulator=data["simulator"],
            joints=list(data["joints"]),
            commanded=np.asarray(data["commanded"], dtype=float),
            measured=np.asarray(data["measured"], dtype=float),
            control_dt=data.get("control_dt"),
        )


def canned_trajectory(
    cfg,
    joints: list[str],
    *,
    steps: int = 400,
    period_steps: int = 200,
    amplitude_frac: float = 0.25,
    amplitude_max: float = 0.3,
) -> np.ndarray:
    """A joint-space sinusoid around the default pose, inside the limits, one phase offset per joint.

    The same ``(steps, n)`` array is written on every backend, so the traces are comparable step by step.
    Amplitude is ``amplitude_frac`` of each joint's half range (0.5 rad for an unlimited joint), at most
    ``amplitude_max`` (so a 0.3 rad swing in 3 s peaks at 0.63 rad/s, the speed of a tabletop demo rather than of
    a joint at its velocity limit), clipped to the limits. A slide joint keeps the fraction (its range is small).
    The amplitude ramps in over the first period, so every joint's command starts at its default and the robot is
    never yanked at step 0 (a step on a phase-shifted joint used to swing an unlimited wrist by a radian).
    """
    limits = getattr(cfg, "joint_limits", None) or {}
    defaults = getattr(cfg, "default_joint_positions", None) or {}
    t = np.arange(int(steps), dtype=float)
    ramp = np.clip(t / float(period_steps), 0.0, 1.0)  # the same for every joint
    out = np.zeros((int(steps), len(joints)))
    for k, j in enumerate(joints):
        q0 = float(defaults.get(j, 0.0))
        lo, hi = limits.get(j, (-np.inf, np.inf))
        lo, hi = float(lo), float(hi)
        amp = amplitude_frac * (hi - lo) / 2 if np.isfinite(lo) and np.isfinite(hi) else 0.5
        if np.isfinite(amplitude_max) and amplitude_max > 0:
            amp = min(amp, amplitude_max)
        q = q0 + ramp * amp * np.sin(2 * np.pi * t / float(period_steps) + k * np.pi / 4)
        out[:, k] = np.clip(q, lo, hi)
    return out


def track_trajectory(
    cfg,
    sim: str,
    *,
    steps: int = 400,
    period_steps: int = 200,
    amplitude_frac: float = 0.25,
    amplitude_max: float = 0.3,
    settle_steps: int = 50,
    dt: float = 0.001,
    decimation: int = 15,
) -> TrackingTrace:
    """Build the robot alone on ``sim``, settle at the default pose, then track :func:`canned_trajectory`.

    The physics step and decimation are pinned (1 ms x 15 = a 15 ms control step by default) so two backends see
    the same command sequence at the same rate. The handler stays open in ``_OPEN_HANDLERS``; call
    :func:`close_handlers` after the report is printed.
    """
    import torch

    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.simulator_params import SimParamCfg
    from metasim.utils.setup_util import get_sim_handler_class

    scenario = ScenarioCfg(
        robots=[cfg], simulator=sim, num_envs=1, headless=True, sim_params=SimParamCfg(dt=dt), decimation=decimation
    )
    handler = get_sim_handler_class(SimType(sim))(scenario)
    handler.launch()
    _OPEN_HANDLERS.append(handler)
    joints = handler.get_joint_names(cfg.name, sort=True)
    defaults = {j: float((cfg.default_joint_positions or {}).get(j, 0.0)) for j in joints}
    for _ in range(int(settle_steps)):
        handler.set_dof_targets([{cfg.name: {"dof_pos_target": dict(defaults)}}])
        handler.simulate()
    commanded = canned_trajectory(
        cfg, joints, steps=steps, period_steps=period_steps, amplitude_frac=amplitude_frac, amplitude_max=amplitude_max
    )
    measured = np.zeros_like(commanded)
    for t in range(commanded.shape[0]):
        target = {j: float(commanded[t, k]) for k, j in enumerate(joints)}
        handler.set_dof_targets([{cfg.name: {"dof_pos_target": target}}])
        handler.simulate()
        q = handler.get_states(mode="tensor").robots[cfg.name].joint_pos[0]
        measured[t] = q.detach().cpu().float().numpy() if isinstance(q, torch.Tensor) else np.asarray(q, dtype=float)
    control_dt = None
    try:
        control_dt = float(handler.physics_dt) * int(decimation) if handler.physics_dt is not None else None
    except Exception:
        pass
    return TrackingTrace(
        simulator=sim, joints=list(joints), commanded=commanded, measured=measured, control_dt=control_dt
    )


@dataclass
class ParityRow:
    """One joint's cross-engine comparison."""

    joint: str
    kind: str
    """``hinge`` (radians) or ``slide`` (metres)."""
    max_diff: float
    rms_diff: float
    tracking_a: float
    tracking_b: float
    tol: float
    lag_a: int = 0
    lag_b: int = 0

    @property
    def ok(self) -> bool:
        """The two engines agree within ``tol`` at every step."""
        return bool(np.isfinite(self.max_diff) and self.max_diff <= self.tol)


@dataclass
class ParityReport:
    """Two backends tracking the same canned trajectory, compared joint by joint."""

    robot: str
    sim_a: str
    sim_b: str
    rows: list[ParityRow] = field(default_factory=list)
    steps: int = 0
    control_dt: float | None = None
    peak_velocity: float | None = None
    """Largest commanded joint speed in the trajectory (units per second), the regime the verdict holds for."""
    judged: bool = True
    """False for a floating-base robot: without a controller it falls, differently on each engine, so the differences
    are reported for the record and no verdict is given (the hold check treats a floating base the same way)."""

    @property
    def ok(self) -> bool:
        """Every joint agrees within its tolerance (a report that is not judged does not fail)."""
        if not self.judged:
            return True
        return bool(self.rows) and all(r.ok for r in self.rows)

    def table(self) -> str:
        """The report as a Markdown table."""
        lines = [
            f"robot {self.robot}: {self.sim_a} vs {self.sim_b}, {self.steps} control steps"
            + (f" of {self.control_dt * 1000:.1f} ms" if self.control_dt else "")
            + (f", peak commanded speed {self.peak_velocity:.2f}/s" if self.peak_velocity is not None else ""),
            f"| joint | unit | max diff | rms diff | tracking {self.sim_a} | tracking {self.sim_b} | lag (steps) | tol | result |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            unit = "rad" if r.kind == "hinge" else "m"
            lines.append(
                f"| {r.joint} | {unit} | {r.max_diff:.4f} | {r.rms_diff:.4f} | {r.tracking_a:.4f} | {r.tracking_b:.4f} "
                f"| {r.lag_a} / {r.lag_b} | {r.tol:.3f} | {('ok' if r.ok else 'FAIL') if self.judged else 'reported'} |"
            )
        if self.judged:
            lines.append(f"verdict: {'pass' if self.ok else 'FAIL'}")
        else:
            lines.append(
                "verdict: not judged (floating base: the robot falls without a controller, differently on each engine)"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "robot": self.robot,
            "sim_a": self.sim_a,
            "sim_b": self.sim_b,
            "steps": self.steps,
            "control_dt": self.control_dt,
            "peak_velocity": self.peak_velocity,
            "judged": self.judged,
            "ok": self.ok,
            "rows": [{**asdict(r), "ok": r.ok} for r in self.rows],
        }


def _joint_kinds(cfg, joints: list[str]) -> dict[str, str]:
    """``hinge`` / ``slide`` per joint from the MJCF when there is one; every joint is a hinge otherwise."""
    kinds = {j: "hinge" for j in joints}
    path = getattr(cfg, "mjcf_path", None)
    if not isinstance(path, str) or not path:
        return kinds
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(path)
        for j in joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid >= 0 and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                kinds[j] = "slide"
    except Exception:
        pass
    return kinds


def compare_traces(
    cfg, a: TrackingTrace, b: TrackingTrace, *, tol: float = 0.05, tol_slide: float = 0.005
) -> ParityReport:
    """Joint-by-joint difference between two traces of the same canned trajectory.

    ``tol`` is the largest allowed ``|q_a - q_b|`` for a hinge joint (radians), ``tol_slide`` for a slide joint
    (metres). Tracking error against the command is reported for each engine but not judged: it depends on the
    gains, and the hold check already covers a robot that cannot hold its pose.
    """
    if a.joints != b.joints:
        raise ValueError(f"joint order differs: {a.joints} vs {b.joints}")
    n = min(a.measured.shape[0], b.measured.shape[0])
    if not np.allclose(a.commanded[:n], b.commanded[:n]):
        raise ValueError("the two traces were not driven by the same command sequence")
    diff = a.measured[:n] - b.measured[:n]
    kinds = _joint_kinds(cfg, a.joints)
    report = ParityReport(
        robot=cfg.name,
        sim_a=a.simulator,
        sim_b=b.simulator,
        steps=n,
        control_dt=a.control_dt,
        judged=bool(getattr(cfg, "fix_base_link", True)),
    )
    if a.control_dt and n > 1:
        report.peak_velocity = float(np.max(np.abs(np.diff(a.commanded[:n], axis=0))) / a.control_dt)
    ta, tb = a.tracking_error, b.tracking_error
    la, lb = a.lag_steps(), b.lag_steps()
    for k, j in enumerate(a.joints):
        kind = kinds[j]
        report.rows.append(
            ParityRow(
                joint=j,
                kind=kind,
                max_diff=float(np.max(np.abs(diff[:, k]))) if n else float("nan"),
                rms_diff=float(np.sqrt(np.mean(diff[:, k] ** 2))) if n else float("nan"),
                tracking_a=float(ta[k]),
                tracking_b=float(tb[k]),
                tol=tol_slide if kind == "slide" else tol,
                lag_a=int(la[k]),
                lag_b=int(lb[k]),
            )
        )
    return report
