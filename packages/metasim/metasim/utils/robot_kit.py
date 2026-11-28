"""The robot kit: generate a ``RobotCfg`` from an MJCF, then put it through the self-check.

``python -m metasim robot add --mjcf <path> --name <robot>`` runs it. The MJCF is the one description every
RoboVerse backend can be held against (MuJoCo loads it, Newton loads it, the kinematics and limits checks read
it), so it is the source the generator reads:

- **joints**: every hinge and slide joint, its range (``joint_limits``), and whether a free joint makes the base
  floating (``fix_base_link``);
- **actuators**: the servo gains of the MJCF actuator that drives each joint (``kp`` from ``gainprm``, ``kv`` from
  ``biasprm``, the force range as ``effort_limit_sim``); a motor becomes an effort-controlled joint; a joint that
  no actuator drives but an ``<equality>`` couples to a driven one is passive (kept in the limits and defaults,
  left out of ``actuators``); a joint with neither gets a placeholder servo the generated file marks ``TODO``;
- **defaults**: ``--default joint=value`` first, then the ``home`` keyframe when the MJCF has one, otherwise zero
  clipped into the range and the open position for the gripper joints; a passive joint mirrors the joint it is
  coupled to;
- **the tool**: gripper joints are the slide joints or the joints named like a finger, jaw or gripper
  (``--gripper-joints`` overrides), the end-effector body is their common parent (``--ee-body`` overrides), and a
  site on that body named like a grasp or tool-centre point becomes ``curobo_tcp_rel_pos``.

Everything the generator had to guess is written into the file as a ``TODO`` comment and listed in
:attr:`RobotSpec.notes`; the self-check (:mod:`metasim.utils.robot_check`) then judges the result the same way it
judges a hand-written config.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_GRIPPER_NAME = re.compile(r"finger|gripper|jaw|claw|hand_joint|thumb|index|middle|ring|pinky", re.IGNORECASE)
_TCP_SITE_RANK = ("grasp", "tcp", "attachment", "tool", "ee", "end_effector", "pinch")
_PLACEHOLDER_GAINS = (100.0, 10.0)
"""Servo gains written for a joint the MJCF does not actuate; the generated file marks them ``TODO``."""


@dataclass
class JointSpec:
    """One hinge or slide joint of the MJCF and what the generator derived for it."""

    name: str
    kind: str
    """``"hinge"`` or ``"slide"``."""
    limited: bool
    low: float
    high: float
    default: float
    actuator: str | None = None
    """The MJCF actuator that drives the joint, if any."""
    control: str = "position"
    """``"position"`` for a servo, ``"effort"`` for a motor."""
    stiffness: float | None = None
    damping: float | None = None
    effort_limit: float | None = None
    is_gripper: bool = False
    passive: bool = False
    """No actuator; coupled to a driven joint by an ``<equality>`` (kept out of ``actuators``)."""
    guessed_gains: bool = False
    """No actuator and no coupling: the gains are placeholders."""


@dataclass
class RobotSpec:
    """What the generator read from an MJCF, and the notes a reviewer must read."""

    model_name: str
    mjcf_path: str
    joints: list[JointSpec] = field(default_factory=list)
    floating_base: bool = False
    ee_body: str | None = None
    ee_body_source: str = ""
    tcp_site: str | None = None
    tcp_offset: tuple[float, float, float] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def gripper_joints(self) -> list[str]:
        """The joints the config marks ``is_ee`` (in MJCF order, passive ones included)."""
        return [j.name for j in self.joints if j.is_gripper]

    @property
    def actuated(self) -> list[JointSpec]:
        """Joints that get an entry in ``actuators`` (everything but the passive ones)."""
        return [j for j in self.joints if not j.passive]


def _body_depth(model, body: int) -> int:
    depth = 0
    while body > 0:
        body = int(model.body_parentid[body])
        depth += 1
    return depth


def _common_ancestor(model, bodies: list[int]) -> int:
    def chain(b: int) -> list[int]:
        out = []
        while b > 0:
            out.append(b)
            b = int(model.body_parentid[b])
        return out

    common = set(chain(bodies[0]))
    for b in bodies[1:]:
        common &= set(chain(b))
    if not common:
        return 0
    return max(common, key=lambda b: _body_depth(model, b))


def inspect_mjcf(
    path: str,
    *,
    ee_body: str | None = None,
    gripper_joints: list[str] | None = None,
    home_key: str = "home",
    defaults: dict[str, float] | None = None,
) -> RobotSpec:
    """Read ``path`` with MuJoCo and derive the :class:`RobotSpec` the config is rendered from.

    ``defaults`` (joint -> position) overrides the home pose for the joints it names; a passive joint coupled by a
    linear ``<equality>`` to a driven one mirrors that joint's default.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)  # "world"; the model name is on the root element
    spec = RobotSpec(model_name=_model_name(path) or name, mjcf_path=path)

    # --- the home keyframe, if any
    key_qpos = None
    for k in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == home_key:
            key_qpos = np.array(model.key_qpos[k])
            break
    if key_qpos is None and model.nkey > 0:
        key_qpos = np.array(model.key_qpos[0])
        spec.notes.append(f"defaults come from the first keyframe (no key named {home_key!r})")
    if key_qpos is None:
        spec.notes.append("no keyframe: defaults are zero clipped into the joint range (gripper joints open)")

    # --- actuators by joint
    act_by_joint: dict[int, int] = {}
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jid = int(model.actuator_trnid[a, 0])
            if jid in act_by_joint:
                spec.notes.append(
                    f"joint {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)!r} has more than one actuator; "
                    "the first one's gains were taken"
                )
                continue
            act_by_joint[jid] = a

    # --- equality-coupled joints
    coupled: set[int] = set()
    couplings: list[
        tuple[int, int, list[float]]
    ] = []  # (joint1, joint2, polycoef): joint1 - q0_1 = a0 + a1 (joint2 - q0_2) + ...
    for e in range(model.neq):
        if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
            j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            coupled.update((j1, j2))
            couplings.append((j1, j2, [float(x) for x in model.eq_data[e, :5]]))
    overrides = dict(defaults or {})

    wanted_grippers = set(gripper_joints or [])
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        jtype = int(model.jnt_type[j])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            spec.floating_base = True
            continue
        if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
            spec.notes.append(f"ball joint {jname!r} skipped: RobotCfg describes one scalar per joint")
            continue
        kind = "slide" if jtype == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge"
        limited = bool(model.jnt_limited[j])
        low, high = (float(x) for x in model.jnt_range[j]) if limited else (-np.inf, np.inf)
        is_gripper = (
            jname in wanted_grippers if wanted_grippers else bool(kind == "slide" or _GRIPPER_NAME.search(jname))
        )
        js = JointSpec(name=jname, kind=kind, limited=limited, low=low, high=high, default=0.0, is_gripper=is_gripper)

        a = act_by_joint.get(j)
        if a is not None:
            js.actuator = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            gain0 = float(model.actuator_gainprm[a, 0])
            bias = [float(x) for x in model.actuator_biasprm[a, :3]]
            affine = int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
            if affine and abs(bias[1] + gain0) <= 1e-6 * max(1.0, abs(gain0)):
                js.control, js.stiffness, js.damping = "position", gain0, max(0.0, -bias[2])
            elif not affine and abs(gain0 - 1.0) < 1e-9:
                js.control = "effort"
                if model.actuator_ctrllimited[a]:
                    js.effort_limit = float(abs(model.actuator_ctrlrange[a, 1]))
            else:
                js.control, js.stiffness, js.damping = "position", gain0, max(0.0, -bias[2])
                spec.notes.append(f"actuator {js.actuator!r} is not a plain position servo; gains read as kp={gain0}")
            if model.actuator_forcelimited[a]:
                js.effort_limit = float(abs(model.actuator_forcerange[a, 1]))
            if js.actuator != jname:
                spec.notes.append(
                    f"actuator {js.actuator!r} drives joint {jname!r}: RoboVerse names an actuator after its joint, "
                    f"rename it in the MJCF or the MuJoCo handler will add a second servo on that joint"
                )
        elif j in coupled:
            js.passive = True
        else:
            js.stiffness, js.damping = _PLACEHOLDER_GAINS
            js.guessed_gains = True
            spec.notes.append(f"joint {jname!r} has no actuator: placeholder servo gains {_PLACEHOLDER_GAINS}")

        if jname in overrides:
            js.default = float(overrides.pop(jname))
        elif key_qpos is not None:
            js.default = float(key_qpos[int(model.jnt_qposadr[j])])
        elif is_gripper and limited:
            js.default = high  # open
        else:
            js.default = float(np.clip(0.0, low, high)) if limited else 0.0
        if limited and not (low - 1e-9 <= js.default <= high + 1e-9):
            spec.notes.append(f"keyframe puts {jname!r} at {js.default:.4f}, outside [{low:.4f}, {high:.4f}]; clipped")
            js.default = float(np.clip(js.default, low, high))
        spec.joints.append(js)

    if overrides:
        raise ValueError(f"--default names joints the MJCF does not have: {sorted(overrides)}")
    by_id = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name): j for j in spec.joints}
    for j1, j2, coef in couplings:
        a, b = by_id.get(j1), by_id.get(j2)
        if a is None or b is None or any(abs(c) > 1e-12 for c in coef[2:]):
            continue  # not two scalar joints, or not linear: left as read
        q1_0, q2_0 = (float(model.qpos0[int(model.jnt_qposadr[k])]) for k in (j1, j2))
        if a.passive and not b.passive:
            a.default = q1_0 + coef[0] + coef[1] * (b.default - q2_0)
        elif b.passive and not a.passive and abs(coef[1]) > 1e-12:
            b.default = q2_0 + (a.default - q1_0 - coef[0]) / coef[1]
        else:
            continue
        for js in (a, b):
            if js.passive and js.limited:
                js.default = float(np.clip(js.default, js.low, js.high))

    # --- the tool
    grippers = [j for j in spec.joints if j.is_gripper]
    if ee_body:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body) < 0:
            raise ValueError(f"{path} has no body {ee_body!r}")
        spec.ee_body, spec.ee_body_source = ee_body, "given"
    elif grippers:
        bodies = [int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, g.name)]) for g in grippers]
        parents = sorted({int(model.body_parentid[b]) for b in bodies})
        anc = parents[0] if len(parents) == 1 else _common_ancestor(model, bodies)
        spec.ee_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, anc)
        spec.ee_body_source = "the parent body of the gripper joints"
    elif spec.joints and not spec.floating_base:
        last = spec.joints[-1]
        bid = int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, last.name)])
        spec.ee_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        spec.ee_body_source = "the body of the last joint (no gripper joints found)"
        spec.notes.append(f"no gripper joints: ee_body_name {spec.ee_body!r} is the last joint's body; pass --ee-body")
    if spec.ee_body:
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body)
        sites = [
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s), s)
            for s in range(model.nsite)
            if model.site_bodyid[s] == eid
        ]

        def rank(item):
            sname = (item[0] or "").lower()
            for i, key in enumerate(_TCP_SITE_RANK):
                if key in sname:
                    return i
            return len(_TCP_SITE_RANK)

        named = sorted((s for s in sites if rank(s) < len(_TCP_SITE_RANK)), key=rank)
        if named:
            sname, sid = named[0]
            spec.tcp_site = sname
            spec.tcp_offset = tuple(round(float(x), 6) for x in model.site_pos[sid])
        else:
            spec.notes.append(f"no grasp/tcp site on {spec.ee_body!r}: curobo_tcp_rel_pos left for review")
    return spec


def _model_name(path: str) -> str | None:
    try:
        head = open(path, encoding="utf-8", errors="replace").read(4096)
    except OSError:
        return None
    m = re.search(r"<mujoco[^>]*\bmodel=\"([^\"]+)\"", head)
    return m.group(1) if m else None


def class_name_for(robot_name: str) -> str:
    """``so_arm100`` -> ``SoArm100Cfg``."""
    parts = re.split(r"[^0-9a-zA-Z]+", robot_name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p) + "Cfg"


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "float('inf')" if x > 0 else "float('-inf')"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0.0"


def render_robot_cfg(
    spec: RobotSpec,
    *,
    name: str,
    class_name: str | None = None,
    mjcf_path: str | None = None,
    usd_path: str | None = None,
    urdf_path: str | None = None,
    command: str | None = None,
) -> str:
    """The Python source of a ``RobotCfg`` subclass for ``spec`` (keyed by joint, in MJCF order)."""
    class_name = class_name or class_name_for(name)
    mjcf_path = mjcf_path or spec.mjcf_path
    today = _dt.date.today().isoformat()
    cmd = command or f"python -m metasim robot add --mjcf {mjcf_path} --name {name}"
    lines: list[str] = []
    w = lines.append
    w('"""' + f"{name}: a RobotCfg generated from its MJCF.")
    w("")
    w(f"Generated by ``{cmd}`` on {today} from ``{mjcf_path}`` (model ``{spec.model_name}``).")
    w("Every value below is what that file declares; the TODO comments mark what the generator had to guess.")
    w("Run ``python -m metasim robot check " + name + "`` after editing.")
    w('"""')
    w("")
    w("from __future__ import annotations")
    w("")
    w("from typing import Literal")
    w("")
    w("from metasim.scenario.robot import BaseActuatorCfg, RobotCfg")
    w("from metasim.utils import configclass")
    w("")
    w("")
    w("@configclass")
    w(f"class {class_name}(RobotCfg):")
    w(
        f'    """{spec.model_name}: {len(spec.joints)} joints, {len(spec.gripper_joints)} gripper joints; see the module docstring."""'
    )
    w("")
    w(f'    name: str = "{name}"')
    w(f"    num_joints: int = {len(spec.joints)}")
    w(f"    fix_base_link: bool = {not spec.floating_base}")
    if spec.floating_base:
        w(
            "    # the MJCF has a free joint: a floating base (legged / humanoid); the hold check reports, not judges, its drift"
        )
    w(f'    mjcf_path: str = "{mjcf_path}"')
    if usd_path:
        w(f'    usd_path: str = "{usd_path}"')
    else:
        w('    # usd_path: str = "roboverse_data/robots/<robot>/usd/<robot>.usd"  # TODO: Isaac Sim needs a USD')
    if urdf_path:
        w(f'    urdf_path: str = "{urdf_path}"')
    w("    enabled_gravity: bool = True")
    w("    enabled_self_collisions: bool = False")
    w("")
    w("    actuators: dict[str, BaseActuatorCfg] = {")
    for j in spec.actuated:
        kw = []
        if j.control == "position":
            kw.append(f"stiffness={_fmt(j.stiffness or 0.0)}")
            kw.append(f"damping={_fmt(j.damping or 0.0)}")
        if j.effort_limit is not None:
            kw.append(f"effort_limit_sim={_fmt(j.effort_limit)}")
        if j.is_gripper:
            kw.append("is_ee=True")
        tail = "  # TODO: no actuator in the MJCF, placeholder gains" if j.guessed_gains else ""
        tail = tail or (f"  # MJCF actuator {j.actuator!r}" if j.actuator and j.actuator != j.name else "")
        w(f'        "{j.name}": BaseActuatorCfg({", ".join(kw)}),{tail}')
    w("    }")
    w("    joint_limits: dict[str, tuple[float, float]] = {")
    for j in spec.joints:
        tail = (
            "  # passive: coupled by an <equality>, not driven"
            if j.passive
            else ("" if j.limited else "  # unlimited in the MJCF")
        )
        w(f'        "{j.name}": ({_fmt(j.low)}, {_fmt(j.high)}),{tail}')
    w("    }")
    w("    default_joint_positions: dict[str, float] = {")
    for j in spec.joints:
        w(f'        "{j.name}": {_fmt(j.default)},')
    w("    }")
    w('    control_type: dict[str, Literal["position", "effort"]] = {')
    for j in spec.actuated:
        w(f'        "{j.name}": "{j.control}",')
    w("    }")
    w("")
    if spec.ee_body:
        todo = "" if spec.ee_body_source in ("given", "the parent body of the gripper joints") else "  # TODO: verify"
        w(f'    ee_body_name: str = "{spec.ee_body}"{todo}  # {spec.ee_body_source}')
    else:
        w("    # ee_body_name: str = ...  # TODO: the tool body (not needed for a robot that is not an arm)")
    driven_grippers = [j.name for j in spec.joints if j.is_gripper and not j.passive]
    if driven_grippers:
        w("    ee_joint_names: list[str] = [" + ", ".join(f'"{g}"' for g in driven_grippers) + "]")
        opens = [_fmt(j.default) for j in spec.joints if j.is_gripper and not j.passive]
        closes = [
            _fmt(j.low if j.default >= (j.low + j.high) / 2 else j.high)
            for j in spec.joints
            if j.is_gripper and not j.passive
        ]
        w(f"    gripper_open_q: list[float] = [{', '.join(opens)}]")
        w(f"    gripper_close_q: list[float] = [{', '.join(closes)}]  # TODO: verify which end of the range closes")
    if spec.tcp_offset is not None:
        w(f"    curobo_tcp_rel_pos: tuple[float, float, float] = {spec.tcp_offset!r}  # site {spec.tcp_site!r}")
        w("    curobo_tcp_rel_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)")
    else:
        w(
            "    # curobo_tcp_rel_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # TODO: tool centre point in the ee body frame"
        )
    if spec.notes:
        w("")
        w("    # Generator notes:")
        for n in spec.notes:
            w(f"    # - {n}")
    return "\n".join(lines) + "\n"


def load_cfg_from_file(path: str, class_name: str) -> Any:
    """Import the generated module by path and instantiate ``class_name``."""
    modname = "_roboverse_robot_add_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    module_spec = importlib.util.spec_from_file_location(modname, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[modname] = module
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)()


def add_robot(
    *,
    mjcf: str,
    name: str,
    out: str,
    usd: str | None = None,
    urdf: str | None = None,
    ee_body: str | None = None,
    gripper_joints: list[str] | None = None,
    class_name: str | None = None,
    overwrite: bool = False,
    command: str | None = None,
    defaults: dict[str, float] | None = None,
) -> tuple[RobotSpec, str]:
    """Generate the config file for ``mjcf`` at ``out``; returns the spec and the class name written."""
    if os.path.exists(out) and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    spec = inspect_mjcf(mjcf, ee_body=ee_body, gripper_joints=gripper_joints, defaults=defaults)
    if not spec.joints:
        raise ValueError(f"{mjcf} declares no hinge or slide joint; nothing to drive")
    cls = class_name or class_name_for(name)
    source = render_robot_cfg(
        spec, name=name, class_name=cls, mjcf_path=mjcf, usd_path=usd, urdf_path=urdf, command=command
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(source)
    return spec, cls
