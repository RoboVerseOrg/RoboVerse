"""Embodiment graph — a robot's morphology inferred from ``RobotCfg``.

A robot is modelled as a general, deterministic **embodiment graph**: a set of named
:class:`Chain` sub-groups (arms, grippers, mobile base, head, torso, legs) inferred from
the fields ``RobotCfg`` already carries — chiefly ``actuators[j].is_ee`` (the
authoritative gripper marker), plus ``gripper_joint_name`` / ``ee_body_name`` and
joint-name tokens. There is no fixed arm-count ceiling: k arms, legs, and a base are all
just chains.

The inference is pure, GPU-free, and unit-testable, and it **never invents a kind it did
not recognize**:

- a joint whose name matches no token, on a robot with no manipulator evidence, lands in
  an ``OTHER`` chain (still controllable in joint space, never silently relabelled
  ``arm``). ``cartpole`` -> ``other(2)``; a standalone ``shadow_hand`` -> ``other(24)``.
- unclassified joints become ``ARM`` **only** when the ``RobotCfg`` gives manipulator
  evidence (an ``ee_body_name``, an ``is_ee`` actuator, a ``gripper_joint_name``, or an
  arm-token joint), so ``kinova_gen3``'s ``joint_1..7`` are an arm but ``cartpole``'s
  ``slider_to_cart`` is not.
- a robot that *declares* a gripper it cannot expose (``gripper_open_q`` set but no
  gripper joint in ``joint_limits``, e.g. ``ur5e_2f85``) raises instead of silently
  dropping the gripper and making every pick task unsolvable.

A single-arm Franka -> ``(arm7, gripper2)``; an ALOHA-style bimanual robot ->
``(left_arm, right_arm, left_gripper, right_gripper)``; ``h1`` ->
``(left_arm4, right_arm4, left_leg5, right_leg5, torso1)``; ``go2`` -> four legs
(``front_left_leg`` … ``rear_right_leg``) — with no branch on arm count anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class ChainKind(str, Enum):
    ARM = "arm"
    GRIPPER = "gripper"
    BASE = "base"  # mobile base (x/y/yaw or wheels)
    HEAD = "head"
    TORSO = "torso"
    LEG = "leg"
    OTHER = "other"  # recognized by no token: kept controllable, never relabelled "arm"


# Kind is decided by matching **whole boundary-delimited name segments** (so "forehead"
# does NOT match "head", and "elbow" never hits a token) — see ``_segments``. Checked
# after ``is_ee``. GRIPPER tokens are used only as a fallback when no ``is_ee`` marker.
_GRIPPER_TOKENS = frozenset({"finger", "gripper", "knuckle", "claw", "thumb", "jaw", "grasp"})
_HEAD_TOKENS = frozenset({"head", "neck"})
_TORSO_TOKENS = frozenset({"torso", "spine", "chest"})  # "waist" excluded: an arm-base joint on viperx/widowx arms
_LEG_TOKENS = frozenset({"hip", "knee", "ankle", "thigh", "calf", "leg", "haa", "hfe", "kfe"})
_BASE_TOKENS = frozenset({"wheel", "mobile", "chassis"})
_BASE_AXES = frozenset({"x", "y", "theta", "yaw", "rot", "trans"})  # only with a "base" segment
# Positive arm evidence. Without one of these (or robot-level manipulator evidence, see
# ``_has_manipulator_evidence``) an unrecognized joint is OTHER, not ARM.
_ARM_TOKENS = frozenset({"arm", "shoulder", "elbow", "wrist", "forearm", "upperarm", "waist", "panda"})
_LEFT_SEGS = frozenset({"left", "l"})
_RIGHT_SEGS = frozenset({"right", "r"})
# Quadruped leg prefixes, both orderings: FL/FR/RL/RR (front/rear + left/right, e.g. go2)
# and LF/RF/LH/RH (left/right + fore/hind, e.g. anymal). The two vocabularies are disjoint,
# so one map resolves both without ambiguity.
_QUAD_SEGS = {
    "fl": "front_left",
    "fr": "front_right",
    "rl": "rear_left",
    "rr": "rear_right",
    "lf": "front_left",
    "rf": "front_right",
    "lh": "rear_left",
    "rh": "rear_right",
}

_SEG_RE = re.compile(r"[A-Za-z]+|\d+")


def _segments(name: str) -> frozenset[str]:
    """Lowercase alnum segments of the joint's *leaf* (after the last namespace ``/``)."""
    leaf = name.rsplit("/", 1)[-1]
    return frozenset(s.lower() for s in _SEG_RE.findall(leaf))


@dataclass(frozen=True)
class Chain:
    """One kinematic sub-group of a robot, in sorted-joint order."""

    name: str  # e.g. "left_arm", "right_gripper", "base"
    kind: ChainKind
    joint_names: tuple[str, ...]
    robot: str  # owning RobotCfg.name
    ee_body_name: str | None = None  # for ARM: link a pose/IK target attaches to (None if the cfg has none)

    @property
    def dof(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class Embodiment:
    """Named morphology of one or more robots — the source of truth for spec derivation."""

    robot_names: tuple[str, ...]
    chains: tuple[Chain, ...]

    def by_kind(self, kind: ChainKind) -> tuple[Chain, ...]:
        return tuple(c for c in self.chains if c.kind == kind)

    @property
    def arms(self) -> tuple[Chain, ...]:
        return self.by_kind(ChainKind.ARM)

    @property
    def grippers(self) -> tuple[Chain, ...]:
        return self.by_kind(ChainKind.GRIPPER)

    def chain(self, name: str) -> Chain:
        for c in self.chains:
            if c.name == name:
                return c
        raise KeyError(f"no chain named {name!r}; have {[c.name for c in self.chains]}")

    @property
    def dof(self) -> int:
        return sum(c.dof for c in self.chains)


@dataclass
class EmbodimentHints:
    """Optional overrides when inference is ambiguous.

    ``chains`` maps a chain name to an explicit ``(kind, joint_names)`` assignment for
    one robot; joints not covered fall back to inference. ``ee_body`` overrides an
    arm's ee link. Keyed by ``RobotCfg.name``.
    """

    chains: dict[str, dict[str, tuple[ChainKind, tuple[str, ...]]]] = field(default_factory=dict)
    ee_body: dict[str, str] = field(default_factory=dict)


def _side_of(joint: str, kind: ChainKind) -> str:
    """Side prefix (``""``, ``"left_"``, ``"right_"``, ``"front_left_"``, …) for a joint.

    Scope is the namespace prefix (before the last ``/``) when present, else the whole name's
    segments — so ``left`` inside a ``left_finger`` *leaf* of a ``right/...`` arm never flips the
    side. Single-letter ``l``/``r`` count only inside a namespace prefix (too ambiguous
    otherwise). LEG joints additionally resolve the quadruped ``FL``/``FR``/``RL``/``RR`` (and
    ``LF``/``RF``/``LH``/``RH``) prefixes, which name four legs rather than two sides.
    """
    if "/" in joint:
        scope = frozenset(s.lower() for s in _SEG_RE.findall(joint.rsplit("/", 1)[0]))
    else:
        scope = frozenset(s.lower() for s in _SEG_RE.findall(joint)) - {"l", "r"}
    if kind == ChainKind.LEG:
        quad = sorted(scope & set(_QUAD_SEGS))
        if len(quad) == 1:
            return f"{_QUAD_SEGS[quad[0]]}_"
    left, right = bool(scope & _LEFT_SEGS), bool(scope & _RIGHT_SEGS)
    if left and not right:
        return "left_"
    if right and not left:
        return "right_"
    return ""


def _token_kind(joint: str) -> ChainKind | None:
    """Kind implied by the joint's name tokens; ``None`` when no token matches."""
    segs = _segments(joint)
    if segs & _BASE_TOKENS or ("base" in segs and segs & _BASE_AXES):
        return ChainKind.BASE
    if segs & _HEAD_TOKENS:
        return ChainKind.HEAD
    if segs & _TORSO_TOKENS:
        return ChainKind.TORSO
    if segs & _LEG_TOKENS:
        return ChainKind.LEG
    if segs & _GRIPPER_TOKENS:
        return ChainKind.GRIPPER
    if segs & _ARM_TOKENS:
        return ChainKind.ARM
    return None


def _has_manipulator_evidence(robot, token_kinds: dict[str, ChainKind | None]) -> bool:
    """Does the ``RobotCfg`` actually say this robot is a manipulator?

    True when it declares an end-effector body, marks an ``is_ee`` actuator, names a gripper
    joint, or carries at least one arm-token joint. Only then may an *unrecognized* joint be
    called an arm — otherwise it stays :attr:`ChainKind.OTHER`.
    """
    actuators = getattr(robot, "actuators", None) or {}
    return bool(
        getattr(robot, "ee_body_name", None)
        or any(getattr(a, "is_ee", False) for a in actuators.values())
        or getattr(robot, "gripper_joint_name", None)
        or any(k == ChainKind.ARM for k in token_kinds.values())
    )


def _gripper_joints(robot, joints: list[str], token_kinds: dict[str, ChainKind | None], manipulator: bool) -> set[str]:
    """Gripper joints: ``is_ee``-marked actuators plus name-token/`gripper_joint_name` matches.

    Token matches count only on a robot with manipulator evidence — a "gripper" is defined
    relative to an arm, so a standalone dexterous hand (``allegro_hand``: ``thumb_joint_*`` but no
    arm, no ``ee_body_name``, no ``is_ee``) must not be split into a 4-DoF "gripper" plus a
    12-DoF "arm". There is no count-based ``gripper_open_q`` guess: that wrongly grabbed arm
    joints on robots whose gripper joints live outside ``joint_limits``.
    """
    actuators = getattr(robot, "actuators", None) or {}
    marked = {j for j, a in actuators.items() if getattr(a, "is_ee", False) and j in joints}
    if not manipulator:
        return marked
    named = {getattr(robot, "gripper_joint_name", None)} & set(joints)
    return marked | named | {j for j in joints if token_kinds[j] == ChainKind.GRIPPER}


def _check_declared_gripper(robot, rname: str, joints: list[str], gripper: set[str], override) -> None:
    """Raise when a robot *declares* a gripper whose joints the harness cannot control.

    ``ur5e_2f85`` / ``kinova_gen3`` set ``gripper_open_q`` but keep no gripper joint in
    ``joint_limits``. Dropping the gripper silently makes every pick task unsolvable by
    construction and reports ``success_rate 0.0`` with no error — a forbidden silent no-op.
    """
    if gripper or any(kind == ChainKind.GRIPPER for kind, _ in override.values()):
        return
    open_q = getattr(robot, "gripper_open_q", None)
    named = getattr(robot, "gripper_joint_name", None)
    if not open_q and not named:
        return
    declared = f"gripper_open_q={list(open_q)!r}" if open_q else f"gripper_joint_name={named!r}"
    raise ValueError(
        f"robot {rname!r} declares a gripper ({declared}) but none of its gripper joints are in "
        f"joint_limits (joints: {joints}), so the harness cannot command the gripper and every "
        "pick/place task on it would be unsolvable. Add the gripper joints to the RobotCfg's "
        "joint_limits/actuators, or pass EmbodimentHints(chains={"
        f"{rname!r}: {{'gripper': (ChainKind.GRIPPER, (<joint names>,))}}}}) to declare them explicitly."
    )


def _infer_one(robot, hints: EmbodimentHints | None) -> list[Chain]:
    rname = robot.name or "robot"
    joints = sorted((robot.joint_limits or {}).keys())
    if not joints:
        if getattr(robot, "ee_body_name", None) or getattr(robot, "actuators", None):
            logger.warning(
                f"[embodiment] robot {rname!r} has empty joint_limits (joints defined via "
                "URDF/USD?); embodiment is empty. Provide joint_limits or EmbodimentHints."
            )
        return []
    override = (hints.chains.get(rname, {}) if hints else {}) or {}
    assigned: set[str] = set()
    chains: list[Chain] = []
    ee_body = (hints.ee_body.get(rname) if hints else None) or getattr(robot, "ee_body_name", None)

    # 1) explicit overrides first
    for cname, (kind, jnames) in override.items():
        js = tuple(j for j in joints if j in set(jnames))
        if js:
            chains.append(Chain(cname, kind, js, rname, ee_body if kind == ChainKind.ARM else None))
            assigned.update(js)

    # 2) classify kind (side-independent) over the remainder
    token_kinds = {j: _token_kind(j) for j in joints}
    manipulator = _has_manipulator_evidence(robot, token_kinds)
    gripper = _gripper_joints(robot, joints, token_kinds, manipulator)
    _check_declared_gripper(robot, rname, joints, gripper, override)
    remaining = [j for j in joints if j not in assigned]
    # An unrecognized joint becomes ARM only with manipulator evidence; otherwise OTHER. It is
    # never dropped, so it stays controllable in joint space either way.
    fallback = ChainKind.ARM if manipulator else ChainKind.OTHER

    def _resolve(j: str) -> ChainKind:
        if j in gripper:
            return ChainKind.GRIPPER
        tk = token_kinds[j]
        # A GRIPPER *token* that did not make the gripper set means the robot has no arm to be a
        # gripper for (see _gripper_joints) — it is unrecognized, not a gripper.
        return fallback if tk is None or tk == ChainKind.GRIPPER else tk

    kind_of = {j: _resolve(j) for j in remaining}
    # "waist" is ambiguous: a humanoid's torso/waist vs a viperx/widowx arm's base joint (which is
    # why "waist" is an ARM token, not a TORSO one). Resolve by robot context — reclassify a
    # waist-segment ARM joint to TORSO only when the robot actually has legs (a legged/humanoid base).
    if any(k == ChainKind.LEG for k in kind_of.values()):
        for j in remaining:
            if kind_of[j] == ChainKind.ARM and "waist" in _segments(j):
                kind_of[j] = ChainKind.TORSO
    # A robot is "sided" (bimanual / legged) only if its ARM/LEG joints carry at least two distinct
    # sides. Otherwise (e.g. a single arm whose parallel gripper jaws are named left/right_finger)
    # side tokens are ignored, so those jaws stay in ONE gripper rather than splitting.
    limb_sides = {_side_of(j, kind_of[j]) for j in remaining if kind_of[j] in (ChainKind.ARM, ChainKind.LEG)}
    sided = len(limb_sides - {""}) >= 2

    def _sd(j: str) -> str:
        return _side_of(j, kind_of[j]) if sided else ""

    groups: dict[tuple[str, ChainKind], list[str]] = {}
    for j in remaining:
        groups.setdefault((_sd(j), kind_of[j]), []).append(j)

    for (side, kind), js in groups.items():
        chains.append(Chain(f"{side}{kind.value}", kind, tuple(js), rname, ee_body if kind == ChainKind.ARM else None))

    other = [c for c in chains if c.kind == ChainKind.OTHER]
    if other:
        named = [j for c in other for j in c.joint_names]
        logger.warning(
            f"[embodiment] robot {rname!r}: {len(named)} joint(s) matched no known chain token and no "
            f"manipulator evidence (ee_body_name / is_ee / gripper_joint_name / arm-token joint) was "
            f"found, so they are kept in an 'other' chain (joint-space controllable) rather than being "
            f"guessed into an arm: {named}. Pass EmbodimentHints to name them explicitly."
        )
    return chains


def infer_embodiment(robots, *, hints: EmbodimentHints | None = None) -> Embodiment:
    """Derive an :class:`Embodiment` from one or more ``RobotCfg`` objects.

    Deterministic and GPU-free. Multiple robots (e.g. two single-arm arms modelled as
    separate ``RobotCfg``s) each contribute their own chains; chain names are made
    unique across robots by prefixing the robot name when a collision occurs.

    Raises:
        ValueError: if no robot is given, or a robot declares a gripper whose joints are not in
            its ``joint_limits`` (see :func:`_check_declared_gripper`).
    """
    robots = list(robots)
    if not robots:
        raise ValueError("infer_embodiment needs at least one robot")
    all_chains: list[Chain] = []
    seen: set[str] = set()
    for robot in robots:
        for c in _infer_one(robot, hints):
            if c.name in seen:  # disambiguate cross-robot collisions
                c = Chain(f"{c.robot}.{c.name}", c.kind, c.joint_names, c.robot, c.ee_body_name)
            seen.add(c.name)
            all_chains.append(c)
    return Embodiment(tuple(r.name or "robot" for r in robots), tuple(all_chains))
