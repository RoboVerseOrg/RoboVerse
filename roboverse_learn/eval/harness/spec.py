"""Typed observation / action specs — the contract that replaces string-key dicts.

The env↔policy contract is a set of typed :class:`FieldSpec`s with **canonical keys**
``"<chain>.<space>"`` (e.g. ``left_arm.joint_pos``, ``gripper.gripper``,
``head_cam.rgb``), derived from an :class:`~.embodiment.Embodiment`. One key scheme
covers every arm count, so there is no per-embodiment key convention to get wrong.

Compatibility between an env's :class:`ObsSpec` and what a policy needs is computed
once, at connect time, as a typed :class:`SpecMatch` — either an adaptation plan or an
actionable error — so a mismatch surfaces before the rollout starts rather than as a
``KeyError`` mid-episode.

Pure and GPU-free: shapes come from chain DOFs and camera configs, not tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .embodiment import ChainKind, Embodiment


class Space(str, Enum):
    JOINT_POS = "joint_pos"
    JOINT_VEL = "joint_vel"
    EE_POSE = "ee_pose"  # (7,): pos(3) + wxyz quat(4)
    GRIPPER = "gripper"
    RGB = "rgb"
    DEPTH = "depth"
    TASK = "task"  # non-tensor payload (ObsBatch.task), e.g. a language instruction


EE_POSE_DIM = 7
# Controls the harness can derive an action spec for. "joint_pos" is the only one the
# EnvAdapter applies today ("ee_pose" needs cuRobo IK); "joint_vel"/"effort" are rejected
# rather than silently handed back a JOINT_POS spec.
SUPPORTED_CONTROLS = ("joint_pos", "ee_pose")


@dataclass(frozen=True)
class FieldSpec:
    """One named tensor field. ``shape`` is per-env (carriers prepend num_envs)."""

    key: str  # canonical "<chain>.<space>" (or "<cam>.rgb", "task.language")
    space: Space
    shape: tuple[int, ...]
    dtype: str = "float32"
    chain: str | None = None  # links field -> Embodiment.chain(name)
    frame: str | None = None  # "world" | "robot_base" | camera name; None = unspecified
    required: bool = True


@dataclass(frozen=True)
class ObsSpec:
    """The complete typed observation contract."""

    fields: tuple[FieldSpec, ...]
    embodiment_id: str = ""

    def keys(self) -> tuple[str, ...]:
        return tuple(f.key for f in self.fields)

    def field(self, key: str) -> FieldSpec:
        for f in self.fields:
            if f.key == key:
                return f
        raise KeyError(f"no field {key!r}; have {list(self.keys())}")

    def subset(self, keys: Iterable[str]) -> ObsSpec:
        want = set(keys)
        return ObsSpec(tuple(f for f in self.fields if f.key in want), self.embodiment_id)

    def compatible_with(self, needs: ObsSpec) -> SpecMatch:
        """Can an env producing *self* satisfy a policy that ``needs`` this spec?"""
        return _match(producer=self, consumer=needs)


@dataclass(frozen=True)
class ActionSpec:
    """What a policy may emit; drives validation + backend translation.

    There is deliberately no ``frequency`` field: the runner steps the policy at the sim rate
    and applies no decimation, so advertising a control-Hz the harness ignores would be a
    contract field that lies. Chunked policies express their rate through ``chunk_len`` and the
    :mod:`.chunking` scheduler/ensembler.
    """

    fields: tuple[FieldSpec, ...]
    control: str = "joint_pos"  # one of SUPPORTED_CONTROLS
    chunk_len: int = 1  # >1 => policy returns an action chunk
    embodiment_id: str = ""

    def keys(self) -> tuple[str, ...]:
        return tuple(f.key for f in self.fields)

    def field(self, key: str) -> FieldSpec:
        for f in self.fields:
            if f.key == key:
                return f
        raise KeyError(f"no field {key!r}; have {list(self.keys())}")

    def compatible_with(self, produces: ActionSpec) -> SpecMatch:
        """Can an env expecting *self* accept the actions a policy ``produces``?

        Checks the declared ``control`` and every field the policy advertises (key, space,
        shape). A policy that advertises no fields (the bundled/scripted adapters, which learn
        the concrete spec via :meth:`bind`) is trivially compatible.
        """
        if not produces.fields:
            return SpecMatch(True)
        errors: list[str] = []
        if produces.control != self.control:
            errors.append(
                f"policy produces control={produces.control!r} but the env's derived action spec is "
                f"control={self.control!r} (pass control={produces.control!r} to evaluate(), or emit "
                f"{self.control!r} actions)"
            )
        mine = {f.key: f for f in self.fields}
        for p in produces.fields:
            m = mine.get(p.key)
            if m is None:
                errors.append(f"policy produces unknown action field {p.key!r} (env expects {list(mine)})")
            elif p.space != m.space:
                errors.append(f"{p.key!r} space {p.space.value} != expected {m.space.value}")
            elif tuple(p.shape) != tuple(m.shape):
                errors.append(f"{p.key!r} shape {tuple(p.shape)} != expected {tuple(m.shape)}")
        missing = [k for k in mine if k not in {f.key for f in produces.fields}]
        if missing:
            errors.append(f"policy produces no action for {missing} (the env expects every chain to be commanded)")
        return SpecMatch(not errors, (), tuple(errors))


# --------------------------------------------------------------------- negotiation
@dataclass(frozen=True)
class FieldOp:
    """One step of an adaptation plan for a single consumer field."""

    key: str
    # "take"          present, exact match
    # "cast"          present, dtype differs -> transport/adapter must convert (note = "a->b")
    # "drop_optional" absent but optional -> policy will not receive it
    op: str
    note: str = ""


@dataclass(frozen=True)
class SpecMatch:
    """Result of comparing a producer spec to a consumer's needs."""

    ok: bool
    plan: tuple[FieldOp, ...] = ()
    errors: tuple[str, ...] = ()

    def raise_if_bad(self) -> SpecMatch:
        if not self.ok:
            raise ValueError("incompatible specs:\n  " + "\n  ".join(self.errors))
        return self


def _match(*, producer: ObsSpec, consumer: ObsSpec) -> SpecMatch:
    prod = {f.key: f for f in producer.fields}
    plan: list[FieldOp] = []
    errors: list[str] = []
    for c in consumer.fields:
        p = prod.get(c.key)
        if p is None:
            if c.required:
                errors.append(f"missing required field {c.key!r} (producer has {list(prod)})")
            else:
                plan.append(FieldOp(c.key, "drop_optional", "absent, optional"))
            continue
        if p.space != c.space:
            errors.append(f"{c.key!r} space {p.space.value} != needed {c.space.value}")
        elif tuple(p.shape) != tuple(c.shape):
            errors.append(f"{c.key!r} shape {tuple(p.shape)} != needed {tuple(c.shape)}")
        elif p.frame and c.frame and p.frame != c.frame:
            # Differing *explicit* pose conventions need a transform we do not apply -> fail fast.
            # A None frame on either side is a wildcard and does NOT error; `derive_obs_spec` sets
            # frame="world" on EE_POSE fields, so a policy that asks for frame="robot_base" gets
            # this error instead of silently receiving world-frame poses.
            errors.append(f"{c.key!r} frame {p.frame!r} != needed {c.frame!r} (no auto-transform)")
        elif p.dtype != c.dtype:
            # dtype mismatch is a legitimate adaptation (e.g. uint8 image -> float).
            # WARNING: the returned `plan` is currently advisory only — NOTHING executes these
            # FieldOps. A uint8->float32 "cast" is recorded here, negotiation passes, and the
            # policy then receives raw uint8 in [0,255] where it expected float in [0,1].
            # Either apply the plan in the obs path or downgrade a dtype mismatch to an error.
            plan.append(FieldOp(c.key, "cast", f"{p.dtype}->{c.dtype}"))
        else:
            plan.append(FieldOp(c.key, "take"))
    return SpecMatch(not errors, tuple(plan), tuple(errors))


# --------------------------------------------------------------------- derivation
def derive_obs_spec(
    emb: Embodiment,
    *,
    cameras: Iterable[tuple[str, tuple[int, int]]] | None = None,
    include_ee_pose: bool = True,
    include_language: bool = False,
) -> ObsSpec:
    """Build a canonical :class:`ObsSpec` from an embodiment (+ optional cameras).

    ``cameras`` is an iterable of ``(name, (H, W))``. Proprio fields: per arm a ``joint_pos``
    (plus an ``ee_pose`` when requested *and* the arm's ``RobotCfg`` names an ``ee_body_name``
    — an arm with no ee link cannot produce a pose, so the field is not emitted rather than
    filled with a constant identity pose); per gripper a ``gripper`` field. ``include_language``
    adds the optional non-tensor ``task.language`` field (carried in ``ObsBatch.task``); the
    :class:`~.env_adapter.EnvAdapter` sets it only for tasks that expose an instruction.
    """
    fields: list[FieldSpec] = []
    for c in emb.chains:
        if c.kind == ChainKind.GRIPPER:
            fields.append(FieldSpec(f"{c.name}.{Space.GRIPPER.value}", Space.GRIPPER, (c.dof,), chain=c.name))
        else:
            fields.append(FieldSpec(f"{c.name}.{Space.JOINT_POS.value}", Space.JOINT_POS, (c.dof,), chain=c.name))
            if include_ee_pose and c.kind == ChainKind.ARM and c.ee_body_name:
                fields.append(
                    FieldSpec(
                        f"{c.name}.{Space.EE_POSE.value}",
                        Space.EE_POSE,
                        (EE_POSE_DIM,),
                        chain=c.name,
                        frame="world",
                    )
                )
    for name, (h, w) in cameras or ():
        fields.append(FieldSpec(f"{name}.{Space.RGB.value}", Space.RGB, (h, w, 3), dtype="uint8", frame=name))
    if include_language:
        fields.append(FieldSpec("task.language", Space.TASK, (), dtype="str", required=False))
    return ObsSpec(tuple(fields), _emb_id(emb))


def derive_action_spec(emb: Embodiment, *, control: str = "joint_pos", chunk_len: int = 1) -> ActionSpec:
    """Build a canonical :class:`ActionSpec`. Arm control key follows ``control``.

    Raises:
        ValueError: on a control outside :data:`SUPPORTED_CONTROLS` (``joint_vel``/``effort``
            would otherwise return a spec whose fields are all ``JOINT_POS``), or on
            ``control="ee_pose"`` for an arm whose ``RobotCfg`` has no ``ee_body_name``.
    """
    if control not in SUPPORTED_CONTROLS:
        raise ValueError(
            f"unsupported control {control!r}; supported: {list(SUPPORTED_CONTROLS)}. "
            "joint_vel/effort control is not implemented (the derived fields would be joint_pos)."
        )
    arm_space = Space.EE_POSE if control == "ee_pose" else Space.JOINT_POS
    fields: list[FieldSpec] = []
    for c in emb.chains:
        if c.kind == ChainKind.GRIPPER:
            fields.append(FieldSpec(f"{c.name}.{Space.GRIPPER.value}", Space.GRIPPER, (c.dof,), chain=c.name))
        elif c.kind == ChainKind.ARM:
            if control == "ee_pose" and not c.ee_body_name:
                raise ValueError(
                    f"control='ee_pose' but arm chain {c.name!r} of robot {c.robot!r} has no ee_body_name; "
                    "set RobotCfg.ee_body_name (or EmbodimentHints.ee_body) or use control='joint_pos'."
                )
            dim = EE_POSE_DIM if control == "ee_pose" else c.dof
            fields.append(FieldSpec(f"{c.name}.{arm_space.value}", arm_space, (dim,), chain=c.name))
        else:
            # non-arm actuated chains (base/torso/legs/other) controlled in joint space
            fields.append(FieldSpec(f"{c.name}.{Space.JOINT_POS.value}", Space.JOINT_POS, (c.dof,), chain=c.name))
    return ActionSpec(tuple(fields), control, chunk_len, _emb_id(emb))


def _emb_id(emb: Embodiment) -> str:
    return "+".join(emb.robot_names) + ":" + ",".join(f"{c.name}#{c.dof}" for c in emb.chains)
