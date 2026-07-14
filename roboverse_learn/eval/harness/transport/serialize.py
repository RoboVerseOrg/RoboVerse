"""Typed wire (de)serialization for ObsBatch / ActionBatch / specs.

The spec travels with the data, so the receiver reconstructs a *typed* carrier and can
validate it — a schema mismatch is a typed error at connect/first-frame, not a
``KeyError`` deep in an adapter. msgpack + msgpack-numpy framing (lightweight,
zero-copy-ish for arrays).

Optional deps (remote path only)::  pip install msgpack msgpack-numpy
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..obs import ActionBatch, ObsBatch
from ..spec import ActionSpec, FieldSpec, ObsSpec, Space


def _to_tensor(v) -> torch.Tensor:
    # msgpack-numpy decodes to read-only arrays; copy so the tensor is writable (avoid UB).
    a = np.asarray(v)
    if not a.flags.writeable:
        a = a.copy()
    return torch.as_tensor(a)


def _codec():
    import msgpack
    import msgpack_numpy

    return msgpack, msgpack_numpy


def encode(obj: dict[str, Any]) -> bytes:
    msgpack, msgpack_numpy = _codec()

    def _default(o):
        # A torch.Tensor only reaches here from a non-tensor field (e.g. obs.task); fail with
        # an actionable message instead of msgpack's opaque "can not serialize 'Tensor'".
        if isinstance(o, torch.Tensor):
            raise TypeError(
                f"a torch.Tensor {tuple(o.shape)} was put in a non-tensor payload (task/meta); "
                "tensors must go in .tensors (they are serialized there) — task is for language/goal data"
            )
        return msgpack_numpy.encode(o)

    return msgpack.packb(obj, default=_default, use_bin_type=True)


def decode(data: bytes) -> dict[str, Any]:
    msgpack, msgpack_numpy = _codec()
    return msgpack.unpackb(data, raw=False, object_hook=msgpack_numpy.decode)


# ---- specs -----------------------------------------------------------------
def field_to_wire(f: FieldSpec) -> dict[str, Any]:
    return {
        "key": f.key,
        "space": f.space.value,
        "shape": list(f.shape),
        "dtype": f.dtype,
        "chain": f.chain,
        "frame": f.frame,
        "required": f.required,
    }


def field_from_wire(d: dict[str, Any]) -> FieldSpec:
    return FieldSpec(
        key=d["key"],
        space=Space(d["space"]),
        shape=tuple(d["shape"]),
        dtype=d.get("dtype", "float32"),
        chain=d.get("chain"),
        frame=d.get("frame"),
        required=d.get("required", True),
    )


def obs_spec_to_wire(s: ObsSpec) -> dict[str, Any]:
    return {"fields": [field_to_wire(f) for f in s.fields], "embodiment_id": s.embodiment_id}


def obs_spec_from_wire(d: dict[str, Any]) -> ObsSpec:
    return ObsSpec(tuple(field_from_wire(f) for f in d["fields"]), d.get("embodiment_id", ""))


def action_spec_to_wire(s: ActionSpec) -> dict[str, Any]:
    return {
        "fields": [field_to_wire(f) for f in s.fields],
        "control": s.control,
        "chunk_len": s.chunk_len,
        "embodiment_id": s.embodiment_id,
    }


def action_spec_from_wire(d: dict[str, Any]) -> ActionSpec:
    return ActionSpec(
        tuple(field_from_wire(f) for f in d["fields"]),
        control=d.get("control", "joint_pos"),
        chunk_len=d.get("chunk_len", 1),
        embodiment_id=d.get("embodiment_id", ""),
    )


# ---- carriers --------------------------------------------------------------
def obs_to_wire(obs: ObsBatch) -> dict[str, Any]:
    return {
        "spec": obs_spec_to_wire(obs.spec),
        "env_ids": obs.env_ids.detach().cpu().numpy(),
        "tensors": {k: v.detach().cpu().numpy() for k, v in obs.tensors.items()},
        "task": obs.task,
    }


def obs_from_wire(d: dict[str, Any]) -> ObsBatch:
    return ObsBatch(
        obs_spec_from_wire(d["spec"]),
        _to_tensor(d["env_ids"]),
        {k: _to_tensor(v) for k, v in d["tensors"].items()},
        d.get("task", {}) or {},
    )


def action_to_wire(action: ActionBatch) -> dict[str, Any]:
    return {
        "spec": action_spec_to_wire(action.spec),
        "env_ids": action.env_ids.detach().cpu().numpy(),
        "tensors": {k: v.detach().cpu().numpy() for k, v in action.tensors.items()},
    }


def action_from_wire(d: dict[str, Any]) -> ActionBatch:
    return ActionBatch(
        action_spec_from_wire(d["spec"]),
        _to_tensor(d["env_ids"]),
        {k: _to_tensor(v) for k, v in d["tensors"].items()},
    )
