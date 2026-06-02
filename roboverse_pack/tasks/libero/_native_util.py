"""Self-contained helpers for the native MetaSim LIBERO tasks (no libero/robosuite).

* :func:`remap_libero_model` — rebase a LIBERO demo's embedded MJCF to local
  assets and make it dm_control-safe (so MetaSim's handler can load it).
* :func:`parse_goal` / :func:`check_bddl_success` — evaluate a BDDL ``(:goal ...)``
  natively from the MuJoCo state (In / On / Open / Close predicates).
"""

from __future__ import annotations

import os
import re

import mujoco
import numpy as np

_RS_ASSETS = "/venv/roboverse/lib/python3.11/site-packages/robosuite/models/assets"
# LIBERO articulated-object open ranges (libero/.../articulated_objects.py)
_OPEN_RANGES = {
    "wooden_cabinet": ([-0.16, -0.14], "lt"),
    "white_cabinet": ([-0.16, -0.14], "lt"),
    "short_cabinet": ([0.10, 0.16], "gt"),
    "short_fridge": ([2.0, 2.7], "gt"),
    "window": ([0.10, 0.16], "gt"),
    "microwave": ([-2.094, -1.3], "lt"),
    "flat_stove": ([-2.094, -1.3], "lt"),
}


def remap_libero_model(model_file: str, libero_assets: str) -> str:
    """Rebase by asset-root suffix (collector machine varies) + dm_control-safe."""
    xml = model_file
    xml = re.sub(r'file="[^"]*?/chiliocosm/assets', f'file="{libero_assets}', xml)
    xml = re.sub(r'file="[^"]*?/robosuite[^"]*?/(?:robosuite/)?models/assets', f'file="{_RS_ASSETS}', xml)
    xml = re.sub(r'<default class="main"\s*/>', "", xml)
    for ref in sorted(set(re.findall(r'file="([^"]+)"', xml))):
        if not os.path.exists(ref):
            cand = ref.replace("/mounts/", "/bases/")
            if os.path.exists(cand):
                xml = xml.replace(ref, cand)
    return xml


def parse_goal(bddl_text: str) -> list[tuple[str, list[str]]]:
    m = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", bddl_text, re.S)
    block = m.group(1) if m else bddl_text
    out = []
    for pred, args in re.findall(r"\((\w+)\s+([^()]*?)\)", block):
        if pred.lower() != "and":
            out.append((pred, args.split()))
    return out


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _in_region(model, data, obj: str, region: str) -> bool:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, region)
    bid = _bid(model, obj if _bid(model, obj) >= 0 else obj + "_main")
    if sid < 0 or bid < 0:
        return False
    sp = data.site_xpos[sid]
    half = np.abs(data.site_xmat[sid].reshape(3, 3) @ model.site_size[sid])
    return bool(np.all(np.abs(data.xpos[bid] - sp) < half))


def _is_open(model, data, region: str) -> bool:
    base = re.sub(r"_(\w+_)?(region|level)$", "", region)
    cls = next((k for k in _OPEN_RANGES if k in base.lower()), None)
    rng, direction = _OPEN_RANGES.get(cls, ([-0.14, -0.14], "lt"))
    for j in range(model.njnt):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if base in jn and any(k in jn for k in ("level", "joint", "cabinet", "door")):
            q = data.qpos[model.jnt_qposadr[j]]
            if (q < max(rng)) if direction == "lt" else (q > min(rng)):
                return True
    return False


def _contact(model, data, body_a: int, body_b: int) -> bool:
    if body_a < 0 or body_b < 0:
        return False
    for i in range(data.ncon):
        c = data.contact[i]
        ba, bb = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if {ba, bb} == {body_a, body_b}:
            return True
    return False


def _on(model, data, a: str, b: str) -> bool:
    """LIBERO On(a,b): a above b, in contact, xy-aligned (<0.03). (b.check_ontop(a))"""
    ba = _bid(model, a) if _bid(model, a) >= 0 else _bid(model, a + "_main")
    bb = _bid(model, b) if _bid(model, b) >= 0 else _bid(model, b + "_main")
    if ba < 0 or bb < 0:
        return False
    pa, pb = data.xpos[ba], data.xpos[bb]
    return bool(pa[2] >= pb[2] and float(np.linalg.norm(pa[:2] - pb[:2])) < 0.03 and _contact(model, data, ba, bb))


def check_bddl_success(model, data, goal_terms) -> bool:
    """True iff every BDDL goal predicate holds in the current MuJoCo state."""
    mujoco.mj_forward(model, data)
    for pred, args in goal_terms:
        p = pred.lower()
        if p == "in" and len(args) == 2:
            ok = _in_region(model, data, args[0], args[1])
        elif p == "on" and len(args) == 2:
            ok = _on(model, data, args[0], args[1])
        elif p == "open" and len(args) == 1:
            ok = _is_open(model, data, args[0])
        elif p == "close" and len(args) == 1:
            ok = not _is_open(model, data, args[0])
        else:
            return False
        if not ok:
            return False
    return True
