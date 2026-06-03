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


def remap_liberoplus_model(model_file: str, libero_assets: str, plus_assets: str) -> str:
    """Rebase a LIBERO-plus perturbed ``model_file`` to local assets.

    Same as :func:`remap_libero_model` (robosuite + chiliocosm roots, dm_control-safe)
    plus the LIBERO-plus overlay asset root — perturbed scenes embed absolute paths
    into ``.../LIBERO-plus/libero/libero/assets`` (light scenes, new-object meshes,
    textures). Rebase that root to ``plus_assets`` so the bundle is portable.
    """
    xml = remap_libero_model(model_file, libero_assets)
    xml = re.sub(r'file="[^"]*?/LIBERO-plus/libero/libero/assets', f'file="{plus_assets}', xml)
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


# turn-on/off ranges (FlatStove etc.): turn_on if qpos >= min(range); off if < max(off)
_TURNON_RANGES = {"flat_stove": [0.5, 2.1]}


def _region_base(region: str) -> str:
    return re.sub(r"_(\w+_)?(region|level)$", "", region)


def _artic_qpos(model, data, region: str):
    """qpos of the articulated (non-free) joint(s) for ``region``.

    A multi-drawer cabinet has one joint per drawer (``<obj>_<level>_level``); a
    region like ``<obj>_<level>_region`` must target ONLY that drawer's joint, not
    every joint on the object. We first try the level-specific joint
    (``region`` with ``region``→``level``); if none match we fall back to all of
    the object's articulated joints (single-DOF objects: microwave, window, …).
    """
    level_joint = re.sub(r"region$", "level", region)
    base = _region_base(region)
    specific, generic = [], []
    for j in range(model.njnt):
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        q = float(data.qpos[model.jnt_qposadr[j]])
        if level_joint in jn:
            specific.append(q)
        if base in jn:
            generic.append(q)
    return specific if specific else generic


def _is_open(model, data, region: str) -> bool:
    base = _region_base(region)
    cls = next((k for k in _OPEN_RANGES if k in base.lower()), None)
    rng, direction = _OPEN_RANGES.get(cls, ([-0.14, -0.14], "lt"))
    for q in _artic_qpos(model, data, region):
        if (q < max(rng)) if direction == "lt" else (q > min(rng)):
            return True
    return False


def _turn_on(model, data, region: str) -> bool:
    base = _region_base(region)
    cls = next((k for k in _TURNON_RANGES if k in base.lower()), None)
    rng = _TURNON_RANGES.get(cls, [0.5, 2.1])
    return any(q >= min(rng) for q in _artic_qpos(model, data, region))


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
    """LIBERO On(a,b) = b.check_ontop(a).

    If ``b`` is a region *site* (e.g. ``cabinet_top_side``) LIBERO uses the
    site's box containment (``SiteObject.under``; site contact is always True) —
    same as :func:`_in_region`. If ``b`` is a movable *object* it's the
    body-center on-top test (above + in contact + xy-aligned <0.03).
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, b)
    if sid >= 0:
        # SiteObject.under: obj within site-local xy half-size, z from surface up to +0.10
        bid = _bid(model, a) if _bid(model, a) >= 0 else _bid(model, a + "_main")
        if bid < 0:
            return False
        size = model.site_size[sid]
        delta = data.site_xmat[sid].reshape(3, 3) @ (data.xpos[bid] - data.site_xpos[sid])
        return bool(size[2] - 0.005 < delta[2] < size[2] + 0.10 and abs(delta[0]) < size[0] and abs(delta[1]) < size[1])
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
        elif p == "turnon" and len(args) == 1:
            ok = _turn_on(model, data, args[0])
        elif p == "turnoff" and len(args) == 1:
            ok = not _turn_on(model, data, args[0])
        else:
            return False
        if not ok:
            return False
    return True
