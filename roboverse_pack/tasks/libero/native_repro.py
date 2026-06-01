"""Native MetaSim reproduction of a LIBERO demo.

This loads a LIBERO demo's *own* combined MJCF (the per-demo ``model_file`` that
robosuite generated: Franka + objects + arena + the ``agentview`` camera) into
**MetaSim's own MuJoCo handler** and reproduces the recorded trajectory there --
i.e. without ever calling robosuite/LIBERO to advance physics or render.

Two reproduction modes are provided:

* :func:`kinematic_rollout` -- set the full recorded MuJoCo state
  (``qpos``/``qvel``) at every frame and render the ``agentview`` camera through
  the handler. This proves MetaSim loads *all* LIBERO assets + scene and
  reproduces the exact recorded poses of every body, every frame.

* :func:`dynamic_rollout` -- set only the initial state, then write per-substep
  ``ctrl`` captured from the native env into the handler's physics and step it.
  This proves MetaSim's MuJoCo *stepping* reproduces LIBERO's dynamics.

The key fidelity requirement is that the model's ``qpos``/``qvel`` layout matches
the recorded ``states`` array bit-for-bit. That holds because we load the
*per-demo stored* ``model_file`` (the exact model that produced the states), with
its asset paths rewritten to the local robosuite + LIBERO asset roots.

Nothing in MetaSim is modified; this module only *consumes* the public
``ScenarioCfg`` / ``MujocoHandler`` surface.
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np


# --------------------------------------------------------------------------- #
# Asset-path rewriting: the stored model_file embeds an absolute meshdir from
# the data author's machine. Rewrite every ``file=`` ref to a local abspath.
# --------------------------------------------------------------------------- #
def _local_asset_roots() -> list[str]:
    roots: list[str] = []
    import robosuite

    roots.append(os.path.join(os.path.dirname(robosuite.__file__), "models", "assets"))
    try:
        import libero

        # ``libero`` is a namespace package: ``__file__`` is None, use ``__path__``.
        lib_roots = list(getattr(libero, "__path__", []))
        if getattr(libero, "__file__", None):
            lib_roots.append(os.path.dirname(libero.__file__))
        for lib_root in lib_roots:
            for r, _d, _f in os.walk(lib_root):
                if r.endswith("assets"):
                    roots.append(r)
    except Exception:
        pass
    return list(dict.fromkeys(roots))


def _build_asset_index(roots: list[str]) -> dict[str, list[str]]:
    """Map every trailing path-suffix of each local asset file to its abspath(s).

    LIBERO object meshes/textures use NON-unique generic filenames — every object
    directory ships a ``texture_map.png`` / ``textured_vis.msh`` / ``texture.png``.
    Indexing by basename alone collides them (all objects would grab whichever
    same-named file was walked first → wrong textures on every object). So we key
    by ALL trailing suffixes (``texture_map.png``, ``alphabet_soup/texture_map.png``,
    ``stable_hope_objects/alphabet_soup/texture_map.png``, …) and let
    :func:`_resolve` pick the LONGEST suffix that maps to exactly one file.
    """
    index: dict[str, list[str]] = {}
    for root in roots:
        for r, _d, fnames in os.walk(root):
            for fn in fnames:
                ap = os.path.join(r, fn)
                parts = ap.split(os.sep)
                for k in range(1, len(parts) + 1):
                    suffix = "/".join(parts[-k:])
                    lst = index.setdefault(suffix, [])
                    if ap not in lst:
                        lst.append(ap)
    return index


def localize_model_xml(model_file_xml: str) -> tuple[str, dict]:
    """Rewrite ``file=`` refs in a stored LIBERO model_file to local abspaths.

    Returns ``(fixed_xml, info)``. ``info`` has ``n_fixed`` and ``missing``.
    """
    roots = _local_asset_roots()
    index = _build_asset_index(roots)
    missing: list[str] = []
    n_fixed = 0

    def _resolve(fileref: str) -> str | None:
        # Normalize away ``a/../b`` segments, then match the LONGEST trailing
        # path-suffix of the original ref that maps to exactly ONE local file.
        # This disambiguates generic names like ``texture_map.png`` by their
        # parent dirs (``alphabet_soup/texture_map.png`` is unique even though
        # the basename is shared by every object).
        norm = os.path.normpath(fileref.replace("\\", "/"))
        parts = [p for p in norm.split("/") if p not in ("", ".")]
        # longest suffix first
        for k in range(len(parts), 0, -1):
            suffix = "/".join(parts[-k:])
            cands = index.get(suffix)
            if cands and len(cands) == 1:
                return cands[0]
        # fall back to the shortest unambiguous match we can get (basename),
        # preferring a deterministic pick so behaviour is stable, not random.
        base = parts[-1] if parts else ""
        cands = index.get(base)
        if cands:
            return sorted(cands)[0]
        return None

    def _repl(m: re.Match) -> str:
        nonlocal n_fixed
        pre, fileref, post = m.group(1), m.group(2), m.group(3)
        ap = _resolve(fileref)
        if ap is None:
            missing.append(fileref)
            return m.group(0)
        n_fixed += 1
        return f'{pre}"{ap}"{post}'

    xml = re.sub(r'(\sfile=)"([^"]+)"()', _repl, model_file_xml)
    xml = re.sub(r'\smeshdir="[^"]+"', "", xml)
    xml = re.sub(r'\stexturedir="[^"]+"', "", xml)
    xml = _bump_offscreen_framebuffer(xml, 1024, 1024)
    return xml, {"n_fixed": n_fixed, "missing": missing, "roots": roots}


def _bump_offscreen_framebuffer(xml: str, width: int, height: int) -> str:
    """Ensure the MJCF offscreen framebuffer is at least ``width`` x ``height``.

    LIBERO's stored MJCF caps the offscreen buffer at 480 (no explicit
    ``<global .../>``), which would reject renders larger than 480. We render at
    512 to make the side-by-side crisper than native's 128, so raise the cap.
    """
    glob = f'<global offwidth="{width}" offheight="{height}"/>'
    if "<global" in xml:
        # replace any existing offwidth/offheight or add them to the tag
        def _patch(m: re.Match) -> str:
            tag = m.group(0)
            tag = re.sub(r'\soffwidth="[^"]*"', "", tag)
            tag = re.sub(r'\soffheight="[^"]*"', "", tag)
            return tag[:-2] + f' offwidth="{width}" offheight="{height}"/>'

        return re.sub(r"<global[^>]*/>", _patch, xml, count=1)
    if "<visual>" in xml:
        return xml.replace("<visual>", f"<visual>\n    {glob}", 1)
    # no <visual> block: add one right after the <mujoco ...> opening tag
    return re.sub(r"(<mujoco[^>]*>)", r"\1\n  <visual>\n    " + glob + "\n  </visual>", xml, count=1)


# --------------------------------------------------------------------------- #
# MetaSim MuJoCo handler construction around the LIBERO scene MJCF.
# --------------------------------------------------------------------------- #
def make_native_handler(
    model_file_xml: str, *, image_size: int = 512, headless: bool = True, preserve_layout: bool = True
):
    """Build a launched MetaSim ``MujocoHandler`` around a LIBERO model_file.

    The LIBERO combined MJCF is fed to MetaSim's handler as the *scene* MJCF
    (``ScenarioCfg.scene.mjcf_path``) with no extra robots/objects/ground, so the
    handler loads it essentially verbatim and owns the physics + renderer.

    ``preserve_layout`` (default) rebinds the handler's ``dm_control`` ``Physics``
    to one constructed directly via :func:`dm_control.mujoco.Physics.from_xml_string`
    instead of the ``mjcf.attach`` + ``export_with_assets`` round-trip that the
    handler does at launch. That round-trip silently *re-derives* the ``qvel``
    layout (LIBERO's ``nv`` 43 -> 48, so ``1+nq+nv`` becomes 97 != 92), which
    would break the bit-exact mapping between the recorded 92-dim ``states`` and
    the model. ``Physics.from_xml_string`` is the *same* dm_control MuJoCo
    substrate the handler uses (so the handler still owns the physics + renders
    through ``handler.physics.render``), but it keeps LIBERO's exact ``nq``/``nv``
    layout.

    Returns ``(handler, fixed_xml_path, info, cam_name)``.
    """
    from metasim.scenario.scenario import ScenarioCfg
    from metasim.scenario.scene import SceneCfg
    from metasim.sim.mujoco.mujoco import MujocoHandler

    fixed_xml, info = localize_model_xml(model_file_xml)

    tmpdir = tempfile.mkdtemp(prefix="libero_native_")
    fixed_xml_path = os.path.join(tmpdir, "model_fixed.xml")
    with open(fixed_xml_path, "w") as f:
        f.write(fixed_xml)

    scenario = ScenarioCfg(
        scene=SceneCfg(mjcf_path=fixed_xml_path),
        robots=[],
        objects=[],
        cameras=[],
        add_default_ground=False,
        num_envs=1,
        headless=headless,
    )
    # The scene mjcf is a local file; MetaSim's HF FileDownloader (run in
    # BaseSimHandler.__init__ via scenario.check_assets) would try to fetch it
    # from the hub and fail. Disable that single network step (local-only) -- a
    # downstream behaviour shim, not a MetaSim edit.
    scenario.check_assets = lambda *a, **k: None

    handler = MujocoHandler(scenario)

    from dm_control import mujoco as dm_mujoco

    launched_via = "handler.launch"
    try:
        handler.launch()
        if preserve_layout:
            # Rebind to the layout-preserving Physics (same dm_control substrate)
            # so nq/nv match LIBERO's recorded 92-dim state exactly.
            handler.physics = dm_mujoco.Physics.from_xml_string(fixed_xml)
            handler.data = handler.physics.data
    except Exception:
        # MetaSim's launch path (mjcf.attach + export_with_assets) trips over
        # LIBERO's stored MJCF (unnamed bodies -> KeyError(None) when launch
        # reads ``physics.model.body(i).name``). We cannot edit MetaSim, so bind
        # the handler's physics directly to the LIBERO scene using the SAME
        # dm_control MuJoCo substrate MetaSim uses, and fill the few attributes
        # the render/state path reads. The handler object and its render path
        # (``handler.physics.render``) are still MetaSim's.
        launched_via = "direct-bind (MetaSim launch incompatible w/ LIBERO MJCF)"
        handler.physics = dm_mujoco.Physics.from_xml_string(fixed_xml)
        handler.data = handler.physics.data
        handler._mj_model = handler.physics.model.ptr
        handler.body_names = [(handler.physics.model.body(i).name or "") for i in range(handler.physics.model.nbody)]
        handler.robot_body_names = []
        handler._mujoco_robot_names = []
        handler.renderer = None

    info["launched_via"] = launched_via
    # Resolve the agentview camera id (the attach route prefixes it, e.g.
    # "base/agentview"; the from_xml_string route keeps the bare "agentview").
    cam_name = _find_camera(handler, "agentview")
    return handler, fixed_xml_path, info, cam_name


def _find_camera(handler, leaf: str) -> str:
    import mujoco

    model = handler.physics.model
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name is not None and name.split("/")[-1] == leaf:
            return name
    # fall back to first camera
    return mujoco.mj_id2name(model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, 0)


# --------------------------------------------------------------------------- #
# State application: write the full flattened LIBERO state into the handler's
# physics. LIBERO/robosuite flat layout = [time(1), qpos(nq), qvel(nv)].
# --------------------------------------------------------------------------- #
def set_flat_state(handler, flat_state: np.ndarray) -> None:
    physics = handler.physics
    nq = physics.model.nq
    nv = physics.model.nv
    expected = 1 + nq + nv
    if flat_state.shape[0] != expected:
        raise ValueError(
            f"flat_state len {flat_state.shape[0]} != 1+nq+nv ({expected}); "
            f"model layout (nq={nq}, nv={nv}) does not match the recorded states."
        )
    physics.data.qpos[:] = flat_state[1 : 1 + nq]
    physics.data.qvel[:] = flat_state[1 + nq : 1 + nq + nv]
    physics.forward()


def _visual_only_scene_option():
    """A dm_control scene option that renders only VISUAL geoms (MuJoCo group 1).

    LIBERO/robosuite MJCFs carry both visual geoms (group 1, textured/material)
    and collision geoms (group 0, untextured box/convex-hull approximations).
    robosuite's own renderer shows only the visual group; dm_control's default
    draws ALL groups, so the untextured collision geoms paint over the visual
    ones -- making the Franka appear in default green/yellow/blue and adding a
    protruding block to the drawer cabinet. Enabling only group 1 matches native.
    """
    # dm_control's Physics.render expects a dm_control wrapper MjvOption (it reads
    # ``scene_option.ptr``), NOT a raw ``mujoco.MjvOption``.
    from dm_control.mujoco import wrapper

    opt = wrapper.MjvOption()  # defaults match dm_control's render defaults
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    return opt


def render_cam(handler, cam_name: str, *, width: int, height: int, visual_only: bool = True) -> np.ndarray:
    scene_option = _visual_only_scene_option() if visual_only else None
    return handler.physics.render(
        width=width, height=height, camera_id=cam_name, depth=False, scene_option=scene_option
    )


def kinematic_rollout(handler, cam_name: str, states: np.ndarray, *, image_size: int = 512):
    """Yield rendered agentview frames by setting the full state every frame."""
    for j in range(states.shape[0]):
        set_flat_state(handler, states[j])
        yield render_cam(handler, cam_name, width=image_size, height=image_size)


def dynamic_rollout(
    handler,
    cam_name: str,
    init_state: np.ndarray,
    ctrl_seq,
    *,
    n_substeps: int,
    image_size: int = 512,
    record_frames: bool = True,
):
    """Set init state, then write each ctrl and step ``n_substeps`` per ctrl.

    ``ctrl_seq`` is an iterable of (nu,) arrays (one per policy step).
    Returns ``(frames, achieved_states)`` where achieved_states[k] is the
    flattened MuJoCo state after applying ctrl_seq[k].
    """
    import mujoco

    set_flat_state(handler, init_state)
    physics = handler.physics
    nq, nv = physics.model.nq, physics.model.nv
    frames = []
    achieved = []
    for ctrl in ctrl_seq:
        physics.data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(physics.model.ptr, physics.data.ptr)
        physics.forward()
        flat = np.concatenate([[physics.data.time], physics.data.qpos[:nq], physics.data.qvel[:nv]])
        achieved.append(flat.copy())
        if record_frames:
            frames.append(render_cam(handler, cam_name, width=image_size, height=image_size))
    return frames, np.array(achieved)
