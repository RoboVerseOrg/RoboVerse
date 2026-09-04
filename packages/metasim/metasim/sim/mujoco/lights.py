"""Translate ``scenario.lights`` into MJCF ``<light>`` elements for the MuJoCo-family backends.

Opt-in: ``SimParamCfg.mujoco_use_scenario_lights=True``. The light configs in
:mod:`metasim.scenario.lights` are written in UsdLux terms (Isaac Sim is the reference renderer),
``ScenarioCfg.lights`` defaults to one 500 cd distant light, and most scenarios in the tree never
set it. Translating that default for everyone would silently re-light every MuJoCo scene, so by
default MuJoCo keeps its own lighting (camera headlight plus whatever the robot / scene MJCF files
declare) and the rig is ignored, as before. With the flag on, the declared rig is the *only*
illumination, as on a USD stage.

MuJoCo's lighting model is the fixed-function OpenGL one, so the translation is a documented
approximation whose goal is that one rig looks *comparable* on both backends, not radiometrically
identical. The rules, each of which produced a wrong picture when missed:

- **``type`` is always explicit, and never ``point``.** An MJCF ``<light>`` without ``type`` is a
  *spot* light with a 45° cutoff and a ``cos^10`` fall-off pointing down, so an untyped bulb becomes
  a narrow cone. ``type="point"`` renders *black* in MuJoCo's classic (OpenGL) renderer, the one
  ``mujoco.Renderer`` uses (the type serves the newer renderer); sphere and cylinder lights are
  therefore hemispherical spots (90° cutoff, no angular fall-off). MuJoCo's spot cutoff saturates at
  a hemisphere: a bulb lights what is below it.
- **Disk lights are oriented by ``cfg.rot``.** UsdLux disks emit along their local ``-Z``; the same
  rig aimed at a workpiece in Isaac Sim would point at the floor in MuJoCo unless the quaternion is
  applied (and normalised: configs carry raw tuples).
- **Sphere and disk lights honour ``normalize``.** With UsdLux ``normalize=True`` the intensity is
  the light's total output regardless of its size; with ``normalize=False`` it is per unit area and
  the output grows with the area. MuJoCo has no area lights, so the factor is folded into
  ``diffuse``. (``CylinderLightCfg`` has no ``normalize`` field.)
- **A dome is ambient plus a shadow-less zenith light.** A uniform sky gives a horizontal face twice
  the irradiance of a vertical one; global ambient alone would render a flat silhouette.
- **Indirect light is a global ambient term.** The fixed-function pipeline has no bounce light, so
  faces turned away from a light (or outside a spot cone, which in OpenGL also gates that light's
  own ambient) would be black where the path tracer shows fill. Every light adds ``AMBIENT_RATIO``
  of its diffuse to ``visual/headlight/ambient``.
- **Known mismatch, on purpose:** MuJoCo's default attenuation is constant (``attenuation="1 0 0"``)
  while UsdLux falls off with ``1/d²``. Quadratic attenuation would invalidate the exposure
  constants below (fitted against renders at 1-3 m), so distance fall-off is left constant and
  rigs that rely on it read brighter on MuJoCo. A consequence: intensities meant for a light many
  metres away map to ``diffuse > 1`` and clip to white; that is warned about, not hidden.
- **The classic renderer uses at most 8 lights** (the headlight, kept for the ambient fill, is one
  of them): the 8th and later scenario lights are ignored by MuJoCo, so that is warned about too.

Exposure constants map UsdLux intensity to MuJoCo ``diffuse`` and were fitted by rendering the same
scene on both backends; the measurement is in ``docs/source/features/lighting.md``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger as log

from metasim.scenario.lights import (
    BaseLightCfg,
    CylinderLightCfg,
    DiskLightCfg,
    DistantLightCfg,
    DomeLightCfg,
    SphereLightCfg,
)

if TYPE_CHECKING:
    from dm_control import mjcf

#: UsdLux distant-light intensity -> MuJoCo ``diffuse``. Fitted so a 2000 cd distant light gives the
#: same foreground mean on both backends (Isaac Sim 5.0 RTX vs mujoco.Renderer).
DISTANT_INTENSITY_TO_DIFFUSE = 0.95 / 2000.0
#: UsdLux area-light (sphere / disk / cylinder, ``normalize=True``) intensity -> ``diffuse``; same fit
#: on a 1500 cd sphere / disk at 1-1.5 m.
AREA_INTENSITY_TO_DIFFUSE = 0.18 / 500.0
#: UsdLux dome-light intensity -> MuJoCo global ambient; the same amount again drives the zenith light.
DOME_INTENSITY_TO_AMBIENT = 0.45 / 1000.0
#: Specular is kept at a fixed fraction of diffuse (MuJoCo default light: diffuse 0.7, specular 0.3).
SPECULAR_RATIO = 0.3 / 0.7
#: Each light adds this fraction of its diffuse to the *global* ambient term (bounce-light stand-in).
AMBIENT_RATIO = 0.18
#: Lights the classic renderer evaluates, headlight included.
CLASSIC_RENDERER_MAX_LIGHTS = 8
#: Emission direction of an un-rotated UsdLux disk / distant light in its local frame.
_LOCAL_EMISSION_DIR = np.array([0.0, 0.0, -1.0])


def _rotate(quat_wxyz, vec) -> np.ndarray:
    """Rotate ``vec`` by the (w, x, y, z) quaternion ``quat_wxyz`` (normalised here: configs are raw tuples)."""
    q = np.asarray(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or norm == 0.0 or not math.isfinite(norm):
        raise ValueError(f"light rotation {quat_wxyz!r} is not a usable (w, x, y, z) quaternion")
    w, x, y, z = q / norm
    axis = np.array([x, y, z])
    v = np.asarray(vec, dtype=float)
    return v + 2.0 * np.cross(axis, np.cross(axis, v) + w * v)


def _rgb(color, scale: float) -> list[float]:
    return [max(0.0, float(c) * scale) for c in color[:3]]


def _is_named(cfg: BaseLightCfg) -> bool:
    return bool(cfg.name) and cfg.name != "light"


def _generated_name(cfg: BaseLightCfg, index: int) -> str:
    """Type-prefixed like Isaac Sim's ``/World/SphereLight_<i>`` (``spherelight_<i>`` here)."""
    return f"{type(cfg).__name__.removesuffix('Cfg').lower()}_{index}"


def _area_factor(cfg: SphereLightCfg | DiskLightCfg, area: float) -> float:
    """``normalize=False`` scales the output with the emitting area (UsdLux semantics)."""
    return 1.0 if cfg.normalize else area


def _add_fill(mjcf_model, color, scale: float) -> None:
    """Accumulate a light's bounce-light stand-in into the global ambient (``visual/headlight/ambient``)."""
    headlight = mjcf_model.visual.headlight
    current = list(headlight.ambient) if headlight.ambient is not None else [0.0, 0.0, 0.0]
    headlight.ambient = [c + a for c, a in zip(current, _rgb(color, scale), strict=True)]


def _finish(light, cfg: BaseLightCfg, diffuse_scale: float):
    light.diffuse = _rgb(cfg.color, diffuse_scale)
    light.specular = _rgb(cfg.color, diffuse_scale * SPECULAR_RATIO)
    light.ambient = [0.0, 0.0, 0.0]
    _add_fill(light.root, cfg.color, diffuse_scale * AMBIENT_RATIO)
    return light


def _hemispherical_spot(mjcf_model, name: str, pos, direction):
    """A spot with a 90° cutoff and no angular fall-off: MuJoCo's nearest thing to an area emitter."""
    light = mjcf_model.worldbody.add("light", name=name, type="spot", pos=[float(p) for p in pos])
    light.dir = [float(d) for d in direction]
    light.cutoff = 90.0
    light.exponent = 0.0
    return light


def add_distant_light(mjcf_model: mjcf.RootElement, cfg: DistantLightCfg, name: str):
    """Directional light; ``cfg.quat`` rotates the default ``-Z`` emission direction."""
    light = mjcf_model.worldbody.add("light", name=name, type="directional")
    light.dir = _rotate(cfg.quat, _LOCAL_EMISSION_DIR).tolist()
    light.castshadow = "true"
    return _finish(light, cfg, cfg.intensity * DISTANT_INTENSITY_TO_DIFFUSE)


def add_sphere_light(mjcf_model: mjcf.RootElement, cfg: SphereLightCfg, name: str):
    """Omnidirectional bulb as a downward hemispherical spot (see the module docstring for why)."""
    light = _hemispherical_spot(mjcf_model, name, cfg.pos, _LOCAL_EMISSION_DIR)
    scale = cfg.intensity * AREA_INTENSITY_TO_DIFFUSE * _area_factor(cfg, 4.0 * math.pi * cfg.radius**2)
    return _finish(light, cfg, scale)


def add_disk_light(mjcf_model: mjcf.RootElement, cfg: DiskLightCfg, name: str):
    """Hemispherical emitter along the disk's rotated ``-Z``.

    A Lambertian ``cos`` lobe (``exponent=1``) measured 45% darker than the RTX reference for a
    side-mounted panel while the downward panel matched, so the lobe is flat like the sphere light.
    """
    light = _hemispherical_spot(mjcf_model, name, cfg.pos, _rotate(cfg.rot, _LOCAL_EMISSION_DIR))
    scale = cfg.intensity * AREA_INTENSITY_TO_DIFFUSE * _area_factor(cfg, math.pi * cfg.radius**2)
    return _finish(light, cfg, scale)


def add_cylinder_light(mjcf_model: mjcf.RootElement, cfg: CylinderLightCfg, name: str):
    """A cylinder emits radially around its axis; MuJoCo has no line light, so it becomes the same
    downward hemispherical spot as a sphere light, centred on the cylinder. ``rot`` and ``length``
    are not representable and are reported when set.
    """
    light = _hemispherical_spot(mjcf_model, name, cfg.pos, _LOCAL_EMISSION_DIR)
    if tuple(float(c) for c in cfg.rot) != (1.0, 0.0, 0.0, 0.0) or float(cfg.length) != 1.0:
        log.warning(
            f"MuJoCo cylinder light {name}: `rot` and `length` are not representable (no line lights); it is a "
            f"downward hemispherical spot at {list(cfg.pos)}. Use a DiskLightCfg to aim a panel."
        )
    return _finish(light, cfg, cfg.intensity * AREA_INTENSITY_TO_DIFFUSE)


def add_dome_light(mjcf_model: mjcf.RootElement, cfg: DomeLightCfg, name: str):
    """Uniform sky light: global ambient plus a shadow-less directional light from the zenith, so
    horizontal faces read brighter than vertical ones as they do under a dome. An HDR
    ``texture_file`` has no MuJoCo equivalent and is reported.
    """
    if cfg.texture_file:
        log.warning(
            f"MuJoCo dome light {name!r}: HDR texture {cfg.texture_file!r} has no MuJoCo equivalent; "
            "using a uniform sky of the configured colour instead."
        )
    scale = cfg.intensity * DOME_INTENSITY_TO_AMBIENT
    _add_fill(mjcf_model, cfg.color, scale)
    light = mjcf_model.worldbody.add("light", name=name, type="directional")
    light.dir = _LOCAL_EMISSION_DIR.tolist()
    light.castshadow = "false"
    light.diffuse = _rgb(cfg.color, scale)
    light.specular = [0.0, 0.0, 0.0]
    light.ambient = [0.0, 0.0, 0.0]
    return light


_BUILDERS = (
    (DistantLightCfg, add_distant_light),
    (SphereLightCfg, add_sphere_light),
    (DiskLightCfg, add_disk_light),
    (CylinderLightCfg, add_cylinder_light),
    (DomeLightCfg, add_dome_light),
)


def _resolve_names(lights: list[BaseLightCfg], asset_names: set[str]) -> list[str]:
    """One MJCF identifier per light.

    Duplicate *explicit* names are an error (Isaac Sim needs one prim per light too). An explicit
    name that an asset light already owns, and every generated name, is suffixed until unique;
    generated names are type-prefixed so they cannot collide with a user's own naming.
    """
    explicit = [cfg.name for cfg in lights if _is_named(cfg)]
    dupes = sorted(name for name, n in Counter(explicit).items() if n > 1)
    if dupes:
        raise ValueError(
            f"scenario.lights has duplicate names {dupes}; give every light a unique `name` (the MJCF "
            "namespace and the USD stage both need one identifier per light)."
        )
    taken = set(asset_names) | set(explicit)
    resolved = []
    for i, cfg in enumerate(lights):
        if _is_named(cfg):
            candidate = cfg.name
            while candidate in asset_names:  # only an asset light can push an explicit name aside
                candidate = f"{candidate}_scenario"
        else:
            candidate = _generated_name(cfg, i)
            while candidate in taken:
                candidate = f"{candidate}_scenario"
        if candidate != (cfg.name if _is_named(cfg) else candidate):
            log.debug(f"MuJoCo: scenario light {cfg.name!r} renamed to {candidate!r} (the name is already taken)")
        taken.add(candidate)
        resolved.append(candidate)
    return resolved


def add_scenario_lights(mjcf_model: mjcf.RootElement, lights: list[BaseLightCfg]) -> list:
    """Write every light of ``lights`` into ``mjcf_model``; returns the created ``<light>`` elements.

    With an empty list the model is left untouched. Otherwise the declared rig is the only
    illumination: the headlight's diffuse and specular are zeroed (its ambient carries the fill
    term) and lights embedded in robot / scene MJCF files are deactivated. The caller decides
    whether a rig was declared at all (``SimParamCfg.mujoco_use_scenario_lights``).
    """
    if not lights:
        log.warning(
            "sim_params.mujoco_use_scenario_lights is on but scenario.lights is empty: MuJoCo keeps its default "
            "lighting (headlight + asset lights). Declare a rig or drop the flag."
        )
        return []
    headlight = mjcf_model.visual.headlight
    headlight.active = 1  # its ambient term carries the bounce-light fill
    headlight.diffuse = [0.0, 0.0, 0.0]
    headlight.specular = [0.0, 0.0, 0.0]
    headlight.ambient = [0.0, 0.0, 0.0]
    existing = list(mjcf_model.find_all("light"))
    # lights that come with an attached robot / object model (namespaced ``model/name``) are switched
    # off: the USD assets carry none. Lights authored in the scene MJCF itself (the model root) stay,
    # as Isaac Sim keeps a scene USD's lights and adds the rig on top.
    embedded = [light for light in existing if "/" in light.full_identifier]
    for light in embedded:
        light.active = "false"
    if embedded:
        log.info(
            f"MuJoCo: switched off {len(embedded)} light(s) embedded in robot / object models "
            f"({[light.full_identifier for light in embedded]}) in favour of the scenario rig"
        )
    names = _resolve_names(lights, {light.full_identifier for light in existing})
    created = []
    for cfg, name in zip(lights, names, strict=True):
        for cfg_type, build in _BUILDERS:
            if isinstance(cfg, cfg_type):
                created.append(build(mjcf_model, cfg, name))
                break
        else:
            raise TypeError(f"MuJoCo backend: unsupported light config {type(cfg).__name__} for light {cfg.name!r}")
    _warn_about_limits(mjcf_model, created)
    return created


def _warn_about_limits(mjcf_model, created: list) -> None:
    hot = [light.name for light in created if max(light.diffuse) > 1.0]
    raw_ambient = mjcf_model.visual.headlight.ambient
    ambient = [0.0, 0.0, 0.0] if raw_ambient is None else [float(a) for a in raw_ambient]
    if hot or max(ambient) > 1.0:
        log.warning(
            f"MuJoCo lights {hot} map to diffuse > 1 (global ambient {np.round(ambient, 2).tolist()}): the frame will clip "
            "to white. MuJoCo keeps constant attenuation, so an intensity meant for a light metres away is too much here; "
            "lower `intensity` (or set `normalize=True`) for the MuJoCo run."
        )
    if len(created) > CLASSIC_RENDERER_MAX_LIGHTS - 1:
        log.warning(
            f"scenario.lights has {len(created)} lights; MuJoCo's classic renderer evaluates at most "
            f"{CLASSIC_RENDERER_MAX_LIGHTS} including the headlight, so {[light.name for light in created[CLASSIC_RENDERER_MAX_LIGHTS - 1 :]]} "
            "will not light the frame."
        )


def scenario_lights_enabled(scenario) -> bool:
    """``True`` when the scenario asked MuJoCo to render its light rig (``SimParamCfg.mujoco_use_scenario_lights``)."""
    return bool(getattr(getattr(scenario, "sim_params", None), "mujoco_use_scenario_lights", False))
