# Lighting across backends

`ScenarioCfg.lights` describes a light rig in UsdLux terms (`DistantLightCfg`, `SphereLightCfg`,
`DiskLightCfg`, `CylinderLightCfg`, `DomeLightCfg` in `metasim.scenario.lights`). Isaac Sim spawns
them as USD lights, Blender maps them onto its light types, and the MuJoCo backend translates them
into MJCF `<light>` elements. This page documents the MuJoCo translation: what each config becomes,
what was measured, and what does not match.

**Opt-in on MuJoCo and MJX:** set `SimParamCfg(mujoco_use_scenario_lights=True)`. `ScenarioCfg.lights`
defaults to one 500 cd distant light and most scenarios in the tree never set it, so translating it
for everyone would re-light every MuJoCo scene; by default MuJoCo keeps its camera headlight and the
lights the robot / scene MJCF files declare, and the rig is ignored as before. With the flag on, the
rig is the only illumination, as on a USD stage: the headlight and the asset-embedded lights are
switched off (the Franka MJCF ships a spot light at 2 m that would otherwise add to the rig).

```python
ScenarioCfg(lights=[...], sim_params=SimParamCfg(mujoco_use_scenario_lights=True), simulator="mujoco")
```

Other backends (SAPIEN, Genesis, PyBullet, Newton) do not consume `scenario.lights` yet.

## Translation rules (`metasim/sim/mujoco/lights.py`)

| config | MJCF | notes |
|---|---|---|
| `DistantLightCfg` | `type="directional"`, `dir` = `-Z` rotated by `polar` / `azimuth`, shadows on | |
| `SphereLightCfg` | `type="spot"` at `pos`, pointing `-Z`, `cutoff=90`, `exponent=0` | hemisphere below the bulb |
| `DiskLightCfg` | `type="spot"` at `pos`, `dir` = `-Z` rotated by `rot`, `cutoff=90`, `exponent=0` | orientation follows `rot` |
| `CylinderLightCfg` | as a sphere light at `pos` | no line lights in MuJoCo; no `normalize` field on this config |
| `DomeLightCfg` | global ambient in the dome colour plus a shadow-less zenith directional light of the same strength | HDR `texture_file` is reported and ignored |

Rules that produced a wrong picture when missed:

- **`type` is always written, and never `point`.** An MJCF `<light>` without `type` is a 45° spot with
  a `cos^10` lobe pointing down, so an untyped bulb becomes a narrow cone. `type="point"` renders
  *black* in MuJoCo's classic OpenGL renderer (the one `mujoco.Renderer` uses; the type serves the
  newer renderer), so bulbs are hemispherical spots. MuJoCo's spot cutoff saturates at a hemisphere:
  a bulb lights what is below it.
- **Disks are aimed with `rot`.** UsdLux disks emit along local `-Z`; a rig aimed sideways at a
  workpiece in Isaac Sim would point at the floor without applying the quaternion.
- **`normalize` is honoured (sphere, disk).** With `normalize=True` (the default) the intensity is
  the light's total output; with `normalize=False` it is per unit area and the output scales with
  the emitting area (`4πr²`, `πr²`).
- **A dome is not ambient alone.** Under a uniform sky a horizontal face receives twice the
  irradiance of a vertical one; global ambient would render a flat silhouette, so a dome also adds a
  shadow-less directional light from the zenith.
- **Names.** Duplicate names inside the rig are an error (USD needs one prim per light too); a
  clash with a light in a loaded scene / robot MJCF is suffixed `_scenario`; unnamed lights become
  `light_<index>` (Isaac Sim names them `DistantLight_<index>` etc., so name your lights if a
  randomizer refers to them).
- **Two warnings.** Intensities meant for lights metres away map to `diffuse > 1` under constant
  attenuation and clip to white; more than 7 rig lights exceed what the classic renderer evaluates
  (8 including the headlight). Both are logged with the fix; neither is silently clamped.
- **Indirect light is a global ambient term.** The fixed-function pipeline has no bounce light, so
  faces turned away from a light (or outside a spot cone, which in OpenGL also gates that light's
  own ambient) would be black where the path tracer shows fill. Every declared light adds
  `AMBIENT_RATIO` (0.18) of its diffuse to the global ambient.

## Known mismatch: distance fall-off

MuJoCo's default attenuation is constant (`attenuation="1 0 0"`), UsdLux falls off with `1/d²`.
Quadratic attenuation in MuJoCo cuts a 1.5 m bulb's contribution by ~65% in the test scene, which
would invalidate the exposure constants below (fitted at 1–3 m). Fall-off is therefore left constant:
rigs that rely on distance fall-off across a large scene read brighter on MuJoCo. If you need it,
set `attenuation` on the created `<light>` elements and re-fit the constants for your scene.

## Measurement

Same scene on both backends (Franka + a red cube + a blue sphere on a plane, one 256×256 pinhole
camera at (1.5, −1.5, 1.2)), foreground = the central 160×160 pixels, luminance 0–255. Isaac Sim 5.0
(RTX ray tracing, default tone mapping) vs `mujoco.Renderer` (MuJoCo 3.12, EGL). Before this
translation every rig rendered identically on MuJoCo (the rig was ignored): mean 116, p05 51.

| rig | Isaac Sim mean / p05 / dark px | MuJoCo mean / p05 / dark px |
|---|---:|---:|
| distant 2000 cd, polar 30° | 209 / 156 / 0% | 210 / 56 / 0.5% |
| sphere 1500 cd, r 0.1 at z 1.5 | 96 / 62 / 0% | 106 / 25 / 3.4% |
| disk 1500 cd, r 0.3 at z 1.5, down | 111 / 74 / 0% | 106 / 25 / 3.8% |
| disk 1500 cd, r 0.3 at z 0.5, sideways | 104 / 76 / 0% | 65 / 25 / 3.0% |
| dome 1000 cd | 195 / 135 / 0% | 191 / 71 / 0% |

"dark px" = fraction of foreground pixels below 20. Means match to within ~10% except the
side-mounted disk (MuJoCo 40% darker: the flat spot lobe still loses light that the path tracer
bounces off the floor). Percentiles do not match: Isaac Sim's tone mapping lifts shadows that the
fixed-function pipeline leaves dark, so the 5th percentile stays lower on MuJoCo even with the
ambient fill. Treat the translation as "same rig, comparable exposure", not pixel parity.

The fit script is a one-scene render on each backend; the constants live at the top of
`metasim/sim/mujoco/lights.py` and should be re-fitted when either renderer changes.
