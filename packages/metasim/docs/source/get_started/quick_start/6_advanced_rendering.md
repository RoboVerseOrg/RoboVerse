# Tutorial 6: Advanced Rendering

**Objective**: Learn how to use different rendering techniques for varying quality/performance trade-offs.

**What you'll learn**:
- Rasterization vs Ray Tracing vs Path Tracing
- Configuring render modes in MetaSim
- When to use each rendering technique

**Prerequisites**: Completed [Tutorial 5: Hybrid Simulation](5_hybrid_sim)

**Estimated time**: 20 minutes

---

MetaSim supports multiple rendering techniques through Isaac Sim's rendering backend. Choose based on your quality vs performance requirements.

| Technique | `--render.mode` | Quality | Speed | Use Case |
|-----------|-----------------|---------|-------|----------|
| Rasterization | `rasterization` | — | — | Unsupported by Isaac Sim |
| Ray Tracing (RTX Real-Time, legacy) | `raytracing` | Great | Medium | Validation, demos |
| Real-Time Path Tracing (RTX Real-Time 2.0) | `realtime_pathtracing` | Great | Medium | Interactive path-traced rendering (recent Isaac Sim) |
| Path Tracing (RTX Interactive) | `pathtracing` | Best | Slow | Final renders, sim2real |

> `realtime_pathtracing` selects Isaac Sim's "RTX - Real-Time 2.0" renderer (`/rtx/rendermode=RealTimePathTracing`). The renderer must be **registered at Kit boot** — it only joins the render-mode list when `/rtx-transient/rt2Enabled` is on (derived at startup from the persistent preference `/persistent/rtx/modes/rt2/enabled`); without registration every `/rtx/rendermode` write is silently refused. MetaSim injects the registration flags into Kit's boot argv when it launches the SimulationApp itself, engages the mode, and verifies via readback (once registered, the mode also switches in/out at runtime). If you hand the handler an already-running `simulation_app`, launch it with `--/persistent/rtx/modes/rt2/enabled=true --/rtx-transient/rt2Enabled=true` yourself. On builds without RTX Real-Time 2.0 MetaSim raises an explicit error so you can fall back to `raytracing` or `pathtracing`.

## Running the tutorial

`examples/6_advanced_rendering.py` starts with cubes, a sphere, a background wall,
a shared collision/render floor, key/fill lights and a camera. No downloaded robot or object assets are needed by
default. The old asset-heavy example is now opt-in with `--scene assets --robot franka`.
Install both editable packages as described in the installation guide first.

The default `--appearance studio` uses restrained linear-RGB palettes, matte
supports, a metal sphere and Blender AgX highlight rolloff. `--appearance stress`
uses broader independent material colors. These are example distributions; calibrate
them against your intended cameras, lighting and material collection.

Generate four deterministic recipes without launching any simulator:

```bash
python examples/6_advanced_rendering.py --recipes-only --variants 4 --seed 0 \
  --output examples/output/recipes
```

Render the same primitive scene with Blender Cycles on CPU (Blender uses Cycles
regardless of `render.mode`):

```bash
python examples/6_advanced_rendering.py --sim blender --render.device CPU \
  --render.samples 32 --width 256 --height 256 --variants 4 \
  --output examples/output/blender_primitives
```

Use MuJoCo for physics and Blender for multi-view, episode-consistent videos:

```bash
python examples/6_advanced_rendering.py --sim blender --physics mujoco \
  --render.device CPU --render.samples 64 --multiview --frames 48 \
  --output examples/output/mujoco_blender
```

Run these commands in an environment with both MuJoCo and the selected renderer.
One recipe is applied per video, then the physics scene advances. Each variant
restores the same initial physical state. `--steps-per-frame` controls simulation
steps between frames. Playback defaults to the actual capture frequency
`1 / (steps_per_frame * decimation * physics_dt)`; `--fps` explicitly overrides
playback speed without changing simulation time.
Intermediate control intervals run through `HybridSimHandler.simulate_steps`, so
only the final state is rendered. The manifest records both time axes explicitly.

Isaac Sim RTX path tracing, with multiple environments on one stage:

```bash
python examples/6_advanced_rendering.py --sim isaacsim --physics mujoco \
  --num-envs 4 --render.mode pathtracing --render.samples 128 --multiview \
  --output examples/output/mujoco_isaacsim
```

The demo applies the same recipe to all envs; use `VisualRandomizer.apply(...,
env_ids=[...])` for materials/cameras-only per-env variants. Lights and HDRIs are
shared across a stage. Blender supports one env per process.

For asset-based rendering, prepare the repository's BBQ-sauce and optional Franka
assets using the normal asset workflow. Supply your own HDRI/PBR files:

```bash
python examples/6_advanced_rendering.py --sim blender --scene assets --robot franka \
  --hdri /assets/studio.exr --textures.base-color /assets/surface/albedo.png \
  --textures.roughness /assets/surface/roughness.png \
  --textures.normal /assets/surface/normal_gl.png \
  --render.samples 256 --render.view-transform AgX \
  --output examples/output/assets_blender
```

The texture set targets the BBQ-sauce object's authored UVs. Unselected robot
materials remain unchanged. Replace these paths with existing, suitable assets;
there is no implicit download or silent HDRI fallback. Environment color,
background-wall material, lights and camera calibration vary. The studio preset
keeps the flat world color neutral when no HDRI is provided.
Scene geometry/collisions stay fixed.

Textured floors and backdrops come from `--material-library DIR`, a directory of
`<name>_BaseColor.png` files with optional sibling `<name>_Normal.png` (OpenGL) and
`<name>_ORM.png` (packed occlusion/roughness/metallic) maps, such as the
`materials/arnold/...` sets of `RoboVerseOrg/roboverse_data`. Each variant picks one
coherent set per surface and applies it through metric box UV projection, so a
primitive floor needs no authored UVs and texture density (`uv_scale` tiles per
metre) is the same on Blender and Isaac Sim:

```bash
python examples/6_advanced_rendering.py --sim blender --render.device OPTIX \
  --material-library /path/to/materials/arnold --hdri /assets/studio.hdr \
  --output examples/output/textured_blender
```

For diverse real-scene backgrounds, render inside one of the interior scenes in
`roboverse_pack/scenes` (Kujiale/InteriorAgent, manycore, Arnold): `--background
kujiale_scene_0009` loads that scene on the renderer (Blender imports the USD,
Isaac Sim references it per env), drops the primitive floor/backdrop, and keeps a
flat MuJoCo ground at z = 0, which is the calibrated floor height of these scenes.
The scene assets must be present locally (see
`roboverse_pack/asset/setup_interior_agent_assets.py` and the InteriorAgent terms
of use); the scene file's hash is recorded in the manifest. `--hdri-library DIR`
adds every `.hdr`/`.exr` under a directory to the HDRI pool, so world lighting is
sampled per variant from the whole library:

```bash
python examples/6_advanced_rendering.py --sim isaacsim --physics mujoco --settle-steps 60 \
  --background kujiale_scene_0009 --hdri-library /path/to/hdris \
  --output examples/output/kujiale_isaac
```

To render a recorded robot trajectory instead of the primitive scene, name a
registered task: its `ScenarioCfg` (robot, objects, ground) is reused with the
chosen renderer, and its demonstration file drives MuJoCo through
`HybridSimHandler.simulate_steps(actions=...)`, one image per `--steps-per-frame`
actions. Task assets keep their authored materials; lighting, HDRI, camera and the
sensor stage still vary per recipe, and `--background` can wrap the task in an
interior scene:

```bash
python examples/6_advanced_rendering.py --sim isaacsim --physics mujoco \
  --task libero.pick_butter --demo 0 --steps-per-frame 2 --multiview \
  --camera-pos 1.2 -1.4 0.9 --look-at 0 0 0.1 --output examples/output/libero_butter
```

The trajectory file must be local (`roboverse_data/trajs/...`) and every object needs
a USD for the renderer plus an MJCF for MuJoCo. The manifest records the trajectory
hash, demo index and action count; each frame records the control steps replayed so
far. Actions are replayed closed-loop in MuJoCo, so the motion is MuJoCo's response
to the recorded targets, not a bit-exact copy of the source simulator's states.

Both presets sample key/fill light color temperatures in Kelvin, thin-lens depth of
field (`--no-depth-of-field` keeps pinhole cameras) and a per-camera image-space
sensor stage (`--no-sensor` disables it): exposure, white balance, OpenCV
distortion, vignetting, defocus blur and shot/read noise, computed identically for
both renderers. The first frame of every camera is saved twice, `<view>_env_<i>.png`
after the sensor stage and `<view>_env_<i>_raw.png` straight from the renderer.

For Blender GPU rendering, choose `--render.device OPTIX` or `CUDA` (or leave
`AUTO`); `GPU` is not a Cycles compute backend name. Importing authored USD
materials also requires compatible `pxr` bindings. The local validation environment
uses `bpy==4.2.0` with `usd-core==25.5`; the 24.5 USD wheel crashed on import in
that environment. See `packages/metasim/ENVIRONMENTS.md` for the tested setup.

Replay a saved realization, passing matching scene and camera options:

```bash
python examples/6_advanced_rendering.py --sim blender \
  --recipe examples/output/recipes/sample_000000_variant_0000/recipe.json \
  --output examples/output/replayed
```

For independent workers use the same seed/sample ID and `--num-shards 4
--shard-index 0` through `3`. Variant IDs are assigned by stride, so worker count
and completion order do not change recipes. Workers can share an output directory
when their variant assignments are disjoint. Use a fresh output directory for a
new run; existing variant directories are never overwritten.

Each completed variant contains `recipe.json`, `manifest.json`, one PNG per
camera/env, per-frame camera calibration and, when requested, MP4 videos.
`camera_<frame>.json` holds `render` (the renderer's pinhole position, world
quaternion and intrinsics per env) and `sensor` (the OpenCV `intrinsics`,
`distortion = [k1, k2, p1, p2, k3]` and `model` describing the processed image).
The manifest includes resolved recipe asset SHA-256 hashes, the hashes of the
scenario's USD/MJCF/URDF files (`scene_asset_sha256`), renderer and physics
versions (`backend_versions`), CLI settings and measured elapsed time. When a sibling manifest exists, replay verifies recipe and texture/HDRI
hashes before launching the simulator; use `--no-verify-assets` only for intentional
edits. Standalone recipes have no previous hashes to verify. Meshes, robot assets
and renderer binaries are not covered by the recipe asset hash list.
Per-frame metrics report mean RGB, standard deviation, near-black pixel fraction
and pixels with a clipped channel; these diagnostics are not a realism score. A contact sheet
shows up to eight completed variants per shard. Each variant is built in a temporary
directory and renamed only when all outputs are complete. Pixel-identical replay
requires more than identical parameters and is not promised across renderer versions,
GPU models or drivers.

## Rendering controls and verification

`RenderCfg` adds `seed`, `denoise`, `max_bounces`, `exposure`, `view_transform`,
`settle_frames` (default 2) and `asset_timeout_s` (default 30).
Seed, exposure and view transform are Blender controls; Isaac Sim retains its own
tone mapper. Cycles uses a fixed seed (no animated seed). `samples` and bounce
budgets must be positive integers. Path-traced Isaac Sim captures schedule at
least `settle_frames` propagation frames plus `ceil(samples / min(samples, 32))`
accumulation frames. Capture waits for pending USD stage assets with a deadline,
reads sensors after accumulation and never steps physics. Stage readiness does
not establish shader-cache completion or perceptual convergence. `samples=None`
retains backend defaults.

Isaac Sim rejects `rasterization`; use a supported RTX mode. Render mode availability
and sample settings should be checked on the supported Isaac Sim version. See the
[RTX path-tracing settings](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)
for per-frame versus accumulated sample counts.

Before producing datasets, inspect both renderer outputs and run the appropriate
integration test in the environment mapped by `packages/metasim/ENVIRONMENTS.md`:

```bash
# From packages/metasim, in the corresponding backend environment:
python -m pytest metasim/test/randomization/test_visual_render.py -k blender
python -m pytest metasim/test/randomization/test_visual_render.py -k isaacsim
```

A passing simulator-free suite verifies parameter/scheduling contracts only.
Rendered image quality, timing and memory behavior need actual backend runs;
no cross-renderer photometric or performance parity is asserted by this tutorial.

### Measured validation (2026-09-16)

On Quadro RTX 6000 GPUs, driver 580.82.07, Python 3.11, Blender 4.2 and Isaac
Sim 5.0 / Isaac Lab 2.2.1, the portable integration test passed on Blender (one
environment) and Isaac Sim (one, two and four environments). It checks rendered
pixel changes, actual camera positions and focal intrinsics against recipes,
replay, physical-state invariance, resource counts and multi-env isolation.
Isaac Lab needed live pose updates, and camera state export needed the sensor's
actual intrinsic matrices instead of the original configuration. Earlier captures
made before these fixes should not be treated as correctly calibrated datasets.

A four-env MuJoCo/Isaac controlled-robot regression also passed: twelve ordinary
control intervals and three batches of four intervals produced matching object
and joint states at absolute tolerance 1e-6, with actuator targets preserved.

In a warmed primitive scene at 128² / 32 samples, eight output frames with four
physics intervals each took 2.031 s with per-step Blender rendering versus 0.754 s
with batched rendering (2.69×). Isaac took 4.010 s versus 1.093 s (3.67×). Final
MuJoCo object states were identical. These are one local comparison, not a throughput
guarantee. An explicit shared floor supported the cube/sphere within 0.4 mm of the
nominal surface after settling; no hidden default floor was present.

PBR stress runs performed 600 applications over 120 albedo paths with roughness,
metallic, normal, UV/IOR, HDRI and camera variation. Blender retained 8 materials,
8 images and 13 objects; Isaac retained 71 USD prims. RSS from iteration 119 to
599 rose from 1934.4 to 1985.2 MiB for Blender and 6594.1 to 6621.1 MiB for Isaac.
Stable authored resources do not imply bounded renderer caches or unlimited dataset
capacity. These scene/UV settings differ from earlier probes, so absolute RSS
between those runs is not a controlled measurement of the cache optimization.

With the schema v3 features (eight `arnold` texture sets through metric box UVs,
Kelvin key/fill lights, depth of field and the sensor stage) the same four recipes
rendered on both backends with matching texture placement and density. Settled
384² stills took 1.38–1.47 s per variant on Blender (OptiX, 96 samples) and
2.56–3.51 s on Isaac Sim (path tracing, 64 samples), excluding launch. Replaying one
recipe with only `f_stop` changed gave a monotone Laplacian variance of 46.28
(pinhole), 43.88 (f/2.9) and 39.81 (f/1.4), so depth of field is measurable rather
than assumed. The sensor stage is an exact identity at zero parameters, its
distortion round trip is accurate to ~1e-15, and one 512² frame takes about 0.3 s.
Two-view 24-frame Franka videos with these features took 25.1–27.0 s per variant
on Blender at 384² / 96 samples and 34.8–36.0 s on Isaac Sim at 256² / 64 samples;
every frame's render and sensor calibration was audited against its recipe. Two
Kujiale interiors (`kujiale_scene_0009`, `kujiale_scene_0032` with a Franka)
rendered on both backends from the same recipes in 10.9–11.3 s (Blender) and
7.7–8.1 s (Isaac Sim) per 384² variant after the scene loaded; auto exposure
brought a living room that clipped 10.3% of Blender pixels down to none. None of
this calibrates the distributions against a real camera.

A fixed-camera 1/4/8× light-intensity comparison exposed an underlit Isaac demo.
The selected 8× baseline increased mean RGB from approximately (51,54,50) to
(107,112,102), with fewer than 0.01% of pixels containing a clipped channel in
that frame. This is scene tuning, not a photometric calibration or realism score.
