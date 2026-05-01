# Blender Offline Render Backend Design

Date: 2026-05-02

## Summary

Add Blender as an offline render backend for robot tasks by using the `bpy`
5.0.1 module installed in the `isaacsim` conda environment. The first milestone
does not make Blender part of the live physics loop. Instead, a simulator run
records MetaSim `TensorState` frames, then a Blender replay renderer loads the
same scenario assets, applies saved object and robot body transforms per frame,
and renders RGB frames with Cycles.

This is the lowest-risk path to realistic rendering because it validates asset
import, transform mapping, camera calibration, and render quality without
requiring Blender to satisfy real-time hybrid stepping.

## Current Status

RoboVerse benchmark code currently accepts only `isaacsim` and `mujoco` for the
`benchmark.cube_reach` task and for `scripts/advanced/run_bidexbench_cube_reach.py`.
The current hybrid path constructs separate physics and render `ScenarioCfg`
instances and synchronizes the render handler from the physics handler state.

MetaSim has a `metasim.sim.blender.BlenderHandler`, but it is experimental and
does not support robot task rendering yet:

- It imports `bpy` directly and is intended as an in-process renderer.
- It rejects `PrimitiveCubeCfg`, `PrimitiveSphereCfg`, and `ArticulationObjCfg`.
- It does not provide a complete render-handler contract for robot tasks:
  `_get_states`, `_simulate`, `render`/`refresh_render`, and `close` need real
  implementations.
- It currently renders only one camera to a temporary PNG and returns a legacy
  observation dict instead of `TensorState.cameras`.

Local environment checks:

- `conda activate isaacsim && python -c "import bpy"` works.
- `bpy.app.version_string` is `5.0.1`.
- `bpy.app.background` is `True`.
- `bpy.ops.wm.usd_import` and `bpy.ops.render.render` are available.
- A small in-process Cycles render succeeded.
- The OpenArm USD imports into Blender as mesh objects plus transform empties,
  not as an armature. Therefore the first implementation should drive imported
  link transform objects from MetaSim body states, not pose bones.

## Official Blender Basis

The design relies on stable, documented Blender APIs and workflows:

- Blender can be built and used as a Python module, though official release
  downloads do not ship that module as the normal application artifact.
  Reference: https://developer.blender.org/docs/handbook/building_blender/python_module/
- Python rendering is available through `bpy.ops.render.render(write_still=True)`.
  Reference: https://docs.blender.org/api/current/bpy.ops.render.html
- USD import is available through `bpy.ops.wm.usd_import`.
  Reference: https://docs.blender.org/api/current/bpy.ops.wm.html
- Cycles quality is controlled through render samples, adaptive sampling,
  denoising, and GPU device settings.
  References:
  - https://docs.blender.org/manual/en/5.0/render/cycles/render_settings/sampling.html
  - https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html

## Goals

- Support offline Blender rendering for robot task runs.
- Use `bpy` in the `isaacsim` conda environment, not a standalone Blender binary.
- Render task cameras from saved MetaSim states.
- Support the `benchmark.cube_reach` scene first: OpenArm bimanual Wuji robot,
  tabletop, cube, and configured benchmark cameras.
- Use Cycles by default for realistic RGB output.
- Keep the first implementation scoped to one environment per frame.
- Fail explicitly when Blender cannot import an asset, cannot map a body/link,
  or cannot render a requested camera.
- Support Blender Python module versions `>=5.0.0`, with local validation on
  `bpy==5.0.1`.

## Non-Goals

- Do not make Blender a real-time online render backend in the first milestone.
- Do not implement Blender physics.
- Do not require Blender to support simulator stepping or control targets.
- Do not solve all MetaSim asset classes immediately.
- Do not add a standalone Blender process/IPC worker unless in-process `bpy`
  proves unstable.

## Architecture

The first implementation should introduce an offline replay renderer rather than
overloading the live benchmark session path.

The intended shape is:

1. A normal task run uses an existing physics backend, usually MuJoCo.
2. The run records a list or stream of MetaSim `TensorState` frames.
3. The offline Blender renderer receives:
   - the original `ScenarioCfg`,
   - the saved `TensorState` frames,
   - render settings,
   - output path options.
4. The renderer builds a Blender scene once.
5. For each frame, it applies object and robot body transforms, renders each
   requested camera, and writes frame images or returns RGB tensors.

This can be implemented in MetaSim as either:

- a completed `BlenderHandler` with an offline helper method, or
- a small `BlenderOfflineRenderer` wrapper that uses shared Blender scene
  construction helpers.

Prefer completing `BlenderHandler` enough to satisfy the render-side contract,
then adding a focused offline replay helper around it. That keeps the backend in
MetaSim where simulator/render backends already live, while leaving RoboVerse to
provide task-specific CLI wiring.

## Data Flow

Physics run:

1. Build benchmark `ScenarioCfg` with simulator `mujoco` or `isaacsim`.
2. Run reset/step loop.
3. Save each frame's `TensorState`, including:
   - `objects[name].root_state`,
   - `robots[name].root_state`,
   - `robots[name].body_names`,
   - `robots[name].body_state`,
   - camera configs from the scenario.

Offline render:

1. Build a render-only `ScenarioCfg` with simulator `blender`.
2. Launch the Blender scene.
3. Import robot USD visuals and create primitive objects.
4. Build name maps from MetaSim object/body names to Blender objects.
5. For each `TensorState`:
   - set primitive object transforms from object root states,
   - set robot link transforms from `RobotState.body_state`,
   - set camera intrinsics/extrinsics from `PinholeCameraCfg`,
   - render RGB for each camera,
   - save or return the RGB frame.

## Asset Strategy

Primitive objects:

- Create `PrimitiveCubeCfg` objects directly with Blender mesh primitives.
- Apply scale from MetaSim size/radius/height fields.
- Assign simple Principled BSDF materials from MetaSim RGB colors.
- Add table/cube support first because that covers `benchmark.cube_reach`.

Robots and articulated objects:

- Prefer USD import for Blender because OpenArm already has USD visual assets.
- Use `bpy.ops.wm.usd_import(filepath=...)`.
- Preserve imported hierarchy and transform empties.
- Map MetaSim `body_names` to imported Blender objects by normalized name:
  exact match first, then match after Blender suffix removal such as `.001`.
- For OpenArm, drive transform empties with names such as `openarm_left_link7`,
  `left_palm_link`, and finger link names.

Fallbacks:

- If a robot has no USD path, fail with a clear error in the first milestone.
- URDF/MJCF mesh extraction can be a later extension.
- If multiple imported objects match one body name, choose the parent transform
  object with children when possible and log the chosen mapping.

## Transform Mapping

MetaSim states use root/body tensors with position, quaternion, linear velocity,
and angular velocity. Blender only needs position and orientation for offline
rendering.

Rules:

- Convert MetaSim quaternion `(w, x, y, z)` to Blender `mathutils.Quaternion`
  order.
- Set Blender object `matrix_world` from translation and quaternion.
- Ignore velocities for rendering.
- For robot bodies, use `RobotState.body_names` and `RobotState.body_state`.
- For rigid/primitive objects, use `ObjectState.root_state`.
- Fail if a required object or body cannot be mapped unless the caller opts into
  a diagnostic "partial render" mode.

Because OpenArm USD imports as transform nodes rather than an armature, pose
bone APIs should not be the initial implementation path.

## Camera And Render Output

Camera creation:

- Support `PinholeCameraCfg`.
- Create one Blender camera per MetaSim camera.
- Use camera position and `look_at` to construct the camera transform.
- Use focal length, horizontal aperture, width, and height to match MetaSim
  intrinsics as closely as Blender allows.
- Set clipping range from the camera config.

Render output:

- For each camera, set `scene.camera`.
- Set `scene.render.resolution_x/y`.
- Render with `bpy.ops.render.render(write_still=True)`.
- Read the output image into a `torch.uint8` RGB tensor with shape
  `(1, H, W, 3)`.
- Populate `TensorState.cameras[camera.name].rgb` when using handler-style APIs.
- For CLI use, write PNG files under an output directory organized by camera and
  frame index.

Render quality:

- Default to Cycles.
- Provide explicit settings for samples, denoising, device type, and resolution.
- Prefer conservative defaults such as 64 samples plus denoising for smoke
  testing, and allow users to increase quality for final renders.

## Error Handling

The renderer should fail fast for unsupported paths:

- `bpy` import failure.
- Blender version below the supported floor.
- unsupported camera config type.
- missing USD path for robot/articulation.
- failed USD import.
- unmapped required body/object.
- requested GPU device unavailable when GPU rendering is explicitly requested.

Warnings are acceptable for non-critical imported USD material binding warnings,
as long as the scene imports and renders.

## RoboVerse CLI Integration

Keep current live command behavior unchanged.

Add an offline mode to `scripts/advanced/run_bidexbench_cube_reach.py`, for
example:

```bash
python scripts/advanced/run_bidexbench_cube_reach.py \
  --sim mujoco \
  --steps 100 \
  --record-states /tmp/cube_reach_states.pt \
  --offline-renderer blender \
  --render-output /tmp/cube_reach_blender
```

The first milestone uses these CLI names:

- `--record-states PATH`: save physics states with `torch.save`.
- `--offline-renderer blender`: select the offline replay renderer.
- `--render-output DIR`: write rendered PNG frames.
- `--render-samples N`: override Cycles sample count.
- `--render-device {CPU,CUDA,OPTIX,HIP,ONEAPI,METAL}`: optional explicit
  Blender/Cycles render device.

The semantics should be explicit: Blender is an offline renderer, not a live
`--renderer blender` backend in the first milestone.

Only after offline rendering works should `--renderer blender` be considered for
live hybrid rendering.

## Testing Strategy

MetaSim general tests:

- Add focused tests for Blender transform helpers that do not require GPU.
- Add primitive mesh creation tests if `bpy` is importable; otherwise skip with a
  clear reason.
- Add a tiny render smoke test using CPU Cycles or EEVEE at very low resolution.
- Add a mapping unit test with fake imported object names containing Blender
  suffixes.

RoboVerse tests:

- Add a dry-run/argument test for offline Blender render options.
- Add a state-recording unit test that verifies frames are captured without
  launching Blender.
- Add an integration smoke command gated on `bpy` availability.

Manual verification:

- In `isaacsim` env, render one `benchmark.cube_reach` frame at 64x64.
- Verify output file exists and is non-empty.
- Verify at least one frame from the overview camera contains non-background
  pixels.

Simulator-backed tests should follow existing MetaSim/RoboVerse simulator test
rules. Do not claim GPU-backed simulator tests pass if the local NVIDIA driver is
not available.

## Milestones

1. Complete Blender scene construction for primitives, cameras, lights, and USD
   robot visual import.
2. Implement state-to-Blender transform mapping for one environment.
3. Render one saved `TensorState` frame to PNG from a small scenario.
4. Record and replay `benchmark.cube_reach` frames offline.
5. Add focused tests and documentation.
6. Consider live hybrid `renderer=blender` only after the offline path is
   reliable.

## Fixed First-Milestone Decisions

- CLI flags are `--record-states`, `--offline-renderer`, `--render-output`,
  `--render-samples`, and `--render-device`.
- Saved state frames use `torch.save` for the first milestone.
- Default quality prioritizes smoke-test speed: Cycles, 64 samples, denoising
  enabled, and CPU rendering unless the user explicitly requests a GPU device.
