# 12. Domain Randomization

This quick-start shows how to replay a demonstration trajectory while progressively enabling scene, material, lighting, and camera randomization. It mirrors the workflow we use for benchmarking: spawn a simple room, randomize it every *N* steps, and capture video for visual inspection.

## Run the demo
```bash
python get_started/12_domain_randomization.py \
    --sim isaacsim \
    --render-mode pathtracing \
    --level 2 \
    --randomize-interval 10 \
    --headless
```

Drop `--headless` if you want to watch the replay live. The script streams RGB frames to `get_started/output/12_dr_level{N}_{sim}.mp4` and logs every randomization pass to the console.

### CLI highlights
- `--level {0,1,2,3}` Progressive bundles (see table below).
- `--randomize-interval N` Apply randomization every *N* simulation steps after reset.
- `--render-mode {raytracing,pathtracing}` Ties into lighting intensity ranges.
- `--object-states` Bypass randomization entirely and simply replay recorded states (useful for debugging trajectories or recording “clean” footage).
- `--seed` Every randomizer shares the same deterministic RNG seed.

## Progressive levels
The script wires all available randomizers, then activates them based on the selected level.

| Level | Scene surfaces | Object randomization | Material randomization | Lighting randomization | Camera randomization |
|-------|----------------|----------------------|------------------------|------------------------|----------------------|
| 0 | Deterministic floor/wall/ceiling/table materials | Box mass/pose only | Disabled | Disabled | Disabled |
| 1 | Material families unlocked | Enabled (same as level 0) | Box visual+physical materials sampled from `paper/wood` | Disabled | Disabled |
| 2 | Same as level 1 | Enabled | Enabled | Ceiling lights randomize intensity, color temperature, pose | Disabled |
| 3 | Same as level 2 | Enabled | Enabled | Enabled | `CameraPresets.surveillance_camera` perturbs intrinsics/extrinsics |

Randomization runs once at reset and then every `randomize_interval` steps. You can edit `apply_randomization()` in the script if you need a different cadence.

## Scene + material pools
`ScenePresets.tabletop_workspace()` accepts `floor_families`, `wall_families`, `ceiling_families`, and `table_families`. Each tuple expands into MDL pools using `MDLCollections`, so you can express "give me concrete floors and painted walls" without hard-coding file paths:

```python
scene_cfg = ScenePresets.tabletop_workspace(
    room_size=10.0,
    wall_height=5.0,
    table_size=(1.8, 1.8, 0.1),
    table_height=0.7,
    floor_families=("concrete", "carpet"),
    wall_families=("wall_board", "paint"),
    ceiling_families=("architecture",),
    table_families=("wood", "plastic"),
)
```

When a listed `.mdl` or texture is missing locally, the randomizer automatically downloads it from the `RoboVerseOrg/roboverse_data` dataset on Hugging Face (matching the `roboverse_data/materials/...` layout). Existing files are reused—no duplicate downloads. If you prefer to prefetch everything manually, you can still run `huggingface_hub.snapshot_download(..., allow_patterns=["materials/**"])` once.

### Material variant randomization
Many MDL files contain multiple material variants (e.g., `Rug_Carpet.mdl` has 4 variants: Base, Lines, Hexagonal, Honeycomb). By default, `randomize_material_variant=True` in material configurations, which means:
- First, a random MDL file is selected from the pool
- Then, a random variant within that file is selected

This significantly expands diversity: the vMaterials_2 collection contains 315 MDL files with 2,605 total variants (8.3x multiplier). All selections use seeded RNG for full reproducibility.

## Randomizers in play
The demo coordinates five randomizer types. Feel free to lift the snippets into your own training loops.

### SceneRandomizer
Creates the floor, four walls, ceiling, and table only when the scenario doesn’t already define a scene. Material pools can be deterministic (level 0) or sampled from the selected families (level ≥1).

### ObjectRandomizer
`ObjectPresets.heavy_object("box_base")` perturbs the box’s mass and friction while keeping the replayed pose fixed in this tutorial. You can edit `box_rand.cfg.pose` to allow position/rotation jitter during replay.

### MaterialRandomizer
`MaterialPresets.mdl_family_object("box_base", family=("paper", "wood"))` handles the object's visual material and optional physical overrides. The system implements two-stage randomization:
1. Select a random MDL file from the configured pool
2. Select a random material variant within that file (default behavior)

For example, if a material pool contains `Rug_Carpet.mdl` (4 variants) and `Caoutchouc.mdl` (93 variants), this creates 97 possible material outcomes. Because MDL pools are resolved lazily, the first time a material is sampled it is downloaded along with any referenced textures.

### LightRandomizer
A `LightRandomizer` is attached to each ceiling light. The script selects different intensity ranges for ray-traced v path-traced renders and jitters color temperature, position, and (for the main disk light) orientation.

### CameraRandomizer
Level 3 enables `CameraPresets.surveillance_camera`, which perturbs intrinsics/extrinsics in a combined mode so that the replayed footage mimics hand-held camera drift.

## Output + customization tips
- All renders land in `get_started/output/`. Delete the directory to keep only the latest clips.
- The `randomizers` dict returned by `initialize_randomizers()` can be modified on the fly. For example, append more `MaterialRandomizer` instances if you add objects, or replace the `ScenePresets` call with `ScenePresets.custom_scene()` for custom geometry.
- Because every randomizer shares the same seed, re-running the script with identical CLI arguments reproduces the full randomization timeline. Change the seed or interval to explore new perturbations.

This tutorial now reflects the production workflow we use internally—select a scene preset, pick material families, and let the randomizers pull assets on demand.
