# Randomization

## Overview
RoboVerse builds on deterministic physics backends, then injects controlled randomness to mimic real‑world variability. Randomization lets you stress test policies across lighting changes, camera drift, texture swaps, and object pose perturbations without rewriting the core task logic.

RoboVerse exposes two broad families of randomizers:
- **Observation randomizers** apply post‑processing noise (e.g., sensor noise, image augmentation).
- **Simulation randomizers** mutate the simulation state itself (e.g., material, lighting, camera intrinsics).

All randomizers share the same interface:

```python
randomizer = SomeRandomizer(cfg, seed=42)
randomizer.bind_handler(handler)
randomizer()  # Apply one round of randomization
```

Calling `bind_handler` connects the randomizer to the active simulator handler. The handler supplies simulator‑specific hooks so randomizers can modify assets without patching simulation code.

> **Backend support**  
> The shipped randomizers call into Omni/Isaac USD APIs during `bind_handler`, so the Isaac Sim backend is currently the only configuration where they execute. Other simulators listed in CLI flags will still load the task, but each randomizer raises `Unsupported handler` until equivalent bindings are implemented.

## Architecture
Custom randomizers inherit from `metasim.randomization.base.BaseRandomizerType`. The base class exposes four key touch points:

- `bind_handler(handler)`: Called once when the task is constructed. Attach simulator‑specific APIs here.
- `__call__(...)`: Implement the actual randomization. Demo scripts call this on demand (e.g., every N steps).
- `set_seed(seed)`: Re-seed the internal RNG at runtime. Every built-in randomizer now respects the seed 1:1, so using the same seed on different randomizers produces identical RNG streams (the resulting scene changes may differ because each randomizer consumes the values in its own way).
- `supported_handlers`: Optional allowlist to guard against unsupported simulators.

Most users never touch the base class directly. Instead, you configure higher level randomizers supplied in `metasim.randomization`:

- `ObjectRandomizer`: Varies mass, friction, restitution, and initial pose.
- `MaterialRandomizer`: Swaps MDL/PBR materials and material physics.
- `LightRandomizer`: Perturbs intensity, color temperature, position, and orientation.
- `CameraRandomizer`: Adjusts extrinsics and intrinsics.
- `SceneRandomizer`: Builds floor, walls, ceilings, and tables with material pools.

Each randomizer takes a config dataclass (e.g., `ObjectRandomCfg`) and optional seed. Configs support different distributions (`uniform`, `gaussian`, etc.) and let you scope updates to a subset of environments via `env_ids`.

## Preparing Material Assets
Scene and material randomizers read MDL files under `roboverse_data/materials`, which mirrors the Hugging Face dataset [`RoboVerseOrg/roboverse_data`](https://huggingface.co/datasets/RoboVerseOrg/roboverse_data/tree/main/materials). You no longer need to pre-download the entire subtree: whenever an `.mdl` or referenced texture is missing locally, `metasim.utils.hf_util.check_and_download_single()` fetches it automatically and stores it in the same relative path.

- **Cold start:** Make sure `huggingface_hub` is installed (it is part of the default dependencies). The dataset is public, so no token is required unless you enabled private mirrors.
- **Cached runs:** Existing files are reused—downloads only happen the first time a specific asset or texture is requested.
- **Offline clusters:** If your environment lacks outbound internet access, run a manual sync once and ship the resulting `roboverse_data/materials` directory with your job. You can still use `snapshot_download(..., allow_patterns=["materials/**"])` to mirror the dataset ahead of time.

### Material variant randomization
Many MDL files contain multiple material definitions. For example:
- `Rug_Carpet.mdl` contains 4 variants: Rug_Carpet_Base, Rug_Carpet_Lines, Rug_Carpet_Hexagonal, Rug_Carpet_Honeycomb
- `Caoutchouc.mdl` (rubber) contains 93 variants with different colors and finishes
- Overall, vMaterials_2 has 315 files containing 2,605 material variants

By default, `randomize_material_variant=True` in `MDLMaterialCfg` and `SceneMaterialPoolCfg`, enabling two-stage randomization:
1. Select a random MDL file from the pool
2. Select a random material variant within that file

This behavior is fully reproducible through seed control. All material name extraction uses deterministic parsing of MDL file contents, and variant selection uses the randomizer's seeded RNG (`self._rng`).

To specify a particular variant explicitly, use the `::` syntax:
```python
mdl_paths=["path/to/Rug_Carpet.mdl::Rug_Carpet_Hexagonal"]
```

To disable variant randomization and always use the first material in each file:
```python
MDLMaterialCfg(mdl_paths=[...], randomize_material_variant=False)
```

### Binding Flow in Tasks
1. Task instantiates the simulator handler.
2. Randomizers are created and bound with `randomizer.bind_handler(handler)`.
3. Randomizers are triggered during reset or at runtime (see `apply_randomization` in the demo).

Because bindings happen inside the task, you can hot‑swap randomizers without modifying the low‑level simulator integration.

## Quick Start Demo
The easiest way to see the system in action is `get_started/12_domain_randomization.py`. Run the script with different levels to watch the progressive randomization pipeline:

```bash
# Ray-traced render mode with Isaac Sim backend
python get_started/12_domain_randomization.py --sim isaacsim --level 2

# Path-traced rendering, faster camera randomization cadence
python get_started/12_domain_randomization.py --level 3 --render-mode pathtracing --randomize-interval 5

# Deterministic replay (no randomization, no physics stepping)
python get_started/12_domain_randomization.py --sim isaacsim --object-states
```

All examples assume Isaac Sim; invoking the randomizers with other simulator backends currently raises an `Unsupported handler` error.

Relevant CLI flags (see `Args` in the script):
- `--level {0,1,2,3}`: Select progressive randomization bundle (details below).
- `--randomize-interval N`: Apply randomizations every `N` simulation steps.
- `--render-mode {raytracing,pathtracing}`: Switch renderer presets (intensity ranges adapt automatically).
- `--seed`: Reproducible RNG for all randomizers.
- `--object-states`: Use recorded states in replay for deterministic debugging; this mode bypasses every randomizer and does not step physics.

`--object-states` switches the demo from action replay to pure state injection: every frame pulls the saved tensor state back into the handler, refreshes the render, and bypasses randomization. It is handy when you want to inspect the original trajectory before layering domain randomization on top.

The script records MP4 outputs under `get_started/output/`:
- `12_dr_level{N}_{sim}.mp4` for action replay (randomization enabled).
- `12_dr_states_{sim}.mp4` when `--object-states` is used.


## Worked Examples
Below are concise snippets showing how to use each built‑in randomizer in isolation. They mirror the patterns used in the demo.

### Object Randomizer
```python
from metasim.randomization import ObjectRandomizer, ObjectPresets

# Perturb mass/friction and initial pose for the "cube" asset
cube_rand = ObjectRandomizer(
    ObjectPresets.grasping_target("cube", physics_randomization="full"),
    seed=123,
)
cube_rand.bind_handler(handler)
cube_rand()  # Apply once, repeat as needed
```

### Material Randomizer
```python
from metasim.randomization import MaterialRandomizer, MaterialPresets

# Swap materials from the MDL wood collections (Arnold + vMaterials) and sync friction
cube_mat_rand = MaterialRandomizer(
    MaterialPresets.mdl_family_object("cube", family="wood", randomization_mode="combined"),
    seed=123,
)
cube_mat_rand.bind_handler(handler)
cube_mat_rand()
```

The randomizer implements two-stage selection:
1. Randomly select an MDL file from the configured pool
2. Randomly select a material variant within that file (controlled by `randomize_material_variant`, default `True`)

For fine-grained control:
```python
from metasim.randomization import MaterialRandomCfg, MDLMaterialCfg

# Explicit variant specification
mat_cfg = MaterialRandomCfg(
    obj_name="cube",
    mdl=MDLMaterialCfg(
        mdl_paths=[
            "roboverse_data/materials/vMaterials_2/Carpet/Rug_Carpet.mdl::Rug_Carpet_Lines",
            "roboverse_data/materials/vMaterials_2/Metal/Aluminum.mdl",  # Random variant
        ],
        randomize_material_variant=True,
    )
)
```

### Light Randomizer
```python
from metasim.randomization import (
    LightRandomizer,
    LightRandomCfg,
    LightIntensityRandomCfg,
    LightColorRandomCfg,
    LightPositionRandomCfg,
)

key_light_cfg = LightRandomCfg(
    light_name="key_light",
    intensity=LightIntensityRandomCfg(intensity_range=(8000.0, 15000.0), enabled=True),
    color=LightColorRandomCfg(temperature_range=(3500.0, 6500.0), use_temperature=True, enabled=True),
    position=LightPositionRandomCfg(position_range=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)), enabled=True),
    randomization_mode="combined",
)
key_light_rand = LightRandomizer(key_light_cfg, seed=7)
key_light_rand.bind_handler(handler)
key_light_rand()
```

### Camera Randomizer
```python
from metasim.randomization import CameraRandomizer, CameraPresets

camera_rand = CameraRandomizer(
    CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
    seed=999,
)
camera_rand.bind_handler(handler)
camera_rand()
```

### Scene Randomizer
```python
from metasim.randomization import ScenePresets, SceneRandomizer

# Tabletop workspace with explicit material families
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
scene_rand = SceneRandomizer(scene_cfg, seed=42)
scene_rand.bind_handler(handler)
scene_rand()
```

Scene material pools also support variant randomization via `SceneMaterialPoolCfg.randomize_material_variant` (default `True`).


## Progressive Randomization Levels
`get_started/12_domain_randomization.py` implements a four‑level schedule that you can reuse in your own benchmarks:

| Level | Description | Randomizers Triggered | Notes |
|-------|-------------|-----------------------|-------|
| 0 | Baseline | Scene randomizer (deterministic materials) | No stochastic variation, actions replayed via physics |
| 1 | Material randomization | Scene + Material | Scene material pools unlocked; object materials randomized |
| 2 | Lighting randomization | Scene + Material + Light | Adds intensity, color, and position variation for ceiling lights |
| 3 | Camera randomization | Scene + Material + Light + Camera | Camera extrinsics/intrinsics perturbed each interval |

The script reuses the same `SceneRandomizer` for all levels, but at Level 0 the material pools are forced to a single deterministic choice. From Level 1 onward, scene surfaces pull from material collections, and the box object gains a randomized wood finish. Level 2 adds ceiling light variations with different ranges depending on `--render-mode`. Level 3 finally enables camera perturbations using the surveillance preset.

Randomization is applied once at reset and then every `randomize_interval` steps. You can tweak the schedule by editing `apply_randomization` if, for example, you need curriculum learning where components activate at different episode counts.


## Choosing Between Action Replay and State Replay
- **Action replay (default)** uses the recorded action sequence and steps the simulator. Domain randomization stays active, so you can evaluate how a fixed trajectory looks (and potentially fails) under different visual conditions.
- **State replay (`--object-states`)** bypasses physics integration and randomization entirely. The script injects the saved simulator state for every frame and simply refreshes the render, giving you the exact visuals that were recorded.

Use state replay for baseline inspection or regression tests, and action replay for studying robustness to randomized visuals.


## Best Practices
- **Seed everything** during debugging. All randomizers accept a seed; set `torch.manual_seed` and `numpy.random.seed` too for reproducibility. Material variant selection is fully deterministic given a seed.
- **Scope randomization** to specific environments via config `env_ids` when running vectorized simulations.
- **Combine wisely**: Use `ObjectRandomizer` for pose/mass adjustments and `MaterialRandomizer` for visual appearance. Avoid duplicating friction edits unless intentional.
- **Profile renderers**: Path tracing benefits from larger light intensity ranges; the demo automatically widens ranges in that mode.
- **Record outputs** often. Visual inspection helps validate that randomization stays within reasonable bounds.
- **Leverage variant diversity**: The default `randomize_material_variant=True` significantly expands material pools. Disable it only if you need strict control over which specific variants are used.
