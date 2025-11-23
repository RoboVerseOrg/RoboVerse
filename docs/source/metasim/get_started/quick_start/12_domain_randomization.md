# 12. Domain Randomization

This tutorial shows how to use domain randomization during trajectory replay. We'll go from simple material changes to full scene randomization with lighting and camera variations - basically everything you need for robust sim-to-real transfer.

## Quick Start

```bash
python get_started/12_domain_randomization.py \
    --scene_mode 1 \
    --level 2 \
    --seed 42 \
    --randomize_interval 60
```

The script replays a close_box demonstration while applying randomization every 60 steps. Output video is saved to `get_started/output/12_dr_*.mp4`.

## Visual Examples

Here are videos showing what each mode and level looks like. All use the same close_box trajectory, just with different randomization settings.

### Mode 0: Manual Geometry

<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode0_manual_level0.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 0: Baseline</b><br>Fixed scene, no randomization</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode0_manual_level1.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 1: Material</b><br>Table material varies between wood/metal</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode0_manual_level2.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 2: + Lighting</b><br>Material + light intensity/color</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode0_manual_level3.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 3: Full</b><br>Material + lighting + camera</p>
    </div>
</div>

### Mode 1: USD Table

<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode1_usd_table_level0.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 0: Baseline</b><br>Single USD table model</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode1_usd_table_level1.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 1: Scene</b><br>Switches between 5 table models</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode1_usd_table_level2.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 2: + Lighting</b><br>Table switching + lighting</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode1_usd_table_level3.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 3: Full</b><br>Table + lighting + camera</p>
    </div>
</div>

### Mode 2: USD Scene

<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode2_usd_scene_level0.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 0: Baseline</b><br>Single Kujiale interior scene</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode2_usd_scene_level1.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 1: Scene</b><br>Switches between 12 interior scenes</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode2_usd_scene_level2.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 2: + Lighting</b><br>Scene switching + lighting</p>
    </div>
    <div style="width: 23%; text-align: center;">
        <video width="100%" autoplay loop muted playsinline>
            <source src="https://roboverse.wiki/_static/standard_output/12_dr_mode2_usd_scene_level3.mp4" type="video/mp4">
        </video>
        <p style="margin-top: 5px;"><b>Level 3: Full</b><br>Scene + lighting + camera</p>
    </div>
</div>

**Notes:**
- Level 0 is useful for debugging - everything stays the same
- Level 1 adds scene/material variation
- Level 2 is where things get interesting with lighting changes
- Level 3 throws in camera randomization for good measure

## Architecture Overview

The randomization system separates two concerns: managing object lifecycle vs. editing their properties.

### Object Types

**Static Objects** (Handler-managed):
- Robot (franka)
- Task objects (box_base)  
- Cameras
- Lights

These are created by the Handler during environment initialization and remain throughout the simulation.

**Dynamic Objects** (SceneRandomizer-managed):
- Environment geometry (floors, walls, ceilings)
- Workspace surfaces (tables, desktops)
- Distractor objects (decorative items)

These can be created, deleted, or switched at runtime for maximum flexibility.

### Randomizer Types

**SceneRandomizer** - Manages object lifecycle:
- Creates and deletes scene elements
- Switches between USD asset variants
- Registers all objects to a central ObjectRegistry

**Property Editors** - Modify object attributes:
- MaterialRandomizer: Visual and physical material properties
- ObjectRandomizer: Mass, friction, pose
- LightRandomizer: Intensity, color, position
- CameraRandomizer: Intrinsics and extrinsics

This separation enables clean composition. For example, you can randomize materials on both static objects (robot links) and dynamic objects (table surface) using the same MaterialRandomizer interface.

## Command Line Arguments

### Scene Configuration

**--scene_mode** (default: 3)

Controls the type of scene geometry:

- **0 (Manual)**: All procedural geometry
  - Environment: Floor, 4 walls, ceiling (manual cubes)
  - Workspace: 1.8m x 1.8m table (manual cube)
  - Objects: None
  
- **1 (USD Table)**: Manual environment with USD table
  - Environment: Manual geometry
  - Workspace: Table785 (5 table models from EmbodiedGen)
  - Objects: None
  
- **2 (USD Scene)**: Full USD environment and workspace
  - Environment: Kujiale scenes (12 interior scenes)
  - Workspace: Table785
  - Objects: None
  
- **3 (Full USD)**: Maximum diversity
  - Environment: Kujiale scenes
  - Workspace: Table785
  - Objects: Desktop supplies (10 fruit models)

### Randomization Control

**--level** (default: 1)

Progressive randomization complexity:

- **Level 0**: Baseline
  - Fixed scene configuration
  - No randomization (deterministic)
  - Useful for debugging and baseline comparisons
  
- **Level 1**: Scene and Material
  - Scene randomization: USD asset selection (modes 1-3)
  - Material randomization: Visual and physical properties
  - For mode 0: Table material randomizes between wood and metal families
  
- **Level 2**: Add Lighting
  - Everything from Level 1
  - Light intensity: 16K-30K (main), 6K-12K (corners) for raytracing
  - Color temperature: 2700K-6000K range
  - 5 lights total: 1 main DiskLight + 4 corner SphereLights
  
- **Level 3**: Full Randomization
  - Everything from Level 2
  - Camera randomization: Position, orientation, look-at point, FOV
  - Surveillance camera preset with small perturbations

**--randomize_interval** (default: 60)

Number of simulation steps between randomization applications. Set to a higher value (e.g., 120) for less frequent changes.

**--seed** (default: 42)

Random seed for reproducibility. All randomizers derive their seeds from this base value using fixed offsets (e.g., `seed+1`, `seed+2`), ensuring independent but deterministic random streams.

**--render_mode** (default: raytracing)

Rendering quality. Choose `pathtracing` for higher quality at slower speed. This affects light intensity ranges automatically.

## Scene Modes in Detail

### Mode 0: Manual Geometry

Good starting point for understanding how things work. Everything is just cube primitives.

**Environment**:
```python
ScenePresets.empty_room(
    room_size=10.0,      # 10m x 10m room
    wall_height=5.0,     # 5m tall walls
)
```

Creates:
- Floor: 10m x 10m x 0.1m at z=0.005 (Carpet_Beige material)
- Walls: 4 walls, 0.1m thick (Brick_Pavers material)
- Ceiling: 10m x 10m x 0.1m at z=5.05 (Roof_Tiles material)

**Workspace**:
```python
ManualGeometryCfg(
    name="table",
    geometry_type="cube",
    size=(1.8, 1.8, 0.1),
    position=(0.0, 0.0, 0.65),
    default_material="roboverse_data/materials/arnold/Wood/Plywood.mdl"
)
```

In Level 1+, the MaterialRandomizer switches table material between wood and metal families while keeping geometry fixed.

### Mode 1: USD Table

Combines manual environment with realistic table models.

**Workspace**:
```python
USDAssetPoolCfg(
    name="table",
    usd_paths=table_paths,  # 5 tables from EmbodiedGen
    per_path_overrides=table_configs,  # Per-table calibrations
    selection_strategy="random" if level >= 1 else "sequential"
)
```

The system downloads EmbodiedGen assets (URDF + meshes + textures), auto-converts URDF to USD, and switches between models at each randomization step.

### Mode 2: USD Scene

Full interior scenes from the Kujiale dataset.

**Environment**:
```python
USDAssetPoolCfg(
    name="kujiale_scene",
    usd_paths=scene_paths,  # 12 interior scenes
    per_path_overrides=scene_configs,  # Position/scale calibrations
    selection_strategy="random" if level >= 1 else "sequential"
)
```

Kujiale scenes use a two-repo setup:
1. Download RoboVerse USDA (scene layout)
2. Download InteriorAgent assets (meshes, textures, materials)
3. Copy USDA to InteriorAgent folder
4. Load with references resolving to InteriorAgent assets

This gives you curated layouts from RoboVerse with the full asset library from InteriorAgent.

### Mode 3: Full USD

Adds desktop objects to Mode 2 for maximum visual diversity.

**Objects**:
```python
ObjectsLayerCfg(
    elements=[
        USDAssetPoolCfg(
            name=f"desktop_object_{i+1}",
            usd_paths=object_paths,  # 10 fruit models
            selection_strategy="random"
        )
        for i in range(3)  # Place 3 objects
    ]
)
```

Three desktop objects are randomly selected and placed in the scene.

## Randomizer Details

### SceneRandomizer

Manages the three-layer hierarchy:

**Layer 0 (Environment)**: Background geometry
- Manual: Procedural shapes with material assignment
- USD: Complete scene models (Kujiale)

**Layer 1 (Workspace)**: Manipulation surfaces
- Manual: Simple table primitive
- USD: Realistic table models (Table785)

**Layer 2 (Objects)**: Distractors
- USD: Desktop items, decorations

Each layer can be shared across all environments or instantiated per-environment. The `only_if_no_scene` flag prevents conflicts with scenario-defined scenes.

Key responsibilities:
- Download assets from HuggingFace on demand
- Convert URDF to USD automatically
- Apply default materials to manual geometry
- Register all created objects to ObjectRegistry

### MaterialRandomizer

Handles both visual and physical material properties.

**MDL Materials** (visual):
```python
MaterialPresets.mdl_family_object("box_base", family=("paper", "wood"))
```

Selects from MDL families. Each MDL file may contain multiple variants. The system:
1. Downloads MDL file if missing (from RoboVerseOrg/roboverse_data)
2. Downloads referenced textures (albedo, normal, roughness maps)
3. Randomly selects a variant within the MDL (if `randomize_material_variant=True`)
4. Applies the material using IsaacSim's official CreateMdlMaterialPrim API
5. Generates UV coordinates for proper texture display

**Physical Materials** (optional):
```python
PhysicalMaterialCfg(
    friction_range=(0.3, 0.7),
    restitution_range=(0.1, 0.3),
    enabled=True
)
```

Randomizes physics properties on objects with RigidBodyAPI. Note: dynamic objects from SceneRandomizer are visual-only, so they skip this.

### ObjectRandomizer

Randomizes physics properties for static objects:

```python
ObjectPresets.heavy_object("box_base")
```

Includes:
- Mass randomization (e.g., 15-25 kg)
- Friction randomization
- Restitution randomization
- Optional pose randomization (disabled in this demo to preserve trajectory)

### LightRandomizer

Controls lighting parameters. The demo creates 5 lights:

**Main Light** (DiskLight at ceiling center):
```python
LightRandomCfg(
    light_name="ceiling_main",
    intensity=LightIntensityRandomCfg(
        intensity_range=(16000, 30000),  # Raytracing
        enabled=True
    ),
    color=LightColorRandomCfg(
        temperature_range=(3000, 6000),  # Warm to cool white
        use_temperature=True,
        enabled=True
    )
)
```

**Corner Lights** (4 SphereLights):
- Slightly lower intensity range
- Warmer temperature range (2700K-5500K)
- Positioned at room corners for even ambient coverage

Intensity ranges automatically adjust based on `--render_mode`. PathTracing requires higher values (22K-40K for main light) compared to RayTracing (16K-30K).

### CameraRandomizer

Active only at Level 3:

```python
CameraPresets.surveillance_camera("main_camera")
```

Applies small perturbations:
- Position: +/- 0.1m around initial position
- Orientation: +/- 5 degrees
- Look-at target: +/- 0.05m around table center
- FOV: 45-60 degrees

These small perturbations simulate installation variations without drastically changing the viewpoint.

## Asset Management

### Automatic Downloading

When an asset is requested but not found locally:

1. **MDL Materials**: Download from `RoboVerseOrg/roboverse_data`
   - Pattern: `materials/arnold/{family}/{file}.mdl`
   - Includes all referenced textures

2. **EmbodiedGen Assets** (Table785, Desktop Supplies): Download complete folders
   - Pattern: `dataset/{category}/{uuid}/`
   - Includes URDF, mesh files, and textures
   - Automatically converts URDF to USD using MeshConverter

3. **Kujiale Scenes**: Two-step process
   - Download RoboVerse USDA (scene description)
   - Download InteriorAgent assets (meshes, materials, HDR)
   - Copy USDA to InteriorAgent folder for correct reference resolution

All downloads use `huggingface_hub.snapshot_download()` with caching, so repeated runs reuse existing files.

### URDF to USD Conversion

EmbodiedGen provides tables and desktop objects as URDF files. The system automatically converts these to USD on first use:

```python
# User provides
usd_path = "EmbodiedGenData/.../table/uuid.urdf"

# System automatically
1. Downloads complete folder (URDF + mesh/*.obj + textures/*.png)
2. Uses AssetConverterFactory with MESH source type
3. Converts to USD: uuid.usd
4. Caches for subsequent use
```

The conversion uses the same logic as the standalone `urdf2usd.py` script, ensuring consistency.

## Material Randomization Strategy

Materials are randomized differently based on object type:

**Mode 0** (Manual Geometry):
- Table: Wood and metal families
- Floor: Carpet, wood, and stone families
- Walls: All 4 walls share same material (masonry/architecture families)
- Ceiling: Architecture and wall_board families

**Mode 1** (USD Table):
- Box: Wood and paper families
- Table: USD original materials (we get diversity from model switching instead)
- Floor, Walls, Ceiling: Same as Mode 0

**Mode 2/3** (USD Assets):
- Scene objects keep their original materials
- Only task objects (like box_base) get randomized
- This keeps USD scenes looking coherent

The idea: geometric randomization (switching USD models) and material randomization (applying MDL) are complementary. Mode 0 is all about materials, modes 1-3 are about geometry.

## Position Update System

After scene creation, the system adjusts robot and object positions to match the table surface:

```python
table_bounds = scene_rand.get_table_bounds(env_id=0)
if table_bounds:
    table_height = table_bounds['height']
    clearance = 0.05
    
    for obj_state in init_state["objects"].values():
        obj_state["pos"][2] = table_height + clearance
    for robot_state in init_state["robots"].values():
        robot_state["pos"][2] = table_height + clearance
```

This moves objects up/down to sit on the table surface. For manual geometry, bounds come from config. For USD assets, we extract them from mesh bounding boxes.

## Reproducibility

All randomization is deterministic when using the same seed:

```python
# Base seed from CLI
base_seed = args.seed  # e.g., 42

# Derived seeds for each randomizer
scene_rand = SceneRandomizer(cfg, seed=base_seed)
box_mat = MaterialRandomizer(cfg, seed=base_seed + 1)
table_mat = MaterialRandomizer(cfg, seed=base_seed + 2)
walls_mat = MaterialRandomizer(cfg, seed=base_seed + 100)  # Shared by all walls
box_physics = ObjectRandomizer(cfg, seed=base_seed + 3)
main_light = LightRandomizer(cfg, seed=base_seed + 4)
corner_lights = [LightRandomizer(cfg, seed=base_seed + 5 + i) for i in range(4)]
camera = CameraRandomizer(cfg, seed=base_seed + 10)
```

Each randomizer maintains an independent random number generator. Running the script twice with the same seed produces identical randomization sequences.

## Customization Examples

### Change Material Families

```python
# In initialize_randomizers()
box_mat = MaterialRandomizer(
    MaterialPresets.mdl_family_object(
        "box_base",
        family=("stone", "ceramic", "plastic")  # Different families
    ),
    seed=args.seed + 1
)
```

### Add More Lights

```python
# Add a fill light
fill_light = LightRandomizer(
    LightRandomCfg(
        light_name="fill_light",
        intensity=LightIntensityRandomCfg(intensity_range=(5000, 10000), enabled=True)
    ),
    seed=args.seed + 20
)
randomizers["light"].append(fill_light)
```

### Use Custom Scene

```python
# Replace ScenePresets.empty_room()
environment_layer = EnvironmentLayerCfg(
    elements=[
        ManualGeometryCfg(
            name="custom_floor",
            geometry_type="cube",
            size=(15.0, 15.0, 0.1),
            position=(0.0, 0.0, 0.005),
            default_material="roboverse_data/materials/arnold/Stone/Marble.mdl"
        ),
        # Add custom walls, ceiling, etc.
    ]
)
```

### Modify Randomization Frequency

```python
# In run_replay()
if step % args.randomize_interval == 0 and step > 0:
    apply_randomization(randomizers, args.level)
    
# Try different patterns:
# - Every N steps: step % N == 0
# - Only at specific steps: step in [60, 120, 180]
# - Probabilistic: if random.random() < 0.1
```

## Performance Considerations

### Memory Usage

The system is designed for efficient memory use:

- **Manual Geometry**: USD DefinePrim is idempotent. Calling create repeatedly on the same path returns the existing prim without duplication.

- **USD Assets**: The `_loaded_usds` cache tracks which assets are loaded at each path. Randomizing to the same USD skips reloading.

- **Materials**: Each object gets its own material binding, but the underlying material prims are reused when the same material is applied multiple times.

For long training runs, monitor memory if you observe growth. The current implementation is optimized for typical training scenarios (fixed object count, periodic randomization).

### Asset Download

First run can take a few minutes depending on your network:

- Table785 (5 tables): ~200MB
- Kujiale scenes (12 scenes): ~2GB (all meshes and textures)
- Desktop supplies (10 objects): ~100MB

Later runs use cached assets. To prefetch everything:

```bash
# Download all materials
huggingface-cli download RoboVerseOrg/roboverse_data materials --repo-type dataset --local-dir roboverse_data

# Download all Kujiale scenes
huggingface-cli download spatialverse/InteriorAgent kujiale_0003 --repo-type dataset --local-dir third_party/InteriorAgent
# Repeat for kujiale_0004, kujiale_0008, etc.
```

## Implementation Notes

### ObjectRegistry

ObjectRegistry gives unified access to all sim objects, whether they're static (Handler-managed) or dynamic (SceneRandomizer-managed).

```python
registry = ObjectRegistry.get_instance()
all_objects = registry.list_objects()
static_only = registry.list_objects(lifecycle='static')
dynamic_only = registry.list_objects(lifecycle='dynamic')
```

Material, Object, Light, and Camera randomizers query the registry automatically to find their targets. No need for manual prim path specification in most cases.

### Hybrid Simulation Support

Randomizers that need specific handlers (e.g., IsaacSim for USD ops) automatically get the right sub-handler in Hybrid mode. It just works:

```python
# Works in both standalone IsaacSim and Hybrid (IsaacLab + IsaacSim) modes
scene_rand = SceneRandomizer(cfg)
scene_rand.bind_handler(handler)  # Automatically uses render_handler if Hybrid
scene_rand()
```

### Why Dynamic Objects Are Visual-Only

Scene elements created by SceneRandomizer can't have physics - it's an IsaacLab constraint. Objects need to be registered with the Handler during init; anything added later is visual-only.

This works fine in practice:
- Background geometry (floors, walls) rarely needs physics anyway
- Tables can use invisible collision geometry if needed
- Task objects with physics are managed by the Handler

## Common Workflows

### Training Workflow

```python
# Set up randomizers once
scene_rand = SceneRandomizer(scene_cfg, seed=base_seed)
mat_rand = MaterialRandomizer(mat_cfg, seed=base_seed + 1)
light_rand = LightRandomizer(light_cfg, seed=base_seed + 2)

# Bind to environment
scene_rand.bind_handler(env.handler)
mat_rand.bind_handler(env.handler)
light_rand.bind_handler(env.handler)

# Training loop
for epoch in range(num_epochs):
    obs = env.reset()
    
    # Apply randomization at episode start
    scene_rand()
    mat_rand()
    light_rand()
    
    for step in range(max_steps):
        # Optional: Periodic mid-episode randomization
        if step % randomize_interval == 0 and step > 0:
            mat_rand()  # Vary materials during episode
            light_rand()  # Vary lighting during episode
        
        action = policy(obs)
        obs, reward, done, info = env.step(action)
```

### Evaluation Workflow

```python
# Level 0: Baseline (no randomization)
python 12_domain_randomization.py --level 0 --seed 42

# Levels 1-3: Progressive randomization
for level in [1, 2, 3]:
    python 12_domain_randomization.py --level {level} --seed 42
```

Compare videos to assess policy robustness across randomization complexity.

## Extending the System

### Add a New Randomizer

```python
# Create configuration
@configclass
class MyRandomCfg:
    obj_name: str
    my_param_range: tuple[float, float]

# Implement randomizer
class MyRandomizer(BaseRandomizerType):
    def __init__(self, cfg: MyRandomCfg, seed: int | None = None):
        super().__init__(seed=seed)
        self.cfg = cfg
    
    def __call__(self):
        obj = self.registry.get_object(self.cfg.obj_name)
        value = self.rng.uniform(*self.cfg.my_param_range)
        # Apply randomization...

# Use in demo
my_rand = MyRandomizer(cfg, seed=args.seed + 100)
my_rand.bind_handler(handler)
randomizers["custom"].append(my_rand)
```

### Add a New Asset Collection

```python
# In scene_presets.py
class SceneUSDCollections:
    @staticmethod
    def my_custom_assets(
        *,
        indices: list[int] | None = None,
        return_configs: bool = False
    ) -> list[str] | tuple[list[str], dict]:
        paths = [...]  # Your asset paths
        configs = {...}  # Optional per-asset configs
        
        if return_configs:
            return (paths, configs)
        return paths
```

Then use in demo:

```python
paths, configs = SceneUSDCollections.my_custom_assets(return_configs=True)
element = USDAssetPoolCfg(usd_paths=paths, per_path_overrides=configs)
```

## Troubleshooting

### Assets Not Downloading

Check network connectivity and HuggingFace access:

```bash
huggingface-cli whoami
```

If behind a firewall, you may need to prefetch assets manually.

### Materials Not Appearing

Ensure UV coordinates exist. The system auto-generates UVs for procedural geometry, but some USD meshes may lack them. Check IsaacSim logs for UV-related warnings.

### Objects Falling Through Table

This can happen if table bounds calculation fails. Check:

```python
table_bounds = scene_rand.get_table_bounds(env_id=0)
print(table_bounds)  # Should show reasonable height (e.g., 0.7m)
```

If bounds are invalid (e.g., astronomical numbers), the position update is skipped. This usually indicates an issue with the USD asset or bounding box computation.

## Summary

This tutorial covers a complete domain randomization setup:

- Four scene modes (manual to full USD)
- Four randomization levels (baseline to full perturbation)
- Automatic asset downloading and conversion
- Reproducible randomization with explicit seeds
- Clean separation between static/dynamic objects
- Unified object access via ObjectRegistry

Works well for both evaluation (short runs, visual diversity) and training (long runs, stability, reproducibility). Customize by editing configs, adding randomizers, or defining custom asset collections.
