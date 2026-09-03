# Domain Randomization System

## Overview

RoboVerse provides a comprehensive domain randomization system designed to bridge the sim-to-real gap by introducing controlled variability across scene composition, material properties, lighting conditions, and camera parameters. The system is built on a principled architecture that separates object lifecycle management from property editing, enabling flexible composition and reproducible experiments.

Domain randomization in RoboVerse operates at multiple levels:
- **Scene-level**: Creating, deleting, and switching between different environments, workspaces, and distractor objects
- **Material-level**: Varying visual appearance (textures, colors, reflectance) and physical properties (friction, restitution)
- **Lighting-level**: Adjusting intensity, color temperature, position, and orientation of light sources
- **Camera-level**: Perturbing intrinsic and extrinsic parameters to simulate sensor variations
- **Object-level**: Randomizing mass, friction, and initial pose of rigid bodies

All randomizers share a consistent interface and integrate seamlessly with the simulation handler, supporting both standalone operation and coordinated multi-randomizer workflows.

## Architectural Principles

### Separation of Concerns

The refactored system distinguishes between two fundamentally different operations:

**Lifecycle Management** - Creating, deleting, and switching objects:
- Handled by `SceneRandomizer`
- Operates directly on USD Stage
- Manages dynamic objects that can appear and disappear
- Examples: Switching between different table models, adding/removing distractor objects

**Property Editing** - Modifying attributes of existing objects:
- Handled by specialized randomizers (`MaterialRandomizer`, `ObjectRandomizer`, etc.)
- Operates through Handler APIs and USD properties
- Works on both static and dynamic objects
- Examples: Changing material roughness, adjusting light intensity

This separation ensures clean interfaces and prevents the complexity explosion that occurs when lifecycle and properties are mixed.

### Object Classification

**Static Objects**:
- Managed by the simulation Handler
- Created during Handler initialization
- Include robots, task objects, cameras, lights
- Have full physics simulation capabilities
- Cannot be added after Handler initialization (architectural constraint)

**Dynamic Objects**:
- Managed by SceneRandomizer
- Created and deleted at runtime via USD Stage manipulation
- Include background geometry, workspace surfaces, distractors
- Visual-only (no physics simulation)
- Registered in ObjectRegistry for unified access

This classification reflects the underlying constraints of physics engines and rendering systems. Static objects participate in physics simulation, while dynamic objects provide visual context and can be freely modified.

### Unified Object Access

The `ObjectRegistry` provides a central database of all simulation objects:

```python
registry = ObjectRegistry.get_instance()

# Query objects
all_objects = registry.list_objects()
static_only = registry.list_objects(lifecycle='static')
dynamic_only = registry.list_objects(lifecycle='dynamic')

# Get object metadata
meta = registry.get_object_metadata("table")
prim_paths = meta.prim_paths  # USD paths for this object
```

Randomizers automatically query the registry to locate their target objects, eliminating the need for manual prim path management in most cases. The registry bridges static and dynamic objects, providing a unified interface regardless of how objects were created.

## Core Components

### BaseRandomizerType

All randomizers inherit from this base class, which provides:

**RNG Management**:
```python
class MyRandomizer(BaseRandomizerType):
    def __init__(self, cfg, seed=None):
        super().__init__(seed=seed)
        # self._rng is now available
```

Each randomizer maintains an independent `random.Random` instance seeded with the provided value. This ensures reproducibility and prevents interference between randomizers.

**Handler Binding**:
```python
randomizer.bind_handler(handler)
```

Binding connects the randomizer to the simulation environment. The base class automatically handles Hybrid simulation mode by selecting the appropriate sub-handler based on the randomizer's `REQUIRES_HANDLER` attribute.

**ObjectRegistry Integration**:

The first randomizer to bind initializes the ObjectRegistry and scans existing Handler objects. Subsequent randomizers reuse the registry, avoiding duplicate scans.

### SceneRandomizer

Manages dynamic object lifecycle through a three-layer hierarchy:

**Layer 0 - Environment**: Background geometry
- Floors, walls, ceilings
- Complete interior scenes (Kujiale)
- Large-scale background elements

**Layer 1 - Workspace**: Manipulation surfaces
- Tables, desks, countertops
- Platforms for object placement

**Layer 2 - Objects**: Distractors and decorations
- Fruits, books, office supplies
- Visual clutter for robustness testing

Each layer can contain multiple elements, where each element is either:

**Manual Geometry**:
```python
ManualGeometryCfg(
    name="floor",
    geometry_type="cube",
    size=(10.0, 10.0, 0.1),
    position=(0.0, 0.0, 0.005),
    default_material="roboverse_data/materials/arnold/Carpet/Carpet_Beige.mdl"
)
```

Creates procedural shapes (cube, sphere, cylinder, plane) with optional default material. The geometry is created using USD primitives, and materials are applied via the IsaacSimAdapter.

**USD Assets**:
```python
USDAssetCfg(
    name="table",
    usd_path="EmbodiedGenData/dataset/basic_furniture/table/uuid.urdf",
    position=(0.0, 0.0, 0.0),
    scale=(1.2, 1.5, 1.0)
)
```

Loads external assets from USD or URDF files. The system automatically:
- Downloads missing assets from HuggingFace
- Converts URDF to USD using IsaacLab's converters
- Caches converted assets for subsequent use

**USD Asset Pools**:
```python
USDAssetPoolCfg(
    name="table_pool",
    usd_paths=[...],  # Multiple USD files
    selection_strategy="random",
    per_path_overrides={...}  # Per-asset calibrations
)
```

Randomly selects from multiple assets, enabling scene diversity. Each selection can have custom position, rotation, and scale overrides.

### MaterialRandomizer

Applies visual and physical material properties to objects.

**Visual Materials** (MDL):

The system supports NVIDIA MDL materials with full texture support:

```python
MaterialRandomCfg(
    obj_name="box_base",
    mdl=MDLMaterialCfg(
        mdl_paths=[...],  # Paths to MDL files
        randomize_material_variant=True  # Select from variants within files
    )
)
```

Material application includes:
- Automatic MDL and texture downloading
- UV coordinate generation for procedural geometry
- Material binding using IsaacSim's CreateMdlMaterialPrim API
- Support for material variants within MDL files

**Physical Materials** (optional):

```python
PhysicalMaterialCfg(
    friction_range=(0.3, 0.7),
    restitution_range=(0.1, 0.3),
    enabled=True
)
```

Modifies friction and restitution on physics-enabled objects. Note that dynamic objects (created by SceneRandomizer) are visual-only and skip physical material randomization.

**Material Families**:

The preset system organizes materials into logical families:

```python
MaterialPresets.mdl_family_object("table", family=("wood", "stone", "metal"))
```

Families are resolved from the MDL collection registry, which indexes materials by category (architecture, wood, metal, stone, fabric, etc.). This provides high-level control without requiring explicit file paths.

### ObjectRandomizer

Randomizes physics properties of static objects:

```python
ObjectRandomCfg(
    obj_name="box_base",
    physics=PhysicsRandomCfg(
        mass_range=(10.0, 30.0),
        friction_range=(0.3, 0.8),
        enabled=True
    ),
    pose=PoseRandomCfg(
        position_range=[(-0.1, 0.1), (-0.1, 0.1), (0, 0)],
        rotation_range=(0, 30),  # Degrees
        enabled=False  # Disabled in trajectory replay
    )
)
```

Physics randomization only applies to objects with `RigidBodyAPI`. The randomizer queries the ObjectRegistry to determine if an object has physics before attempting modifications.

### LightRandomizer

Controls lighting parameters with support for multiple light types:

```python
LightRandomCfg(
    light_name="ceiling_main",
    intensity=LightIntensityRandomCfg(
        intensity_range=(16000, 30000),
        enabled=True
    ),
    color=LightColorRandomCfg(
        temperature_range=(3000, 6000),  # Kelvin
        use_temperature=True,
        enabled=True
    ),
    position=LightPositionRandomCfg(
        position_range=((-1, 1), (-1, 1), (-0.2, 0.2)),
        relative_to_origin=True,
        enabled=True
    ),
    orientation=LightOrientationRandomCfg(
        angle_range=((-20, 20), (-20, 20), (-180, 180)),
        relative_to_origin=True,
        enabled=True
    )
)
```

Supported light types: DomeLight, DistantLight, SphereLight, DiskLight, CylinderLight, RectLight.

Color can be specified either as RGB values or as color temperature in Kelvin. The temperature mode is often more intuitive for realistic lighting (2700K = warm incandescent, 6500K = cool daylight).

### CameraRandomizer

Perturbs camera parameters to simulate sensor variations:

```python
CameraRandomCfg(
    camera_name="main_camera",
    position=CameraPositionRandomCfg(
        delta_range=((-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05)),
        use_delta=True,
        enabled=True
    ),
    orientation=CameraOrientationRandomCfg(
        rotation_delta=((-5, 5), (-5, 5), (-5, 5)),
        enabled=True
    ),
    look_at=CameraLookAtRandomCfg(
        look_at_delta=((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)),
        use_delta=True,
        enabled=True
    ),
    intrinsics=CameraIntrinsicsRandomCfg(
        fov_range=(45, 60),
        use_fov=True,
        enabled=True
    )
)
```

Position and orientation randomization can operate in absolute or delta mode. Delta mode is recommended for small perturbations around the nominal camera pose.

## Asset Management

### Automatic Asset Downloading

When a material or USD asset is requested but not found locally, the system downloads it automatically from HuggingFace:

**MDL Materials**:
- Repository: `RoboVerseOrg/roboverse_data`
- Pattern: `materials/arnold/{family}/{file}.mdl`
- Includes: MDL shader + all referenced textures (albedo, normal, roughness)

**EmbodiedGen Assets** (Tables, Desktop Objects):
- Repository: `HorizonRobotics/EmbodiedGenData`
- Pattern: `dataset/{category}/{subcategory}/{uuid}/`
- Downloads: Complete folder (URDF + mesh/*.obj + textures/)
- Conversion: URDF → USD automatically

**Kujiale Scenes**:
- Primary: `RoboVerseOrg/roboverse_data` (scene USDA files)
- Assets: `spatialverse/InteriorAgent` (meshes, textures, materials)
- Strategy: Download RoboVerse USDA + InteriorAgent assets, then copy USDA to InteriorAgent folder for correct relative reference resolution

All downloads use `snapshot_download` with caching. Repeated requests for the same asset are instant.

### URDF to USD Conversion

EmbodiedGen provides furniture and objects as URDF files. The system converts these to USD automatically:

```python
# User requests
usd_path = "EmbodiedGenData/.../table/uuid.urdf"

# System performs
1. Check if uuid.usd exists (converted cache)
2. If not, download complete asset folder
3. Convert using AssetConverterFactory with MESH source type
4. Save as uuid.usd in same directory
5. Load the USD file
```

The conversion preserves mesh geometry and creates proper material bindings. Physics properties are stripped since scene objects are visual-only.

### Material Variant System

Many MDL files contain multiple material definitions. For example:
- `Rug_Carpet.mdl`: 4 variants (Base, Lines, Hexagonal, Honeycomb)
- `Caoutchouc.mdl`: 93 variants (different colors and finishes)
- `Wood.mdl`: 20+ wood species

The system supports two-stage randomization:

1. **Select MDL file** from configured pool
2. **Select variant** within that file

This is controlled by `randomize_material_variant`:

```python
# Default: Random variant selection
MDLMaterialCfg(mdl_paths=[...], randomize_material_variant=True)

# Explicit variant specification
mdl_paths=["path/to/Wood.mdl::Oak"]

# Always use first variant in file
randomize_material_variant=False
```

Variant selection is deterministic given a seed. The system parses MDL files to extract all available material names and uses the randomizer's RNG to select one.

## Usage Patterns

### Basic Workflow

```python
# 1. Create randomizer with configuration
scene_cfg = ScenePresets.tabletop_workspace(
    room_size=10.0,
    wall_height=5.0
)
scene_rand = SceneRandomizer(scene_cfg, seed=42)

# 2. Bind to simulation handler
scene_rand.bind_handler(env.handler)

# 3. Apply randomization
scene_rand()  # Creates scene with randomized materials
```

### Progressive Randomization

Start simple and add complexity incrementally:

```python
# Level 0: Baseline (deterministic)
scene_rand()  # Fixed materials

# Level 1: Add material randomization
scene_rand()
mat_rand()

# Level 2: Add lighting
scene_rand()
mat_rand()
light_rand()

# Level 3: Add camera
scene_rand()
mat_rand()
light_rand()
cam_rand()
```

This staged approach helps isolate the impact of each randomization component.

### Coordinated Randomization

Multiple randomizers can operate on the same scene:

```python
# Initialize all randomizers
randomizers = {
    "scene": SceneRandomizer(scene_cfg, seed=42),
    "material": MaterialRandomizer(mat_cfg, seed=43),
    "object": ObjectRandomizer(obj_cfg, seed=44),
    "light": [LightRandomizer(light_cfg, seed=45+i) for i in range(5)],
    "camera": CameraRandomizer(cam_cfg, seed=50)
}

# Bind all
for rand_list in randomizers.values():
    if isinstance(rand_list, list):
        for rand in rand_list:
            rand.bind_handler(handler)
    else:
        rand.bind_handler(handler)

# Apply all
def apply_all(level):
    randomizers["scene"]()
    if level >= 1:
        randomizers["material"]()
    if level >= 2:
        for light in randomizers["light"]:
            light()
    if level >= 3:
        randomizers["camera"]()
```

### Training Integration

```python
class MyTask(BaseTask):
    def __init__(self, ...):
        super().__init__(...)
        
        # Create randomizers
        self.scene_rand = SceneRandomizer(cfg, seed=42)
        self.mat_rand = MaterialRandomizer(cfg, seed=43)
        
        # Bind after Handler initialization
        self.scene_rand.bind_handler(self.handler)
        self.mat_rand.bind_handler(self.handler)
    
    def reset(self, env_ids=None):
        # Apply randomization at episode start
        self.scene_rand()
        self.mat_rand()
        
        return super().reset(env_ids)
```

## Scene Randomization in Detail

### Three-Layer Hierarchy

The `SceneRandomCfg` organizes scene elements into three semantic layers:

```python
from metasim.randomization import (
    SceneRandomCfg,
    EnvironmentLayerCfg,
    WorkspaceLayerCfg,
    ObjectsLayerCfg,
    ManualGeometryCfg,
    USDAssetPoolCfg
)

scene_cfg = SceneRandomCfg(
    environment_layer=EnvironmentLayerCfg(
        elements=[
            ManualGeometryCfg(name="floor", ...),
            ManualGeometryCfg(name="walls", ...),
        ],
        shared=True,  # Single instance for all environments
        z_offset=0.0
    ),
    workspace_layer=WorkspaceLayerCfg(
        elements=[
            USDAssetPoolCfg(name="table", usd_paths=[...])
        ],
        shared=True
    ),
    objects_layer=ObjectsLayerCfg(
        elements=[
            USDAssetPoolCfg(name="distractors", usd_paths=[...])
        ],
        shared=False  # Different objects per environment
    )
)
```

Layers process in order: Environment → Workspace → Objects. The `shared` flag controls whether elements are instantiated once (shared across all environments) or per-environment (for independent scenes).

### Manual Geometry Creation

Procedural shapes are created using USD primitives:

```python
ManualGeometryCfg(
    name="table",
    geometry_type="cube",
    size=(1.8, 1.8, 0.1),  # x, y, z dimensions
    position=(0.0, 0.0, 0.65),
    rotation=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w, x, y, z)
    add_collision=True,  # Add CollisionAPI for spatial queries
    default_material="path/to/material.mdl"  # Optional default appearance
)
```

The system creates the geometry using `UsdGeom.Cube.Define()` and applies scale transformations to achieve the desired dimensions. A default material can be specified for initial appearance, though material randomization should be handled separately by MaterialRandomizer for maximum flexibility.

### USD Asset Loading

External assets are loaded via USD references:

```python
USDAssetCfg(
    name="table",
    usd_path="path/to/table.usd",
    position=(0.0, 0.0, 0.37),
    rotation=(1.0, 0.0, 0.0, 0.0),
    scale=(1.2, 1.5, 1.0),
    auto_download=True
)
```

For URDF files, the system:
1. Downloads the complete asset folder (URDF + meshes + textures)
2. Converts URDF to USD using MeshConverter
3. Caches the USD file
4. Loads the USD file with proper transforms

### USD Asset Pools and Randomization

Asset pools enable geometric diversity:

```python
USDAssetPoolCfg(
    name="table",
    usd_paths=[
        "path/to/table1.urdf",
        "path/to/table2.urdf",
        "path/to/table3.urdf"
    ],
    per_path_overrides={
        "table1.urdf": {"position": (0, 0, 0.37), "scale": (1.2, 1.5, 1.0)},
        "table2.urdf": {"position": (0.3, 0, 0.37), "scale": (1.2, 1.4, 1.0)}
    },
    selection_strategy="random"  # or "sequential"
)
```

Each call to `scene_rand()` selects a random asset from the pool. The system tracks which USD is currently loaded at each prim path and performs replacement only when necessary:

```python
# First call: Loads table1
scene_rand()  # Creates table1 at /World/scene_workspace_table

# Second call: Randomly selects table2
scene_rand()
# Detects different USD → deletes old prim → loads new USD
# Result: table2 replaces table1 seamlessly
```

To avoid consecutive repetition, the random strategy filters out the currently loaded asset when pool size exceeds 1.

## Material Randomization in Detail

### MDL Material Workflow

```python
from metasim.randomization import MaterialRandomizer, MaterialPresets

# Using preset
mat_rand = MaterialRandomizer(
    MaterialPresets.mdl_family_object("box_base", family=("wood", "metal")),
    seed=123
)

# Or manual configuration
from metasim.randomization import MaterialRandomCfg, MDLMaterialCfg

mat_rand = MaterialRandomizer(
    MaterialRandomCfg(
        obj_name="box_base",
        mdl=MDLMaterialCfg(
            mdl_paths=["path/to/Wood.mdl", "path/to/Metal.mdl"],
            randomize_material_variant=True
        )
    ),
    seed=123
)

mat_rand.bind_handler(handler)
mat_rand()  # Selects random MDL + random variant + applies
```

The randomizer:
1. Selects a random MDL file from the pool
2. If `randomize_material_variant=True`, lists all variants in that file and selects one
3. Downloads MDL and textures if missing
4. Generates UV coordinates if needed
5. Creates material prim and binds to object

### Dynamic Object Material Randomization

A key feature of the refactored system is the ability to randomize materials on dynamic objects:

```python
# Manual table created by SceneRandomizer
scene_rand()  # Creates table with default Plywood material

# Later: Randomize table material
table_mat = MaterialRandomizer(
    MaterialPresets.mdl_family_object("table", family=("wood", "metal")),
    seed=99
)
table_mat.bind_handler(handler)
table_mat()  # Changes table material to random wood or metal
```

This works because ObjectRegistry tracks both static and dynamic objects, providing MaterialRandomizer with the necessary prim paths.

### Material Collections

The preset system provides curated collections:

```python
# Architectural materials (concrete, brick, tiles, etc.)
MaterialPresets.mdl_family_object("wall", family="architecture")

# Multiple families
MaterialPresets.mdl_family_object("floor", family=("carpet", "wood", "stone"))

# Specific collection
from metasim.randomization.presets import MDLCollections
paths = MDLCollections.family("metal")  # All metal materials
```

Collections are dynamically resolved from:
- Local directory scan (`roboverse_data/materials/`)
- HuggingFace manifest (if available)

This remote-first strategy ensures you get the complete asset list even if you only have a few files downloaded locally.

## Reproducibility and Seeding

### Seed Management

Every randomizer accepts an explicit seed:

```python
base_seed = 42

scene_rand = SceneRandomizer(cfg, seed=base_seed)
mat_rand = MaterialRandomizer(cfg, seed=base_seed + 1)
obj_rand = ObjectRandomizer(cfg, seed=base_seed + 2)
```

Using different offsets ensures independent randomization streams while maintaining global reproducibility. Running the same script with the same base seed produces identical results.

### Shared Seeds for Consistency

Sometimes you want related objects to share randomization:

```python
# All walls get the same material
wall_seed = base_seed + 100
for wall_name in ["wall_front", "wall_back", "wall_left", "wall_right"]:
    wall_mat = MaterialRandomizer(
        MaterialPresets.mdl_family_object(wall_name, family="masonry"),
        seed=wall_seed  # Same seed for all
    )
    wall_mat.bind_handler(handler)
    wall_mat()
```

This ensures visual coherence (all walls match) while still randomizing which specific material is selected.

### Internal RNG

Each randomizer maintains `self._rng = random.Random(seed)`, an independent random number generator. This prevents interference between randomizers and allows precise control over randomization order.

For reproducibility, always provide explicit seeds. If `seed=None`, the randomizer derives a seed from Python's global `random`, which may not be seeded in your script.

## Advanced Topics

### Hybrid Simulation Support

RoboVerse supports Hybrid mode, where IsaacLab manages physics and IsaacSim handles rendering. Randomizers that require specific capabilities declare their needs:

```python
class SceneRandomizer(BaseRandomizerType):
    REQUIRES_HANDLER = "render"  # Needs IsaacSim for USD operations

class ObjectRandomizer(BaseRandomizerType):
    REQUIRES_HANDLER = "physics"  # Needs IsaacLab for physics APIs
```

When binding in Hybrid mode, the base class automatically selects the appropriate sub-handler. This is transparent to users.

### Per-Environment Randomization

For vectorized training with different randomization per environment:

```python
# Create scene per-environment
scene_cfg = SceneRandomCfg(
    workspace_layer=WorkspaceLayerCfg(
        shared=False,  # Different table per env
        elements=[USDAssetPoolCfg(...)]
    )
)

# Randomize specific environments
scene_rand(env_ids=[0, 2, 4])  # Only randomize even envs
```

### Material Property Modification

Beyond swapping entire materials, you can modify specific properties:

```python
# Get current material properties
props = mat_rand.get_material_properties("box_base")
# Returns: {"roughness": 0.5, "metallic": 0.0, ...}

# Modify
props["roughness"] = 0.8

# Apply back
mat_rand.set_material_properties("box_base", props)
```

This get-modify-set pattern preserves other material attributes while changing specific ones.

### Custom Presets

Define reusable configurations:

```python
# In your code
class MyPresets:
    @staticmethod
    def industrial_workspace():
        return SceneRandomCfg(
            environment_layer=EnvironmentLayerCfg(
                elements=[
                    ManualGeometryCfg(
                        name="floor",
                        size=(20.0, 20.0, 0.1),
                        default_material="roboverse_data/materials/arnold/Concrete/Concrete_Polished.mdl"
                    ),
                    # ... walls, ceiling
                ]
            ),
            workspace_layer=WorkspaceLayerCfg(
                elements=[
                    USDAssetPoolCfg(
                        name="workbench",
                        usd_paths=[...]  # Industrial tables
                    )
                ]
            )
        )

# Use in task
scene_rand = SceneRandomizer(MyPresets.industrial_workspace(), seed=42)
```

## Performance and Scalability

### Memory Efficiency

The system is designed for memory-stable operation:

**USD Primitives**: The `DefinePrim()` API is idempotent. Calling it repeatedly on the same path returns the existing prim without duplication. Manual geometry randomization has zero memory growth.

**USD Assets**: The `_loaded_usds` cache tracks which assets are loaded at each path. Randomizing to the same asset skips reloading. Only asset switching (intentional) incurs the cost of deletion and recreation.

**Materials**: Each object receives its own material binding, but underlying material prims are reused when the same material is applied to multiple objects or the same object repeatedly. For PBR materials, `UsdShade.Material.Define()` returns the existing material if the path matches.

### Computational Cost

Randomization overhead per call:

- **SceneRandomizer**: O(elements) - Checks each element, creates/skips based on cache
- **MaterialRandomizer**: O(prims) - Applies material to each target prim
- **ObjectRandomizer**: O(objects × properties) - Modifies each randomized property
- **LightRandomizer**: O(1) - Single light update
- **CameraRandomizer**: O(1) - Single camera update

For typical scenes (10-20 static objects, 5-10 dynamic elements, 5 lights, 1 camera), total randomization time is under 100ms on modern GPUs.

### Training Recommendations

**Short runs** (demo, evaluation):
- Any configuration works
- Memory and performance are not concerns

**Medium runs** (1-2 hour training):
- Use `shared=True` for scene layers when possible
- Fixed object count recommended
- Monitor memory if using many per-environment objects

**Long runs** (8+ hour training):
- Profile memory after first hour
- If growth exceeds 500MB/hour, investigate
- Consider reducing randomization frequency if needed

The current implementation is validated for runs up to 24 hours with fixed object counts.

## Troubleshooting

### Assets Not Loading

**Symptom**: USD references show as empty Xforms

**Causes**:
1. Missing mesh files for URDF assets
2. Network download failure
3. URDF conversion error

**Solutions**:
- Check logs for download error messages
- Verify HuggingFace connectivity: `huggingface-cli whoami`
- Try manual download: `huggingface-cli download HorizonRobotics/EmbodiedGenData dataset/basic_furniture/table/uuid --local-dir EmbodiedGenData`
- Check that mesh files exist: `ls EmbodiedGenData/.../mesh/`

### Materials Not Displaying

**Symptom**: Objects show solid colors instead of textures

**Causes**:
1. Missing UV coordinates
2. Texture files not downloaded
3. Material binding failed

**Solutions**:
- Check logs for "Failed to generate UVs" warnings
- Verify texture files exist alongside MDL
- For procedural geometry, UV generation should be automatic
- For USD meshes, ensure the source asset includes UVs

### Randomization Not Triggering

**Symptom**: Scene looks identical across randomization calls

**Causes**:
1. Using `selection_strategy="sequential"` with same index
2. Random selection happened to pick the same asset
3. Material pool has only one option

**Solutions**:
- Verify pool contains multiple options: `len(cfg.mdl_paths) > 1`
- Check logs for "Replacing USD" or "Selected material" messages
- For sequential strategy, ensure counter increments
- For random strategy, system now avoids consecutive repeats automatically

### Objects Falling or Misplaced

**Symptom**: Robot or task objects fall through table or appear at wrong height

**Causes**:
1. Table bounds calculation failed
2. Position update skipped due to invalid bounds
3. USD asset has unexpected origin

**Solutions**:
- Check table bounds: `scene_rand.get_table_bounds(env_id=0)`
- Bounds should be reasonable (height ~0.7m for typical table)
- If bounds show astronomical numbers, the USD bounding box is invalid
- For manual geometry, bounds are computed from configuration (always reliable)

## Best Practices

### Design Principles

**Separate lifecycle from properties**: Use SceneRandomizer to manage what exists, and property randomizers to modify how things look and behave.

**Use explicit seeds**: Always provide seeds for reproducibility. Deriving seeds with offsets (base+1, base+2) keeps streams independent.

**Leverage presets**: Start with `ScenePresets` and `MaterialPresets` before writing custom configurations. Presets encode best practices and handle common use cases.

**Test progressively**: Begin with Level 0 (baseline), then add complexity one layer at a time (materials, lights, camera). This isolates issues and quantifies each component's impact.

**Monitor in production**: Add memory and time profiling when deploying to training clusters. The system is designed for stability, but validation on your specific hardware is recommended.

### Common Patterns

**Deterministic baseline**:
```python
# Level 0: Fixed materials, no variation
scene_rand = SceneRandomizer(cfg, seed=42)
scene_rand()  # Creates scene
# No further randomization calls
```

**Periodic randomization**:
```python
for step in range(max_steps):
    if step % interval == 0:
        scene_rand()
        mat_rand()
```

**Episode-level randomization**:
```python
for episode in range(num_episodes):
    # Randomize at start
    scene_rand()
    mat_rand()
    
    # Fixed scene for entire episode
    for step in range(episode_length):
        ...
```

**Curriculum learning**:
```python
def apply_randomization(epoch):
    scene_rand()
    
    if epoch > 100:
        mat_rand()  # Add materials after 100 epochs
    
    if epoch > 200:
        light_rand()  # Add lighting after 200 epochs
```

## Summary

The domain randomization system provides:

- **Clean architecture**: Separation between lifecycle (SceneRandomizer) and properties (specialized randomizers)
- **Unified access**: ObjectRegistry bridges static and dynamic objects
- **Comprehensive coverage**: Scene, material, object, light, camera randomization
- **Asset integration**: Automatic downloading and conversion from HuggingFace repositories
- **Reproducibility**: Explicit seed management with independent RNG streams
- **Flexibility**: Manual geometry, USD assets, URDF conversion, material variants
- **Performance**: Optimized for training stability with minimal overhead

The system supports both evaluation workflows (short runs with maximum diversity) and training workflows (long runs with reproducible randomization). Customize by composing randomizers, defining custom presets, or extending the base classes.

For practical examples, see `get_started/12_domain_randomization.py` and the detailed tutorial in the quick-start guide.
