# Randomization

## Overview
Metasim itself is a deterministic simulation system. In order to mimic the uncertainties we met in real world, we use randomizers to create uncertainties.

There are two types of randomizers: 1) the ones that only changes the simulation result (noise adder), and 2) the ones that modifies the simulation iteself (light, mass, friction, etc.).

The first type of randomizers are easy to implement: simply write a wrapper around the simulation. The second type requires some tricks. We are now adopting the same approach as we constructed extra queries: use hooks to modify the simulation without changing the code of handlers.

<!-- Every type 2 randomizers will be called at `reset()` of a simulation. -->

## Architecture

A custom randomizer can be inherited from `metasim.randomizers.base.BaseRandomizerType`.

This class has only one important method you need to overwrite **__call__()** .

When an Extra Observation is used by a Task, it will be automatically bound to the underlying handler. You can then use `self.handler` to access and modify the handler instance within the Extra Observation class.

When `randomizer()` is called, it needs to do the randomization.

<!-- **Currently, the automaticall execution of randomizers in a task pipelien is not implemented.** -->

## Step-by-Step Usage Guide
We provide different kinds of randomizers under `metasim/randomization`. The availability of each randomization function in the simulator depends on whether the corresponding randomizer API is implemented in the simulator handler. You can run 

```
python3 ./get_started/12_domain_randomization.py --sim isaacsim --eval_level level3
```

to see the randomization effects in action. We also provide a four-level generalization benchmarking protocol. See the details below.



### Object Randomizers
Generally, we use this 
Use the unified `ObjectRandomizer` with predefined presets:

```python
# Create object randomizers with different presets
cube_randomizer = ObjectRandomizer(
    ObjectPresets.grasping_target("cube"), 
    seed=42
)
cube_randomizer.bind_handler(handler)

sphere_randomizer = ObjectRandomizer(
    ObjectPresets.bouncy_object("sphere"), 
    seed=42
)
sphere_randomizer.bind_handler(handler)

franka_randomizer = ObjectRandomizer(
    ObjectPresets.robot_base("franka"), 
    seed=42
)
franka_randomizer.bind_handler(handler)
```

###  Material Randomizers

Apply material randomization for visual and physical properties:

```python
# Material randomizers with different strategies
cube_material_randomizer = MaterialRandomizer(
    MaterialPresets.wood_object("cube", use_mdl=True, randomization_mode="combined"),
    seed=42,
)
cube_material_randomizer.bind_handler(handler)

sphere_material_randomizer = MaterialRandomizer(
    MaterialPresets.rubber_object("sphere", randomization_mode="combined"),
    seed=42,
)
sphere_material_randomizer.bind_handler(handler)
```

### Light Randomizers

Set up lighting randomization using scenarios or individual presets:

```python
# Option 1: Using LightScenarios for complex setups
light_configs = LightScenarios.indoor_room()
light_randomizers = []
for config in light_configs:
    randomizer = LightRandomizer(config, seed=42)
    randomizer.bind_handler(handler)
    light_randomizers.append(randomizer)

# Option 2: Using individual presets
# For DistantLight (outdoor sun)
sun_randomizer = LightRandomizer(
    LightPresets.distant_outdoor_sun("sun_light", randomization_mode="combined"),
    seed=42,
)
sun_randomizer.bind_handler(handler)

# For SphereLight (ceiling light)
ceiling_randomizer = LightRandomizer(
    LightPresets.sphere_ceiling_light("ceiling_light", randomization_mode="combined"),
    seed=43,
)
ceiling_randomizer.bind_handler(handler)
```

### Camera Randomizers

Configure camera randomization for different aspects:

```python
# Camera randomizer with different modes
camera_randomizer = CameraRandomizer(
    CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
    seed=42,
)
camera_randomizer.bind_handler(handler)
```

### Scene Randomizers

Add environmental elements like floors, walls, and tables:

```python
from metasim.randomization import SceneGeometryCfg, SceneMaterialPoolCfg, SceneRandomCfg
from metasim.randomization.presets.scene_presets import SceneMaterialCollections

# Configure scene elements
floor_cfg = SceneGeometryCfg(
    enabled=True,
    size=(10.0, 10.0, 0.01),
    position=(0.0, 0.0, 0.005),
    material_randomization=True,
)
floor_materials_cfg = SceneMaterialPoolCfg(
    material_paths=SceneMaterialCollections.floor_materials(),
    selection_strategy="random",
)

# Create scene randomizer
scene_cfg = SceneRandomCfg(
    floor=floor_cfg,
    floor_materials=floor_materials_cfg,
    only_if_no_scene=True,
)
scene_randomizer = SceneRandomizer(scene_cfg, seed=42)
scene_randomizer.bind_handler(handler)
```
### Randomization Level

The system supports systematic evaluation levels:

- **Level 0**: Task space 
- **Level 1**: + Environment randomization (scene geometry & materials)
- **Level 2**: + Lighting & reflection randomization 
- **Level 3**: + Camera randomization (position, orientation, intrinsics)

