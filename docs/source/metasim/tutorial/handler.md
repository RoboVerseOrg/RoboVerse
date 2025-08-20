
# Metasim Handlers

## Architecture and Philosophy

A **handler** is a basic class that transforms a scenario config into a simulated instance. We support multiple handler backends including `mujoco`, `mujoco-mjx`, `isaacgym`, `isaaclab`, `isaacsim`, `pybullet`, `genesis`, `sapien` and more.


## Handler Lifecycle

Every handler follows a common lifecycle:

1. **Initialization (`__init__`)**:
    Receives a `ScenarioCfg` which includes simulation metadata such as robots, objects, sensors, lights, task checker, etc.
    It extracts these components and stores useful references like `self.robots`, `self.cameras`, and `self.object_dict`.
2. **Launch (`launch`)**:
    This function builds the simulator model (e.g., loading MJCF/URDF files), compiles it, allocates buffers, and optionally initializes renderers or viewers.
3. **Close (`close`)**
    Releases all simulator resources, such as viewers, renderers, or GPU memory buffers.

------

##  Key Interface Functions

1. `get_state() → TensorState | DictEnvState`

> **Purpose:** Extracts structured simulator state for all robots, objects, and cameras into a unified `TensorState` data structure.

This includes:

- Root position/orientation of each object
- Joint positions & velocities
- Actuator states
- Camera outputs (RGB / depth)
- Optional per-task "extras"

It supports multi-env batched extraction, and ensures consistent structure across backends.

------

2. `set_state(ts: TensorState)`

> **Purpose:** Restores or manually sets the simulator state using a full `TensorState` snapshot.

This is often used for:

- Episode resets to a known state
- State injection during training
- Replaying trajectories

Internally this maps the unified `TensorState` back to simulator-specific structures (`qpos`, `qvel`, `ctrl`, etc.)

------

3. `simulate()`

> **Purpose:** Executes the physics update (step function) in the simulator.

This is typically called after applying actions or updating the state. It may involve multiple substeps (based on decimation rate) and handles model-specific quirks.

------
