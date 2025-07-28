# Achitecture Overview
## Metasim Overview

**Metasim** is a standalone simulator layer designed to provide a unified interface to different underlying physics backends (e.g. MuJoCo, Isaac). It is simulator-agnostic, and only contains code and configuration necessary for simulating scenes and extracting structured state information.

>  Its design principle:
>
> 1. **Metasim is a standalone simulation interface that supports multiple use cases.**
> 2. **Configurations only describe static, simulator-related properties.**
> 3. **New tasks should be easy to migrate or implement from scratch without modifying simulator logic.**

------

###  Directory Structure

Inside the `metasim/` folder:

| Folder/File               | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `cfg/`                    | Contains static py configs that define simulation-related properties — such as robot models, scenes, objects, and task setups. |
| `sim/`                    | Simulator-specific **handlers**. Each simulator has a handler that defines how to set, step, and get state. |
| `scripts/`                | Includes runnable tools that operate within Metasim  — e.g. trajectory replay, asset conversion. |
| `test/`                   | Contains consistency tests for handler behavior debug. In particular, it ensures information does not change after using `get_state()` and `set_state()` |
| `utils/`                  | Shared utility functions that Metasim uses internally        |
| `constants.py / types.py` | Global definitions for enums and shared constants used throughout the metasim system. |



------

###  Core Components

The **two most important folders** in Metasim are:

1. #### sim folder — Simulator Adapters

   The `sim/` module defines simulation-specific handlers that bridge between low-level simulators (like MuJoCo, IsaacGym) and RoboVerse’s unified task interface.

   Each simulator implements a handler class (e.g., `MJXHandler`, `IsaacHandler`) by inheriting from `BaseSimHandler`. These handlers are responsible for loading assets, stepping physics, setting/resetting environment state, and extracting structured state for upper layers.

   ------

   #####  Handler Lifecycle

   Every handler follows a common lifecycle:

   1. **Initialization (`__init__`)**:
       Receives a `ScenarioCfg` which includes simulation metadata such as robots, objects, sensors, lights, task checker, etc.
       It extracts these components and stores useful references like `self.robots`, `self.cameras`, and `self.object_dict`.
   2. **Launch (`launch`)**:
       This function builds the simulator model (e.g., loading MJCF/URDF files), compiles it, allocates buffers, and optionally initializes renderers or viewers.
   3. **Close (`close`)**
       Releases all simulator resources, such as viewers, renderers, or GPU memory buffers.

   ------

   #####  Key Interface Functions

   1. `get_state() → TensorState`

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

   4. `get_extras(env_ids=None) → dict[str, Tensor]`

   > **Purpose:** Returns **task-specific, non-standard information** not present in the core `TensorState`.

   Examples include:

   - Site positions

   - Contact forces

   - Body mass

   - IMU sensor data

     ......


   #### Usage overview

   The full pipeline looks like this:

   ```text
   Task.extra_spec()       # Declares what is needed
           │
           ▼
   SimHandler.get_extras()                 # Called by RL wrapper
           │
           ▼
   SimHandler.query_derived_obs(spec)      # Parses query dict
           │
           ▼
   Querier.query(query_obj, handler)       # Resolves each field
   ```

   #### Task-level declaration

   ```python
   from metasim.cfg.query_type import SitePos, SensorData, GeomCollision, BodyMass
   
   def extra_spec(self):
       return {
           "head_pos"        : SitePos(["head"]),
           "gyro_torso"      : SensorData("gyro_torso"),
           "torso_mass"      : BodyMass("torso"),
           "left_foot_touch" : GeomCollision("left_foot", "floor"),
       }
   ```

   #### Output from `get_extras()`

   The returned dictionary will look like:

   ```python
   {
       "head_pos":        Tensor of shape (N_env, 3),
       "gyro_torso":      Tensor of shape (N_env, 3),
       "torso_mass":      Tensor of shape (N_env,),  # scalar per env
       "left_foot_touch": Tensor of shape (N_env,),  # bool mask
   }
   ```

   Each value is resolved independently via the corresponding query type and handler logic.

   

2. #### cfg folder — Simulator Configuration

   ##### What Belongs in Config

   Each config file under `cfg/` specifies *only* information required to build and launch the simulation. This includes:

   | Key Section  | Purpose                                                      |
   | ------------ | ------------------------------------------------------------ |
   | `robots`     | List of robot instances, including model path (e.g. MJCF or URDF), initial pose, joint limits, etc. |
   | `objects`    | Static or dynamic scene objects, such as tables, cubes, buttons. Each has position, type, and optional fixations. |
   | `lights`     | Light source settings for visual fidelity or vision-based tasks (e.g. color, direction, intensity). |
   | `cameras`    | Camera positions and intrinsics, e.g., for RGB, depth, or offscreen rendering. |
   | `scene`      | Ground plane, friction, or other high-level environment descriptors. |
   | `sim_params` | Physics timestep, solver config, gravity toggle, etc.        |

   ------

   #####  What Does Not Belong in Config

   To keep `cfg/` clean and portable across tasks and RL settings, the following things are **explicitly excluded**:

   - Reward functions
   - Observation definitions
   - Success checkers
   - Task-level logic or termination conditions
   - Algorithm-specific parameters (policy type, optimizer, etc.)

   > These should all live in upper-level wrappers in Roboverse_learn

   ------

   #####  Integration with ScenarioCfg

   Every handler is initialized with a `ScenarioCfg` object parsed from these configs.
    The `ScenarioCfg` aggregates all static config elements (robot, objects, lights, etc.), and passes them to the simulation backend during launch.

   This decoupling ensures that you can:

   - Reuse one config across multiple RL tasks
   - Load the same config for visualization, trajectory replay, or debugging
   - Build new tasks without touching simulator configs


## RoboVerse Learn  Overeview

RoboVerse Learn consists of Task Wrappers and Learning Framework.  
Its goal is to present *one* standard interface that:

* Lets any algorithm (PPO, SAC, BC, etc.) work with any task  
* Hides simulator & task differences, so you can swap tasks, simulators or algorithms with minimal friction  

---

###  Design Principles

| #     | Principle                                | Key Points                                                   |
| ----- | ---------------------------------------- | ------------------------------------------------------------ |
| **1** | **Standardised Wrapper API**             | • `TaskWrapper` exposes `step / reset / _reward / _observation / _success`.<br>• Once an algorithm is connected to a single `TaskWrapper`, it can seamlessly switch to any other task simply by replacing the wrapper.<br/>• Upper‑level algorithms need not care whether the backend is MuJoCo, Isaac, etc. |
| **2** | **Minimise Task‑Migration Cost**         | • Add a task: just subclass / compose a wrapper.<br>• Switch simulator: wrappers/algorithms stay unchanged.<br>• Directory layout, Configs management（except the sim-related part）, training scripts all stay the same. |
| **3** | **Reusable Reward & Checker Primitives** | • Tasks build complex logic by *composing* primitives → no copy‑paste across tasks. |

---

### 1. Module Composition

| Sub‑module              | Responsibilities                                             |
| ----------------------- | ------------------------------------------------------------ |
| **Task Wrapper**        | • Combines a `Handler` & exposes `step / reset`.<br>• Assembles Reward / Observation / Success .<br>• Provides `pre_sim_step` & `post_sim_step` callbacks for *task‑level* DR. |
| **Handler (Metasim)**   | • `set_state / get_state / get_extras` unified across engines.<br>• *Physics‑level* DR (`pre_sim_step`).<br>• Pure simulator adapter—no algorithm logic. |
| **Learning Framework**  | • Any RL / IL algorithm.<br>• No simulator knowledge.        |
| **Custom Util Wrapper** | • Provide lightweight extensions (e.g., NumPy-to-Torch conversion, first-frame caching) to support logging, preprocessing, or offline data collection without modifying core task logic. |

---

### 2. Interface List

| Method                        | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `step(action)`                | Runs one simulation step: calls `pre_sim_step`, then `handler.simulate()`, then `post_sim_step`; returns `(obs, reward, done, info)` |
| `reset()`                     | Resets the environment and applies `reset_callback`, returns initial observation |
| `pre_sim_step()`              | (Optional) Hook for task-level domain randomization before simulation |
| `post_sim_step()`             | (Optional) Hook for post-processing (e.g., observation noise) |
| `get_state()` / `set_state()` | Unified simulator-agnostic state interface using `TensorState` |
| `get_extras(spec)`            | Returns task-specific quantities (e.g., site poses, contact forces) via query descriptors |

### 3. Domain Randomisation Layers

| Layer             | Location                      | Examples                                             |
| ----------------- | ----------------------------- | ---------------------------------------------------- |
| **Physics‑level** | `Handler`                     | Friction, mass, light, material                      |
| **Task‑level**    | `Wrapper.pre/post_sim_step()` | Action noise, observation noise, initial‑pose jitter |

*Rule:* Simulator parameters → Handler; task‑coupled noise → Wrapper.

------

### 4.Migrating a New Task into RoboVerse

We support two ways to bring an external task into the RoboVerse Learn pipeline:

#### Approach 1: Direct Integration (Quick Migration)

The fastest way to integrate a new task is to:

1. **Copy the task codebase** (from an external repo) into `roboversa_learn/`
2. Replace any simulator-specific API calls with `Handler` equivalents
3. Convert raw observations into RoboVerse `TensorState` via `get_state()`
4. Move simulator-related config (e.g. robot model path, asset layout, `dt`, `decimation`, `n_substeps`) into `ScenarioCfg` and Metasim config files

This transforms the original task into a RoboVerse-compatible format while preserving its logic and structure.

**Cross-simulator support is now enabled for this task.**

####  Approach 2: Structured Wrapper Integration

To enable better reuse and cross-task comparison:

1. **Subclass `BaseTaskWrapper`**
2. Implement standardized interfaces: `_reward()`, `_observation()`, `_terminated()`
3. Use callbacks (`pre_sim_step`, `post_sim_step`, `reset_callback`) as needed
4. Leverage existing `Handler` and `ScenarioCfg` setup from Approach 1

This approach supports full compatibility with:

- **Multi-task learning benchmarks**
- **One-click algorithm switching**
- **Clean architectural separation between task, sim, and learning logic**

------

>  With either approach, you can quickly benchmark new tasks under different simulators or algorithms — with no boilerplate or duplicate integration.