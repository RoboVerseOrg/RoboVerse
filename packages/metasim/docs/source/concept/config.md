# Configuration System

## Overview

The **configuration system** in RoboVerse ensures that all simulation-related settings remain **simulator-agnostic**. Instead of embedding parameters directly into task logic, each simulation instance is fully defined by a **ScenarioCfg**.

A `ScenarioCfg` specifies both static and runtime properties of a simulation, including robots, objects, lights, cameras, scenes, rendering, and physics parameters. When passed to a **Handler**, it is instantiated into an actual running simulation environment.

This design enables:

* **Portability**: The same config works across multiple simulators.
* **Reusability**: A single config can be shared across tasks, benchmarks, or visualization tools.
* **Clarity**: Configurations are declarative, while execution details remain in handlers.

---

## Organization

### Root Config: `ScenarioCfg`

```python
@configclass
class ScenarioCfg:
    """Scenario configuration."""

    # Assets
    scene: SceneCfg | None = None
    robots: list[RobotCfg] = []
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    objects: list[BaseObjCfg] = []
    cameras: list[BaseCameraCfg] = []

    # Runtime
    render: RenderCfg = RenderCfg()
    sim_params: SimParamCfg = SimParamCfg()
    simulator: Literal["isaaclab","isaacgym","sapien2","sapien3",
                       "genesis","pybullet","mujoco"] | None = None
    renderer:  Literal["isaaclab","isaacgym","sapien2","sapien3",
                       "genesis","pybullet","mujoco"] | None = None

    # Misc
    num_envs: int = 1
    headless: bool = False
    env_spacing: float = 1.0
    decimation: int = 25
```

---

### Key Sections

| Section          | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| **scene**        | Global scene setup/                                                              |
| **robots**       | List of robot configs (`RobotCfg`), name,acutators, joint name&limit, init pose. |
| **lights**       | Lighting setup for rendering (default: `DistantLightCfg`).                       |
| **objects**      | Dynamic or static scene objects (`BaseObjCfg`).                                  |
| **cameras**      | Configurations for camera sensors (intrinsics, pose, type).                      |
| **render**       | Rendering options .                                                              |
| **sim_params**   | Physics parameters: timestep, solver settings, gravity, etc.                     |
| **simulator**    | Physics backend selection (`isaacgym`, `mujoco`, `sapien`, etc.).                |
| **renderer**     | Rendering backend (simulator for rendering).                                     |
| **num_envs**     | Number of parallel environments to instantiate.                                  |
| **headless**     | Run without viewer.                                                              |
| **env_spacing**  | Distance between environments when instantiated in parallel.                     |
| **decimation**   | Simulation decimation factor (steps per control action).                         |

### Validation at construction

Values that every backend would choke on later are rejected when the config is built (and inside
`ScenarioCfg.update`, which restores the previous value when the new one is rejected). The error names
the field, the value and what is accepted, e.g. `ScenarioCfg.num_envs=0 is invalid: expected an integer >= 1.`

| Field | Accepted |
| --- | --- |
| `ScenarioCfg.num_envs`, `decimation` | integers >= 1 (numpy / torch integer scalars are fine; floats and bools are not) |
| `ScenarioCfg.robots`, `objects`, `cameras`, `lights` | a list of configs (a tuple is converted; a bare config is rejected) |
| `SimParamCfg.dt` | `None` (the backend's default step) or a finite number > 0 |
| `SimParamCfg.substeps` | integer >= 1 |
| camera `width`, `height` | integers >= 1 |
| camera `pos`, `look_at` | three finite numbers |
| `PinholeCameraCfg.focal_length`, `horizontal_aperture` | finite numbers > 0 |

`from_dict` and direct attribute assignment bypass these checks; call `update(...)` to patch a config
with validation.

---

## Utility Methods

The class provides mechanisms for asset management and dynamic updates:

* **`__post_init__()`**
  Resolves string-based shortcuts (e.g., `"franka"` → `RobotCfg("franka")`) and fetches scene assets when the simulator is set.

* **`check_assets()`**
  Ensures that all referenced assets are available and automatically downloads missing files. Typically invoked when a handler instantiates the scenario.

* **`update(**kwargs)`**
  Dynamically patches fields, re-runs `__post_init__`, and returns the updated config. Useful for quick overrides (e.g., changing gravity, swapping robots).

  ------
##  What Does Not Belong in Config

  To keep `cfg/` clean and portable across tasks and RL settings, the following things are **explicitly excluded**:

  - Reward functions
  - Observation definitions
  - Success checkers
  - Task-level logic or termination conditions
  - Algorithm-specific parameters (policy type, optimizer, etc.)

  > These should all live in upper-level wrappers in Roboverse_learn



## Robot Configuration Specification

### Purpose

`RobotCfg` defines robots in a simulator-agnostic way: asset paths, joints, actuators, and control types. Handlers consume this config and adapt it to MuJoCo, IsaacLab/IsaacGym, Sapien, Genesis, or PyBullet.

`BaseActuatorCfg` specifies per-joint actuation properties (limits, stiffness/damping, EE flags).

---

### RobotCfg

* **name / num_joints**: Metadata.
* **usd_path / mjcf_path / urdf_path**: Asset file path.
* **fix_base_link / enabled_gravity**: Physical flags.
* **actuators**: Dict of joint → `BaseActuatorCfg`.
* **control_type**: Dict of joint → control mode.
* **joint_limits**
* **(Optional)  default_joint_positions, curobo_ref_cfg_name**.

---

### Minimal Setup Steps

1. Copy the template file and rename it.
2. Change `class` name and `name` attribute.
3. Set `num_joints`.
4. Add correct asset file path (USD/MJCF/URDF).
5. Configure actuator parameters (velocity/torque/stiffness/damping).
6. Define control types for each joint.
7. (Optional) Add joint limits and default positions.
8. Keep or adjust `fix_base_link` and `enabled_gravity` as needed.

---

### Example (Minimal)

```python
RobotCfg(
  name="robot_template",
  num_joints=2,
  urdf_path="roboverse_data/robots/your_robot/urdf/your_robot.urdf",
  fix_base_link=True,
  enabled_gravity=True,
  control_type={"joint1": "position", "joint2": "effort"},
  actuators={"joint1": BaseActuatorCfg(stiffness=500, damping=10),
             "joint2": BaseActuatorCfg(effort_limit_sim=50)}
)
```
