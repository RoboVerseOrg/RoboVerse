# Task System

## 1.Overview

In RoboVerse, a **task** is a wrapper built on top of a Handler and exposes **Gym-style APIs** (`step`, `reset`, etc.).

* **Simulation contents** (robots, objects, scene, physics params) live in a `ScenarioCfg` and are instantiated by a Handler.
* **Task logic** (reward, observation, termination, etc.) is layered on top via wrappers.
* This enforces clean separation between simulation, task, and algorithm.

A task is created with:

* **scenario**: a `ScenarioCfg` describing the simulation.
* **device**: execution device (e.g., CPU/GPU).

When defining a new task, inherit from `BaseTaskEnv` and implement methods like `_observation`, `_reward`, `_terminated`, `_time_out`, `_observation_space`, `_action_space`, and `_extra_spec`.

Tasks are managed by a **registry system**, where each task is bound to a unique string ID (e.g., `"example.my_task"`). This design provides:

* **One-click switching**: run a different task by simply changing a string in configs or CLI args.
* **Unified interface**: all tasks share the same API, regardless of simulator or logic.
---

## 2. Task Instantiation Workflow

Typical instantiation of a task for training looks like:

```python
task_cls = get_task_class(args.task)

# Get default scenario from task class and update with overrides
scenario = task_cls.scenario.update(
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=args.headless,
    cameras=[],
)

# Create task env via registry
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = task_cls(scenario=scenario, device=device)

```

**Key points:**

* `get_task_class(name)` fetches the task class by string identifier from registry.
* Each task class provides a **default scenario config** (`task_cls.scenario`) with standard robot, object, and asset definitions.
* Users can **update** this config (simulator choice, camera list, env count, etc.) via `scenario.update()`.
* The updated `ScenarioCfg` is then passed into the task class to instantiate a working environment.

This workflow ensures tasks are:

* **Customizable** (override any part of the scenario at runtime).
* **Consistent** (task class always defines a sane default).
* **Simulator‑agnostic** (only the Handler changes underneath).

---


## 3. Task Instantiation Workflow

### 3.1 Via Task Registry

```python
"""Train PPO for a reaching task using RLTaskEnv."""
from metasim.task.registry import get_task_class
import torch
task_cls = get_task_class(args.task)  # e.g., "example.my_task"

# Start from the class-provided default scenario and override as needed
scenario = task_cls.scenario.update(
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=args.headless,
    cameras=[],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = task_cls(scenario=scenario, device=device)
```

### 3.2 Via `make_vec`

`make_vec` provides a standardized helper that wraps task instantiation in a **Gym‑compatible API**. It is the recommended entry point for creating environments.

```python
import metasim # triggering the registration of tasks with Gymnasium
from gymnasium import make_vec

env = make_vec(
    env_id,                      # e.g., "example.my_task"
    num_envs=args.num_envs,
    robots=[args.robot],
    simulator=args.sim,
    headless=args.headless,
    cameras=[camera] if args.save_video else [],
    device=args.device,
)
```

**Key points:**

* Each task class provides a **default scenario** (`task_cls.scenario`) with standard robots/objects/assets.
* Use `scenario.update(...)` to override simulator, cameras, env count, etc.
* The final `ScenarioCfg` is passed into the task class or wrapped via `make_vec` for Gym API compatibility.

---

## 4. Task Registration & Auto‑Import

### 4.1 Auto‑import paths

Task modules under the following directories are **auto‑imported and registered** at runtime:

* `metasim/example/example_pack/tasks`
* `roboverse_pack/tasks`

> For new project tasks, place modules under **`roboverse_pack/tasks`**.

### 4.2 How to register a task

```python
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.scenario.scenario import ScenarioCfg

@register_task("example.my_task")
class MyExampleTask(BaseTaskEnv):
    scenario = ScenarioCfg(robots=["franka"], simulator="mujoco", cameras=[])
    def _observation(self, state): ...
    def _privileged_observation(self, state): ...
    def _reward(self, state, action, next_state=None): ...
    def _terminated(self, state): ...
    def _time_out(self, step_count): ...
    def _observation_space(self): ...
    def _action_space(self): ...
    def _extra_spec(self): ...
    def step(self,actions): ...
    def reset(self,states,env_ids): ...
```

### 4.3 Using the Task Template

For quickly creating a new task, we provide a **task template** at `roboverse_pack/tasks/task_template.py`.

The template includes:

* Complete task structure with all necessary methods
* Example implementations for `_observation`, `_reward`, `_terminated`
* Scenario configuration with common objects and robots
* Detailed comments explaining each component

**Usage:**

1. Copy `task_template.py` to your desired location under `roboverse_pack/tasks/`
2. Rename the file and class to match your task name
3. Update the `@register_task()` decorator with your task ID
4. Modify the scenario, reward, observation logic as needed
5. The task will be auto-imported and registered

---

## 5. Migration New Task

### 5.1 Direct Integration (Quick)

1. Copy external task code into `roboverse_learn/`.
2. Replace simulator‑specific APIs with `Handler` equivalents.
3. Convert observations to `TensorState` via `get_state()`.
4. Move sim details (assets, timestep, decimation) into `ScenarioCfg`.

### 5.2 Structured Wrapper Integration

1. Subclass `BaseTaskWrapper`.
2. Implement `_reward()`, `_observation()`, `_terminated()`.
3. Use hooks `pre_sim_step`, `post_sim_step`, `reset_callback`.
4. Reuse `Handler` + `ScenarioCfg` separation.

---
## 6.BaseTaskEnv & RLTaskEnv

### BaseTaskEnv (core behavior)

* **Default observation**: returns the simulator’s **TensorState** directly via `_observation(env_states)` (structured tensor, not flattened).
* **Initialization**: accepts a `ScenarioCfg` or a pre‑built `BaseSimHandler`. Internally resolves the handler and calls `launch()`.
* **Callbacks**: `pre_physics_step_callback`, `post_physics_step_callback`, `reset_callback`, `close_callback`.
* **Episode control**: per‑env step counter `self._episode_steps`; timeout handled by `_time_out` (default based on `max_episode_steps`).
* **Step flow**:

  1. `pre_physics_step_callback(actions)`
  2. `handler.set_dof_targets(actions)`
  3. `handler.simulate()`
  4. `env_states = handler.get_states()`
  5. `post_physics_step_callback(env_states)`
  6. Compute `reward`, `terminated`, `timeout` and return `(obs, reward, terminated, timeout, info)` with `privileged_observation`.
* **Reset flow**: can use external `states` or fall back to `_initial_states`. Calls `handler.set_states(...)`, fetches `env_states`, and resets episode counters.
* **Flexible override**: You can also override `step()` and `reset()` functions directly, bypassing the callback system entirely.

### RLTaskEnv (RL‑friendly extension)

* **Observation shape**: flattens `TensorState` into a 1D tensor and builds `observation_space = Box(num_obs,)`.
* **Action handling**: derives `action_space` from `robot.joint_limits` and `handler.get_joint_names(...)`. In `step()`, actions are clamped before being passed to `set_dof_targets`.
* **Auto device**: defaults to CUDA if available.
* **Auto reset on done**: after each step, envs flagged by `terminated | time_out` are reset in-place, and their observations refreshed.
* **Initial state acceleration**: uses `list_state_to_tensor(handler, _get_initial_states())` to convert list states to tensor states for faster resets.
* **Info payload**: includes `privileged_observation`, `episode_steps`, and cached raw observations `observations.raw.obs`.
* **Utilities**: `unnormalise_action(a)` maps actions from `[-1,1]` to joint physical ranges.

### Differences at a Glance

| Aspect               | BaseTaskEnv                        | RLTaskEnv                         |
| -------------------- | ---------------------------------- | --------------------------------- |
| Observation return   | TensorState (not flattened)        | Flattened tensor (1D)             |
| Auto reset           | No                                 | Yes (on done/timeout)             |
| Space construction   | Decided by subclass or upper layer | Auto‑derived obs/action spaces    |
| Action clamping      | Decided by subclass or upper layer | Built‑in clamping to joint limits |
| Initial state format | list or tensor                     | Auto conversion list → tensor     |
| Device selection     | Passed by user                     | Auto‑select CUDA/CPU              |

---
## 7. Summary

* Tasks = **glue layer** between `ScenarioCfg/Handler` (simulation) and learning algorithms
* Registry system (`get_task_class`) makes tasks discoverable by string names.
* Default `ScenarioCfg` in each task class ensures reproducibility and easy overrides.
* Two migration methods (Quick vs. Structured) cover integration.
