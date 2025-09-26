# Guide

This guide is designed for users who want to understand RoboVerse's core concepts and get started quickly. 

## Core Concepts Overview

RoboVerse is built around three key components that work together:

```
Handler → (Task, Optional) → Algorithm/Script
```

Let's understand each component:

---

## 1. Handler: The Physics Engine Interface

### What is a Handler?

A **Handler** is the core component that directly interfaces with physics engines (like MuJoCo, IsaacSim, PyBullet, etc.). Think of it as a translator that speaks the language of different simulators.

### Key Handler Methods

```python
# Get current simulation state, including RGB Data
state = handler.get_state()

# Set simulation state (for resets)
handler.set_state(new_state)

# Apply robot actions
handler.set_dof_targets(actions)

# Step physics simulation
handler.simulate()

# Get robot joint/body names
joint_names = handler.get_joint_names()
body_names = handler.get_body_names()
```

---

## 2. Scenario: The Configuration Blueprint

### What is a Scenario?

A **Scenario** (`ScenarioCfg`) is a configuration that tells the Handler what to simulate. It's like a blueprint that describes everything needed for a simulation.

### Creating a Scenario

Here's how to create a basic scenario:

```python
from metasim.scenario.scenario import ScenarioCfg

# Simple scenario with just a robot
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="mujoco",
    num_envs=1
)

```

### Key Scenario Fields Explained

Fields include: `robots` (robot configurations), `objects` (scene objects), `simulator` (physics engine), `num_envs` (number of parallel environments), `headless` (run without GUI), `cameras` (camera sensors), `lights` (lighting setup), and `sim_params` (physics parameters like timestep and gravity).
---

## 3. Task: The High-Level Interface

### What is a Task?

A **Task** is a wrapper built on top of a Handler that provides Gym-style APIs (`step`, `reset`, `observation_space`, `action_space`). It adds task-specific logic like rewards, termination conditions, and observations.

### Task vs Handler

| Aspect | Handler | Task |
|--------|---------|------|
| **Purpose** | Direct physics engine interface | High-level learning interface |
| **API** | Low-level (`get_state`, `simulate`...) | Gym-style (`step`, `reset`) |
| **Logic** | No task logic, Only simulator-related | Contains reward, termination, etc. |
| **Usage** | For custom control | For training, data collection, evaluation |


## Getting Started: Two User Paths

### Path A: Use RoboVerse as Unified Simulator

**Goal**: Integrate RoboVerse into your existing project

**Steps**:
1. **Choose a simulator** based on your needs
2. **Define your scenario** with robots, objects, cameras
3. **Integrate with your workflow** using Handler methods:

**Example**:
```python
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils.setup_util import get_handler

# Define your scenario
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="mujoco",
    num_envs=1
)

# Create handler directly - one step
handler = get_handler(scenario)

# Control loop
states = ...
handler.set_state(states)     # Set state (for resets)

actions = ...  # Your control actions
handler.set_dof_targets(actions)     # Apply actions to robot

handler.simulate()                  # Step physics
obs = handler.get_state(mode="tensor")  # Get updated state
```

### Path B: Use Pre-built Tasks

**Goal**: Run existing tasks and training scripts

**What Tasks Provide**:
- Pre-configured scenarios (robots, objects, cameras)
- Task logic (max episode length, initial states)
- Reward functions and termination conditions
- Ready-to-use training scripts

---
。
There are two ways to use pre-built tasks:

#### Method 1: Via Task Registry (Direct Task Class)

```python
from metasim.task.registry import get_task_class
import torch

# Get task class by name
task_cls = get_task_class(args.task)  # e.g., "example.reaching"

# Get default scenario and update with specific parameters
scenario = task_cls.scenario.update(
    robots=[args.robot], 
    simulator=args.sim, 
    num_envs=args.num_envs, 
    headless=args.headless, 
    cameras=[]
)

# Create task environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = task_cls(scenario=scenario, device=device)


```

```python
obs, info = env.reset()
action = ...
obs, reward, terminated, truncated, info = env.step(action)
```


---

## Next Steps

For more detailed information, you can explore the concept tutorials which cover advanced topics such as state system content and format, domain randomization techniques, handler implementation details, task system architecture, and configuration management.


---

