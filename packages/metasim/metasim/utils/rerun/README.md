# TaskRerunWrapper Usage Guide

TaskRerunWrapper is a wrapper that adds real-time Rerun visualization to RLTaskEnv.

## Main Features

- Automatic Rerun visualization setup
- Real-time robot and object state updates
- Only renders the first environment (to avoid multi-environment complexity)
- Transparent proxy for all environment attributes and methods
- Error handling: visualization failures don't affect training

## Usage

### 1. Basic Usage

```python
from metasim.task.registry import get_task_class
from metasim.utils.rerun.rerun_env_wrapper import TaskRerunWrapper

# Create environment
task_cls = get_task_class('reach_origin')
scenario = task_cls.scenario.update(
    robots=['franka'],
    simulator='mujoco',
    num_envs=1024,
    headless=True,
    cameras=[]
)

env = task_cls(scenario, device='cuda')

# Wrap environment to enable visualization
rerun_env = TaskRerunWrapper(env, app_name="RL Training", update_freq=10)

# Use normally
obs = rerun_env.reset()
for _ in range(100):
    actions = policy(obs)  # your policy
    obs, reward, terminated, timeout, info = rerun_env.step(actions)
    if terminated.any() or timeout.any():
        obs = rerun_env.reset()

rerun_env.close()
```

### 2. Integration with fast_td3

Enable visualization in fast_td3:

```python
CONFIG = {
    "sim": "mujoco",
    "robots": ["franka"],
    "task": "reach_origin",
    "headless": True,
    "use_rerun": True,  # Enable Rerun visualization
    # ... other config
}
```

## How It Works

1. **On Initialization**:
   - Creates RerunVisualizer instance
   - Downloads necessary URDF files
   - Visualizes all robots and objects in the scene
   - Opens the Rerun viewer

2. **During Runtime**:
   - Updates visualization after each `reset()` and `step()` (based on update_freq)
   - Extracts first environment's state
   - Updates all robot and object positions/orientations

3. **Error Handling**:
   - Visualization failures don't affect training
   - Errors are logged but don't raise exceptions

## Comparison with TaskViserWrapper

| Feature | TaskRerunWrapper | TaskViserWrapper |
|---------|-----------------|------------------|
| Viewer | Desktop app | Web browser (port 8080) |
| Timeline | Built-in scrubbing | No |
| Recording | Native .rrd files | No |
| Interactive Controls | No | Joint/IK sliders |

## API Reference

### TaskRerunWrapper

```python
TaskRerunWrapper(
    task_env,           # RLTaskEnv or similar
    app_name="RoboVerse Training",  # Rerun application name
    update_freq=10,     # Update visualization every N steps
    spawn=True,         # Auto-open Rerun viewer
)
```

### Methods

- `step(action)` - Step environment and update visualization
- `reset(**kwargs)` - Reset environment and update visualization
- `close()` - Close environment and visualizer
- All other methods/attributes proxied to wrapped environment

