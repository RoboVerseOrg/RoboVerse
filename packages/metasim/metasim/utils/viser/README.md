# TaskViserWrapper

`TaskViserWrapper` wraps an `RLTaskEnv` and streams the first environment to a live Viser viewer.

## What it does

- sets up the Viser visualizer for the scenario
- updates robot and object poses after every `reset()` and `step()`
- renders the first environment only (keeps the viewer independent of `num_envs`)
- forwards every other attribute and method to the wrapped environment
- never interrupts training: a visualization failure is logged and skipped

## Usage

### 1. Wrap an environment

```python
from metasim.task.registry import get_task_class
from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

task_cls = get_task_class("reach_origin")
scenario = task_cls.scenario.update(
    robots=["franka"],
    simulator="mujoco",
    num_envs=1024,
    headless=False,  # rendering must be on for Viser
    cameras=[],
)

env = task_cls(scenario, device="cuda")

# wrap the environment to enable the viewer
viser_env = TaskViserWrapper(env, port=8080)

# use it like the plain environment
obs = viser_env.reset()
for _ in range(100):
    actions = policy(obs)  # your policy
    obs, reward, terminated, timeout, info = viser_env.step(actions)
    if terminated.any() or timeout.any():
        obs = viser_env.reset()

viser_env.close()
```

### 2. With fast_td3

Enable the viewer from the fast_td3 example:

```bash
# with the Viser viewer
python examples/rl/fast_td3/1_fttd3.py --viser-port 8080

# without it (default)
python examples/rl/fast_td3/1_fttd3.py
```

Configuration example:

```python
CONFIG = {
    "sim": "mujoco",
    "robots": ["franka"],
    "task": "reach_origin",
    "headless": True,
    "viser_port": 8080,  # > 0 enables the viewer
    # ... other settings
}
```

## How it works

1. **At construction**:
   - creates a `ViserVisualizer`
   - downloads the URDF files it needs
   - adds every robot and object of the scene to the viewer
   - sets up the camera controls

2. **At run time**:
   - refreshes the viewer after each `reset()` and `step()`
   - extracts the state of the first environment
   - updates the position and orientation of every robot and object

3. **On errors**:
   - a visualization failure does not stop training
   - import errors are handled quietly

## Attribute forwarding

`TaskViserWrapper` forwards every environment attribute:

```python
wrapper = TaskViserWrapper(env)
print(wrapper.num_envs)      # forwarded to env.num_envs
print(wrapper.num_actions)   # forwarded to env.num_actions
print(wrapper.num_obs)       # forwarded to env.num_obs
print(wrapper.action_space)  # forwarded to env.action_space
```

## Details

- uses the same state extraction as `viser_demo.py`
- works with every simulator backend (MuJoCo, Isaac Sim, ...)
- renders the first environment only, for performance
- indexes multi-environment tensors itself
