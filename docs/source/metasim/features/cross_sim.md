# Cross-Simulator Support

MetaSim enables you to write code once and run it across multiple physics simulators without modifications. This is one of the core features that makes RoboVerse a unified platform for robot learning.

---

## Supported Simulators

| Simulator | Backend | GPU Parallel | Best Use Case |
|-----------|---------|--------------|---------------|
| `mujoco` | MuJoCo | Via MJX | Fast prototyping, CPU training |
| `mjx` | MuJoCo MJX | Native | Massively parallel RL |
| `isaacsim` | Isaac Sim | Native | High-fidelity sim2real |
| `isaacgym` | Isaac Gym | Native | GPU-accelerated RL |
| `sapien3` | SAPIEN 3 | Native | Manipulation research |
| `sapien2` | SAPIEN 2 | - | Legacy support |
| `genesis` | Genesis | Native | Large-scale training |
| `pybullet` | PyBullet | - | Debugging, education |

---

## Basic Usage

### Switching Simulators

Change the simulator by setting the `simulator` parameter in your scenario:

```python
from metasim.scenario.scenario import ScenarioCfg

# Run with MuJoCo
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="mujoco",  # Change this to switch simulators
    num_envs=1,
)

# Or switch to Isaac Sim
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="isaacsim",
    num_envs=1,
)
```

### Command Line Examples

Most scripts support the `--sim` argument to select the simulator:

```bash
# Run with MuJoCo
python scripts/advanced/replay_demo.py --sim=mujoco --task=StackCube

# Run with Isaac Gym
python scripts/advanced/replay_demo.py --sim=isaacgym --task=StackCube

# Run with SAPIEN 3
python scripts/advanced/replay_demo.py --sim=sapien3 --task=PickCube
```

---

## Considerations

### Feature Parity

Not all features are available across all simulators. Check the [Support Matrix](support_matrix.rst) for detailed compatibility information.

### Performance Differences

| Simulator | Single Env Speed | Parallel Scaling | Rendering Quality |
|-----------|------------------|------------------|-------------------|
| MuJoCo | Fast | Good (via MJX) | Medium |
| Isaac Sim | Medium | Excellent | Excellent (RTX) |
| SAPIEN 3 | Fast | Good | Good |
| Genesis | Fast | Excellent | Good |

### Asset Compatibility

Different simulators may require different asset formats:

| Simulator | Preferred Format |
|-----------|------------------|
| MuJoCo | MJCF (.xml) |
| Isaac Sim | USD (.usd, .usda) |
| SAPIEN | URDF (.urdf) |
| PyBullet | URDF (.urdf) |

MetaSim handles format conversion automatically when possible.

---

## Example: Multi-Simulator Evaluation

Evaluate the same policy across different simulators to test sim2sim transfer:

```python
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_class

task_cls = get_task_class("StackCube")

simulators = ["mujoco", "sapien3", "isaacsim"]
results = {}

for sim in simulators:
    scenario = task_cls.scenario.update(
        simulator=sim,
        num_envs=10,
        headless=True,
    )
    env = task_cls(scenario=scenario)
    
    # Run evaluation
    success_rate = evaluate_policy(env, policy)
    results[sim] = success_rate
    
    env.close()

print("Cross-simulator results:", results)
```
