# 9. Config-based Task
In this tutorial, we will show you how to use MetaSim to config a task.

## Common Usage

```bash
python get_started/9_cfg_task.py --sim <simulator> --num_envs <num_envs> --task <task_name>
```

### Examples

#### IsaacSim
```bash
python get_started/9_cfg_task.py --sim isaacsim --num_envs 4 --task close_box
```

#### Isaac Gym
```bash
python get_started/9_cfg_task.py --sim isaacgym --num_envs 4 --task close_box
```

## Code Highlights

**Gym-style Task Creation**: Use `make_vec()` to create environments with Gym-style API:
```python
from metasim.task.gym_registration import make_vec

env = make_vec(
    env_id=f"RoboVerse/{task_name}",
    robots=[args.robot],
    simulator=args.sim,
    num_envs=args.num_envs,
    headless=args.headless,
)

# Gym-style API
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(actions)
```

**Key Differences from Direct Task Creation**:
- **Normal task**: Access handler via `env.handler`
- **Gym-style task**: Access handler via `env.task_env.handler` (one extra layer)
- **Same functionality**: Both provide identical Gym-style APIs and task logic
- **Wrapper layer**: Gym-style adds a compatibility wrapper for standard RL libraries
