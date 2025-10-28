# 8. Replay Trajectories

There are two control modes for replay. The `physics` mode replays the physics actions and the `states` mode replays the states, so the `physics` mode has greater possibilities to fail across different simulators, but the `states` mode is bound to succeed across different simulators.

## Physics replay

```bash
python scripts/advanced/replay_demo.py --sim=isaacsim --task=close_box --num_envs 4
```

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
- `BasketballInHoop`

## States replay

```bash
python scripts/advanced/replay_demo.py --sim=isaacsim --task=close_box --num_envs 4 --object-states
```
task could also be:
- `CloseBox`
- `BasketballInHoop`

## Varifies commands

### Libero

e.g.

```bash
python scripts/advanced/replay_demo.py --sim=isaacsim --task=LiberoPickButter
```

Simulator:
- `isaacsim`
- `mujoco`

Task:
- `LiberoPickAlphabetSoup`
- `LiberoPickBbqSauce`
- `LiberoPickChocolatePudding`
- `LiberoPickCreamCheese`
- `LiberoPickMilk`
- `LiberoPickOrangeJuice`
- `LiberoPickSaladDressing`
- `LiberoPickTomatoSauce`

### Humanoid

e.g.

```bash
python scripts/advanced/replay_demo.py --sim=isaacsim --num_envs=1 --robot=h1 --task=Stand --object-states
```

```bash
python scripts/advanced/replay_demo.py --sim=mujoco --num_envs=1 --robot=h1 --task=Stand --object-states
```

Simulator:
- `isaacsim`
- `mujoco`

Task:
- `Stand`
- `Walk`
- `Run`

Note:
- `MuJoCo` replay supports only one environment at a time, aka `num_envs` should be 1 (but training supports multiple environments).

### Add scene:
Note: only single environment is supported for adding scene.
```bash
python scripts/advanced/replay_demo.py --sim=isaacsim --task=CloseBox --num_envs 1 --scene=tapwater_scene_131
```

## Code Highlights

**Trajectory Loading**: Use `get_traj()` to automatically download and load trajectories from roboverse_data:
```python
from metasim.utils.demo_util import get_traj

# Load trajectory data (auto-downloads from HuggingFace if needed)
init_states, all_actions, _ = get_traj(traj_filepath, scenario.robots[0], env.handler)

# Action replay: Execute actions step by step
for step in range(len(all_actions)):
    actions = get_actions(all_actions, step, num_envs, robot)
    obs, reward, success, time_out, extras = env.step(actions)

# State replay: Directly set states (more reliable)
for i, state in enumerate(captured_states):
    env.handler.set_states(state)
    env.handler.refresh_render()
    obs = env.handler.get_states()
```

**Two Replay Modes**:
- **Action replay**: Executes recorded actions through physics simulation (may fail across simulators)
- **State replay**: Directly sets system states (guaranteed to succeed across simulators)
- **Automatic download**: Trajectories are automatically downloaded from HuggingFace when needed
