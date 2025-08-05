# 8. Replay Trajectories

There are two control modes for replay. The `physics` mode replays the physics actions and the `states` mode replays the states, so the `physics` mode has greater possibilities to fail across different simulators, but the `states` mode is bound to succeed across different simulators.

## Physics replay

```bash
python metasim/scripts/replay_demo.py --sim=isaaclab --task=CloseBox --num_envs 4
```

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
- `BasketballInHoop`

## States replay

```bash
python metasim/scripts/replay_demo.py --sim=isaaclab --task=CloseBox --num_envs 4 --object-states
```
task could also be:
- `CloseBox`
- `BasketballInHoop`

## Varifies commands

### Libero

e.g.

```bash
python metasim/scripts/replay_demo.py --sim=isaaclab --task=LiberoPickButter
```

Simulator:
- `isaaclab`
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
python metasim/scripts/replay_demo.py --sim=isaaclab --num_envs=1 --robot=h1 --task=Stand --object-states
```

```bash
python metasim/scripts/replay_demo.py --sim=mujoco --num_envs=1 --robot=h1 --task=Stand --object-states
```

Simulator:
- `isaaclab`
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
python metasim/scripts/replay_demo.py --sim=isaaclab --task=CloseBox --num_envs 1 --scene=tapwater_scene_131
```
