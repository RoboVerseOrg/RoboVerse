# Adding New Robots

Define a new robot in `metasim/cfg/robots/{robot_name}_cfg.py`.

You can debug robot by applying random actions:

```bash
python metasim/scripts/random_action.py --sim=isaaclab --num_envs=4 --robot=franka
```

Other examples for unitree h1 robot:

```bash
python metasim/scripts/random_action_pure.py --sim=isaaclab --num_envs=1 --robot=h1 --task=humanoidbench:Walk
```

```bash
python metasim/scripts/random_action_pure.py --sim=isaacgym --num_envs=1 --robot=h1 --task=humanoidbench:Walk
```

```bash
python metasim/scripts/random_action_pure.py --sim=mujoco --num_envs=1 --robot=h1 --task=humanoidbench:Walk
```
