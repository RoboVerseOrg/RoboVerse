# Parallel Simulation

By default, the simulator only has one environment. You can create multiple environments by setting the `num_envs` argument.

For example, to create 4 environments in IsaacLab, you can run:
```bash
python metasim/scripts/replay_demo.py --task=StackCube --num_envs=4
```

Currently, we support the following simulators to use multiple environments:
- IsaacLab
- IsaacGym
- SAPIEN (GPU-based parallel under development, currently supported with multi-processing)
- Genesis
- PyBullet (supported by multi-processing)

We won't support other simulators to use multiple environments due to the limitation of the simulators:
- CoppeliaSim/PyRep
- MuJoCo (until MuJoCo 3 release)
