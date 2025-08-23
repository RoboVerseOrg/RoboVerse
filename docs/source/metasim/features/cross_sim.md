# Cross Simulator
## Basic usage
By default, the simulator is set to `isaaclab`. You can change it to other simulators by setting the `sim` argument. Currently, we support:
- `isaaclab`
- `isaacgym`
- `pyrep`

## IsaacGym example
For example, to replay the demo in IsaacGym, you can run:
```bash
python metasim/scripts/replay_demo.py --sim=isaacgym --task=StackCube
```

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
