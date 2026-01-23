# Cross Simulator
## Basic usage
By default, the simulator is set to `isaacsim`. You can change it to other simulators by setting the `sim` argument. Currently, we support:
- `isaacsim`
- `isaacgym`
- `pyrep`

## IsaacGym example
For example, to replay the demo in IsaacGym, you can run:
```bash
python scripts/advanced/replay_demo.py --sim=isaacgym --task=StackCube
```

task could also be:
- `PickCube`
- `StackCube`
- `CloseBox`
