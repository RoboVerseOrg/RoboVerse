#  0. Static Scene
In this tutorial, we will show you how to use MetaSim to simulate a static scene.

## Common Usage

```bash
python get_started/0_static_scene.py  --sim <simulator>
```
you can also render in the headless mode by adding `--headless` flag. By using this, there will be no window popping up and the rendering will also be faster.

### Examples

#### Isaac Lab
```bash
python get_started/0_static_scene.py  --sim isaacsim
```
/isaacsim
#### Isaac Gym
```bash
python get_started/0_static_scene.py  --sim isaacgym
```

#### Mujoco
```bash
# For mac users, replace python with mjpython.
python get_started/0_static_scene.py  --sim mujoco --headless
```
Note that we find the `non-headless` mode of Mujoco is not stable. So we recommend using the `headless` mode.

#### Genesis
```bash
python get_started/0_static_scene.py  --sim genesis
```
Note that we find the `headless` mode of Genesis is not stable. So we recommend using the `non-headless` mode.

#### Sapien
```bash
python get_started/0_static_scene.py  --sim sapien3
```

#### Pybullet
```bash
python get_started/0_static_scene.py  --sim pybullet
```



You will get the following image:
---
| Isaac Lab | Isaac Gym | Mujoco |
|:---:|:---:|:---:|
| ![Isaac Lab](../../../_static/standard_output/0_static_scene_isaaclab.png) | ![Isaac Gym](../../../_static/standard_output/0_static_scene_isaacgym.png) | ![Mujoco](../../../_static/standard_output/0_static_scene_mujoco.png) |

| Genesis | Sapien | PyBullet |
|:---:|:---:|:---:|
| ![Genesis](../../../_static/standard_output/0_static_scene_genesis.png) | ![Sapien](../../../_static/standard_output/0_static_scene_sapien3.png) | ![Pybullet](../../../_static/standard_output/0_static_scene_pybullet.png) |

## Code Highlights

**Object Configuration**: Objects are added to `scenario.objects` with different types:
- `PrimitiveCubeCfg` / `PrimitiveSphereCfg`: Simple geometric objects
- `RigidObjCfg`: Static objects with physics properties  
- `ArticulationObjCfg`: Objects with joints (like the box_base)

**Initial State Setup**: Use `handler.set_states()` to position objects:
```python
init_states = [{
    "objects": {
        "cube": {"pos": torch.tensor([0.3, -0.2, 0.05]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        "sphere": {"pos": torch.tensor([0.4, -0.6, 0.05]), "rot": torch.tensor([1.0, 0.0, 0.0, 0.0])},
    }
}]
```