# Unitree RL

Train and deploy locomotion policies for Unitree robots across three stages:
- Training in IsaacGym, IsaacSim
- Sim2Sim evaluation in IsaacGym, IsaacSim, MuJoCo
- Real-world deployment (networked controller)

Well Supported robots: `g1_dof29` (full-body with without hands) and `g1_dof12` (lower-body).


## Environment Setup

### Core Dependencies

#### For Python > 3.8 (IsaacSim/Mujoco)

You can install rsl-rl-lib directly via pip:
```bash
pip install rsl-rl-lib
```

#### For Python 3.8 (IsaacGym)
Due to compatibility requirements with IsaacGym's Python 3.8 environment, you'll need to install from source with modified dependencies:
```bash
# Clone the repository and checkout v3.1.0
git clone https://github.com/leggedrobotics/rsl_rl && \
cd rsl_rl && \
git checkout v3.1.0

# Apply compatibility patches
sed -i 's/"torch>=2\.6\.0"/"torch>=2.4.1"/' pyproject.toml && \
sed -i 's/"torchvision>=0\.5\.0"/"torchvision>=0.19.1"/' pyproject.toml && \
sed -i 's/"tensordict>=0\.7\.0"/"tensordict>=0.5.0"/' pyproject.toml && \
sed -i '/^# SPDX-License-Identifier: BSD-3-Clause$/a from __future__ import annotations' rsl_rl/algorithms/distillation.py

# Install in editable mode
pip install -e .
```

### Optional Dependencies

#### LiDAR Sensor Support (OmniPerception)

The LiDAR implementation uses the [OmniPerception](https://github.com/aCodeDog/OmniPerception) package for GPU-accelerated ray tracing.

**For IsaacGym and MuJoCo:**

Install the LidarSensor package:
```bash
cd /path/to/OmniPerception/LidarSensor
pip install -e .
```

For complete IsaacGym/MuJoCo integration details, see the [OmniPerception IsaacGym example](https://github.com/aCodeDog/OmniPerception/tree/main/LidarSensor/LidarSensor/example/isaacgym).

**For IsaacSim (IsaacLab):**

First you need to install IsaacLab from source. Follow the official [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Then install the LiDAR sensor extension:

```bash
cd /path/to/OmniPerception/LidarSensor/LidarSensor/example/isaaclab
./install_lidar_sensor.sh /path/to/your/IsaacLab
```

For complete IsaacSim integration details, see the [OmniPerception IsaacLab example](https://github.com/aCodeDog/OmniPerception/tree/main/LidarSensor/LidarSensor/example/isaaclab/isaaclab).

#### Real-World Deployment (unitree_sdk2_python)

For real-world deployment, install the `unitree_sdk2_python` package:
```bash
cd third_party
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```


## Training

General form:
```
python roboverse_learn/rl/unitree_rl/main.py \
  --task <your_task> \
  --sim isaacgym \
  --num_envs 8192 \
  --robot <your_robot>
```

Examples:
- G1 humanoid walking (IsaacSim):
```
python roboverse_learn/rl/unitree_rl/main.py --task walk_g1_dof29 --sim isaacsim --num_envs 8192 --robot g1_dof29
```
- G1Dof12 walking (IsaacGym):
```
python roboverse_learn/rl/unitree_rl/main.py --task walk_g1_dof12 --sim isaacgym --num_envs 8192 --robot g1_dof12
```
- G1 humanoid walking with terrain (slope):
```
python roboverse_learn/rl/unitree_rl/main.py --task walk_g1_dof29 --sim isaacgym --num_envs 8192 --robot g1_dof29 --ground slope_cfg
```

Outputs and checkpoints are saved to:
```
outputs/unitree_rl/<robot>_<task>/<datetime>/
```
Each checkpoint is named `model_<iter>.pt`.

## Evaluation / Play

You can evaluate trained policies in both MuJoCo, Isaacsim and IsaacGym. In evaluation, `main.py` also exports the jit version policy to the directory `outputs/unitree_rl/<robot>_<task>/<datetime>/exported/model_exported_jit.pt`, which can be further used for real-world deployment.

IsaacGym evaluation:
```
python roboverse_learn/rl/unitree_rl/main.py \
  --task walk_g1_dof29 \
  --sim isaacgym \
  --num_envs 1 \
  --robot g1_dof29 \
  --resume <datetime_from_outputs> \
  --checkpoint <iter> \
  --eval
```

MuJoCo evaluation (e.g., DOF12 with public policy):
```
python roboverse_learn/rl/unitree_rl/main.py \
  --checkpoint <iter> \
  --task walk_g1_dof12 \
  --sim mujoco \
  --robot g1_dof12 \
  --resume <datetime_from_outputs> \
  --eval
```
the `--resume` and `--checkpoint` option can also be used during training for checkpoint resume.

## Real-World Deployment

Real-world deployment entry point:
```
python roboverse_learn/rl/unitree_rl/deploy/deploy_real.py <network_interface> <robot_yaml>
```
Example:
```
python roboverse_learn/rl/unitree_rl/deploy/deploy_real.py eno1 g1_dof29_dex3.yaml
```
where you should modify the corresponding `yaml` file in `roboverse_learn/rl/unitree_rl/deploy/configs`, setting the `policy_path` to the exported jit policy.
This will initialize the real controller and stream commands to the robot. Ensure your networking and safety interlocks are correctly configured.

## Advanced Features

### Terrain Configuration

The framework now supports customizable terrain generation for training locomotion policies on varied ground conditions. Terrain can be configured via the `ground` parameter in your scenario configuration.

#### Supported Terrain Types

The `GroundCfg` class supports multiple terrain primitives:
- **Slope**: Planar inclined surfaces
- **Stair**: Staircase features
- **Obstacle**: Random rectangular obstacle fields
- **Stone**: Stone-like protrusions
- **Gap**: Gaps that robots must traverse
- **Pit**: Rectangular pits

#### Terrain Parameters

Key configuration parameters:
- `width`, `length`: Terrain dimensions in meters
- `horizontal_scale`, `vertical_scale`: Resolution and height scaling
- `margin`: Border margin around terrain
- `static_friction`, `dynamic_friction`, `restitution`: Physics material properties
- `elements`: Dictionary of terrain primitives to include
- `difficulty`: Difficulty progression settings

#### Example Usage

```python
from metasim.scenario.grounds import GroundCfg, SlopeCfg, StairCfg

ground_cfg = GroundCfg(
    width=20.0,
    length=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    static_friction=1.0,
    dynamic_friction=1.0,
    elements={
        "slope": [SlopeCfg(origin=[0, 0], size=[2.0, 2.0], slope=0.3)],
        "stair": [StairCfg(origin=[5, 0], size=[2.0, 2.0], step_height=0.1)]
    }
)
```

Terrain configuration is supported across all simulators (IsaacGym, IsaacSim, MuJoCo).

### LiDAR Sensor Support

The framework now includes LiDAR point cloud sensing capabilities for enhanced perception-based locomotion policies.

#### Overview

The `LidarPointCloud` query provides 3D point cloud data from a simulated LiDAR sensor:
- Supports IsaacGym, IsaacSim, and MuJoCo simulators
- Returns point clouds in both local (sensor frame) and world frames
- Configurable sensor mounting location and type
- Raycasts against terrain, ground, and scenario objects

#### Configuration

Add LiDAR to your environment via the `callbacks_query` parameter:

```python
from roboverse_learn.rl.unitree_rl.configs.cfg_queries import LidarPointCloud

callbacks_query = {
    "lidar_point_cloud": LidarPointCloud(
        link_name="mid360_link",           # Robot link to attach sensor
        sensor_type="mid360",               # Sensor type (e.g., mid360, mid70)
        apply_optical_center_offset=True,
        optical_center_offset_z=0.03503,
        enabled=True
    )
}
```

#### Sensor Parameters

- `link_name` (str): Name of the robot link where LiDAR is mounted (default: "mid360_link")
- `sensor_type` (str): Type of LiDAR sensor pattern (default: "mid360")
- `apply_optical_center_offset` (bool): Apply optical center offset correction
- `optical_center_offset_z` (float): Z-axis offset for optical center
- `enabled` (bool): Enable/disable LiDAR query

#### Output Format

The query returns a dictionary with:
- `points_local`: Point cloud in sensor frame (E, N, 3)
- `points_world`: Point cloud in world frame (E, N, 3)
- `dist`: Distance measurements (when available)
- `link`: Name of the link the sensor is attached to

where E is the number of environments and N is the number of points per scan.

#### Example Task Configuration

See [walk_g1_dof29.py](roboverse_learn/rl/unitree_rl/configs/locomotion/walk_g1_dof29.py) for a complete example:

```python
callbacks_query = {
    "contact_forces": ContactForces(history_length=3),
    "lidar_point_cloud": LidarPointCloud(enabled=True)
}
```

## Command-line Arguments

The most relevant flags (see `helper/utils.py`):
- `--task` (str): Task name. CamelCase or snake_case accepted. Examples: `walk_g1_dof29`, `walk_g1_dof12`.
- `--robot` (str): Robot identifier. Common: `g1_dof29`, `g1_dof12`.
- `--num_envs` (int): Number of parallel environments.
- `--sim` (str): Simulator. Supported: `isaacgym` (training), `mujoco` (evaluation).
- `--ground` (str): Ground/terrain configuration to load. References predefined configurations in `roboverse_pack/grounds/`. Examples: `slope_cfg`, `stair_cfg`, `obstacle_cfg`, `stone_cfg`, `gap_cfg`, `pit_cfg`. If not specified, uses default flat ground.
- `--run_name` (str): Required run tag for training logs/checkpoints.
- `--learning_iterations` (int): Number of learning iterations (default 15000).
- `--resume` (flag): Resume training from a checkpoint dir (datetime) in the specified run.
- `--checkpoint` (int): Which checkpoint to load. `-1` loads the latest.
- `--headless` (flag): Headless rendering (IsaacGym).
- `--jit_load` (flag): Load the jit policy.

Notes:
- Checkpoints: `outputs/unitree_rl/<task>/<run_name or datetime>/model_<iter>.pt`
- Exported JIT model (when used): `outputs/unitree_rl/<task>/<run_name or datetime>/exported/model_exported_jit.pt`