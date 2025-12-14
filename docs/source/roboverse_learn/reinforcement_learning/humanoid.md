# Humanoid Locomotion

Train and deploy locomotion policies for humanoid robots across three stages:
- Training in IsaacGym, IsaacSim, MuJoCo
- Sim2Sim evaluation across multiple simulators
- Real-world deployment (networked controller)

Supported robots: `g1_dof29` (full-body without hands) and `g1_dof12` (lower-body).


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

### RSL-RL PPO Training

General form:
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task <your_task> \
  --sim <simulator> \
  --num-envs <num_envs> \
  --robot <your_robot>
```

Examples:
- G1 humanoid walking (IsaacGym):
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task walk_g1_dof29 \
  --sim isaacgym \
  --num-envs 8192 \
  --robot g1_dof29
```

- G1 DOF12 walking (IsaacGym):
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task walk_g1_dof12 \
  --sim isaacgym \
  --num-envs 8192 \
  --robot g1_dof12
```

- G1 humanoid walking (IsaacSim):
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task walk_g1_dof29 \
  --sim isaacsim \
  --num-envs 4096 \
  --robot g1_dof29
```

### Resuming Training

To resume training from a checkpoint:
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task walk_g1_dof29 \
  --sim isaacgym \
  --num-envs 8192 \
  --robot g1_dof29 \
  --resume <timestamp_directory> \
  --checkpoint -1
```
The `--checkpoint -1` flag loads the latest checkpoint. You can specify a specific iteration number instead.

### Other RL Algorithms

The framework supports other RL algorithms in `roboverse_learn/rl/`:
- **CleanRL implementations**: PPO, SAC, TD3
  - Training: `python roboverse_learn/rl/clean_rl/{ppo,sac,td3}.py`
- **Stable-Baselines3**: Available in `roboverse_learn/rl/sb3/`

Refer to the respective training scripts for algorithm-specific parameters.

### Output Directory Structure

Outputs and checkpoints are saved to:
```
outputs/<robot>/<task>/<timestamp>/
```
For example:
```
outputs/g1_dof29/walk_g1_dof29/2025_1203_150000/
```

Each training run contains:
- `model_<iter>.pt` - Checkpoint files saved at specified intervals
- `policy.pt` - Final exported JIT policy (for deployment)
- Training logs (TensorBoard/WandB depending on configuration)


## Evaluation

Evaluate trained policies across different simulators using the evaluation script.

General form:
```bash
python roboverse_learn/rl/rsl_rl/eval.py \
  --task <your_task> \
  --sim <simulator> \
  --num-envs <num_envs> \
  --robot <your_robot> \
  --resume <timestamp_directory> \
  --checkpoint <iter>
```

Examples:

IsaacGym evaluation:
```bash
python roboverse_learn/rl/rsl_rl/eval.py \
  --task walk_g1_dof29 \
  --sim isaacgym \
  --num-envs 1 \
  --robot g1_dof29 \
  --resume 2025_1203_150000 \
  --checkpoint -1
```

MuJoCo evaluation:
```bash
python roboverse_learn/rl/rsl_rl/eval.py \
  --task walk_g1_dof12 \
  --sim mujoco \
  --num-envs 1 \
  --robot g1_dof12 \
  --resume 2025_1203_150000 \
  --checkpoint -1
```

IsaacSim evaluation:
```bash
python roboverse_learn/rl/rsl_rl/eval.py \
  --task walk_g1_dof29 \
  --sim isaacsim \
  --num-envs 1 \
  --robot g1_dof29 \
  --resume 2025_1203_150000 \
  --checkpoint -1
```

The evaluation script runs for 1,000,000 steps with fixed velocity commands for thorough policy assessment.


## Real-World Deployment

Real-world deployment entry point:
```bash
python roboverse_pack/tasks/humanoid/unitree_deploy/deploy_real.py <network_interface> <robot_yaml>
```

Example:
```bash
python roboverse_pack/tasks/humanoid/unitree_deploy/deploy_real.py eno1 g1_dof29_dex3.yaml
```

Configuration files are located in `roboverse_pack/tasks/humanoid/unitree_deploy/configs/`:
- `g1_dof12.yaml` - 12 DOF lower-body control
- `g1_dof29.yaml` - 29 DOF full-body control
- `g1_dof29_dex3.yaml` - 43 DOF with DeX3 hands

In the YAML file, set the `policy_path` to your exported JIT policy (the `policy.pt` file from training output).

This will initialize the real controller and stream commands to the robot. Ensure your networking and safety interlocks are correctly configured.


## Advanced Features

### Terrain Configuration

The framework supports customizable terrain generation for training locomotion policies on varied ground conditions. Predefined terrain configurations are available in `roboverse_pack/grounds/`.

#### Supported Terrain Types

- **Slope**: Planar inclined surfaces (`slope_cfg.py`)
- **Stair**: Staircase features (`stair_cfg.py`)
- **Obstacle**: Random rectangular obstacle fields (`obstacle_cfg.py`)
- **Stone**: Stone-like protrusions (`stone_cfg.py`)
- **Gap**: Gaps that robots must traverse (`gap_cfg.py`)
- **Pit**: Rectangular pits (`pit_cfg.py`)

#### Example Usage

Terrain configurations are imported and used in task definitions. See task files in `roboverse_pack/tasks/humanoid/locomotion/` for examples.

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

The framework includes LiDAR point cloud sensing capabilities for enhanced perception-based locomotion policies.

#### Overview

The `LidarPointCloud` query provides 3D point cloud data from a simulated LiDAR sensor:
- Supports IsaacGym, IsaacSim, and MuJoCo simulators
- Returns point clouds in both local (sensor frame) and world frames
- Configurable sensor mounting location and type
- Raycasts against terrain, ground, and scenario objects

#### Configuration

LiDAR configuration is defined in task files. See `roboverse_pack/tasks/humanoid/locomotion/walk_g1_dof29.py` for examples.

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


## Command-line Arguments

### Common Training/Evaluation Arguments

- `--task` (str): Task name. Examples: `walk_g1_dof29`, `walk_g1_dof12`
- `--robot` (str): Robot identifier. Examples: `g1_dof29`, `g1_dof12`
- `--num-envs` (int): Number of parallel environments
- `--sim` (str): Simulator. Supported: `isaacgym`, `isaacsim`, `mujoco`
- `--headless` (bool): Headless rendering (default: False)
- `--device` (str): Device for training (default: "cuda:0")
- `--seed` (int): Random seed (default: 1)

### Training-Specific Arguments (RSL-RL PPO)

- `--max-iterations` (int): Number of training iterations (default: 50000)
- `--save-interval` (int): Checkpoint save interval (default: 100)
- `--logger` (str): Logger type. Options: `tensorboard`, `wandb`, `neptune` (default: "tensorboard")
- `--wandb-project` (str): WandB project name (default: "rsl_rl_ppo")
- `--use-wandb` (bool): Enable WandB logging (default: False)

### Evaluation/Checkpoint Arguments

- `--resume` (str): Timestamp directory containing checkpoints
- `--checkpoint` (int): Checkpoint iteration to load. `-1` loads the latest (default: -1)

### Examples

Training with custom parameters:
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task walk_g1_dof29 \
  --robot g1_dof29 \
  --sim isaacgym \
  --num-envs 8192 \
  --max-iterations 30000 \
  --save-interval 200 \
  --logger wandb \
  --use-wandb
```

Resume from checkpoint:
```bash
python roboverse_learn/rl/rsl_rl/ppo.py \
  --task walk_g1_dof29 \
  --robot g1_dof29 \
  --sim isaacgym \
  --num-envs 8192 \
  --resume 2025_1203_150000 \
  --checkpoint 5000
```

Evaluation with specific checkpoint:
```bash
python roboverse_learn/rl/rsl_rl/eval.py \
  --task walk_g1_dof29 \
  --robot g1_dof29 \
  --sim mujoco \
  --num-envs 1 \
  --resume 2025_1203_150000 \
  --checkpoint 10000
```