# RoboVerse Reinforcement Learning

This directory contains the reinforcement learning infrastructure for RoboVerse, supporting multiple state-of-the-art RL algorithms across various robotic tasks and benchmarks.

## Supported Algorithms

- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **Dreamer** (Model-based RL with world models)

## Supported Tasks

### IsaacGymEnvs Tasks

#### AllegroHand - Dexterous Manipulation
The AllegroHand task involves dexterous object manipulation with a 4-finger robotic hand. The task requires rotating objects to match target orientations.

**Supported Objects:**
- **Block**: Cube object for basic manipulation
- **Pen**: Cylindrical object requiring precision grasping (configuration available via `object_type="pen"`)
- **Egg**: Ellipsoid object requiring careful handling (configuration available via `object_type="egg"`)

**Key Features:**
- 16 DOF control (4 joints per finger)
- Object reorientation to match goal poses
- Dense rewards based on position/rotation errors
- Support for different observation types (full, full_no_vel, full_state)

#### Ant - Quadruped Locomotion
The Ant environment features a 4-legged robot learning to walk and navigate.

**Key Features:**
- 8 DOF control (2 joints per leg)
- Forward locomotion with stability
- Energy-efficient movement rewards
- Configurable action scaling and torque limits

#### Anymal - Advanced Quadruped Locomotion
The Anymal tasks feature a realistic quadruped robot based on ANYbotics' ANYmal robot.

**Variants:**
- **Anymal**: Basic locomotion on flat terrain
- **AnymalTerrain**: Locomotion on rough/uneven terrain with curriculum learning

**Key Features:**
- 12 DOF control (3 joints per leg)
- Command-following (velocity tracking)
- Terrain adaptation capabilities
- Base orientation and contact force rewards
- Configurable terrain difficulty levels

### DMControl Tasks
#### Locomotion Tasks
- **Acrobot**: Swingup control of underactuated double pendulum
- **Cartpole**: Balance, balance_sparse, swingup, swingup_sparse variants
- **Cheetah**: High-speed running
- **Hopper**: Single-leg hopping and standing
- **Humanoid**: Bipedal walking
- **Pendulum**: Classic swingup control
- **Walker**: Bipedal walking, running, and standing

#### Manipulation Tasks
- **Cup**: Ball-in-cup catching task
- **Finger**: Object spinning and turning (easy/hard variants)
- **Reacher**: 2-link arm reaching (easy/hard variants)

## Installation

Before running RL training, ensure you have the proper environment set up:

```bash
# Activate the Isaac Gym environment
conda activate isaacgym

# Set the library path (required for every new terminal)
export LD_LIBRARY_PATH=/home/handsomeyoungman/anaconda3/envs/isaacgym/lib
```

## Training Commands

### Basic Training

Train with a specific configuration:
```bash
python roboverse_learn/rl/train_rl.py train=<TaskAlgorithm>
```

### IsaacGymEnvs Tasks

#### AllegroHand Object Manipulation
```bash
# Basic block manipulation
python roboverse_learn/rl/train_rl.py train=AllegroHandPPO
python roboverse_learn/rl/train_rl.py train=AllegroHandTD3

# Different object types (override object_type parameter)
python roboverse_learn/rl/train_rl.py train=AllegroHandPPO task.object_type="pen"
python roboverse_learn/rl/train_rl.py train=AllegroHandPPO task.object_type="egg"

# Custom observation types
python roboverse_learn/rl/train_rl.py train=AllegroHandPPO task.obs_type="full"
python roboverse_learn/rl/train_rl.py train=AllegroHandPPO task.obs_type="full_state"
```

#### Ant Locomotion
```bash
# Standard Ant locomotion
python roboverse_learn/rl/train_rl.py train=AntPPO
python roboverse_learn/rl/train_rl.py train=AntIsaacGymPPO

# Custom parameters
python roboverse_learn/rl/train_rl.py train=AntPPO task.heading_weight=0.7
python roboverse_learn/rl/train_rl.py train=AntPPO task.actions_cost_scale=0.01
```

#### Anymal Quadruped Robot
```bash
# Flat terrain locomotion
python roboverse_learn/rl/train_rl.py train=AnymalPPO

# Rough terrain with curriculum learning
python roboverse_learn/rl/train_rl.py train=AnymalTerrainPPO

# Custom command ranges
python roboverse_learn/rl/train_rl.py train=AnymalPPO task.command_x_range="[-3.0,3.0]"
python roboverse_learn/rl/train_rl.py train=AnymalPPO task.command_yaw_range="[-2.0,2.0]"

# Terrain difficulty settings
python roboverse_learn/rl/train_rl.py train=AnymalTerrainPPO task.terrain_num_levels=10
python roboverse_learn/rl/train_rl.py train=AnymalTerrainPPO task.terrain_curriculum=True
```

### DMControl Tasks

#### Acrobot
```bash
python roboverse_learn/rl/train_rl.py train=AcrobotSwingupPPO
python roboverse_learn/rl/train_rl.py train=AcrobotSwingupSAC
python roboverse_learn/rl/train_rl.py train=AcrobotSwingupTD3
python roboverse_learn/rl/train_rl.py train=AcrobotSwingupDreamer
```

#### Cartpole Variants
```bash
# Balance
python roboverse_learn/rl/train_rl.py train=CartpoleBalancePPO
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceSAC
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceTD3
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceDreamer

# Balance Sparse
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceSparsePPO
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceSparseSAC
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceSparseTD3
python roboverse_learn/rl/train_rl.py train=CartpoleBalanceSparseDreamer

# Swingup
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupPPO
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupSAC
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupTD3
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupDreamer

# Swingup Sparse
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupSparsePPO
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupSparseSAC
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupSparseTD3
python roboverse_learn/rl/train_rl.py train=CartpoleSwingupSparseDreamer
```

#### Cheetah
```bash
python roboverse_learn/rl/train_rl.py train=CheetahRunPPO
python roboverse_learn/rl/train_rl.py train=CheetahRunSAC
python roboverse_learn/rl/train_rl.py train=CheetahRunTD3
python roboverse_learn/rl/train_rl.py train=CheetahRunDreamer
```

#### Cup (Ball-in-Cup)
```bash
python roboverse_learn/rl/train_rl.py train=CupCatchPPO
python roboverse_learn/rl/train_rl.py train=CupCatchSAC
python roboverse_learn/rl/train_rl.py train=CupCatchTD3
python roboverse_learn/rl/train_rl.py train=CupCatchDreamer
```

#### Finger Manipulation
```bash
# Spin
python roboverse_learn/rl/train_rl.py train=FingerSpinPPO
python roboverse_learn/rl/train_rl.py train=FingerSpinSAC
python roboverse_learn/rl/train_rl.py train=FingerSpinTD3
python roboverse_learn/rl/train_rl.py train=FingerSpinDreamer

# Turn Easy
python roboverse_learn/rl/train_rl.py train=FingerTurnEasyPPO
python roboverse_learn/rl/train_rl.py train=FingerTurnEasySAC
python roboverse_learn/rl/train_rl.py train=FingerTurnEasyTD3
python roboverse_learn/rl/train_rl.py train=FingerTurnEasyDreamer

# Turn Hard
python roboverse_learn/rl/train_rl.py train=FingerTurnHardPPO
python roboverse_learn/rl/train_rl.py train=FingerTurnHardSAC
python roboverse_learn/rl/train_rl.py train=FingerTurnHardTD3
python roboverse_learn/rl/train_rl.py train=FingerTurnHardDreamer
```

#### Hopper
```bash
# Hop
python roboverse_learn/rl/train_rl.py train=HopperHopPPO
python roboverse_learn/rl/train_rl.py train=HopperHopSAC
python roboverse_learn/rl/train_rl.py train=HopperHopTD3
python roboverse_learn/rl/train_rl.py train=HopperHopDreamer

# Stand
python roboverse_learn/rl/train_rl.py train=HopperStandPPO
python roboverse_learn/rl/train_rl.py train=HopperStandSAC
python roboverse_learn/rl/train_rl.py train=HopperStandTD3
python roboverse_learn/rl/train_rl.py train=HopperStandDreamer
```

#### Humanoid
```bash
python roboverse_learn/rl/train_rl.py train=HumanoidWalkPPO
python roboverse_learn/rl/train_rl.py train=HumanoidWalkSAC
python roboverse_learn/rl/train_rl.py train=HumanoidWalkTD3
python roboverse_learn/rl/train_rl.py train=HumanoidWalkDreamer
```

#### Pendulum
```bash
python roboverse_learn/rl/train_rl.py train=PendulumSwingupPPO
python roboverse_learn/rl/train_rl.py train=PendulumSwingupSAC
python roboverse_learn/rl/train_rl.py train=PendulumSwingupTD3
python roboverse_learn/rl/train_rl.py train=PendulumSwingupDreamer
```

#### Reacher
```bash
# Easy
python roboverse_learn/rl/train_rl.py train=ReacherEasyPPO
python roboverse_learn/rl/train_rl.py train=ReacherEasySAC
python roboverse_learn/rl/train_rl.py train=ReacherEasyTD3
python roboverse_learn/rl/train_rl.py train=ReacherEasyDreamer

# Hard
python roboverse_learn/rl/train_rl.py train=ReacherHardPPO
python roboverse_learn/rl/train_rl.py train=ReacherHardSAC
python roboverse_learn/rl/train_rl.py train=ReacherHardTD3
python roboverse_learn/rl/train_rl.py train=ReacherHardDreamer
```

#### Walker
```bash
# Walk
python roboverse_learn/rl/train_rl.py train=WalkerWalkPPO
python roboverse_learn/rl/train_rl.py train=WalkerWalkSAC
python roboverse_learn/rl/train_rl.py train=WalkerWalkTD3
python roboverse_learn/rl/train_rl.py train=WalkerWalkDreamer

# Run
python roboverse_learn/rl/train_rl.py train=WalkerRunPPO
python roboverse_learn/rl/train_rl.py train=WalkerRunSAC
python roboverse_learn/rl/train_rl.py train=WalkerRunTD3
python roboverse_learn/rl/train_rl.py train=WalkerRunDreamer

# Stand
python roboverse_learn/rl/train_rl.py train=WalkerStandPPO
python roboverse_learn/rl/train_rl.py train=WalkerStandSAC
python roboverse_learn/rl/train_rl.py train=WalkerStandTD3
python roboverse_learn/rl/train_rl.py train=WalkerStandDreamer
```

### Legacy Walker Tasks
```bash
python roboverse_learn/rl/train_rl.py train=WalkerPPO
python roboverse_learn/rl/train_rl.py train=WalkerDreamer
```

## Advanced Usage

### Custom Hyperparameters

You can override any configuration parameter:
```bash
python roboverse_learn/rl/train_rl.py train=HumanoidWalkPPO train.ppo.learning_rate=0.0001
```

### Multi-GPU Training

For supported algorithms (PPO, SAC, TD3):
```bash
python roboverse_learn/rl/train_rl.py train=CheetahRunPPO experiment.multi_gpu=True
```

### Visualization

To enable visualization during training (not recommended for performance):
```bash
python roboverse_learn/rl/train_rl.py train=CartpoleBalancePPO environment.headless=False
```

## Evaluation

To evaluate a trained model:
```bash
python roboverse_learn/rl/eval_rl.py train=<TaskAlgorithm> checkpoint_path=<path_to_checkpoint>
```

## Configuration Files

All training configurations are stored in `configs/train/`. Each configuration specifies:
- Task name and robot configuration
- Algorithm-specific hyperparameters
- Training schedule (epochs, steps, etc.)
- Network architecture
- Normalization settings
- Logging and checkpointing frequency

## Output Structure

Training outputs are saved to:
```
outputs/
└── <timestamp>/
    ├── stage1_nn/     # Model checkpoints
    │   ├── best.pth   # Best performing model
    │   └── last.pth   # Most recent checkpoint
    └── stage1_tb/     # TensorBoard logs
```

## Troubleshooting

1. **CUDA/Device Errors**: Ensure you've activated the correct conda environment and set LD_LIBRARY_PATH
2. **Out of Memory**: Reduce `environment.num_envs` or `train.<algo>.batch_size`
3. **Import Errors**: Make sure you've installed RoboVerse with the appropriate extras (e.g., `pip install -e ".[isaacgym]"`)

## Adding New Tasks

To add support for new tasks:
1. Create a task configuration in `metasim/cfg/tasks/`
2. Create corresponding training configurations in `configs/train/`
3. Follow the naming convention: `<TaskName><Algorithm>.yaml`

### Adding IsaacGymEnvs Tasks

For tasks from NVIDIA's IsaacGymEnvs benchmark:
1. Create a wrapper configuration in `metasim/cfg/tasks/isaacgym_envs/`
2. Import the task in `metasim/cfg/tasks/isaacgym_envs/__init__.py`
3. Create training configs with appropriate observation/action dimensions
4. Tasks can be accessed via the `isaacgym_envs:` prefix (e.g., `isaacgym_envs:AllegroHand`)

#### Available IsaacGymEnvs Tasks for Integration
The following tasks from IsaacGymEnvs can be integrated following the above pattern:
- **Manipulation**: FrankaCabinet, FrankaClothManipulation, Ingenuity (drone), ShadowHand
- **Locomotion**: BallBalance, Cartpole, Humanoid, Quadcopter, Trifinger
- **Other**: Factory environments, Industreal tasks

Each task would require creating appropriate wrapper configurations and training configs

## OGBench Tasks

OGBench provides offline goal-conditioned RL benchmarks with various locomotion and manipulation tasks. All tasks are goal-conditioned and use sparse rewards.

### Supported OGBench Environments

#### Navigation Tasks
- **PointMaze**: Simple 2D navigation (medium, large, giant, teleport)
- **AntMaze**: Ant robot navigation (medium, large, giant, teleport)
- **HumanoidMaze**: Humanoid robot navigation (medium, large, giant)
- **AntSoccer**: Ant robot soccer tasks (arena, medium)

#### Manipulation Tasks
- **Cube**: Single to quadruple cube stacking tasks
- **Scene**: Complex scene manipulation
- **Puzzle**: Sliding puzzle solving (3x3, 4x4, 4x5, 4x6)
- **Powderworld**: Discrete action powder simulation (easy, medium, hard)

#### Visual Tasks (Image-based observations)
- **Visual-AntMaze**: Vision-based ant navigation
- **Visual-HumanoidMaze**: Vision-based humanoid navigation
- **Visual-Cube**: Vision-based cube manipulation
- **Visual-Scene**: Vision-based scene manipulation
- **Visual-Puzzle**: Vision-based puzzle solving

### Training Commands

#### PointMaze Navigation
```bash
python roboverse_learn/rl/train_rl.py train=PointMazeMediumPPO
python roboverse_learn/rl/train_rl.py train=PointMazeLargePPO
python roboverse_learn/rl/train_rl.py train=PointMazeGiantPPO
python roboverse_learn/rl/train_rl.py train=PointMazeTeleportPPO
```

#### AntMaze Navigation
```bash
python roboverse_learn/rl/train_rl.py train=AntMazeMediumNavigatePPO
python roboverse_learn/rl/train_rl.py train=AntMazeLargeNavigatePPO
python roboverse_learn/rl/train_rl.py train=AntMazeGiantNavigatePPO
python roboverse_learn/rl/train_rl.py train=AntMazeTeleportPPO
```

#### HumanoidMaze Navigation
```bash
python roboverse_learn/rl/train_rl.py train=HumanoidMazeMediumNavigatePPO
python roboverse_learn/rl/train_rl.py train=HumanoidMazeLargeNavigatePPO
python roboverse_learn/rl/train_rl.py train=HumanoidMazeGiantNavigatePPO
```

#### AntSoccer Tasks
```bash
python roboverse_learn/rl/train_rl.py train=AntSoccerArenaPPO
python roboverse_learn/rl/train_rl.py train=AntSoccerMediumPPO
```

#### Cube Manipulation
```bash
python roboverse_learn/rl/train_rl.py train=CubeSinglePPO
python roboverse_learn/rl/train_rl.py train=CubeDoublePPO
python roboverse_learn/rl/train_rl.py train=CubeTriplePPO
python roboverse_learn/rl/train_rl.py train=CubeQuadruplePPO

# Play variants (different reward structure)
python roboverse_learn/rl/train_rl.py train=CubeDoublePlayPPO
python roboverse_learn/rl/train_rl.py train=CubeQuadruplePlayPPO
```

#### Scene Manipulation
```bash
python roboverse_learn/rl/train_rl.py train=ScenePPO
```

#### Puzzle Solving
```bash
python roboverse_learn/rl/train_rl.py train=Puzzle3x3PPO
python roboverse_learn/rl/train_rl.py train=Puzzle4x4PPO
python roboverse_learn/rl/train_rl.py train=Puzzle4x5PPO
python roboverse_learn/rl/train_rl.py train=Puzzle4x6PPO
```

#### Powderworld (Discrete Actions)
```bash
python roboverse_learn/rl/train_rl.py train=PowderworldEasyPPO
python roboverse_learn/rl/train_rl.py train=PowderworldMediumPPO
python roboverse_learn/rl/train_rl.py train=PowderworldHardPPO
```

### Single-Task Variants

For focused training on specific goals:
```bash
python roboverse_learn/rl/train_rl.py train=AntMazeLargeNavigateSingleTaskPPO
python roboverse_learn/rl/train_rl.py train=CubeDoublePlaySingleTaskPPO
```

### Visual Tasks

Note: Visual tasks are not yet fully supported but configurations are available for future implementation.

### Important Notes

1. **Goal-Conditioned**: All OGBench tasks are goal-conditioned, meaning the agent must learn to reach different goals
2. **Sparse Rewards**: Tasks use sparse rewards (only given when goal is achieved)
3. **Episode Lengths**: Vary by task complexity (200-4000 steps)
4. **Observation Dimensions**: Range from 2 (PointMaze) to 6144 (Powderworld visual)
5. **Action Spaces**: Most tasks use continuous actions except Powderworld (discrete)

Note: OGBench integration is experimental and may require additional configuration for optimal performance. Consider using offline RL algorithms for better results with the provided datasets.
