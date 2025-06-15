# RoboVerse Reinforcement Learning

This directory contains the reinforcement learning infrastructure for RoboVerse, supporting multiple state-of-the-art RL algorithms across various robotic tasks and benchmarks.

## Supported Algorithms

- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)
- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **Dreamer** (Model-based RL with world models)

## Supported Tasks

### IsaacGymEnvs Tasks
- **AllegroHand**: Dexterous manipulation with the Allegro robotic hand
- **Ant**: Quadruped locomotion
- **Anymal**: Quadruped robot locomotion on flat and rough terrain

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

```bash
# AllegroHand
python roboverse_learn/rl/train_rl.py train=AllegroHandPPO
python roboverse_learn/rl/train_rl.py train=AllegroHandTD3

# Ant
python roboverse_learn/rl/train_rl.py train=AntPPO
python roboverse_learn/rl/train_rl.py train=AntIsaacGymPPO

# Anymal
python roboverse_learn/rl/train_rl.py train=AnymalPPO
python roboverse_learn/rl/train_rl.py train=AnymalTerrainPPO
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

## OGBench Tasks

OGBench provides offline goal-conditioned RL benchmarks with locomotion and manipulation tasks:

### Supported OGBench Environments
- **Locomotion**: AntMaze (large, medium, giant), HumanoidMaze (large, medium, giant)
- **Manipulation**: Cube (double, triple, quadruple play)

### Training Commands
```bash
# AntMaze tasks
python roboverse_learn/rl/train_rl.py train=AntMazeLargeNavigatePPO

# HumanoidMaze tasks
python roboverse_learn/rl/train_rl.py train=HumanoidMazeLargeNavigatePPO

# Cube manipulation tasks
python roboverse_learn/rl/train_rl.py train=CubeDoublePlayPPO
python roboverse_learn/rl/train_rl.py train=CubeQuadruplePlayPPO
```

Note: OGBench integration is experimental and may require additional configuration for optimal performance (we need offline RL algos)
