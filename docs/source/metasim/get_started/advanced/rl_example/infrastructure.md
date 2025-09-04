# RL Infrastructure Overview

> ⚠️ **CONSTRUCTION WARNING** ⚠️  
> This RL infrastructure is currently under active construction. Features, APIs, and interfaces may change significantly. Use at your own discretion and expect potential breaking changes.

## Available Scripts

All scripts are located in `get_started/rl/`:

### 1. PPO Training (`0_ppo.py`)

Direct implementation using RLTaskEnv with Stable Baselines3 PPO.

**Usage:**

```bash
# Basic usage with default settings
python get_started/rl/0_ppo.py

# Custom task and robot
python get_started/rl/0_ppo.py --task reach_origin --robot franka --sim isaacgym

# Adjust environment count and headless mode
python get_started/rl/0_ppo.py --num-envs 256 --headless

# Different simulators
python get_started/rl/0_ppo.py --sim mujoco
python get_started/rl/0_ppo.py --sim genesis
python get_started/rl/0_ppo.py --sim isaaclab
```

**Arguments:**

- `--task`: Task name (default: `reach_origin`)
- `--robot`: Robot type (default: `franka`)
- `--num-envs`: Number of parallel environments (default: `128`)
- `--sim`: Simulator backend (`isaacgym`, `isaaclab`, `mujoco`, `genesis`, `mjx`)
- `--headless`: Run without GUI (flag)

**Outputs:**

- Model saved to: `get_started/output/rl/0_ppo_reaching_{sim}`

### 2. PPO Training with Gym Interface (`0_ppo_gym.py`)

Uses Gymnasium-compatible interface with cleaner integration.

**Usage:**

```bash
# Basic training
python get_started/rl/0_ppo_gym.py

# With different backend
python get_started/rl/0_ppo_gym.py --sim mjx --device cuda

# Custom configuration
python get_started/rl/0_ppo_gym.py --task reach_origin --robot franka --num-envs 64
```

**Arguments:**

- `--task`: Task name (default: `reach_origin`)
- `--robot`: Robot type (default: `franka`)
- `--num-envs`: Number of environments (default: `128`)
- `--sim`: Simulator (`isaaclab`, `isaacgym`, `mujoco`, `genesis`, `mjx`)
- `--headless`: Headless mode (flag)
- `--device`: Device (`cuda`, `cpu`)

### 3. Fast TD3 Training (`1_fttd3.py`)

Advanced TD3 implementation with distributional critics and various optimizations.

**Usage:**

```bash
# Basic training
python get_started/rl/1_fttd3.py

# The script uses a CONFIG dictionary for configuration
# Key parameters can be modified in the CONFIG section
```

**Key Configuration Options:**

```python
CONFIG = {
    "sim": "mjx",                    # Simulator backend
    "robots": ["h1"],               # Robot type
    "task": "humanoid.run",         # Task name
    "num_envs": 1024,              # Number of environments
    "total_timesteps": 1500,       # Training timesteps
    "batch_size": 32768,           # Batch size
    "learning_rate": 0.0003,       # Learning rate
    "use_wandb": False,            # Weights & Biases logging
}
```


## Environment Integration

### Supported Tasks

- `reach_origin`: Basic reaching task
- `humanoid.run`: Humanoid locomotion
- Custom tasks via task registry

### Supported Robots

- `franka`: Franka Panda arm
- `h1`: H1 humanoid robot
- Additional robots available in robot configurations

### Simulator Backends

1. **Isaac Gym**: NVIDIA's physics simulation
2. **Isaac Lab**: Next-generation Isaac simulation
3. **MuJoCo**: Fast physics simulation
4. **Genesis**: Multi-physics simulation
5. **MJX**: JAX-based MuJoCo implementation

## Dependencies

### Core Requirements

```bash
# Core metasim framework
pip install -e .

# RL libraries
pip install stable-baselines3
pip install torch torchvision
pip install tensordict
pip install loguru
pip install tyro
```
