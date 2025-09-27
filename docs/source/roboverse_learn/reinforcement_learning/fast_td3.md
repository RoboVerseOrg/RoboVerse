# Fast TD3

## Overview

Fast TD3 (Twin Delayed Deep Deterministic Policy Gradient) is an advanced implementation with distributional critics and various optimizations for high-performance reinforcement learning training.

## Quick Start

## Configuration

The script uses a `CONFIG` dictionary for all parameters. Key options include:

```python
CONFIG = {
    # Environment
    "sim": "mjx",                    # Simulator backend
    "robots": ["h1"],               # Robot type
    "task": "humanoid.run",         # Task name
    "num_envs": 1024,              # Number of parallel environments
    "decimation": 10,              # Control decimation
    
    # Training
    "total_timesteps": 1500,       # Total training steps
    "batch_size": 32768,           # Batch size for updates
    "buffer_size": 20480,          # Replay buffer size
    
    # Algorithm
    "gamma": 0.99,                 # Discount factor
    "tau": 0.1,                    # Target network update rate
    "policy_frequency": 2,         # Policy update frequency
    "num_updates": 12,             # Updates per step
    
    # Networks
    "critic_learning_rate": 0.0003,
    "actor_learning_rate": 0.0003,
    "critic_hidden_dim": 1024,
    "actor_hidden_dim": 512,
    
    # Distributional Q-learning
    "num_atoms": 101,
    "v_min": -250.0,
    "v_max": 250.0,
    
    # Optimizations
    "use_cdq": True,               # Clipped Double Q-learning
    "compile": True,               # PyTorch compilation
    "obs_normalization": True,     # Observation normalization
    "amp": True,                   # Automatic mixed precision
    "amp_dtype": "fp16",          # Precision type
    
    # Logging
    "use_wandb": False,           # Weights & Biases integration
    "eval_interval": 700,         # Evaluation frequency
    "save_interval": 700,         # Model saving frequency
}
```

## Supported Tasks

- **Humanoid Locomotion**: `humanoid.run`, `humanoid.walk`, `humanoid.stand`
- **Reaching Tasks**: `reach_origin` (modify config)
- **Custom Tasks**: Via task registry

## Supported Robots

- **G1 Humanoid**: Default configuration optimized for locomotion
- **Franka**: Supported with configuration changes
- **Custom Robots**: Define in robot configurations

## See Also

- [RL Infrastructure](../../metasim/get_started/advanced/rl_example/infrastructure.md) - Complete setup guide
- [PPO Training](ppo.md) - Alternative on-policy algorithm