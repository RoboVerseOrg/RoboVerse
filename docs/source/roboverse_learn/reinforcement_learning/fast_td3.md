# Fast TD3

## Overview

Fast TD3 (Twin Delayed Deep Deterministic Policy Gradient) is an advanced implementation with distributional critics and various optimizations for high-performance reinforcement learning training.

## Quick Start

```bash
# Basic training with default H1 humanoid configuration
python get_started/rl/1_fttd3.py
```

## Key Features

- **Distributional Critics**: Uses distributional Q-networks with 101 atoms for better value estimation
- **Automatic Mixed Precision**: Supports FP16/BF16 training for faster performance
- **Empirical Normalization**: Online observation normalization for stable training
- **Advanced Replay Buffer**: N-step returns and asymmetric observations support
- **PyTorch Compilation**: Optional compilation for additional speed improvements
- **Comprehensive Evaluation**: Periodic evaluation with detailed metrics
- **Automatic Video Generation**: Creates rollout videos for visualization

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

- **H1 Humanoid**: Default configuration optimized for locomotion
- **Franka**: Supported with configuration changes
- **Custom Robots**: Define in robot configurations

## Supported Simulators

1. **MJX**: JAX-based MuJoCo (default, fastest)
2. **Isaac Gym**: NVIDIA GPU-accelerated physics
3. **Isaac Lab**: Next-generation Isaac simulation
4. **MuJoCo**: Standard MuJoCo physics
5. **Genesis**: Multi-physics simulation

## Algorithm Details

### Distributional Q-Learning

Fast TD3 uses distributional Q-networks that model the full return distribution rather than just the expected value:

- **Atoms**: 101 discrete points representing the value distribution
- **Value Range**: Configurable min/max values (`v_min`, `v_max`)
- **Projection**: Categorical projection for target distribution

### Clipped Double Q-Learning (CDQ)

When `use_cdq=True`, the algorithm uses the minimum of two Q-networks for more conservative value estimation:

```python
qf_value = torch.minimum(qf1_value, qf2_value)  # CDQ enabled
```

### Advanced Replay Buffer

The replay buffer supports:

- **N-step Returns**: Multi-step bootstrapping
- **Asymmetric Observations**: Separate actor/critic observations
- **Efficient Sampling**: Vectorized operations for speed

### Empirical Normalization

Online normalization of observations using Welford's algorithm:

- **Running Statistics**: Maintains mean and variance
- **Stability**: Prevents gradient explosion from unnormalized inputs
- **Efficiency**: Updates during training without extra passes

## Performance Optimizations

### Automatic Mixed Precision

```python
# Enable AMP for faster training
CONFIG["amp"] = True
CONFIG["amp_dtype"] = "fp16"  # or "bf16"
```

### PyTorch Compilation

```python
# Compile critical functions for speed
CONFIG["compile"] = True
```

### Vectorized Environments

```python
# Use many environments for sample efficiency
CONFIG["num_envs"] = 1024
```

## Output and Monitoring

### Files Generated

- **Model Checkpoints**: Saved if `save_interval > 0`
- **Rollout Video**: `output/rollout.mp4`
- **WandB Logs**: If `use_wandb=True`

### Metrics Tracked

- Actor/critic losses
- Q-function statistics (min/max values)
- Gradient norms
- Episode returns and lengths
- Training speed (steps per second)

## Customization Examples

### Different Robot

```python
CONFIG["robots"] = ["franka"]
CONFIG["task"] = "reach_origin"
```

### Different Simulator

```python
CONFIG["sim"] = "isaacgym"
```

### Hyperparameter Tuning

```python
CONFIG["critic_learning_rate"] = 0.001
CONFIG["batch_size"] = 16384
CONFIG["num_updates"] = 8
```

## Troubleshooting

### Memory Issues

- Reduce `num_envs` or `batch_size`
- Disable mixed precision: `CONFIG["amp"] = False`

### Slow Training

- Enable compilation: `CONFIG["compile"] = True`
- Use faster simulator: `CONFIG["sim"] = "mjx"`
- Increase batch size if memory allows

### Unstable Training

- Disable CDQ: `CONFIG["use_cdq"] = False`
- Reduce learning rates
- Adjust value distribution range (`v_min`, `v_max`)

## Advanced Usage

### Custom Task Integration

1. Define task in task registry
2. Update `CONFIG["task"]`
3. Adjust network sizes and hyperparameters as needed

### Multi-Task Training

Currently single-task focused. For multi-task training, consider:

- Task-specific value ranges
- Shared vs. separate networks
- Task sampling strategies

### Distributed Training

Current implementation is single-GPU. For multi-GPU:

- Use data parallel environments
- Adjust batch sizes accordingly
- Consider gradient accumulation

## Comparison with PPO

| Feature | Fast TD3 | PPO |
|---------|----------|-----|
| **Algorithm Type** | Off-policy | On-policy |
| **Sample Efficiency** | Higher | Lower |
| **Stability** | Requires tuning | More stable |
| **Continuous Control** | Excellent | Good |
| **Implementation** | Custom | Stable-Baselines3 |
| **Memory Usage** | Higher (replay buffer) | Lower |

## See Also

- [RL Infrastructure](../../metasim/get_started/advanced/rl_example/infrastructure.md) - Complete setup guide
- [PPO Training](ppo.md) - Alternative on-policy algorithm
- [Humanoid Bench RL](humanoidbench_rl.md) - Specialized humanoid tasks