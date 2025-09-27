# PPO

RoboVerse provides two PPO implementations with different features and use cases:

## 1. Stable-Baselines3 PPO (Recommended for Beginners)

Based on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), this implementation provides a more user-friendly interface with comprehensive configuration options.

### Usage

```bash
# Basic PPO training with Franka robot
python get_started/rl/0_ppo.py --task reach_origin --robot franka --sim isaacgym

# PPO with Gym interface
python get_started/rl/0_ppo_gym_style.py --sim mjx --num-envs 256
```

### Configuration

Check the file header in `get_started/rl/0_ppo.py` for available configuration options including:
- Task selection (`--task`)
- Robot type (`--robot`) 
- Simulator backend (`--sim`)
- Environment settings

## 2. CleanRL PPO 

Based on [CleanRL](https://github.com/vwxyzjn/cleanrl), this implementation provides a more minimal and educational approach with direct algorithm implementation.

### Usage

```bash
# CleanRL PPO with RoboVerse environment
python roboverse_learn/rl/clean_rl/ppo.py --task reach_origin --robot franka --sim mjx --num_envs 2048
```

### Configuration

Check the file header in `roboverse_learn/rl/clean_rl/ppo.py` for available configuration options including:
- Task selection (`--task`)
- Robot type (`--robot`)
- Simulator backend (`--sim`) 
- Training hyperparameters (`--num_envs`, `--learning_rate`, etc.)

## Quick Start Examples

For detailed tutorials and infrastructure setup:

- **Infrastructure Overview**: See [RL Infrastructure](../../metasim/get_started/advanced/rl_example/infrastructure.md) for complete setup
- **Quick Examples**: See [Quick Start Examples](../../metasim/get_started/advanced/rl_example/quick_examples.md) for ready-to-run commands
