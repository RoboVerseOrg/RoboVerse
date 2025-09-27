# CleanRL Algorithms for RoboVerse

This directory contains RL algorithms adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl) for use with RoboVerse environments.

## Available Algorithms

- **PPO** (`ppo.py`) - Proximal Policy Optimization
- **SAC** (`sac.py`) - Soft Actor-Critic
- **TD3** (`td3.py`) - Twin Delayed Deep Deterministic Policy Gradient


## Usage

Each algorithm can be run directly with command line arguments:

```bash
# PPO
python roboverse_learn/rl/clean_rl/ppo.py --task reach_origin --robot franka --sim mjx --num_envs 2048

# SAC
python roboverse_learn/rl/clean_rl/sac.py --task reach_origin --robot franka --sim mjx --num_envs 128

# TD3
python roboverse_learn/rl/clean_rl/td3.py --task reach_origin --robot franka --sim mjx --num_envs 128
```

## License

- Based on CleanRL code licensed under MIT License
- See individual file headers for specific license information

## Acknowledgments

- Based on CleanRL by [vwxyzjn](https://github.com/vwxyzjn/cleanrl)
- Adapted for RoboVerse by the RoboVerse team
- Buffer implementation adapted from stable-baselines3
