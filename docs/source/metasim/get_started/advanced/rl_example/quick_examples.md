# Quick Start Examples

This page provides ready-to-run examples for getting started with RL training in RoboVerse.

## Basic PPO Training

### Reaching Task with Franka Robot

```bash
# Train PPO on reaching task with Franka robot in Isaac Gym
python get_started/rl/0_ppo.py --task reach_origin --robot franka --sim isaacgym --num-envs 128

# Train with MuJoCo backend
python get_started/rl/0_ppo.py --sim mujoco --headless

# Use Gym interface for cleaner integration
python get_started/rl/0_ppo_gym.py --sim mjx --num-envs 256
```

### Different Simulators

```bash
# Isaac Gym (GPU-accelerated)
python get_started/rl/0_ppo.py --sim isaacgym --headless

# Isaac Lab (next-generation)
python get_started/rl/0_ppo.py --sim isaaclab --headless

# MuJoCo (CPU/GPU)
python get_started/rl/0_ppo.py --sim mujoco --headless

# Genesis (multi-physics)
python get_started/rl/0_ppo.py --sim genesis --headless

# MJX (JAX-based, fastest)
python get_started/rl/0_ppo_gym.py --sim mjx --headless
```

## Fast TD3 Humanoid Training

```bash
# Train H1 humanoid on running task
python get_started/rl/1_fttd3.py

# Modify CONFIG in the script for different settings:
# - Change "robots": ["franka"] for different robot
# - Change "task": "reach_origin" for different task
# - Change "sim": "isaacgym" for different simulator
```
