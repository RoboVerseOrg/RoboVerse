# PPO

RoboVerse provides three PPO implementations with different features and use cases:

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

Configuration defaults live in `roboverse_learn/rl/configs/clean_rl/ppo.py` (parsed with `tyro`). Use `--help` for all options, including:
- Task selection (`--task`)
- Robot type (`--robot`)
- Simulator backend (`--sim`) 
- Training hyperparameters (`--num_envs`, `--learning_rate`, etc.)

## 3. RSL-RL PPO (OnPolicyRunner)

Based on [rsl_rl](https://github.com/leggedrobotics/rsl_rl) for high-throughput on-policy training with asymmetric observations.

### Usage

```bash
# RSL-RL PPO for Unitree G1 walking
python -m roboverse_learn.rl.rsl_rl.ppo --task walk_g1_dof29 --robot g1 --sim isaacgym --num-envs 4096
```

### Configuration

- Install dependency: `pip install rsl_rl`
- CLI defaults: `roboverse_learn/rl/configs/rsl_rl/ppo.py` (tyro). Run with `--help` to see environment, training, and PPO hyperparameters (`--num-steps-per-env`, `--max-iterations`, `--clip-param`, etc.).
- Outputs: checkpoints, TensorBoard logs, and final scripted policy are saved under `models/{exp_name}/{task}/` by default (override with `--model-dir`).

## Quick Start Examples

For detailed tutorials and infrastructure setup:

- **Infrastructure Overview**: See [RL Infrastructure](../../metasim/get_started/advanced/rl_example/infrastructure.md) for complete setup
- **Quick Examples**: See [Quick Start Examples](../../metasim/get_started/advanced/rl_example/quick_examples.md) for ready-to-run commands
