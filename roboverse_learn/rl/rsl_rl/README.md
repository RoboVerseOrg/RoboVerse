# RSL-RL for RoboVerse

Proximal Policy Optimization (PPO) implementation using the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) library.

## Installation

```bash
pip install rsl_rl
```

## Quick Start

```bash
# Train G1 walking with default settings
python -m roboverse_learn.rl.rsl_rl.ppo \
    --task walk_g1_dof29 \
    --robot g1 \
    --sim isaacgym \
    --num-envs 4096

# Train with custom hyperparameters
python -m roboverse_learn.rl.rsl_rl.ppo \
    --task walk_h1_dof29 \
    --learning-rate 5e-4 \
    --num-learning-epochs 8 \
    --clip-param 0.3

# Enable WandB logging
python -m roboverse_learn.rl.rsl_rl.ppo \
    --task walk_g1_dof29 \
    --use-wandb \
    --wandb-project my-project
```

## Configuration

All parameters are specified via command-line arguments. Use `--help` to see all options:

```bash
python -m roboverse_learn.rl.rsl_rl.ppo --help
```

### Key Parameters

**Environment**:
- `--task`: Task name (e.g., walk_g1_dof29, walk_h1_dof29)
- `--robot`: Robot type (g1, h1, etc.)
- `--sim`: Simulator backend (isaacgym, isaacsim, mujoco, mjx)
- `--num-envs`: Number of parallel environments (default: 4096)

**Training**:
- `--max-iterations`: Training iterations (default: 50000)
- `--num-steps-per-env`: Steps per environment per update (default: 24)
- `--save-interval`: Checkpoint save frequency (default: 100)

**PPO Algorithm**:
- `--learning-rate`: Learning rate (default: 1e-3)
- `--num-learning-epochs`: Epochs per update (default: 5)
- `--clip-param`: PPO clipping parameter (default: 0.2)
- `--gamma`: Discount factor (default: 0.99)
- `--lam`: GAE lambda (default: 0.95)

## Output

Models are saved to `models/{exp_name}/{task}/`:
- Checkpoints: `model_*.pt` (every `save_interval` iterations)
- Final policy: `policy.pt` (JIT-scripted for inference)
- Logs: TensorBoard logs

## Credits

Based on [rsl_rl](https://github.com/leggedrobotics/rsl_rl) by Robotic Systems Lab, ETH Zurich.
