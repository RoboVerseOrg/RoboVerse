# SAC

RoboVerse provides CleanRL-based SAC implementation for continuous control tasks.

## Usage

```bash
# SAC training with RoboVerse environment
python roboverse_learn/rl/clean_rl/sac.py --task reach_origin --robot franka --sim mjx --num_envs 128
```

## Configuration

Check the file header in `roboverse_learn/rl/clean_rl/sac.py` for available configuration options including:
- Task selection (`--task`)
- Robot type (`--robot`)
- Simulator backend (`--sim`)
- Training hyperparameters (`--num_envs`, `--learning_rate`, etc.)

## Features

- Based on [CleanRL](https://github.com/vwxyzjn/cleanrl) SAC implementation
- Vectorized environment support with RoboVerse
- Automatic entropy tuning
- Episode tracking without relying on info dicts
