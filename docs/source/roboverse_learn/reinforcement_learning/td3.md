# TD3 Training

RoboVerse provides CleanRL-based TD3 implementation for continuous control tasks.

## Usage

```bash
# TD3 training with RoboVerse environment
python roboverse_learn/rl/clean_rl/td3.py --task reach_origin --robot franka --sim mjx --num_envs 128
```

## Configuration

Check the file header in `roboverse_learn/rl/clean_rl/td3.py` for available configuration options including:
- Task selection (`--task`)
- Robot type (`--robot`)
- Simulator backend (`--sim`)
- Training hyperparameters (`--num_envs`, `--learning_rate`, etc.)

## Features

- Based on [CleanRL](https://github.com/vwxyzjn/cleanrl) TD3 implementation
- Vectorized environment support with RoboVerse
- Twin delayed updates for stable training
- Episode tracking without relying on info dicts
