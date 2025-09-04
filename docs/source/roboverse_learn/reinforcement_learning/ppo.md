# PPO Training

## Get Started PPO (Simple Examples)

For quick start examples and simpler PPO training, see the MetaSim get started tutorials:

- **Infrastructure Overview**: See [RL Infrastructure](../../metasim/get_started/advanced/rl_example/infrastructure.md) for complete setup
- **Quick Examples**: See [Quick Start Examples](../../metasim/get_started/advanced/rl_example/quick_examples.md) for ready-to-run commands
- **PPO Reaching Tutorial**: See [PPO Reaching](../../metasim/get_started/advanced/rl_example/0_ppo_reaching.md) for detailed walkthrough

### Simple PPO Commands

```bash
# Basic PPO training with Franka robot
python get_started/rl/0_ppo.py --task reach_origin --robot franka --sim isaacgym

# PPO with Gym interface
python get_started/rl/0_ppo_gym.py --sim mjx --num-envs 256
```

## Choosing Between Approaches

- **Use RoboVerse Learn PPO** for: Advanced research, complex tasks, customizable configs
- **Use Get Started PPO** for: Quick prototyping, simple tasks, learning the basics
