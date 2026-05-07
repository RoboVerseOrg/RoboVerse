# RoboVerse Learn

**RoboVerse Learn** provides a comprehensive suite of learning algorithms for robot policy training. It integrates seamlessly with MetaSim environments, enabling end-to-end training pipelines for both imitation learning and reinforcement learning.

---

## Overview

::::{grid} 2
:gutter: 3

:::{grid-item-card} Imitation Learning
:link: imitation_learning/diffusion_policy
:link-type: doc

Learn from demonstrations using state-of-the-art IL algorithms including Diffusion Policy, ACT, and Vision-Language-Action models.
:::

:::{grid-item-card} Reinforcement Learning
:link: reinforcement_learning/ppo
:link-type: doc

Train policies through trial and error with PPO, TD3, SAC, and specialized algorithms for humanoid control.
:::

::::

---

## Quick Start

### Training with Imitation Learning

```bash
# Collect demonstrations
python scripts/collect_demo.py --task pick_cube --episodes 100

# Train Diffusion Policy
python roboverse_learn/il/train_dp.py \
    --task pick_cube \
    --data_path ./demos/pick_cube \
    --epochs 100
```

### Training with Reinforcement Learning

```bash
# Train PPO on a manipulation task
python roboverse_learn/rl/train_ppo.py \
    --task pick_cube \
    --robot franka \
    --num_envs 1024 \
    --steps 10000000

# Train FastTD3 with MJX backend
python roboverse_learn/rl/train_fast_td3.py \
    --task pick_cube \
    --simulator mjx \
    --num_envs 4096
```

---

## Features

### Unified Interface

All algorithms share a common interface with MetaSim:

```python
from roboverse_learn.il import DiffusionPolicy
from roboverse_learn.rl import PPO

# IL training
policy = DiffusionPolicy(config)
policy.train(env, demonstrations)

# RL training
agent = PPO(config)
agent.train(env, total_steps=1000000)
```

### GPU-Accelerated Training

- Vectorized environments for parallel data collection
- Batch policy inference on GPU
- Mixed-precision training support

### Experiment Management

- Weights & Biases integration
- TensorBoard logging
- Checkpoint management
- Hyperparameter sweeps

---

## Installation

Most algorithms are included in the base installation. For specific algorithms:

```bash
# Full IL suite
pip install -e ".[il]"

# Full RL suite
pip install -e ".[rl]"

# Vision-Language models
pip install -e ".[vla]"
```

---

## Contributing

Want to add a new algorithm? See our [Contributing Guide](imitation_learning/contributing.md) for instructions on integrating new methods.

---

```{toctree}
:caption: Imitation Learning
:titlesonly:

imitation_learning/diffusion_policy
imitation_learning/ACT
imitation_learning/openvla
imitation_learning/smolvla
imitation_learning/rdt
imitation_learning/octo
imitation_learning/contributing
```

```{toctree}
:caption: Reinforcement Learning
:titlesonly:

reinforcement_learning/ppo
reinforcement_learning/fast_td3
reinforcement_learning/sac
reinforcement_learning/td3
reinforcement_learning/skillblender_rl
reinforcement_learning/humanoid
```
