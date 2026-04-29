# Frequently Asked Questions

This page answers common questions about RoboVerse. For simulator-specific issues, see the [Troubleshooting](../metasim/troubleshooting/common.md) section.

---

## General Questions

### What is RoboVerse?

RoboVerse is a unified platform for scalable and generalizable robot learning. See the [project page](https://roboverseorg.github.io) and [paper](https://arxiv.org/abs/2504.18904) for details.

### What simulators are supported?

See the [Support Matrix](../metasim/features/support_matrix.rst) for the full list of supported simulators and features.

### What robots are available?

See [Robots Dataset](../dataset_benchmark/dataset/robots.md) for the complete list of available robots.

---

## Installation

### What are the system requirements?

**Minimum Requirements:**
- Ubuntu 20.04+ or macOS 12+
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with CUDA 11.8+ (for GPU features)

**Recommended:**
- Ubuntu 22.04
- Python 3.10
- 32GB RAM
- NVIDIA RTX 3090 or better

### How do I install RoboVerse?

```bash
# Clone the repository
git clone https://github.com/RoboVerseOrg/RoboVerse.git
cd RoboVerse

# Create conda environment
conda create -n roboverse python=3.10
conda activate roboverse

# Install RoboVerse and the MetaSim simulator extra you need
python -m pip install -e ".[mujoco]"
```

See the [Installation Guide](../metasim/get_started/installation.rst) for detailed instructions.

### Can I use Docker?

Yes! We provide official Docker images:

```bash
docker pull roboverseorg/roboverse:latest
docker run -it --gpus all roboverseorg/roboverse:latest
```

See [Docker Guide](../metasim/get_started/docker.md) for more details.

### How do I install optional dependencies?

Different features require additional packages:

```bash
# For development tools and MuJoCo support
python -m pip install -e ".[dev,mujoco]"

# For Isaac Sim support
python -m pip install -e ".[isaacsim]"

# For learning dependencies
python -m pip install -e ".[learn]"

# For VLA dependencies
python -m pip install -e ".[vla]"
```

See [Advanced Installation](../metasim/get_started/advanced_installation/index.md) for all options.

---

## Usage

### How do I run my first simulation?

```python
from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.mujoco_handler import MujocoHandler
from roboverse_pack.robots.franka import FrankaCfg

# Configure scenario
scenario = ScenarioCfg(
    robots=[FrankaCfg()],
    num_envs=1,
)

# Create and launch handler
handler = MujocoHandler(scenario)
handler.launch()

# Step simulation
for _ in range(1000):
    handler.step()
```

See [Quick Start Tutorials](../metasim/get_started/quick_start/index.md) for more examples.

### How do I switch between simulators?

Simply change the handler class while keeping the same scenario:

```python
from metasim.sim.mujoco_handler import MujocoHandler
from metasim.sim.sapien_handler import SapienHandler
from metasim.sim.isaacsim_handler import IsaacSimHandler

# Same scenario works with any simulator
handler = MujocoHandler(scenario)    # or
handler = SapienHandler(scenario)    # or
handler = IsaacSimHandler(scenario)
```

### How do I run parallel environments?

```python
scenario = ScenarioCfg(
    robots=[FrankaCfg()],
    num_envs=1024,  # Run 1024 environments in parallel
)
handler = MujocoHandler(scenario)
handler.launch()
```

GPU-accelerated simulators (Isaac Sim, MJX, Genesis) provide best performance for large-scale parallelism.

### How do I train a policy?

**Reinforcement Learning:**
```bash
python roboverse_learn/rl/train_ppo.py --task pick_cube --robot franka
```

**Imitation Learning:**
```bash
python roboverse_learn/il/train_dp.py --task pick_cube --data_path /path/to/demos
```

See [RoboVerse Learn](../roboverse_learn/index.md) for algorithm-specific guides.

---

## Troubleshooting

### I get "libGL error" on Linux

Install the required OpenGL libraries:

```bash
conda install -c conda-forge libstdcxx-ng
# or
sudo apt-get install libgl1-mesa-glx libegl1
```

### MuJoCo shows "egl error"

Install EGL libraries:

```bash
sudo apt-get install libegl1 libgl1-mesa-glx
```

### MJX shows "DNN library initialization failed"

Install a compatible cuDNN version:

```bash
pip install nvidia-cudnn-cu12==9.10.2.21
```

### Isaac Sim fails to start

Ensure you have:
1. NVIDIA driver version 525+
2. Isaac Sim 2023.1.1 or later installed
3. Sufficient GPU memory (8GB+ recommended)

See [Isaac Sim Troubleshooting](../metasim/troubleshooting/isaaclab.md) for more solutions.

### How do I report a bug?

1. Check existing [GitHub Issues](https://github.com/RoboVerseOrg/RoboVerse/issues)
2. If not found, create a new issue with:
   - System information (OS, Python version, GPU)
   - Steps to reproduce
   - Full error traceback
   - Expected vs actual behavior

---

## Contributing

### How can I contribute?

We welcome contributions! See our guides:

- [Contributing New Robots](../metasim/developer_guide/contributing_new_robot.md)
- [Converting Assets](../metasim/developer_guide/converting_asset.md)
- [Code Style Guide](../metasim/developer_guide/docstring.md)

### How do I add a new robot?

1. Prepare robot assets (URDF/MJCF/USD)
2. Create a configuration class in `roboverse_pack/robots/`
3. Add tests for the new robot
4. Submit a pull request

See [Contributing New Robot](../metasim/developer_guide/contributing_new_robot.md) for details.

### How do I add a new task?

1. Create a task class inheriting from `BaseTaskEnv`
2. Define reward function and success criteria
3. Configure default scenario with appropriate robots/objects
4. Add to the task registry

See [Task System](../metasim/concept/task.md) for the task API.

---

## Community

### Where can I get help?

- **Documentation**: [roboverse.wiki](https://roboverse.wiki)
- **GitHub Discussions**: [Ask questions](https://github.com/RoboVerseOrg/RoboVerse/discussions)
- **Discord**: [Join community](https://discord.gg/6e2CPVnAD3)
- **GitHub Issues**: [Report bugs](https://github.com/RoboVerseOrg/RoboVerse/issues)

### How do I cite RoboVerse?

```bibtex
@misc{geng2025roboverse,
    title={RoboVerse: Towards a Unified Platform, Dataset and Benchmark 
           for Scalable and Generalizable Robot Learning},
    author={Haoran Geng and Feishi Wang and Songlin Wei and ...},
    year={2025},
    eprint={2504.18904},
    archivePrefix={arXiv},
}
```
