# Prerequisites

## Use All Simulators Supported by MetaSim

To develop with the full capability of MetaSim, it is recommended to [install with Docker](./docker.md) on Ubuntu. The recommended setup is as follows:

### System Requirements

| Component        | Ideal           | Minimum         |
|------------------|-----------------|-----------------|
| **CPU**          | Intel Core i9   | Intel Core i7   |
| **Cores**        | 32 cores        | 4 cores         |
| **GPU**          | RTX 4090        | RTX 3070 / 4060 |
| **VRAM**         | 24GB            | 8GB             |
| **RAM**          | 64GB            | 32GB            |
| **Free Storage** | 500GB SSD       | 50GB SSD        |

### Environment Setup

| Component           | Recommended Setup (tested) | Alternative Setup (not fully tested) |
|---------------------|----------------------------|----------------------------------|
| **Operating System** | Ubuntu 22.04 LTS          | Ubuntu 20.04 LTS                 |
| **NVIDIA Driver**    | 535                       | 550, 570                         |
| **CUDA Toolkit**     | 11.8                      | 12.*                             |

- Check your NVIDIA driver version with `nvidia-smi`.
- Check your CUDA toolkit version with `nvcc --version`.


## Light-weight Installation

If you only want to try MetaSim, you can [directly install](./installation.rst) the specific simulator backends as you need.

You may choose one or more of the following simulators, according to your operating system.

| OS      | IsaacLab v1.4 | IsaacLab v2 | IsaacGym | MuJoCo | SAPIEN2 | SAPIEN3 | Genesis | PyBullet |
|---------|---------------|-------------|----------|--------|---------|---------|---------|----------|
| MacOS   |               |             |          | ✓      |         | ✓       | ✓       |          |
| Windows | ✓             | ✓           |          | ✓      |         | ✓       | ✓       |          |
| Ubuntu  | ✓             | ✓           | ✓        | ✓      | ✓       | ✓       | ✓       | ✓        |

```{note}
RoboVerse team hasn't got the chance to fully test MetaSim on MacOS and Windows. Please let us know if you have any issues.
```

For more information about the supported platforms, please refer to the official guide of each simulator:
- [IsaacLab v1.4](https://isaac-sim.github.io/IsaacLab/v1.4.1/source/setup/installation/pip_installation.html)
- [IsaacLab v2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [IsaacGym](https://docs.robotsfan.com/isaacgym/install.html)
- [MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html)
- [SAPIEN2](https://sapien.ucsd.edu/docs/latest/tutorial/basic/installation.html)
- [SAPIEN3](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)
- [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)
- [PyBullet](https://github.com/bulletphysics/bullet3)
