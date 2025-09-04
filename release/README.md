# Meta-Sim

A unified simulation framework for robotics that supports multiple physics simulators.

## Installation

### Basic Installation

```bash
pip install metasim-core
```

### Installation with Specific Simulators

Install with support for specific simulators:

```bash
# For MuJoCo support
pip install metasim-core[mujoco]

# For PyBullet support
pip install metasim-core[pybullet]

# For Sapien3 support
pip install metasim-core[sapien3]

# For Genesis support
pip install metasim-core[genesis]

# For all simulators (except Isaac)
pip install metasim-core[all]
```


## Quick Start

```python
import metasim
from metasim.sim import base

# Your simulation code here
```

## Supported Simulators

- **MuJoCo**: High-performance physics simulation
- **PyBullet**: Open-source physics simulation
- **Sapien**: Photorealistic rendering and physics
- **Genesis**: Next-generation simulation platform
- **Isaac Gym**: NVIDIA's physics simulation (requires separate installation)
- **Isaac Lab**: Advanced robotics simulation
- **Blender**: 3D creation and rendering

## License

MIT License


## For developers:

compiling the package:

```bash
cd release
python -m build .
```

uploading the package to pypi:

```bash
python -m twine upload dist/*
```
