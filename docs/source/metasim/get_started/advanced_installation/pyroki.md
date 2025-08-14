# PyRoki Installation

[Pyroki](https://github.com/chungmin99/pyroki) is a differentiable kinematics library that provides FK, IK, collision detection, and more for robotics research and development.

<!-- ---

## Warning

PyRoki requires **CUDA Toolkit version **.
Please make sure your system meets this requirement before installation.

--- -->

## Note

PyRoki installation is tested on **Ubuntu 22.04 LTS** and **Ubuntu 24.04 LTS**, with Python versions **3.10**.
PyRoki can also be used inside a Docker container with GPU support for a consistent and isolated environment.

For other operating systems or Python versions, please refer to the official [Pyroki](https://github.com/chungmin99/pyroki) documentation.

---

## Installation

Please follow the steps below to install PyRoki:

```bash
cd third_party
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
````
*Note:* The installation process may take around 2 minutes.

If you encounter errors related to graphics rendering or compatibility issues with PyOpenGL, please try downgrading PyOpenGL to a compatible version by running:

```bash
pip install PyOpenGL==3.1.0
```
---
**Limitations**

- PyRoki currently does **not support parallel computation** and relies on sequential looping for calculations, which results in higher computational cost and slower performance compared to CuRobo, especially in complex or large-scale scenarios.
