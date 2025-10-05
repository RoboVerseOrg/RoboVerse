# PyRoki Installation

MetaSim uses [PyRoki](https://github.com/chungmin99/pyroki) for modular and scalable robotics kinematics optimization, including inverse kinematics.

```{note}
PyRoki requires Python 3.10 or higher. Python 3.12+ is recommended for best compatibility.
```

## Installation

```bash
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .
```
For Isaacsim, also need the following commands:
```bash
pip install numpy==1.26.0 # For Isaacsim
pip install jax==0.6.0 # For Isaacsim
```




