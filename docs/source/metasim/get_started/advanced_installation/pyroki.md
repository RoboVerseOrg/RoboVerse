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

If you encounter a NumPy version mismatch between lsaacSim 5.0.0 and PyRoki, for example, an error
`TypeError: asarray() got an unexpected keyword argument 'copy'`, try running the following commands:
```bash
pip install numpy==1.26.0 # For Isaacsim
pip install jax==0.4.30 jaxlib==0.4.30
pip install sentry-sdk==1.43.0 typing-extensions==4.12.2 websockets==12.0
pip install --upgrade websockets
```


