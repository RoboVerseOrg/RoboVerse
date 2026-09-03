# MJX Installation

```{warning}
MJX is not fully supported by MetaSim yet. We are still actively developing with it.
```

Install MJX:
```bash
pip install -e ".[mjx]"
```

The following package versions are tested on Python 3.10:
```bash
mujoco                     3.2.7
mujoco-mjx                 3.2.7
jax                        0.6.0
jax-cuda12-pjrt            0.6.0
jax-cuda12-plugin          0.6.0
jaxlib                     0.6.0
dm_control                 1.0.27
```

## Troubleshooting

If you encounter cuDNN-related issues, try the following solution:

1. Install cuDNN 9.1 and CUDA 12 in your conda environment
2. Add the following line before running:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```