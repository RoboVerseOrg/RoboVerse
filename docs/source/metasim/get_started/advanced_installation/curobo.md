# cuRobo Installation

MetaSim uses [cuRobo](https://github.com/NVlabs/curobo) to perform IK, motion planning, and retargeting of trajectories across different embodiments.

```{note}
cuRobo installation is tested on Ubuntu 22.04LTS, with both Python 3.8 and 3.10. For other platforms, please refer to the [official guide](https://curobo.org/get_started/1_install_instructions.html).
```

## Installation

Please refer to the [official guide](https://curobo.org/get_started/1_install_instructions.html#library-installation).
```bash
sudo apt install git-lfs
git submodule update --init --recursive
cd third_party/curobo
uv pip install -e . --no-build-isolation
```

This may take ~5 minutes.

## Troubleshooting
- If you encounter `cc1plus: warning: command-line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++` when building cuRobo, try adding the following lines at the top of `third_party/curobo/setup.py`:
    ```python
    import distutils.sysconfig
    cfg_vars = distutils.sysconfig.get_config_vars()
    for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")
    ```
    For more details, please refer to this [answer](https://stackoverflow.com/a/29634231).
- If you encounter `error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.` when building cuRobo, try installing gcc-10 and g++-10 on your machine:
    ```bash
    sudo apt install gcc-10 g++-10
    ```
    and switch to gcc-10 and g++-10 as the default compiler:
    ```bash
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
    sudo update-alternatives --config gcc
    sudo update-alternatives --config g++
    ```
    Note that gcc and g++ must be both installed and switched to the same version.
