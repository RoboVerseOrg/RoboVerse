# OpenUSD Installation

This instruction is tested on Ubuntu 22.04LTS. For more information, please refer to the [official guide](https://github.com/PixarAnimationStudios/OpenUSD).

The following instruction does not have to be executed under the RoboVerse directory.

```bash
git clone https://github.com/PixarAnimationStudios/OpenUSD.git
git checkout v25.02a
mkdir -p ../openusd_build
python build_scripts/build_usd.py ../openusd_build --no-python
```

Note that cmake >= 3.27 is required. If you have not installed it, please refer to [this guide](https://askubuntu.com/a/1157132).
