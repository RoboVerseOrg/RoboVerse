# PyRep Installation

```{warning}
PyRep is not fully supported by MetaSim yet. We are still actively developing with it.
```

This instruction is tested on Ubuntu 22.04LTS. For other platforms, please refer to the [official guide](https://github.com/stepjam/PyRep?tab=readme-ov-file#install).

Create a new conda environment
```bash
conda create -n pyrep python=3.10
conda activate pyrep
```

Install CoppeliaSim v4.1.0
```bash
cd third_party
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```


Install PyRep
```bash
# utils function to setup pyrep
setup_pyrep(){
   export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
   export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
}

# Install PyRep
cd third_party
git clone https://github.com/stepjam/PyRep.git pyrep && cd pyrep
setup_pyrep
pip install -e .
python examples/example_youbot_navigation.py  # test

# Install RLBench
cd ..
git clone https://github.com/stepjam/RLBench.git rlbench && cd rlbench
pip install -e .
pip install gymnasium
python rlbench/examples/imitation_learning.py  # test
```

Install MetaSim dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install rootutils loguru rich tyro tqdm huggingface-hub
```
