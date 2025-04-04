# PyBullet Installation

This instruction is tested on Ubuntu 22.04LTS. For other system versions, please refer to the [official guide](https://github.com/bulletphysics/bullet3?tab=readme-ov-file#pybullet).

Create a new conda environment
```bash
conda create -n pybullet python=3.10
conda activate pybullet
```

Install PyBullet
```bash
pip install pybullet --upgrade --user
pip install setuptools==65.5.0 pip==21 wheel==0.38.0  # see https://stackoverflow.com/a/77205046
pip install gym==0.19.0
pip install numpy==1.26.4
conda install -c conda-forge libstdcxx-ng  # see https://stackoverflow.com/a/71421355
python -m pip install --upgrade pip  # see https://stackoverflow.com/a/58707130
```

Test PyBullet Installation
```bash
python3 -m pybullet_envs.examples.enjoy_TF_AntBulletEnv_v0_2017may
python3 -m pybullet_envs.examples.enjoy_TF_HumanoidFlagrunHarderBulletEnv_v1_2017jul
```

Install MetaSim dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python "numpy<2" "imageio[ffmpeg]" gymnasium==0.29.0
pip install rootutils loguru rich tyro tqdm huggingface-hub dill
```
