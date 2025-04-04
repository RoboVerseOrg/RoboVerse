# IsaacGym Installation

This instruction is tested on Ubuntu 22.04LTS.

Create a new conda environment
```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
```

Install IsaacGym
```bash
# Download IsaacGym if you haven't
cd third_party/
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xf isaac-gym-preview-4
rm isaac-gym-preview-4
cd isaacgym/python

# Install IsaacGym
pip install -e .

# test
cd examples
python asset_info.py
# you may meet some errors like module 'numpy' has no attribute 'float', change np.float to np.float32 in the source code of isaacgym
cd ../../../../
```

Install MetaSim dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python "numpy<2" "imageio[ffmpeg]" gymnasium==0.29.0
pip install rootutils loguru rich tyro tqdm huggingface-hub dill
```

Install optional dependencies
```bash
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl # Optional: retargeting, multi-embodiment support
```
