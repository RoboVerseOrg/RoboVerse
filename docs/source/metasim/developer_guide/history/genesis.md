# Genesis Installation

This instruction is tested on Ubuntu 22.04LTS.

Create a new conda environment
```bash
conda create -n genesis python=3.10
conda activate genesis
```

Install Genesis
```bash
pip install genesis-world
```


Install MetaSim dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python "numpy<2" "imageio[ffmpeg]" gymnasium==0.29.0
pip install rootutils loguru rich tyro tqdm huggingface-hub dill
```


Install optional dependencies
```bash
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl # Optional: retargeting, multi-embodiment support
pip install types-usd  # Optional: for OpenUSD development
pip install fake-bpy-module-4.2  # Optional: for Blender code linting
```
