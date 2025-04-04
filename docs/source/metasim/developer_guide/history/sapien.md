# SAPIEN Installation

This instruction is tested on Ubuntu 22.04LTS. For more information, please refer to the [official guide](https://github.com/haosulab/ManiSkill/tree/v0.5.3?tab=readme-ov-file#installation).

Create a new conda environment
```bash
conda create -n sapien python=3.10
conda activate sapien
```

Install SAPIEN2
```bash
cd third_party
git clone https://github.com/haosulab/ManiSkill.git  # It is safe to install ManiSkill instead of directly installing SAPIEN
cd ManiSkill
git checkout v0.5.3  # use ManiSkill2
pip install -e .
```

Test Installation
```bash
python -m mani_skill2.examples.demo_random_action
```

Install MetaSim dependencies
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python "numpy<2" "imageio[ffmpeg]" gymnasium==0.29.0
pip install rootutils loguru rich tyro tqdm huggingface-hub dill
```
