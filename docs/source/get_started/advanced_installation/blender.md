# Blender Installation

```{warning}
Blender is not fully supported by MetaSim yet. We are still actively developing with it.
```

Please refer to the official guide: [Building Blender on Linux](https://developer.blender.org/docs/handbook/building_blender/linux/) and [Building Blender as a Python Module](https://developer.blender.org/docs/handbook/building_blender/python_module/).

Download Blender source code:
```bash
sudo apt install python3 git git-lfs
mkdir ~/Downloads/blender-git
cd blender-git
git clone https://projects.blender.org/blender/blender.git
git checkout v4.2.7
```

Building Blender:
```bash
cd ~/Downloads/blender-git/blender/
./build_files/build_environment/install_linux_packages.py
./build_files/utils/make_update.py --use-linux-libraries
make update
make bpy
```

Building the wheel:
```bash
python3 ./build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin/
```

Create a new conda environment and install the wheel:
```bash
conda create -n blender python=3.11
conda activate blender
cd ~/Downloads/blender-git/build_linux_bpy/bin/
pip install bpy-4.2.7-*.whl
```

Install other dependencies:
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python "numpy<2" "imageio[ffmpeg]" gymnasium==0.29.0
pip install rootutils loguru rich tyro tqdm huggingface-hub
```
