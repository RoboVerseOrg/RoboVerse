# IsaacLab Installation

This instruction is tested on Ubuntu 22.04LTS. For other system versions, please refer to the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).


Clone the repository
```
git clone https://github.com/RoboVerseOrg/RoboVerse
cd RoboVerse
git submodule update --init --recursive
```

Create a new conda environment
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
```

Install Isaac Sim
```bash
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl==4.2.0.2 isaacsim-replicator==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-app==4.2.0.2 --extra-index-url https://pypi.nvidia.com
```

Install Isaac Lab
```bash
cd third_party
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 4d9914702  # Ensure reproducibility
sudo apt install cmake build-essential
./isaaclab.sh --install
ln -s ${CONDA_ROOT}/envs/isaaclab/lib/python3.10/site-packages/isaacsim _isaac_sim  # Optional: for linting
```

- If you encounter error about rsl-rl-lib, try the following commands:



    Replace line 46 of IsaacLab/source/extensions/omni.isaac.lab_tasks/setup.py
    with
    ```python
    "rsl-rl": ["rsl-rl-lib@git+https://github.com/leggedrobotics/rsl_rl.git"],
    ```
    (Just add the -lib in the repository name)

    Then retry the installation.

- If you encounter any version conflicts when running `./isaaclab.sh --install`, try the following commands to fix it:
    ```bash
    cd ../..  # go back to the root directory
    pip install -r requirements.txt --no-deps
    ```

If `pip check` doesn't report any issues, you should be good to go.

Run the Isaac Sim GUI for testing (optional)
```bash
cd third_party/IsaacLab
./isaaclab.sh -s
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
