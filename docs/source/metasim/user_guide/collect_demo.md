# Collect Demonstrations
```bash
python metasim/scripts/collect_demo.py --task=CloseBox --num_envs=1 --run_all
```
The collected demos will be saved under `./data_isaaclab/demo/{task}/robot-franka`.

When collecting demo, the RAM will grow up, during which the collected rendered data are gathered before writing to the disk. After a while, the RAM occupation should become steady.

On RTX 4090, the ideal num_envs is 64.


## Domain Randomization
Protocal:
- Level 0: No randomization
- Level 1: Randomize the visual material ("texture") of ground - and table
- Level 2: Randomize camera pose
- Level 3: Randomize object reflection ("material") and lighting

<!-- First, prepare the material dataset [vMaterial 2.4](https://developer.nvidia.com/vmaterials).
```bash
wget https://developer.nvidia.com/downloads/designworks/vmaterials/secure/2.4/nvidia-vmaterials_2-linux-x86-64-2.4.0-373000.4039.run
sudo sh ./nvidia-vmaterials_2-linux-x86-64-2.4.0-373000.4039.run
# The installed path should be /opt/nvidia/mdl/vMaterials_2
# If not, change the path to the actual intallation path on your machine
ln -s /opt/nvidia/mdl/vMaterials_2 third_party
``` -->
