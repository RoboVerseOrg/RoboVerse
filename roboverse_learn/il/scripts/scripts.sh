sudo pkill -9 pt_main_thread

sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia

sudo modprobe nvidia

omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 1
omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 1
omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 1


omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 2
omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 2
omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 2


omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 3
omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 3
omni_python metasim/scripts/collect_demo.py --task=StackGreenCube \
--num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 3

docker run --name rv_6 -it --runtime=nvidia --gpus "device=6" --rm --network=host \
  -v /home/yutong/ws/RoboVerse:/root/Projects/RoboVerse:rw \
  -v /home/yutong/anaconda3/envs:/data/miniconda3/envs:rw \
  -v /mnt/disk1/:/mnt/disk1/:rw \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/carb/logs:/isaac-sim/kit/logs/Kit/Isaac-Sim:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v ~/docker/isaac-lab/docs:/workspace/isaaclab/docs/_build:rw \
  -v ~/docker/isaac-lab/logs:/workspace/isaaclab/logs:rw \
  -v ~/docker/isaac-lab/data:/workspace/isaaclab/data_storage:rw \
  rv:20250121 # your docker name
