import subprocess
import time

experiments = [
    # "/isaac-sim/python.sh  metasim/scripts/collect_demo.py --num_envs=1 --task=square_d0 --robot=franka_stable \
    #     --random.level=0 --all-rerender --no-table --headless",
    # "/isaac-sim/python.sh  metasim/scripts/collect_demo.py --num_envs=1 --task=square_d0 --robot=franka_stable \
    #     --random.level=1 --all-rerender --no-table --headless",
    # "/isaac-sim/python.sh  metasim/scripts/collect_demo.py --num_envs=1 --task=square_d0 --robot=franka_stable \
    #     --random.level=2 --all-rerender --no-table --headless",
    # "/isaac-sim/python.sh  metasim/scripts/collect_demo.py --num_envs=1 --task=square_d0 --robot=franka_stable \
    #     --random.level=3 --all-rerender --no-table --headless",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/SquareD0-Level0 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/SquareD0-Level1 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/SquareD0-Level2 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/SquareD0-Level3 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/StackCube-Level0 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/StackCube-Level1 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/StackCube-Level2 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/StackCube-Level3 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/LiberoPickChocolatePudding-Level0 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/LiberoPickChocolatePudding-Level1 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/LiberoPickChocolatePudding-Level2 .",
    # "rsync -rvp ji_10:/home/ghr/Projects/RoboVerse/data_isaaclab/demo/LiberoPickChocolatePudding-Level3 .",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 3000 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 3000 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 3000 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 3000 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 400 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 400 --max_demo_idx 700 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackGreenCube \
    # --num_envs=1 --headless --demo_start_idx 700 --max_demo_idx 1000 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube \
    # --num_envs=1 --headless --demo_start_idx 800 --max_demo_idx 1000 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube \
    # --num_envs=1 --headless --demo_start_idx 800 --max_demo_idx 1000 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube \
    # --num_envs=1 --headless --demo_start_idx 800 --max_demo_idx 1000 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube \
    # --num_envs=1 --headless --demo_start_idx 800 --max_demo_idx 1000 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=PickCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=PickCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=PickCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=PickCube \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox-Level0_90_3 90 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox-Level1_90_3 90 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox-Level2_90_3 90 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox-Level3_90_3 90 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh MoveSliderLeftA-Level0_153_1 153 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh MoveSliderLeftA-Level1_153_1 153 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh MoveSliderLeftA-Level2_153_1 153 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh MoveSliderLeftA-Level3_153_1 153 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh StackGreenCube-Level3_900_3 900 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh StackGreenCube-Level2_900_3 900 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh StackGreenCube-Level1_900_3 900 0 0",
    # "bash roboverse_learn/algorithms/diffusion_policy/train.sh StackGreenCube-Level0_900_3 900 0 0",
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1 --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 0 --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=SquareD0 \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 0 --robot franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=SquareD0 \
    # --num_envs=1 --headless --demo_start_idx 613 --max_demo_idx 1000 --random.level 1 --robot franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=SquareD0 \
    # --num_envs=1 --headless --demo_start_idx 790 --max_demo_idx 1000 --random.level 2 --robot franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=SquareD0 \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 --robot franka_stable",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackGreenCube --headless \
    # --random.level 1 --checkpoint_path /home/ghr/Projects/RoboVerse/outputs/SGC_900_3_1000_l1.ckpt",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task MoveSliderLeftA --headless --random.level 0 --checkpoint_path outputs/MSLA_153_2_1000_l0.ckpt --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task MoveSliderLeftA --headless --random.level 1 --checkpoint_path outputs/MSLA_153_2_1000_l1.ckpt --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task MoveSliderLeftA --headless --random.level 2 --checkpoint_path outputs/MSLA_153_2_1000_l2.ckpt --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task MoveSliderLeftA --headless --random.level 3 --checkpoint_path outputs/MSLA_153_2_1000_l3.ckpt --robot franka_with_gripper_extension",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task CloseBox --headless --random.level 0 --checkpoint_path outputs/CB_900_3_1000_l0.ckpt --action_set_steps 0",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task CloseBox --headless --random.level 1 --checkpoint_path outputs/CB_900_3_1000_l1.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task CloseBox --headless --random.level 1 --checkpoint_path outputs/CB_900_3_1000_l1.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task CloseBox --headless --random.level 3 --checkpoint_path outputs/CB_900_3_1000_l3.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackCube --headless --random.level 2 --checkpoint_path outputs/SC_900_3_1000_l2.ckpt --action_set_steps 1",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task PickCube --headless --random.level 1 --checkpoint_path outputs/PC_900_2_1000_l1.ckpt --action_set_steps 1",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task PickCube --headless --random.level 2 --checkpoint_path outputs/PC_900_2_1000_l2.ckpt --action_set_steps 1",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task PickCube --headless --random.level 3 --checkpoint_path outputs/PC_900_2_1000_l3.ckpt --action_set_steps 1 --robot=franka_stable",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task CloseBox --headless --random.level 2 --checkpoint_path outputs/CB_90_3_1000_l2.ckpt --action_set_steps 1",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task CloseBox --headless --random.level 2 --checkpoint_path outputs/CB_90_3_1000_l2.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task square_d0 --headless --random.level 0 --checkpoint_path outputs/SD_900_2_700_l0.ckpt --action_set_steps 1 --robot=franka_stable"
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackCube --headless --random.level 0 --checkpoint_path outputs/SC_900_3_1000_l0.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackCube --headless --random.level 1 --checkpoint_path outputs/SC_900_3_1000_l1.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackCube --headless --random.level 2 --checkpoint_path outputs/SC_900_3_1000_l2.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackCube --headless --random.level 3 --checkpoint_path outputs/SC_900_3_1000_l3.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task PickCube --headless --random.level 3 --checkpoint_path outputs/SC_900_3_1000_l3_resnet50_no_crop.ckpt --action_set_steps 1",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task StackCube --headless --random.level 3 --checkpoint_path outputs/SC_900_3_1000_l3_resnet50_no_crop.ckpt --action_set_steps 3",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task LiberoPickChocolatePudding --headless --random.level 2 --checkpoint_path outputs/LPCP_1800_2_1000_l2.ckpt --action_set_steps 1 --robot=franka_stable",
    # "/isaac-sim/python.sh roboverse_learn/eval.py --task LiberoPickChocolatePudding --headless --random.level 3 --checkpoint_path outputs/LPCP_1800_2_1000_l3.ckpt --action_set_steps 1 --robot=franka_stable",
    #  docker run --name rv_3_2 -it --runtime=nvidia --gpus "device=3" --rm --network=host \
    # docker run --name rv_7_1 -it --runtime=nvidia --gpus "device=7" --rm --network=host \
    #   -v /home/ghr/Projects/RoboVerse:/root/Projects/RoboVerse:rw \
    #   -v /home/ghr/anaconda3/envs:/data/miniconda3/envs:rw \
    #   -v /testdata:/testdata:rw -v /home:/home_:rw \
    #   -v /scratch/partial_datasets/droid_video:/scratch/partial_datasets/droid_video:rw \
    #   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    #   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    #   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    #   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    #   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    #   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    #   -v ~/docker/isaac-sim/carb/logs:/isaac-sim/kit/logs/Kit/Isaac-Sim:rw \
    #   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    #   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    #   -v ~/docker/isaac-lab/docs:/workspace/isaaclab/docs/_build:rw \
    #   -v ~/docker/isaac-lab/logs:/workspace/isaaclab/logs:rw \
    #   -v ~/docker/isaac-lab/data:/workspace/isaaclab/data_storage:rw \
    #   rv:20250121
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 20 --random.level 0 \
    # --render.mode pathtracing --cust_name demo_path_tracing",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 20 --random.level 1  \
    # --render.mode pathtracing --cust_name demo_path_tracing",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 20 --max_demo_idx 40 --random.level 0 \
    # --render.mode pathtracing --cust_name demo_path_tracing_512",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 20 --max_demo_idx 40 --random.level 1 \
    # --render.mode pathtracing --cust_name demo_path_tracing_512",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 20 --random.level 3 \
    # --render.mode pathtracing --cust_name demo_path_tracing",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 1 \
    # --cust_name ray_tracing_real_sense1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 1 \
    # --cust_name ray_tracing_real_sense2",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 1 \
    # --cust_name ray_tracing_real_sense3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.level 1 \
    # --render.mode pathtracing  --cust_name path_tracing_real_sense1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 20 --random.level 2 \
    #  --cust_name demo",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.light --random.ground --random.reflection --random.table --random.wall \
    # --cust_name raytracing_real_sense1_randomcolor",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 100 --random.light --random.ground --random.reflection --random.table --random.wall \
    # --render.mode pathtracing  --cust_name pathtracing_real_sense1_randomcolor",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 10 --random.level 3 \
    # --render.mode pathtracing  --cust_name pathtracing_real_sense1_randomcolor  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 10 --random.level 3 \
    # --render.mode pathtracing  --cust_name pathtracing_real_sense1_randomcolor --robot franka_with_gripper_extension",
    # python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py CloseBox-Level-all_450_1 450 \
    # data_isaaclab/demo/CloseBox-Level-all/robot-franka 1
    #############################
    # docker run --name rv_6 -it --runtime=nvidia --gpus "device=6" --rm --network=host \
    #   -v /home/ghr/Projects/RoboVerse:/root/Projects/RoboVerse:rw \
    #   -v /home/ghr/anaconda3/envs:/data/miniconda3/envs:rw \
    #   -v /testdata:/testdata:rw \
    #   -v /scratch/partial_datasets/roboverse:/scratch/partial_datasets/roboverse:rw \
    #   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    #   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    #   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    #   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    #   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    #   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    #   -v ~/docker/isaac-sim/carb/logs:/isaac-sim/kit/logs/Kit/Isaac-Sim:rw \
    #   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    #   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    #   -v ~/docker/isaac-lab/docs:/workspace/isaaclab/docs/_build:rw \
    #   -v ~/docker/isaac-lab/logs:/workspace/isaaclab/logs:rw \
    #   -v ~/docker/isaac-lab/data:/workspace/isaaclab/data_storage:rw \
    #   rv:20250121
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_0 --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_1 --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_1 --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_2 --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=CloseBox  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_3 --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_2  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_3  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_0  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 1 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_1  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_0",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_1",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.scene \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_scene_0 --robot=franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.scene \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_scene_1 --robot=franka_with_gripper_extension",
    "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
--num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.scene \
--render.mode pathtracing  --cust_name new_render_256_pathtracing_scene_0 ",
    "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=LiberoPickChocolatePudding  \
--num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.scene \
--render.mode pathtracing  --cust_name new_render_256_pathtracing_scene_1 ",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_3",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=PickCube  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_2  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=PickCube  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_3  --robot=franka_stable",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_1  --robot=franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_1  --robot=franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=MoveSliderLeftA  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 2 \
    # --render.mode pathtracing  --cust_name new_render_256_pathtracing_0  --robot=franka_with_gripper_extension",
    # "/isaac-sim/python.sh metasim/scripts/collect_demo.py --task=StackCube  \
    # --num_envs=1 --headless --demo_start_idx 0 --max_demo_idx 1000 --random.level 3 \
    # --cust_name new_render_256_raytracing_scene --random.scene",
]


def run_experiment(experiment):
    command = experiment
    subprocess.Popen(command, shell=True)


def schedule_experiments():
    for experiment in experiments:
        run_experiment(experiment)
        time.sleep(10)


if __name__ == "__main__":
    schedule_experiments()
