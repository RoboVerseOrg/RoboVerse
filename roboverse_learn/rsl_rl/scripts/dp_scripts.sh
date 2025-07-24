python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
StackCube_Level0_200_3 200 data_isaaclab/demo/StackCube-Level0/robot-franka 3
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
StackCube_Level1_200_3 200 data_isaaclab/demo/StackCube-Level1/robot-franka 3
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
StackCube_Level2_200_3 200 data_isaaclab/demo/StackCube-Level2/robot-franka 3
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
CloseBox_Level0_90_3 90 data_isaaclab/demo/CloseBox-Level0/robot-franka 3
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
CloseBox_Level1_90_3 90 data_isaaclab/demo/CloseBox-Level1/robot-franka 3
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
CloseBox_Level2_90_3 90 data_isaaclab/demo/CloseBox-Level2/robot-franka 3
python roboverse_learn/algorithms/diffusion_policy/data2zarr_dp.py \
CloseBox_Level3_90_3 90 data_isaaclab/demo/CloseBox-Level3/robot-franka 3



bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox_Level0_90_3 90 0 0
bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox_Level1_90_3 90 0 2
bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox_Level2_90_3 90 0 2
bash roboverse_learn/algorithms/diffusion_policy/train.sh CloseBox_Level3_90_3 90 0 1

bash roboverse_learn/algorithms/diffusion_policy/train.sh StackCube_Level3_200_3 200 0 0
