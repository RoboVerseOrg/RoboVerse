# Examples:
# bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 200 joint_pos joint_pos

# 'metadata_dir' means path to metadata directory. e.g. roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka
# 'task_name' gives a name to the policy, which can include the task robot and level ie CloseBoxFrankaL0
# 'expert_data_num' means number of training data. e.g.100
# 'gpu_id' means single gpu id, e.g.0

metadata_dir=${1}
task_name=${2}
expert_data_num=${3}
gpu_id=${4}
num_epochs=${5}
obs_space=${6} # joint_pos or ee
act_space=${7} # joint_pos or ee
delta_ee=${8:-0} # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )

config_name=robot_dp
horizon=8
n_obs_steps=3
n_action_steps=4
seed=42

# adding the obs and action space as additional info
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name ${task_name}_${extra} \
--expert_data_num ${expert_data_num} \
--metadata_dir ${metadata_dir} \
--action_space ${act_space} \
--observation_space ${obs_space} \
--delta_ee ${delta_ee}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python roboverse_learn/algorithms/diffusion_policy/train.py --config-name=${config_name}.yaml \
task.name=${task_name}_${extra} \
task.dataset.zarr_path="data_policy/${task_name}_${extra}_${expert_data_num}.zarr" \
training.seed=${seed} \
horizon=${horizon} \
n_obs_steps=${n_obs_steps} \
n_action_steps=${n_action_steps} \
training.num_epochs=${num_epochs} \
policy_runner.obs.obs_type=${obs_space} \
policy_runner.action.action_type=${act_space} \
policy_runner.action.delta=${delta_ee} \
training.device="cuda:${gpu_id}"
