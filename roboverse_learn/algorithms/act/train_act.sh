# Examples:
# bash roboverse_learn/algorithms/act/train_act.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 2000 ee ee

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

alg_name=ACT
seed=42


extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi
#python roboverse_learn/algorithms/data2zarr_dp.py \
#--task_name ${task_name}_${extra} \
#--expert_data_num ${expert_data_num} \
#--metadata_dir ${metadata_dir} \
#--action_space ${act_space} \
#--observation_space ${obs_space} \
#--delta_ee ${delta_ee}


export CUDA_VISIBLE_DEVICES=${gpu_id}
python -m roboverse_learn.algorithms.act.train \
--task_name ${task_name}_${extra} \
--num_episodes ${expert_data_num} \
--dataset_dir data_policy/${task_name}_${extra}_${expert_data_num}.zarr \
--policy_class ${alg_name} --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs ${num_epochs}  --lr 1e-5 --state_dim 9 \
--seed ${seed}
