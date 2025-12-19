## Separate script for training and evaluation
train_enable=True
eval_enable=True


## Parameters
task_name_set=close_box
expert_data_num=100
gpu_id=0
sim_set=isaacsim
num_epochs=100

obs_space=joint_pos # joint_pos or ee
act_space=joint_pos # joint_pos or ee
delta_ee=0 # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control)

alg_name=ACT
seed=42
collect_level=3

# ACT hyperparameters
chunk_size=40
kl_weight=10
hidden_dim=512
lr=1e-5
batch_size=32
dim_feedforward=3200

# Domain Randomization parameters for evaluation
eval_level=3
eval_scene_mode=0     # 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD
eval_seed=42          # Randomization seed (optional)

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

# Training
if [ "${train_enable}" = "True" ]; then
  echo "=== Training ==="
  export CUDA_VISIBLE_DEVICES=${gpu_id}
  python -m roboverse_learn.il.policies.act.train \
  --task_name ${task_name_set} \
  --num_episodes ${expert_data_num} \
  --dataset_dir data_policy/${task_name_set}FrankaL${collect_level}_${extra}_${expert_data_num}.zarr \
  --policy_class ${alg_name} --kl_weight ${kl_weight} --chunk_size ${chunk_size} \
  --hidden_dim ${hidden_dim} --batch_size ${batch_size} --dim_feedforward ${dim_feedforward} \
  --num_epochs ${num_epochs}  --lr ${lr} --state_dim 9 \
  --seed ${seed} \
  --level ${collect_level}
fi

# Evaluation
if [ "${eval_enable}" = "True" ]; then
  echo "=== Evaluation ==="
  # # export TORCH_CUDA_ARCH_LIST="8.9"
  ckpt_path=$(cat ./roboverse_learn/il/policies/act/ckpt_dir_path.txt)


  python -m roboverse_learn.il.policies.act.act_eval_runner \
  --task ${task_name_set} \
  --robot franka \
  --num_envs 1 \
  --sim ${sim_set} \
  --algo act \
  --ckpt_path  ./${ckpt_path} \
  --headless True \
  --num_eval ${expert_data_num} \
  --temporal_agg True \
  --chunk_size ${chunk_size} \
  --level ${eval_level} \
  --scene_mode ${eval_scene_mode} \
  --randomization_seed ${eval_seed}
fi
