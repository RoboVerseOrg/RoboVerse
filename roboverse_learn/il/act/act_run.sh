# Separate script for training and evaluation
train_enable=true
eval_enable=true


task_name=close_box  # close_box pick_butter
expert_data_num=100
expert_data_num_real=100

gpu_id=0
sim_set=mujoco
num_epochs=100

obs_space=joint_pos # joint_pos or ee
act_space=joint_pos # joint_pos or ee
delta_ee=0 # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )

alg_name=ACT
seed=42
level=0

# ACT hyperparameters
chunk_size=120
kl_weight=10
hidden_dim=512
lr=1e-5
batch_size=8
dim_feedforward=3200

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

# Loop over different chunk sizes
start_chunk=10
step_chunk=20

for ((i=0; i<1; i++)); do

  chunk_size=$((start_chunk + i * step_chunk))
  echo "=== Current chunk_size: $chunk_size ==="

  # Training
  if [ "${train_enable}" = "true" ]; then
    echo "=== Training ==="
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    python -m roboverse_learn.il.utils.act.train \
    --task_name ${task_name}_${extra}_chunk${chunk_size} \
    --num_episodes ${expert_data_num_real} \
    --dataset_dir data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr \
    --policy_class ${alg_name} --kl_weight ${kl_weight} --chunk_size ${chunk_size} \
    --hidden_dim ${hidden_dim} --batch_size ${batch_size} --dim_feedforward ${dim_feedforward} \
    --num_epochs ${num_epochs}  --lr ${lr} --state_dim 9 \
    --seed ${seed}
  fi

  # Evaluation
  if [ "${eval_enable}" = "true" ]; then
    echo "=== Evaluation ==="
    # # export TORCH_CUDA_ARCH_LIST="8.9"
    ckpt_path=$(cat ./roboverse_learn/il/act/ckpt_dir_path.txt)
    python -m roboverse_learn.il.act.act_eval_runner \
    --task ${task_name} \
    --robot franka \
    --num_envs 1 \
    --sim ${sim_set} \
    --algo act \
    --ckpt_path  /home/jjindou/RoboVerse/${ckpt_path} \
    --headless True \
    --num_eval 100 \
    --temporal_agg True \
    --chunk_size ${chunk_size}
  fi

done
