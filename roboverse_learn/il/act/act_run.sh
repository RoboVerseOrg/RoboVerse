train_enable=false
eval_enable=true


task_name=close_box
expert_data_num=100
gpu_id=0
sim_set=mujoco

num_epochs=100
obs_space=joint_pos # joint_pos or ee
act_space=joint_pos # joint_pos or ee
delta_ee=0 # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )

alg_name=ACT
seed=42
level=0

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi


if [ "${train_enable}" = "true" ]; then
  echo "=== Training ==="
  export CUDA_VISIBLE_DEVICES=${gpu_id}
  python -m roboverse_learn.il.utils.act.train \
  --task_name ${task_name}_${extra} \
  --num_episodes ${expert_data_num} \
  --dataset_dir data_policy/${task_name}FrankaL${level}_${extra}_${expert_data_num}.zarr \
  --policy_class ${alg_name} --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs ${num_epochs}  --lr 1e-5 --state_dim 9 \
  --seed ${seed}
fi


if [ "${eval_enable}" = "true" ]; then
  echo "=== Evaluation ==="
  # # export TORCH_CUDA_ARCH_LIST="8.9"
  python -m roboverse_learn.il.act.act_eval_runner \
  --task ${task_name} \
  --robot Franka \
  --num_envs 1 \
  --sim ${sim_set} \
  --algo act \
  --ckpt_path /home/jjindou/RoboVerse/info/outputs/ACT/2025.10.09/12.45.48_close_box_obs:joint_pos_act:joint_pos_100 \
  --headless True \
  --num_eval 10
fi
