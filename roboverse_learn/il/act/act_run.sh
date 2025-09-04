# Examples:
# bash roboverse_learn/il/utils/act/train_act.sh roboverse_demo/demo_mujoco/close_box-/robot-franka close_box 100 0 2000 ee ee

# 'metadata_dir' means path to metadata directory. e.g. roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka
# 'task_name' gives a name to the policy, which can include the task robot and level ie CloseBoxFrankaL0
# 'expert_data_num' means number of training data. e.g.100
# 'gpu_id' means single gpu id, e.g.0


train_enable=true
eval_enable=false


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
level=2

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
  --ckpt_path ~/RoboVerse/info/outputs/ACT/2025.09.04/01.37.14_close_box_obs:joint_pos_act:joint_pos_100 \
  --headless True
fi
