
## Seperate training and evaluation
train_enable=False  # True for training, False for evaluation
eval_enable=True

task_name_set=track_il
config_name=default_runner
num_epochs=200
port=50010
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300
expert_data_num=100
sim_set=isaacsim
eval_ckpt_name=200           # Evaluate the last checkpoint (epoch 3)

# Weights & Biases logging
# - online: upload logs to your W&B account
# - offline: save locally, no upload
# - disabled: no wandb
wandb_mode="${wandb_mode:-online}"

if [ "${wandb_mode}" = "online" ]; then
  # Prefer non-interactive login via env var (recommended on servers)
  if [ -n "${WANDB_API_KEY:-}" ]; then
    python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("wandb login ok")
PY
  else
    echo "[WARN] wandb_mode=online 但未检测到 WANDB_API_KEY。若你尚未登录，请先运行：wandb login"
  fi
fi

## Domain Randomization Configuration
level=3              # 0=None, 1=Scene+Material, 2=+Light, 3=+Camera
scene_mode=0         # 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD
dr_seed=42          # Random seed for reproducible DR (null for random)


## Choose training or inference algorithm
# Supported models:
#   "ddpm_unet", "ddpm_dit", "ddim_unet", "vita", "fm_dit", "fm_unet", "score"
policy_name="ddpm_dit"
# 注意：Hydra 配置组 `policy_config` 的可选项是 `ddpm_dit/ddpm_unet/...`
# default_runner.yaml 中使用 `${oc.env:policy_name,ddpm_dit}` 选择 policy_config，
# 因此这里的环境变量必须是 *不带* `_model` 后缀的名字。
export policy_name
eval_path="./il_outputs/${policy_name}/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Selected model: $policy_name"
echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

# Note: level variable is now used for DR, not in zarr filename
# The zarr filename should use the data collection level (e.g., L3)
data_level=3  # Level used when collecting data (matches the converted zarr file)
zarr_path="./data_policy/${task_name_set}FrankaL${data_level}_${extra}_${expert_data_num}.zarr"

python ./roboverse_learn/il/train.py --config-name=${config_name}.yaml \
task_name=${task_name_set} \
dataset_config.zarr_path="${zarr_path}" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${gpu} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name_set} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.sim=${sim_set} \
++eval_config.eval_args.max_demo=${expert_data_num} \
++eval_config.eval_args.level=${level} \
++eval_config.eval_args.scene_mode=${scene_mode} \
++eval_config.eval_args.randomization_seed=${dr_seed} \
logging.mode=${wandb_mode} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path}
