#!/bin/bash
# Usage: bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name ddpm_dit --dr_level_eval 2 -- train_enable False

task_name_set="close_box" # Tasks, e.g., close_box, stack_cube, pick_cube
policy_name="ddpm_dit"    # IL policy, opts: ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, vita, act, score
sim_set="isaacsim"          # Simulator, e.g., mujoco, isaacsim
demo_num=100              # Number of demonstrations to collect, train, and eval

# Training/eval control
train_enable=True
eval_enable=True

# Training parameters
num_epochs=100
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300

# Domain Randomization Level
dr_level_collect=0
dr_level_eval=0

# Parse parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_name_set)
            task_name_set="$2"
            shift 2
            ;;
        --policy_name)
            policy_name="$2"
            shift 2
            ;;
        --sim_set)
            sim_set="$2"
            shift 2
            ;;
        --demo_num)
            demo_num="$2"
            shift 2
            ;;
        --train_enable)
            train_enable="$2"
            shift 2
            ;;
        --eval_enable)
            eval_enable="$2"
            shift 2
            ;;
        --dr_level_collect)
            dr_level_collect="$2"
            shift 2
            ;;
        --dr_level_eval)
            dr_level_eval="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Optional parameters: --task_name_set --policy_name --sim_set --demo_num --train_enable --eval_enable --num_epochs --gpu"
            exit 1
            ;;
    esac
done

# Collect demo
echo "=== Running collect_demo.sh ==="
sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^num_demo_success=.*/num_demo_success=$demo_num/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^random_level=.*/random_level=$dr_level_collect/" ./roboverse_learn/il/collect_demo.sh
bash ./roboverse_learn/il/collect_demo.sh

# Map policy_name to model config
config_name="default_runner"
main_script="./roboverse_learn/il/train.py"

# if policy_name is ACT
if [ "${policy_name}" = "act" ]; then
    echo "=== Running ACT training ==="
    sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^train_enable=.*/train_enable=$train_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_enable=.*/eval_enable=$eval_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_level=.*/eval_level=$dr_level_eval/" ./roboverse_learn/il/policies/act/act_run.sh
    bash ./roboverse_learn/il/policies/act/act_run.sh
    echo "=== Completed all data collection, training, and evaluation ==="
    exit 0
fi

# Run training/evaluation for DP/FM/VITA policies
echo "=== Running ${policy_name} ==="

eval_ckpt_name=$demo_num
output_dir="./il_outputs/${policy_name}"
eval_path="${output_dir}/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

export policy_name="${policy_name}"
python ${main_script} --config-name=${config_name}.yaml \
task_name=${task_name_set} \
"dataset_config.zarr_path=./data_policy/${task_name_set}FrankaL${dr_level_collect}_${extra}_${demo_num}.zarr" \
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
eval_config.eval_args.level=${dr_level_eval} \
+eval_config.eval_args.max_demo=${demo_num} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path}

echo "=== Completed all data collection, training, and evaluation ==="
