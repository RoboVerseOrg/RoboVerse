#!/bin/bash
# Usage: bash roboverse_learn/il/il_run.sh --task_name_set close_box --algo_choose ddpm_dit --demo_num 100 --sim_set mujoco

task_name_set="close_box" # Tasks, e.g., close_box, stack_cube, pick_cube
algo_choose="ddpm_dit"    # IL algorithm, opts: ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, vita, act, score
sim_set="mujoco"          # Simulator, e.g., mujoco, isaacsim
demo_num=90              # Number of demonstrations to collect, train, and eval

# Training/eval control
train_enable=True
eval_enable=False

# Training parameters
level=0
num_epochs=100
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300

# Parse parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_name_set)
            task_name_set="$2"
            shift 2
            ;;
        --algo_choose)
            algo_choose="$2"
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
            echo "Optional parameters: --task_name_set --algo_choose --sim_set --demo_num --train_enable --eval_enable --num_epochs --gpu"
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
bash ./roboverse_learn/il/collect_demo.sh

# Map algo_choose to model config
case "$algo_choose" in
    "ddpm_unet")
        algo_model="ddpm_unet_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="DP"
        ;;
    "ddpm_dit")
        algo_model="ddpm_dit_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="DP"
        ;;
    "ddim_unet")
        algo_model="ddim_unet_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="DP"
        ;;
    "fm_unet")
        algo_model="fm_unet_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="FM"
        ;;
    "fm_dit")
        algo_model="fm_dit_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="FM"
        ;;
    "score")
        algo_model="score_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="DP"
        ;;
    "vita")
        algo_model="vita_model"
        config_name="dp_runner"
        main_script="./roboverse_learn/il/train.py"
        output_dir="VITA"
        ;;
    "act")
        echo "=== Running ACT training ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/act/act_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/act/act_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/act/act_run.sh
        bash ./roboverse_learn/il/act/act_run.sh
        echo "=== Completed all data collection, training, and evaluation ==="
        exit 0
        ;;
    *)
        echo "Unsupported algorithm: $algo_choose"
        echo "Available options: act, ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, score, vita"
        exit 1
        ;;
esac

# Run training/evaluation for DP/FM/VITA policies
echo "=== Running ${algo_choose} (${algo_model}) ==="
echo "Selected model: $algo_model"

eval_ckpt_name=$demo_num
eval_path="./info/outputs/${output_dir}/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

export algo_model
python ${main_script} --config-name=${config_name}.yaml \
task_name=${task_name_set} \
"dataset_config.zarr_path=./data_policy/${task_name_set}FrankaL${level}_${extra}_${demo_num}.zarr" \
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
+eval_config.eval_args.max_demo=${demo_num} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path}

echo "=== Completed all data collection, training, and evaluation ==="
