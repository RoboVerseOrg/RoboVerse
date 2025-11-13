#!/bin/bash
# Try： bash roboverse_learn/il/il_run.sh --task_name_set close_box --algo_choose dp_DDPM --demo_num 100 --sim_set mujoco

task_name_set="close_box" # Tasks, opts: close_box, stack_cube pick_cube
algo_choose="dp_DDPM"     # IL algorithm, opts: act, dp_DDPM, dp_DDIM, dp_FM_UNet, dp_FM_DiT, dp_Score, dp_VITA
sim_set="mujoco"          # Simulator, opts: mujoco, isaacsim
demo_num=100              # Number of demonstration to collect, train, and eval

# parse parameters
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
        *)
            echo "Unknown parameter: $1，optional parameter：--task_name_set --algo_choose --sim_set --demo_num"
            exit 1
            ;;
    esac
done

# 1. collect_demo
echo "=== Running collect_demo.sh ==="
sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^max_demo_idx=.*/max_demo_idx=$demo_num/" ./roboverse_learn/il/collect_demo.sh
sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/collect_demo.sh
bash ./roboverse_learn/il/collect_demo.sh

# 2. il algorithm
case "$algo_choose" in
    "dp_DDPM")
        echo "=== Running dp_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^algo_choose=.*/algo_choose=0/" ./roboverse_learn/il/dp/dp_run.sh
        bash ./roboverse_learn/il/dp/dp_run.sh
        ;;
    "dp_DDIM")
        echo "=== Running dp_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^algo_choose=.*/algo_choose=1/" ./roboverse_learn/il/dp/dp_run.sh
        bash ./roboverse_learn/il/dp/dp_run.sh
        ;;
    "dp_FM_UNet")
        echo "=== Running dp_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^algo_choose=.*/algo_choose=2/" ./roboverse_learn/il/dp/dp_run.sh
        bash ./roboverse_learn/il/dp/dp_run.sh
        ;;
    "dp_FM_DiT")
        echo "=== Running dp_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^algo_choose=.*/algo_choose=3/" ./roboverse_learn/il/dp/dp_run.sh
        bash ./roboverse_learn/il/dp/dp_run.sh
        ;;
    "dp_Score")
        echo "=== Running dp_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^algo_choose=.*/algo_choose=4/" ./roboverse_learn/il/dp/dp_run.sh
        bash ./roboverse_learn/il/dp/dp_run.sh
        ;;
    "dp_VITA")
        echo "=== Running dp_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$demo_num/" ./roboverse_learn/il/dp/dp_run.sh
        sed -i "s/^algo_choose=.*/algo_choose=5/" ./roboverse_learn/il/dp/dp_run.sh
        bash ./roboverse_learn/il/dp/dp_run.sh
        ;;
    "act")
        echo "=== Running act_run.sh ==="
        sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/act/act_run.sh
        sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/act/act_run.sh
        sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/act/act_run.sh
        bash ./roboverse_learn/il/act/act_run.sh
        ;;
    *)
        echo "Unavailable chose: $algo_choose, optional options: act, dp_DDPM, dp_DDIM, dp_FM_UNet, dp_FM_DiT, dp_Score, dp_VITA"
        exit 1
        ;;
esac

echo "=== Completed all data collection, training, and evaluation ==="
