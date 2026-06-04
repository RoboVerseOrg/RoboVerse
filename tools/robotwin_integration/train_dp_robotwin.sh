#!/bin/bash
# Train a RoboVerse Diffusion Policy on RoboTwin demos (policy-reproduction track).
#
# Chains the bridge -> RoboVerse IL pipeline end to end, all in the roboverse env:
#   1. robotwin_to_demo.py : bridge pkls (collect_bridge --rgb) -> demo_<id>/ dirs
#   2. data2zarr_dp.py     : demo dirs -> zarr (UNMODIFIED RoboVerse tool)
#   3. train.py            : DP training, with the bimanual 14-D / RoboTwin-image
#                            shape overrides (the default config is single-arm 9-D).
#
# Eval is intentionally OFF here: a RoboTwin policy is evaluated closed-loop in the
# native RoboTwin env via tools/robotwin_integration/eval_robotwin_policy.py
# (--policy dp --ckpt <ckpt>), not in a RoboVerse handler env.
#
# Usage:
#   bash tools/robotwin_integration/train_dp_robotwin.sh \
#       --task beat_block_hammer \
#       --bridge-dir ~/projects/robotwin/data/_rv_bridge/bbh_train50 \
#       --num 50 --epochs 300 --policy ddpm_unet
set -euo pipefail

task="beat_block_hammer"
bridge_dir=""
num=50
epochs=300
policy="ddpm_unet"   # ResNet+UNet image encoder: robust to the 240x320 RoboTwin frame
camera="head_camera"
gpu=0
# Match RoboTwin's OWN DP spec (policy/DP/diffusion_policy/config/robot_dp_14.yaml):
# horizon=8, n_obs_steps=3, n_action_steps=6. roboverse_learn's default_runner.yaml
# ships n_action_steps=4, which deviates -> under-reproduces RoboTwin on precise tasks
# (e.g. place_empty_cup). Default to RoboTwin's value so the pipeline matches the
# published DP numbers out of the box; override per-task if needed.
n_action_steps=6
n_obs_steps=3
horizon=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="$2"; shift 2;;
    --bridge-dir) bridge_dir="$2"; shift 2;;
    --num) num="$2"; shift 2;;
    --epochs) epochs="$2"; shift 2;;
    --n-action-steps) n_action_steps="$2"; shift 2;;
    --policy) policy="$2"; shift 2;;
    --camera) camera="$2"; shift 2;;
    --gpu) gpu="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[[ -n "$bridge_dir" ]] || { echo "ERROR: --bridge-dir required"; exit 1; }

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
demo_dir="data/robotwin_demo/${task}"
zarr_path="data_policy/${task}_${num}.zarr"

# This stage runs in the RoboVerse IL env (zarr + torch + roboverse_learn), NOT the robotwin
# collection env. Fail fast with an actionable message if `python` is the wrong interpreter --
# otherwise data2zarr_dp.py dies with a bare `ModuleNotFoundError: zarr` mid-pipeline.
if ! python -c "import zarr, torch" 2>/dev/null; then
  echo "ERROR: the active 'python' lacks zarr/torch ($(command -v python))." >&2
  echo "       Run this in the RoboVerse IL env, e.g.: conda activate roboverse  (or set PATH to its bin)." >&2
  exit 1
fi

echo "=== [1/3] bridge -> RoboVerse demo dirs ==="
python tools/robotwin_integration/robotwin_to_demo.py \
  --bridge "${bridge_dir}/demo_*.pkl" --out "$demo_dir" --camera "$camera"

echo "=== [2/3] demo dirs -> zarr (unmodified data2zarr_dp.py) ==="
python roboverse_learn/il/data2zarr_dp.py \
  --task_name "$task" --expert_data_num "$num" --metadata_dir "$demo_dir"

echo "=== [3/3] train DP (${policy}, bimanual 14-D, ${epochs} epochs) ==="
# The default shape_meta is single-arm Franka (agent_pos/action=9, image 256x256);
# RoboTwin is bimanual (14-D) with a 240x320 head camera -> override all three.
# Run via `python -m` with the worktree on PYTHONPATH so `roboverse_learn` resolves
# to THIS checkout (the editable install otherwise points it at the main checkout,
# bypassing the worktree's numba-optional fix). Do NOT override batch_size: the
# dataset's own batch_size (dataset_config) must match the dataloader's, and both
# default to 32 -- overriding only one trips an assert in the batched __getitem__.
PYTHONPATH="$repo_root" policy_name="$policy" python -m roboverse_learn.il.train \
  --config-name=default_runner.yaml \
  task_name="$task" \
  "dataset_config.zarr_path=${zarr_path}" \
  "image_shape=[3,240,320]" \
  "shape_meta.obs.head_cam.shape=[3,240,320]" \
  "shape_meta.obs.agent_pos.shape=[14]" \
  "shape_meta.action.shape=[14]" \
  train_config.training_params.num_epochs="$epochs" \
  n_action_steps="$n_action_steps" \
  n_obs_steps="$n_obs_steps" \
  horizon="$horizon" \
  "train_config.training_params.device=cuda:${gpu}" \
  logging.mode=offline \
  train_enable=True eval_enable=False

echo "=== DONE. checkpoints under il_outputs/${policy}/${task}/ ==="
echo "Eval closed-loop in native RoboTwin:"
echo "  conda run -n robotwin env MUJOCO_GL=egl python \\"
echo "    tools/robotwin_integration/eval_robotwin_policy.py --task ${task} \\"
echo "    --policy dp --ckpt il_outputs/${policy}/${task}/checkpoints/<ckpt>"
