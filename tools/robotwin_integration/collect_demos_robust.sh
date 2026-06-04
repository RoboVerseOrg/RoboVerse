#!/bin/bash
# Hang-resistant RGB demo collection for IL training.
#
# Headless RT + OIDN can intermittently deadlock SAPIEN on a long episode, hanging
# a long in-process multi-demo collect_bridge run (observed: stalls after a handful
# of demos even with the denoiser disabled). This wrapper runs ONE seed per
# subprocess under `timeout`, so a hang kills only that seed and the sweep
# continues -- trading a little startup overhead for robustness at training scale.
#
# Usage:
#   bash tools/robotwin_integration/collect_demos_robust.sh \
#       --task beat_block_hammer --out-dir ~/.../bbh_train50 \
#       --want 50 --max-seed 200 --camera head_camera --per-demo-timeout 240
set -uo pipefail

task="beat_block_hammer"
out_dir=""
want=50
start_seed=0
max_seed=200
camera="head_camera"
timeout_s=240

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="$2"; shift 2;;
    --out-dir) out_dir="$2"; shift 2;;
    --want) want="$2"; shift 2;;
    --start-seed) start_seed="$2"; shift 2;;
    --max-seed) max_seed="$2"; shift 2;;
    --camera) camera="$2"; shift 2;;
    --per-demo-timeout) timeout_s="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[[ -n "$out_dir" ]] || { echo "ERROR: --out-dir required"; exit 1; }

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
mkdir -p "$out_dir"
PY="${ROBOTWIN_PY:-$HOME/miniconda3/envs/robotwin/bin/python}"

# Count demos already present (resume-friendly): find the next free index.
have=$(ls "$out_dir"/demo_*.pkl 2>/dev/null | wc -l)
echo "starting with $have/$want demos already in $out_dir"

seed=$start_seed
while [[ $have -lt $want && $seed -lt $max_seed ]]; do
  idx=$(printf "%04d" "$have")
  tmp="$out_dir/_try_${idx}.pkl"
  rm -f "$tmp"
  echo "[seed $seed] collecting -> demo_${idx} (timeout ${timeout_s}s) ..."
  # Single-demo mode tries exactly this seed (--max-seeds 1) and writes --out verbatim.
  if timeout "${timeout_s}" env MUJOCO_GL=egl SAPIEN_HEADLESS=1 PYTHONUNBUFFERED=1 \
      "$PY" tools/robotwin_integration/collect_bridge.py \
      --task "$task" --start-seed "$seed" --max-seeds 1 --num-demos 1 \
      --rgb --cameras "$camera" --out "$tmp" \
      > "$out_dir/_log_seed${seed}.txt" 2>&1; then
    if [[ -f "$tmp" ]]; then
      mv "$tmp" "$out_dir/demo_${idx}.pkl"
      have=$((have + 1))
      echo "  -> SAVED demo_${idx} (now $have/$want)"
    else
      echo "  -> seed $seed: no success (plan/check failed), skipping"
    fi
  else
    echo "  -> seed $seed: TIMEOUT/error after ${timeout_s}s, skipping"
    rm -f "$tmp"
  fi
  seed=$((seed + 1))
done

echo "DONE: $have/$want demos in $out_dir (last seed tried: $((seed - 1)))"
