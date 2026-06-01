#!/bin/bash
# One-command closed-loop DP eval in native RoboTwin (the RoboTwin "eval.sh" experience).
#
# Hides the two-env server/client split: starts dp_policy_server.py in the roboverse
# env (loads the ckpt + serves inference), waits for it to listen, runs
# eval_robotwin_policy.py --policy dp in the robotwin env for N seeds, prints the
# success rate, and tears the server down. A RoboVerse user just points at a
# checkpoint and a task and gets a success rate -- exactly like running RoboTwin's
# own policy eval.
#
# Usage:
#   bash tools/robotwin_integration/eval_dp_robotwin.sh \
#       --task beat_block_hammer \
#       --ckpt il_outputs/ddpm_unet/beat_block_hammer/checkpoints/300.ckpt \
#       --num-eval 20 --start-seed 100
set -uo pipefail

task="beat_block_hammer"
ckpt=""
num_eval=20
start_seed=100
port=5599
gpu=0
per_ep_timeout=420   # kill a hung episode after this many seconds (a full 400-step fail is ~6min)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="$2"; shift 2;;
    --ckpt) ckpt="$2"; shift 2;;
    --num-eval) num_eval="$2"; shift 2;;
    --start-seed) start_seed="$2"; shift 2;;
    --port) port="$2"; shift 2;;
    --gpu) gpu="$2"; shift 2;;
    --per-ep-timeout) per_ep_timeout="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[[ -n "$ckpt" ]] || { echo "ERROR: --ckpt required"; exit 1; }
[[ -f "$ckpt" ]] || { echo "ERROR: checkpoint not found: $ckpt"; exit 1; }

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
ROBOVERSE_PY="${ROBOVERSE_PY:-conda run -n roboverse python}"
ROBOTWIN_PY="${ROBOTWIN_PY:-conda run -n robotwin python}"
srv_log="$(mktemp /tmp/dp_server.XXXXXX.log)"

echo "=== [1/3] starting DP policy server (roboverse env) on port ${port} ==="
PYTHONPATH="$repo_root" $ROBOVERSE_PY tools/robotwin_integration/dp_policy_server.py \
  --ckpt "$ckpt" --port "$port" --device "cuda:${gpu}" > "$srv_log" 2>&1 &
srv_pgid=$!
trap 'kill $(pgrep -f "dp_policy_server.py --ckpt $ckpt") 2>/dev/null; pkill -P $srv_pgid 2>/dev/null' EXIT

echo "=== [2/3] waiting for server to listen ==="
for i in $(seq 1 60); do
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then exec 3>&- 3<&-; echo "  server up"; break; fi
  sleep 3
  if [[ $i -eq 60 ]]; then echo "ERROR: server didn't come up in 180s; log:"; tail -20 "$srv_log"; exit 1; fi
done

echo "=== [3/3] closed-loop eval: ${num_eval} episodes from seed ${start_seed} ==="
# Run ONE episode per timeout-wrapped client against the persistent server. RoboTwin's
# sapien sim/render can intermittently HANG inside take_action/get_obs (not a socket
# stall, so the client-side socket timeout can't catch it) -- a per-episode process
# timeout kills a hung episode, counts it FAIL, and the sweep continues. The server
# (model) stays loaded across episodes; only the lightweight robotwin client restarts.
succ=0
for i in $(seq 0 $((num_eval - 1))); do
  seed=$((start_seed + i))
  out=$(timeout "${per_ep_timeout}" env MUJOCO_GL=egl SAPIEN_HEADLESS=1 $ROBOTWIN_PY \
    tools/robotwin_integration/eval_robotwin_policy.py \
    --task "$task" --policy dp --server "127.0.0.1:${port}" \
    --num-eval 1 --start-seed "$seed" 2>&1 | grep -aE "seed ${seed}\] (SUCCESS|FAIL)" | grep -av "step:")
  if echo "$out" | grep -qa "SUCCESS"; then
    succ=$((succ + 1)); echo "[seed $seed] SUCCESS"
  elif echo "$out" | grep -qa "FAIL"; then
    echo "[seed $seed] FAIL"
  else
    echo "[seed $seed] FAIL (timeout/hang after ${per_ep_timeout}s)"
  fi
done
rate=$(awk "BEGIN{printf \"%.1f\", ${succ}/${num_eval}*100}")
echo "RESULT ${task} | policy=dp | success ${succ}/${num_eval} = ${rate}%"

echo "=== done (server will be torn down) ==="
