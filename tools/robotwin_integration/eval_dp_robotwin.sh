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

# Guard against train->eval GPU contention. If a DP training run is still on the GPU when the
# policy server loads, the server's first inference can HANG indefinitely -- every episode then
# burns its full --per-ep-timeout and the run reports a spurious 0/N (looks like total policy
# failure but is purely a scheduling artifact). Wait for any training process to exit + a brief
# settle so the CUDA caching allocator releases, before the server claims the GPU. No-op when
# nothing is training (the normal standalone-eval case), so this is backward-compatible.
echo "=== [0/3] ensuring the GPU is free of DP training (avoids false 0/N from contention) ==="
for i in $(seq 1 60); do
  if pgrep -f "roboverse_learn.il.train" >/dev/null 2>&1; then
    [[ $i -eq 1 ]] && echo "  a DP training process is still on the GPU; waiting for it to finish..."
    sleep 5
    [[ $i -eq 60 ]] && echo "  WARN: training still running after 300s; proceeding anyway"
  else
    [[ $i -gt 1 ]] && echo "  training finished; GPU released"
    break
  fi
done
sleep 3  # let the CUDA caching allocator hand the memory back before the server grabs it

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
hangstreak=0
for i in $(seq 0 $((num_eval - 1))); do
  seed=$((start_seed + i))
  # Write the client's (very chatty -- per-step "step: N/None" + svulkan) output to a file and
  # grep the file AFTERWARDS, rather than piping it live through grep. Piping a long episode's
  # stdout through a grep inside $() can deadlock on pipe back-pressure for high-step tasks (e.g.
  # handover_block's 800-step budget), which looked exactly like an RT-render "hang". A plain file
  # redirect decouples the client from any consumer, so it never blocks on write.
  ep_log="$(mktemp /tmp/dp_ep.XXXXXX.log)"
  # PYTHONUNBUFFERED=1 is REQUIRED: without it the client's very chatty per-step stdout is
  # block-buffered to the redirected file, and that buffering reproducibly wedges the episode
  # (standalone clients that set it complete the SAME seed that hangs here without it). With it,
  # the eval matches the known-good standalone path.
  timeout "${per_ep_timeout}" env MUJOCO_GL=egl SAPIEN_HEADLESS=1 PYTHONUNBUFFERED=1 $ROBOTWIN_PY \
    tools/robotwin_integration/eval_robotwin_policy.py \
    --task "$task" --policy dp --server "127.0.0.1:${port}" \
    --num-eval 1 --start-seed "$seed" > "$ep_log" 2>&1
  out=$(grep -aE "seed ${seed}\] (SUCCESS|FAIL)" "$ep_log" | grep -av "step:")
  rm -f "$ep_log"
  if echo "$out" | grep -qa "SUCCESS"; then
    succ=$((succ + 1)); hangstreak=0; echo "[seed $seed] SUCCESS"
  elif echo "$out" | grep -qa "FAIL"; then
    hangstreak=0; echo "[seed $seed] FAIL"
  else
    hangstreak=$((hangstreak + 1))
    echo "[seed $seed] FAIL (timeout/hang after ${per_ep_timeout}s)"
    # An episode that produces NO result line hit the wall-clock timeout = a hang, not a
    # policy decision. The first episodes all hanging means the server is wedged (e.g. GPU
    # still contended) -- abort fast with a diagnostic instead of burning N*timeout and
    # reporting a misleading 0/N. (A real policy failure prints "FAIL" and resets the streak.)
    if [[ $hangstreak -ge 3 ]]; then
      echo "ERROR: ${hangstreak} consecutive episodes hung with no result -- the policy server" >&2
      echo "       is not responding (likely GPU contention or a wedged server). Aborting." >&2
      echo "       Server log tail:" >&2; tail -15 "$srv_log" >&2
      exit 2
    fi
  fi
done
rate=$(awk "BEGIN{printf \"%.1f\", ${succ}/${num_eval}*100}")
echo "RESULT ${task} | policy=dp | success ${succ}/${num_eval} = ${rate}%"

echo "=== done (server will be torn down) ==="
