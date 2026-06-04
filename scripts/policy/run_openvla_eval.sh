#!/bin/bash
# Guarded launcher: wait for the (shared) GPU to have enough free memory, then
# run the official LIBERO-plus openVLA-OFT checkpoint closed-loop through the
# sim<->policy bridge in bf16 (no quantization), and save the eval results.
#
# bf16 needs ~15 GB; this machine's RTX 5090 is shared with other sessions, so
# we poll until enough frees up rather than fighting for it. Run detached:
#   setsid nohup bash scripts/policy/run_openvla_eval.sh > .../launcher.log 2>&1 < /dev/null &
set -u

# Local-dev launcher: override these for your machine (or export before running).
RVLIB=${RVLIB:-$(cd "$(dirname "$0")/../.." && pwd)}
OVREPO=${OPENVLA_OFT_REPO:-/path/to/openvla-oft}
CKPT=$RVLIB/third_party/ckpt/openvla-oft-libero-plus
SRV_LOG=$RVLIB/third_party/ckpt/bf16_server.log
RES_LOG=$RVLIB/third_party/ckpt/openvla_eval_result.log
PORT=5555
NEED_MIB=${NEED_MIB:-15500}
EPISODES=${EPISODES:-5}
BASE=${BASE:-pick_up_the_alphabet_soup_and_place_it_in_the_basket}
SUITE=${SUITE:-libero_object}

echo "[launcher] waiting for >= ${NEED_MIB} MiB free GPU ..."
for i in $(seq 1 120); do   # up to ~60 min
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "[launcher] attempt $i: ${free} MiB free"
  [ "${free:-0}" -ge "$NEED_MIB" ] && break
  sleep 30
done
if [ "${free:-0}" -lt "$NEED_MIB" ]; then
  echo "[launcher] GPU never freed enough (${free} MiB). Aborting; re-run later."
  exit 1
fi

echo "[launcher] launching bf16 policy server ..."
cd "$OVREPO"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  /home/ghr/miniconda3/envs/openvla/bin/python "$RVLIB/scripts/policy/bridge_policy_server.py" \
  --ckpt "$CKPT" --port "$PORT" --quant none > "$SRV_LOG" 2>&1 < /dev/null &
SRV=$!

echo "[launcher] waiting for server to load (PID $SRV) ..."
for i in $(seq 1 120); do   # up to 10 min to load
  grep -qa "listening on" "$SRV_LOG" && { echo "[launcher] server ready"; break; }
  if grep -qaE "out of memory|CUDA out|Traceback" "$SRV_LOG"; then
    echo "[launcher] server failed to load:"; tail -5 "$SRV_LOG"; kill -9 $SRV 2>/dev/null; exit 1
  fi
  kill -0 $SRV 2>/dev/null || { echo "[launcher] server died"; tail -5 "$SRV_LOG"; exit 1; }
  sleep 5
done

echo "[launcher] running sim client eval (suite=$SUITE base=$BASE episodes=$EPISODES) ..."
cd "$RVLIB"
LIBERO_CONFIG_PATH=$HOME/.libero_plus MUJOCO_GL=egl \
  /home/ghr/miniconda3/envs/liberoplus/bin/python -m scripts.policy.bridge_sim_client \
  --suite "$SUITE" --base "$BASE" --episodes "$EPISODES" --port "$PORT" 2>&1 | tee "$RES_LOG"

echo "[launcher] eval done; shutting down server"
kill -9 $SRV 2>/dev/null
echo "[launcher] COMPLETE"
