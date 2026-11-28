#!/bin/bash
# tools/gate/nightly.sh <sim> <out_dir>
# The task gate on one backend: `bench health --floor --repeat 2` over every registered task, isolated per task
# for backends that boot once per process (Isaac Sim). Environment: ROBOVERSE_GATE_PYTHON points at the venv
# for <sim> (defaults: $PWD/.venv/bin/python; for isaacsim $ISAACSIM_PYTHON); ACCEPT_EULA etc. for Isaac.
set -euo pipefail
sim=${1:?sim}
out=${2:?out_dir}
py=${ROBOVERSE_GATE_PYTHON:-$PWD/.venv/bin/python}
if [ "$sim" = isaacsim ]; then
  py=${ISAACSIM_PYTHON:-$py}
  export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y OPENBLAS_NUM_THREADS=1
fi
export PYTHONPATH="$PWD/packages/metasim:$PWD${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=egl
mkdir -p "$out"
"$py" -m metasim bench health --sim "$sim" --episodes 5 --threshold 0.8 --floor --repeat 2 --isolate auto \
  --device "${ROBOVERSE_GATE_DEVICE:-cpu}" --out-dir "$out" || status=$?
echo "gate exit: ${status:-0} (1 = at least one task is low / unreliable / flaky / error)"
exit "${status:-0}"
