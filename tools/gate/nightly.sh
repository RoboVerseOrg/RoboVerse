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
echo "gate exit: ${status:-0} (1 = at least one task is low / unreliable / flaky / invalid / error)"
# the robot sweep: every config the robot packages define, six checks each (robots.md is published with health.md);
# on Isaac Sim each robot's simulator check runs in its own process (--isolate auto)
"$py" -m metasim robot check --all --sim "$sim" --isolate auto --out-dir "$out/robots" || robots_status=$?
echo "robot sweep exit: ${robots_status:-0} (1 = at least one robot fails a check)"
if [ "${status:-0}" -ne 0 ] || [ "${robots_status:-0}" -ne 0 ]; then exit 1; fi
exit 0
