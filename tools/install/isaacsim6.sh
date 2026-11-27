#!/usr/bin/env bash
# Install the Isaac Sim 6.0 + Isaac Lab 3.0 backend for MetaSim into the *current* Python 3.12 environment.
#
# This script is the documented procedure (packages/metasim/requirements/isaacsim6.txt), executed
# in the order that actually works:
#   1. isaacsim[all,extscache]==6.0.1.0 (pip, NVIDIA index)
#   2. Isaac Lab release/3.0.0 from source (not on PyPI): isaaclab, isaaclab_physx, isaaclab_contrib
#   3. LAST: CUDA 13.0 torch 2.11.0 — so nothing above leaves a build without Blackwell (sm_120) kernels.
# Then `python -m metasim doctor --backend isaacsim` must report every row ok.
#
# Usage:  tools/install/isaacsim6.sh [ISAACLAB_DIR]      (default: ./IsaacLab next to the repo)
# Env:    METASIM_TORCH_INDEX (default https://download.pytorch.org/whl/cu130)
#         TMPDIR / PIP_CACHE_DIR — point them at a disk with >20 GB free; the Isaac wheels are ~6 GB.
# Run-time: the EULA variables on first launch (see the requirements file).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_DIR="${1:-${REPO_ROOT}/../IsaacLab}"
ISAACLAB_BRANCH="release/3.0.0"
TORCH_INDEX="${METASIM_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

py_minor="$(python -c 'import sys; print(sys.version_info.minor)')"
if [[ "$(python -c 'import sys; print(sys.version_info.major)')" != "3" || "${py_minor}" != "12" ]]; then
  echo "Isaac Sim 6.0 needs Python 3.12 (found $(python --version)); create the venv with 3.12 first." >&2
  exit 1
fi

echo "== 1/3 isaacsim 6.0.1 (+ MetaSim extra)"
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/packages/metasim[isaacsim]" --extra-index-url https://pypi.nvidia.com

echo "== 2/3 Isaac Lab ${ISAACLAB_BRANCH} from source -> ${ISAACLAB_DIR}"
if [[ ! -d "${ISAACLAB_DIR}" ]]; then
  git clone --depth 1 --branch "${ISAACLAB_BRANCH}" https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi
python -m pip install -e "${ISAACLAB_DIR}/source/isaaclab" -e "${ISAACLAB_DIR}/source/isaaclab_physx" -e "${ISAACLAB_DIR}/source/isaaclab_contrib"

echo "== 3/3 CUDA torch (last, on purpose)"
python -m pip install --index-url "${TORCH_INDEX}" "torch==2.11.0" torchvision

echo "== verify"
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y python -m metasim doctor --backend isaacsim
