#!/usr/bin/env bash
# Install the Isaac Sim 5.0 backend for MetaSim into the *current* Python 3.11 environment.
#
# This script is the documented procedure (packages/metasim/requirements/isaacsim5.txt), executed
# in the order that actually works:
#   1. isaacsim[all,extscache]==5.0.0 (pip, NVIDIA index)
#   2. Isaac Lab v2.2.1 from source (not on PyPI); its build needs setuptools<70 + flatdict
#   3. numpy<2, h5py
#   4. LAST: CUDA 12.8 torch 2.7.0 — the steps above silently replace torch with the default
#      cu126 wheel, which has no Blackwell (sm_120) kernels.
# Then `python -m metasim doctor --backend isaacsim` must report every row ok.
#
# Usage:  tools/install/isaacsim5.sh [ISAACLAB_DIR]      (default: ./IsaacLab next to the repo)
# Env:    METASIM_TORCH_INDEX (default https://download.pytorch.org/whl/cu128)
#         TMPDIR / PIP_CACHE_DIR — point them at a disk with >20 GB free; the Isaac wheels are ~6 GB.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_DIR="${1:-${REPO_ROOT}/../IsaacLab}"
ISAACLAB_TAG="v2.2.1"
TORCH_INDEX="${METASIM_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

py_minor="$(python -c 'import sys; print(sys.version_info.minor)')"
if [[ "$(python -c 'import sys; print(sys.version_info.major)')" != "3" || "${py_minor}" != "11" ]]; then
  echo "Isaac Sim 5.0 needs Python 3.11 (found $(python --version)); create the venv with 3.11 first." >&2
  exit 1
fi

echo "== 1/4 isaacsim 5.0.0 (+ MetaSim extra)"
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/packages/metasim[isaacsim]" --extra-index-url https://pypi.nvidia.com

echo "== 2/4 Isaac Lab ${ISAACLAB_TAG} from source -> ${ISAACLAB_DIR}"
if [[ ! -d "${ISAACLAB_DIR}" ]]; then
  git clone --depth 1 --branch "${ISAACLAB_TAG}" https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi
python -m pip install "setuptools<70" wheel
python -m pip install --no-build-isolation flatdict
python -m pip install -e "${ISAACLAB_DIR}/source/isaaclab" -e "${ISAACLAB_DIR}/source/isaaclab_assets" -e "${ISAACLAB_DIR}/source/isaaclab_tasks"

echo "== 3/4 numpy<2, h5py"
python -m pip install "numpy<2" h5py

echo "== 4/4 CUDA torch (last, on purpose)"
python -m pip install --index-url "${TORCH_INDEX}" "torch==2.7.0" "torchvision==0.22.0"

echo "== verify"
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y python -m metasim doctor --backend isaacsim
