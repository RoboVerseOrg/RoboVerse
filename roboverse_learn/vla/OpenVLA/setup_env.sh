#!/bin/bash
# Setup RLDS and OpenVLA environments

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=========================================="
echo "OpenVLA Environment Setup"
echo "=========================================="

# RLDS Environment
echo "[1/2] RLDS environment..."
cd "$SCRIPT_DIR/../rlds_utils"
if conda env list | grep -q "^rlds_env "; then
    echo "✓ rlds_env exists"
else
    conda env create -f environment_ubuntu.yml
    echo "✓ rlds_env created"
fi

# OpenVLA Environment
echo "[2/2] OpenVLA environment..."
cd "$PROJECT_ROOT/third_party"

if [ ! -d "openvla" ]; then
    git clone https://github.com/openvla/openvla.git
    echo "✓ OpenVLA cloned"
else
    echo "✓ OpenVLA exists"
fi

if conda env list | grep -q "^openvla "; then
    echo "✓ openvla env exists"
else
    conda create -n openvla python=3.10 -y
    echo "✓ openvla env created"
fi

eval "$(conda shell.bash hook)"
conda activate openvla

echo "Installing dependencies..."
cd openvla && pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

echo ""
echo "✓ Setup Complete!"
echo "  - rlds_env: Data conversion"
echo "  - openvla: Training & evaluation"
echo ""
