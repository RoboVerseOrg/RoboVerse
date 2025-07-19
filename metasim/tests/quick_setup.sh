#!/bin/bash
# Quick setup script for metasim test environment

echo "🚀 Quick Metasim Test Setup"
echo "=========================="

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ No conda environment activated!"
    echo "Please run: conda activate roboverse-test"
    exit 1
fi

echo "📦 Using conda environment: $CONDA_DEFAULT_ENV"

# Install numpy first (many packages depend on it)
echo ""
echo "📦 Installing numpy first..."
pip install numpy --quiet

# Install all test requirements
echo ""
echo "📦 Installing test requirements..."
pip install -r metasim/tests/test_requirements.txt

# Install the package itself in editable mode
echo ""
echo "📦 Installing metasim in editable mode..."
pip install -e . --quiet

# Quick verification
echo ""
echo "✅ Verifying installation..."
python -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>/dev/null || echo "  ❌ numpy"
python -c "import torch; print(f'  ✅ torch {torch.__version__}')" 2>/dev/null || echo "  ❌ torch"
python -c "import pytest; print(f'  ✅ pytest {pytest.__version__}')" 2>/dev/null || echo "  ❌ pytest"
python -c "import coverage; print(f'  ✅ coverage {coverage.__version__}')" 2>/dev/null || echo "  ❌ coverage"
python -c "import metasim; print('  ✅ metasim (local)')" 2>/dev/null || echo "  ❌ metasim"

echo ""
echo "🎯 Setup complete! You can now run:"
echo "  python metasim/tests/generate_comprehensive_report.py --non-interactive"
echo ""
echo "Or for a quick test:"
echo "  pytest metasim/tests -v --tb=short"
