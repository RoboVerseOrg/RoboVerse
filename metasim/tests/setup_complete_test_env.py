"""Complete test environment setup for RoboVerse metasim tests.
This script ensures all dependencies are properly installed in the conda environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description="", check=True, capture_output=True):
    """Run a command and return success status."""
    if description:
        print(f"\n{description}")

    print(f"Running: {' '.join(cmd)}")

    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        else:
            result = subprocess.run(cmd, check=check)

        if capture_output and result.stdout:
            print(result.stdout)

        return True, result
    except subprocess.CalledProcessError as e:
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return False, e


def check_conda_env():
    """Check if we're in the correct conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
    print(f"\n📦 Current conda environment: {conda_env}")

    if conda_env == "base":
        print("\n⚠️  WARNING: You're in the base environment!")
        print("It's strongly recommended to use a dedicated environment.")
        print("\nTo create and activate a test environment, run:")
        print("  conda create -n roboverse-test python=3.10 -y")
        print("  conda activate roboverse-test")
        print("\nThen run this script again.")

        response = input("\nContinue in base environment anyway? (y/n): ")
        if response.lower() != "y":
            print("Setup cancelled.")
            return False

    return True


def install_conda_packages():
    """Install packages via conda for better compatibility."""
    print("\n🐍 Installing core packages via conda...")

    conda_packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pillow",
        "pyyaml",
        "tqdm",
    ]

    cmd = ["conda", "install", "-y"] + conda_packages
    success, _ = run_command(cmd, "Installing conda packages...", check=False)

    if not success:
        print("⚠️  Some conda packages failed to install, will retry with pip")

    return success


def upgrade_pip():
    """Upgrade pip and core tools."""
    print("\n📦 Upgrading pip, setuptools, and wheel...")

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    success, _ = run_command(cmd, check=False)

    if not success:
        print("⚠️  Failed to upgrade pip, continuing anyway...")

    return success


def install_pytorch():
    """Install PyTorch with appropriate backend."""
    print("\n🤖 Installing PyTorch...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    ]

    success, _ = run_command(cmd, "Installing PyTorch (CPU version for testing)...", check=False)

    if not success:
        print("❌ Failed to install PyTorch!")
        print("You may need to install it manually from https://pytorch.org/")

    return success


def install_test_dependencies():
    """Install all testing framework dependencies."""
    print("\n🧪 Installing test dependencies...")

    core_test_packages = {
        "pytest>=8.0.0": "Core testing framework",
        "pytest-asyncio": "Async test support",
        "pytest-mock": "Mocking support",
        "pytest-timeout": "Test timeout support",
        "pytest-xdist": "Parallel test execution",
    }

    reporting_packages = {
        "pytest-html>=4.0.0": "HTML test reports",
        "pytest-cov>=4.0.0": "Coverage integration",
        "coverage[toml]>=7.0": "Code coverage",
        "pytest-json-report": "JSON test reports",
        "pytest-benchmark": "Performance benchmarking",
        "pytest-metadata": "Test metadata",
    }

    quality_packages = {
        "black": "Code formatting",
        "isort": "Import sorting",
        "mypy": "Type checking",
    }

    all_packages = {**core_test_packages, **reporting_packages, **quality_packages}

    failed_packages = []

    for package, description in all_packages.items():
        print(f"\n📦 Installing {package} ({description})...")
        cmd = [sys.executable, "-m", "pip", "install", package]
        success, _ = run_command(cmd, check=False, capture_output=True)

        if success:
            print(f"✅ {package} installed")
        else:
            print(f"❌ {package} failed to install")
            failed_packages.append(package)

    return failed_packages


def install_metasim_dependencies():
    """Install metasim-specific dependencies."""
    print("\n📦 Installing metasim dependencies...")

    metasim_packages = {
        "omegaconf": "Configuration management",
        "hydra-core": "Hydra configuration",
        "tyro": "CLI interface",
        "einops": "Tensor operations",
        "trimesh": "3D mesh processing",
        "mujoco": "MuJoCo physics simulator",
        "dm-control": "DeepMind Control Suite",
        "gymnasium": "Gymnasium RL environments",
        "gym": "OpenAI Gym (legacy)",
    }

    failed_packages = []

    for package, description in metasim_packages.items():
        print(f"\n📦 Installing {package} ({description})...")
        cmd = [sys.executable, "-m", "pip", "install", package]
        success, _ = run_command(cmd, check=False, capture_output=True)

        if success:
            print(f"✅ {package} installed")
        else:
            print(f"⚠️  {package} failed to install (may be optional)")
            failed_packages.append(package)

    return failed_packages


def verify_installation():
    """Verify that critical packages are installed."""
    print("\n✅ Verifying installations...")
    print("-" * 60)

    critical_imports = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("pytest", "pytest"),
        ("coverage", "coverage"),
        ("pytest_html", "pytest-html"),
        ("pytest_cov", "pytest-cov"),
    ]

    optional_imports = [
        ("pytest_benchmark", "pytest-benchmark"),
        ("pytest_json_report", "pytest-json-report"),
        ("mujoco", "mujoco"),
        ("dm_control", "dm-control"),
    ]

    all_good = True

    print("\nCritical packages:")
    for import_name, package_name in critical_imports:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✅ {package_name} ({version})")
        except ImportError:
            print(f"  ❌ {package_name} - MISSING!")
            all_good = False

    print("\nOptional packages:")
    for import_name, package_name in optional_imports:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✅ {package_name} ({version})")
        except ImportError:
            print(f"  ⚠️  {package_name} - not installed")

    return all_good


def create_test_runner_script():
    """Create a simple test runner script."""
    script_content = '''"""Simple test runner for metasim tests."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run metasim tests with basic reporting."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    cmd = [
        "pytest",
        "-v",
        "--tb=short",
        "--cov=metasim",
        "--cov-report=term",
        "--cov-report=html:test_reports/coverage_html",
        "--html=test_reports/test_report.html",
        "--self-contained-html",
        "metasim/tests"
    ]

    print("Running tests...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        print("\\n✅ Tests passed!")
        print("\\nReports available at:")
        print("  - Coverage: test_reports/coverage_html/index.html")
        print("  - Test results: test_reports/test_report.html")
    else:
        print("\\n❌ Some tests failed. Check the reports for details.")

    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
'''

    script_path = Path("run_basic_tests.py")
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    print(f"\n✅ Created test runner script: {script_path}")


def main():
    """Main setup function."""
    print("🚀 RoboVerse Complete Test Environment Setup")
    print("=" * 60)

    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.minor}")

    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return 1

    if not check_conda_env():
        return 1

    install_conda_packages()

    upgrade_pip()

    if not install_pytorch():
        print("\n⚠️  PyTorch installation failed, but continuing...")

    failed_test_deps = install_test_dependencies()

    failed_metasim_deps = install_metasim_dependencies()

    all_critical_installed = verify_installation()

    create_test_runner_script()

    print("\n" + "=" * 60)
    print("📊 Setup Summary")
    print("=" * 60)

    if all_critical_installed:
        print("✅ All critical packages installed successfully!")

        print("\n🎯 Next steps:")
        print("1. Run the comprehensive test suite:")
        print("   python metasim/tests/generate_comprehensive_report.py --non-interactive")
        print("\n2. Or run basic tests:")
        print("   python run_basic_tests.py")
        print("\n3. Or run pytest directly:")
        print("   pytest metasim/tests -v")

    else:
        print("❌ Some critical packages are missing!")
        print("\nTry installing missing packages manually:")
        print("  pip install <package-name>")

        if failed_test_deps:
            print(f"\nFailed test packages: {', '.join(failed_test_deps[:5])}")

    print("\n💡 Tips:")
    print("- If you encounter import errors, try: pip install -e .")
    print("- For GPU support, install PyTorch with CUDA from https://pytorch.org/")
    print("- Check CLAUDE.md for project-specific guidelines")

    print("\n" + "=" * 60)

    return 0 if all_critical_installed else 1


if __name__ == "__main__":
    sys.exit(main())
