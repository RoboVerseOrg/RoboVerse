"""Quick environment check for metasim tests."""

import subprocess
import sys
from pathlib import Path


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def check_command(cmd):
    """Check if a command is available."""
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    """Check test environment status."""
    print("🔍 Metasim Test Environment Check")
    print("=" * 60)

    print(f"\n🐍 Python: {sys.version}")
    print(f"   Executable: {sys.executable}")

    import os

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "not activated")
    print(f"\n📦 Conda environment: {conda_env}")

    print("\n📋 Critical Packages:")
    critical = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("pytest", "pytest"),
        ("coverage", "coverage"),
        ("pytest_html", "pytest-html"),
        ("pytest_cov", "pytest-cov"),
    ]

    missing_critical = []
    for import_name, package_name in critical:
        installed, version = check_import(import_name)
        if installed:
            print(f"  ✅ {package_name}: {version}")
        else:
            print(f"  ❌ {package_name}: NOT INSTALLED")
            missing_critical.append(package_name)

    print("\n📋 Optional Packages:")
    optional = [
        ("pytest_json_report", "pytest-json-report"),
        ("mujoco", "mujoco"),
        ("omegaconf", "omegaconf"),
    ]

    missing_optional = []
    for import_name, package_name in optional:
        installed, version = check_import(import_name)
        if installed:
            print(f"  ✅ {package_name}: {version}")
        else:
            print(f"  ⚠️  {package_name}: not installed")
            missing_optional.append(package_name)

    print("\n📋 Metasim Import:")
    try:
        import metasim

        print("  ✅ metasim can be imported")
    except ImportError as e:
        print(f"  ❌ metasim cannot be imported: {e}")
        print("     Try: pip install -e . (from project root)")

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)

    if not missing_critical:
        print("✅ All critical packages are installed!")
        print("\nYou can run tests with:")
        print("  pytest metasim/tests -v")
        print("\nOr generate comprehensive reports:")
        print("  python metasim/tests/generate_comprehensive_report.py")
    else:
        print("❌ Missing critical packages:")
        for pkg in missing_critical:
            print(f"  - {pkg}")

        print("\n🔧 Quick fix - install all at once:")
        print(f"  pip install {' '.join(missing_critical)}")

        print("\n🔧 Or run the complete setup:")
        print("  python metasim/tests/setup_complete_test_env.py")

    if missing_optional:
        print("\n⚠️  Missing optional packages (for enhanced reports):")
        print(f"  pip install {' '.join(missing_optional[:3])}...")

    test_dir = Path("metasim/tests")
    if test_dir.exists():
        test_files = list(test_dir.glob("**/test_*.py"))
        print(f"\n📁 Found {len(test_files)} test files in {test_dir}")
    else:
        print(f"\n❌ Test directory not found: {test_dir}")
        print("   Make sure you're running from the project root!")

    return 0 if not missing_critical else 1


if __name__ == "__main__":
    sys.exit(main())
