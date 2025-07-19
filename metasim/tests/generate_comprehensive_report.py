#!/usr/bin/env python3
"""Generate comprehensive test reports with coverage, performance, and detailed results."""

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path


def ensure_in_project_root():
    """Ensure we're running from the project root directory."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    os.chdir(project_root)
    print(f"📍 Working directory: {os.getcwd()}")
    return project_root


def run_command(cmd, description, check=False):
    """Run a command and capture output."""
    print(f"\n{'=' * 60}")
    print(f"🔧 {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)

        if result.returncode != 0:
            print(f"⚠️  Warning: Command failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")

        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return None
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        return None


def is_interactive():
    """Check if we're running in an interactive terminal."""
    return sys.stdin.isatty()


def install_missing_dependencies():
    """Install required pytest plugins if missing."""
    required_deps = ["pytest-html", "pytest-cov", "pytest-xdist"]

    optional_deps = ["pytest-json-report", "pytest-benchmark", "pytest-profiling", "pytest-metadata"]

    print("\n📦 Checking dependencies...")

    # Check required dependencies
    missing_required = []
    for dep in required_deps:
        try:
            __import__(dep.replace("-", "_").replace("pytest_", "pytest-"))
        except ImportError:
            missing_required.append(dep)

    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        print("Installing required dependencies...")
        cmd = [sys.executable, "-m", "pip", "install"] + missing_required
        subprocess.run(cmd, check=True)
        print("✅ Required dependencies installed!")

    # Check optional dependencies
    missing_optional = []
    for dep in optional_deps:
        try:
            __import__(dep.replace("-", "_").replace("pytest_", "pytest-"))
        except ImportError:
            missing_optional.append(dep)

    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies for enhanced reports: {', '.join(missing_optional)}")
        if is_interactive():
            response = input("Install optional dependencies for better reports? (y/n): ")
            if response.lower() == "y":
                cmd = [sys.executable, "-m", "pip", "install"] + missing_optional
                subprocess.run(cmd, check=False)
                print("✅ Optional dependencies installed!")
        else:
            print("⚠️  Running in non-interactive mode, skipping optional dependency installation.")
            print("   To install manually: pip install " + " ".join(missing_optional))


def generate_comprehensive_report():
    """Generate all possible test reports."""
    # Ensure we're in the project root
    project_root = ensure_in_project_root()

    # Create reports directory
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)

    # Clean up old reports (optional)
    if is_interactive():
        response = input("\n🧹 Clean up old reports? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(reports_dir)
            reports_dir.mkdir()
    else:
        print("\n🧹 Keeping existing reports (non-interactive mode)")

    # Timestamp for unique report names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n🚀 Starting Comprehensive Test Report Generation")
    print(f"📁 Reports will be saved in: {reports_dir.absolute()}")

    # Basic pytest command that should always work
    pytest_cmd = [
        "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        # Coverage options (these are essential)
        "--cov=metasim",
        "--cov-report=html:test_reports/coverage_html",
        "--cov-report=term",  # Show in terminal too
        "--cov-report=json:test_reports/coverage.json",
        "--cov-report=xml:test_reports/coverage.xml",
        # Test result reports (essential)
        f"--html=test_reports/pytest_report_{timestamp}.html",
        "--self-contained-html",
        # JUnit XML for CI integration
        f"--junit-xml=test_reports/junit_{timestamp}.xml",
        # Path to tests
        "metasim/tests",
    ]

    # Add optional features if available
    try:
        import pytest_json_report

        pytest_cmd.extend([
            "--json-report",
            f"--json-report-file=test_reports/test_results_{timestamp}.json",
            "--json-report-summary",
        ])
        print("✅ JSON report plugin available")
    except ImportError:
        print("⚠️  pytest-json-report not installed, skipping JSON report")

    try:
        import pytest_profiling

        pytest_cmd.extend(["--profile", "--profile-svg"])
        print("✅ Profiling plugin available")
    except ImportError:
        print("⚠️  pytest-profiling not installed, skipping profiling")

    try:
        import pytest_benchmark

        print("✅ Benchmark plugin available")
    except ImportError:
        print("⚠️  pytest-benchmark not installed")

    # Add parallel execution if xdist is available
    try:
        import xdist

        pytest_cmd.extend(["-n", "auto"])
        print("✅ Parallel execution enabled")
    except ImportError:
        print("⚠️  pytest-xdist not installed, running tests sequentially")

    # Run the tests
    print("\n" + "=" * 60)
    print("🧪 Running pytest with comprehensive reporting...")
    print("=" * 60)

    result = subprocess.run(pytest_cmd, check=False, text=True)

    # Check if coverage report was generated
    coverage_html_dir = reports_dir / "coverage_html"
    if not coverage_html_dir.exists() or not (coverage_html_dir / "index.html").exists():
        print("\n❌ Coverage HTML report was not generated!")
        print("This might be because tests failed or coverage plugin is not properly installed.")
    else:
        print(f"\n✅ Coverage report generated at: {coverage_html_dir / 'index.html'}")

    # Check if test report was generated
    test_report = reports_dir / f"pytest_report_{timestamp}.html"
    if not test_report.exists():
        print("\n❌ Test results HTML was not generated!")
    else:
        print(f"✅ Test report generated at: {test_report}")

    # Generate additional reports (removed coverage badge and flake8 as requested)

    # Create test summary
    generate_test_summary(reports_dir, timestamp, result)

    # Create master index.html
    create_master_index(reports_dir, timestamp)

    # Final summary
    print("\n" + "=" * 60)
    print("📊 Report Generation Complete!")
    print("=" * 60)

    print("\n📁 Generated files:")
    for file in sorted(reports_dir.iterdir()):
        if file.is_file():
            print(f"   - {file.name}")
        elif file.is_dir():
            print(f"   - {file.name}/")
            for subfile in sorted(file.iterdir())[:3]:
                print(f"      - {subfile.name}")
            if len(list(file.iterdir())) > 3:
                print(f"      ... and {len(list(file.iterdir())) - 3} more files")

    # Open the master report in browser
    index_path = reports_dir / "index.html"
    print(f"\n✅ Master report: {index_path.absolute()}")

    # Ask if user wants to open in browser
    if is_interactive():
        response = input("\n🌐 Open report in browser? (y/n): ")
        if response.lower() == "y":
            webbrowser.open(f"file://{index_path.absolute()}")
    else:
        print("\n🌐 To view the report, open: file://" + str(index_path.absolute()))

    return reports_dir


def create_master_index(reports_dir, timestamp):
    """Create a master index.html that links all reports."""
    # Check which reports actually exist
    coverage_exists = (reports_dir / "coverage_html" / "index.html").exists()
    test_report_exists = (reports_dir / f"pytest_report_{timestamp}.html").exists()
    junit_exists = (reports_dir / f"junit_{timestamp}.xml").exists()
    json_exists = (reports_dir / f"test_results_{timestamp}.json").exists()
    # Removed badge and flake8 checks as they are no longer generated
    summary_exists = (reports_dir / f"test_summary_{timestamp}.txt").exists()

    index_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metasim Test Reports - {timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .report-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .report-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
        }}
        .report-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .report-card.disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .report-card.disabled:hover {{
            transform: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .report-card h3 {{
            margin-top: 0;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .report-card a {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.2s;
        }}
        .report-card a:hover {{
            background-color: #2980b9;
        }}
        .report-card.disabled a {{
            background-color: #95a5a6;
            cursor: not-allowed;
        }}
        .not-available {{
            color: #e74c3c;
            font-style: italic;
            margin-top: 10px;
        }}
        .summary {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .metric {{
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 8px;
            text-align: center;
        }}
        .metric .value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric .label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .status-passed {{ color: #27ae60; }}
        .status-failed {{ color: #e74c3c; }}
        .status-skipped {{ color: #f39c12; }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #7f8c8d;
        }}
        .icon {{
            font-size: 1.5em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .status-badge {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-available {{
            background-color: #27ae60;
            color: white;
        }}
        .status-unavailable {{
            background-color: #e74c3c;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Metasim Test Reports</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <h2>📊 Report Status</h2>
        <p>This page links to all test reports generated. Some reports may not be available if the corresponding plugins are not installed or if tests failed.</p>
    </div>

    <h2>📑 Available Reports</h2>
    <div class="report-grid">
        <div class="report-card {"disabled" if not coverage_exists else ""}">
            <span class="status-badge {"status-available" if coverage_exists else "status-unavailable"}">
                {"Available" if coverage_exists else "Not Generated"}
            </span>
            <h3><span class="icon">📈</span> Coverage Report</h3>
            <p>Detailed code coverage analysis with line-by-line coverage information.</p>
            {'<a href="coverage_html/index.html">View Coverage Report</a>' if coverage_exists else '<p class="not-available">Coverage report not generated. Run with --cov flag.</p>'}
        </div>

        <div class="report-card {"disabled" if not test_report_exists else ""}">
            <span class="status-badge {"status-available" if test_report_exists else "status-unavailable"}">
                {"Available" if test_report_exists else "Not Generated"}
            </span>
            <h3><span class="icon">🧪</span> Test Results</h3>
            <p>Detailed test execution report with pass/fail status and error messages.</p>
            {'<a href="pytest_report_' + timestamp + '.html">View Test Report</a>' if test_report_exists else '<p class="not-available">Test report not generated. Check if pytest-html is installed.</p>'}
        </div>

        <div class="report-card {"disabled" if not junit_exists else ""}">
            <span class="status-badge {"status-available" if junit_exists else "status-unavailable"}">
                {"Available" if junit_exists else "Not Generated"}
            </span>
            <h3><span class="icon">📊</span> JUnit XML Report</h3>
            <p>Machine-readable test results for CI/CD integration.</p>
            {'<a href="junit_' + timestamp + '.xml">View JUnit XML</a>' if junit_exists else '<p class="not-available">JUnit XML not generated.</p>'}
        </div>

        <div class="report-card {"disabled" if not summary_exists else ""}">
            <span class="status-badge {"status-available" if summary_exists else "status-unavailable"}">
                {"Available" if summary_exists else "Not Generated"}
            </span>
            <h3><span class="icon">📝</span> Test Summary</h3>
            <p>Quick overview of test results and key metrics.</p>
            {'<a href="test_summary_' + timestamp + '.txt">View Summary</a>' if summary_exists else '<p class="not-available">Summary not generated.</p>'}
        </div>
    </div>

    <h2>🔧 Troubleshooting</h2>
    <div class="summary">
        <h3>Missing Reports?</h3>
        <p>If some reports are not available, you may need to install additional dependencies:</p>
        <pre><code>pip install pytest-html pytest-cov pytest-json-report pytest-benchmark \\
            pytest-profiling pytest-metadata</code></pre>

        <h3>Regenerate Reports</h3>
        <pre><code>python metasim/tests/generate_comprehensive_report.py</code></pre>

        <h3>Manual Report Generation</h3>
        <pre><code>pytest --cov=metasim --cov-report=html:test_reports/coverage_html \\
       --html=test_reports/pytest_report.html --self-contained-html \\
       metasim/tests</code></pre>
    </div>

    <div class="footer">
        <p>Generated by Metasim Test Suite | Report timestamp: {timestamp}</p>
    </div>
</body>
</html>
"""

    index_path = reports_dir / "index.html"
    index_path.write_text(index_content)
    print(f"✅ Created master index at: {index_path}")


def generate_test_summary(reports_dir, timestamp, test_result):
    """Generate a text summary of test results."""
    summary_path = reports_dir / f"test_summary_{timestamp}.txt"

    # Get some basic stats from output
    output = test_result.stdout if test_result and test_result.stdout else "No output captured"

    summary_content = f"""
METASIM TEST SUMMARY
====================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

TEST EXECUTION RESULTS
----------------------
Exit Code: {test_result.returncode if test_result else "N/A"}

STDOUT:
{output[:5000]}... (truncated) if output is long

ERROR OUTPUT (if any):
{test_result.stderr[:2000] if test_result and test_result.stderr else "No errors"}

REPORT LOCATIONS
----------------
- Coverage HTML: test_reports/coverage_html/index.html
- Test Results: test_reports/pytest_report_{timestamp}.html
- JUnit XML: test_reports/junit_{timestamp}.xml
- This Summary: test_reports/test_summary_{timestamp}.txt

QUICK COMMANDS
--------------
# View coverage report
open test_reports/coverage_html/index.html

# View test results
open test_reports/pytest_report_{timestamp}.html

# Re-run failed tests only
pytest --lf

# Run tests with specific marker
pytest -m unit

# Run tests for specific simulator
pytest -k "mujoco"
"""

    summary_path.write_text(summary_content)
    print(f"✅ Created test summary at: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive test reports for metasim")
    parser.add_argument("--non-interactive", "-n", action="store_true", help="Run in non-interactive mode (no prompts)")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean up old reports before generating new ones")
    args = parser.parse_args()

    # Override interactive mode if requested
    if args.non_interactive:
        # Monkey patch is_interactive to always return False
        globals()["is_interactive"] = lambda: False

    # Check and install dependencies
    install_missing_dependencies()

    # Handle clean flag in non-interactive mode
    if args.clean and args.non_interactive:
        reports_dir = Path("test_reports")
        if reports_dir.exists():
            print("\n🧹 Cleaning up old reports...")
            shutil.rmtree(reports_dir)

    # Generate comprehensive reports
    generate_comprehensive_report()
