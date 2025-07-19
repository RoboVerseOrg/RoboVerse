# Metasim Unit Tests Guide

```bash
conda create -n roboverse-test python=3.10
conda activate roboverse-test
```

```bash
python metasim/tests/setup_complete_test_env.py
```


### Verify Installation

```bash
python metasim/tests/check_test_env.py
```

## Running Tests

### Basic Test Execution

#### Run all tests:
```bash
pytest metasim/tests -v
```

#### Run tests for a specific module:
```bash
pytest metasim/tests/simulators/test_mujoco.py -v
```

#### Run tests matching a pattern:
```bash
pytest metasim/tests -k "replay_demo" -v
```

#### Run tests in parallel (faster):
```bash
pytest metasim/tests -n auto -v
```

## Run tests with report/dashboard

```bash
python metasim/tests/generate_comprehensive_report.py
```

Reports are saved in `test_reports/` directory.

### Non-Interactive Mode

For CI/CD or automated testing:

```bash
python metasim/tests/generate_comprehensive_report.py --non-interactive
```

## Current Test Structure

```
metasim/tests/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures and configuration
├── generate_comprehensive_report.py  # Report generation script
├── setup_complete_test_env.py   # Environment setup script
├── check_test_env.py           # Environment verification
│
├── common/                      # Core functionality tests
│   ├── test_api_compliance.py  # API consistency tests
│   ├── test_state_management.py # State handling tests
│   ├── test_error_handling.py  # Error handling tests
│   └── test_performance.py     # Performance benchmarks
│
├── simulators/                  # Simulator-specific tests
│   ├── test_mujoco.py          # MuJoCo tests
│   ├── test_genesis.py         # Genesis tests
│   ├── test_isaacgym.py        # Isaac Gym tests
│   └── test_rendering_all_simulators.py  # Rendering tests
│
├── integration/                 # Integration tests
│   ├── test_cross_simulator_consistency.py  # Cross-sim tests
│   ├── test_replay_demo.py     # Replay functionality
│   └── test_replay_demo_functional.py  # Black-box tests
│
└── assets/                      # Test assets (URDF/MJCF files)
    ├── robots/                  # Test robot models
    └── objects/                 # Test object models
```
