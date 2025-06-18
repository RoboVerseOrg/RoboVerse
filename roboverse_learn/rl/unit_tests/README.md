# RoboVerse RL Unit Tests

This directory contains unit tests for the RoboVerse RL framework, with a focus on the AllegroHand task implementation.

## Test Files

### `test_allegrohand_task.py`
Tests the AllegroHand task configuration and core functionality:
- Task initialization and configuration
- Observation space setup
- Observation computation for different observation types (full_no_vel, full, full_state)
- Reward function computation
- Termination conditions
- Quaternion distance calculations
- Task registration and retrieval

### `test_training_config.py`
Tests the training configuration for AllegroHand with PPO:
- Configuration loading from YAML files
- PPO hyperparameter validation
- Network architecture configuration
- Environment settings
- Configuration override functionality
- Compatibility with default configuration

### `test_env_integration.py`
Integration tests for AllegroHand environment with the RL framework:
- RLEnvWrapper initialization
- Environment reset and step functionality
- Action and observation space compatibility
- Reward computation under various conditions
- Termination condition handling
- Multi-environment batch processing

## Running Tests

To run all tests:
```bash
cd roboverse_learn/rl
pytest unit_tests/ -v
```

To run a specific test file:
```bash
pytest unit_tests/test_allegrohand_task.py -v
```

To run a specific test:
```bash
pytest unit_tests/test_allegrohand_task.py::TestAllegroHandTask::test_reward_function -v
```

## Requirements

- pytest
- torch
- numpy
- omegaconf
- hydra-core

## Adding New Tests

When adding new tasks or features:
1. Create test files following the naming convention `test_*.py`
2. Use pytest fixtures for common setup
3. Test both positive and edge cases
4. Mock external dependencies when appropriate
5. Ensure tests are deterministic (use fixed seeds)

## Coverage

To run tests with coverage:
```bash
pytest unit_tests/ --cov=roboverse_learn.rl --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.