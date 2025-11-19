# test_scene_randomizer.py — Quick Guide

This file verifies the core behavior of the Scene Randomizer, ensuring that:
- Parameter samples fall within valid ranges.
- Random seeds yield reproducible results.
- Edge cases and error paths are handled correctly.

Refer to the test cases for exact assertions and coverage.

## Location
`metasim/test/randomization/test_scene_randomizer.py`

## Prerequisites
- Project dependencies installed (see `pyproject.toml` or `docs/requirements.txt`).
- `pytest` installed: `pip install pytest`.

## How to Run (from repo root)
- Run this file only:
  ```zsh
  pytest -q metasim/test/randomization/test_scene_randomizer.py
  ```
- Run all randomization tests:
  ```zsh
  pytest -q metasim/test/randomization
  ```
- Optional: more verbose output or filter cases:
  ```zsh
  pytest -vv metasim/test/randomization/test_scene_randomizer.py
  pytest -q metasim/test/randomization -k scene_randomizer
  ```

## Troubleshooting
- Import errors (ImportError) or missing packages: ensure you run from the repo root, or set `PYTHONPATH=.` temporarily:
  ```zsh
  PYTHONPATH=. pytest -q metasim/test/randomization/test_scene_randomizer.py
  ```

## Expected Result
- All assertions pass: pytest exits with code 0 and shows `passed`.
