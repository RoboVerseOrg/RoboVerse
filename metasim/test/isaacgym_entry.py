import sys

import isaacgym  # noqa: F401  # must be imported before torch
import pytest

if __name__ == "__main__":
    # pass through any CLI args, e.g. `python run_tests.py tests/ -k foo`
    sys.exit(pytest.main(sys.argv[1:]))
