import sys

import isaacgym  # noqa: F401  # must be imported before torch
import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
