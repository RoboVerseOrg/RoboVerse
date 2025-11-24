"""
Centralized conftest for test suite.
Handles isaacgym import before any test modules are loaded.
"""


def pytest_configure(config):
    """Called after command line options have been parsed and all plugins and initial conftest files been loaded."""
    # Import isaacgym early to handle ImportError gracefully
    # This runs before test collection, so it affects all test modules
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        pass


# Also import at module level for direct imports (non-pytest usage)
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
