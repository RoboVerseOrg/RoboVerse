### How to write and run tests

#### Test Discovery
Pytest **(by default)** automatically discovers and runs **test functions** that meet ALL of the following criteria:

* Function name starts with `test_` (e.g., `test_contact_forces_mujoco()`) OR method name starts with `test_` in classes named `Test*` (that do not define `__init__`)
* Located in files whose names start with `test_` or end with `_test.py` (e.g., `test_site.py`, `test_contact_force.py`)
* Within the directories pytest is invoked on (e.g., the current directory or paths given on the command line) and their subdirectories

**Note**: Not all functions in test files are run—only those whose names start with `test_`. Helper functions like `_pick_robot_site_name()` or `contact_forces_mujoco_query()` are not executed as tests by pytest.


#### Markers and Fixtures
- **Markers**: Use `@pytest.mark.isaacsim`, `isaacgym`, `mujoco`, `mjx`, or `@pytest.mark.sim("sim1","sim2")` to declare which backends a test needs. Use `@pytest.mark.general` for tests that need no simulator/handler.
- **Fixtures**:
  - `handler`: for simulator-backed tests. When present, the test body is executed in the child process with the handler instance passed in.
  - **Important**: Tests marked `@pytest.mark.general` should **NOT** request `handler`. General tests are for pure unit tests without simulators.
- **Param selection**: Marker declarations + `get_test_parameters` determine which `(sim,num_envs)` combos get applied. `-k` still filters collected tests by substring after parametrization.

#### Registering Your Test Suite with Shared Handler

The `handler` fixture provides **significant performance benefits** by reusing simulator instances across multiple tests. Instead of creating and destroying a handler for every test, it creates one handler per `(sim, num_envs)` combination and reuses it for all tests in a session that share the same scenario configuration.

**Best Practice: Group tests by scenario**
- **Why**: Tests that share the same scenario (robot configuration, environment setup, etc.) can reuse the same handler process, dramatically reducing test execution time.
- **How**: Organize all tests that need the same scenario into a single directory (or file), then register one scenario builder for that entire directory using `register_shared_suite()`.
- **Example**: The `queries/` test suite has tests for contact forces, site positions, etc. All these tests use the same robot and environment setup, so they all share one handler per `(sim, num_envs)` combo.

**Directory Structure Example:**
```
metasim/test/
├── conftest.py                    # Central shared handler machinery
├── test_utils.py                  # get_test_parameters()
│
├── queries/                       # Suite 1: All query-related tests (ONE scenario)
│   ├── conftest.py               # Register ONE scenario for all query tests
│   ├── test_contact_force.py     # Tests using G1 robot scenario
│   ├── test_site.py              # Tests using G1 robot scenario
│
├── manipulation/                  # Suite 2: Manipulation tests (ONE scenario)
│   ├── conftest.py               # Register ONE scenario for manipulation tests
│   ├── test_gripper.py           # Tests using gripper robot scenario
│   └── test_grasping.py          # Tests using gripper robot scenario
│
└── locomotion/                    # Suite 3: Locomotion tests (MULTIPLE scenarios + general tests)
    ├── conftest.py               # Register TWO scenarios: walking + running
    ├── test_walking.py           # Tests using bipedal walking scenario
    ├── test_running.py           # Tests using quadruped running scenario
    └── test_locomotion_general.py           # General tests (no handler - conftest has no effect)
```

**Key Points**:
- **Single scenario per suite**: `queries/` and `manipulation/` each register ONE scenario for all tests in that directory - maximizes handler reuse and minimizes startup overhead.
- **Multiple scenarios in one suite**: `locomotion/` registers TWO scenarios because walking tests need a bipedal robot while running tests need a quadruped robot. Each scenario is registered with a different prefix matching the specific test file.
- **General tests can coexist**: `locomotion/test_locomotion_general.py` contains tests marked `@pytest.mark.general` that don't request the `handler` fixture. The `conftest.py` registrations have no effect on these tests since they don't use the handler.

**Example: Registering the `manipulation/` test suite**

**In `metasim/test/manipulation/conftest.py`:**
```python
from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import register_shared_suite
from roboverse_pack.robots.panda_cfg import PandaCfg

def get_manipulation_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Build scenario for all manipulation tests.

    All tests in metasim/test/manipulation/ will share this scenario,
    so we create ONE handler per (sim, num_envs) that handles gripper
    tests, grasping tests, pick-and-place tests, etc.
    """
    return ScenarioCfg(
        robots=[PandaCfg()],  # Gripper robot for manipulation
        objects=[],
        num_envs=num_envs,
        simulator=sim,
        headless=True,
        # ... other config
    )

# Register this scenario for ALL tests in the manipulation/ directory
register_shared_suite("metasim.test.manipulation", get_manipulation_scenario)
```

**In `metasim/test/manipulation/test_gripper.py`:**
```python
import pytest

@pytest.mark.mujoco
def test_gripper_open_close(handler):
    """Test gripper opening and closing."""
    # Runs in the child process; this arg is the handler instance
    assert handler.scenario.simulator == "mujoco"

@pytest.mark.sim("mujoco", "isaacsim")
def test_gripper_force_limits(handler):
    """Test gripper force limits on multiple backends."""
    assert handler.scenario.num_envs >= 1
```

**In `metasim/test/manipulation/test_grasping.py`:**
```python
import pytest

@pytest.mark.isaacsim
def test_grasp_cube(handler):
    """Test grasping a cube object."""
    # Reuses the same handler as other manipulation tests for isaacsim
    assert handler.scenario.simulator == "isaacsim"
```

**How it works:**
1. The central [conftest.py](metasim/test/conftest.py) finds your registration by matching the `"metasim.test.manipulation"` prefix to all test modules in that directory
2. When a test in your suite requests `handler`, pytest parametrizes it based on the test's markers (e.g., `@pytest.mark.mujoco`)
3. Your `get_manipulation_scenario(sim, num_envs)` is called **once** to build the scenario for each `(sim, num_envs)` combination
4. The handler is **reused** across all tests in `manipulation/` that have the same `(sim, num_envs)` - dramatically faster than creating a new handler for each test
5. The test body runs inside the child process automatically; the handler instance is passed as the argument tied to the `handler` fixture.

**Important**: Do not request `handler` on `@pytest.mark.general` tests - those are for pure unit tests without simulators.

---

**Example: Registering multiple scenarios in `locomotion/` suite**

When tests in the same directory need different robot configurations, register multiple scenarios with different prefixes:

**In `metasim/test/locomotion/conftest.py`:**
```python
from metasim.scenario.scenario import ScenarioCfg
from metasim.test.conftest import register_shared_suite
from roboverse_pack.robots.bipedal_cfg import BipedalCfg
from roboverse_pack.robots.quadruped_cfg import QuadrupedCfg

def get_walking_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Build scenario for bipedal walking tests."""
    return ScenarioCfg(
        robots=[BipedalCfg()],  # Bipedal robot for walking
        num_envs=num_envs,
        simulator=sim,
        headless=True,
    )

def get_running_scenario(sim: str, num_envs: int) -> ScenarioCfg:
    """Build scenario for quadruped running tests."""
    return ScenarioCfg(
        robots=[QuadrupedCfg()],  # Quadruped robot for running
        num_envs=num_envs,
        simulator=sim,
        headless=True,
    )

# Register TWO scenarios with file-specific prefixes
register_shared_suite("metasim.test.locomotion.test_walking", get_walking_scenario)
register_shared_suite("metasim.test.locomotion.test_running", get_running_scenario)
```

**In `metasim/test/locomotion/test_walking.py`:**
```python
import pytest

@pytest.mark.mujoco
def test_bipedal_gait(handler):
    """Test bipedal walking gait."""
    # Uses BipedalCfg scenario
    assert len(handler.scenario.robots) == 1

@pytest.mark.isaacsim
def test_balance_control(handler):
    """Test balance during walking."""
    # Reuses the same bipedal handler for isaacsim
    assert handler.scenario.simulator == "isaacsim"
```

**In `metasim/test/locomotion/test_running.py`:**
```python
import pytest

@pytest.mark.mujoco
def test_quadruped_trot(handler):
    """Test quadruped trotting gait."""
    # Uses QuadrupedCfg scenario (different from walking!)
    assert len(handler.scenario.robots) == 1

@pytest.mark.isaacsim
def test_high_speed_stability(handler):
    """Test stability at high running speeds."""
    # Reuses the same quadruped handler for isaacsim
    assert handler.scenario.simulator == "isaacsim"
```

The key difference: `test_walking.py` gets the bipedal scenario, `test_running.py` gets the quadruped scenario, because their full module paths match different registration prefixes.

---

**Example: General tests (no simulator required)**

**In `metasim/test/locomotion/test_locomotion_general.py`:**
```python
import pytest
from metasim.utils import some_pure_function

@pytest.mark.general
def test_pure_math():
    """Test a pure function that doesn't need any simulator."""
    # NO handler fixture requested!
    result = some_pure_function(2, 3)
    assert result == 5

@pytest.mark.general
def test_config_validation():
    """Test configuration validation logic."""
    # NO handler fixture requested!
    from metasim.scenario.scenario import ScenarioCfg

    # This just tests the config class itself, no simulator needed
    cfg = ScenarioCfg(robots=[], num_envs=1, simulator="mujoco")
    assert cfg.num_envs == 1
    assert cfg.simulator == "mujoco"

@pytest.mark.general
def test_string_parsing():
    """Test string parsing utilities."""
    # NO handler fixture requested!
    from metasim.utils import parse_robot_name

    assert parse_robot_name("robot/link_1") == "robot"
```

**Key points**:
- Tests marked `@pytest.mark.general` should NEVER request the `handler` fixture. They're for testing pure Python logic, utilities, and classes without needing a simulator.
- General tests can be placed in any subdirectory alongside simulator tests. The `conftest.py` scenario registrations have no effect on them since they don't use the `handler` fixture.

#### Examples
- Run MuJoCo-only tests: `pytest <folder_or_file> -k mujoco`
- Run isaacgym tests ensuring import order: `python metasim/test/isaacgym_entry.py <folder_or_file> -k isaacgym`
- General tests (no sim): `pytest <folder_or_file> -k general`

