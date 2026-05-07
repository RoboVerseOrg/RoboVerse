# RoboVerse Architecture Review & Improvement Roadmap

> **Document Version**: 1.0  
> **Last Updated**: January 2026  
> **Status**: Active Development

This document provides a comprehensive architecture review of the RoboVerse codebase and outlines a structured improvement roadmap. It is intended for core maintainers and contributors who want to understand the current state of the codebase and contribute to its improvement.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Identified Issues](#identified-issues)
4. [Improvement Roadmap](#improvement-roadmap)
5. [Detailed TODO List](#detailed-todo-list)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Strengths

RoboVerse is a well-architected multi-simulator robotics framework with several notable strengths:

| Aspect | Assessment | Details |
|--------|------------|---------|
| **Modularity** | Excellent | Clear separation between `metasim`, `roboverse_pack`, and `roboverse_learn` |
| **Simulator Abstraction** | Good | Unified interface across MuJoCo, IsaacSim, SAPIEN, PyBullet, Genesis |
| **Configuration System** | Good | Type-safe `@configclass` decorator with validation |
| **Domain Randomization** | Excellent | Comprehensive DR system with hybrid simulation support |
| **Documentation** | Good | Extensive tutorials and API documentation |

### Critical Issues Requiring Attention

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | State cache consistency bug | Data corruption | Low |
| P0 | Test coverage severely lacking | Reliability | High |
| P1 | Configuration system fragmentation | Usability | Medium |
| P1 | Environment creation interface inconsistency | Usability | Medium |
| P2 | Code quality issues | Maintainability | Low |

---

## Architecture Overview

### Module Structure

```
RoboVerse/
├── metasim/                 # Core simulation framework
│   ├── sim/                 # Simulator handlers (MuJoCo, Isaac, SAPIEN, etc.)
│   ├── scenario/            # Scene configuration (robots, objects, cameras)
│   ├── task/                # Task environment abstraction
│   ├── randomization/       # Domain randomization system
│   ├── queries/             # Extended query system (contacts, sensors)
│   └── utils/               # Utilities (configclass, math, state conversion)
│
├── roboverse_pack/          # Assets and task definitions
│   ├── robots/              # 50+ robot configurations
│   ├── tasks/               # 200+ task environments
│   ├── scenes/              # Scene configurations
│   └── queries/             # Custom query implementations
│
├── roboverse_learn/         # Learning algorithms
│   ├── il/                  # Imitation learning (ACT, Diffusion Policy, etc.)
│   ├── rl/                  # Reinforcement learning (PPO, TD3, SAC)
│   └── vla/                 # Vision-Language-Action models
│
└── generation/              # Asset generation and conversion tools
```

### Core Abstractions

```
┌─────────────────────────────────────────────────────────────────┐
│                        BaseSimHandler                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ MujocoHandler│  │IsaacHandler │  │SAPIENHandler│  ...        │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BaseTaskEnv                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │PickPlaceTask│  │LocomotionTask│  │ManipulationTask│  ...     │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ScenarioCfg                               │
│  robots[] + objects[] + cameras[] + lights[] + sim_params       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Identified Issues

### Issue 1: State Cache Consistency Bug (P0 - Critical)

**Location**: `metasim/sim/base.py:118-129`

**Problem**: The state cache is modified in-place when switching between `tensor` and `dict` modes, causing data corruption.

```python
# Current problematic implementation
def get_states(self, env_ids=None, mode="tensor"):
    if self._state_cache_expire:
        self._states = self._get_states(env_ids=env_ids)
        self._state_cache_expire = False
    # BUG: This overwrites the cache!
    if isinstance(self._states, TensorState) and mode == "dict":
        self._states = state_tensor_to_nested(self, self._states)
    elif isinstance(self._states, list) and mode == "tensor":
        self._states = list_state_to_tensor(self, self._states)
    return self._states
```

**Reproduction**:
```python
handler = create_handler(...)
states_t1 = handler.get_states(mode="tensor")  # TensorState
states_d = handler.get_states(mode="dict")     # Converts cache to dict
states_t2 = handler.get_states(mode="tensor")  # Returns dict, not tensor!
```

**Impact**: Silent data corruption in training pipelines that use both state formats.

**Solution**: See [TODO-001](#todo-001-fix-state-cache-consistency).

---

### Issue 2: Incomplete Abstract Method Declarations (P1)

**Location**: `metasim/sim/base.py:86-91`

**Problem**: `_set_dof_targets()` should be marked as `@abstractmethod` but is commented out.

```python
# @abstractmethod  # <-- This is commented out!
def _set_dof_targets(self, actions: list[Action]) -> None:
    raise NotImplementedError
```

**Impact**: 
- Static analysis tools cannot detect missing implementations
- IDE cannot provide proper warnings
- New simulator implementations may forget to implement this method

---

### Issue 3: Configuration System Fragmentation (P1)

**Problem**: Three different configuration systems are used across modules:

| Module | System | Tools |
|--------|--------|-------|
| `metasim` | `@configclass` | Python dataclass + custom wrapper |
| `roboverse_learn/il` | Hydra | YAML + OmegaConf |
| `roboverse_learn/rl` | Mixed | YAML + tyro + argparse |

**Impact**:
- Steep learning curve for contributors
- Configuration cannot be easily shared between modules
- Inconsistent user experience

---

### Issue 4: Test Coverage Severely Lacking (P0)

**Current Coverage Estimate**:

| Module | Test Files | Estimated Coverage |
|--------|------------|-------------------|
| `metasim` | 29 | ~60-70% |
| `roboverse_pack` | 0 | 0% |
| `roboverse_learn` | 1 | <5% |
| `generation` | 1 | ~20% |
| **Overall** | 31 | **~15-20%** |

**Critical Gaps**:
- 50+ robot configurations have no validation tests
- 200+ task environments have no tests
- All learning algorithms (ACT, Diffusion Policy, PPO, TD3) have no tests
- No code coverage reporting in CI

---

### Issue 5: Environment Creation Interface Inconsistency (P1)

**Problem**: Two different ways to create environments:

```python
# Method 1: Gymnasium style (used in clean_rl, vla)
from gymnasium import make_vec
env = make_vec("RoboVerse/task", robots=[...], simulator=sim)

# Method 2: Direct task class (used in fast_td3, rsl_rl, il)
from metasim.task.registry import get_task_class
env = get_task_class(task)(scenario)
```

**Impact**: Confusion for new users; inconsistent integration patterns.

---

### Issue 6: Code Quality Issues (P2)

| Issue | Location | Count |
|-------|----------|-------|
| Commented-out code | Multiple files | ~15 instances |
| Magic numbers | `sim/isaacsim/`, `sim/mujoco/` | ~20 instances |
| Missing type annotations | Various | ~100+ functions |
| Inconsistent error handling | All simulator handlers | Varies |

---

### Issue 7: Parallel Simulation Error Handling (P1)

**Location**: `metasim/sim/parallel.py:127-132`

**Problem**: `_check_error()` is only called at specific times, potentially missing subprocess errors.

```python
def _check_error(self):
    # Only checks for errors in error queue
    # May miss errors if not called at right time
```

**Impact**: Silent failures in parallel training runs.

---

## Improvement Roadmap

### Phase 1: Critical Fixes (Weeks 1-2)

```
┌────────────────────────────────────────────────────────┐
│ PHASE 1: Critical Fixes                                │
│                                                        │
│ □ TODO-001: Fix state cache consistency                │
│ □ TODO-002: Add @abstractmethod decorators             │
│ □ TODO-003: Add robot configuration validation tests   │
│ □ TODO-004: Integrate pytest-cov in CI                 │
└────────────────────────────────────────────────────────┘
```

### Phase 2: Test Coverage (Weeks 3-6)

```
┌────────────────────────────────────────────────────────┐
│ PHASE 2: Test Coverage Improvement                     │
│                                                        │
│ □ TODO-005: Add task environment integration tests     │
│ □ TODO-006: Add learning algorithm unit tests          │
│ □ TODO-007: Add state conversion tests                 │
│ □ TODO-008: Add domain randomization tests             │
└────────────────────────────────────────────────────────┘
```

### Phase 3: Interface Unification (Weeks 7-10)

```
┌────────────────────────────────────────────────────────┐
│ PHASE 3: Interface Unification                         │
│                                                        │
│ □ TODO-009: Create unified environment factory         │
│ □ TODO-010: Standardize configuration loading          │
│ □ TODO-011: Unify error handling across simulators     │
│ □ TODO-012: Add deprecation warnings for old APIs      │
└────────────────────────────────────────────────────────┘
```

### Phase 4: Code Quality (Weeks 11-14)

```
┌────────────────────────────────────────────────────────┐
│ PHASE 4: Code Quality                                  │
│                                                        │
│ □ TODO-013: Remove commented-out code                  │
│ □ TODO-014: Extract magic numbers to constants         │
│ □ TODO-015: Add comprehensive type annotations         │
│ □ TODO-016: Add performance benchmarks                 │
└────────────────────────────────────────────────────────┘
```

### Phase 5: Architecture Evolution (Long-term)

```
┌────────────────────────────────────────────────────────┐
│ PHASE 5: Architecture Evolution                        │
│                                                        │
│ □ TODO-017: Implement plugin architecture for sims     │
│ □ TODO-018: Create unified configuration system        │
│ □ TODO-019: Add async simulation support               │
│ □ TODO-020: Performance profiling integration          │
└────────────────────────────────────────────────────────┘
```

---

## Detailed TODO List

### TODO-001: Fix State Cache Consistency

**Priority**: P0 (Critical)  
**Effort**: Low (1-2 days)  
**Risk**: Low  

**Description**: Fix the state cache to maintain separate caches for tensor and dict formats.

**Implementation**:

```python
# metasim/sim/base.py

class BaseSimHandler(ABC):
    def __init__(self, scenario, optional_queries=None):
        # ... existing code ...
        self._state_cache_expire = True
        self._tensor_state_cache: TensorState | None = None
        self._dict_state_cache: list[DictEnvState] | None = None

    def _invalidate_cache(self) -> None:
        """Invalidate all state caches."""
        self._state_cache_expire = True
        self._tensor_state_cache = None
        self._dict_state_cache = None

    def set_states(self, states, env_ids=None) -> None:
        """Set states and invalidate cache."""
        self._invalidate_cache()
        self._set_states(states, env_ids)

    def get_states(
        self, 
        env_ids: list[int] | None = None, 
        mode: Literal["tensor", "dict"] = "tensor"
    ) -> TensorState | list[DictEnvState]:
        """Get states with independent caching for each mode."""
        if self._state_cache_expire:
            self._tensor_state_cache = self._get_states(env_ids=env_ids)
            self._dict_state_cache = None  # Lazy conversion
            self._state_cache_expire = False

        if mode == "tensor":
            return self._tensor_state_cache
        else:
            if self._dict_state_cache is None:
                self._dict_state_cache = state_tensor_to_nested(
                    self, self._tensor_state_cache
                )
            return self._dict_state_cache
```

**Test Case**:

```python
# metasim/test/sim/test_state_cache.py

@pytest.mark.general
def test_state_cache_mode_independence():
    """Verify that switching modes doesn't corrupt cache."""
    handler = create_test_handler()
    handler.launch()
    
    # Get tensor state
    states_t1 = handler.get_states(mode="tensor")
    assert isinstance(states_t1, TensorState)
    
    # Get dict state (should not affect tensor cache)
    states_d = handler.get_states(mode="dict")
    assert isinstance(states_d, list)
    
    # Get tensor state again (should return same type)
    states_t2 = handler.get_states(mode="tensor")
    assert isinstance(states_t2, TensorState)
    
    # Values should match
    assert torch.allclose(states_t1.pos, states_t2.pos)
```

**Acceptance Criteria**:
- [ ] Test case passes
- [ ] Existing tests still pass
- [ ] No breaking changes to public API

---

### TODO-002: Add @abstractmethod Decorators

**Priority**: P1  
**Effort**: Very Low (1 hour)  
**Risk**: Very Low  

**Files to modify**:
- `metasim/sim/base.py`

**Changes**:

```python
# Before
# @abstractmethod
def _set_dof_targets(self, actions: list[Action]) -> None:
    raise NotImplementedError

# After
@abstractmethod
def _set_dof_targets(self, actions: list[Action]) -> None:
    """Set DOF targets. Subclasses must implement this method."""
    raise NotImplementedError
```

**Verification**:
```bash
# Run mypy to verify abstract method detection
mypy metasim/sim/base.py
```

---

### TODO-003: Add Robot Configuration Validation Tests

**Priority**: P0  
**Effort**: Medium (2-3 days)  
**Risk**: Low  

**Implementation**:

```python
# metasim/test/test_robot_configs.py

import pytest
from pathlib import Path
import importlib
import pkgutil

def get_all_robot_configs():
    """Dynamically discover all robot configuration classes."""
    import roboverse_pack.robots as robots_module
    
    configs = []
    for importer, modname, ispkg in pkgutil.iter_modules(robots_module.__path__):
        if modname.endswith('_cfg'):
            module = importlib.import_module(f'roboverse_pack.robots.{modname}')
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    hasattr(obj, 'name') and 
                    name.endswith('Cfg')):
                    configs.append(obj)
    return configs

ALL_ROBOT_CONFIGS = get_all_robot_configs()

@pytest.mark.general
@pytest.mark.parametrize("robot_cfg_cls", ALL_ROBOT_CONFIGS, 
                         ids=lambda x: x.__name__)
def test_robot_config_instantiation(robot_cfg_cls):
    """Verify robot config can be instantiated."""
    cfg = robot_cfg_cls()
    assert cfg.name is not None
    assert isinstance(cfg.name, str)
    assert len(cfg.name) > 0

@pytest.mark.general
@pytest.mark.parametrize("robot_cfg_cls", ALL_ROBOT_CONFIGS,
                         ids=lambda x: x.__name__)
def test_robot_config_has_asset_path(robot_cfg_cls):
    """Verify robot config has at least one asset path."""
    cfg = robot_cfg_cls()
    
    asset_paths = [
        getattr(cfg, 'usd_path', None),
        getattr(cfg, 'urdf_path', None),
        getattr(cfg, 'mjcf_path', None),
    ]
    
    valid_paths = [p for p in asset_paths if p is not None and len(p) > 0]
    assert len(valid_paths) > 0, f"{cfg.name} has no valid asset path"

@pytest.mark.general
@pytest.mark.parametrize("robot_cfg_cls", ALL_ROBOT_CONFIGS,
                         ids=lambda x: x.__name__)
def test_robot_config_has_actuators(robot_cfg_cls):
    """Verify robot config has actuator definitions."""
    cfg = robot_cfg_cls()
    
    if hasattr(cfg, 'actuators'):
        assert len(cfg.actuators) > 0, f"{cfg.name} has no actuators defined"

@pytest.mark.general
@pytest.mark.parametrize("robot_cfg_cls", ALL_ROBOT_CONFIGS,
                         ids=lambda x: x.__name__)
def test_robot_config_joint_limits_valid(robot_cfg_cls):
    """Verify joint limits are valid (lower < upper)."""
    cfg = robot_cfg_cls()
    
    if hasattr(cfg, 'joint_limits'):
        for joint_name, (lower, upper) in cfg.joint_limits.items():
            assert lower < upper, \
                f"{cfg.name}.{joint_name}: lower ({lower}) >= upper ({upper})"
```

---

### TODO-004: Integrate pytest-cov in CI

**Priority**: P0  
**Effort**: Low (1 day)  
**Risk**: Very Low  

**Changes to `.github/workflows/premerge-ci.yml`**:

```yaml
# Add coverage reporting step
- name: Run tests with coverage
  run: |
    pytest metasim/test \
      --cov=metasim \
      --cov-report=xml \
      --cov-report=html \
      --cov-fail-under=30 \
      -k ${{ matrix.test_type }}

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: ${{ matrix.test_type }}
    fail_ci_if_error: false
```

**Add `pyproject.toml` configuration**:

```toml
[tool.coverage.run]
source = ["metasim"]
omit = ["metasim/test/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

---

### TODO-005: Add Task Environment Integration Tests

**Priority**: P1  
**Effort**: High (1 week)  
**Risk**: Medium  

**Implementation**:

```python
# metasim/test/test_task_environments.py

import pytest
from metasim.task.registry import get_task_class, TASK_REGISTRY

# Select representative tasks for testing
CORE_TASKS = [
    "pick_cube",
    "place_cube", 
    "open_drawer",
    "close_drawer",
    "push_button",
]

@pytest.mark.mujoco
@pytest.mark.parametrize("task_name", CORE_TASKS)
def test_task_reset_step_mujoco(task_name):
    """Test basic reset/step cycle for core tasks on MuJoCo."""
    task_cls = get_task_class(task_name)
    scenario = task_cls.scenario.copy()
    scenario.update(
        simulator="mujoco",
        num_envs=1,
        headless=True,
    )
    
    env = task_cls(scenario, device="cpu")
    env.launch()
    
    try:
        # Test reset
        obs, info = env.reset()
        assert obs is not None
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float, torch.Tensor))
        assert isinstance(terminated, (bool, torch.Tensor))
        assert isinstance(truncated, (bool, torch.Tensor))
    finally:
        env.close()
```

---

### TODO-006: Add Learning Algorithm Unit Tests

**Priority**: P1  
**Effort**: High (1-2 weeks)  
**Risk**: Medium  

**Example for Diffusion Policy**:

```python
# roboverse_learn/il/tests/test_diffusion_policy.py

import pytest
import torch
from roboverse_learn.il.policies.dp.ddpm_dit_image_policy import DDPMDiTImagePolicy

@pytest.fixture
def sample_obs():
    """Create sample observation for testing."""
    return {
        "image": torch.randn(1, 3, 224, 224),
        "agent_pos": torch.randn(1, 7),
    }

def test_diffusion_policy_forward():
    """Test forward pass of diffusion policy."""
    policy = DDPMDiTImagePolicy(
        obs_dim=7,
        action_dim=7,
        horizon=16,
        # ... minimal config
    )
    
    obs = sample_obs()
    action = policy.predict_action(obs)
    
    assert action.shape == (1, 16, 7)  # (batch, horizon, action_dim)

def test_diffusion_policy_training_step():
    """Test single training step."""
    policy = DDPMDiTImagePolicy(...)
    optimizer = torch.optim.Adam(policy.parameters())
    
    batch = create_training_batch()
    loss = policy.compute_loss(batch)
    
    assert loss.requires_grad
    loss.backward()
    optimizer.step()
```

---

### TODO-009: Create Unified Environment Factory

**Priority**: P1  
**Effort**: Medium (3-5 days)  
**Risk**: Medium  

**Implementation**:

```python
# metasim/env_factory.py

from typing import Union, List, Optional
from metasim.scenario.robot import RobotCfg

def make_env(
    task: str,
    robots: Optional[List[Union[str, RobotCfg]]] = None,
    simulator: str = "mujoco",
    num_envs: int = 1,
    headless: bool = True,
    device: str = "cuda",
    cameras: Optional[List] = None,
    **kwargs
):
    """
    Unified environment factory for RoboVerse.
    
    This is the recommended way to create environments.
    
    Args:
        task: Task name (e.g., "pick_cube", "locomotion_walk")
        robots: List of robot names or RobotCfg instances
        simulator: Simulator backend ("mujoco", "isaacsim", "sapien3", etc.)
        num_envs: Number of parallel environments
        headless: Whether to run in headless mode
        device: Device for tensor computations
        cameras: Camera configurations for observations
        **kwargs: Additional task-specific arguments
    
    Returns:
        BaseTaskEnv: Configured environment instance
    
    Example:
        >>> env = make_env(
        ...     task="pick_cube",
        ...     robots=["franka"],
        ...     simulator="mujoco",
        ...     num_envs=16,
        ... )
        >>> obs, info = env.reset()
        >>> obs, reward, term, trunc, info = env.step(action)
    """
    from metasim.task.registry import get_task_class
    from metasim.utils.setup_util import get_robot
    
    # Resolve task class
    task_cls = get_task_class(task)
    
    # Build scenario from task default
    scenario = task_cls.scenario.copy()
    
    # Resolve robots
    if robots is not None:
        resolved_robots = []
        for robot in robots:
            if isinstance(robot, str):
                resolved_robots.append(get_robot(robot))
            else:
                resolved_robots.append(robot)
        scenario.robots = resolved_robots
    
    # Apply overrides
    scenario.update(
        simulator=simulator,
        num_envs=num_envs,
        headless=headless,
        cameras=cameras or [],
    )
    
    # Create and return environment
    env = task_cls(scenario, device=device, **kwargs)
    env.launch()
    
    return env

# Also register with gymnasium for compatibility
def register_gymnasium_envs():
    """Register all RoboVerse tasks with Gymnasium."""
    import gymnasium
    from metasim.task.registry import TASK_REGISTRY
    
    for task_name in TASK_REGISTRY:
        gymnasium.register(
            id=f"RoboVerse/{task_name}",
            entry_point="metasim.env_factory:make_env",
            kwargs={"task": task_name},
        )
```

---

### TODO-013: Remove Commented-Out Code

**Priority**: P2  
**Effort**: Very Low (2 hours)  
**Risk**: Very Low  

**Files to clean**:

| File | Line | Content |
|------|------|---------|
| `metasim/sim/base.py` | 18 | `# from metasim.utils.hf_util import FileDownloader` |
| `metasim/sim/base.py` | 36 | `# FileDownloader(scenario).do_it()` |
| `metasim/sim/base.py` | 86 | `# @abstractmethod` |
| `metasim/scenario/scenario.py` | 72 | Various commented code |

**Approach**: 
1. Search for `# ` patterns followed by code
2. Review each instance
3. Either remove or convert to proper TODO comment

---

### TODO-014: Extract Magic Numbers to Constants

**Priority**: P2  
**Effort**: Low (1-2 days)  
**Risk**: Low  

**Create constants file**:

```python
# metasim/constants.py

"""Framework-wide constants."""

# Simulation defaults
DEFAULT_DT = 0.015  # 15ms physics timestep
DEFAULT_DECIMATION = 2
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

# Cache settings
STATE_CACHE_SIZE = 1000
MAX_PARALLEL_ENVS = 4096

# Timeouts
DEFAULT_LAUNCH_TIMEOUT = 30.0
DEFAULT_STEP_TIMEOUT = 5.0

# Numerical tolerances
POSITION_TOLERANCE = 1e-5
ROTATION_TOLERANCE = 1e-4
```

---

## Implementation Guidelines

### Safe Modification Protocol

Before making any changes, follow this checklist:

```
□ Pre-modification
  ├── □ Read existing tests for the module
  ├── □ Run existing tests (ensure they pass)
  ├── □ Understand the change's impact scope
  └── □ Check for downstream dependencies

□ Implementation
  ├── □ Write tests first (TDD preferred)
  ├── □ Make small, focused commits
  ├── □ Maintain backward compatibility
  └── □ Add deprecation warnings if needed

□ Post-modification
  ├── □ Run full test suite locally
  ├── □ Check for linter errors
  ├── □ Update documentation if needed
  └── □ Create detailed PR description
```

### Feature Flag Pattern

For high-risk changes, use feature flags:

```python
# metasim/config.py

class FeatureFlags:
    """Feature flags for gradual rollout of changes."""
    
    # State cache v2 with independent tensor/dict caches
    USE_INDEPENDENT_STATE_CACHE = False
    
    # New unified environment factory
    USE_UNIFIED_ENV_FACTORY = False
    
    # Strict type checking in configs
    STRICT_CONFIG_VALIDATION = False
```

### Deprecation Pattern

```python
import warnings
from functools import wraps

def deprecated(message: str, removal_version: str = "0.3.0"):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {message}. "
                f"Will be removed in version {removal_version}.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@deprecated("Use make_env() instead", removal_version="0.4.0")
def create_environment_legacy(...):
    ...
```

---

## Testing Strategy

### Test Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    End-to-End Tests                         │
│              (Full training pipeline tests)                 │
│                      ~5% of tests                           │
├─────────────────────────────────────────────────────────────┤
│                   Integration Tests                         │
│         (Task reset/step, multi-env, rendering)             │
│                     ~25% of tests                           │
├─────────────────────────────────────────────────────────────┤
│                      Unit Tests                             │
│    (Individual functions, state conversion, configs)        │
│                     ~70% of tests                           │
└─────────────────────────────────────────────────────────────┘
```

### Running Tests Locally

```bash
# Run all general tests (no simulator required)
pytest metasim/test -k general -v

# Run MuJoCo tests
pytest metasim/test -k mujoco -v

# Run with coverage
pytest metasim/test --cov=metasim --cov-report=html

# Run specific test file
pytest metasim/test/sim/test_state_cache.py -v
```

### Coverage Targets

| Phase | Target Coverage | Timeline |
|-------|-----------------|----------|
| Current | ~15-20% | - |
| Phase 1 | 30% | Week 2 |
| Phase 2 | 50% | Week 6 |
| Phase 3 | 60% | Week 10 |
| Long-term | 70%+ | Week 14+ |

---

## Appendix: Quick Reference

### Priority Definitions

| Priority | Definition | Response Time |
|----------|------------|---------------|
| P0 | Critical - blocks users or causes data corruption | Immediate |
| P1 | High - significant usability or reliability issue | 1-2 weeks |
| P2 | Medium - code quality or maintainability | 2-4 weeks |
| P3 | Low - nice to have improvements | Backlog |

### Effort Definitions

| Effort | Time Estimate | Description |
|--------|---------------|-------------|
| Very Low | < 4 hours | Single file, obvious change |
| Low | 1-2 days | Few files, clear scope |
| Medium | 3-5 days | Multiple files, some complexity |
| High | 1-2 weeks | Significant refactoring |
| Very High | 2+ weeks | Architecture-level changes |

---

## Contributing

If you want to contribute to any of these improvements:

1. Comment on the relevant GitHub issue (or create one)
2. Follow the [Safe Modification Protocol](#safe-modification-protocol)
3. Start with lower-risk items to familiarize yourself with the codebase
4. Ask questions in discussions if anything is unclear

Recommended starting points for new contributors:
- TODO-002: Add @abstractmethod decorators (Very Low effort)
- TODO-013: Remove commented-out code (Very Low effort)
- TODO-003: Add robot configuration validation tests (Medium effort, high impact)
