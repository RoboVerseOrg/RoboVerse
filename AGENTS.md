# AGENTS.md

This file defines repository-level development rules for people and AI agents working in this repo.

The goals are:
- keep the repo simple and easy to read
- prefer local consistency over clever abstractions
- follow existing project infrastructure instead of inventing parallel workflows

For testing infrastructure details, also refer to:
- [docs/source/metasim/developer_guide/autotest.md](./docs/source/metasim/developer_guide/autotest.md)
- https://roboverse.wiki/metasim/developer_guide/autotest
- [ENVIRONMENTS.md](./ENVIRONMENTS.md)

## General Development Rules

### Keep Changes Small And Local

- Prefer modifying existing files over adding new files.
- Do not introduce a new helper/module/doc file when an existing one already owns that responsibility.
- Do not add architectural layers or abstractions unless the current code clearly needs them.
- If a change can be expressed as a small local fix, prefer that over a framework-style refactor.

### Preserve Existing User-Facing Contracts

- Do not silently change public semantics when the repo already has established behavior.
- When broadening a base contract, make sure concrete subclasses actually satisfy it.
- If a behavior is only compatibility support, document it as such instead of treating it as a primary contract.

### Be Explicit About Scope

- Only test simulators that are related to or influenced by the current change.
- If the correct simulator env mapping is not known from repo context or user instructions, ask before running simulator-backed tests.
- Do not claim “all tests pass” unless the exact requested test commands were actually run in the correct envs.

### Prefer Fail-Fast Over Silent Degradation

- For unsupported backends or unsupported code paths, prefer an explicit error over silent success.
- Do not turn an explicit unsupported-operation failure into a silent no-op unless there is a strong reason.

## Testing Rules For `metasim/test`

### Source Of Truth

The authoritative test workflow is the MetaSim autotest guide:
- [docs/source/metasim/developer_guide/autotest.md](./docs/source/metasim/developer_guide/autotest.md)

When there is any doubt about pytest structure, markers, suite registration, handler reuse, or backend-specific commands, follow that guide.

### Test Discovery And File Format

For files under `metasim/test`:

- Test file names should start with `test_` or end with `_test.py`.
- Test function names should start with `test_`.
- Helper functions should not start with `test_`.
- Test directories that participate in the shared suite/registration system must contain `__init__.py`.
- If a suite uses the shared `handler` fixture, its package or file prefix must be registered through `register_shared_suite(...)` in the relevant `conftest.py`.

### Markers And Fixtures

Use the existing marker model described in the autotest guide:

- `@pytest.mark.mujoco`
- `@pytest.mark.isaacgym`
- `@pytest.mark.isaacsim`
- `@pytest.mark.newton`
- `@pytest.mark.mjx`
- `@pytest.mark.sim(...)`
- `@pytest.mark.general`

Rules:

- `@pytest.mark.general` tests should not request the `handler` fixture.
- Simulator-backed tests should use the repo’s shared fixture/registration system instead of custom one-off setup.
- Prefer extending an existing relevant test file under `metasim/test` before creating a new one.

### Backend-Specific Autotest Commands

When running tests for `metasim/test`, use these commands in the corresponding conda envs:

```bash
# MuJoCo only
pytest metasim/test/ -k mujoco

# IsaacGym (use entry script for proper import order)
python metasim/test/isaacgym_entry.py metasim/test/ -k isaacgym

# IsaacSim only
pytest metasim/test/ -k isaacsim

# Newton only
pytest metasim/test/ -k newton

# General tests (no simulator)
pytest metasim/test/ -k general
```

### Environment Correspondence

Read simulator-to-conda-env mapping from [ENVIRONMENTS.md](./ENVIRONMENTS.md).

### Test Selection Rule

- Only run a simulator’s tests when current changes affect that simulator or shared code used by it.
- For shared code changes, run the relevant backend tests one-by-one, not through improvised filtering logic unless necessary.
- Always consider whether `general` tests should also be run for the changed area.
- Run simulator-specific tests in an environment where the simulator can access a usable GPU when that backend expects or requires GPU execution.
- Do not claim simulator-backed tests were meaningfully exercised if the backend failed only because GPU access was unavailable.

### IsaacGym Special Rule

- For IsaacGym-backed tests, use:
  `python metasim/test/isaacgym_entry.py ...`
- Do not replace it with plain `pytest ... -k isaacgym`, because the import order is part of the required test harness.

### GPU Rule For Simulator Tests

- `isaacgym`, `isaacsim`, and `newton` test runs should be treated as GPU-backed simulator tests unless there is clear repo-specific evidence that a given run is intended to be CPU-only.
- Before relying on simulator-backed test results, verify that the selected env can actually see a GPU.
- If GPU access is missing, report that as an environment blocker instead of treating the run as a normal simulator test result.

## AI-Agent Rules

### Before Changing Code

- Read the relevant existing code paths first.
- Read the existing test file for the area before adding a new test.
- If the change touches testing infrastructure or simulator-specific behavior, check `docs/source/metasim/developer_guide/autotest.md`.

### Before Adding Files

- Ask: can this be done by extending an existing file cleanly?
- If yes, do that instead.
- New files should be the exception, not the default.

### Before Declaring Success

- Say exactly what was verified.
- If tests were blocked by environment problems, say so explicitly and include the real blocker.
- Do not compress environment/setup failures into “test failures” without explanation.
