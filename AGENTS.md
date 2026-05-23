# AGENTS.md

This file defines repository-level development rules for people and AI coding agents working in
the **RoboVerse** content repo (`roboverse-py`). It is read automatically by Claude Code, Codex,
Cursor, and similar agents at the start of a session.

RoboVerse is the **downstream** half of a two-repo system:

- **MetaSim** (`metasim`, sibling repo) owns the core simulator abstractions, scenario config types,
  task registry, package discovery, and simulator backends. Its rules live in the MetaSim
  [`AGENTS.md`](../MetaSim/AGENTS.md) — follow that file for anything touching core, backends, or the
  `metasim/test` suite.
- **RoboVerse** (this repo) owns tasks, robots, scenes, grounds, assets, learning code
  (`roboverse_learn`), examples (`get_started`), tooling, and reports. It depends on MetaSim via
  `metasim @ git+https://github.com/RoboVerseOrg/MetaSim.git`.

The goals here are:

- keep RoboVerse easy to extend with new tasks / robots / sims, with a low on-ramp for contributors;
- protect **multi-simulator parity** — correctness across backends is load-bearing, not a nice-to-have;
- keep public APIs stable so RoboVerse can become a dependable standard library;
- prefer local consistency and existing infrastructure over clever new abstractions.

For details that this file points to:

- Contributor setup: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- Priorities and version targets: [`ROADMAP.md`](./ROADMAP.md)
- Tutorials, task migration, API: <https://roboverse.wiki/metasim/>
- Simulator → environment mapping: MetaSim [`ENVIRONMENTS.md`](../MetaSim/ENVIRONMENTS.md)

---

## Multi-Repo Workflow

- The two repos are sibling directories. Install both editable
  (`python -m pip install -e ".[dev,mujoco]"` in RoboVerse), with `metasim` resolved from the local
  MetaSim checkout when developing cross-cutting features.
- **Know which repo owns the change before editing.** A simulator-backend bug, a scenario-config
  type, or the task registry is a *MetaSim* change. A new task/robot/scene, a reward, a learning
  script, or an example is a *RoboVerse* change.
- For a feature that spans both, land (or at least draft) the **MetaSim change first**, then the
  RoboVerse change that consumes it. Do not work around a missing core capability by duplicating it
  downstream.
- Do not push core logic into MetaSim that is really RoboVerse content, and do not fork core types
  into RoboVerse. Use MetaSim's package-discovery mechanisms (`metasim.toml`, entry points,
  `METASIM_*_PACKAGES`) to register downstream content.

## General Workflow

- **Orient first.** When unfamiliar with an area, check the docs (<https://roboverse.wiki>,
  `docs/source/`), `ROADMAP.md`, and the relevant `roboverse_pack/` subtree before searching source
  from scratch.
- **Pre-commit is the gate.** Install once with `pre-commit install`. Before opening a PR, run
  `pre-commit run --all-files` and resolve errors.
- **Semantic commits.** Use [Conventional Commits](https://www.conventionalcommits.org/):
  `<type>(<scope>): <description>`, types `feat|fix|docs|style|refactor|test|chore`, scope = pack or
  module (e.g. `feat(tasks): add mjlab cartpole balance`, `fix(robots): correct go1 base-vel obs`).
- Keep commits and pull requests focused, reviewable, and scoped to one concern.

## Parity Is Load-Bearing (RoboVerse-Specific)

RoboVerse's value is that the same task behaves consistently across MuJoCo, Newton, SAPIEN3,
PyBullet, IsaacSim, etc. Treat parity as a correctness contract:

- **Run the task end-to-end first; chase numerical parity second.** A task that "matches" only
  because both sides are equally broken (e.g. both fall into the void) is not parity. See the parity
  harness pattern in `scripts/parity_obs_reward_cartpole.py` and `scripts/eval_*_cross_sim.py`.
- When porting a reward / observation / dynamics from another framework (mjlab, ManiSkill, …), aim
  for **bitwise or machine-eps agreement** on obs and reward, and verify with an actual cross-sim
  comparison rather than asserting it.
- **Closed-loop dynamics parity ≠ obs-bitwise parity.** A policy that trains on one backend may not
  transfer to another even when observations match; report which backend a trained demo actually
  ran on instead of implying transfer.
- Reports must show pain points honestly. Do not present a clean number that hides a real failure.

## Adding Content (Tasks / Robots / Scenes)

- Follow the task-migration developer guide: <https://roboverse.wiki/metasim/developer_guide/new_task>.
- Tasks/robots/scenes/grounds live under `roboverse_pack/`. Configs use the `@configclass` dataclass
  pattern from `metasim.utils`; prefer composing existing Cfg types over inventing parallel ones.
- Prefer extending an existing task family over a new top-level scaffold when the new task is a
  variant. Keep additions additive — do not modify unrelated legacy task files to make a new one work.
- Register learning entry points and example usage where the existing ones live
  (`roboverse_learn/{rl,il,vla}`, `get_started/`); don't scatter new top-level scripts.

## Design Principles

- Be a library, not a framework. Keep public APIs small, orthogonal, and composable.
- Duplication is cheaper than the wrong abstraction. Don't add layers the code doesn't yet need.
- Make it work, make it right, then make it fast — measure before optimizing.
- Single source of truth: centralize shared constants, configs, and logic; don't fork them per pack.
- Validate and normalize at boundaries (CLI, config, data loaders). Fail fast on invalid states with
  clear, actionable errors; never fail silently or turn an unsupported path into a quiet no-op.
- Favor composition over inheritance. Keep side effects explicit and localized.
- Optimize for readability first; future maintainers and downstream users are users too.

## API Design Guidelines

- **Prefer keyword-only arguments** over long positional lists, so the API can grow without breaking
  call sites: `def attention(*, query, key, value, query_mask, kv_mask): ...`. Obvious operators
  (`matmul(a, b)`) may stay positional.
- **Return a dataclass / `@configclass`, not a dynamically-sized tuple.** New return fields should not
  break existing callers. A well-known operator signature (e.g. `lstm` → `(output, h, c)`) is the
  exception.
- For composite configs, hold sub-module `.Config` objects rather than flattening their fields, so
  sub-modules can be swapped without changing the composite surface.
- Keep public APIs stable. If you must break one, document a migration path.

## Code Style

RoboVerse uses **ruff** (lint + format) and **pre-commit**. The authoritative config is in
`pyproject.toml` — do not invent rules that contradict it.

### Python

- **Double quotes** (ruff default — match this repo even if a sibling repo uses single quotes).
- **Line length 120**; `target-version = "py38"`, so keep syntax 3.8-compatible.
- **Google-style docstrings** (`pydocstyle convention = "google"`).
- Use `from __future__ import annotations` (the codebase relies on string annotations and `FA`).
- **Local / lazy imports are allowed here** (`E402`, `PLC0415` are intentionally ignored) because
  heavy optional simulator backends (isaacgym, isaacsim, newton) and the `try: import isaacgym`
  ordering shim must be imported lazily. Prefer top-level imports for stdlib and hard deps, but do
  not "fix" an existing deliberate lazy/ordered import.
- Don't add global `noqa`/ignore rules to `pyproject.toml`; use per-line `# noqa: CODE` for local
  exceptions. The existing per-file docstring (`"D"`) ignores are tracked debt — when you add a
  docstring, you may remove the matching `FIXME` ignore.

### Commands

- Prefer `python -m <module>` form (e.g. `python -m pip`, `python -m pytest`, `python -m ruff`).
- Prefer `git -C <dir>` over `cd <dir> && git ...` to avoid changing the working directory.

## Testing

- RoboVerse content/integration tests live in `tests/` (`test_*.py`, functions `test_*`); run with
  `python -m pytest tests/`. Core simulator tests live in MetaSim's `metasim/test` — run those there,
  following MetaSim's `AGENTS.md`.
- **Be explicit about simulator scope.** Only run a simulator's tests when the change affects that
  simulator or shared code it uses. Read the sim→environment mapping from MetaSim
  [`ENVIRONMENTS.md`](../MetaSim/ENVIRONMENTS.md); if the mapping is unknown, **ask before running
  simulator-backed tests**.
- **GPU rule.** `isaacgym`, `isaacsim`, and `newton` runs are GPU-backed. If the environment can't
  see a GPU, report it as an environment blocker — don't record it as a normal test result, and
  don't compress setup/GPU failures into "test failures".
- Add a regression test for every bug fix. Do not claim "all tests pass" unless the exact requested
  commands ran in the correct environments.

## Documentation Workflow

For every non-trivial change (new feature, public behavior, CLI/config/data-format, module
responsibility, or durable design choice), update docs together with the code:

- User-facing docs are Sphinx under `docs/source/` and publish to <https://roboverse.wiki>.
- Ground docs in the repo: inspect code before naming paths, functions, classes, configs, or CLI
  flags. If something can't be verified, write `TODO: verify ...` instead of guessing.

## AI-Agent Rules

### Before changing code

- Read the relevant existing code paths and the existing test for the area first.
- Confirm whether the change belongs in RoboVerse or MetaSim (see Multi-Repo Workflow).

### Before adding files

- Ask: can this be done by cleanly extending an existing file? If yes, do that. New files — and
  especially new top-level scripts or docs — are the exception, not the default.

### Before declaring success

- Say exactly what was verified and how (which command, which sim environment).
- If a run was blocked by environment/GPU problems, say so and include the real blocker.
- For parity claims, state the measured delta and the backend(s) actually exercised.
