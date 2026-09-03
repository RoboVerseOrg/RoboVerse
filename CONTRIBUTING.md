# Contributing to MetaSim

Thanks for helping. Two documents define how work lands here:

- [`AGENTS.md`](./AGENTS.md) — engineering rules (repo boundaries, parity, code style, testing).
- [`RELEASING.md`](./RELEASING.md) — branches, pull-request gates, versioning and the release
  checklist.

Quick start:

```bash
git clone https://github.com/RoboVerseOrg/MetaSim.git && cd MetaSim
python -m pip install -e ".[dev,mujoco]"   # or the simulator extra you work on, see docs/source/get_started/installation.rst
pre-commit install                          # ruff lint + format at the pinned version
python -m pytest -k general                 # no simulator needed
```

Open a PR from a `type/scope-slug` branch with a Conventional Commit title, add a line under
`## [Unreleased]` in `CHANGELOG.md` for anything a user can notice, and keep one concern per PR.

By contributing you agree that your contribution is licensed under the Apache License 2.0
(see `LICENSE`).
