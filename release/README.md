# Legacy Release Metadata

The root `pyproject.toml` now builds and installs the `roboverse-py` downstream package from this repository.

This directory contains older release scaffolding and should not be used for new package builds. Build or install RoboVerse from the repository root; MetaSim is resolved from the standalone GitHub dependency declared in the root `pyproject.toml`:

```bash
python -m pip install -e ".[mujoco]" --no-build-isolation
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/roboverse-wheelhouse
```

For local two-repo development, install the local MetaSim checkout first, then install RoboVerse without resolving dependencies:

```bash
python -m pip install -e "../MetaSim[dev,mujoco]" --no-build-isolation
python -m pip install -e . --no-deps --no-build-isolation
```
