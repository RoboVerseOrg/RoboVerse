# MetaSim

MetaSim is a unified simulation framework for robotics. It provides the core simulator abstraction, scenario configuration types, task registry, package discovery utilities, built-in example assets, and simulator backends used by RoboVerse and downstream content packs.

## Install

```bash
python -m pip install -e ".[dev]"
```

Install a simulator extra for the backend you need:

```bash
python -m pip install -e ".[dev,mujoco]"
```

## Content Packages

MetaSim core does not ship RoboVerse task, robot, scene, or ground packs. Downstream packages can register content through entry points, a local `metasim.toml`, `[tool.metasim.packages]` in `pyproject.toml`, or the `METASIM_*_PACKAGES` environment variables.

## Verification

```bash
pytest metasim/test/ -k general
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/metasim-wheelhouse
```

## Import Source

This repository was split from RoboVerse with Git history preserved for `metasim/`, `docs/source/metasim/`, `pyproject.toml`, and `LICENSE`. The split includes RoboVerse source commit `56ca0d70371c7b5757a62342a061be220849e8a6` plus the local packaging commit that renamed the root distribution to `metasim`.

## License

Apache License 2.0. See `LICENSE`.
