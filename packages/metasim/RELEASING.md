# Releasing MetaSim

MetaSim is released from the RoboVerse monorepo: see [`../../RELEASING.md`](../../RELEASING.md). One `vX.Y.Z` tag on `main` builds and publishes `roboverse-metasim` (this directory) and `roboverse-py` (repo root) together; both `pyproject.toml` versions must equal the tag. Library changes under `metasim/` need a line under `## [Unreleased]` in this directory's `CHANGELOG.md`.
