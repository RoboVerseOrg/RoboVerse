# RoboVerse Documentation

1. Install the dependencies from the repository root

```bash
cd RoboVerse
conda create -n roboverse_page python=3.11
conda activate roboverse_page
pip install -r docs/requirements-dev.txt
pip install -e ".[mujoco]" --no-build-isolation
```

2. Build the documentation and watch the change lively

```bash
cd docs
rm -rf build/; make html; sphinx-autobuild ./source ./build/html
```

3. If on your system, the autobuild loops forever although you did not change any file, you can use the command:

```bash
rm -rf build/; make html; sphinx-autobuild ./source ./build/html --ignore source/dataset_benchmark/tasks
```

This is due to some files are automatically generated in `source/dataset_benchmark/tasks` while building. This may interfere the change detection mechanism of sphinx-autobuild.

## Cloudflare Pages deployment

The site is deployed via the `.github/workflows/deploy-docs.yml` GitHub
Action, which builds in CI and uploads the static `public/` directory to
the Cloudflare Pages project `roboverse-release` with
`wrangler pages deploy`. Cloudflare itself is **not** asked to build —
this avoids its framework auto-detection running `pip install .` against
the repository's `pyproject.toml`, which would otherwise pull in `torch`,
`metasim` (with all of its sim/runtime deps), and friends.

Required GitHub repository secrets:

- `CLOUDFLARE_API_TOKEN` — token with `Account → Cloudflare Pages → Edit`
- `CLOUDFLARE_ACCOUNT_ID` — the Cloudflare account ID hosting the Pages project

In the Cloudflare Pages dashboard for `roboverse-release`, disable the
git integration's build (or set Build command + Build output directory
to empty) so Cloudflare doesn't shadow-build on every push. The project
should be set to receive deployments via Direct Upload only.

The workflow triggers on `push` to `main` when `docs/`, `scripts/docs/`,
or `roboverse_pack/tasks/` changes, plus manual `workflow_dispatch`.

To reproduce the build locally:

```bash
python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
bash scripts/docs/build_roboverse_wiki.sh public
```
