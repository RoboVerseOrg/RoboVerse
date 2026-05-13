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

Cloudflare Pages should build the static docs without installing the RoboVerse Python package. Set the Pages build variable `SKIP_DEPENDENCY_INSTALL=1` so Cloudflare skips the automatic `pip install .` phase, which otherwise downloads simulator/runtime packages such as `torch`.

Use this build command:

```bash
python -m pip install --upgrade pip && python -m pip install -r docs/requirements.txt && bash scripts/docs/build_roboverse_wiki.sh public
```

Set the build output directory to `public`.
