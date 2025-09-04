# RoboVerse Documentation

1. Install the dependencies

```bash
conda create -n roboverse_page python=3.11
conda activate roboverse_page
pip install -r requirements.txt
```

2. Build the documentation and watch the change lively

```bash
rm -rf build/; make html; sphinx-autobuild ./source ./build/html
```

3. If on your system, the autobuild loops forever although you did not change any file, you can use the command:

```bash
rm -rf build/; make html; sphinx-autobuild ./source ./build/html --ignore source/dataset_benchmark/tasks
```

This is due to some files are automatically generated in `source/dataset_benchmark/tasks` while building. This may interfere the change detection mechanism of sphinx-autobuild.
