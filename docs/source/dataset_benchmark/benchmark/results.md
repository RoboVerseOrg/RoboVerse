# Benchmark Results

The benchmark results RoboVerse stands behind are the ones produced by the protocol in
[usage](usage.md): one evaluator, one result schema, task-macro success with Wilson intervals, scored per
information-access tier. The current table is the methods x perspectives matrix in that page
("Methods x perspectives: the first matrix"), reproducible with the commands listed there.

## Retired: the 2025 imitation-learning table

This page used to carry an imitation-learning table (RGB / RGBD / point-cloud encoders on `CloseBox`,
`PickCube`, `StackCube`, ... in- and out-of-distribution). It was produced before the evaluation protocol
existed: episode counts, success rules and seeds were not recorded, several of its tasks have since been
retired because their demos no longer replay (see `roboverse_pack/tasks/RETIRED_TASKS.md`), and the numbers
cannot be reproduced with the current tree. It is kept in git history, not here:

```bash
git show 56a0e67b:docs/source/dataset_benchmark/benchmark/results.md
```

A new row enters the benchmark only through `python -m metasim bench run` / `bench suite run` result files
checked against the schema (`bench validate`), under a named access tier.
