# Results registry

One directory per submission: `results/<suite>/<date>-<name>/`, holding the result files one run of
`python -m metasim bench run` or `bench suite run` produced (any sub-directory layout). A submission is a
pull request; CI validates every file against the result schema and regenerates the leaderboard page
(`tools/bench/leaderboard.py`), so no number is typed by hand.

What a submission must contain:

- the result files themselves (`bench validate` passes), unmodified;
- a run under one information-access tier per file (the tier is recorded in `protocol.access`);
- a served or packaged policy that a maintainer can re-run: `policy.spec`, `checkpoint_sha256` or the
  server metadata are in the file; a row is marked verified once the nightly runner reproduces it.

The first submission, `tabletop-mujoco/2026-11-27-poc`, is the methods x perspectives proof of concept:
replay, hold, random, scripted skills, an LLM agent under two tiers, and OpenVLA-7B zero-shot, 10 episodes
per task.
`tabletop-mujoco/2026-11-28-isaacsim-twin` is the same suite on Isaac Sim 6 (replay and scripted skills, 10
episodes per task); `tabletop-mujoco/2026-11-28-il-ddpm-dit` is the first learned row, a diffusion policy trained
on scripted demos per task and scored under the visual tier, 20 episodes per task.
