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

`results/robots/` holds the robot kit's runs, one directory per run: `<date>-<sim>-check/robots.json` from
`python -m metasim robot check --all --sim <sim> --out-dir`, and `<date>-parity-<a>-<b>/<robot>.json` from
`python -m metasim robot parity <robot> --sims <a> <b> --json`. `generate_robot_docs.py` renders the newest of each
into the Robots page (`--check` in CI keeps it current). The first entries are the MuJoCo sweep of 2026-11-28
(42 robots: 17 pass, 21 fail, 4 n/a), the Isaac Sim 6 sweep of the same day (9 pass, 23 fail, 10 n/a; one child process per robot) and the MuJoCo vs Isaac Sim 6 parity runs of the same day (five arms within 0.04 rad, the
H1 humanoids reported not judged).
