"""``python -m metasim <command>``.

* ``api-snapshot`` — public API surface vs ``metasim/test/api_snapshot.json`` (``--update`` to accept).
* ``doctor`` — every simulator backend: installed package versions vs the supported range and the
  last verified version (``metasim/sim/_versions.py``). Exit status 1 when an installed backend is
  outside its supported range, so it can gate an environment in CI or a Dockerfile.
* ``bench list | run | health | summarize | validate`` — the benchmark protocol (``metasim.bench``): list
  the registered tasks, score a policy on one task under a protocol and write a result JSON, replay
  every task's demos as a health gate, pool results into a leaderboard table, check a result file.
* ``bench suite list | run | report`` — multi-task suites with splits, scored task-macro per split.
* ``bench serve`` — serve a policy over HTTP so ``bench run --policy-url`` can score it from another
  process (a VLA in its own venv).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from metasim.constants import SimType
from metasim.sim._versions import BACKEND_REQUIREMENTS, doctor, format_reports
from metasim.sim.gl import apply_headless_gl


def _doctor(args: argparse.Namespace) -> int:
    sims = [SimType(b) for b in args.backend] if args.backend else None
    reports = doctor(sims)
    if args.json:
        payload = [
            {
                "backend": r.sim.value,
                "installed": r.installed,
                "requirements": [
                    {
                        "dist": s.requirement.dist,
                        "installed": s.installed,
                        "supported": s.requirement.spec,
                        "tested": s.requirement.tested,
                        "status": s.label,
                    }
                    for s in r.statuses
                ],
            }
            for r in reports
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(format_reports(reports))
        bad = [r for r in reports if r.unsupported]
        if bad:
            print(f"\nUNSUPPORTED versions installed for: {', '.join(r.sim.value for r in bad)}", file=sys.stderr)
    return 1 if any(r.unsupported for r in reports) else 0


def _api_snapshot(args: argparse.Namespace) -> int:
    from metasim.utils.api_surface import SNAPSHOT_PATH, collect_api, diff_api, load_snapshot, write_snapshot

    current = collect_api()
    if args.update or not SNAPSHOT_PATH.exists():
        write_snapshot(current)
        print(f"wrote {SNAPSHOT_PATH} ({sum(len(v) for v in current.values())} symbols)")
        return 0
    breaking, additions = diff_api(load_snapshot(), current)
    for line in additions:
        print(f"+ {line}")
    for line in breaking:
        print(f"! {line}")
    if breaking:
        print(f"{len(breaking)} breaking change(s); run `python -m metasim api-snapshot --update` to accept them.")
        return 1
    print("public API unchanged" if not additions else f"{len(additions)} addition(s), nothing broken")
    return 0


def _bench_access(args: argparse.Namespace) -> int:
    """``bench access``: the preset tiers, one row each."""
    from metasim.bench import ACCESS_PROFILES

    print("| name | observation | cameras | proprioception | tools | memory | wall time | budgets |")
    print("|---|---|---|---|---|---|---|---|")
    for name, p in ACCESS_PROFILES.items():
        cams = "all" if p.cameras is None else ", ".join(p.cameras)
        wall = "-" if p.wall_time_s is None else f"{p.wall_time_s:g}s"
        budgets = ", ".join(f"{k}={v}" for k, v in p.budgets.items()) or "-"
        print(
            f"| {name} | {p.observation} | {cams} | {p.proprioception} | {p.tools} | {p.memory} | {wall} | {budgets} |"
        )
    return 0


def _bench_list(args: argparse.Namespace) -> int:
    from metasim.task.registry import list_tasks

    names = list_tasks()
    if args.filter:
        names = [n for n in names if args.filter in n]
    if args.json:
        print(json.dumps({"count": len(names), "tasks": names}, indent=2))
    else:
        for n in names:
            print(n)
        print(f"{len(names)} task(s)", file=sys.stderr)
    return 0


def _parse_kv(items: list[str] | None) -> dict:
    """``k=v`` pairs with JSON values when they parse (``lr=0.1`` -> 0.1, ``name=x`` -> "x")."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--policy-arg expects key=value, got {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _make_env(
    task: str,
    *,
    sim: str,
    robots: list[str] | None,
    num_envs: int,
    gui: bool,
    device: str | None,
    cameras: list[str] | None = None,
):
    """Build the registry task ``task`` on ``sim`` from its class-level scenario, plus the preset ``cameras``."""
    import torch

    from metasim.task.registry import get_task_class

    task_cls = get_task_class(task)
    if getattr(task_cls, "scenario", None) is None:
        raise SystemExit(f"{task} has no class-level scenario; it cannot be built from the command line")
    updates = {"simulator": sim, "num_envs": num_envs, "headless": not gui}
    if robots:
        updates["robots"] = list(robots)
    scenario = task_cls.scenario.replace(**updates)
    if cameras:
        from metasim.bench.cameras import with_cameras

        scenario = with_cameras(scenario, cameras)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return task_cls(scenario, device)


def _make_policy(args: argparse.Namespace):
    """``(policy, policy_info)`` from ``--policy`` / ``--policy-url`` / ``--checkpoint`` / ``--policy-arg``."""
    from metasim.bench import RemotePolicy, checkpoint_sha256, load_policy

    if getattr(args, "policy_url", None):
        ignored = [
            flag
            for flag, given in (
                ("--policy", args.policy not in (None, "hold")),
                ("--policy-arg", bool(args.policy_arg)),
                ("--checkpoint", bool(args.checkpoint)),
            )
            if given
        ]
        if ignored:
            raise SystemExit(
                f"--policy-url scores the served policy; {', '.join(ignored)} would be ignored. Give the checkpoint to "
                "`bench serve` and pin it here with --require-metadata checkpoint=<path>."
            )
        policy = RemotePolicy(
            args.policy_url,
            required_metadata=_parse_kv(args.require_metadata),
            obs_keys=args.obs_key or None,
        )
        return policy, {"spec": "remote", "url": args.policy_url, "server": policy.metadata}
    policy_kwargs = _parse_kv(args.policy_arg)
    if args.checkpoint and ":" in args.policy:
        policy_kwargs.setdefault("checkpoint", args.checkpoint)  # a user policy takes it; the baselines have none
    policy = load_policy(args.policy, **policy_kwargs)
    policy_info = {"spec": args.policy, "args": policy_kwargs}
    if args.checkpoint:
        policy_info.update({"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha256(args.checkpoint)})
    return policy, policy_info


def _report_episode(ep) -> None:
    print(
        f"episode {ep.index:4d}  env {ep.env_id}  instance {ep.instance_id}  "
        f"{'SUCCESS' if ep.success else 'fail   '}  steps={ep.steps}  return={ep.return_sum:.3f}"
        + (f"  prompt={ep.prompt_variant}" if ep.prompt_variant not in (None, "canonical") else ""),
        file=sys.stderr,
    )


def _score_line(summary: dict) -> str:
    if summary["success_defined"]:
        low, high = summary["success_rate_ci95"]
        return f"{summary['successes']}/{summary['episodes']} = {summary['success_rate']:.3f} (95% CI {low:.3f}-{high:.3f})"
    return f"{summary['episodes']} episode(s), success undefined (the task declares no success signal)"


def _bench_run(args: argparse.Namespace) -> int:
    import dataclasses
    import re
    import time

    from metasim.bench import EvalProtocol, evaluate, get_access_profile

    try:
        protocol = EvalProtocol(
            episodes=args.episodes,
            success_rule=args.success_rule,
            hold_steps=args.hold_steps,
            max_episode_steps=args.max_steps,
            seed=args.seed,
            instances=tuple(args.instance) if args.instance else None,
            name=args.protocol_name,
            demo_budget_factor=args.demo_budget,
            demo_budget_quantile=args.demo_budget_quantile,
            prompt_split=args.prompt_split,
            access=get_access_profile(args.access),
        )
    except ValueError as err:
        raise SystemExit(f"invalid protocol: {err}") from err
    policy, policy_info = _make_policy(args)
    try:
        env = _make_env(
            args.task,
            sim=args.sim,
            robots=args.robot,
            num_envs=args.num_envs,
            gui=args.gui,
            device=args.device,
            cameras=getattr(args, "camera", None),
        )
    except (KeyError, ValueError) as err:
        raise SystemExit(f"cannot build {args.task} on {args.sim}: {err}") from err
    _apply_demos_override(env, getattr(args, "demos", None))
    out = args.out
    if out is None:
        robots = "+".join(r.name for r in env.scenario.robots) or "norobot"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{args.task}__{robots}__{args.sim}__{policy.name}__{stamp}")
        out = f"bench_results/{slug}.json"
    # every finished episode is appended to <out>.episodes.jsonl as it ends, so a crash or a kill at
    # episode 49 of 50 keeps 49 rows; the file is removed once the complete result is saved
    partial = f"{out}.episodes.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(partial)), exist_ok=True)
    with open(partial, "w") as stream:

        def on_episode(ep) -> None:
            _report_episode(ep)
            stream.write(json.dumps(dataclasses.asdict(ep)) + "\n")
            stream.flush()

        try:
            result = evaluate(
                env, policy, protocol, policy_info=policy_info, on_episode=on_episode, task_name=args.task
            )
            if getattr(args, "camera", None):
                result.notes["cameras"] = list(args.camera)
            if getattr(args, "demos", None):
                result.notes["demos"] = args.demos  # scored from these instances, not the task's own
            # saved before close(): a backend whose shutdown ends the process (Isaac Sim's fast shutdown) must
            # not take the result with it
            path = result.save(out)
        finally:
            env.close()
    os.remove(partial)
    print(
        f"{args.task} [{', '.join(result.robots)} @ {args.sim}] {policy.name}: {_score_line(result.summary)}, "
        f"mean {result.summary['mean_steps']:.1f} steps -> {path}"
    )
    return 0


def _apply_demos_override(env, demos: str | None) -> None:
    """``--demos PATH``: evaluate against another demo file / directory than the task's own (a freshly recorded set)."""
    if demos:
        env.traj_filepath = demos
        env.__dict__.pop("_bench_instance_pool", None)


def _bench_record(args: argparse.Namespace) -> int:
    """Record demos with a policy and write the ``{robot: [demos]}`` file the gate replays."""
    from metasim.bench import EvalProtocol
    from metasim.bench.record import record_demos

    try:
        protocol = EvalProtocol(
            episodes=args.episodes,
            success_rule=args.success_rule,
            hold_steps=args.hold_steps,
            max_episode_steps=args.max_steps,
            seed=args.seed,
            name="record",
        )
    except ValueError as err:
        raise SystemExit(f"invalid protocol: {err}") from err
    policy, policy_info = _make_policy(args)
    try:
        env = _make_env(
            args.task,
            sim=args.sim,
            robots=args.robot,
            num_envs=args.num_envs,
            gui=False,
            device=args.device,
            cameras=getattr(args, "camera", None),
        )
    except (KeyError, ValueError) as err:
        raise SystemExit(f"cannot build {args.task} on {args.sim}: {err}") from err
    _apply_demos_override(env, getattr(args, "demos", None))
    try:
        result, summary = record_demos(
            env,
            policy,
            protocol,
            out=args.out,
            success_only=not args.keep_failures,
            task_name=args.task,
            on_episode=_report_episode,
            policy_info=policy_info,
        )
    finally:
        env.close()
    print(
        f"{args.task} [{summary['robot']} @ {args.sim}] {policy.name}: {_score_line(result.summary)}; "
        f"kept {summary['kept']} demo(s), dropped {summary['dropped']} -> {summary['path']}"
    )
    return 0 if summary["kept"] else 1


def _bench_serve(args: argparse.Namespace) -> int:
    """Serve a policy for ``bench run --policy-url`` clients until interrupted."""
    from metasim.bench import PolicyServer

    policy, policy_info = _make_policy(argparse.Namespace(**{**vars(args), "policy_url": None}))
    metadata = {k: v for k, v in policy_info.items() if k in ("spec", "checkpoint", "checkpoint_sha256")}
    metadata.update(_parse_kv(args.metadata))
    server = PolicyServer(policy, host=args.host, port=args.port, device=args.device or "cpu", metadata=metadata)
    print(f"serving {policy.name} on {server.url}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def _bench_conformance(args: argparse.Namespace) -> int:
    """Check a policy server against the wire protocol without a simulator; exit 1 when a contract fails."""
    from metasim.bench import conformance

    robot_cfg = None
    if args.robot:
        from metasim.utils.setup_util import get_robot

        robot_cfg = get_robot(args.robot)
    report = conformance(
        args.policy_url, num_envs=args.num_envs, robot_cfg=robot_cfg, action_dim=args.action_dim, timeout=args.timeout
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.table())
    return 0 if report.ok else 1


def _bench_suite_list(args: argparse.Namespace) -> int:
    from metasim.bench import get_suite, list_suites

    names = list_suites()
    if args.json:
        out = []
        for name in names:
            suite = get_suite(name)
            out.append({
                "name": name,
                "simulator": suite.simulator,
                "description": suite.description,
                "splits": {s: [t.task for t in tasks] for s, tasks in suite.splits.items()},
                "protocol": suite.protocol.to_dict(),
            })
        print(json.dumps(out, indent=2))
        return 0
    for name in names:
        suite = get_suite(name)
        splits = ", ".join(f"{s} ({len(t)})" for s, t in suite.splits.items())
        print(f"{name}  [{suite.simulator or 'any'}]  {splits}")
    print(f"{len(names)} suite(s)", file=sys.stderr)
    return 0


def _write_suite_report(suite, results, out_dir: str, *, policy: str | None = None) -> dict:
    from metasim.bench import suite_report, suite_table

    report = suite_report(suite, results, policy=policy)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "suite.json"), "w") as f:
        json.dump(report, f, indent=2)
    table = suite_table(report)
    with open(os.path.join(out_dir, "suite.md"), "w") as f:
        f.write(table)
    print(table)
    return report


def _suite_task_isolated(entry, protocol, args: argparse.Namespace, sim: str, out_path: str):
    """Score one suite task in a child interpreter (``bench run``) and load its result file.

    Isaac Sim boots Kit once per process and ends the process in ``close()``, so the tasks of a suite cannot
    share one interpreter there. The child gets the entry's protocol as flags; a child that exits without a
    result raises with the last line of its log, which the suite loop records as that task's failure.
    """
    import json as _json
    import subprocess
    import sys

    from metasim.bench import ACCESS_PROFILES, BenchmarkResult

    cmd = [sys.executable, "-m", "metasim", "bench", "run", "--task", entry.task, "--sim", sim, "--out", out_path]
    cmd += ["--num-envs", str(args.num_envs), "--episodes", str(protocol.episodes)]
    cmd += ["--success-rule", protocol.success_rule, "--hold-steps", str(protocol.hold_steps)]
    cmd += ["--seed", str(protocol.seed), "--protocol-name", protocol.name, "--prompt-split", protocol.prompt_split]
    cmd += ["--demo-budget-quantile", str(protocol.demo_budget_quantile)]
    if protocol.max_episode_steps is not None:
        cmd += ["--max-steps", str(protocol.max_episode_steps)]
    if protocol.demo_budget_factor is not None:
        cmd += ["--demo-budget", str(protocol.demo_budget_factor)]
    for i in protocol.instances or ():
        cmd += ["--instance", str(i)]
    preset = ACCESS_PROFILES.get(protocol.access.name)
    if preset is not None and preset == protocol.access:
        cmd += ["--access", protocol.access.name]
    else:  # a custom tier travels as a file the child reads
        logs_dir = os.path.join(os.path.dirname(out_path), "logs")  # not next to the results: readers glob *.json there
        os.makedirs(logs_dir, exist_ok=True)
        access_path = os.path.join(logs_dir, f"access__{protocol.access.name}.json")
        with open(access_path, "w") as f:
            _json.dump(protocol.access.to_dict(), f)
        cmd += ["--access", access_path]
    for robot in entry.robots:
        cmd += ["--robot", robot]
    for cam in getattr(args, "camera", None) or ():
        cmd += ["--camera", cam]
    if getattr(args, "policy_url", None):
        cmd += ["--policy-url", args.policy_url]
        for kv in args.require_metadata or ():
            cmd += ["--require-metadata", kv]
        for key in args.obs_key or ():
            cmd += ["--obs-key", key]
    else:
        cmd += ["--policy", args.policy]
        for kv in args.policy_arg or ():
            cmd += ["--policy-arg", kv]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
    if args.device:
        cmd += ["--device", args.device]
    log_dir = os.path.join(os.path.dirname(out_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{entry.task}.log")
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT)
    if not os.path.exists(out_path):
        raise RuntimeError(
            f"child exited with {proc.returncode} before saving a result ({_last_error_line(log_path)}; {log_path})"
        )
    return BenchmarkResult.load(out_path)


def _bench_suite_run(args: argparse.Namespace) -> int:
    """Every task of a suite (or one split) under the suite protocol; one result file per task."""
    import dataclasses

    from metasim.bench import evaluate, get_suite

    suite = get_suite(args.suite)
    sim = args.sim or suite.simulator
    if not sim:
        raise SystemExit(f"suite {suite.name!r} names no simulator; pass --sim")
    out_dir = args.out_dir or os.path.join("bench_results", "suites", suite.name, sim)
    isolate = args.isolate == "always" or (args.isolate == "auto" and sim in ISOLATED_HEALTH_BACKENDS)
    overrides = {k: v for k, v in {"episodes": args.episodes, "seed": args.seed}.items() if v is not None}
    suite = _suite_under_access(suite, args.access)
    if args.episodes is not None and any(
        suite.protocol_for(entry).instances is not None for _, entry in suite.entries(args.split)
    ):
        raise SystemExit(
            f"suite {suite.name!r} names explicit instances, so its episode count is theirs; drop --episodes"
        )
    results, failures = [], []
    for split, entry in suite.entries(args.split):
        print(f"=== {suite.name}/{split}: {entry.task}", file=sys.stderr, flush=True)
        try:
            protocol = suite.protocol_for(entry)
            if overrides:
                protocol = dataclasses.replace(protocol, **overrides)
            if isolate:
                policy_name = "remote" if getattr(args, "policy_url", None) else args.policy.rsplit(":", 1)[-1]
                robots = "+".join(entry.robots) or "task"
                out_path = os.path.join(out_dir, f"{split}__{entry.task}__{robots}__{policy_name}.json")
                result = _suite_task_isolated(entry, protocol, args, sim, out_path)
                result.notes["suite"] = {"name": suite.name, "split": split}
                result.save(out_path)
            else:
                policy, policy_info = _make_policy(args)
                env = _make_env(
                    entry.task,
                    sim=sim,
                    robots=list(entry.robots) or None,
                    num_envs=args.num_envs,
                    gui=False,
                    device=args.device,
                    cameras=getattr(args, "camera", None),
                )
                try:
                    result = evaluate(
                        env, policy, protocol, policy_info=policy_info, on_episode=_report_episode, task_name=entry.task
                    )
                    result.notes["suite"] = {"name": suite.name, "split": split}
                    if getattr(args, "camera", None):
                        result.notes["cameras"] = list(args.camera)
                    robots = "+".join(result.robots) or "norobot"
                    result.save(
                        os.path.join(out_dir, f"{split}__{entry.task}__{robots}__{policy.name}.json")
                    )  # before close()
                finally:
                    env.close()
        except Exception as err:  # one broken task is reported as missing, the others still run
            detail = f"{type(err).__name__}: {err}".splitlines()[0][:300]
            print(f"{entry.task}: ERROR {detail}", file=sys.stderr)
            failures.append({"split": split, "task": entry.task, "error": detail})
            continue
        print(f"{entry.task}: {_score_line(result.summary)}", file=sys.stderr)
        results.append(result)
    report = _write_suite_report(suite, results, out_dir)
    errors_path = os.path.join(out_dir, "errors.json")
    if failures:
        with open(errors_path, "w") as f:
            json.dump(failures, f, indent=2)
    elif os.path.exists(errors_path):
        os.remove(errors_path)  # a clean re-run into the same directory must not keep the last run's failures
    complete = all(block["complete"] for block in report["splits"].values())
    return 0 if complete and not failures else 1


def _suite_under_access(suite, spec: str | None):
    """``suite`` evaluated and scored under the access tier ``spec``; ``suite`` itself when ``spec`` is None.

    ``spec`` is a preset name or a JSON file. The tier is part of the protocol a suite scores against, so
    running and reporting must see the same one.
    """
    if not spec:
        return suite
    import dataclasses

    from metasim.bench import get_access_profile

    try:
        access = get_access_profile(spec)
    except ValueError as err:
        raise SystemExit(f"invalid --access: {err}") from err
    return dataclasses.replace(suite, protocol=dataclasses.replace(suite.protocol, access=access))


def _bench_suite_report(args: argparse.Namespace) -> int:
    from metasim.bench import get_suite

    suite = _suite_under_access(get_suite(args.suite), args.access)
    results, _ = _load_results(args.paths)
    try:
        if args.out_dir:
            report = _write_suite_report(suite, results, args.out_dir, policy=args.policy)
        else:
            from metasim.bench import suite_report, suite_table

            report = suite_report(suite, results, policy=args.policy)
            print(suite_table(report))
    except ValueError as err:
        raise SystemExit(str(err)) from err
    return 0 if all(block["complete"] for block in report["splits"].values()) else 1


HEALTH_STATUSES = ("ok", "low", "unreliable", "flaky", "invalid", "undefined", "unsupported", "no-demos", "error")
"""``unreliable``: the hold policy also succeeds (the checker accepts the initial state); ``flaky``: the same replay
scores differently on a second run (``--repeat``)."""
HEALTH_FAILING = ("low", "unreliable", "flaky", "invalid", "error")
"""The statuses that fail the gate."""
"""Outcomes of ``bench health`` for one task, in the order the table sorts them."""

ISOLATED_HEALTH_BACKENDS = frozenset({"isaacsim", "isaacgym"})
"""Backends whose simulator boots once per process: ``bench health`` runs each task in its own subprocess."""


def _error_detail(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}".splitlines()[0][:200]


def _health_rows_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "rows")


def _write_health_row(out_dir: str, row: dict) -> str:
    """Persist one task's row as ``<out_dir>/rows/<task>.json`` (the isolated runner and crash recovery read it)."""
    os.makedirs(_health_rows_dir(out_dir), exist_ok=True)
    path = os.path.join(_health_rows_dir(out_dir), f"{row['task']}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2)
    return path


def _read_health_row(out_dir: str, task: str) -> dict | None:
    path = os.path.join(_health_rows_dir(out_dir), f"{task}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _health_task_isolated(name: str, args: argparse.Namespace) -> dict:
    """Run :func:`_health_task` for one task in a child interpreter and collect its row.

    Isaac Sim boots Kit once per process and its shutdown may end the process, so a sweep over several
    tasks cannot share one interpreter. The child runs ``bench health --task <name> --isolate never`` into the
    same ``--out-dir`` and writes its row before it closes the simulator; a child that dies without a row is
    reported as ``error`` with its exit code.
    """
    import subprocess
    import sys

    row_path = os.path.join(_health_rows_dir(args.out_dir), f"{name}.json")
    if os.path.exists(row_path):
        os.remove(row_path)
    cmd = [sys.executable, "-m", "metasim", "bench", "health", "--sim", args.sim, "--task", name, "--isolate", "never"]
    cmd += ["--num-envs", str(args.num_envs), "--episodes", str(args.episodes), "--seed", str(args.seed)]
    cmd += ["--threshold", str(args.threshold), "--out-dir", args.out_dir]
    env = {**os.environ, "METASIM_HEALTH_ROWS_ONLY": "1"}  # the child writes its row, not the sweep's report
    for robot in args.robot or ():
        cmd += ["--robot", robot]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.device:
        cmd += ["--device", args.device]
    if getattr(args, "demos", None):
        cmd += ["--demos", args.demos]
    if getattr(args, "floor", False):
        cmd += ["--floor"]
    if getattr(args, "repeat", 1) and args.repeat > 1:
        cmd += ["--repeat", str(args.repeat)]
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    log_path = os.path.join(args.out_dir, "logs", f"{name}.log")
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, check=False, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    row = _read_health_row(args.out_dir, name)
    if row is None:
        last = _last_error_line(log_path)
        # a simulator that ends the process in close() (Isaac Sim 6) takes the launch-time error with it;
        # the log still tells an unsupported task from a broken one
        status = _health_error_status(last)
        row = {
            "task": name,
            "status": status,
            "success_rate": None,
            "episodes": 0,
            "result": None,
            "detail": f"child exited with {proc.returncode} before reporting ({last}; {log_path})",
        }
    return row


def _health_error_status(detail: str) -> str:
    """The health status an exception text stands for.

    ``unsupported`` for the handler's "no asset for this backend" refusal (the task cannot run there), ``no-demos``
    for a demo file under ``roboverse_data/trajs`` that is neither local nor on the Hub (the task ships no replayable
    demo for this robot), ``error`` otherwise.
    """
    if "asset path set for the" in detail:
        return "unsupported"
    if "roboverse_data/trajs/" in detail and ("neither exists" in detail or "Cannot fetch" in detail):
        return "no-demos"
    return "error"


def _last_error_line(log_path: str) -> str:
    """The last line of a child log that names a Python exception, else a hint that there is none."""
    try:
        with open(log_path, errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return "no log"
    for line in reversed(lines):
        if ("Error" in line or "Exception" in line) and not line.startswith(("File ", "Traceback")):
            return line[:160]
    return lines[-1][:160] if lines else "empty log"


def _health_task(name: str, args: argparse.Namespace) -> dict:
    """Replay one task's demos under the protocol and classify the outcome.

    Returns ``{"task", "status", "success_rate", "episodes", "result", "detail"}``: ``ok`` at or above the
    threshold, ``low`` below it, ``invalid`` when an initial state puts a robot joint outside its limits,
    ``undefined`` when the task declares no success signal, ``unsupported`` when it does not run on the
    backend, ``no-demos`` when it ships no demo file, ``error`` with the exception text.
    """
    import torch

    from metasim.bench import EvalProtocol, ReplayPolicy, evaluate, instance_pool, validate_instances
    from metasim.task.registry import get_task_class

    row = {"task": name, "status": "error", "success_rate": None, "episodes": 0, "result": None, "detail": ""}
    try:
        task_cls = get_task_class(name)
        if getattr(task_cls, "scenario", None) is None:
            row.update(status="unsupported", detail="no class-level scenario")
            return row
        supported = getattr(task_cls, "supported_simulators", None)
        if supported is not None and args.sim not in supported:
            row.update(status="unsupported", detail=f"supported_simulators={tuple(supported)}")
            return row
        updates = {"simulator": args.sim, "num_envs": args.num_envs, "headless": True}
        if args.robot:
            updates["robots"] = list(args.robot)
        scenario = task_cls.scenario.replace(**updates)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        protocol = EvalProtocol(episodes=args.episodes, max_episode_steps=args.max_steps, seed=args.seed, name="health")
        env = task_cls(scenario, device)
        _apply_demos_override(env, getattr(args, "demos", None))
        try:
            try:
                # the demo file is a property of the built task: some tasks pick it per robot or in __init__
                if getattr(env, "traj_filepath", None) is None:
                    row.update(status="no-demos")
                elif problems := validate_instances(env, instance_pool(env)):
                    # an unphysical initial state replays differently in every consumer; the score would be noise
                    row.update(status="invalid", detail=f"{len(problems)} out-of-limit joint value(s): {problems[0]}")
                else:
                    result = evaluate(env, ReplayPolicy(), protocol, policy_info={"spec": "replay"}, task_name=name)
                    robots = "+".join(result.robots) or "norobot"
                    if getattr(args, "demos", None):
                        result.notes["demos"] = args.demos
                    path = result.save(os.path.join(args.out_dir, f"{name}__{robots}__{args.sim}__replay.json"))
                    summary = result.summary
                    row.update(episodes=summary["episodes"], result=str(path))
                    if not summary["success_defined"]:
                        row.update(status="undefined")
                    else:
                        rate = float(summary["success_rate"])
                        row.update(success_rate=rate, status="ok" if rate >= args.threshold else "low")
                        if row["status"] == "ok":
                            _health_reliability(env, name, protocol, result, row, args)
            except Exception as err:
                detail = _error_detail(err)
                row.update(status=_health_error_status(detail), detail=detail)
            # Written before close(): a simulator whose shutdown ends or aborts the process (Isaac Sim)
            # must not take the verdict, or the error, with it.
            _write_health_row(args.out_dir, row)
            return row
        finally:
            env.close()
    except Exception as err:  # one broken task must not hide the others: it is reported as a row
        detail = _error_detail(err)
        row.update(status=_health_error_status(detail), detail=detail)
        return row


def _health_reliability(env, name: str, protocol, result, row: dict, args: argparse.Namespace) -> None:
    """The checks a passing replay must also survive, when asked for.

    ``--repeat N`` replays the same instances and seed N times; a different per-episode verdict is ``flaky``.
    ``--floor`` scores the hold policy (no motion) under the same protocol; any success is ``unreliable``: the
    checker accepts the initial state, so a success rate on that task measures nothing.
    """
    from metasim.bench import HoldPolicy, ReplayPolicy, evaluate

    flags = [e.success for e in result.episodes]
    for k in range(1, int(getattr(args, "repeat", 1) or 1)):
        again = evaluate(env, ReplayPolicy(), protocol, policy_info={"spec": "replay"}, task_name=name)
        other = [e.success for e in again.episodes]
        if other != flags:
            row.update(
                status="flaky",
                detail=f"replay run {k + 1} scored {sum(other)}/{len(other)} against {sum(flags)}/{len(flags)} on the "
                "same instances and seed",
            )
            return
    if getattr(args, "floor", False):
        hold = evaluate(env, HoldPolicy(), protocol, policy_info={"spec": "hold"}, task_name=name)
        floor = hold.summary["success_rate"]
        row["floor_rate"] = None if floor is None else float(floor)
        if floor:
            row.update(
                status="unreliable",
                detail=f"the hold policy (no motion) scores {float(floor):.2f}: the checker accepts the initial state",
            )


def _health_table(rows: list[dict], threshold: float) -> str:
    order = {status: i for i, status in enumerate(HEALTH_STATUSES)}
    lines = [
        "| task | status | replay success | episodes | detail |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (order.get(r["status"], 99), r["task"])):
        rate = "" if row["success_rate"] is None else f"{row['success_rate']:.3f}"
        lines.append(f"| {row['task']} | {row['status']} | {rate} | {row['episodes']} | {row['detail']} |")
    counts = {status: sum(r["status"] == status for r in rows) for status in HEALTH_STATUSES}
    lines.append("")
    lines.append(
        f"{len(rows)} task(s), threshold {threshold:.2f}: "
        + ", ".join(f"{n} {status}" for status, n in counts.items() if n)
    )
    return "\n".join(lines)


def _bench_health(args: argparse.Namespace) -> int:
    """The task health gate: the replay baseline on every task that ships demos, one row per task.

    A task whose own demos do not replay to the threshold on the backend they were recorded on is
    broken (asset, checker or physics), not hard; the gate exits 1 when any task is ``low``,
    ``unreliable``, ``flaky`` or ``error`` so it can guard a release. Results go to ``--out-dir`` as ordinary result files plus a
    ``health.json`` with the rows.
    """
    from metasim.task.registry import list_tasks

    names = list(args.task) if args.task else list_tasks()
    os.makedirs(args.out_dir, exist_ok=True)
    isolate = args.isolate == "always" or (args.isolate == "auto" and args.sim in ISOLATED_HEALTH_BACKENDS)
    if (
        not isolate
        and args.sim == "isaacsim"
        and len(names) > 1
        and os.environ.get("METASIM_ISAAC_FAST_SHUTDOWN", "1") != "0"
    ):
        raise SystemExit(
            f"bench health: isaacsim ends the process when the first task's simulator closes, so {len(names)} tasks "
            "cannot share one interpreter; use --isolate auto|always (or METASIM_ISAAC_FAST_SHUTDOWN=0)"
        )
    rows = []
    for name in names:
        row = _health_task_isolated(name, args) if isolate else _health_task(name, args)
        if not isolate:
            _write_health_row(args.out_dir, row)
        if row["status"] == "no-demos" and not args.task:
            continue  # the default sweep is over tasks with demos; an explicit --task is always reported
        rows.append(row)
        print(f"{row['task']:48s} {row['status']:12s} {row['detail']}", file=sys.stderr)
    if os.environ.get("METASIM_HEALTH_ROWS_ONLY") == "1":  # a child of an isolated sweep: the parent reports
        return 1 if any(r["status"] in HEALTH_FAILING for r in rows) else 0
    table = _health_table(rows, args.threshold)
    with open(os.path.join(args.out_dir, "health.json"), "w") as f:
        json.dump(
            {"simulator": args.sim, "threshold": args.threshold, "episodes": args.episodes, "rows": rows}, f, indent=2
        )
    with open(os.path.join(args.out_dir, "health.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    return 1 if any(r["status"] in HEALTH_FAILING for r in rows) else 0


_REPORT_FILES = frozenset({"suite.json", "errors.json", "health.json"})
"""Files the bench commands write next to result files; directory globs skip them."""


def _load_results(paths: list[str]):
    """Result files from paths and directories (``*.json``, minus report files); unreadable ones are skipped."""
    from pathlib import Path

    from metasim.bench import BenchmarkResult

    files = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(f for f in p.glob("*.json") if f.name not in _REPORT_FILES) if p.is_dir() else [p])
    results, bad = [], 0
    for f in files:
        try:
            results.append(BenchmarkResult.load(f))
        except (OSError, ValueError, json.JSONDecodeError) as err:
            bad += 1
            print(f"skipping {f}: {err}", file=sys.stderr)
    return results, bad


def _bench_summarize(args: argparse.Namespace) -> int:
    from metasim.bench import summarize

    results, bad = _load_results(args.paths)
    if not results:
        print("no result files found", file=sys.stderr)
        return 1
    print(summarize(results))
    return 1 if bad else 0


def _bench_matrix(args: argparse.Namespace) -> int:
    from metasim.bench import matrix

    results, bad = _load_results(args.paths)
    if not results:
        print("no result files found", file=sys.stderr)
        return 1
    print(matrix(results))
    return 1 if bad else 0


def _bench_validate(args: argparse.Namespace) -> int:
    from metasim.bench import validate_result

    bad = 0
    for p in args.paths:
        try:
            with open(p) as f:
                validate_result(json.load(f))
            print(f"ok    {p}")
        except (OSError, ValueError, json.JSONDecodeError) as err:
            bad += 1
            print(f"FAIL  {p}: {err}")
    return 1 if bad else 0


#: Mirrors ``metasim.bench.protocol.SUCCESS_RULES`` / ``metasim.bench.policy.BUILTIN_POLICIES`` (pinned by a test)
#: so building the parser does not import the bench package, and with it torch, for ``doctor``.
_SUCCESS_RULES = ("first", "final", "held")
_BUILTIN_POLICIES = ("hold", "random", "replay")
_PROMPT_SPLITS = ("canonical", "train", "heldout")


def _add_remote_args(parser) -> None:
    parser.add_argument("--policy-url", help="score a policy served by `bench serve` at this URL (replaces --policy)")
    parser.add_argument(
        "--require-metadata",
        action="append",
        metavar="KEY=VALUE",
        help="the server must report this metadata value, e.g. a checkpoint id (repeatable)",
    )
    parser.add_argument("--obs-key", action="append", help="observation path prefix to send (repeatable; default all)")


def _add_bench(sub) -> None:
    b = sub.add_parser("bench", help="benchmark protocol: list tasks, run an evaluation, summarize results")
    bs = b.add_subparsers(dest="bench_command", required=True)
    ls = bs.add_parser("list", help="registered task names (static index, no simulator import)")
    ls.add_argument("--filter", help="substring to match")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_bench_list)
    r = bs.add_parser("run", help="score a policy on one task and write a result JSON")
    r.add_argument("--task", required=True, help="registry name, e.g. stack_cube")
    r.add_argument("--sim", required=True, help="simulator backend, e.g. mujoco")
    r.add_argument("--robot", action="append", help="robot config name(s) to evaluate with (repeatable)")
    r.add_argument("--num-envs", type=int, default=1)
    r.add_argument("--episodes", type=int, help="default 10, or the number of --instance values given")
    r.add_argument("--policy", default="hold", help=f"one of {list(_BUILTIN_POLICIES)} or 'pkg.module:attr'")
    r.add_argument("--policy-arg", action="append", metavar="KEY=VALUE", help="constructor argument (repeatable)")
    r.add_argument("--checkpoint", help="checkpoint path; passed to the policy and hashed into the result")
    r.add_argument("--success-rule", choices=_SUCCESS_RULES, default="first")
    r.add_argument("--hold-steps", type=int, default=1)
    r.add_argument("--max-steps", type=int, help="evaluator-side episode cap (default: the task's)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--instance", type=int, action="append", help="explicit instance index (repeatable)")
    r.add_argument("--protocol-name", default="default")
    r.add_argument(
        "--demo-budget",
        type=float,
        metavar="FACTOR",
        help="episode cap = FACTOR x a quantile of the task's demo lengths (instead of --max-steps)",
    )
    r.add_argument("--demo-budget-quantile", type=float, default=0.75)
    r.add_argument(
        "--prompt-split",
        choices=_PROMPT_SPLITS,
        default="canonical",
        help="instruction per episode: the task's own, a training paraphrase, or a held-out one",
    )
    r.add_argument(
        "--access",
        help="information-access tier: a preset (`bench access`) or a JSON file; default privileged",
    )
    r.add_argument(
        "--camera",
        action="append",
        help="add a preset camera to the scenario (`front`, `side`, `top`; repeatable), for the visual tiers",
    )
    r.add_argument("--demos", help="demo file or directory to take the instances from instead of the task's own")
    r.add_argument("--device", help="torch device (default: cuda if available)")
    r.add_argument("--gui", action="store_true", help="run with the simulator viewer (default headless)")
    r.add_argument("--out", help="result path (default: bench_results/<task>__<robot>__<sim>__<policy>__<time>.json)")
    _add_remote_args(r)
    r.set_defaults(func=_bench_run)
    rc = bs.add_parser("record", help="record demos with a policy into the {robot: [demos]} file the gate replays")
    rc.add_argument("--task", required=True)
    rc.add_argument("--sim", required=True)
    rc.add_argument("--robot", action="append", help="robot config name (one robot per demo file)")
    rc.add_argument("--num-envs", type=int, default=1)
    rc.add_argument("--episodes", type=int, default=50, help="episodes to attempt")
    rc.add_argument(
        "--policy", default="metasim.bench.skills:ScriptedPolicy", help="policy spec (default: the scripted skills)"
    )
    rc.add_argument("--policy-arg", action="append", metavar="KEY=VALUE")
    rc.add_argument("--checkpoint")
    rc.add_argument("--success-rule", choices=_SUCCESS_RULES, default="first")
    rc.add_argument("--hold-steps", type=int, default=1)
    rc.add_argument("--max-steps", type=int, help="episode cap (default: the task's)")
    rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("--demos", help="start the episodes from the instances of this demo file instead of the task's")
    rc.add_argument(
        "--keep-failures", action="store_true", help="also write the episodes the task did not score as success"
    )
    rc.add_argument("--camera", action="append", help="preset camera to add (for policies that need an image)")
    rc.add_argument("--device", help="torch device (default: cuda if available)")
    rc.add_argument("--out", required=True, help="output file, e.g. demos/pick_cube/v2/franka_v2.pkl.gz")
    _add_remote_args(rc)
    rc.set_defaults(func=_bench_record)
    sv = bs.add_parser("serve", help="serve a policy over HTTP for `bench run --policy-url` (run in the policy's env)")
    sv.add_argument("--policy", required=True, help=f"one of {list(_BUILTIN_POLICIES)} or 'pkg.module:attr'")
    sv.add_argument("--policy-arg", action="append", metavar="KEY=VALUE", help="constructor argument (repeatable)")
    sv.add_argument("--checkpoint", help="checkpoint path; passed to the policy and reported in the metadata")
    sv.add_argument("--host", default="127.0.0.1", help="interface to bind (default: this machine only)")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--device", help="torch device the policy runs on (default cpu)")
    sv.add_argument("--metadata", action="append", metavar="KEY=VALUE", help="extra metadata field (repeatable)")
    sv.set_defaults(func=_bench_serve)
    cf = bs.add_parser(
        "conformance",
        help="check a policy server against the wire protocol (metadata, reset, act, chunks) without a simulator",
    )
    cf.add_argument("--policy-url", required=True, help="the server, e.g. http://127.0.0.1:8000")
    cf.add_argument("--num-envs", type=int, default=2, help="envs in the synthetic batch (default 2)")
    cf.add_argument("--robot", help="use this robot config's action box instead of a synthetic one, e.g. franka")
    cf.add_argument("--action-dim", type=int, default=7, help="joints of the synthetic robot when --robot is not given")
    cf.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    cf.add_argument("--json", action="store_true")
    cf.set_defaults(func=_bench_conformance)
    su = bs.add_parser("suite", help="multi-task suites: list them, run one, report task-macro scores")
    sus = su.add_subparsers(dest="suite_command", required=True)
    sl = sus.add_parser("list", help="registered suites")
    sl.add_argument("--json", action="store_true")
    sl.set_defaults(func=_bench_suite_list)
    sr = sus.add_parser("run", help="score a policy on every task of a suite; exit 1 unless every task scored")
    sr.add_argument("--suite", required=True)
    sr.add_argument("--split", help="one split only (default: all)")
    sr.add_argument("--sim", help="backend (default: the suite's)")
    sr.add_argument("--num-envs", type=int, default=1)
    sr.add_argument("--episodes", type=int, help="override the suite protocol's episode count (smoke runs)")
    sr.add_argument("--seed", type=int, help="override the suite protocol's seed")
    sr.add_argument("--policy", default="hold", help=f"one of {list(_BUILTIN_POLICIES)} or 'pkg.module:attr'")
    sr.add_argument("--policy-arg", action="append", metavar="KEY=VALUE", help="constructor argument (repeatable)")
    sr.add_argument("--checkpoint", help="checkpoint path; passed to the policy and hashed into the results")
    sr.add_argument("--access", help="information-access tier for every task (preset name or JSON file)")
    sr.add_argument(
        "--camera",
        action="append",
        help="add a preset camera to the scenario (`front`, `side`, `top`; repeatable), for the visual tiers",
    )
    sr.add_argument("--device", help="torch device (default: cuda if available)")
    sr.add_argument("--out-dir", help="default: bench_results/suites/<suite>/<sim>")
    sr.add_argument(
        "--isolate",
        choices=("auto", "always", "never"),
        default="auto",
        help="run each task in its own interpreter (auto: for backends that boot once per process, e.g. isaacsim)",
    )
    _add_remote_args(sr)
    sr.set_defaults(func=_bench_suite_run)
    sp = sus.add_parser("report", help="score existing result files against a suite")
    sp.add_argument("--suite", required=True)
    sp.add_argument("--access", help="information-access tier the results were run under (as in `suite run`)")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--out-dir", help="also write suite.json / suite.md here")
    sp.add_argument("--policy", help="policy name to score when the results hold several")
    sp.set_defaults(func=_bench_suite_report)
    a = bs.add_parser("access", help="list the information-access tiers (`--access` presets)")
    a.set_defaults(func=_bench_access)
    h = bs.add_parser("health", help="task health gate: the replay baseline on every task that ships demos")
    h.add_argument("--sim", required=True, help="simulator backend the demos were recorded on, e.g. mujoco")
    h.add_argument("--task", action="append", help="task(s) to check (repeatable; default: every task with demos)")
    h.add_argument("--robot", action="append", help="robot config name(s) (repeatable; default: the task's)")
    h.add_argument("--num-envs", type=int, default=1)
    h.add_argument("--episodes", type=int, default=5)
    h.add_argument("--max-steps", type=int, help="evaluator-side episode cap (default: the task's)")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--demos", help="demo file or directory to replay instead of the task's own (one --task only)")
    h.add_argument("--threshold", type=float, default=0.8, help="replay success rate a healthy task reaches")
    h.add_argument(
        "--floor",
        action="store_true",
        help="also score the hold policy: a task it succeeds on is `unreliable` (the checker accepts the initial state)",
    )
    h.add_argument(
        "--repeat", type=int, default=1, help="replay N times; a different verdict on the same instances is `flaky`"
    )
    h.add_argument("--device", help="torch device (default: cuda if available)")
    h.add_argument("--out-dir", default="bench_results/health", help="result files, health.json and health.md")
    h.add_argument(
        "--isolate",
        choices=("auto", "always", "never"),
        default="auto",
        help="run each task in its own subprocess (auto: for backends that boot once per process, e.g. isaacsim)",
    )
    h.set_defaults(func=_bench_health)
    s = bs.add_parser("summarize", help="pool result files / directories into a Markdown table")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=_bench_summarize)
    mx = bs.add_parser("matrix", help="methods x access tiers: task-macro success from result files / directories")
    mx.add_argument("paths", nargs="+")
    mx.set_defaults(func=_bench_matrix)
    v = bs.add_parser("validate", help="check result files against the schema")
    v.add_argument("paths", nargs="+")
    v.set_defaults(func=_bench_validate)


def _robot_list(args: argparse.Namespace) -> int:
    """Every robot config the robot packages define (``robot check --all`` runs on this list)."""
    from metasim.utils.setup_util import list_robots

    robots = list_robots()
    if args.json:
        print(json.dumps(robots, indent=2))
    else:
        for name, where in robots.items():
            print(f"{name:32s} {where}")
        print(f"{len(robots)} robot(s)", file=sys.stderr)
    return 0


def _robot_sweep_table(reports) -> str:
    """One row per robot: each check's verdict and the overall one (the nightly gate publishes this)."""
    checks = ("config", "assets", "kinematics", "limits", "actuators", "hold")
    lines = ["| robot | " + " | ".join(checks) + " | verdict |", "|---|" + "---|" * (len(checks) + 1)]
    from metasim.utils.robot_check import row_verdict

    for r in reports:
        by = {row.check: row for row in r.rows}
        cells = ["-" if by.get(c) is None else row_verdict(by[c]) for c in checks]
        lines.append(f"| {r.robot} | " + " | ".join(cells) + f" | {r.verdict} |")
    n_na = sum(not r.applicable for r in reports)
    n_ok = sum(r.ok and r.applicable for r in reports)
    n_fail = len(reports) - n_ok - n_na
    lines.append(f"\n{len(reports)} robot(s), {n_ok} pass, {n_fail} fail, {n_na} n/a (no asset for this backend)")
    for r in reports:
        for row in r.rows:
            if row.applicable and not row.ok:
                lines.append(f"- {r.robot} / {row.check}: {row.detail}")
    return "\n".join(lines)


def _robot_check_isolated(name: str, args: argparse.Namespace):
    """One robot's self-check in a child interpreter (Isaac Sim boots Kit once per process and may end it in close()).

    The cheap rows (config, assets, kinematics, limits, actuators) run here first; the child, which costs a simulator
    boot, is spawned only when they pass, so a robot without an asset for this backend is a named failure in
    milliseconds. The child writes its report (``--out-dir``) before it exits; a child that dies without one is reported as
    a failed ``hold`` row with its exit code and last log line.
    """
    import subprocess

    from metasim.utils.robot_check import CheckRow, RobotReport, check_robot
    from metasim.utils.setup_util import get_robot

    cfg = get_robot(name)
    report = check_robot(cfg.name, sim=args.sim, steps=args.steps, hold=False)
    if not report.applicable:
        report.rows.append(CheckRow("hold", False, f"n/a: no asset declared for {args.sim}", applicable=False))
        return report
    if not report.ok or args.no_hold:
        if not args.no_hold:
            report.rows.append(CheckRow("hold", False, "not attempted: a cheaper check already failed"))
        return report
    import tempfile

    scratch = None if args.out_dir else tempfile.TemporaryDirectory(prefix="robot_check_")
    child_dir = os.path.join(args.out_dir, "children", name) if args.out_dir else scratch.name
    os.makedirs(child_dir, exist_ok=True)
    report_path = os.path.join(child_dir, "robots.json")
    if os.path.exists(report_path):
        os.remove(report_path)  # a child that dies must not be read as the last run's report
    cmd = [sys.executable, "-m", "metasim", "robot", "check", name, "--sim", args.sim, "--steps", str(args.steps)]
    cmd += ["--isolate", "never", "--out-dir", child_dir]  # the report is read from the child's robots.json, not stdout
    log_path = None
    if args.out_dir:
        os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
        log_path = os.path.join(args.out_dir, "logs", f"{name}.log")
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if log_path:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(proc.stdout)
                fh.write(proc.stderr)
        try:
            with open(report_path, encoding="utf-8") as fh:
                child = json.load(fh)["robots"][0]
            rows = [
                CheckRow(r["check"], bool(r["ok"]), str(r["detail"]), bool(r.get("applicable", True)))
                for r in child["rows"]
            ]
            return RobotReport(robot=child["robot"], simulator=child["simulator"], rows=rows)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            lines = [ln.strip() for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
            # the exception line of a traceback ("ValueError: ...") beats any later log line about closing the
            # simulator, which beats the last line of all ("SimulationContext cleared")
            raised = [ln for ln in lines if re.match(r"^[A-Za-z_.]+(Error|Exception)\b", ln)]
            markers = ("Error", "error", "Exception", "Traceback", "fault", "abort")
            flagged = [ln for ln in lines if any(m in ln for m in markers) and not ln.startswith("[")]
            last = (raised or flagged or lines or [""])[-1]
            report.rows.append(
                CheckRow("hold", False, f"child exited {proc.returncode} without a report: {last.strip()[:200]}")
            )
            return report
    finally:
        if scratch is not None:
            scratch.cleanup()


def _robot_check(args: argparse.Namespace) -> int:
    """The robot self-check (config, assets, kinematics, limits, actuators, hold); exit 1 when a check fails."""
    from metasim.utils.robot_check import CheckRow, RobotReport, check_robot

    names = list(args.robot or [])
    if args.all:
        from metasim.utils.setup_util import list_robots

        names = list(list_robots())
    if not names:
        print("robot check needs robot name(s) or --all", file=sys.stderr)
        return 2
    isolate = args.isolate == "always" or (
        args.isolate == "auto" and args.sim in ISOLATED_HEALTH_BACKENDS and len(names) > 1
    )
    reports = []
    for name in names:
        try:
            if isolate:
                reports.append(_robot_check_isolated(name, args))
            else:
                reports.append(check_robot(name, sim=args.sim, steps=args.steps, hold=not args.no_hold))
        except Exception as err:  # one robot that cannot even be resolved must not end the sweep
            row = CheckRow("config", False, f"{type(err).__name__}: {str(err)[:200]}")
            reports.append(RobotReport(robot=name, simulator=args.sim, rows=[row]))
        if args.all and not args.json:
            r = reports[-1]
            print(f"{r.robot:32s} {r.verdict}", file=sys.stderr, flush=True)
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    elif args.all:
        print(_robot_sweep_table(reports))
    else:
        print("\n\n".join(r.table() for r in reports))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "robots.json"), "w", encoding="utf-8") as fh:
            json.dump({"simulator": args.sim, "robots": [r.to_dict() for r in reports]}, fh, indent=2)
            fh.write("\n")
        with open(os.path.join(args.out_dir, "robots.md"), "w", encoding="utf-8") as fh:
            fh.write(_robot_sweep_table(reports) + "\n")
    sys.stdout.flush()
    if not all(r.ok for r in reports):
        # Isaac Sim ends the process with status 0 inside close(); a failed check must not be laundered into
        # exit 0 by its own cleanup, so the process ends here with the verdict and the simulator dies with it.
        sys.stderr.flush()
        os._exit(1)
    from metasim.utils.robot_check import close_handlers

    close_handlers()  # may end the process on Isaac Sim; the report is out and the verdict is pass
    return 0


def _robot_add(args) -> int:
    """Generate a RobotCfg from an MJCF, write it, import it back and run the self-check on it."""
    from metasim.utils.robot_check import check_robot_cfg, close_handlers
    from metasim.utils.robot_kit import add_robot, load_cfg_from_file

    defaults = {}
    for item in args.default or []:
        if "=" not in item:
            print(f"--default expects joint=value, got {item!r}", file=sys.stderr)
            return 2
        joint, value = item.split("=", 1)
        defaults[joint.strip()] = float(value)
    out = args.out or os.path.join("roboverse_pack", "robots", f"{args.name}_cfg.py")
    try:
        spec, cls = add_robot(
            mjcf=args.mjcf,
            name=args.name,
            out=out,
            usd=args.usd,
            urdf=args.urdf,
            ee_body=args.ee_body,
            gripper_joints=args.gripper_joints,
            class_name=args.class_name,
            overwrite=args.overwrite,
            defaults=defaults or None,
            command="python -m metasim " + " ".join(sys.argv[1:]),
        )
    except (FileExistsError, ValueError) as err:
        print(str(err), file=sys.stderr)
        return 1
    print(
        f"wrote {out}: class {cls}, {len(spec.joints)} joints ({len(spec.gripper_joints)} gripper), ee body {spec.ee_body!r}"
    )
    for note in spec.notes:
        print(f"  note: {note}")
    if args.no_check:
        return 0
    cfg = load_cfg_from_file(out, cls)
    report = check_robot_cfg(cfg, sim=args.sim, steps=args.steps, hold=not args.no_hold)
    print()
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.table())
    print(
        f"\nnext: export {cls} from roboverse_pack/robots/__init__.py, review the TODO lines in {out}, "
        f"then `python -m metasim robot check {args.name}` and one gated task."
    )
    sys.stdout.flush()
    if not report.ok:
        sys.stderr.flush()
        os._exit(1)
    close_handlers()
    return 0


def _robot_parity(args) -> int:
    """Track one canned joint trajectory on two backends and compare the traces joint by joint."""
    from metasim.utils.robot_check import TrackingTrace, close_handlers, compare_traces, track_trajectory
    from metasim.utils.setup_util import get_robot

    cfg = get_robot(args.robot)
    traces: list[TrackingTrace] = []
    for path in args.load or []:
        with open(path, encoding="utf-8") as fh:
            traces.append(TrackingTrace.from_dict(json.load(fh)))
    for sim in args.sims or []:
        traces.append(
            track_trajectory(
                cfg,
                sim,
                steps=args.steps,
                period_steps=args.period,
                amplitude_frac=args.amplitude,
                amplitude_max=args.amplitude_max,
                settle_steps=args.settle,
            )
        )
    if args.save:
        if len(args.sims or []) != 1:
            print("--save stores the trace of exactly one --sims backend", file=sys.stderr)
            return 2
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(traces[-1].to_dict(), fh)
            fh.write("\n")
        print(f"saved {traces[-1].simulator} trace to {args.save}")
    if len(traces) < 2:
        if args.save:
            close_handlers()
            return 0
        print("robot parity needs two traces: --sims a b, or --load <trace.json> plus --sims b", file=sys.stderr)
        return 2
    report = compare_traces(cfg, traces[0], traces[1], tol=args.tol, tol_slide=args.tol_slide)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.table())
    sys.stdout.flush()
    if not report.ok:
        sys.stderr.flush()
        os._exit(1)  # Isaac Sim ends the process with status 0 in close(); the verdict must survive that
    close_handlers()
    return 0


def _add_robot(sub) -> None:
    r = sub.add_parser(
        "robot", help="robot configs: generate one from an MJCF, self-check before a task is built on one"
    )
    rs = r.add_subparsers(dest="robot_command", required=True)
    a = rs.add_parser(
        "add", help="generate a RobotCfg file from an MJCF (joints, limits, gains, defaults, tool) and check it"
    )
    a.add_argument("--mjcf", required=True, help="the robot's MJCF (local path or roboverse_data/... on the Hub)")
    a.add_argument("--name", required=True, help="the robot name every task and result will use, e.g. so_arm101")
    a.add_argument("--out", help="where to write the config (default roboverse_pack/robots/<name>_cfg.py)")
    a.add_argument("--class-name", help="config class name (default from --name, e.g. SoArm101Cfg)")
    a.add_argument("--usd", help="USD path to declare for Isaac Sim")
    a.add_argument("--urdf", help="URDF path to declare for the URDF backends")
    a.add_argument("--ee-body", help="the tool body (default: the parent body of the gripper joints)")
    a.add_argument(
        "--gripper-joints", nargs="+", help="joints marked is_ee (default: slide joints and finger/jaw/gripper names)"
    )
    a.add_argument("--default", action="append", metavar="JOINT=VALUE", help="default joint position (repeatable)")
    a.add_argument("--overwrite", action="store_true", help="replace an existing file at --out")
    a.add_argument("--no-check", action="store_true", help="write the file only, skip the self-check")
    a.add_argument("--sim", default="mujoco", help="backend for the self-check (default mujoco)")
    a.add_argument("--steps", type=int, default=200, help="simulator steps to hold the default pose")
    a.add_argument("--no-hold", action="store_true", help="skip the hold test in the self-check")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=_robot_add)
    pa = rs.add_parser("parity", help="track one canned joint trajectory on two backends and compare the traces")
    pa.add_argument("robot", help="robot config name, e.g. franka")
    pa.add_argument(
        "--sims", nargs="+", help="backends to run, e.g. mujoco isaacsim (the first two traces are compared)"
    )
    pa.add_argument("--load", action="append", metavar="TRACE.json", help="a trace saved with --save (repeatable)")
    pa.add_argument("--save", metavar="TRACE.json", help="save the one backend's trace for a later run in another env")
    pa.add_argument("--steps", type=int, default=400, help="control steps of the trajectory (default 400 = 6 s)")
    pa.add_argument("--period", type=int, default=200, help="sinusoid period in control steps (default 200 = 3 s)")
    pa.add_argument(
        "--amplitude", type=float, default=0.25, help="amplitude as a fraction of the half range (default 0.25)"
    )
    pa.add_argument(
        "--amplitude-max",
        type=float,
        default=0.3,
        help="cap on a hinge joint's amplitude, rad (default 0.3; 0 = no cap)",
    )
    pa.add_argument("--settle", type=int, default=50, help="control steps at the default pose before the trajectory")
    pa.add_argument(
        "--tol", type=float, default=0.05, help="largest allowed cross-engine difference for a hinge joint, rad"
    )
    pa.add_argument("--tol-slide", type=float, default=0.005, help="the same for a slide joint, metres")
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=_robot_parity)
    ls = rs.add_parser("list", help="every robot config the robot packages define")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=_robot_list)
    c = rs.add_parser("check", help="config, assets, kinematics, limits, actuators and a hold test on a backend")
    c.add_argument("robot", nargs="*", help="robot config name(s), e.g. franka")
    c.add_argument("--all", action="store_true", help="every robot `robot list` names (the sweep; one table)")
    c.add_argument("--out-dir", help="write robots.json and robots.md here (the nightly gate publishes them)")
    c.add_argument(
        "--isolate",
        choices=("auto", "always", "never"),
        default="auto",
        help="run each robot's simulator check in its own process (auto: on backends that boot once per process, "
        "e.g. isaacsim, when checking several robots)",
    )
    c.add_argument("--sim", default="mujoco", help="backend for the assets and hold checks (default mujoco)")
    c.add_argument("--steps", type=int, default=200, help="simulator steps to hold the default pose")
    c.add_argument(
        "--no-hold", action="store_true", help="skip the simulator hold test (config, assets, kinematics only)"
    )
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_robot_check)


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected sub-command; returns the process exit status."""
    apply_headless_gl()  # one decision about MuJoCo's GL backend, before any command imports mujoco
    parser = argparse.ArgumentParser(prog="python -m metasim")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="check installed simulator versions against the supported ranges")
    d.add_argument(
        "--backend",
        action="append",
        choices=[s.value for s in BACKEND_REQUIREMENTS],
        help="limit to one backend (repeatable)",
    )
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_doctor)
    a = sub.add_parser("api-snapshot", help="compare the public API against metasim/test/api_snapshot.json")
    a.add_argument("--update", action="store_true", help="rewrite the snapshot from the current code")
    a.set_defaults(func=_api_snapshot)
    _add_bench(sub)
    _add_robot(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
