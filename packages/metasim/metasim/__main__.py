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
import sys

from metasim.constants import SimType
from metasim.sim._versions import BACKEND_REQUIREMENTS, doctor, format_reports


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


def _make_env(task: str, *, sim: str, robots: list[str] | None, num_envs: int, gui: bool, device: str | None):
    """Build the registry task ``task`` on ``sim`` from its class-level scenario."""
    import torch

    from metasim.task.registry import get_task_class

    task_cls = get_task_class(task)
    if getattr(task_cls, "scenario", None) is None:
        raise SystemExit(f"{task} has no class-level scenario; it cannot be built from the command line")
    updates = {"simulator": sim, "num_envs": num_envs, "headless": not gui}
    if robots:
        updates["robots"] = list(robots)
    scenario = task_cls.scenario.replace(**updates)
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
    import re
    import time

    from metasim.bench import EvalProtocol, evaluate

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
        )
    except ValueError as err:
        raise SystemExit(f"invalid protocol: {err}") from err
    policy, policy_info = _make_policy(args)
    env = _make_env(
        args.task, sim=args.sim, robots=args.robot, num_envs=args.num_envs, gui=args.gui, device=args.device
    )
    try:
        result = evaluate(
            env, policy, protocol, policy_info=policy_info, on_episode=_report_episode, task_name=args.task
        )
    finally:
        env.close()
    out = args.out
    if out is None:
        robots = "+".join(result.robots) or "norobot"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{args.task}__{robots}__{args.sim}__{policy.name}__{stamp}")
        out = f"bench_results/{slug}.json"
    path = result.save(out)
    print(
        f"{args.task} [{', '.join(result.robots)} @ {args.sim}] {policy.name}: {_score_line(result.summary)}, "
        f"mean {result.summary['mean_steps']:.1f} steps -> {path}"
    )
    return 0


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


def _bench_suite_run(args: argparse.Namespace) -> int:
    """Every task of a suite (or one split) under the suite protocol; one result file per task."""
    import dataclasses

    from metasim.bench import evaluate, get_suite

    suite = get_suite(args.suite)
    sim = args.sim or suite.simulator
    if not sim:
        raise SystemExit(f"suite {suite.name!r} names no simulator; pass --sim")
    out_dir = args.out_dir or os.path.join("bench_results", "suites", suite.name, sim)
    overrides = {k: v for k, v in {"episodes": args.episodes, "seed": args.seed}.items() if v is not None}
    results, failures = [], []
    for split, entry in suite.entries(args.split):
        print(f"=== {suite.name}/{split}: {entry.task}", file=sys.stderr, flush=True)
        try:
            protocol = suite.protocol_for(entry)
            if overrides:
                protocol = dataclasses.replace(protocol, **overrides)
            policy, policy_info = _make_policy(args)
            env = _make_env(
                entry.task,
                sim=sim,
                robots=list(entry.robots) or None,
                num_envs=args.num_envs,
                gui=False,
                device=args.device,
            )
            try:
                result = evaluate(
                    env, policy, protocol, policy_info=policy_info, on_episode=_report_episode, task_name=entry.task
                )
            finally:
                env.close()
        except Exception as err:  # one broken task is reported as missing, the others still run
            detail = f"{type(err).__name__}: {err}".splitlines()[0][:300]
            print(f"{entry.task}: ERROR {detail}", file=sys.stderr)
            failures.append({"split": split, "task": entry.task, "error": detail})
            continue
        result.notes["suite"] = {"name": suite.name, "split": split}
        robots = "+".join(result.robots) or "norobot"
        result.save(os.path.join(out_dir, f"{split}__{entry.task}__{robots}__{policy.name}.json"))
        print(f"{entry.task}: {_score_line(result.summary)}", file=sys.stderr)
        results.append(result)
    report = _write_suite_report(suite, results, out_dir)
    if failures:
        with open(os.path.join(out_dir, "errors.json"), "w") as f:
            json.dump(failures, f, indent=2)
    complete = all(block["complete"] for block in report["splits"].values())
    return 0 if complete and not failures else 1


def _bench_suite_report(args: argparse.Namespace) -> int:
    from metasim.bench import get_suite

    suite = get_suite(args.suite)
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


HEALTH_STATUSES = ("ok", "low", "undefined", "unsupported", "no-demos", "error")
"""Outcomes of ``bench health`` for one task, in the order the table sorts them."""


def _health_task(name: str, args: argparse.Namespace) -> dict:
    """Replay one task's demos under the protocol and classify the outcome.

    Returns ``{"task", "status", "success_rate", "episodes", "result", "detail"}``: ``ok`` at or above the
    threshold, ``low`` below it, ``undefined`` when the task declares no success signal, ``unsupported`` when
    it does not run on the backend, ``no-demos`` when it ships no demo file, ``error`` with the exception text.
    """
    import torch

    from metasim.bench import EvalProtocol, ReplayPolicy, evaluate
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
        try:
            # the demo file is a property of the built task: some tasks pick it per robot or in __init__
            if getattr(env, "traj_filepath", None) is None:
                row.update(status="no-demos")
                return row
            result = evaluate(env, ReplayPolicy(), protocol, policy_info={"spec": "replay"}, task_name=name)
        finally:
            env.close()
        robots = "+".join(result.robots) or "norobot"
        path = result.save(os.path.join(args.out_dir, f"{name}__{robots}__{args.sim}__replay.json"))
        summary = result.summary
        row.update(episodes=summary["episodes"], result=str(path))
        if not summary["success_defined"]:
            row.update(status="undefined")
        else:
            rate = float(summary["success_rate"])
            row.update(success_rate=rate, status="ok" if rate >= args.threshold else "low")
        return row
    except Exception as err:  # one broken task must not hide the others: it is reported as a row
        detail = f"{type(err).__name__}: {err}".splitlines()[0][:200]
        # an object with no description for this backend is a backend the task does not support
        status = "unsupported" if "has no" in detail and "asset path set for the" in detail else "error"
        row.update(status=status, detail=detail)
        return row


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
    broken (asset, checker or physics), not hard; the gate exits 1 when any task is ``low`` or
    ``error`` so it can guard a release. Results go to ``--out-dir`` as ordinary result files plus a
    ``health.json`` with the rows.
    """
    from metasim.task.registry import list_tasks

    names = list(args.task) if args.task else list_tasks()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for name in names:
        row = _health_task(name, args)
        if row["status"] == "no-demos" and not args.task:
            continue  # the default sweep is over tasks with demos; an explicit --task is always reported
        rows.append(row)
        print(f"{row['task']:48s} {row['status']:12s} {row['detail']}", file=sys.stderr)
    table = _health_table(rows, args.threshold)
    with open(os.path.join(args.out_dir, "health.json"), "w") as f:
        json.dump(
            {"simulator": args.sim, "threshold": args.threshold, "episodes": args.episodes, "rows": rows}, f, indent=2
        )
    with open(os.path.join(args.out_dir, "health.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    return 1 if any(r["status"] in ("low", "error") for r in rows) else 0


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
    r.add_argument("--device", help="torch device (default: cuda if available)")
    r.add_argument("--gui", action="store_true", help="run with the simulator viewer (default headless)")
    r.add_argument("--out", help="result path (default: bench_results/<task>__<robot>__<sim>__<policy>__<time>.json)")
    _add_remote_args(r)
    r.set_defaults(func=_bench_run)
    sv = bs.add_parser("serve", help="serve a policy over HTTP for `bench run --policy-url` (run in the policy's env)")
    sv.add_argument("--policy", required=True, help=f"one of {list(_BUILTIN_POLICIES)} or 'pkg.module:attr'")
    sv.add_argument("--policy-arg", action="append", metavar="KEY=VALUE", help="constructor argument (repeatable)")
    sv.add_argument("--checkpoint", help="checkpoint path; passed to the policy and reported in the metadata")
    sv.add_argument("--host", default="127.0.0.1", help="interface to bind (default: this machine only)")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--device", help="torch device the policy runs on (default cpu)")
    sv.add_argument("--metadata", action="append", metavar="KEY=VALUE", help="extra metadata field (repeatable)")
    sv.set_defaults(func=_bench_serve)
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
    sr.add_argument("--device", help="torch device (default: cuda if available)")
    sr.add_argument("--out-dir", help="default: bench_results/suites/<suite>/<sim>")
    _add_remote_args(sr)
    sr.set_defaults(func=_bench_suite_run)
    sp = sus.add_parser("report", help="score existing result files against a suite")
    sp.add_argument("--suite", required=True)
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--out-dir", help="also write suite.json / suite.md here")
    sp.add_argument("--policy", help="policy name to score when the results hold several")
    sp.set_defaults(func=_bench_suite_report)
    h = bs.add_parser("health", help="task health gate: the replay baseline on every task that ships demos")
    h.add_argument("--sim", required=True, help="simulator backend the demos were recorded on, e.g. mujoco")
    h.add_argument("--task", action="append", help="task(s) to check (repeatable; default: every task with demos)")
    h.add_argument("--robot", action="append", help="robot config name(s) (repeatable; default: the task's)")
    h.add_argument("--num-envs", type=int, default=1)
    h.add_argument("--episodes", type=int, default=5)
    h.add_argument("--max-steps", type=int, help="evaluator-side episode cap (default: the task's)")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--threshold", type=float, default=0.8, help="replay success rate a healthy task reaches")
    h.add_argument("--device", help="torch device (default: cuda if available)")
    h.add_argument("--out-dir", default="bench_results/health", help="result files, health.json and health.md")
    h.set_defaults(func=_bench_health)
    s = bs.add_parser("summarize", help="pool result files / directories into a Markdown table")
    s.add_argument("paths", nargs="+")
    s.set_defaults(func=_bench_summarize)
    v = bs.add_parser("validate", help="check result files against the schema")
    v.add_argument("paths", nargs="+")
    v.set_defaults(func=_bench_validate)


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the selected sub-command; returns the process exit status."""
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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
