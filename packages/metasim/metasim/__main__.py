"""``python -m metasim <command>``.

* ``api-snapshot`` — public API surface vs ``metasim/test/api_snapshot.json`` (``--update`` to accept).
* ``doctor`` — every simulator backend: installed package versions vs the supported range and the
  last verified version (``metasim/sim/_versions.py``). Exit status 1 when an installed backend is
  outside its supported range, so it can gate an environment in CI or a Dockerfile.
* ``bench list | run | summarize | validate`` — the benchmark protocol (``metasim.bench``): list the
  registered tasks, score a policy on one task under a protocol and write a result JSON, pool
  results into a leaderboard table, check a result file against the schema.
"""

from __future__ import annotations

import argparse
import json
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


def _bench_run(args: argparse.Namespace) -> int:
    import copy
    import re
    import time

    import torch

    from metasim.bench import EvalProtocol, checkpoint_sha256, evaluate, load_policy
    from metasim.task.registry import get_task_class

    task_cls = get_task_class(args.task)
    scenario = copy.deepcopy(task_cls.scenario)
    if scenario is None:
        raise SystemExit(f"{args.task} has no class-level scenario; it cannot be built from the command line")
    updates = {"simulator": args.sim, "num_envs": args.num_envs, "headless": not args.gui}
    if args.robot:
        updates["robots"] = list(args.robot)
    scenario = scenario.update(**updates)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    protocol = EvalProtocol(
        episodes=args.episodes,
        success_rule=args.success_rule,
        hold_steps=args.hold_steps,
        max_episode_steps=args.max_steps,
        seed=args.seed,
        instances=tuple(args.instance) if args.instance else None,
        name=args.protocol_name,
    )
    policy_kwargs = _parse_kv(args.policy_arg)
    if args.checkpoint:
        policy_kwargs.setdefault("checkpoint", args.checkpoint)
    policy = load_policy(args.policy, **policy_kwargs)
    policy_info = {"spec": args.policy, "args": policy_kwargs}
    if args.checkpoint:
        policy_info.update({"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_sha256(args.checkpoint)})

    env = task_cls(scenario, device)
    try:

        def report(ep):
            print(
                f"episode {ep.index:4d}  env {ep.env_id}  instance {ep.instance_id}  "
                f"{'SUCCESS' if ep.success else 'fail   '}  steps={ep.steps}  return={ep.return_sum:.3f}",
                file=sys.stderr,
            )

        result = evaluate(env, policy, protocol, policy_info=policy_info, on_episode=report, task_name=args.task)
    finally:
        env.close()
    summary = result.summary
    out = args.out
    if out is None:
        robots = "+".join(result.robots) or "norobot"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{args.task}__{robots}__{args.sim}__{policy.name}__{stamp}")
        out = f"bench_results/{slug}.json"
    path = result.save(out)
    low, high = summary["success_rate_ci95"]
    print(
        f"{args.task} [{', '.join(result.robots)} @ {args.sim}] {policy.name}: "
        f"{summary['successes']}/{summary['episodes']} = {summary['success_rate']:.3f} "
        f"(95% CI {low:.3f}-{high:.3f}), mean {summary['mean_steps']:.1f} steps -> {path}"
    )
    return 0


def _bench_summarize(args: argparse.Namespace) -> int:
    from pathlib import Path

    from metasim.bench import BenchmarkResult, summarize

    paths = []
    for p in args.paths:
        p = Path(p)
        paths.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    results = [BenchmarkResult.load(p) for p in paths]
    if not results:
        print("no result files found", file=sys.stderr)
        return 1
    print(summarize(results))
    return 0


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


def _add_bench(sub) -> None:
    from metasim.bench import BUILTIN_POLICIES, SUCCESS_RULES

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
    r.add_argument("--episodes", type=int, default=10)
    r.add_argument("--policy", default="hold", help=f"one of {sorted(BUILTIN_POLICIES)} or 'pkg.module:attr'")
    r.add_argument("--policy-arg", action="append", metavar="KEY=VALUE", help="constructor argument (repeatable)")
    r.add_argument("--checkpoint", help="checkpoint path; passed to the policy and hashed into the result")
    r.add_argument("--success-rule", choices=SUCCESS_RULES, default="first")
    r.add_argument("--hold-steps", type=int, default=1)
    r.add_argument("--max-steps", type=int, help="evaluator-side episode cap (default: the task's)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--instance", type=int, action="append", help="explicit instance index (repeatable)")
    r.add_argument("--protocol-name", default="default")
    r.add_argument("--device", help="torch device (default: cuda if available)")
    r.add_argument("--gui", action="store_true", help="run with the simulator viewer (default headless)")
    r.add_argument("--out", help="result path (default: bench_results/<task>__<robot>__<sim>__<policy>__<time>.json)")
    r.set_defaults(func=_bench_run)
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
