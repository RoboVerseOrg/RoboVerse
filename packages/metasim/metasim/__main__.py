"""``python -m metasim <command>``.

* ``api-snapshot`` — public API surface vs ``metasim/test/api_snapshot.json`` (``--update`` to accept).
* ``doctor`` — every simulator backend: installed package versions vs the supported range and the
  last verified version (``metasim/sim/_versions.py``). Exit status 1 when an installed backend is
  outside its supported range, so it can gate an environment in CI or a Dockerfile.
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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
