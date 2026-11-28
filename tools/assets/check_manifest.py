"""Every ``roboverse_data/...`` path the content packs name must exist, locally or on the Hub.

This week's curation retired 2587 generated ManiSkill tasks whose assets or demos were never on the Hub.
This check runs on every pull request so a task cannot reference an asset that does not exist::

    python tools/assets/check_manifest.py                 # every path under roboverse_pack
    python tools/assets/check_manifest.py --paths roboverse_pack/tasks/maniskill   # a subtree
    python tools/assets/check_manifest.py --hub-list hub_files.txt                 # offline, from a saved listing

It scans string literals in Python files for ``roboverse_data/<...>`` references, resolves a reference to the
Hub listing (``RoboVerseOrg/roboverse_data``) by exact file or directory prefix, and exits 1 with one line per
missing reference. Paths built at run time (f-strings with a variable) are reported separately as "dynamic"
and never fail the check.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HUB_REPO = "RoboVerseOrg/roboverse_data"
REF = re.compile(r"roboverse_data/[A-Za-z0-9_./\-]+")


def hub_files(cache: str | None) -> set[str]:
    """The Hub dataset's file list, from ``cache`` when given (one path per line) else the Hub API."""
    if cache and os.path.exists(cache):
        return {line.strip() for line in open(cache) if line.strip()}
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(HUB_REPO, repo_type="dataset"))
    if cache:
        Path(cache).write_text("\n".join(sorted(files)))
    return files


def references(paths: list[Path]) -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[tuple[str, int]]]]:
    """``roboverse_data/...`` literals -> [(file, line)]; a literal that is part of an f-string or a concatenation is dynamic."""
    static: dict[str, list[tuple[str, int]]] = {}
    dynamic: dict[str, list[tuple[str, int]]] = {}
    for root in paths:
        for py in sorted(root.rglob("*.py") if root.is_dir() else [root]):
            if "__pycache__" in py.parts:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = py.relative_to(REPO).as_posix()
            fstring_parts: set[int] = set()  # ast.walk visits an f-string's constant parts again as plain Constants
            for scope in ast.walk(tree):  # docstrings are prose about the data, not references to it
                if isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and scope.body:
                    first = scope.body[0]
                    if (
                        isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)
                    ):
                        fstring_parts.add(id(first.value))
            for node in ast.walk(tree):
                if isinstance(node, ast.JoinedStr):  # f-string: dynamic
                    for part in node.values:
                        if isinstance(part, ast.Constant) and isinstance(part.value, str):
                            fstring_parts.add(id(part))
                            for m in REF.findall(part.value):
                                dynamic.setdefault(m, []).append((rel, node.lineno))
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "roboverse_data/" in node.value
                    and id(node) not in fstring_parts
                ):
                    for m in REF.findall(node.value):
                        static.setdefault(m, []).append((rel, node.lineno))
    # a literal that is also the constant part of some f-string stays a static reference where it is written in full
    return static, dynamic


def resolve(ref: str, files: set[str], prefixes: set[str], *, hub_only: bool = False) -> bool:
    """``ref`` names a Hub file, a Hub directory (prefix), or (unless ``hub_only``) exists locally."""
    key = ref[len("roboverse_data/") :].rstrip("/")
    if key in files or key in prefixes:
        return True
    if not hub_only and (REPO / ref).exists():
        return True
    # a stem reference (``trajs/<task>/v2``) is a directory; a ``.xml`` next to ``assets/`` dirs may be hashed
    return any(f.startswith(key + "/") for f in files) if "/" in key else False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paths", nargs="*", default=["roboverse_pack"], help="files or directories to scan")
    ap.add_argument("--hub-list", help="cache file of the Hub listing (written when missing)")
    ap.add_argument("--allow", nargs="*", default=[], help="reference prefixes that are allowed to be absent")
    ap.add_argument(
        "--hub-only",
        action="store_true",
        help="ignore local files: what CI sees (a file only on this machine is not published)",
    )
    args = ap.parse_args(argv)
    files = hub_files(args.hub_list)
    prefixes = {p for f in files for p in [f.rsplit("/", 1)[0]] if "/" in f}
    static, dynamic = references([REPO / p for p in args.paths])
    allowed = lambda r: any(r.startswith(a) for a in args.allow)  # noqa: E731
    missing = {
        r: locs
        for r, locs in static.items()
        if not resolve(r, files, prefixes, hub_only=args.hub_only) and not allowed(r)
    }
    local_only = sorted(
        r
        for r in static
        if not allowed(r) and resolve(r, files, prefixes) and not resolve(r, files, prefixes, hub_only=True)
    )
    if local_only:
        print(
            f"{len(local_only)} reference(s) exist only on this machine, not on the Hub (upload them): "
            + ", ".join(local_only[:8])
            + (" ..." if len(local_only) > 8 else "")
        )
    print(
        f"{len(static)} static references, {len(dynamic)} dynamic, {len(missing)} missing ({len(files)} files on the Hub)"
    )
    for ref, locs in sorted(missing.items()):
        where = ", ".join(f"{f}:{n}" for f, n in locs[:3]) + (" ..." if len(locs) > 3 else "")
        print(f"MISSING {ref}  <- {where}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
