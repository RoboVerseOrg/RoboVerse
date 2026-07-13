"""Guardrail: vendored / adapted third-party code must carry its attribution header.

RoboVerse is Apache-2.0 and vendors code from MIT / Apache-2.0 / BSD-3 projects. Those licenses
require the upstream notice to travel with the copy, and Apache-2.0 §4(b) requires a statement of
changes. `AGENTS.md` therefore mandates a header on every copied/adapted file:

    # Copyright (c) <year> <holder>
    # SPDX-License-Identifier: <license>
    #
    # Adapted from <Project> (<url>).
    # Changes: <...>
    # Full license: <path to the license text in this repo>

This test *scans* the tree rather than trusting a hand-maintained list of vendored paths (a list
only ever proves that the files someone remembered are fine). It fails when:

* a file shows a **copy-tell** — "copied/adapted/ported/vendored from <Project>", "based on
  <url>", or an upstream copyright line that is not RoboVerse's — but has no complete header;
* a header's ``Full license:`` path does not exist, or does not contain the license text the
  SPDX tag claims (an MIT header pointing at a file with no MIT grant is worthless);
* `THIRD_PARTY_NOTICES.md` and the tree disagree in either direction: a row names a path that
  does not exist, or a headered file is not covered by any row.

Three escape hatches exist, all narrow and all machine-checked here:

1. **Reimplemented / Integration rows.** A `THIRD_PARTY_NOTICES.md` row whose *Relationship* is
   "Reimplemented" (written from the upstream's ideas, no source copied) or "Integration" (we
   import the upstream package) does not need file headers — but the row must exist, and the
   paths it names must exist.
2. **The "Unresolved" section.** Code whose provenance/license could not be established is
   recorded there instead of being silently headered with a guessed license. Those paths are
   exempt from the header rule; the paths must exist, and the list is the *only* place such code
   may hide.
3. **`# RoboVerse-original: <reason>`.** A file inside a vendored tree that contains no upstream
   code must say so — silence is indistinguishable from a forgotten header. It is narrowed here:
   the marker needs a written reason, and a file claiming to be RoboVerse-original may not carry
   somebody else's copyright line. It is still an unverified assertion by the author, which is
   why it may not be used to shut this test up about a file with a copy-tell in it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTICES = REPO / "THIRD_PARTY_NOTICES.md"

#: Directories that hold no first-party source (caches, build output, data, vendored binaries).
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "roboverse_data",
    "roboverse_py.egg-info",
    "wandb",
}

#: This file quotes header fragments and license markers; scanning it would flag itself.
SELF = Path(__file__).resolve()

#: How far into a file attribution may live. The header must be at the top (after a shebang),
#: not buried — so a copyright/SPDX line beyond this is not a header.
HEADER_LINES = 40

#: "copied from X", "adapted from X", ... where X looks like a project (capitalised) or a URL.
#: Requiring that shape keeps prose out ("values taken from env_states", "derived from config")
#: while catching the admissions that matter ("vendored from SimplerEnv").
COPY_TELL = re.compile(
    r"\b(?i:copied|adapted|ported|vendored|derived|taken)\s+from\s+(?:[A-Z][\w.'-]*|https?://\S+)"
    r"|(?i:\bbased on\b)[^\n]{0,60}?https?://",
)

#: An upstream copyright line in the header region is itself an admission.
COPYRIGHT = re.compile(r"\bCopyright\b")

ORIGINAL_MARKER = re.compile(r"#\s*RoboVerse-original\b[:.]?\s*(?P<reason>.*)")

#: What the license text must actually say for the SPDX tag to be true.
LICENSE_FINGERPRINTS = {
    "MIT": "Permission is hereby granted",
    "BSD-3-Clause": "Redistributions of source code",
    "BSD-2-Clause": "Redistributions of source code",
    "Apache-2.0": "Apache License",
}

#: Top-level entries of the repo — used to tell a repo path in a header/table apart from prose.
TOP_LEVEL = {p.name for p in REPO.iterdir()}

PATH_TOKEN = re.compile(r"[\w.]+(?:/[\w.+-]+)*/?")


def _is_repo_path(token: str) -> bool:
    return token.split("/", 1)[0] in TOP_LEVEL


def _python_files() -> list[Path]:
    out = []
    for path in REPO.rglob("*.py"):
        if SKIP_DIRS & set(path.parts) or path.resolve() == SELF:
            continue
        out.append(path)
    return sorted(out)


def _header(path: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:HEADER_LINES])


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


# --------------------------------------------------------------------------------------------
# THIRD_PARTY_NOTICES.md — parsed once, and treated as the index it claims to be.
# --------------------------------------------------------------------------------------------


def _expand_braces(token: str) -> list[str]:
    """`a/{b,c}/d` -> [`a/b/d`, `a/c/d`] (the table uses this shorthand)."""
    match = re.search(r"\{([^{}]*)\}", token)
    if not match:
        return [token]
    out = []
    for option in match.group(1).split(","):
        out += _expand_braces(token[: match.start()] + option.strip() + token[match.end() :])
    return out


def _table_rows() -> list[tuple[list[str], str]]:
    """[(paths in column 1, relationship), ...] from the Components table."""
    rows: list[tuple[list[str], str]] = []
    for line in NOTICES.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5 or cells[0] == "Path in this repo":
            continue
        paths = []
        for quoted in re.findall(r"`([^`]+)`", cells[0]):
            paths += [p for p in _expand_braces(quoted) if _is_repo_path(p)]
        if paths:
            rows.append((paths, cells[3]))
    return rows


def _unresolved_paths() -> list[str]:
    """Paths recorded in the 'Unresolved' section — code we could not attribute."""
    text = NOTICES.read_text(encoding="utf-8")
    section = text.split("## Unresolved", 1)[1].split("\n## ", 1)[0]
    return sorted({p for p in re.findall(r"`([^`]+)`", section) if _is_repo_path(p)})


TABLE_ROWS = _table_rows()
UNRESOLVED = _unresolved_paths()
INDEXED_PATHS = [p for paths, _ in TABLE_ROWS for p in paths]
NO_HEADER_NEEDED = [p for paths, rel in TABLE_ROWS if rel.startswith(("Reimplemented", "Integration")) for p in paths]


def _covered_by(rel_path: str, candidates: list[str]) -> bool:
    return any(rel_path == c or rel_path.startswith(c.rstrip("/") + "/") for c in candidates)


# --------------------------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------------------------


def _licenses(header: str) -> list[str]:
    spdx = re.search(r"SPDX-License-Identifier:\s*(.+)", header)
    if not spdx:
        return []
    return [tok for tok in re.split(r"\s+(?:AND|OR)\s+|\s+", spdx.group(1).strip()) if tok]


def _license_paths(header: str) -> list[str]:
    """Repo paths named by the `Full license:` clause (the line plus indented continuations)."""
    lines = header.splitlines()
    start = next((i for i, line in enumerate(lines) if "Full license:" in line), None)
    if start is None:
        return []
    clause = [lines[start].split("Full license:", 1)[1]]
    for line in lines[start + 1 :]:
        if not re.match(r"#\s{2,}\S", line):  # indented comment = continuation of the clause
            break
        clause.append(line.lstrip("# "))
    return [t for t in PATH_TOKEN.findall(" ".join(clause)) if _is_repo_path(t)]


HEADERED = [p for p in _python_files() if "SPDX-License-Identifier:" in _header(p)]


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


def test_attribution_index_exists() -> None:
    """NOTICE + THIRD_PARTY_NOTICES.md are what ship to downstream users."""
    for name in ("NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSE"):
        assert (REPO / name).is_file(), f"{name} is missing from the repo root"
    assert TABLE_ROWS, "THIRD_PARTY_NOTICES.md has no parseable component table"


@pytest.mark.parametrize("path", _python_files(), ids=_rel)
def test_copied_code_declares_its_upstream(path: Path) -> None:
    """A file that admits copying must carry the header — or be an audited exception."""
    rel = _rel(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    head = _header(path)

    if "SPDX-License-Identifier:" in head:
        return  # completeness is checked by test_header_is_complete
    if _covered_by(rel, UNRESOLVED) or _covered_by(rel, NO_HEADER_NEEDED):
        return  # recorded in THIRD_PARTY_NOTICES.md as unresolved / reimplemented / integration

    tell = COPY_TELL.search(text)
    foreign_copyright = [line for line in head.splitlines() if COPYRIGHT.search(line) and "RoboVerse" not in line]
    if not tell and not foreign_copyright:
        return

    marker = ORIGINAL_MARKER.search(head)
    admission = tell.group(0) if tell else foreign_copyright[0].strip()
    assert not (marker and foreign_copyright), (
        f"{rel} claims to be RoboVerse-original but carries somebody else's copyright line "
        f"({foreign_copyright[0].strip()!r}). One of the two is wrong."
    )
    if marker and not tell:
        assert len(marker.group("reason").strip()) >= 20, (
            f"{rel} uses the '# RoboVerse-original' opt-out without saying why. The marker is an "
            f"unverified assertion; it must at least state what the file is and why it contains "
            f"no upstream code."
        )
        return

    raise AssertionError(
        f"{rel} looks like third-party code — it says {admission!r} — but carries no attribution "
        f"header.\nAdd the header (see AGENTS.md):\n"
        f"    # Copyright (c) <year> <upstream holder>\n"
        f"    # SPDX-License-Identifier: <MIT|Apache-2.0|BSD-3-Clause>\n"
        f"    #\n"
        f"    # Adapted from <Project> (<url>).\n"
        f"    # Changes: <what we changed>, or 'none (vendored verbatim).'\n"
        f"    # Full license: <path to the license text in this repo>\n"
        f"...and a row in THIRD_PARTY_NOTICES.md. A '# copied from X' comment is an admission of "
        f"copying, not a license grant. If the upstream and its license cannot be named, the code "
        f"must not be merged (record it under 'Unresolved' in THIRD_PARTY_NOTICES.md instead)."
    )


@pytest.mark.parametrize("path", HEADERED, ids=_rel)
def test_header_is_complete(path: Path) -> None:
    """SPDX alone is not a header: name the upstream, the changes, and the license text."""
    rel = _rel(path)
    head = _header(path)

    assert COPYRIGHT.search(head), f"{rel}: header has an SPDX tag but no copyright line."
    assert _licenses(head), f"{rel}: empty SPDX-License-Identifier."
    assert re.search(r"\b(?:Adapted|Vendored|Ported|Copied|adapted|vendored|ported) from\b", head), (
        f"{rel}: header declares a license but does not name its upstream project.\n"
        f"Add 'Adapted from <Project> (<url>).'"
    )
    assert re.search(r"https?://", head), f"{rel}: header names no upstream URL."
    assert "Changes:" in head, (
        f"{rel}: no statement of changes. Apache-2.0 §4(b) requires one; add 'Changes: ...' "
        f"(or 'Changes: none (vendored verbatim).')."
    )
    assert "Full license:" in head, f"{rel}: header does not say where the license text lives."


@pytest.mark.parametrize("path", HEADERED, ids=_rel)
def test_header_license_text_resolves(path: Path) -> None:
    """`Full license:` must point at a real file that really contains that license."""
    rel = _rel(path)
    head = _header(path)
    licenses = _licenses(head)
    paths = _license_paths(head)

    assert paths, f"{rel}: 'Full license:' names no in-repo path."
    for candidate in paths:
        assert (REPO / candidate).exists(), f"{rel}: 'Full license: {candidate}' does not exist."

    texts = {c: (REPO / c).read_text(encoding="utf-8", errors="replace") for c in paths}
    for license_id in licenses:
        fingerprint = LICENSE_FINGERPRINTS.get(license_id)
        assert fingerprint, f"{rel}: unknown SPDX identifier {license_id!r} — add it to this test."
        assert any(fingerprint in text for text in texts.values()), (
            f"{rel} declares {license_id} but none of {paths} contains {fingerprint!r}. "
            f"The header points at the wrong license text."
        )


@pytest.mark.parametrize("indexed", sorted(set(INDEXED_PATHS)))
def test_indexed_path_exists(indexed: str) -> None:
    """Every path THIRD_PARTY_NOTICES.md claims to cover must be there (no stale rows)."""
    if "*" in indexed:  # a row may name a family, e.g. `.../envs/tracking_*.py`
        assert list(REPO.glob(indexed)), f"THIRD_PARTY_NOTICES.md names {indexed}, which matches nothing"
        return
    assert (REPO / indexed).exists(), f"THIRD_PARTY_NOTICES.md names {indexed}, which does not exist"


@pytest.mark.parametrize("unresolved", UNRESOLVED)
def test_unresolved_path_exists(unresolved: str) -> None:
    """The 'Unresolved' list is a live inventory, not a memo — its paths must exist."""
    assert (REPO / unresolved).exists(), (
        f"THIRD_PARTY_NOTICES.md lists {unresolved} as unresolved, but it does not exist. "
        f"If it was removed, drop it from the list."
    )


@pytest.mark.parametrize("path", HEADERED, ids=_rel)
def test_headered_file_is_indexed(path: Path) -> None:
    """The other direction: third-party code in the tree must appear in the index."""
    rel = _rel(path)
    assert _covered_by(rel, INDEXED_PATHS), (
        f"{rel} carries a third-party attribution header but no row in THIRD_PARTY_NOTICES.md "
        f"covers it. The index must stay complete — add a row."
    )
