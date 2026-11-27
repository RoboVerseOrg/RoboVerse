"""The repository's text is English: no CJK characters in tracked source, docs or assets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# CJK unified ideographs (U+4E00-U+9FFF), CJK punctuation (U+3000-U+303F) and the full-width ASCII
# variants (U+FF00-U+FFEF; full-width comma, parentheses, colon), the characters that slip into comments
# and docs written with a CJK input method. Built from code points so this file never contains one.
CJK = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x4E00, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF))) + "]"
)
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".npz",
    ".npy",
    ".pkl",
    ".pt",
    ".pth",
    ".ttf",
    ".woff",
    ".woff2",
    ".bin",
    ".mp4",
    ".usd",
    ".usdz",
    ".glb",
    ".obj",
    ".stl",
    ".dae",
    ".zip",
}


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [REPO / p for p in out.decode().split("\0") if p and Path(p).suffix.lower() not in SKIP_SUFFIXES]


def test_tracked_files_contain_no_cjk_characters():
    offenders: list[str] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not text we author
        for lineno, line in enumerate(text.splitlines(), 1):
            if CJK.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()[:80]}")
    assert not offenders, "CJK text in the repository (write comments, docs and asset ids in English):\n" + "\n".join(
        offenders[:40]
    )
