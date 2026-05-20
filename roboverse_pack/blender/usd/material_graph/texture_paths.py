"""Texture asset path normalization and UDIM tile discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class UdimInfo:
    normalized_path: str
    first_tile: int | None


@dataclass(frozen=True)
class TexturePathResult:
    raw: str
    normalized: str
    is_udim: bool
    candidates: tuple[Path, ...]
    exists: bool
    warnings: tuple[str, ...] = ()


_UDIM_TOKEN = "<UDIM>"
_PRINTF_UDIM_TOKEN = "%(UDIM)d"
_UDIM_TILE_RE = re.compile(r"(?P<prefix>.*[./])(?P<tile>1[0-9]{3})(?P<suffix>\.[A-Za-z0-9]+)$")


def detect_udim_pattern(path: str) -> UdimInfo | None:
    if _UDIM_TOKEN in path:
        return UdimInfo(normalized_path=path, first_tile=None)

    if _PRINTF_UDIM_TOKEN in path:
        return UdimInfo(normalized_path=path.replace(_PRINTF_UDIM_TOKEN, _UDIM_TOKEN), first_tile=None)

    match = _UDIM_TILE_RE.match(path)
    if match is None:
        return None

    tile = int(match.group("tile"))
    if not 1001 <= tile <= 1100:
        return None

    return UdimInfo(
        normalized_path=f"{match.group('prefix')}{_UDIM_TOKEN}{match.group('suffix')}",
        first_tile=tile,
    )


def find_texture_file_candidates(raw: str, base_dir: Path) -> list[Path]:
    clean = _strip_asset_path(raw)
    if not clean:
        return []

    udim = detect_udim_pattern(clean)
    if udim is not None:
        return _find_udim_candidates(udim.normalized_path, base_dir)

    candidate = Path(clean)
    if not candidate.is_absolute():
        candidate = base_dir / candidate

    return [candidate] if candidate.is_file() else []


def normalize_texture_asset_path(raw: str, base_dir: Path) -> TexturePathResult:
    clean = _strip_asset_path(raw)
    udim = detect_udim_pattern(clean)
    normalized = udim.normalized_path if udim is not None else clean
    candidates = tuple(find_texture_file_candidates(clean, base_dir))
    exists = bool(candidates)

    warnings: tuple[str, ...] = ()
    if not exists:
        if udim is not None:
            warnings = (f"UDIM texture tile/file not found: {normalized}",)
        else:
            warnings = (f"Texture file not found: {normalized}",)

    return TexturePathResult(
        raw=clean,
        normalized=normalized,
        is_udim=udim is not None,
        candidates=candidates,
        exists=exists,
        warnings=warnings,
    )


def _strip_asset_path(raw: str) -> str:
    return str(raw).strip().strip("@").strip()


def _find_udim_candidates(normalized_path: str, base_dir: Path) -> list[Path]:
    path = Path(normalized_path)
    search_path = path if path.is_absolute() else base_dir / path
    parent = search_path.parent
    if not parent.is_dir():
        return []

    name_re = _udim_name_regex(search_path.name)
    candidates: list[Path] = []
    for child in sorted(parent.iterdir(), key=lambda candidate: candidate.name):
        if not child.is_file():
            continue
        match = name_re.fullmatch(child.name)
        if match is None:
            continue
        tile = int(match.group("tile"))
        if 1001 <= tile <= 1100:
            candidates.append(child)
    return candidates


def _udim_name_regex(name: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    tile_pattern = r"(?P<tile>1[0-9]{3})"
    return re.compile(escaped.replace(re.escape(_UDIM_TOKEN), tile_pattern))
