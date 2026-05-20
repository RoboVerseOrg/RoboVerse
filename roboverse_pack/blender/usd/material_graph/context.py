"""Context carried through material graph adapter conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class MaterialContext:
    source_path: Path
    texture_base_dir: Path
    material_path: str
    bound_prim_paths: tuple[str, ...] = ()
    uv_primvars_by_prim: Mapping[str, tuple[str, ...]] | None = None
    enable_mdl_bake: bool = False
