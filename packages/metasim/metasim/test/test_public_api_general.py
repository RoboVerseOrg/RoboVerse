"""The public API only changes on purpose.

``metasim/test/api_snapshot.json`` records every public function and class (methods, dataclass
fields, signatures) of the modules in ``metasim.utils.api_surface.PUBLIC_MODULES``. A removed
symbol, method or field, or a changed signature, fails here until the snapshot is regenerated with
``python -m metasim api-snapshot --update`` and the change is written into the CHANGELOG.
Additions are printed, not rejected.
"""

from __future__ import annotations

import warnings

from metasim.utils.api_surface import SNAPSHOT_PATH, collect_api, diff_api, load_snapshot


def test_public_api_matches_snapshot():
    """The live surface has everything the snapshot has, with the same signatures."""
    assert SNAPSHOT_PATH.exists(), "run `python -m metasim api-snapshot --update` to create the snapshot"
    breaking, additions = diff_api(load_snapshot(), collect_api())
    if additions:
        warnings.warn("public API additions not yet in the snapshot:\n  " + "\n  ".join(additions), stacklevel=1)
    assert not breaking, (
        "public API changed:\n  "
        + "\n  ".join(breaking)
        + "\n\nIf intended: `python -m metasim api-snapshot --update`, then record it in CHANGELOG.md."
    )


def test_diff_api_detects_each_kind_of_break():
    """Removed symbol/method/field and changed signatures are breaking; new ones are additions."""
    old = {
        "m": {
            "f": {"kind": "function", "signature": "(a, b=1)"},
            "C": {"kind": "class", "bases": [], "methods": {"step": "(self)", "close": "(self)"}, "fields": ["x"]},
        }
    }
    new = {
        "m": {
            "f": {"kind": "function", "signature": "(a)"},
            "C": {"kind": "class", "bases": [], "methods": {"step": "(self, n=1)", "reset": "(self)"}, "fields": []},
            "g": {"kind": "function", "signature": "()"},
        }
    }
    breaking, additions = diff_api(old, new)
    assert [b.split(":")[0] for b in breaking] == [
        "signature changed",
        "signature changed",
        "method removed",
        "field removed",
    ]
    assert sorted(additions) == ["added: m.g", "method added: m.C.reset"]
