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
    """Removed symbol/method/field, lost defaults and new required parameters are breaking; new
    optional parameters and new symbols are additions."""
    from metasim.utils.api_surface import _params

    def f_old(a, b=1): ...

    def f_new_required(a, b): ...

    def f_new_optional(a, b=1, *, c=None): ...

    def m_old(self): ...

    def m_new(self, n=1): ...

    old = {
        "m": {
            "f": {"kind": "function", "signature": "(a, b=1)", "params": _params(f_old)},
            "g": {"kind": "function", "signature": "(a, b=1)", "params": _params(f_old)},
            "C": {
                "kind": "class",
                "bases": [],
                "methods": {
                    "step": {"signature": "(self)", "params": _params(m_old)},
                    "close": {"signature": "(self)", "params": _params(m_old)},
                },
                "fields": ["x"],
            },
        }
    }
    new = {
        "m": {
            "f": {"kind": "function", "signature": "(a, b)", "params": _params(f_new_required)},
            "g": {"kind": "function", "signature": "(a, b=1, *, c=None)", "params": _params(f_new_optional)},
            "C": {
                "kind": "class",
                "bases": [],
                "methods": {
                    "step": {"signature": "(self, n=1)", "params": _params(m_new)},
                    "reset": {"signature": "(self)", "params": _params(m_old)},
                },
                "fields": [],
            },
            "h": {"kind": "function", "signature": "()", "params": []},
        }
    }
    breaking, additions = diff_api(old, new)
    assert [b.split(":")[0] for b in breaking] == [
        "signature changed (parameter lost its default",
        "method removed",
        "field removed",
    ]
    assert sorted(additions) == [
        "added: m.h",
        "method added: m.C.reset",
        "signature extended: m.C.step(self) -> (self, n=1)",
        "signature extended: m.g(a, b=1) -> (a, b=1, *, c=None)",
    ]


def test_compare_params_rules():
    """The exact rules callers rely on."""
    from metasim.utils.api_surface import _params, compare_params

    def base(a, b=1): ...

    def renamed(a, c=1): ...

    def reordered(b, a=1): ...

    def kw_only(a, *, b=1): ...

    def sink(a, b=1, **kw): ...

    assert compare_params(_params(base), _params(base)) is None
    assert compare_params(_params(base), _params(sink)) is None
    assert "removed" in compare_params(_params(base), _params(renamed))
    assert "kind changed" in compare_params(_params(base), _params(kw_only))
    assert compare_params(_params(base), _params(reordered)) is not None
