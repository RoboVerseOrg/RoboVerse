"""Defused XML parsing for asset descriptors (MJCF / URDF / SDF).

MJCF and URDF assets are auto-downloaded from HuggingFace without
integrity verification (see ``metasim/utils/hf_util.py``). The stdlib
``xml.etree.ElementTree`` parser is vulnerable to several classic
attacks on untrusted XML:

- **Billion-laughs / quadratic-blowup**: nested entity expansion can
  inflate a tiny file to gigabytes in memory, DoS'ing the process.
- **External-entity** loads: ``<!ENTITY x SYSTEM "file:///etc/passwd">``
  reads arbitrary files; ``SYSTEM "http://..."`` can exfiltrate.
- **DTD remote loads**: SSRF via DTD ``SYSTEM`` references.

``defusedxml`` patches the C-accelerated ``expat`` parser to reject
entity expansion and external loads. This module exposes a single
``ET`` symbol that prefers ``defusedxml.ElementTree`` when available
and falls back to the stdlib with a one-time WARNING if not — so
existing installations don't break, but the warning makes the
unhardened state visible. The hardened path activates as soon as
``defusedxml`` is installed.

Usage:
    from metasim.utils.xml_safe import ET
    tree = ET.parse(path)
    root = ET.fromstring(xml_str)

Same surface as the stdlib's ``xml.etree.ElementTree``. The
``Element`` type comes from the stdlib regardless; only the *parser*
is swapped.
"""

from __future__ import annotations

from loguru import logger as log

_warned = False

try:
    import defusedxml.ElementTree as _ET_module  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — exercised when defusedxml absent
    import xml.etree.ElementTree as _ET_module

    if not _warned:
        log.warning(
            "metasim.utils.xml_safe: ``defusedxml`` not installed — falling "
            "back to stdlib ``xml.etree.ElementTree``, which is vulnerable "
            "to billion-laughs and external-entity attacks on untrusted "
            "MJCF/URDF assets. Install with ``pip install defusedxml`` to "
            "activate the hardened parser."
        )
        _warned = True


# Re-export under the canonical ``ET`` alias every consumer uses.
ET = _ET_module


__all__ = ["ET"]
