"""XML parsing of MJCF/URDF assets goes through ``defusedxml`` (M4).

MJCF/URDF descriptors are auto-downloaded from HuggingFace without
integrity verification (see ``metasim/utils/hf_util.py``). The stdlib
``xml.etree.ElementTree`` parser is vulnerable to:

- **billion-laughs / quadratic entity expansion** → DoS
- **external-entity loads** (``SYSTEM "file://..."`` or
  ``SYSTEM "http://..."``) → arbitrary file read or SSRF
- **DTD remote loads** → SSRF via the doctype declaration

The fix centralises the parser choice in ``metasim.utils.xml_safe``
which prefers ``defusedxml.ElementTree`` and falls back to the stdlib
with a one-time WARNING. All asset-parsing modules import ``ET`` from
that helper instead of the stdlib directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.general
def test_xml_safe_exposes_ET():
    from metasim.utils.xml_safe import ET

    assert hasattr(ET, "parse")
    assert hasattr(ET, "fromstring")


@pytest.mark.general
def test_xml_safe_module_is_defusedxml_when_available():
    """If ``defusedxml`` is installed, ``ET`` is the defused version.

    Skipped when defusedxml isn't installed in this environment — the
    fallback path is still legitimate (with a logged WARNING).
    """
    defusedxml = pytest.importorskip("defusedxml.ElementTree")
    from metasim.utils.xml_safe import ET

    assert ET is defusedxml, "xml_safe should prefer defusedxml when it's importable"


@pytest.mark.general
def test_xml_safe_rejects_billion_laughs_when_defused():
    """defusedxml's parser must refuse exponential entity expansion."""
    pytest.importorskip("defusedxml.ElementTree")
    from defusedxml.common import EntitiesForbidden

    from metasim.utils.xml_safe import ET

    bomb = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE lolz ['
        '  <!ENTITY lol "lol">'
        '  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        ']>'
        '<lolz>&lol2;</lolz>'
    )
    with pytest.raises(EntitiesForbidden):
        ET.fromstring(bomb)


@pytest.mark.general
def test_xml_safe_rejects_external_entity_when_defused(tmp_path):
    """External-entity loads must be blocked."""
    pytest.importorskip("defusedxml.ElementTree")
    from defusedxml.common import EntitiesForbidden

    secret = tmp_path / "secret.txt"
    secret.write_text("SHOULD-NOT-LEAK")
    payload = (
        f'<?xml version="1.0"?>'
        f'<!DOCTYPE foo ['
        f'  <!ENTITY xxe SYSTEM "file://{secret}">'
        f']>'
        f'<foo>&xxe;</foo>'
    )

    from metasim.utils.xml_safe import ET

    with pytest.raises(EntitiesForbidden):
        ET.fromstring(payload)


@pytest.mark.general
def test_xml_safe_still_parses_benign_xml():
    """Sanity: normal MJCF-like XML still parses."""
    from metasim.utils.xml_safe import ET

    xml = '<mujoco><worldbody><body name="root"/></worldbody></mujoco>'
    root = ET.fromstring(xml)
    assert root.tag == "mujoco"
    assert root.find("worldbody/body").get("name") == "root"


@pytest.mark.general
def test_all_asset_parsers_use_xml_safe_helper():
    """Lock the regression: every asset-parsing module imports ``ET``
    from ``metasim.utils.xml_safe``, never from the stdlib directly."""
    import metasim

    repo_root = Path(metasim.__file__).resolve().parent

    # Files that touch untrusted MJCF/URDF and must use the defused
    # parser. Listed by repo-relative path so an additional consumer is
    # noticed in code review.
    must_be_defused = [
        "utils/parse_util.py",
        "utils/isaacsim_asset_util.py",
        "sim/blender/blender.py",
        "sim/newton/newton.py",
        "sim/genesis/genesis.py",
        "sim/sapien/sapien3.py",
        "utils/viser/viser_util.py",
        "utils/rerun/rerun_util.py",
    ]

    offenders = []
    for rel in must_be_defused:
        text = (repo_root / rel).read_text()
        if "from metasim.utils.xml_safe import ET" not in text:
            offenders.append(rel)
        if "import xml.etree.ElementTree as ET" in text:
            offenders.append(rel + " (still imports stdlib ET directly)")
    assert not offenders, f"asset parsers must import ET from xml_safe: {offenders}"
