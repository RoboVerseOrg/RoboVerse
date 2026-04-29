from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_docs_conf():
    conf_path = Path(__file__).resolve().parents[1] / "docs" / "source" / "conf.py"
    spec = importlib.util.spec_from_file_location("roboverse_docs_conf", conf_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_navbar() -> str:
    return """
<ul class="bd-navbar-elements navbar-nav">
<li class="nav-item "><a class="nav-link nav-internal" href="../metasim/index.html">MetaSim User Guide</a></li>
<li class="nav-item "><a class="nav-link nav-internal" href="../dataset_benchmark/index.html">Dataset and Benchmark</a></li>
<li class="nav-item current active"><a class="nav-link nav-internal" href="#">RoboVerse Learn</a></li>
<li class="nav-item "><a class="nav-link nav-internal" href="../API/index.html">API</a></li>
<li class="nav-item "><a class="nav-link nav-internal" href="../FAQ/index.html">Frequently Asked Questions</a></li>
</ul>
"""


def _sample_faq_navbar() -> str:
    return """
<ul class="bd-navbar-elements navbar-nav">
<li class="nav-item "><a class="nav-link nav-internal" href="../metasim/index.html">MetaSim User Guide</a></li>
<li class="nav-item "><a class="nav-link nav-internal" href="../dataset_benchmark/index.html">Dataset and Benchmark</a></li>
<li class="nav-item "><a class="nav-link nav-internal" href="../roboverse_learn/index.html">RoboVerse Learn</a></li>
<li class="nav-item "><a class="nav-link nav-internal" href="../API/index.html">API</a></li>
<li class="nav-item current active"><a class="nav-link nav-internal" href="#">Frequently Asked Questions</a></li>
</ul>
"""


def test_docs_navbar_keeps_only_active_section_full_length():
    conf = _load_docs_conf()

    html = conf._rewrite_navbar_labels(_sample_navbar(), "roboverse_learn/index")

    assert ">MetaSim<" in html
    assert ">Dataset<" in html
    assert ">RoboVerse Learn<" in html
    assert ">API<" in html
    assert ">FAQ<" in html
    assert "MetaSim User Guide" not in html
    assert "Dataset and Benchmark" not in html
    assert "Frequently Asked Questions" not in html


def test_docs_navbar_expands_faq_only_when_faq_is_active():
    conf = _load_docs_conf()

    html = conf._rewrite_navbar_labels(_sample_faq_navbar(), "FAQ/index")

    assert ">Learn<" in html
    assert ">Frequently Asked Questions<" in html
    assert "RoboVerse Learn" not in html
