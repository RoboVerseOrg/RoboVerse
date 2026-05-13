from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_roboverse_docs_root_exposes_only_roboverse_sections():
    index = (REPO_ROOT / "docs/source/index.md").read_text(encoding="utf-8")
    conf = (REPO_ROOT / "docs/source/conf.py").read_text(encoding="utf-8")

    assert "MetaSim <metasim/index>" not in index
    assert "API <API/index>" not in index
    assert '"/metasim/"' in index
    assert '"/roboverse/"' in index
    assert '"metasim"' not in conf
    assert '"API"' not in conf


def test_docs_deployment_requirements_avoid_runtime_and_live_preview_packages():
    requirements = (REPO_ROOT / "docs/requirements.txt").read_text(encoding="utf-8")

    assert "sphinx==8.1.3" in requirements
    for package in [
        "metasim",
        "torch",
        "torchvision",
        "gymnasium",
        "sphinx-autobuild",
        "sphinxcontrib.spelling",
    ]:
        assert package not in requirements


def test_split_site_build_script_assembles_landing_roboverse_and_metasim(tmp_path):
    fake_pythonpath = tmp_path / "pythonpath"
    fake_sphinx = fake_pythonpath / "sphinx"
    fake_sphinx.mkdir(parents=True)
    (fake_sphinx / "__main__.py").write_text(
        """from __future__ import annotations

import sys
from pathlib import Path

src = sys.argv[3]
dst = Path(sys.argv[4])
dst.mkdir(parents=True, exist_ok=True)
(dst / "index.html").write_text(f"<html>{src}</html>\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )

    metasim_dir = tmp_path / "MetaSim"
    (metasim_dir / "docs/source").mkdir(parents=True)
    (metasim_dir / "docs/source/index.md").write_text("# MetaSim\n", encoding="utf-8")
    (metasim_dir / "docs/source/images").mkdir()
    (metasim_dir / "docs/source/images/tea.jpg").write_bytes(b"fake image")
    output_dir = tmp_path / "public"

    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(fake_pythonpath)
    env["METASIM_DIR"] = str(metasim_dir)

    subprocess.run(
        [str(REPO_ROOT / "scripts/docs/build_roboverse_wiki.sh"), str(output_dir)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    assert (output_dir / "index.html").is_file()
    landing = (output_dir / "index.html").read_text(encoding="utf-8")
    assert 'href="/metasim/get_started/installation.html"' in landing
    assert "/metasim/metasim/" not in landing
    assert (output_dir / "roboverse/index.html").read_text(encoding="utf-8").endswith("docs/source</html>\n")
    assert (output_dir / "metasim/index.html").read_text(encoding="utf-8").endswith("MetaSim/docs/source</html>\n")
    assert (output_dir / "metasim/_images/tea.jpg").read_bytes() == b"fake image"
    assert (output_dir / ".nojekyll").is_file()
    assert (output_dir / "CNAME").read_text(encoding="utf-8") == "roboverse.wiki\n"
