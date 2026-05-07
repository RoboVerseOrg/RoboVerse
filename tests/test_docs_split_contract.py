from __future__ import annotations

import os
import subprocess
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


def test_split_site_build_script_assembles_landing_roboverse_and_metasim(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sphinx = fake_bin / "sphinx-build"
    fake_sphinx.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
src="${3:?missing source}"
dst="${4:?missing destination}"
mkdir -p "$dst"
printf '<html>%s</html>\\n' "$src" > "$dst/index.html"
""",
        encoding="utf-8",
    )
    fake_sphinx.chmod(0o755)

    metasim_dir = tmp_path / "MetaSim"
    (metasim_dir / "docs/source").mkdir(parents=True)
    (metasim_dir / "docs/source/index.md").write_text("# MetaSim\n", encoding="utf-8")
    (metasim_dir / "docs/source/images").mkdir()
    (metasim_dir / "docs/source/images/tea.jpg").write_bytes(b"fake image")
    output_dir = tmp_path / "public"

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
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
