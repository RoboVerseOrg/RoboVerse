"""Asset-cache root is absolute and path-traversal-safe (M5).

Before the fix:

- ``LOCAL_DIR`` was the literal string ``"roboverse_data"``; every
  downstream use (``os.path.exists``, ``os.makedirs``, ``hf_hub_download
  (local_dir=...)``) resolved it against the *current* process CWD.
  Running the same script from two different directories produced two
  separate caches.
- ``os.path.relpath(filepath, LOCAL_DIR)`` against a filepath that
  contained ``..`` segments (e.g. from a malformed MJCF that referenced
  ``../../../etc/passwd``) returned a path starting with ``..``; that
  string was then sent to the HuggingFace API. Bad hygiene at minimum,
  and a latent leak of intent shape outside the dataset.

After the fix:

- ``LOCAL_DIR`` is resolved to an absolute path at module import, with
  ``ROBOVERSE_DATA_DIR`` env var override for shared installations.
- ``check_and_download_single`` refuses to fetch any filepath whose
  relpath against ``LOCAL_DIR`` starts with ``..``. Required assets
  raise; optional textures/mtls warn-and-skip.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.mark.general
def test_local_dir_is_absolute():
    """The resolved cache root must not be CWD-relative at runtime."""
    from metasim.utils import hf_util

    assert os.path.isabs(hf_util.LOCAL_DIR), (
        f"LOCAL_DIR={hf_util.LOCAL_DIR!r} should have been abspath-resolved"
    )


@pytest.mark.general
def test_env_var_override(monkeypatch, tmp_path):
    """``ROBOVERSE_DATA_DIR`` env var pins the cache root.

    Importing inside the test would need a module reset; using
    ``_resolve_local_dir`` directly is sufficient.
    """
    from metasim.utils.hf_util import _resolve_local_dir

    target = tmp_path / "shared-cache"
    monkeypatch.setenv("ROBOVERSE_DATA_DIR", str(target))
    resolved = _resolve_local_dir()
    assert resolved == str(target.resolve())


@pytest.mark.general
def test_env_var_expanduser(monkeypatch):
    """``~`` in the env var is expanded."""
    from metasim.utils.hf_util import _resolve_local_dir

    monkeypatch.setenv("ROBOVERSE_DATA_DIR", "~/.cache/roboverse-pinned")
    resolved = _resolve_local_dir()
    assert resolved == os.path.abspath(os.path.expanduser("~/.cache/roboverse-pinned"))


@pytest.mark.general
def test_no_env_var_falls_back_to_abs_cwd(monkeypatch, tmp_path):
    """Without the env var, the cache root is ``./roboverse_data`` made absolute."""
    monkeypatch.delenv("ROBOVERSE_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    from metasim.utils.hf_util import _resolve_local_dir

    resolved = _resolve_local_dir()
    assert resolved == str((tmp_path / "roboverse_data").resolve())


@pytest.mark.general
def test_traversal_rejected_for_required_file(monkeypatch):
    """Required asset whose path escapes LOCAL_DIR via ``..`` must raise."""
    from metasim.utils import hf_util

    # File that would relpath() to "../escape.urdf" — and importantly does
    # NOT exist locally so we go down the relpath / HF branch. Use a
    # known-not-local sibling of LOCAL_DIR.
    bad_path = os.path.join(os.path.dirname(hf_util.LOCAL_DIR), "escape.urdf")
    assert not os.path.exists(bad_path), "test pre-condition: bad path must not exist"

    # No network call: the traversal guard fires before HfApi is touched.
    # Patch HfApi just in case to make the test fully hermetic.
    monkeypatch.setattr(hf_util.hf_api, "file_exists", lambda *a, **kw: pytest.fail("HfApi reached"))

    with pytest.raises(ValueError, match="outside LOCAL_DIR"):
        hf_util.check_and_download_single(bad_path)


@pytest.mark.general
def test_traversal_warn_and_skip_for_optional_file(monkeypatch, caplog):
    """Optional textures/mtls whose path escapes must warn-and-skip, not raise."""
    from metasim.utils import hf_util

    bad_path = os.path.join(os.path.dirname(hf_util.LOCAL_DIR), "escape.png")
    assert not os.path.exists(bad_path)

    monkeypatch.setattr(hf_util.hf_api, "file_exists", lambda *a, **kw: pytest.fail("HfApi reached"))

    # Should not raise.
    hf_util.check_and_download_single(bad_path)


@pytest.mark.general
def test_normal_path_inside_local_dir_passes_guard(monkeypatch, tmp_path):
    """A path inside LOCAL_DIR with no ``..`` segments hits the HF branch as before."""
    from metasim.utils import hf_util

    # An in-cache path that doesn't exist on disk → the function reaches the
    # HF check. Stub HfApi to say "not found" — for a required asset that
    # results in the existing "neither local nor on HF" exception, which
    # is exactly what we want to assert: the traversal guard did NOT short-
    # circuit, so the function got past it.
    inside_path = os.path.join(hf_util.LOCAL_DIR, "fake_subdir", "model.urdf")
    monkeypatch.setattr(hf_util.hf_api, "file_exists", lambda *a, **kw: False)

    with pytest.raises(Exception, match="neither exists in the local directory"):
        hf_util.check_and_download_single(inside_path)


@pytest.mark.general
def test_local_existing_file_short_circuits(tmp_path, monkeypatch):
    """If the file exists locally, the function returns without touching HF.

    Independent of the traversal guard but verifies it doesn't break the
    local-cache-hit happy path.
    """
    from metasim.utils import hf_util

    real_file = tmp_path / "asset.urdf"
    real_file.write_text("<robot/>")
    monkeypatch.setattr(hf_util.hf_api, "file_exists", lambda *a, **kw: pytest.fail("HfApi reached"))

    hf_util.check_and_download_single(str(real_file))
