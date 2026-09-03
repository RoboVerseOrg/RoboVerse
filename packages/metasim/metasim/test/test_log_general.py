"""``metasim.utils.log``: one-shot warnings are process-wide; the level switch is opt-in."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from loguru import logger

from metasim.utils import log as mlog


@pytest.mark.general
def test_warn_once_emits_a_key_once_per_process():
    mlog.reset_warn_once()
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert mlog.warn_once(("t", "a"), "first") is True
        assert mlog.warn_once(("t", "a"), "first again") is False
        assert mlog.warn_once(("t", "b"), "other") is True
    finally:
        logger.remove(sink)
    assert sum("first" in m for m in messages) == 1 and any("other" in m for m in messages)


@pytest.mark.general
def test_import_does_not_touch_logging_unless_level_is_set():
    code = "import metasim; from metasim.utils import log as l; print(l._CONFIGURED)"
    env = {k: v for k, v in os.environ.items() if k != "METASIM_LOG_LEVEL"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    assert out.stdout.strip() == "False", out.stderr[-500:]
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**env, "METASIM_LOG_LEVEL": "WARNING"},
        check=False,
    )
    assert out.stdout.strip() == "True", out.stderr[-500:]
