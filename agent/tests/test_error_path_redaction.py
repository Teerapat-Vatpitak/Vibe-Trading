"""Regression tests for P10 — user-facing errors must not leak internal
absolute filesystem paths (CWE-209 / CWE-497).

Pre-fix: an unknown preset / workspace-escape / missing run dir embedded the
full install path (OS username, .venvs/site-packages topology). Post-fix the
message keeps the actionable bits (name, Available list, the boundary) but
not the absolute path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.swarm.presets import load_preset
from src.swarm.store import SwarmStore
from src.swarm.models import SwarmRun
from src.tools.path_utils import safe_path

_LEAKS = (str(Path.home()), "site-packages", ".venvs", str(Path.cwd()))


def _assert_no_abs(msg: str):
    for marker in _LEAKS:
        assert marker not in msg, f"leaked {marker!r} in: {msg}"


def test_unknown_preset_does_not_leak_path():
    with pytest.raises(FileNotFoundError) as ei:
        load_preset("nope_xyz_not_a_preset")
    msg = str(ei.value)
    _assert_no_abs(msg)
    assert "nope_xyz_not_a_preset" in msg and "Available" in msg  # still actionable


def test_workspace_escape_does_not_leak_root(tmp_path):
    with pytest.raises(ValueError) as ei:
        safe_path("../../../../etc/passwd", workdir=tmp_path)
    msg = str(ei.value)
    _assert_no_abs(msg)
    assert "escapes the workspace root" in msg


def test_missing_run_dir_does_not_leak_abs(tmp_path):
    store = SwarmStore(base_dir=tmp_path / "runs")
    with pytest.raises(FileNotFoundError) as ei:
        store.update_run(SwarmRun(id="swarm-rid-001", preset_name="demo", created_at="2026-01-01T00:00:00Z"))
    msg = str(ei.value)
    _assert_no_abs(msg)
    assert "swarm-rid-001" in msg  # logical id retained, absolute path dropped
