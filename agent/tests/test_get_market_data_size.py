"""Regression test for P07 — get_market_data must bound its per-symbol output.

Pre-fix: every row of every symbol was emitted, so "1 symbol, 1 year, daily"
(~251 rows) already breached the MCP token cap and had to spool to a file.
Post-fix: a `max_rows` cap (default 250) returns a head+tail window plus
truncation metadata for oversized symbols; small queries are unchanged
(plain list), and `max_rows=0` restores the unbounded legacy behavior.
"""

from __future__ import annotations

import json

import pandas as pd

import mcp_server

_gmd = getattr(mcp_server.get_market_data, "fn", None) or getattr(
    mcp_server.get_market_data, "__wrapped__", mcp_server.get_market_data
)


def _loader_with_rows(n: int):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    df.index.name = "trade_date"

    class _L:
        def fetch(self, codes, start, end, interval="1D"):
            return {"X.US": df}

    return _L


def _call(monkeypatch, n, **kw):
    monkeypatch.setattr(mcp_server, "_get_loader", lambda src: _loader_with_rows(n))
    return json.loads(_gmd(codes=["X.US"], start_date="2025-01-01", end_date="2026-01-01", source="yfinance", **kw))


def test_oversized_symbol_is_capped_with_metadata(monkeypatch):
    out = _call(monkeypatch, 300)["X.US"]
    assert out["truncated"] is True
    assert out["rows"] == 300
    assert out["returned"] == 250
    assert len(out["data"]) == 251  # 125 head + 1 gap marker + 125 tail
    assert any("_gap" in row for row in out["data"])


def test_small_query_unchanged_plain_list(monkeypatch):
    """No-regression: under the cap the shape is the original plain list."""
    out = _call(monkeypatch, 50)["X.US"]
    assert isinstance(out, list) and len(out) == 50


def test_max_rows_zero_disables_cap(monkeypatch):
    out = _call(monkeypatch, 300, max_rows=0)["X.US"]
    assert isinstance(out, list) and len(out) == 300


def test_default_caps_canonical_one_year_daily(monkeypatch):
    """The canonical ~251-row 1y-daily request must no longer be unbounded."""
    out = _call(monkeypatch, 251)["X.US"]
    assert isinstance(out, dict) and out["truncated"] is True
