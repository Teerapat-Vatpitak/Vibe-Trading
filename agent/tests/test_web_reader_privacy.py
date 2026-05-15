"""Regression tests for P11 — read_url third-party (Jina) hardening.

Network is mocked; no live r.jina.ai calls. Asserts: HTTP/exception errors
no longer leak the vendor name or response body (R2); a cached snapshot is
surfaced via `cached: true`; `no_cache=True` sends the x-no-cache header
while the default path is byte-identical (no extra header).
"""

from __future__ import annotations

import json

import pytest

import src.tools.web_reader_tool as wr
from src.tools.web_reader_tool import read_url

URL = "https://example.com/page"


class _Resp:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def captured(monkeypatch):
    box = {}

    def fake_get(url, headers=None, timeout=None):
        box["url"] = url
        box["headers"] = headers or {}
        r = box["resp"]
        if isinstance(r, BaseException):
            raise r
        return r

    monkeypatch.setattr(wr.requests, "get", fake_get)
    return box


def test_http_error_does_not_leak_vendor_or_body(captured):
    captured["resp"] = _Resp(451, "ParamValidationError at r.jina.ai internal detail")
    out = json.loads(read_url(URL))
    assert out["status"] == "error"
    assert out["error"] == "remote reader returned HTTP 451"
    assert "jina" not in out["error"].lower() and "ParamValidation" not in out["error"]


def test_exception_error_is_generic(captured):
    captured["resp"] = RuntimeError("boom: connect to r.jina.ai failed (10.0.0.1)")
    out = json.loads(read_url(URL))
    assert out["status"] == "error"
    assert out["error"] == "remote reader request failed"
    assert "jina" not in out["error"].lower() and "boom" not in out["error"]


def test_cached_snapshot_is_flagged(captured):
    captured["resp"] = _Resp(200, "Title: X\n\nWarning: This is a cached snapshot\n\nbody")
    out = json.loads(read_url(URL))
    assert out["status"] == "ok"
    assert out.get("cached") is True


def test_fresh_response_has_no_cached_key(captured):
    captured["resp"] = _Resp(200, "Title: X\n\nlive body content")
    out = json.loads(read_url(URL))
    assert out["status"] == "ok"
    assert "cached" not in out  # additive: absent on the normal path


def test_no_cache_header_opt_in_only(captured):
    captured["resp"] = _Resp(200, "Title: X\n\nbody")
    read_url(URL)  # default
    assert "x-no-cache" not in {k.lower() for k in captured["headers"]}
    read_url(URL, no_cache=True)
    assert captured["headers"].get("x-no-cache") == "true"
