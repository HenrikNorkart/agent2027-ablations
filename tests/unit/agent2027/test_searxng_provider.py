"""Web-search provider switch (HQ #266): SearXNG client and provider factory."""

from __future__ import annotations

import pytest

from rogueone.llm.agents.core.knowledge import web_search as ws


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


def _patch_get(monkeypatch, payload, status=200, seen=None):
    import requests

    def fake_get(url, params=None, timeout=None):
        if seen is not None:
            seen.update(url=url, params=params, timeout=timeout)
        return _Resp(payload, status)

    monkeypatch.setattr(requests, "get", fake_get)


def test_run_formats_answers_infobox_then_top_k_snippets(monkeypatch):
    seen = {}
    payload = {
        "answers": ["42"],
        "infoboxes": [{"content": "Infobox text."}],
        "results": [{"content": f"snippet {i}"} for i in range(15)] + [{"content": ""}],
    }
    _patch_get(monkeypatch, payload, seen=seen)
    out = ws.SearxngSearch("http://h:8888/", k=10).run("reynolds number")
    assert out == "42 Infobox text. " + " ".join(f"snippet {i}" for i in range(10))
    assert seen["url"] == "http://h:8888/search"
    assert seen["params"]["format"] == "json" and seen["params"]["q"] == "reynolds number"


def test_zero_results_returns_empty_string(monkeypatch):
    _patch_get(monkeypatch, {"results": [], "answers": [], "infoboxes": []})
    assert ws.SearxngSearch("http://h:8888").run("x") == ""


def test_http_error_raises(monkeypatch):
    import requests

    _patch_get(monkeypatch, {}, status=503)
    with pytest.raises(requests.HTTPError):
        ws.SearxngSearch("http://h:8888").run("x")


def test_provider_factory(monkeypatch):
    monkeypatch.delenv("ROGUE_ONE_WEB_SEARCH_PROVIDER", raising=False)
    assert ws.web_search_provider() == "serper"
    monkeypatch.setenv("ROGUE_ONE_WEB_SEARCH_PROVIDER", "SearXNG")
    monkeypatch.setenv("ROGUE_ONE_SEARXNG_URL", "http://127.0.0.1:8888/")
    client = ws.make_search_client()
    assert isinstance(client, ws.SearxngSearch) and client.base_url == "http://127.0.0.1:8888"
    monkeypatch.setenv("ROGUE_ONE_WEB_SEARCH_PROVIDER", "bing")
    with pytest.raises(ValueError):
        ws.make_search_client()
