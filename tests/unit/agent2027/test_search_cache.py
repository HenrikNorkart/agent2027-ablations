"""E4 web-search query cache, URL logging and caller attribution (01 §3.3 E4, §7 R4)."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from rogueone.llm.agents.core.knowledge import web_search as ws
from rogueone.utils import flag_registry as fr
from rogueone.utils import instrumentation as ins
from rogueone.utils import retrieval_audit
from rogueone.utils import search_cache as sc
from rogueone.utils.search_cache import SearchCache

PROVIDER = "searxng"


class FakeSearch:
    """Upstream stand-in with the ``run_with_meta`` shape of SearxngSearch."""

    def __init__(self, answers=None, raise_exc=None):
        self.calls: list[str] = []
        self.answers = answers or {}
        self.raise_exc = raise_exc

    def run_with_meta(self, query):
        self.calls.append(query)
        if self.raise_exc is not None:
            raise self.raise_exc
        text = self.answers.get(query, f"result for {query}")
        return text, [f"https://example.org/{len(self.calls)}"], ["mojeek"]


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    for k in (sc.ENV_MODE, sc.ENV_PATH, retrieval_audit.ENV_TRANSCRIPT_PATH):
        monkeypatch.delenv(k, raising=False)
    sc.reset_for_testing()
    retrieval_audit.reset_for_testing()
    ins.reset()
    yield
    sc.reset_for_testing()
    retrieval_audit.reset_for_testing()
    ins.reset()


def _transcript(tmp_path) -> Path:
    path = tmp_path / "retrieval_transcript.jsonl"
    retrieval_audit.configure(path)
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- flags
def test_flag_defaults_registered_and_off_by_default():
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_SEARCH_CACHE"] == "off"
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_SEARCH_CACHE_PATH"] == sc.DEFAULT_PATH
    assert sc.cache_mode() == "off"
    assert sc.get_cache().enabled is False


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv(sc.ENV_MODE, "sometimes")
    with pytest.raises(ValueError):
        sc.get_cache()


def test_get_cache_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.ENV_MODE, "Shared")
    monkeypatch.setenv(sc.ENV_PATH, str(tmp_path / "c.sqlite"))
    c = sc.get_cache()
    assert c.mode == "shared" and c.disk_ok and sc.get_cache() is c
    monkeypatch.setenv(sc.ENV_MODE, "run")
    assert sc.get_cache().mode == "run" and sc.get_cache().path is None


# --------------------------------------------------------------------------- SearchCache
def test_off_never_stores():
    c = SearchCache("off")
    c.put(PROVIDER, "q", "text", ["u"], ["e"])
    assert c.get(PROVIDER, "q") is None


def test_run_mode_is_in_process_only(tmp_path):
    c = SearchCache("run", str(tmp_path / "ignored.sqlite"))
    c.put(PROVIDER, "q", "text", ["https://a"], ["yahoo"])
    hit = c.get(PROVIDER, "q")
    assert hit.text == "text" and hit.urls == ["https://a"] and hit.source == "memo"
    assert not (tmp_path / "ignored.sqlite").exists()
    assert SearchCache("run").get(PROVIDER, "q") is None  # nothing shared across instances


def test_shared_persists_across_processes_and_first_writer_wins(tmp_path):
    path = str(tmp_path / "sub" / "cache.sqlite")
    a = SearchCache("shared", path)
    a.put(PROVIDER, "q", "first", ["https://1"], ["mojeek"])
    b = SearchCache("shared", path)  # a second run mounting the same file
    hit = b.get(PROVIDER, "q")
    assert hit.text == "first" and hit.urls == ["https://1"] and hit.engines == ["mojeek"] and hit.source == "disk"
    b.put(PROVIDER, "q", "second")
    assert SearchCache("shared", path).get(PROVIDER, "q").text == "first"
    # provider is part of the key
    assert b.get("serper", "q") is None


def test_empty_answers_are_not_cached(tmp_path):
    c = SearchCache("shared", str(tmp_path / "c.sqlite"))
    c.put(PROVIDER, "q", "   ")
    c.put(PROVIDER, "q2", "")
    assert c.get(PROVIDER, "q") is None and c.get(PROVIDER, "q2") is None


def test_record_overwrites_and_never_reads(tmp_path):
    path = str(tmp_path / "c.sqlite")
    SearchCache("shared", path).put(PROVIDER, "q", "old")
    rec = SearchCache("record", path)
    assert rec.get(PROVIDER, "q") is None
    rec.put(PROVIDER, "q", "new")
    assert SearchCache("shared", path).get(PROVIDER, "q").text == "new"


def test_replay_is_read_only(tmp_path):
    path = str(tmp_path / "c.sqlite")
    SearchCache("shared", path).put(PROVIDER, "q", "stored")
    rp = SearchCache("replay", path)
    assert rp.reads_upstream is False
    assert rp.get(PROVIDER, "q").text == "stored"
    rp.put(PROVIDER, "other", "x")
    assert SearchCache("shared", path).get(PROVIDER, "other") is None
    missing = SearchCache("replay", str(tmp_path / "nope.sqlite"))
    assert missing.disk_ok is False and missing.get(PROVIDER, "q") is None
    assert not (tmp_path / "nope.sqlite").exists()


def test_corrupt_cache_file_degrades_to_memo(tmp_path):
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not a sqlite database" * 100)
    c = SearchCache("shared", str(bad))
    assert c.disk_ok is False
    c.put(PROVIDER, "q", "text")
    assert c.get(PROVIDER, "q").text == "text"


def test_cache_schema(tmp_path):
    path = tmp_path / "c.sqlite"
    SearchCache("shared", str(path)).put(PROVIDER, "q", "t", ["https://u"], ["yahoo"])
    con = sqlite3.connect(str(path))
    row = con.execute("SELECT key, provider, query, result, urls, engines, schema FROM search_cache").fetchone()
    con.close()
    assert row[0] == hashlib.sha256(f"{PROVIDER}\x1fq".encode()).hexdigest()
    assert row[1:] == (PROVIDER, "q", "t", '["https://u"]', '["yahoo"]', sc.SCHEMA_VERSION)


# --------------------------------------------------------------------------- tool body
def test_off_mode_hits_upstream_every_time_and_logs_urls(tmp_path):
    tpath = _transcript(tmp_path)
    up = FakeSearch()
    cache = sc.get_cache()
    with ins.caller_scope("scientist"):
        out1 = ws._web_search_impl(up, PROVIDER, cache, "feature engineering ideas")
        out2 = ws._web_search_impl(up, PROVIDER, cache, "feature engineering ideas")
    assert out1 == out2 == "result for feature engineering ideas"
    assert len(up.calls) == 2
    rows = _rows(tpath)
    assert [r["ok"] for r in rows] == [True, True]
    assert rows[0]["urls"] == ["https://example.org/1"] and rows[0]["engines"] == ["mojeek"]
    assert rows[0]["caller_role"] == "scientist"
    assert "cache" not in rows[0] and "cache_state" not in rows[0]
    s = ins.compute_summary()
    assert s["serper_calls"] == 2 and s["serper_ok_false_rate"] == 0.0 and s["search_cache_hits"] == 0


def test_same_query_twice_hits_upstream_once(monkeypatch, tmp_path):
    """E4 done-when: the same query twice in one run hits the provider once (hit logged)."""
    monkeypatch.setenv(sc.ENV_MODE, "shared")
    monkeypatch.setenv(sc.ENV_PATH, str(tmp_path / "cache.sqlite"))
    tpath = _transcript(tmp_path)
    up = FakeSearch()
    out1 = ws._web_search_impl(up, PROVIDER, sc.get_cache(), "ratio features credit")
    out2 = ws._web_search_impl(up, PROVIDER, sc.get_cache(), "ratio  features (credit)")  # same once sanitised
    assert out1 == out2 and up.calls == ["ratio features credit"]
    rows = _rows(tpath)
    assert [r["cache_state"] for r in rows] == ["miss", "hit_memo"]
    assert rows[1]["urls"] == rows[0]["urls"] and rows[1]["cache"] == "shared"
    s = ins.compute_summary()
    assert s["serper_calls"] == 1 and s["search_cache_hits"] == 1 and s["search_cache_misses"] == 1


def test_shared_cache_serves_a_later_run_without_upstream(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.ENV_MODE, "shared")
    monkeypatch.setenv(sc.ENV_PATH, str(tmp_path / "cache.sqlite"))
    ws._web_search_impl(FakeSearch(), PROVIDER, sc.get_cache(), "blood donation features")
    sc.reset_for_testing()  # a new process mounting the same file
    later = FakeSearch()
    out = ws._web_search_impl(later, PROVIDER, sc.get_cache(), "blood donation features")
    assert out == "result for blood donation features" and later.calls == []


def test_run_mode_dedupes_in_process(monkeypatch):
    monkeypatch.setenv(sc.ENV_MODE, "run")
    up = FakeSearch()
    for _ in range(3):
        ws._web_search_impl(up, PROVIDER, sc.get_cache(), "census income log transform")
    assert len(up.calls) == 1


def test_replay_miss_never_reaches_upstream(monkeypatch, tmp_path):
    path = tmp_path / "cache.sqlite"
    SearchCache("shared", str(path)).put(PROVIDER, "known query", "stored answer", ["https://s"], ["yahoo"])
    monkeypatch.setenv(sc.ENV_MODE, "replay")
    monkeypatch.setenv(sc.ENV_PATH, str(path))
    tpath = _transcript(tmp_path)
    up = FakeSearch()
    assert ws._web_search_impl(up, PROVIDER, sc.get_cache(), "known query") == "stored answer"
    assert ws._web_search_impl(up, PROVIDER, sc.get_cache(), "unknown query words here") == "No results found."
    assert up.calls == []
    rows = _rows(tpath)
    assert rows[0]["cache_state"] == "hit_disk"
    assert all(r["error"] == "cache_miss_replay" for r in rows[1:]) and len(rows) == 1 + 2  # 2 variants


def test_fallback_variants_and_ok_false_rate(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.ENV_MODE, "shared")
    monkeypatch.setenv(sc.ENV_PATH, str(tmp_path / "cache.sqlite"))
    long_q = "one two three four five six seven eight nine ten"
    short = "one two three four five six seven eight"
    up = FakeSearch(answers={long_q: "", short: "short answer"})
    assert ws._web_search_impl(up, PROVIDER, sc.get_cache(), long_q) == "short answer"
    assert up.calls == [long_q, short]
    s = ins.compute_summary()
    assert s["serper_calls"] == 2 and s["serper_ok_false"] == 1 and s["serper_ok_false_rate"] == 0.5
    # the empty answer was not cached, the variant that answered was
    assert sc.get_cache().get(PROVIDER, long_q) is None
    assert sc.get_cache().get(PROVIDER, short).text == "short answer"


def test_errors_are_not_cached_and_return_unchanged_message(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.ENV_MODE, "shared")
    monkeypatch.setenv(sc.ENV_PATH, str(tmp_path / "cache.sqlite"))
    up = FakeSearch(raise_exc=RuntimeError("boom"))
    out = ws._web_search_impl(up, PROVIDER, sc.get_cache(), "abc")
    assert out == "All search attempts failed. Last error: boom"
    assert sc.get_cache().get(PROVIDER, "abc") is None


def test_empty_after_sanitization_unchanged():
    out = ws._web_search_impl(FakeSearch(), PROVIDER, sc.get_cache(), "=<>")
    assert out.startswith("Search query was empty after sanitization.")


# --------------------------------------------------------------------------- serper / searxng metadata
def test_searxng_run_with_meta_urls_follow_text():
    class S(ws.SearxngSearch):
        def results(self, query):
            return {
                "answers": [],
                "infoboxes": [{"content": "Box.", "id": "https://en.wikipedia.org/wiki/X", "engine": "wikipedia"}],
                "results": [
                    {"content": "a", "url": "https://a", "engines": ["mojeek", "yahoo"]},
                    {"content": "", "url": "https://empty", "engines": ["bing"]},
                    {"content": "b", "url": "https://b", "engine": "arxiv"},
                ],
            }

    text, urls, engines = S("http://h").run_with_meta("q")
    assert text == "Box. a b" == S("http://h").run("q")
    assert urls == ["https://en.wikipedia.org/wiki/X", "https://a", "https://b"]
    assert engines == ["arxiv", "mojeek", "wikipedia", "yahoo"]


def test_serper_path_uses_results_and_parse_results():
    class FakeSerper:
        def __init__(self):
            self.n = 0

        def results(self, query):
            self.n += 1
            return {"answerBox": {"link": "https://ab"}, "organic": [{"link": "https://o1"}, {"link": "https://ab"}]}

        def _parse_results(self, payload):
            return "parsed text"

        def run(self, query):  # pragma: no cover - must not be used
            raise AssertionError("run() must not be called when results() is available")

    s = FakeSerper()
    assert ws._search_with_meta(s, "q") == ("parsed text", ["https://ab", "https://o1"], ["serper"])
    assert s.n == 1


def test_plain_run_wrapper_falls_back():
    class Plain:
        def run(self, query):
            return "plain"

    assert ws._search_with_meta(Plain(), "q") == ("plain", [], [])


# --------------------------------------------------------------------------- caller attribution
def test_caller_scope_restores_previous_caller():
    ins.set_caller("tester")
    with ins.caller_scope("extractor"):
        assert ins.current_caller() == "extractor"
    assert ins.current_caller() == "tester"


def test_explain_query_charges_caller_and_logs_role(monkeypatch, tmp_path):
    tpath = _transcript(tmp_path)
    seen = {}

    async def fake_run(query):
        seen["caller"] = ins.current_caller()
        return "answer"

    monkeypatch.setattr(ws, "_run_search_agent", fake_run)
    agent = ws.WebSearchAgent(cfg=None)
    assert asyncio.run(agent.explain_query("q", caller_role="extractor")) == "answer"
    assert seen["caller"] == "extractor" and ins.current_caller() is None
    row = _rows(tpath)[-1]
    assert row["event"] == "answer" and row["caller_role"] == "extractor"


# --------------------------------------------------------------------------- prompt invariance
#: sha256 of the ``_web_search`` tool docstring and of the summariser instructions
#: before E4 (commit 2032f6b). E4 must leave both byte-identical.
_TOOL_DOC_SHA = "ba563473dae61ccab06fa23499b2c88645ef396202ba43a6b3f17ad35677b462"
_INSTRUCTIONS_SHA = "62e0d12c041e5682e232b5923361bd78d9c9bf7b153bc3297074d238dcf6aa36"


def test_tool_name_docstring_and_prompt_byte_identical():
    tree = ast.parse(Path(ws.__file__).read_text(encoding="utf-8"))
    doc = instr = None
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_web_search":
            names.append(node.name)
            doc = ast.get_docstring(node, clean=False)
        if isinstance(node, ast.keyword) and node.arg == "instructions":
            value = node.value
            # E9e wraps the literal in knowledge_backend.summariser_instructions(...),
            # which returns it unchanged under the default ``web`` backend.
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            if isinstance(value, ast.JoinedStr):
                instr = "".join(p.value for p in value.values if isinstance(p, ast.Constant))
    assert names == ["_web_search"]
    assert hashlib.sha256(doc.encode("utf-8")).hexdigest() == _TOOL_DOC_SHA
    assert hashlib.sha256(instr.encode("utf-8")).hexdigest() == _INSTRUCTIONS_SHA
