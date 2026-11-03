"""Tests for the query-level retrieval audit trail (baGY Q1, checklist T2.1).

The recorder is deliberately importable without the retrieval stack, so these
tests exercise the record format directly rather than through an agent.
"""

import importlib.util
import json
from pathlib import Path

import pytest

# Loaded straight from its file rather than via `rogueone.utils`, which pulls in
# openai/langchain through its package __init__. The module claims to be
# importable without the retrieval stack; loading it this way is what pins that
# claim -- if someone adds a heavy import to it, these tests fail.
_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "rogueone" / "utils" / "retrieval_audit.py"
)
_spec = importlib.util.spec_from_file_location("_retrieval_audit_under_test", _MODULE_PATH)
retrieval_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval_audit)


@pytest.fixture(autouse=True)
def _clean_recorder(monkeypatch):
    """Each test starts with no destination and no env override."""
    monkeypatch.delenv(retrieval_audit.ENV_TRANSCRIPT_PATH, raising=False)
    monkeypatch.delenv(retrieval_audit.ENV_TRANSCRIPT_FULL, raising=False)
    retrieval_audit.reset_for_testing()
    yield
    retrieval_audit.reset_for_testing()


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- disabled by default -----------------------------------------------------


def test_recorder_is_a_noop_until_configured():
    assert retrieval_audit.is_enabled() is False
    assert retrieval_audit.record(channel="web_search", event="query", caller="x") is None


def test_configure_none_disables(tmp_path):
    retrieval_audit.configure(tmp_path / "t.jsonl")
    assert retrieval_audit.is_enabled() is True
    retrieval_audit.configure(None)
    assert retrieval_audit.is_enabled() is False


# --- destination resolution --------------------------------------------------


def test_configure_from_output_dir_uses_conventional_filename(tmp_path):
    path = retrieval_audit.configure_from_output_dir(tmp_path)
    assert path == tmp_path / retrieval_audit.TRANSCRIPT_FILENAME


def test_env_var_overrides_output_dir(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "custom.jsonl"
    monkeypatch.setenv(retrieval_audit.ENV_TRANSCRIPT_PATH, str(override))
    path = retrieval_audit.configure_from_output_dir(tmp_path)
    assert path == override
    assert override.parent.exists()


def test_parent_directory_is_created(tmp_path):
    target = tmp_path / "deep" / "nested" / "t.jsonl"
    retrieval_audit.configure(target)
    retrieval_audit.record(channel="web_search", event="query", caller="x", query="q")
    assert target.exists()


# --- record content ----------------------------------------------------------


def test_record_writes_one_json_line_per_call(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    retrieval_audit.record(channel="web_search", event="query", caller="a", query="q1")
    retrieval_audit.record(channel="web_search", event="query", caller="a", query="q2")
    rows = _read(path)
    assert [r["query"] for r in rows] == ["q1", "q2"]
    assert [r["seq"] for r in rows] == [1, 2]


def test_record_captures_query_result_and_outcome(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    retrieval_audit.record(
        channel="web_search",
        event="query",
        caller="WebSearchAgent._web_search",
        query="torque balance scale",
        result="some results",
        ok=True,
        latency_s=1.25,
        query_sanitized="torque balance scale",
        variant_index=0,
    )
    (row,) = _read(path)
    assert row["channel"] == "web_search"
    assert row["event"] == "query"
    assert row["caller"] == "WebSearchAgent._web_search"
    assert row["query"] == "torque balance scale"
    assert row["result"] == "some results"
    assert row["result_chars"] == len("some results")
    assert row["ok"] is True
    assert row["latency_s"] == 1.25
    assert row["variant_index"] == 0
    assert "ts" in row


def test_none_result_is_distinct_from_empty_string(tmp_path):
    """The whole point of the transcript: "returned nothing" != "returned ''"."""
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    retrieval_audit.record(channel="vector_db", event="query", caller="a", result=None)
    retrieval_audit.record(channel="vector_db", event="query", caller="a", result="")
    none_row, empty_row = _read(path)
    assert none_row["result"] is None
    assert none_row["result_is_none"] is True
    assert empty_row["result"] == ""
    assert empty_row["result_is_none"] is False
    assert empty_row["result_chars"] == 0


def test_error_is_recorded(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    retrieval_audit.record(
        channel="web_search", event="query", caller="a", ok=False, error="HTTPError: 400"
    )
    (row,) = _read(path)
    assert row["ok"] is False
    assert row["error"] == "HTTPError: 400"


def test_none_extras_are_omitted_but_falsy_values_kept(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    retrieval_audit.record(
        channel="vector_db", event="query", caller="a", collection=None, doc_count=0
    )
    (row,) = _read(path)
    assert "collection" not in row
    assert row["doc_count"] == 0


# --- truncation --------------------------------------------------------------


def test_long_results_are_clipped_and_flagged(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    big = "x" * (retrieval_audit._MAX_RESULT_CHARS + 100)
    retrieval_audit.record(channel="web_search", event="answer", caller="a", result=big)
    (row,) = _read(path)
    assert row["result_chars"] == len(big)
    assert len(row["result"]) == retrieval_audit._MAX_RESULT_CHARS
    assert row["result_truncated"] is True


def test_full_storage_can_be_turned_down_to_a_preview(tmp_path, monkeypatch):
    monkeypatch.setenv(retrieval_audit.ENV_TRANSCRIPT_FULL, "0")
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    text = "y" * 2000
    retrieval_audit.record(channel="web_search", event="answer", caller="a", result=text)
    (row,) = _read(path)
    assert len(row["result"]) == retrieval_audit._PREVIEW_CHARS
    assert row["result_chars"] == 2000


# --- failure isolation: a diagnostic must never break a run ------------------


def test_record_never_raises_when_destination_is_unwritable(tmp_path, monkeypatch):
    retrieval_audit.configure(tmp_path / "t.jsonl")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    assert retrieval_audit.record(channel="web_search", event="query", caller="a") is None


def test_configure_never_raises_on_a_bad_path(monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("pathlib.Path.mkdir", boom)
    assert retrieval_audit.configure("/nope/t.jsonl") is None
    assert retrieval_audit.is_enabled() is False


def test_unserialisable_values_do_not_break_the_record(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")

    class Weird:
        def __str__(self):
            return "<weird>"

    retrieval_audit.record(channel="vector_db", event="query", caller="a", thing=Weird())
    (row,) = _read(path)
    assert row["thing"] == "<weird>"


# --- timed_retrieval ---------------------------------------------------------


def test_timed_retrieval_records_success_with_latency(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    with retrieval_audit.timed_retrieval(
        channel="vector_db", event="answer", caller="KnowledgeAgent.explain_query",
        query="q", collection="c", doc_count=243,
    ) as audit:
        audit.result = "an answer"
    (row,) = _read(path)
    assert row["ok"] is True
    assert row["result"] == "an answer"
    assert row["collection"] == "c"
    assert row["doc_count"] == 243
    assert row["latency_s"] >= 0


def test_timed_retrieval_records_then_reraises(tmp_path):
    path = retrieval_audit.configure(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="kaboom"):
        with retrieval_audit.timed_retrieval(
            channel="web_search", event="answer", caller="a", query="q"
        ):
            raise ValueError("kaboom")
    (row,) = _read(path)
    assert row["ok"] is False
    assert "ValueError: kaboom" in row["error"]


def test_timed_retrieval_is_a_noop_when_disabled():
    with retrieval_audit.timed_retrieval(
        channel="web_search", event="answer", caller="a", query="q"
    ) as audit:
        audit.result = "x"
    assert retrieval_audit.is_enabled() is False
