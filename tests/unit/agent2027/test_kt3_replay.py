"""E8b / A14-T4: offline TF-IDF replay of logged P7 queries (toy corpora)."""

import json

import pytest

from scripts.agent2027.analysis import kt3_replay as R
from rogueone.llm.agents.core.knowledge import local_corpus as lc

pytestmark = pytest.mark.unit


def _corpus(root, p7_text):
    root.mkdir(parents=True)
    (root / R.P7_DOC).write_text(f"# Marenko index\n{p7_text}\n", encoding="utf-8")
    (root / "d1.md").write_text("# Cable corrosion\ncorrosion humidity temperature cable\n", encoding="utf-8")
    (root / "d2.md").write_text("# Cable vibration\nvibration damping cable\n", encoding="utf-8")
    return lc.load_corpus(root)


def test_replay_counts_and_criteria(tmp_path):
    corpora = {"kt3_correct": _corpus(tmp_path / "a", "tension divided by spacing"),
               "kt3_wrong": _corpus(tmp_path / "b", "tension divided by pit depth")}
    queries = [{"row_id": "A02-016", "seq": i, "query": q} for i, q in enumerate([
        "Marenko index cable", "Marenko", "cable corrosion humidity", "quasar nebula",
        "spacing",
    ])]
    res = R.replay(queries, corpora)
    s = res["summary"]
    assert s["n_calls"] == 5 and s["n_calls_with_term"] == 2
    assert s["kt3_correct"]["term_top1"] == 2 and s["criterion_top3_term"]["pass"]
    assert s["kt3_correct"]["no_results"] == 1
    # "spacing" finds the P7 doc only in kt3_correct -> ranks differ once
    assert res["rows"][4]["kt3_correct_p7_rank"] == 1 and res["rows"][4]["kt3_wrong_p7_rank"] is None
    assert s["kt3_wrong"]["no_results"] == 2
    assert s["same_rank_both"] == 4


def test_transcript_reader_keeps_one_row_per_tool_call(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"event": "query", "variant_index": 0, "seq": 1, "query": "a b c d e f g h i"},
            {"event": "query", "variant_index": 1, "seq": 2, "query": "a b c d e f g h i"},
            {"event": "answer", "seq": 3, "query": "x"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert R.read_transcript_queries(p, "A02-008") == [{"row_id": "A02-008", "seq": 1, "query": "a b c d e f g h i"}]
