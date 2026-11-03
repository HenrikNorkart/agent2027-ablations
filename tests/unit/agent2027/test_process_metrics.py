"""E11b process metrics (process_metrics.py) on hand-computed toy examples."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import process_metrics as pm

pytestmark = pytest.mark.unit


def table(rows):
    cols = ["Attribute", "Pandas Command", "Status", "Added", "Pruned Iteration", "Seeded", "Fallback", "Scored", "Trial",
            "fold_run_id", "iteration_in_fold"]
    return pd.DataFrame(rows, columns=cols)


def test_norm_ast():
    assert pm.norm_ast("result = df['a'] + 1") == pm.norm_ast('result=df["a"]+1  # same')
    assert pm.norm_ast("result = df['a'] + 1") != pm.norm_ast("result = df['a'] + 2")
    assert pm.norm_ast("result = (") is None and pm.norm_ast(None) is None


def test_reproposals_ast_and_value():
    t = table([
        ["x", "result = df['x']", "Active", "Seed", "", 1, 0, 1, "Trial 1", 0, 1],
        ["a", "result = df['x'] * 2", "Pruned", "Trial 0", "Trial 0", 0, 0, 1, "Trial 1", 0, 1],
        ["b", "result=df['x']*2", "Active", "Trial 1", "", 0, 0, 1, "Trial 1", 0, 1],      # same AST as a
        ["c", "result = df['x'] * 3 + 1", "Active", "Trial 1", "", 0, 0, 1, "Trial 1", 0, 1],  # r = 1 with x
        ["d", "result = df['y']", "Active", "Trial 1", "", 0, 0, 1, "Trial 1", 0, 1],
        ["f", "result = df['x'] * 2", "Active", "Trial 1", "", 0, 1, 1, "Trial 1", 0, 1],    # fallback: ignored
    ])
    rp = pm.reproposals(t)  # AST only
    assert rp.attribute.tolist() == ["a", "b", "c", "d"]
    assert rp.reproposal.tolist() == [False, True, False, False]
    x = np.arange(10, dtype=float)
    y = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=float)
    vals = pd.DataFrame({"x": x, "a": 2 * x, "b": 2 * x, "c": 3 * x + 1, "d": y, "f": 2 * x})
    rp = pm.reproposals(t, vals)
    # a duplicates seed x by value (|r| = 1), b by AST and value, c by value, d is new
    assert rp.reproposal.tolist() == [True, True, True, False]
    assert rp.dup_ast.tolist() == [False, True, False, False]
    # a matches only the seed x; b and c also match earlier LLM rows
    assert rp.reproposal_of_llm_row.tolist() == [False, True, True, False]


def test_reproposal_a10_and_prune_counts():
    t0 = table([["s", "result = df['s']", "Active", "Seed", "", 1, 0, 1, "Trial 0", 0, 0],
                ["a", "result = df['s'] + 1", "Pruned", "Trial 0", "Trial 0", 0, 0, 1, "Trial 0", 0, 0]])
    # restart-like table: a re-appears under a new name with the same command
    t1 = table([["s", "result = df['s']", "Active", "Seed", "", 1, 0, 1, "Trial 1", 0, 1],
                ["a2", "result = df['s']+1", "Active", "Trial 1", "", 0, 0, 1, "Trial 1", 0, 1],
                ["b", "result = df['s'] - 1", "Pruned", "Trial 1", "Trial 1", 0, 0, 1, "Trial 1", 0, 1]])
    assert pm.reproposal_a10([t0, t1]) == (1, 3)  # a2 re-proposes a; a, b new
    assert pm.prune_counts(t0) == (1, 2)
    assert pm.prune_counts(t1) == (1, 3)


METRIC_MSG = """Assess the current feature set based on pre-computed metrics. Use this metric table as the baseline:

| feature_name | mutual_information | vif | pfi_mean | pfi_std | noise_sensitivity | mean_abs_shap | warnings |
|---|---:|---:|---:|---:|---:|---:|---|
| a | 0.1 | 1e+12 | 0.2 | 0.01 | 0 | 0.05 |  |
| b | 0.3 | 1e+12 | 0.2 | 0.01 | 0 | 0.02 |  |
| c | 0.01 | 20 | 0.001 | 0.002 | 0 | 0.05 |  |
| d | 0.2 | 20 | 0.5 | 0.01 | 0 | 0.05 |  |
| e | 0.0 | 1e+12 | 0.0 | 0.01 | 0 | 0.001 |  |
| g | 0.0 | 3 | 0.0 | 0.01 | 0 | 0.0 |  |

Then decide."""


def test_metric_table_and_posthoc_ja():
    mt = pm.parse_metric_table(METRIC_MSG)
    assert mt.feature.tolist() == list("abcdeg")
    assert mt.loc[1, "vif"] == 1e12 and mt.loc[2, "pfi_std"] == 0.002
    # cluster: lowest SHAP among VIF >= 1e10 & SHAP <= 0.1, e protected (seeded) -> b;
    # pfi rule: VIF in [10, 1e10), |pfi| <= std, SHAP < 0.1 -> c; g has VIF < 10
    assert pm.ja_posthoc(mt, protected=["e"]) == ["b", "c"]
    assert pm.ja_posthoc(mt) == ["e", "c"]
    assert pm.jaccard(["b", "d"], ["b", "c"]) == pytest.approx(1 / 3)
    assert pm.jaccard([], []) == 1.0
    assert pm.parse_metric_table("no table") is None


def test_recover_within3():
    E, X, A = {"error": True}, {}, {"accepted": True}
    assert pm.recover_within3([E, X, X, A]) == (1, 1)
    assert pm.recover_within3([E, X, X, X, A]) == (0, 1)
    assert pm.recover_within3([E, E, A]) == (2, 2)
    assert pm.recover_within3([X, A]) == (0, 0)


def test_extractor_calls_from_trace(tmp_path):
    def rec(seq, ts, idx, tool_calls, results):
        msgs = [{"role": "tool", "tool_call_id": cid, "content": txt} for cid, txt in results]
        return {"agent_name": "ExtractorAgent", "seq": seq, "index": idx, "ts_start": ts - 1, "ts_end": ts,
                "messages": msgs, "message": {"tool_calls": [{"id": c, "function": {"name": n}} for c, n in tool_calls]}}
    lines = [rec(1, 100.0, 0, [("c1", "append_new_attribute")], []),
             rec(2, 110.0, 0, [("c2", "generic_pandas_tool")], [("c1", "Error updating attributes: boom")]),
             rec(3, 120.0, 0, [("c3", "append_new_attribute")], [("c1", "Error updating attributes: boom"),
                                                                  ("c2", "Result: 3")]),
             rec(4, 130.0, 0, [], [("c3", "Attribute z explained, documented, and added to df_features.")])]
    p = tmp_path / "llm_traces.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in lines:
            fh.write(json.dumps(r) + "\n")
    errs = [{"agent_name": "ExtractorAgent", "tool_name": "append_new_attribute", "message": "Error updating attributes: boom",
             "timestamp_utc": "1970-01-01T00:01:45+00:00", "step": 0, "context": {"site": "h4"}},
            {"agent_name": "ExtractorAgent", "tool_name": "runner", "message": "x",
             "timestamp_utc": "1970-01-01T00:01:46+00:00", "step": 0, "context": {}}]  # runner: not a tool error
    by = pm.extractor_calls(p, errs)
    calls = by[0]
    assert [c["id"] for c in calls] == ["c1", "c2", "c3"]
    assert [c["error"] for c in calls] == [True, False, False]
    assert [c["accepted"] for c in calls] == [False, False, True]
    assert pm.recover_within3(calls) == (1, 1)


def test_focus_repeat_with_fixed_vectors():
    vec = {"a": [1, 0], "a'": [0.95, 0.312], "b": [0, 1], "c": [1, 1]}
    emb = lambda xs: np.array([vec[x] for x in xs], dtype=float)  # noqa: E731
    r = pm.focus_repeat([(0, 0, "a"), (0, 1, "a'"), (0, 2, "b"), (1, 0, "c"), (1, 1, "c")], emb)
    # fold 0: a' vs a cos = 0.95 (repeat); b vs {a, a'} max 0.312 (no); fold 1: c vs c = 1 (repeat)
    assert r["n_focus_compared"] == 3 and r["focus_novelty"] == pytest.approx(2 / 3)
    t = pm.tfidf_embedder(["add ratio of a and b", "add ratio of a and b", "log transform"])
    assert t.shape[0] == 3


def test_stratified_sample():
    df = pd.DataFrame({"i": range(100), "lab": [1] * 10 + [0] * 90})
    s = pm.stratified_sample(df, "lab", n=60)
    assert len(s) == 60 and int(s.lab.sum()) == 10  # only 10 positives exist
    s = pm.stratified_sample(pd.DataFrame({"i": range(100), "lab": [1, 0] * 50}), "lab", n=60)
    assert int(s.lab.sum()) == 30


def test_row_metrics_on_toy_output(tmp_path):
    out = tmp_path / "output"
    (out / "intermediate_steps").mkdir(parents=True)
    (out / "instrumentation_summary.json").write_text(json.dumps(
        {"err_per_accepted": 0.5, "n_tool_errors": 2, "n_accepted": 4, "tok_compl_extractor": 400,
         "restore_pruned_count": 1, "floor_activations": 0, "report_read_rate": 0.25}))
    t0 = table([["s", "result = df['s']", "Active", "Seed", "", 1, 0, 1, "Trial 0", 0, 0],
                ["a", "result = df['s'] + 1", "Pruned", "Trial 0", "Trial 0", 0, 0, 1, "Trial 0", 0, 0]])
    t1 = table([["s", "result = df['s']", "Active", "Seed", "", 1, 0, 1, "Trial 1", 0, 1],
                ["a", "result = df['s'] + 1", "Pruned", "Trial 0", "Trial 0", 0, 0, 0, "Trial 1", 0, 1],
                ["b", "result = df['s']+1", "Active", "Trial 1", "", 0, 0, 1, "Trial 1", 0, 1]])
    t0.to_csv(out / "intermediate_steps" / "df_attributes_explanations_trail_0.csv", index=False)
    t1.to_csv(out / "intermediate_steps" / "df_attributes_explanations_trail_1.csv", index=False)
    t2 = table([["s", "result = df['s']", "Active", "Seed", "", 1, 0, 1, "Trial 2", 0, 2],
                ["a", "result = df['s'] + 1", "Pruned", "Trial 0", "Trial 0", 0, 0, 0, "Trial 2", 0, 2],
                ["b", "result = df['s']+1", "Active", "Trial 1", "", 0, 0, 1, "Trial 2", 0, 2]])
    t2.to_csv(out / "intermediate_steps" / "df_attributes_explanations_trail_2.csv", index=False)
    with open(out / "focus_log.jsonl", "w") as fh:
        for i, txt in enumerate(["ratios", "ratios", "logs"]):
            fh.write(json.dumps({"index": i, "fold_run_id": 0, "iteration_in_fold": i, "focus": txt, "source": "llm"}) + "\n")
    with open(out / "iteration_log.jsonl", "w") as fh:
        for i in range(3):
            fh.write(json.dumps({"index": i, "fold_run_id": 0, "test_fold_id": 0, "iteration_in_fold": i,
                                 "raw_coverage": 1.0}) + "\n")
    sheets = {"impl": [], "rec": [], "lineage": []}
    r = pm.row_metrics(out, embed=pm.tfidf_embedder, embed_name="tfidf", sheets=sheets, row_id="X-1")
    assert r["err_per_accepted"] == 0.5 and r["tok_compl_extractor_per_accepted"] == 100
    # last table: a and b new; b same AST as a -> 1/2
    assert r["reproposal_ratio"] == pytest.approx(0.5) and r["reproposal_basis"] == "ast"
    assert r["reproposal_ratio_a10"] == pytest.approx(0.5)  # t0: a new; t1: b dup of a
    # pruned at Trial 0: a (scored 2); Trial 1: none (scored 2); Trial 2: none (scored 2)
    assert (r["pruned_n"], r["scored_n"]) == (1, 6) and r["prune_rate"] == pytest.approx(1 / 6)
    # focus 0 added a, focus 1 added b, focus 2 added nothing
    assert r["focus_implementable"] == pytest.approx(2 / 3)
    # tfidf: "ratios" twice -> cos 1 (repeat); "logs" vs both -> 0
    assert r["focus_novelty"] == pytest.approx(0.5)
    assert [s["rule_label"] for s in sheets["impl"]] == [1, 1, 0]
    assert "prune_jaccard" not in r  # no shadow and no trace


def test_write_sheets_hides_labels(tmp_path):
    sheets = {"impl": [{"row_id": f"R{i}", "focus": f"f{i}", "rule_label": i % 2} for i in range(80)], "rec": [],
              "lineage": []}
    pm.write_sheets(sheets, tmp_path)
    s = pd.read_csv(tmp_path / "kappa_implementability_sheet.csv")
    k = pd.read_csv(tmp_path / "kappa_implementability_sheet_key.csv")
    assert len(s) == 60 and "rule_label" not in s.columns and {"rater1_label", "rater2_label"} <= set(s.columns)
    assert int(k.rule_label.sum()) == 30 and (s.sheet_id == k.sheet_id).all()
