"""A18 RP1: ``rp1_score.py sample`` (A18-T6 done-when), the §2 scoring rules, the A18 move
test in ``replay_tool.py`` and ``rp1_clients.sh`` (A18-T7: bash -n, DRY_RUN)."""

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.agent2027 import replay_tool as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "agent2027" / "analysis"))
import rp1_score as rp  # noqa: E402

pytestmark = [pytest.mark.unit]
REPO = Path(__file__).resolve().parents[3]

S_TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {},
                                                                       "required": []}}}
           for n in ("generic_pandas_tool", "search_tool", "attribute_lookup_tool", "list_all_attributes",
                     "take_a_note", "read_notes")]
S_TOOLS[0]["function"]["parameters"] = {"type": "object", "properties": {"command": {"type": "string"}},
                                        "required": ["command"]}
E_TOOLS = [
    {"type": "function", "function": {"name": "generic_pandas_tool", "parameters": {
        "type": "object", "required": ["args"], "properties": {"args": {
            "type": "object", "required": ["command", "explanation"],
            "properties": {"command": {}, "explanation": {}}}}}}},
    {"type": "function", "function": {"name": "append_new_attribute", "parameters": {
        "type": "object", "required": ["args"], "properties": {"args": {
            "type": "object", "required": ["name", "command", "argumentation"],
            "properties": {"name": {}, "command": {}, "argumentation": {}}}}}}},
    {"type": "function", "function": {"name": "list_known_attributes_tool",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "search_in_literature_tool", "parameters": {
        "type": "object", "properties": {"query": {}}, "required": ["query"]}}},
]
TA_TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {},
                                                                        "required": []}}}
            for n in ("generic_python_executor_tool", "get_precomputed_metrics_tool", "attribute_lookup_tool",
                      "take_note_tool", "get_notes_tool", "search_in_literature_tool")]
TP_TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {},
                                                                        "required": []}}}
            for n in ("attribute_pruning_tool", "get_precomputed_metrics_tool", "attribute_lookup_tool",
                      "generic_python_executor_tool", "get_notes_tool")]
W_TOOLS = [{"type": "function", "function": {"name": "_web_search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]


def _call(name, args):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _trace(seq, agent, tools, message, index, stage, caller=None):
    return {"seq": seq, "call_seq": 1, "session_name": f"s_{index}", "agent_name": agent, "caller": caller,
            "role": "x", "stage": stage, "index": index, "fold_run_id": index // 10,
            "iteration_in_fold": index % 10, "messages": [{"role": "system", "content": "sys"},
                                                          {"role": "user", "content": f"go {seq}"}],
            "tools": tools, "tool_choice": "<openai.Omit object at 0x1>", "extra_body": {}, "temperature": 0.7,
            "max_tokens": 32768, "model": rp.O120, "seed": 42, "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            "usage": {"prompt_tokens": 100 + seq, "completion_tokens": 5}, "attempts": [],
            "response_model": rp.O120}


#: per run and cell, how many candidate turns the fixture writes (default 30)
def _make_run(root: Path, name: str, per_cell: dict, drop_field=None, skip_tables=()) -> None:
    out = root / name / "run-output"
    (out / "intermediate_steps").mkdir(parents=True)
    recs = []
    seq = 0
    builders = {
        "S_answer": ("ScientistAgent", S_TOOLS, {"role": "assistant", "content": "Build ratio features."}, "focus", None),
        "S_action": ("ScientistAgent", S_TOOLS, {"role": "assistant", "tool_calls": [
            _call("generic_pandas_tool", {"command": "result = 1"})]}, "focus", None),
        "E_append": ("ExtractorAgent", E_TOOLS, {"role": "assistant", "tool_calls": [_call(
            "append_new_attribute", {"args": {"name": "f", "command": "result = 1", "argumentation": "a"}})]},
            "append", None),
        "E_other": ("ExtractorAgent", E_TOOLS, {"role": "assistant", "tool_calls": [
            _call("list_known_attributes_tool", {})]}, "append", None),
        "Ta_answer": ("TesterAgent", TA_TOOLS, {"role": "assistant", "content": "# Summary"}, "assessment", None),
        "Ta_action": ("TesterAgent", TA_TOOLS, {"role": "assistant", "tool_calls": [
            _call("get_precomputed_metrics_tool", {})]}, "assessment", None),
        "Tp_prune": ("TesterAgent", TP_TOOLS, {"role": "assistant", "tool_calls": [
            _call("attribute_pruning_tool", {"attribute_names_list": ["f"]})]}, "pruning", None),
        "Tp_answer": ("TesterAgent", TP_TOOLS, {"role": "assistant", "content": "done"}, "pruning", None),
        "W_search": ("WebSearchAgent", W_TOOLS, {"role": "assistant", "tool_calls": [
            _call("_web_search", {"query": "credit risk features"})]}, "search", "scientist"),
        "W_answer": ("WebSearchAgent", W_TOOLS, {"role": "assistant", "content": "Answer."}, "search", "extractor"),
    }
    for cell, (agent, tools, msg, stage, caller) in builders.items():
        for j in range(per_cell.get(cell, 30)):
            seq += 1
            idx = (j * 7) % 50
            c = caller if not cell.startswith("W_") else ("scientist" if j % 2 else "extractor")
            recs.append(_trace(seq, agent, tools, msg, idx, stage, c))
    # ineligible records: E answer, S no-tool fallback, an error, a T pruning "other" action
    for agent, tools, msg, stage in (("ExtractorAgent", E_TOOLS, {"role": "assistant", "content": "ok"}, "append"),
                                     ("ScientistAgent", [], {"role": "assistant", "content": "fallback focus"}, "focus"),
                                     ("TesterAgent", TP_TOOLS, {"role": "assistant", "tool_calls": [
                                         _call("get_notes_tool", {})]}, "pruning")):
        seq += 1
        recs.append(_trace(seq, agent, tools, msg, 3, stage))
    seq += 1
    err = _trace(seq, "ScientistAgent", S_TOOLS, {}, 3, "focus")
    err["message"] = None
    err["error"] = {"type": "APIError"}
    recs.append(err)
    if drop_field:
        del recs[5][drop_field]
    with gzip.open(out / "llm_traces.jsonl.gz", "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    (out / "instrumentation_summary.json").write_text(json.dumps({"llm_calls_total": len(recs)}))
    with open(out / "focus_log.jsonl", "w") as fh:
        for i in range(50):
            fh.write(json.dumps({"index": i, "fold_run_id": i // 10, "focus": f"focus {i}"}) + "\n")
    for i in range(50):
        if i % 10 == 9:
            continue  # an iteration-skipped trial writes no trail files
        for kind in ("df_entities_attributes", "df_attributes_explanations", "df_scientist_feedback"):
            if (kind, i) in skip_tables:
                continue
            (out / "intermediate_steps" / f"{kind}_trail_{i}.csv").write_text("id\n1\n")
    (root / name / "meta.json").write_text(json.dumps({"id": name[-8:], "artifact": f"x/y/run-output:v{len(name)}"}))


@pytest.fixture
def sources(tmp_path):
    def build(per_cell=None, **kw):
        root = tmp_path / "src"
        if root.exists():
            shutil.rmtree(root)
        for row_id, ds, _tt, _twin in rp.SOURCE_ROWS:
            _make_run(root, f"A05__full__{ds}__f42__r0", per_cell or {}, **kw)
        return rp.load_local_runs(root)
    return build


# --------------------------------------------------------------------------- sample (A18-T6)
def test_allocation_counts_match_section4_exactly(sources):
    res = rp.build_prompt_set(sources(), seed=42)
    assert len(res["records"]) == 800
    got = {row["cell"]: row["drawn"] for row in res["allocation"]}
    assert got == {c: n for c, (_r, _s, n) in rp.ALLOCATION.items()}
    roles = {r: sum(1 for x in res["records"] if x["role"] == r) for r in rp.ROLES}
    assert roles == {"S": 200, "E": 300, "T": 200, "W": 100}
    # round-robin over the 18 runs: every run contributes to the 80-prompt S answer cell
    s_ans = [x for x in res["records"] if x["stratum"] == "S_answer"]
    assert len({x["provenance"]["run_name"] for x in s_ans}) == 18
    # W balanced by caller
    w = [x for x in res["records"] if x["stratum"] == "W_search"]
    assert abs(sum(x["caller"] == "scientist" for x in w) - 30) <= 1
    assert not any(f.startswith("fill") for f in res["log"])
    rec = res["records"][0]
    assert "tool_choice" not in rec["request"] and rec["request"]["seed"] == 42
    assert rec["recorded_response"]["message"] and rec["state"]["data_csv"].endswith("/data.csv")


def test_two_runs_with_seed_42_are_identical_and_seed_matters(sources, tmp_path):
    runs = sources()
    a = rp.build_prompt_set(runs, seed=42)
    b = rp.build_prompt_set(rp.load_local_runs(tmp_path / "src"), seed=42)
    key = lambda r: [(x["prompt_id"], x["provenance"]["run_name"], x["provenance"]["seq"]) for x in r["records"]]  # noqa: E731
    assert key(a) == key(b) and a["replay_order"] == b["replay_order"]
    pa = rp.write_prompt_set(a, tmp_path / "o1")
    pb = rp.write_prompt_set(b, tmp_path / "o2")
    assert pa["prompts"].read_bytes() == pb["prompts"].read_bytes()
    c = rp.build_prompt_set(runs, seed=43)
    assert key(c) != key(a)
    # replay order is a shuffle, not the canonical order
    assert a["replay_order"] != sorted(a["replay_order"])


def test_fill_rule_triggers_and_is_logged(sources):
    res = rp.build_prompt_set(sources(per_cell={"Ta_answer": 1}), seed=42)  # 18 available < 40
    got = {row["cell"]: row for row in res["allocation"]}
    assert got["Ta_answer"]["drawn"] == 18 and got["Ta_answer"]["available"] == 18
    assert got["Ta_action"]["drawn"] == 60 + 22 and got["Ta_action"]["fill_donated"] == 22
    assert any("fill: 22 from Ta_action for short cell Ta_answer" in line for line in res["log"])
    assert any("QC: stratum Ta_answer has 18 < 40" in line for line in res["log"])
    assert len(res["records"]) == 800
    filled = [r for r in res["records"] if r["provenance"]["filled_for"] == "Ta_answer"]
    assert len(filled) == 22 and all(r["stratum"] == "Ta_action" for r in filled)


def test_fill_rule_crosses_stage_within_role(sources):
    res = rp.build_prompt_set(sources(per_cell={"Ta_answer": 1, "Ta_action": 2}), seed=42)
    t = [r for r in res["records"] if r["role"] == "T"]
    assert len(t) == 200
    assert any("from Tp_" in line for line in res["log"])


def test_missing_t2_field_fails_qc(sources):
    with pytest.raises(rp.QCError, match="A18-T2 fields"):
        rp.build_prompt_set(sources(drop_field="caller"), seed=42)


def test_missing_t2_table_and_count_mismatch_fail_qc(sources, tmp_path):
    with pytest.raises(rp.QCError, match="T2 per-trial tables missing"):
        rp.build_prompt_set(sources(skip_tables={("df_scientist_feedback", 4)}), seed=42)
    runs = sources()
    (runs[0].out / "instrumentation_summary.json").write_text(json.dumps({"llm_calls_total": 1}))
    with pytest.raises(rp.QCError, match="llm_calls_total"):
        rp.build_prompt_set(rp.load_local_runs(tmp_path / "src"), seed=42)


def test_classify_record_exclusions():
    ok = _trace(1, "ScientistAgent", S_TOOLS, {"role": "assistant", "content": "x" * 20}, 0, "focus")
    assert rp.classify_record(ok) == ("S_answer", "ok")
    assert rp.classify_record({**ok, "tools": []})[1] == "s_no_tool_fallback"
    assert rp.classify_record({**ok, "error": {"t": 1}})[1] == "error"
    assert rp.classify_record({**ok, "message": None})[1] == "no_response"
    e = _trace(2, "ExtractorAgent", E_TOOLS, {"role": "assistant", "content": "done"}, 0, "append")
    assert rp.classify_record(e) == (None, "e_answer")
    mixed = {"role": "assistant", "tool_calls": [_call("generic_pandas_tool", {}), _call("append_new_attribute", {})]}
    assert rp.classify_record({**e, "message": mixed})[0] == "E_append"
    tp = _trace(3, "TesterAgent", TP_TOOLS, {"role": "assistant", "tool_calls": [_call("get_notes_tool", {})]},
                0, "pruning")
    assert rp.classify_record(tp) == (None, "tp_other_action")
    assert rp.iter_bucket(0) == "it0-2" and rp.iter_bucket(6) == "it3-6" and rp.iter_bucket(9) == "it7-9"


def test_latest_trail_skips_iteration_skipped_trials():
    assert rp.latest_trail({10, 11, 13}, 15, 1) == 13
    assert rp.latest_trail({10, 11, 13}, 10, 1) is None
    assert rp.latest_trail({8, 9}, 10, 1) is None


def test_sample_cli_from_dir(sources, tmp_path):
    sources()
    out = tmp_path / "out"
    assert rp.main(["sample", "--out", str(out), "--from-dir", str(tmp_path / "src")]) == 0
    recs = rt.read_records(out / rp.PROMPTS_FILE)
    assert len(recs) == 800 and [r["replay_order"] for r in recs] == list(range(800))
    meta = json.loads((out / "rp1-prompts-v0.meta.json").read_text())
    assert meta["n"] == 800 and len(meta["prompts_sha256"]) == 64
    # replay_tool reads the prompt-set records as they are (A18 metadata survives)
    n = rt.normalise_record(recs[0], 0)
    assert n["stratum"] == recs[0]["stratum"] and n["role"] in rp.ROLES


# --------------------------------------------------------------------------- §2 rules
def test_lexicons_and_sections():
    assert rp.is_additive("We should create interaction terms") and not rp.is_additive("additional pruning")
    assert rp.is_pruning_only("Drop the noisy columns; keep only age.")
    assert not rp.is_pruning_only("Remove x and build y")
    txt = "# Summary\nok\n**Feature-by-feature assessment**\n- Risks and uncertainties\nRecommended candidates"
    assert rp.t_sections_present(txt) == 3
    assert rp.share_named("age and bmi", ["age", "bmi", "chol", "hr"]) == 0.5


def test_w_query_rule_and_sanitize_matches_web_search():
    assert rp.w_query_usable("credit  risk features")
    assert not rp.w_query_usable("x = y + z")
    assert not rp.w_query_usable(" ".join(["w"] * 13))
    ws = pytest.importorskip("rogueone.llm.agents.core.knowledge.web_search")
    for q in ("a = b", "θ (phi) [x] & y", "plain query", "x" * 300 + " y"):
        assert rp.sanitize_query(q) == ws._sanitize_query(q)


def test_t_pruned_in_session_reads_the_tool_result():
    def turn(i, names, result):
        return [{"role": "assistant", "tool_calls": [{"id": f"p{i}", "function": {
                    "name": "attribute_pruning_tool", "arguments": json.dumps({"attribute_names_list": names})}}]},
                {"role": "tool", "tool_call_id": f"p{i}", "content": result}]
    msgs = (turn(1, ["a", "seed1"], "Attributes pruned successfully: ['a']. Not pruned (protected seeded raw "
                                    "columns): ['seed1'].")
            + turn(2, ["b"], "Attributes pruned successfully.")
            + turn(3, ["c"], "Error pruning attributes: boom"))
    assert rp.t_pruned_in_session(msgs) == ["a", "b"]


def test_focus_novelty_and_metric_table_and_ja():
    cos, novel = rp.focus_novelty("build ratio of income to debt", ["build ratio of income to debt"])
    assert cos > 0.99 and not novel
    assert rp.focus_novelty("anything", []) == (None, True)
    msgs = [{"role": "user", "content": "x\n| feature_name | mutual_information | vif | pfi_mean | pfi_std | "
             "noise_sensitivity | mean_abs_shap | warnings |\n|---|---:|---:|---:|---:|---:|---:|---|\n"
             "| a | 0.1 | 12 | 0.0 | 0.1 | 0 | 0.01 |  |\n| b | 0.3 | 2 | 0.2 | 0.01 | 0 | 0.5 |  |\nend"}]
    table = rp.parse_metric_table(msgs)
    assert [r["feature_name"] for r in table] == ["a", "b"] and table[0]["vif"] == 12.0
    ja = rp.ja_prune_set(table)
    if ja is not None:  # E9d rules present in this checkout
        assert ja == ["a"]


@pytest.fixture
def state_env(tmp_path):
    """data.csv + trail k for an E/S/T prompt of fold 0, trial 3 (latest trail k = 2)."""
    tasks = tmp_path / "tasks" / "classification" / "toy"
    tasks.mkdir(parents=True)
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "age": [30, 40, 50, 60, 35, 45],
                       "income": [1.0, 2.0, 3.5, 4.0, 2.5, 3.0]})
    df.to_csv(tasks / "data.csv", index=False)
    run = tmp_path / "work" / "sources" / "R" / "run-output" / "intermediate_steps"
    run.mkdir(parents=True)
    ents = pd.DataFrame({"id": df["id"], "age": df["age"], "old": [0, 1, 1, 1, 0, 1], "gone": [1] * 6})
    ents.to_csv(run / "df_entities_attributes_trail_2.csv", index=False)
    pd.DataFrame({"Attribute": ["age", "old", "gone"], "Status": ["Active", "Active", "Pruned"],
                  "Seeded": [1, 0, 0]}).to_csv(run / "df_attributes_explanations_trail_2.csv", index=False)
    pd.DataFrame({"id": ["Trial 1"], "cv_accuracy": [0.7]}).to_csv(run / "df_scientist_feedback_trail_2.csv",
                                                                    index=False)
    pd.DataFrame({"Attribute": ["age", "old", "gone"], "Status": ["Active", "Pruned", "Pruned"],
                  "Scored": [1, 1, 0]}).to_csv(run / "df_attributes_explanations_trail_3.csv", index=False)
    ents.to_csv(run / "df_entities_attributes_trail_3.csv", index=False)
    files = lambda i: {k: f"intermediate_steps/{p}_trail_{i}.csv" for k, p in (  # noqa: E731
        ("entities", "df_entities_attributes"), ("attributes", "df_attributes_explanations"),
        ("feedback", "df_scientist_feedback"))}

    def rec(role, stage, stratum, tools, messages=None, recorded=None):
        return {"prompt_id": "rp1-001", "role": role, "stage": stage, "stratum": stratum,
                "recorded_move": rp.CELL_MOVE[stratum],
                "request": {"messages": messages or [{"role": "user", "content": "go"}], "tools": tools},
                "recorded_response": {"message": recorded or {"content": "x"}, "finish_reason": "stop"},
                "provenance": {"run_name": "R", "dataset_id": "toy", "index": 3, "fold_run_id": 0,
                               "iteration_in_fold": 3},
                "state": {"trail_k": 2, "trail_k_files": files(2), "trail_index_files": files(3),
                          "data_csv": "classification/toy/data.csv", "focus_history": ["study age effects"]}}

    def state(r):
        return rp.PromptState(r, tmp_path / "work" / "sources", tmp_path / "tasks")
    return rec, state


def _action(name, args):
    return {"final": {"finish_reason": "tool_calls", "content": None,
                      "tool_calls": [{"name": name, "arguments": json.dumps(args)}]},
            "raw": None, "error_kind": None, "clipped": False}


def _answer(text):
    return {"final": {"finish_reason": "stop", "content": text, "tool_calls": []},
            "raw": {"finish_reason": "stop", "content": text, "tool_calls": []}, "error_kind": None, "clipped": False}


def test_score_e_append_accepted_duplicate_unprune_and_correlated(state_env):
    rec, state = state_env
    r = rec("E", "append", "E_append", E_TOOLS)
    good = rp.score_move(r, _action("append_new_attribute", {"args": {
        "name": "inc_sq", "command": "result = df.groupby('id')['income'].first() ** 2", "argumentation": "a"}}),
        state(r))
    assert good["rp1_valid_call_rescued"] and good["rp1_executes"] and good["rp1_usable"]
    corr = rp.score_move(r, _action("append_new_attribute", {"args": {
        "name": "age2", "command": "result = df.groupby('id')['age'].first() * 2", "argumentation": "a"}}),
        state(r))
    assert corr["rp1_executes"] and not corr["rp1_usable"]  # |r| = 1 with the Active seeded 'age'
    dup = rp.score_move(r, _action("append_new_attribute", {"args": {
        "name": "old", "command": "result = 1", "argumentation": "a"}}), state(r))
    assert dup["rp1_executes"] is False and not dup["rp1_usable"] and "duplicate_active" in dup["exec_errors"]
    unp = rp.score_move(r, _action("append_new_attribute", {"args": {
        "name": "gone", "command": "", "argumentation": "a"}}), state(r))
    assert unp["auto_unprune"] and unp["rp1_executes"] is None and unp["rp1_usable"]
    bad = rp.score_move(r, _action("append_new_attribute", {"args": {
        "name": "z", "command": "result = df['nope']", "argumentation": "a"}}), state(r))
    assert bad["rp1_executes"] is False and not bad["rp1_usable"]
    ans = rp.score_move(r, _answer("I am done."), state(r))
    assert ans["rp1_valid_call_rescued"] and not ans["rp1_usable"] and ans["answer_instead_of_action"]


def test_score_e_history_appends_are_replayed(state_env):
    rec, state = state_env
    hist = [{"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [_call("append_new_attribute", {"args": {
                "name": "inc2", "command": "result = df.groupby('id')['income'].first() + 1",
                "argumentation": "a"}})]},
            {"role": "tool", "tool_call_id": "c-append_new_attribute",
             "content": "Attribute inc2 explained, documented, and added to df_features."}]
    r = rec("E", "append", "E_append", E_TOOLS, messages=hist)
    res = rp.score_move(r, _action("generic_pandas_tool", {"args": {
        "command": "result = df_features['inc2'].sum()", "explanation": "x"}}), state(r))
    assert res["rp1_executes"] and res["rp1_usable"]
    illegal = rp.score_move(r, _action("generic_pandas_tool", {"args": {
        "command": "df = df.dropna()", "explanation": "x"}}), state(r))
    assert illegal["rp1_executes"] is False and "illegal" in illegal["exec_errors"]


def test_score_s_answer_and_action(state_env):
    rec, state = state_env
    r = rec("S", "focus", "S_answer", S_TOOLS)
    ok = rp.score_move(r, _answer("Build interaction features between income and age groups."), state(r))
    assert ok["rp1_valid_call_rescued"] and ok["rp1_usable"] and ok["s_names_column"] and ok["s_additive"]
    short = rp.score_move(r, _answer("ok go"), state(r))
    assert not short["rp1_valid_call_rescued"] and short["valid_reason"] == "answer_too_short"
    prune_only = rp.score_move(r, _answer("Drop income, it is noisy and weak overall."), state(r))
    assert prune_only["rp1_valid_call_rescued"] and not prune_only["rp1_usable"]
    act = rp.score_move(r, _action("generic_pandas_tool", {"command": "result = df_test_results['cv_accuracy'].mean()"}),
                        state(r))
    assert act["rp1_executes"] and act["rp1_usable"]
    lookup = rp.score_move(r, _action("attribute_lookup_tool", {"attribute_names_list": ["nope"]}), state(r))
    assert lookup["rp1_valid_call_rescued"] and not lookup["rp1_usable"]


def test_score_t_and_w(state_env):
    rec, state = state_env
    r = rec("T", "pruning", "Tp_prune", TP_TOOLS,
            recorded={"tool_calls": [_call("attribute_pruning_tool", {"attribute_names_list": ["old"]})]})
    p = rp.score_move(r, _action("attribute_pruning_tool", {"attribute_names_list": ["old"]}), state(r))
    assert p["rp1_usable"] and p["jaccard_O120"] == 1.0 and p["phi_check"] == "n/a"
    p2 = rp.score_move(r, _action("attribute_pruning_tool", {"attribute_names_list": ["gone"]}), state(r))
    assert not p2["rp1_usable"]  # not Active (not scored in this trial)
    a = rec("T", "assessment", "Ta_answer", TA_TOOLS)
    rep = "# Summary\nage and old\n## Feature-by-feature\n## Risks\n## Recommended candidates\n"
    assert rp.score_move(a, _answer(rep), state(a))["rp1_usable"]
    assert not rp.score_move(a, _answer("# Summary\nnothing"), state(a))["rp1_usable"]
    w = rec("W", "search", "W_answer", W_TOOLS)
    assert rp.score_move(w, _answer("Here is what I found."), state(w))["rp1_usable"]
    assert not rp.score_move(w, _answer("Unable to provide an explanation at this time."), state(w))["rp1_usable"]
    ws = rec("W", "search", "W_search", W_TOOLS)
    assert rp.score_move(ws, _action("_web_search", {"query": "age risk"}), state(ws))["rp1_usable"]
    assert not rp.score_move(ws, _action("_web_search", {"query": "x = y"}), state(ws))["rp1_usable"]


def test_score_all_common_set_and_rates(state_env, tmp_path):
    rec, state = state_env
    r1 = rec("W", "search", "W_answer", W_TOOLS)
    r2 = dict(rec("W", "search", "W_answer", W_TOOLS), prompt_id="rp1-002")
    replies = {"Q4": {"rp1-001": {"final_message": {"content": "fine", "tool_calls": []},
                                  "final_finish_reason": "stop",
                                  "attempt_details": [{"finish_reason": "stop", "content": "fine", "tool_calls": []}]},
                      "rp1-002": {"error_kind": "context", "attempt_details": [{"error": "x"}]}}}
    rows = rp.score_all([r1, r2], replies, sources_root=tmp_path / "work" / "sources",
                        tasks_root=tmp_path / "tasks", jobs=1)
    assert {(r["target"], r["prompt_id"]): r["common_set"] for r in rows} == {
        ("recorded_O120", "rp1-001"): True, ("Q4", "rp1-001"): True,
        ("recorded_O120", "rp1-002"): False, ("Q4", "rp1-002"): False}
    q4 = next(r for r in rows if r["target"] == "Q4" and r["prompt_id"] == "rp1-001")
    assert q4["rp1_valid_call_raw"] and q4["rp1_valid_call_rescued"] and q4["rp1_usable"]
    assert next(r for r in rows if r["target"] == "recorded_O120")["rp1_valid_call_raw"] is None
    rates = {(x["target"], x["role"], x["score"]): x for x in rp.rate_table(rows)}
    assert rates[("Q4", "W", "rp1_usable")]["rate"] == 1.0 and rates[("Q4", "W", "rp1_usable")]["den"] == 1


# --------------------------------------------------------------------------- replay_tool A18 mode
def test_classify_move_answers_and_actions():
    assert rt.classify_move("stop", [], "x" * 11, S_TOOLS, answer_min_chars=10)["valid"]
    assert rt.classify_move("stop", [], "x" * 10, S_TOOLS, answer_min_chars=10)["reason"] == "answer_too_short"
    assert rt.classify_move("length", [], "long text", S_TOOLS)["reason"] == "finish_reason=length"
    assert not rt.classify_move("stop", [], "", W_TOOLS)["valid"]
    a = rt.classify_move("tool_calls", [{"name": "_web_search", "arguments": '{"query": "q"}'}], None, W_TOOLS)
    assert a["valid"] and a["move"] == "action"


def test_score_call_a18_and_summary_rates():
    from types import SimpleNamespace

    def final(content=None, calls=None, fr="stop"):
        tcs = [SimpleNamespace(function=SimpleNamespace(name=n, arguments=a)) for n, a in (calls or [])]
        msg = SimpleNamespace(content=content, tool_calls=tcs or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=fr)], usage=None, model="M")

    rows = []
    for i, (content, calls, fr, rm) in enumerate([
        ("A long enough focus statement", None, "stop", "answer"),
        ("short", None, "stop", "answer"),
        (None, [("generic_pandas_tool", '{"command": "result = 1"}')], "tool_calls", "action"),
        ("I answer instead", None, "stop", "action"),
    ]):
        row = {"prompt_id": f"p{i}", "role": "S", "recorded_move": rm, "stratum": "S_x", "error_kind": None}
        att = [{"finish_reason": fr, "server_finish_reason": fr, "content": content,
                "tool_calls": [{"name": n, "arguments": a} for n, a in (calls or [])], "usage": {}}]
        rows.append(rt.score_call(row, att, final(content, calls, fr), S_TOOLS, validity="a18",
                                  answer_min_chars=10))
    assert [r["rescued_valid"] for r in rows] == [True, False, True, True]
    assert [r["final_move"] for r in rows] == ["answer", "answer", "action", "answer"]
    s = rt.summarise(rows, prefix="rp1", validity="a18")
    assert s["rp1_valid_call_rescued__S"] == 0.75 and s["rp1_valid_call_rescued_n__S"] == 3
    assert s["answer_instead_of_action__S"] == 0.5 and s["n_prompts__S"] == 4
    a04 = rt.score_call({"prompt_id": "q"}, [], final("A long enough answer"), S_TOOLS)
    assert not a04["rescued_valid"]  # the A04 test is unchanged


# --------------------------------------------------------------------------- rp1_clients.sh
def _bash():
    exe = shutil.which("bash")
    if exe is None:
        pytest.skip("bash not available")
    return exe


def test_rp1_clients_syntax():
    assert subprocess.run([_bash(), "-n", str(REPO / "scripts/agent2027/rp1_clients.sh")]).returncode == 0


def test_rp1_clients_dry_run_prints_three_steps(tmp_path):
    import os

    env = {k: v for k, v in os.environ.items() if k not in ("SESSION_FILE", "HOST_SID", "WANDB_API_KEY")}
    env.update({"HOME": str(tmp_path), "RP1_ROW": "A18-003", "DRY_RUN": "1", "PORT": "8034",
           "MODEL": "Qwen/Qwen3.5-4B", "SERVER_NAME": "rogueone-vllm-q4-s1019g", "SESSION_ID": "S1019g",
           "RP1_MANIFEST": str(REPO / "manifests/agent2027/A18_RP1.csv"),
           "RUNGS_CSV": str(REPO / "scripts/agent2027/rungs.csv")})
    p = subprocess.run([_bash(), str(REPO / "scripts/agent2027/rp1_clients.sh")], env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    for needle in ("DRY_RUN step 1", "DRY_RUN step 2", "DRY_RUN step 3", "--seed 7042", "--concurrency 8",
                   "--clip-max-tokens 131072", "--validity a18", "A18__replay_Q4__rp1set__f42__r0",
                   "rp1-replies-Q4", "ro27-s1019g-a18-003", "rogue_one:agent2027-v2.0", "--init",
                   "RP1_END A18-003 rc=0"):
        assert needle in out, needle
    assert "WANDB_API_KEY=" not in out  # passed by name only
    bad = dict(env, MODEL="Qwen/Qwen3.5-0.8B")
    p2 = subprocess.run([_bash(), str(REPO / "scripts/agent2027/rp1_clients.sh")], env=bad,
                        capture_output=True, text=True)
    assert p2.returncode == 1 and "is not rung Q4" in p2.stdout
