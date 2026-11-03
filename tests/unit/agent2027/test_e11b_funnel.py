"""Unit tests for funnel.py (E11b feature funnel) on hand-computed toy examples."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import funnel as fn

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("msg,kind", [
    ("Attribute x already exists and is active.", "rej_dup_name"),
    ("Error: Command did not set 'result' variable for attribute 'x'.", "rej_no_result"),
    ("Result for attribute 'x' contains NaN or Inf values.", "rej_nan_inf"),
    ("Result for attribute 'x' contains non-numeric values.", "rej_non_numeric"),
    ("Error updating attributes: KeyError 'foo'", "rej_exec_error")])
def test_classify_append_error(msg, kind):
    assert fn.classify_append_error(msg) == kind


def test_validity_constant_near_dup_valid():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    vals = pd.DataFrame({"a": a, "const": np.ones(50), "dup": 2 * a + 1, "new": rng.normal(size=50),
                         "late": -a})
    refs = {"const": ["a"], "dup": ["a"], "new": ["a"], "late": []}
    v = fn.validity(vals, ["const", "dup", "new", "late", "missing"], refs)
    assert v == {"const": "rej_constant", "dup": "rej_near_dup", "new": "", "late": "", "missing": ""}
    # |r| is used: a sign flip of an Active feature is a duplicate too
    assert fn.validity(vals, ["late"], {"late": ["a"]}) == {"late": "rej_near_dup"}


def test_funnel_counts_hand_example():
    # 5 accepted (a..e) + 4 pre-acceptance rejections (1 dup name, 2 exec errors, 1 NaN/Inf)
    c = fn.funnel_counts(["a", "b", "c", "d", "e"],
                         {"rej_dup_name": 1, "rej_exec_error": 2, "rej_nan_inf": 1},
                         {"b": "rej_constant"},
                         {"a": True, "c": True, "d": False, "e": True},
                         final_active=["a", "c", "seed1"], selected_active=["a", "e", "seed1"])
    assert [c[s] for s in fn.STAGES] == [9, 6, 4, 3, 2, 1]
    assert c["rej_constant"] == 1 and c["rej_replay"] == 1 and c["rej_pruned"] == 1 and c["rej_not_selected"] == 1
    assert c["sel_any_valid"] == 2  # e was selected but pruned later
    # stage bookkeeping: each drop is explained by a rej_* column
    assert c["fun_proposed"] - c["fun_executes"] == c["rej_dup_name"] + c["rej_exec_error"] + c["rej_no_result"]
    assert c["fun_executes"] - c["fun_valid"] == (c["rej_nan_inf"] + c["rej_non_numeric"]
                                                  + c["rej_constant"] + c["rej_near_dup"])


def test_funnel_counts_without_replay_and_selection():
    c = fn.funnel_counts(["a", "b"], {}, {}, None, ["a", "b"], None)
    assert np.isnan(c["fun_replay_ok"]) and c["fun_survives_prune"] == 2 and c["fun_selected"] == 0


def test_phi_stage():
    inv = {"x": "rej_near_dup"}
    rok = {"y": True, "z": False}
    assert fn.phi_stage([], inv, rok, [], []) == "none"
    assert fn.phi_stage(["x"], inv, rok, [], []) == "invalid"
    assert fn.phi_stage(["z"], inv, rok, ["z"], ["z"]) == "valid"          # lost at replay
    assert fn.phi_stage(["y"], inv, rok, [], []) == "replay_ok"           # lost at pruning
    assert fn.phi_stage(["y"], inv, rok, ["y"], []) == "survives_prune"   # not in the selected iteration
    assert fn.phi_stage(["x", "y"], inv, rok, ["y"], ["y"]) == "selected"  # furthest stage wins


def _table(rows):
    return pd.DataFrame(rows, columns=["Attribute", "Pandas Command", "Status", "Added", "Seeded", "Fallback"])


def test_llm_rows_and_active_refs():
    t0 = _table([["r1", "c", "Active", "Seed", 1, 0], ["f1", "c", "Active", "Trial 0", 0, 0],
                 ["fb", "c", "Active", "Trial 0", 0, 1]])
    t1 = _table([["r1", "c", "Active", "Seed", 1, 0], ["f1", "c", "Pruned", "Trial 0", 0, 0],
                 ["fb", "c", "Active", "Trial 0", 0, 1], ["f2", "c", "Active", "Trial 1", 0, 0],
                 ["f3", "c", "Active", "Trial 1", 0, 0]])
    trails = [{"iteration_in_fold": 0, "trial": "Trial 0", "attributes": t0, "active": ["r1", "f1", "fb"]},
              {"iteration_in_fold": 1, "trial": "Trial 1", "attributes": t1, "active": ["r1", "fb", "f2", "f3"]}]
    llm = fn.llm_rows(t1)
    assert llm.Attribute.tolist() == ["f1", "f2", "f3"]  # seeded and fallback rows are not proposals
    refs = fn.active_refs(trails, llm)
    assert refs["f1"] == ["r1"]                        # first iteration: the seeded rows
    assert refs["f2"] == ["r1", "f1", "fb"]            # Active after trial 0
    assert refs["f3"] == ["r1", "f1", "fb", "f2"]      # + appended earlier in the same trial


def test_row_funnel_on_toy_output(tmp_path):
    """Two folds of a toy run folder: no replay (no task data), errors from the jsonl log."""
    out = tmp_path / "output"
    (out / "intermediate_steps").mkdir(parents=True)
    log = []
    for idx in range(4):
        fr, it = idx // 2, idx % 2
        log.append({"index": idx, "fold_run_id": fr, "test_fold_id": fr, "iteration_in_fold": it,
                    "status": "completed", "tokens": {}})
        if fr == 0:
            rows = [["r1", "c", "Active", "Seed", 1, 0], ["g1", "c", "Active", "Trial 0", 0, 0]]
            if it == 1:
                rows += [["g2", "c", "Active", "Trial 1", 0, 0], ["g3", "c", "Active", "Trial 1", 0, 0]]
                rows[1][2] = "Pruned"
        else:
            rows = [["r1", "c", "Active", "Seed", 1, 0]]
        t = _table(rows).assign(Trial=f"Trial {idx}", fold_run_id=fr, iteration_in_fold=it)
        t.to_csv(out / "intermediate_steps" / f"df_attributes_explanations_trail_{idx}.csv", index=False)
    x = np.arange(10.0)
    pd.DataFrame({"id": range(10), "r1": x, "g1": x ** 2, "g2": 3 * x, "g3": np.ones(10)}).to_csv(
        out / "intermediate_steps" / "df_entities_attributes_trail_1.csv", index=False)
    (out / "iteration_log.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")
    pd.DataFrame({"test_fold_id": [0, 1, 2, 3, 4], "status": ["completed"] * 2 + ["skipped"] * 3,
                  "id": ["Trial 0", "Trial 3", "", "", ""]}).to_csv(out / "fold_aggregation_best_iterations.csv",
                                                                    index=False)
    errs = [{"tool_name": "append_new_attribute", "step": 1, "message": "Error updating attributes: boom"},
            {"tool_name": "append_new_attribute", "step": 3, "message": "Attribute r1 already exists and is active."},
            {"tool_name": "generic_pandas_tool", "step": 1, "message": "ignored"}]
    (out / "agent_toolcall_errors.jsonl").write_text("\n".join(json.dumps(e) for e in errs), encoding="utf-8")
    eps = {e["fold"]: e for e in fn.row_funnel(out, replay=False)}
    f0 = eps[0]
    # fold 0: 3 accepted (g1..g3) + 1 exec error; g2 = 3*r1 (near-dup of Active r1), g3 constant;
    # g1 valid but Pruned at fold end; selected iteration 0 has g1 Active
    assert [f0[s] for s in ("fun_proposed", "fun_executes", "fun_valid", "fun_survives_prune", "fun_selected")] \
        == [4, 3, 1, 0, 0]
    assert f0["rej_near_dup"] == 1 and f0["rej_constant"] == 1 and f0["sel_any_valid"] == 1
    assert eps[1]["fun_proposed"] == 1 and eps[1]["fun_executes"] == 0  # the duplicate-name rejection


def test_replay_status_mirrors_fold_safe_replay():
    df_raw = pd.DataFrame({"id": [1, 2, 3, 4], "a": [1.0, 2.0, 3.0, 4.0]})
    table = pd.DataFrame({"Attribute": ["ok", "chain", "boom", "empty", "after_boom"],
                          "Pandas Command": ["result = df['a'] * 2", "result = df_features['ok'] + 1",
                                             "result = df['nope']", "", "result = df_features['boom'] + 1"]})
    mask = np.array([True, True, False, False])
    st = fn.replay_status(df_raw, table, mask)
    # a later command may read an earlier (NaN-filled) feature: it still replays, as in the code
    assert st == {"ok": True, "chain": True, "boom": False, "empty": False, "after_boom": True}


def test_row_funnel_restart_pool(tmp_path):
    """A10 restarts reset the pool: proposals of a discarded pool still count and are judged in their last table."""
    out = tmp_path / "output"
    (out / "intermediate_steps").mkdir(parents=True)
    tables = {0: [["g1", "c", "Active", "Trial 0", 0, 0]], 1: [["g2", "c", "Active", "Trial 1", 0, 0]]}
    log = []
    for idx, rows in tables.items():
        log.append({"index": idx, "fold_run_id": 0, "test_fold_id": 0, "iteration_in_fold": idx, "tokens": {}})
        _table(rows).assign(Trial=f"Trial {idx}", fold_run_id=0, iteration_in_fold=idx).to_csv(
            out / "intermediate_steps" / f"df_attributes_explanations_trail_{idx}.csv", index=False)
    pd.DataFrame({"id": range(5), "g1": [1.0, 2, 3, 4, 6]}).to_csv(
        out / "intermediate_steps" / "df_entities_attributes_trail_0.csv", index=False)
    pd.DataFrame({"id": range(5), "g2": [5.0, 1, 3, 2, 2]}).to_csv(
        out / "intermediate_steps" / "df_entities_attributes_trail_1.csv", index=False)
    (out / "iteration_log.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")
    pd.DataFrame({"test_fold_id": [0], "status": ["completed"], "id": ["Trial 1"]}).to_csv(
        out / "fold_aggregation_best_iterations.csv", index=False)
    f0 = {e["fold"]: e for e in fn.row_funnel(out, replay=False)}[0]
    assert [f0[s] for s in ("fun_proposed", "fun_valid", "fun_survives_prune", "fun_selected")] == [2, 2, 2, 1]
    assert f0["n_pool_tables"] == 2
