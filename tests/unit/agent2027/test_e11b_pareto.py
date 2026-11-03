"""Unit tests for pareto.py (A23 CAP-6) on hand-computed toy examples."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import pareto as p

pytestmark = pytest.mark.unit


def test_front_mask_five_points():
    # (rec, cost): A=(0.8,100) B=(0.6,50) C=(0.5,80) D=(0.9,200) E=(0.8,150)
    y = [0.8, 0.6, 0.5, 0.9, 0.8]
    x = [100, 50, 80, 200, 150]
    # C is dominated by B (higher rec, lower cost); E by A (same rec, lower cost: one strict)
    assert p.front_mask(y, x).tolist() == [True, True, False, True, False]


def test_front_mask_ties_and_nan():
    # identical points do not dominate each other; NaN is never on the front
    assert p.front_mask([0.5, 0.5, np.nan], [10, 10, 1]).tolist() == [True, True, False]


@pytest.mark.parametrize("pf,word", [(0.8, "on the front"), (0.79, "near the front"), (0.5, "near the front"),
                                     (0.49, "dominated"), (float("nan"), "n/a")])
def test_wording_r_a23_1(pf, word):
    assert p.wording(pf) == word


@pytest.mark.parametrize("rid,notes,tag", [
    ("A17-401", "", "record"), ("A17-031", "", "excluded"), ("A17-505", "Amendment 19: re-run of A17-432", "infguard"),
    ("A19-412", "", "record"), ("A19-012", "", "excluded"), ("A20-003", "", "record"),
    ("A20-405", "", "a20_relaunch"), ("A20-905", "", "excluded"), ("A05-805", "", "power"),
    ("A05-032", "", "record"), ("A05-901", "freeze smoke", "excluded"), ("A16-001", "K-SET pilot", "excluded"),
    ("A12-705", "Amendment 16", "excluded"), ("A12-070", "", "record")])
def test_row_tag(rid, notes, tag):
    assert p.row_tag(rid, notes) == tag


def test_config_of():
    assert p.config_of("A05", "full", "record") == "full"
    assert p.config_of("A17", "f2_Q0.8", "record") == "f2_Q0.8"
    assert p.config_of("A20", "fx_pajM_Q4_cap12", "a20_relaunch") == "fx_pajM_Q4"
    assert p.config_of("A12", "fx_pajm", "record") == "fx_pajm"  # RV3 foldover cell
    assert p.config_of("A04", "x", "record") is None


def test_flop_proxy_hand_example():
    m = {"rung_S": "Q0.8", "rung_E": "Q122", "rung_T": "Q0.8",
         "env_overrides": json.dumps({"LLM_MODEL": "Qwen/Qwen3.5-122B-A10B-FP8"})}
    par = p.role_params(m)
    assert par == {"scientist": 0.8e9, "extractor": 10e9, "tester": 0.8e9, "unattributed": 10e9}
    pr = {"scientist": 100, "extractor": 1000, "tester": 10}
    co = {"scientist": 10, "extractor": 100, "tester": 1}
    # 2*0.8e9*(110+11) + 2*10e9*1100 ; completion only: 2*0.8e9*11 + 2*10e9*100
    assert p.flop(pr, co, par) == pytest.approx(1.936e11 + 2.2e13)
    assert p.flop(pr, co, par, compl_only=True) == pytest.approx(1.76e10 + 2e12)


def _write_output(d, folds_tokens, summary=None):
    d.mkdir(parents=True)
    lines = []
    for i, (f, pr, co) in enumerate(folds_tokens):
        lines.append(json.dumps({"index": i, "fold_run_id": f, "test_fold_id": f, "iteration_in_fold": 0,
                                 "tokens": {"scientist": {"prompt": pr, "compl": co, "calls": 1}}}))
    (d / "iteration_log.jsonl").write_text("\n".join(lines), encoding="utf-8")
    if summary is not None:
        (d / "instrumentation_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_run_costs_per_fold_and_capped(tmp_path):
    m = {"rung_S": "O120", "rung_E": "O120", "rung_T": "O120", "env_overrides": "{}"}
    # capped run: no summary, folds 0 and 1 reached only; unreached folds cost 0 (spent, never imputed)
    _write_output(tmp_path / "o", [(0, 100, 10), (0, 100, 20), (1, 50, 5)])
    cost, fc = p.run_costs(tmp_path / "o", m)
    assert cost["cost_source"] == "iteration_log"
    assert cost["tok_compl_total"] == 35
    assert fc[:, 0].tolist() == [30, 5, 0, 0, 0]
    assert fc[0, 1] == pytest.approx(2 * 5.1e9 * 230)
    assert cost["flop_proxy"] == pytest.approx(2 * 5.1e9 * (250 + 35))
    # finished run: summary totals are the run cost; iteration coverage reported
    _write_output(tmp_path / "s", [(f, 10, 1) for f in range(5)],
                  {"tok_compl_scientist": 5, "tok_prompt_scientist": 50, "tok_compl_total": 5, "tok_prompt_total": 50})
    cost, fc = p.run_costs(tmp_path / "s", m)
    assert cost["cost_source"] == "summary" and cost["iter_tok_coverage"] == pytest.approx(1.0)
    assert cost["flop_proxy_compl"] == pytest.approx(2 * 5.1e9 * 5)
    # no output at all: NaN cost
    cost, fc = p.run_costs(None, m)
    assert cost["cost_source"] == "missing" and np.isnan(fc).all()


def _units(spec):
    """spec: {config: [(task, rec_per_fold(list of 5), run_cost)]} -> units, points (one run per tuple)."""
    us, ps = [], []
    for cfg, runs in spec.items():
        for k, (task, recs, cost) in enumerate(runs):
            rid = f"{cfg}-{task}-{k}"
            ps.append({"row_id": rid, "config": cfg, "tag": "record", "itt_imputed": 0, "rerun_of": None,
                       "fold_seed": 42, "replicate": k, "tok_compl_total": cost, "flop_proxy": cost * 10,
                       "flop_proxy_compl": cost, "energy_j": np.nan, "energy_j_equal": np.nan, "fallback_share": 0.0})
            for f in range(5):
                us.append({"row_id": rid, "config": cfg, "tag": "record", "dataset_id": task, "fold_seed": 42,
                           "replicate": k, "fold": f, "rec_sel": recs[f], "tok_compl_total": cost / 5,
                           "flop_proxy": cost * 2, "flop_proxy_compl": cost / 5, "energy_j": np.nan,
                           "energy_j_equal": np.nan})
    return pd.DataFrame(us), pd.DataFrame(ps)


def test_p_front_known_front_four_configs():
    # constant data per configuration -> every resample gives the same picture (A23-T11 toy)
    one, zero = [1] * 5, [0] * 5
    spec = {"full": [("synp_p2", one, 100), ("synp_p3", one, 100)],
            "f2_Q0.8": [("synp_p2", zero, 10), ("synp_p3", zero, 10)],
            "f2_Q4": [("synp_p2", zero, 50), ("synp_p3", zero, 50)],
            "f2_Q122": [("synp_p2", one, 200), ("synp_p3", one, 200)]}
    u, pts = _units(spec)
    fr = p.panel_front(u, pts, "B", B=200, axes=("tok_compl_total", "flop_proxy"))
    f = fr[fr.axis == "flop_proxy"].set_index("config")
    assert f.loc[["full", "f2_Q0.8"], "p_front"].tolist() == [1.0, 1.0]
    assert f.loc[["f2_Q4", "f2_Q122"], "p_front"].tolist() == [0.0, 0.0]
    assert f.loc["full", "wording"] == "on the front" and f.loc["f2_Q4", "wording"] == "dominated"
    assert f.loc["f2_Q122", "cost"] == pytest.approx(2000)  # 5 x mean fold cost (200*2) = run cost x 10
    assert f.loc["f2_Q122", "ratio_vs_full"] == pytest.approx(2.0)
    assert f.loc["f2_Q122", "equal_band"] and not f.loc["f2_Q122", "cheaper_at_equal"]
    assert f.loc["f2_Q0.8", "diff_vs_full"] == pytest.approx(-1.0)
    assert f.loc["f2_Q0.8", "cost_per_recovered"] == "no recovery"
    assert f.loc["full", "cost_per_recovered"] == pytest.approx(2000 / 10)  # flop 1000 per run x 2 runs / 10 eps


def test_point_estimates_and_cheaper_at_equal():
    # y = mean over fold-episodes; x = mean run cost; Q2 equal on recovery at 30% of Full's cost
    spec = {"full": [("synp_p2", [1, 1, 0, 0, 0], 100), ("synp_p4", [1, 0, 0, 0, 0], 300)],
            "f2_Q2": [("synp_p2", [1, 1, 0, 0, 0], 30), ("synp_p4", [1, 0, 0, 0, 0], 90)]}
    u, pts = _units(spec)
    fr = p.panel_front(u, pts, "B", B=300, axes=("tok_compl_total",))
    f = fr.set_index("config")
    assert f.loc["full", "rec_sel"] == pytest.approx(0.3)
    assert f.loc["full", "cost"] == pytest.approx(200)
    assert f.loc["f2_Q2", "ratio_vs_full"] == pytest.approx(0.3)
    # identical recovery per (task, fold) -> paired difference is 0 in every resample
    assert f.loc["f2_Q2", "diff_lo90"] == 0 and f.loc["f2_Q2", "diff_hi90"] == 0
    assert f.loc["f2_Q2", "ratio_hi95"] == pytest.approx(0.3)
    assert f.loc["f2_Q2", "cheaper_at_equal"]
    assert f.loc["f2_Q2", "p_front"] == 1.0 and f.loc["full", "p_front"] == 0.0


def test_reference_points_never_in_front():
    spec = {"full": [("synp_p2", [0] * 5, 100)], "oracle": [("synp_p2", [1] * 5, 1)],
            "restarts": [("synp_p2", [0] * 5, 50)]}
    u, pts = _units(spec)
    fr = p.panel_front(u, pts, "A", B=50, axes=("tok_compl_total",)).set_index("config")
    assert fr.loc["oracle", "role"] == "reference" and np.isnan(fr.loc["oracle", "p_front"])
    assert fr.loc["restarts", "p_front"] == 1.0 and fr.loc["full", "p_front"] == 0.0


def test_select_population_and_infguard():
    units = pd.DataFrame([
        {"row_id": "A17-434", "config": "f2_Q4", "tag": "record", "dataset_id": "synp_p2", "fold_seed": 42},
        {"row_id": "A17-501", "config": "f2_Q4", "tag": "infguard", "dataset_id": "synp_p2", "fold_seed": 42},
        {"row_id": "A17-508", "config": "f2_Q4", "tag": "infguard", "dataset_id": "synp_p2", "fold_seed": 42},
        {"row_id": "A17-401", "config": "f2_Q0.8", "tag": "record", "dataset_id": "synp_p1", "fold_seed": 42},
        {"row_id": "A17-410", "config": "f2_Q0.8", "tag": "record", "dataset_id": "synp_p3", "fold_seed": 4},
        {"row_id": "A05-805", "config": "full", "tag": "power", "dataset_id": "synp_p2", "fold_seed": 42}])
    pts = pd.DataFrame({"row_id": ["A17-434", "A17-501", "A17-508", "A17-401", "A17-410", "A05-805"],
                        "tag": ["record", "infguard", "infguard", "record", "record", "power"],
                        "rerun_of": [None, "A17-434", "A17-434", None, None, None]})
    assert p.select(units, pts).row_id.tolist() == ["A17-434"]
    assert set(p.select(units, pts, "a_all7").row_id) == {"A17-434", "A17-401"}
    assert set(p.select(units, pts, "b_allseeds").row_id) == {"A17-434", "A17-410"}
    assert set(p.select(units, pts, "power").row_id) == {"A17-434", "A05-805"}
    assert p.select(units, pts, "infguard").row_id.tolist() == ["A17-508"]  # latest re-run replaces the row


def test_classify_panels():
    def fr(rows):
        base = {"primary_axis": True, "cheaper_at_equal": False, "diff95_excludes_0": False}
        return pd.DataFrame([{**base, **r} for r in rows])
    a = fr([{"panel": "A", "config": "full", "wording": "on the front"},
            {"panel": "A", "config": "llmfe", "wording": "dominated"}])
    assert p.classify(a, "A")["outcome"] == "Hoped (A)"
    a2 = fr([{"panel": "A", "config": "full", "wording": "near the front"},
             {"panel": "A", "config": "set_phased", "wording": "on the front", "cheaper_at_equal": True}])
    assert p.classify(a2, "A")["outcome"] == "Opposite"
    assert p.classify(fr([{"panel": "A", "config": "full", "wording": "near the front"}]), "A")["outcome"] == "Null (A)"
    assert p.classify(a, "A", token_invalid=True)["outcome"] == "Measurement failure"
    b = fr([{"panel": "B", "config": "mix_1b", "wording": "near the front", "cheaper_at_equal": True},
            {"panel": "B", "config": "f2_Q4", "wording": "on the front", "cheaper_at_equal": True}])
    assert p.classify(b, "B")["outcome"] == "Hoped variant (B)"
    b2 = fr([{"panel": "B", "config": "f2_Q4", "wording": "dominated"}])
    assert p.classify(b2, "B")["outcome"] == "Null B"
    b3 = fr([{"panel": "B", "config": "f2_Q4", "wording": "dominated", "diff95_excludes_0": True}])
    assert p.classify(b3, "B")["outcome"] == "Null A"


def test_external_units_llmfe_and_screen(tmp_path):
    ep = pd.DataFrame({"dataset_id": ["synp_p2"] * 3, "arm": ["llmfe-r0-best3", "llmfe-r0-any", "llmfe-r1-best3"],
                       "fold_seed": [42] * 3, "fold": [0, 0, 1], "rec_sel": [1, 0, 0]})
    ep.to_csv(tmp_path / "x6.csv", index=False)
    (tmp_path / "t.json").write_text(json.dumps({"a08-llmfe-synp_p2-r0": {"prompt_tokens": 90, "completion_tokens": 10}}))
    sc = pd.DataFrame({"dataset_id": ["synp_p2"] * 3, "arm": ["openfe", "raw", "autofeat"], "fold_seed": [42, 42, 4],
                       "fold": [0, 0, 0], "rec_sel": [1, 1, 1]})
    sc.to_csv(tmp_path / "sc.csv", index=False)
    u = p.external_units(str(tmp_path / "x6.csv"), [str(tmp_path / "t.json")], str(tmp_path / "sc.csv"))
    assert u.config.tolist() == ["llmfe", "llmfe", "openfe"]  # best-3 only; screen arms at seed 42 only
    assert u.iloc[0].tok_compl_total == 10 and u.iloc[0].flop_proxy == pytest.approx(2 * 5.1e9 * 100)
    assert np.isnan(u.iloc[1].tok_compl_total)  # r1 has no proxy totals -> no cost
    assert u.iloc[2].tok_compl_total == 0


def test_sensitivity_and_outcome_changes():
    prim = pd.DataFrame({"panel": ["A", "A", "AB"], "axis": ["t"] * 3, "config": ["full", "l0", "l0"],
                         "primary_axis": [True] * 3, "wording": ["on the front", "dominated", "dominated"],
                         "cheaper_at_equal": [np.nan, False, False]})
    sens = prim[prim.panel == "A"].copy()
    assert p.sensitivity_changes(prim, sens, "x").empty  # NaN == NaN; panels absent from the variant are skipped
    sens.loc[1, "wording"] = "near the front"
    assert p.sensitivity_changes(prim, sens, "x").config.tolist() == ["l0"]
    oc = p.outcome_changes([{"panel": "A", "outcome": "Null (A)", "variant": "primary"},
                            {"panel": "A", "outcome": "Null (A)", "variant": "a_all7"},
                            {"panel": "A", "outcome": "Opposite", "variant": "power"}])
    assert oc.sensitivity.tolist() == ["power"]
