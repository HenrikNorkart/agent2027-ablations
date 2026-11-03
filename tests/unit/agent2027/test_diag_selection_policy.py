"""A21 (M2) anytime / iso-cost extension of paper/scripts/diag_selection_policy.py (A21-T7, T8).

Hand-computed toy pools. The script is loaded from its file (as test_retrieval_audit.py does).
The A21-T7 real-data halves (A21_ARTIFACTS_DIR, A21_T17_RUNS) skip when the env vars are unset.
"""
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

_PATH = Path(__file__).resolve().parents[3] / "paper" / "scripts" / "diag_selection_policy.py"
_spec = importlib.util.spec_from_file_location("_diag_selection_policy_under_test", _PATH)
dsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsp)


def _fold(rows):
    """rows: (iteration, status, heldout, inner, rec[, fold_scores])"""
    out = []
    for r in rows:
        it, st, h, inner, rec = r[:5]
        sc = r[5] if len(r) > 5 else None
        out.append({"test_fold_id": 0, "iteration_in_fold": it, "status": st, "heldout": h,
                    "protocol": h, "inner": inner, "rec_iter": rec,
                    "inner_fold_scores": None if sc is None else json.dumps(sc)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ rules (classification)
FOLD_C = _fold([
    (0, "completed", 0.70, 0.81, 0, [0.81, 0.81]),
    (1, "completed", 0.75, 0.85, 1, [0.80, 0.90]),   # inner 0.85, SE = 0.0707/sqrt2 = 0.05
    (2, "iteration_skipped", np.nan, np.nan, 0),
    (3, "completed", 0.90, 0.85, 1, [0.85, 0.85]),   # tie with iteration 1 -> first (it 1) wins
    (4, "completed", 0.60, 0.84, 0, [0.84, 0.84]),
    (5, "completed", 0.65, 0.83, 0, [0.83, 0.83]),
    (6, "completed", 0.66, 0.82, 0, [0.82, 0.82]),
    (7, "completed", 0.67, 0.81, 0, [0.81, 0.81]),
    (8, "completed", 0.68, 0.80, 0, [0.80, 0.80]),
    (9, "completed", 0.95, 0.79, 1, [0.79, 0.79]),
])


def test_rules_k10_classification():
    p = dsp.fold_rule_picks(FOLD_C, 10, isreg=False, run_inner_ok=True)
    assert p["inner_cv"]["iter"] == 1 and p["inner_cv"]["heldout"] == 0.75  # tie -> first row
    assert p["iter0"]["iter"] == 0 and p["last"]["iter"] == 9
    assert p["oracle"]["iter"] == 9 and p["oracle"]["heldout"] == 0.95
    # mean_all over the 9 completed rows (the skipped row never counts)
    hs = [0.70, 0.75, 0.90, 0.60, 0.65, 0.66, 0.67, 0.68, 0.95]
    assert math.isclose(p["mean_all"]["heldout"], np.mean(hs))
    assert math.isclose(p["mean_all"]["rec"], 3 / 9)
    # patience_2: it0 gain (0.81), it1 gain (0.85), it2 skipped (no gain, 1), it3 0.85 not > 0.85 (2) -> stop at 3
    assert p["patience_2"]["stop_iter"] == 3 and p["patience_2"]["iter"] == 1
    # patience_3: ... it4 0.84 no gain (3) -> stop at 4; pick among 0..4 = it 1
    assert p["patience_3"]["stop_iter"] == 4 and p["patience_3"]["iter"] == 1
    # one_se: best row it1 (0.85), SE = std([0.8, 0.9], ddof=1)/sqrt(2) = 0.05 -> earliest >= ~0.80 = it 0 (0.81)
    assert p["one_se"]["iter"] == 0


def test_rules_prefix_k3():
    p = dsp.fold_rule_picks(FOLD_C, 3, isreg=False, run_inner_ok=True)
    assert p["inner_cv"]["iter"] == 1
    assert p["last"]["iter"] == 1          # iteration 2 is skipped -> never selected
    assert math.isclose(p["mean_all"]["heldout"], (0.70 + 0.75) / 2)
    assert p["oracle"]["iter"] == 1
    assert p["patience_2"]["stop_iter"] == 2  # walked to k-1 (2): one no-gain only, cap at k-1


def test_all_nan_inner_fold_takes_last_completed_row():
    f = _fold([(0, "completed", 0.5, np.nan, 0), (1, "completed", 0.6, np.nan, 0),
               (2, "iteration_skipped", np.nan, np.nan, 0)])
    p = dsp.fold_rule_picks(f, 10, isreg=False, run_inner_ok=True)
    assert p["inner_cv"]["iter"] == 1


def test_no_finite_inner_in_run_selects_on_heldout():
    f = _fold([(0, "completed", 0.5, np.nan, 0), (1, "completed", 0.8, np.nan, 1), (2, "completed", 0.6, np.nan, 0)])
    p = dsp.fold_rule_picks(f, 10, isreg=False, run_inner_ok=False)
    assert p["inner_cv"]["iter"] == 1      # held-out argmax (flagged inner_cv_unavailable by the caller)
    assert p["one_se"] is None             # no inner values -> rule not computable


def test_regression_minimises():
    f = _fold([(0, "completed", 10.0, 4.9, 0, [4.9, 4.9]), (1, "completed", 9.0, 4.0, 0, [3.0, 5.0]),
               (2, "completed", 8.0, 4.5, 0, [4.5, 4.5])])
    p = dsp.fold_rule_picks(f, 10, isreg=True, run_inner_ok=True)
    assert p["inner_cv"]["iter"] == 1 and p["oracle"]["iter"] == 2
    # SE = std([3,5])/sqrt2 = 1.0 -> earliest with inner <= ~5.0 is iteration 0 (4.9)
    assert p["one_se"]["iter"] == 0


# ------------------------------------------------------------------ costs and AUC
ICOST = pd.DataFrame({"test_fold_id": [0, 0, 0, 1, 1, 1], "iteration_in_fold": [0, 1, 2, 0, 1, 2],
                      "tok_compl_total": [10.0, 20.0, 30.0, 1.0, 2.0, 3.0]})


def test_cost_curve():
    assert dsp.run_cost(ICOST, "tok_compl_total", 1) == 11.0
    assert dsp.run_cost(ICOST, "tok_compl_total", 2) == 33.0
    assert dsp.run_cost(ICOST, "tok_compl_total", 10) == 66.0
    assert dsp.cost_through(ICOST, "tok_compl_total", 0, 1) == 30.0


def test_anytime_auc_tok_hand():
    # C(1)=10, C(2)=20; scores 0.6, 0.8; base 0.5; T=40:
    # [0,10) 0.5 -> 5 ; [10,20) 0.6 -> 6 ; [20,40] 0.8 -> 16 ; total 27 / 40
    assert math.isclose(dsp.anytime_auc_tok({1: 10, 2: 20}, {1: 0.6, 2: 0.8}, 0.5, 40), 27 / 40)
    # T below C(1): only the base
    assert math.isclose(dsp.anytime_auc_tok({1: 10, 2: 20}, {1: 0.6, 2: 0.8}, 0.5, 5), 0.5)


def test_anytime_auc_iter_needs_all_k():
    assert math.isclose(dsp.anytime_auc_iter({k: k / 10 for k in range(1, 11)}), 0.55)
    assert math.isnan(dsp.anytime_auc_iter({1: 0.5}))


def test_first_rec():
    assert dsp.first_rec({0: 0, 1: 0, 2: 1}, {0: 5, 1: 5, 2: 5, 3: 5}) == (3, 15.0)
    assert dsp.first_rec({0: 0}, {0: 5, 1: 5}) == (11, 10.0)


# ------------------------------------------------------------------ iso-cost
CF = {k: 100.0 * k for k in range(1, 11)}


def test_iso_k_star():
    assert dsp.iso_k_star(CF, 350.0) == (3, "")
    assert dsp.iso_k_star(CF, 300.0) == (3, "")          # C_F(k) <= C_A: equality allowed
    assert dsp.iso_k_star(CF, 50.0) == (None, "no iso-cost point")
    assert dsp.iso_k_star(CF, 0.0) == (None, "no iso-cost point")   # 0-token CPU arm
    assert dsp.iso_k_star(CF, 1000.0) == (10, "arm >= Full's whole schedule")
    assert dsp.iso_k_star(CF, 5000.0) == (10, "arm >= Full's whole schedule")


def test_classify_and_rd():
    assert dsp.classify_delta(0.1, 0.02, 0.2) == ("+", "+")
    assert dsp.classify_delta(0.1, -0.05, 0.2) == ("+", "0")
    assert dsp.classify_delta(-0.1, -0.2, -0.01) == ("-", "-")
    assert dsp.rd_reading("+", "+", False) == "at matched cost +"
    assert dsp.rd_reading("+", "-", False) == "budget-sensitive"
    assert dsp.rd_reading("-", None, True).endswith("(arm >= Full's whole schedule)")


def test_n_active():
    assert dsp.n_active_b("O120") == 5.1 and dsp.n_active_b("Q122") == 10.0
    assert dsp.n_active_b("Q4") == 4.0 and dsp.n_active_b("Q0.8") == 0.8
    assert dsp.n_active_b("XYZ") is None


# ------------------------------------------------------------------ reading rules
def _eps(task, rec_any_by_k, rec_sel_by_k, n=10):
    rows = []
    for k in range(1, 11):
        for i in range(n):
            rows.append({"dataset_id": task, "k": k, "rec_any": int(i < rec_any_by_k(k)),
                         "rec_sel": int(i < rec_sel_by_k(k))})
    return pd.DataFrame(rows)


def test_rule_RA_early_late_na():
    early = _eps("synp_p2", lambda k: 9 if k >= 5 else 2, lambda k: 8 if k >= 5 else 1)
    assert dsp.rule_RA(early)["reading"] == "early"      # S5 = 1.0, rec_sel@5 = rec_sel@10
    late = _eps("synp_p2", lambda k: 4 if k < 8 else 10, lambda k: 3)
    assert dsp.rule_RA(late)["reading"] == "late"        # S5 = 0.4
    grad = _eps("synp_p2", lambda k: 6 if k < 8 else 10, lambda k: 3)
    assert dsp.rule_RA(grad)["reading"] == "gradual"     # S5 = 0.6
    none = _eps("synp_p2", lambda k: 0, lambda k: 0)
    assert dsp.rule_RA(none)["reading"].startswith("not applicable")
    p1only = _eps("synp_p1", lambda k: 10, lambda k: 10)  # non-discriminating task is out of the primary pop
    assert dsp.rule_RA(p1only)["reading"].startswith("not applicable")


def test_rule_RB():
    sc = {"heart": {k: 0.5 - 0.125 * (10 - k) for k in range(1, 11)},         # rises to 0.5
          "wine": {k: 0.25 + 0.0625 * (10 - k) for k in range(1, 11)}}         # nRMSE falls to 0.25
    sig = {"heart": 0.25, "wine": 0.125}  # binary-exact steps
    r = dsp.rule_RB(sc, sig, {"heart": "classification", "wine": "regression"})
    # k: heart (-(10-k)*0.5), wine (-(10-k)*0.5) -> mean -(10-k)/2 >= -0.5 iff k >= 9
    assert math.isclose(r["delta_std_k"][9], -0.5)
    assert r["k_hat"] == 9


def test_rule_RC():
    tt = {f"d{i}": "classification" for i in range(10)}
    # (a): k10 below k5 on 6 datasets
    ds = {f"d{i}": {"inner_cv": {3: 0.7, 5: 0.8, 10: 0.79 if i < 6 else 0.81},
                    "oracle": {3: 0.9, 10: 0.95}, "mean_all": {3: 0.6, 10: 0.6}} for i in range(10)}
    r = dsp.rule_RC(ds, tt)
    assert r["a_n_worse_k10_vs_k5"] == 6 and r["a"]
    # (b): share k3 = 1 - 0.1/0.3 = 2/3; k10 = mean(1 - 0.19/0.35 (x6), 1 - 0.21/0.35 (x4)) ~ 0.43 -> falls, b False
    assert not r["b"] and r["flag"]


# ------------------------------------------------------------------ legacy mode unchanged
def test_legacy_mode_unchanged(tmp_path, capsys):
    d = tmp_path / "runs" / "tasks" / "classification" / "heart" / "output"
    d.mkdir(parents=True)
    pd.DataFrame({
        "status": ["completed"] * 4, "test_fold_id": [0, 0, 1, 1], "iteration_in_fold": [0, 1, 0, 1],
        "accuracy": [0.70, 0.80, 0.60, 0.90], "inner_cv_accuracy": [0.75, 0.70, 0.50, 0.60],
    }).to_csv(d / "df_test_pool_final.csv", index=False)
    base = tmp_path / "b.csv"
    base.write_text("dataset,baseline,value\nheart,base_no_fe,0.65\n", encoding="utf-8")
    out = tmp_path / "o.csv"
    assert dsp.main(["--runs", str(tmp_path / "runs"), "--baselines", str(base), "--out", str(out)]) == 0
    r = pd.read_csv(out).iloc[0]
    # inner_cv picks it0 (0.70) and it1 (0.90) -> 0.80; mean_all = 0.75; oracle 0.85; iter0 0.65; last 0.85
    assert r["inner_cv"] == 0.8 and r["mean_all"] == 0.75 and r["oracle"] == 0.85
    assert r["iter0"] == 0.65 and r["last"] == 0.85 and r["inner_cv_vs_mean_all"] == 0.05
    assert r["mean_all_vs_base_no_fe"] == 0.1


def test_legacy_requires_runs_and_baselines(tmp_path):
    with pytest.raises(SystemExit):
        dsp.main(["--out", str(tmp_path / "o.csv")])


# ------------------------------------------------------------------ A21-T7 real-data checks (skip if unset)
@pytest.mark.skipif(not os.environ.get("A21_T17_RUNS"), reason="A21_T17_RUNS unset")
def test_legacy_reproduces_selection_policy_audit(tmp_path):
    root = Path(__file__).resolve().parents[3]
    out = tmp_path / "audit.csv"
    dsp.main(["--runs", os.environ["A21_T17_RUNS"], "--baselines",
              str(root / "paper" / "results_matched_harness_baselines.csv"), "--out", str(out)])
    ref = pd.read_csv(root / "paper" / "selection_policy_audit.csv")
    got = pd.read_csv(out)
    m = ref.merge(got, on="dataset", suffixes=("_ref", ""))
    for c in ("inner_cv", "iter0", "last", "mean_all", "oracle"):
        assert np.allclose(m[c + "_ref"], m[c], equal_nan=True)


@pytest.mark.skipif(not (os.environ.get("A21_ARTIFACTS_DIR") and os.environ.get("A21_MANIFEST")),
                    reason="A21_ARTIFACTS_DIR / A21_MANIFEST unset")
def test_k10_reproduces_protocol_on_real_runs(tmp_path):
    """A21-T7 test 2: on real runs (session roots in A21_ARTIFACTS_DIR, ':'-separated; groups in
    A21_GROUPS, default A05), the k = 10 inner_cv pick reproduces fold_aggregation_best_iterations.csv
    and heldout@10 reproduces fold_aggregation_summary.csv on every finished run."""
    groups = os.environ.get("A21_GROUPS", "A05").split(",")
    out = tmp_path / "o"
    rc = dsp.main(["--mode", "anytime", "--artifacts", *os.environ["A21_ARTIFACTS_DIR"].split(":"),
                   "--manifest", *os.environ["A21_MANIFEST"].split(":"), "--groups", *groups, "--out", str(out)])
    assert rc == 0
    v = pd.read_csv(out / "validity.csv")
    v = v[v["itt"] == 0]
    assert len(v) and v["k10_pick_reproduces"].eq(1).all() and v["heldout10_reproduces"].eq(1).all()
