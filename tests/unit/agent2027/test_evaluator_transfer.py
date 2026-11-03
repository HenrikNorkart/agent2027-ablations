"""A22 (M3) evaluator_transfer.py: hand-computed toy checks (A22-T10, T11 done-when)."""
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.agent2027.analysis import evaluator_transfer as et

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------ T10: xgb400 == Tester source
def _source_params(task_type):
    src = (ROOT / "rogueone" / "llm" / "agents" / "core" / "tester" / "agent.py").read_text(encoding="utf-8")
    fn = "get_regression_metrics" if task_type == "regression" else "get_classification_metrics_tabular"
    body = src[src.index(f"def {fn}("):]
    block = body[body.index("base_params = {"):body.index("}", body.index("base_params = {"))]
    out = {}
    for k, v in re.findall(r'"(\w+)":\s*([^,\n]+),', block):
        out[k] = v.strip()
    return out


@pytest.mark.parametrize("task_type", ["regression", "classification"])
def test_xgb400_params_equal_tester_source(task_type):
    src = _source_params(task_type)
    mine = et.xgb400_params(task_type, fold_seed=4, n_jobs=8)
    assert set(src) == set(mine)
    for k, v in src.items():
        if k == "random_state":
            assert v == "self.fold_seed" and mine[k] == 4
        elif k == "n_jobs":
            assert v == "_XGB_N_JOBS" and mine[k] == 8
        else:
            assert mine[k] == eval(v), k  # noqa: S307 - literal from the source


# ------------------------------------------------------------------ evaluators: phi gain > 0, duplicate ~ 0
def _toy_regression(n=600, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(1, 3, n), rng.uniform(1, 3, n)
    phi = a / b
    y = 3 * phi + rng.normal(0, 0.05, n)
    X = pd.DataFrame({"a": a, "b": b, "n1": rng.normal(size=n)})
    return X, y, phi


def _toy_classification(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(1, 3, n), rng.uniform(1, 3, n)
    phi = a * b  # a curved boundary in (a, b): no raw column or linear mix of them gives it
    y = (phi + rng.normal(0, 0.05, n) > 4.0).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "n1": rng.normal(size=n)})
    return X, y, phi


@pytest.mark.parametrize("ev", ["xgb400", "linear", "rf"])
def test_injected_phi_positive_gain_regression(ev):
    X, y, phi = _toy_regression()
    tr, te = slice(0, 450), slice(450, None)
    raw = et.fit_score(ev, X.iloc[tr], y[tr], X.iloc[te], y[te], "regression", 42, 2)
    Xp = X.assign(phi=phi)
    wp = et.fit_score(ev, Xp.iloc[tr], y[tr], Xp.iloc[te], y[te], "regression", 42, 2)
    assert et.gain(wp, raw, "regression", "synp") > 0.0      # Delta R2
    assert et.gain(wp, raw, "regression", "core10") > 0.0    # relative nRMSE reduction


@pytest.mark.parametrize("ev", ["xgb400", "linear", "rf"])
def test_injected_phi_positive_gain_classification(ev):
    X, y, phi = _toy_classification()
    tr, te = slice(0, 800), slice(800, None)
    raw = et.fit_score(ev, X.iloc[tr], y[tr], X.iloc[te], y[te], "classification", 42, 2)
    Xp = X.assign(phi=phi)
    wp = et.fit_score(ev, Xp.iloc[tr], y[tr], Xp.iloc[te], y[te], "classification", 42, 2)
    assert et.gain(wp, raw, "classification", "synp") > 0.0


@pytest.mark.parametrize("ev", ["xgb400", "linear", "rf"])
def test_duplicated_raw_column_gain_about_zero(ev):
    X, y, _ = _toy_regression(seed=3)
    tr, te = slice(0, 450), slice(450, None)
    raw = et.fit_score(ev, X.iloc[tr], y[tr], X.iloc[te], y[te], "regression", 42, 2)
    Xd = X.assign(a_copy=X["a"])
    wd = et.fit_score(ev, Xd.iloc[tr], y[tr], Xd.iloc[te], y[te], "regression", 42, 2)
    assert abs(et.gain(wd, raw, "regression", "synp")) < 0.02


# ------------------------------------------------------------------ scores and parity (hand)
def test_balanced_accuracy_and_regression_scores():
    # classes 0 (3 rows) and 1 (1 row): predict all 0 -> balanced accuracy 0.5
    assert math.isclose(et.balanced_accuracy_tester(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0])), 0.5)
    s = et.regression_scores(np.array([1.0, 3.0]), np.array([2.0, 4.0]), np.array([3.0, 4.0]))
    # rmse = sqrt((1 + 0)/2); train mean 2 -> nrmse = rmse / 2
    assert math.isclose(s["rmse"], math.sqrt(0.5)) and math.isclose(s["nrmse"], math.sqrt(0.5) / 2)


def test_gain_core10_relative_nrmse():
    assert math.isclose(et.gain({"nrmse": 0.18}, {"nrmse": 0.20}, "regression", "core10"), 0.1)
    assert math.isclose(et.gain({"r2": 0.7}, {"r2": 0.6}, "regression", "synp"), 0.1)
    assert math.isclose(et.gain({"bacc": 0.8}, {"bacc": 0.75}, "classification", "core10"), 0.05)


def test_parity_check_tolerances():
    ok = et.parity_check("classification", {"bacc": 0.801, "auroc": 0.9}, {"accuracy": 0.8, "AUROC": 0.9015})
    assert ok["et_parity_ok"] == 1
    bad = et.parity_check("classification", {"bacc": 0.8025, "auroc": 0.9}, {"accuracy": 0.8, "AUROC": 0.9})
    assert bad["et_parity_ok"] == 0
    assert et.parity_check("classification", {"bacc": 0.8}, {"accuracy": float("nan")})["rescorable"] == 0
    assert et.parity_check("regression", {"rmse": 100.4}, {"RMSE": 100.0})["et_parity_ok"] == 1
    assert et.parity_check("regression", {"rmse": 100.6}, {"RMSE": 100.0})["et_parity_ok"] == 0


# ------------------------------------------------------------------ preprocessing and matrices (hand)
def test_linear_preprocess_hand():
    tr = np.array([[1.0, 5.0, np.inf], [np.nan, 5.0, 2.0], [3.0, 5.0, 4.0]])
    te = np.array([[np.nan, 7.0, -np.inf]])
    a, b, keep = et.linear_preprocess(tr, te, winsorise=False)
    assert keep == [0, 2]                       # column 1 is constant on train
    # col 0: NaN -> median 2 -> train [1,2,3], test 2 -> standardised 0
    assert math.isclose(b[0, 0], 0.0, abs_tol=1e-12)
    # col 2: inf -> NaN -> median(2,4)=3 -> train [3,2,4], test -inf -> 3 -> standardised 0
    assert math.isclose(b[0, 1], 0.0, abs_tol=1e-12)
    a2, b2, _ = et.linear_preprocess(np.arange(1000, dtype=float)[:, None], np.array([[5000.0]]), winsorise=True)
    assert b2[0, 0] == a2.max()                 # test value clipped at the training 99.5th percentile


def test_tester_matrices_order_and_mask():
    folds = [pd.DataFrame({"id": [10 * i + 1, 10 * i + 2], "f1": [i, i], "f2": [-i, -i], "target": [i, i]})
             for i in range(5)]
    m = et.tester_matrices(folds, 2, ["f2", "f1"])
    assert m["cols"] == ["f1", "f2"]                         # attribute order, not mask order
    assert list(m["ids_train"]) == [1, 2, 11, 12, 31, 32, 41, 42]
    assert list(m["ids_test"]) == [21, 22]
    snap = pd.DataFrame({"Attribute": ["f1", "f2", "f3"], "Scored_post_prune": [1, 0, 0], "Scored": [1, 1, 0]})
    assert et.scored_names(snap) == ["f1"]
    snap0 = snap.assign(Scored_post_prune=0)
    assert et.scored_names(snap0) == ["f1", "f2"]            # Tester keeps the pre-pruning matrix


def test_ro_extra_columns_live_seeded_fallback():
    snap = pd.DataFrame({"Attribute": ["raw_a", "llm1", "llm_const", "fb1"], "Seeded": [1, 0, 0, 0],
                         "Fallback": [0, 0, 0, 1]})
    X = pd.DataFrame({"raw_a": [1, 2, 3], "llm1": [1.0, np.nan, 2.0], "llm_const": [7, 7, 7], "fb1": [0, 1, 0]})
    r = et.ro_extra_columns(snap, X, list(X.columns))
    assert r == {"live": ["llm1", "fb1"], "dead": ["llm_const"]}
    r2 = et.ro_extra_columns(snap, X, list(X.columns), drop_fallback=True)
    assert r2["live"] == ["llm1"]
    assert not et.is_live([np.nan, np.nan]) and et.is_live([np.inf, 1.0])


def test_raw_block_trees_and_linear_one_hot():
    data = pd.DataFrame({"id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0], "c": ["b", "a", "b", "z"]})
    tr, te = et.raw_block(data, [1, 2, 3], [4], "xgb400")
    assert list(tr["c"]) == [1, 0, 1] and list(te["c"]) == [2]   # alphabetical codes over the full column
    tr, te = et.raw_block(data, [1, 2, 3], [4], "linear")
    assert list(tr.columns) == ["x", "c==a", "c==b"]             # levels from the training fold
    assert list(te.iloc[0]) == [4.0, 0.0, 0.0]                   # unseen level -> all zeros


# ------------------------------------------------------------------ section 6 labels and gate
def _H(v):
    return dict(zip(et.HEADROOM, v))


def test_label_T():
    S = {"synp_p2": 0.1, "synp_p4": 0.05}
    r = et.assign_label(S, S, _H([0.1, 0, 0.1, -0.1, 0.2, 0]), _H([0.1, 0, 0.1, -0.1, 0.2, 0]),
                        S, _H([0.1] * 6), {e: _H([0.1] * 6) for e in et.EVALUATORS},
                        {e: S for e in et.EVALUATORS}, h_usable=et.HEADROOM)
    assert r["T"] is True and r["X"] is None and r["B"] is False and r["label"] == "T"


def test_label_X_and_B():
    S = {"synp_p2": 0.1}
    lin = {"synp_p2": -0.1}
    r = et.assign_label(lin, lin, _H([-0.1] * 6), _H([-0.1] * 6), S, _H([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]),
                        {e: _H([-0.01] * 4 + [0.1, 0.1]) for e in et.EVALUATORS},
                        {e: {"synp_p2": -0.2} for e in et.EVALUATORS}, h_usable=et.HEADROOM)
    # U = synp_p2 + 3 H units (4); linear and rf <= 0 on all of U -> X; vs_ofe <= 0 on 4/6 H and S median <= 0 -> B
    assert r["T"] is False and r["X"] is True and r["B"] is True and r["label"] == "X+B"


def test_label_not_evaluable_and_N():
    S = {"synp_p2": 0.1}
    r = et.assign_label(S, S, _H([0.1] * 6), _H([0.1] * 6), S, _H([0.1] * 6), {}, {}, h_usable=et.HEADROOM[:3])
    assert r["label"] == "not evaluable"
    lin = {"synp_p2": 0.1}
    r = et.assign_label(lin, {"synp_p2": -0.1}, _H([0.1] * 6), _H([0.1] * 6), {"synp_p2": -0.1}, _H([-0.1] * 6),
                        {e: _H([np.nan] * 6) for e in et.EVALUATORS}, {}, h_usable=et.HEADROOM)
    assert r["label"] == "N"   # T fails (rf on S), |U| = 0 -> no X, control unavailable -> no B


def test_parity_gate_and_s_set():
    eps = pd.DataFrame({"row_id": ["r1"] * 5 + ["r2"] * 5, "fold": list(range(5)) * 2,
                        "rescorable": [1] * 9 + [0], "et_parity_ok": [1, 1, 0, 0, 0, 1, 1, 1, 1, None]})
    g = et.parity_gate(eps)
    assert g["n_rescorable"] == 9 and g["n_pass"] == 6 and g["excluded_run_rows"] == ["r1"]
    assert not g["gate_pass"]
    gains = pd.DataFrame({"row_id": ["a"] * 4, "fold": range(4), "dataset_id": ["synp_p2"] * 3 + ["synp_p1"],
                          "rec_sel": [1, 1, 1, 1]})
    assert et.s_set(gains) == ["synp_p2"]                        # p1 is non-discriminating


def test_boot_median_deterministic():
    a = et.boot_median([0.1, 0.2, 0.3])
    b = et.boot_median([0.1, 0.2, 0.3])
    assert a == b and math.isclose(a["median"], 0.2) and a["lo"] <= 0.2 <= a["hi"]


# ------------------------------------------------------------------ folds (T7): Rogue One's own splitter
def test_folds_partition(tmp_path):
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"id": np.arange(50), "x": rng.normal(size=50)})
    data.loc[10, "x"] = data.loc[11, "x"]  # duplicate raw rows must share a fold
    labels = pd.DataFrame({"id": np.arange(50), "target": rng.integers(0, 2, 50)})
    et._ensure_rogueone_importable()
    f = et.fold_test_ids(data, labels, 42, stratified=False, task_type="classification")
    flat = [i for fold in f for i in fold]
    assert sorted(flat) == list(range(50)) and len(f) == 5
    assert any(10 in fold and 11 in fold for fold in f)
    assert f == et.fold_test_ids(data, labels, 42, stratified=False, task_type="classification")
