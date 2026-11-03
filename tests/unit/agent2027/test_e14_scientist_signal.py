"""E14: ROGUE_ONE_SCIENTIST_SIGNAL (train_fit | inner_cv_l1) and the 01 4.4 item 2 check.

``train_fit`` (code default) must leave the Scientist's prompt, the feedback pool and
the trails byte-identical; ``inner_cv_l1`` replaces the ``train_*`` columns with one
inner-CV scalar computed on a split crossed with (fold-disjoint from) the selection
split, inside the outer-train rows only.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# rogue_one_main / the Tester need the LLM endpoint keys at import; values are irrelevant.
for _k, _v in {"LLM_API_KEY": "x", "LLM_PORT": "1"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from rogueone.utils import flag_registry as fr
from rogueone.utils import instrumentation, rouge_one_cfg
from rogueone.utils import scientist_signal as ss
from rogueone.utils.config import rogueoneConfig

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
SMALL_PARAMS = {
    "tree_method": "hist", "random_state": 42, "device": "cpu", "n_estimators": 20,
    "learning_rate": 0.1, "max_depth": 3, "n_jobs": 1,
}


@pytest.fixture
def signal(monkeypatch):
    def _set(mode):
        monkeypatch.setattr(rouge_one_cfg, "scientist_signal", mode)

    return _set


def _clf_data(n=120, seed=0, classes=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    if classes == 3:
        y = np.digitize(X[:, 0], [-0.5, 0.5])
    return X, y


# --------------------------------------------------------------------------- flag
def test_flag_default_and_validation(monkeypatch):
    assert fr.FLAG_DEFAULTS["ROGUE_ONE_SCIENTIST_SIGNAL"] == "train_fit"
    assert ss.parse_scientist_signal(None) == "train_fit"
    assert ss.parse_scientist_signal("") == "train_fit"
    assert ss.parse_scientist_signal(" Inner_CV_L1 ") == "inner_cv_l1"
    for bad in ("l1", "inner_cv", "train", "test"):
        with pytest.raises(ValueError):
            ss.parse_scientist_signal(bad)
    cfg = rogueoneConfig()
    monkeypatch.delenv("ROGUE_ONE_SCIENTIST_SIGNAL", raising=False)
    cfg.__init__()
    assert cfg.scientist_signal == "train_fit"
    monkeypatch.setenv("ROGUE_ONE_SCIENTIST_SIGNAL", "inner_cv_l1")
    cfg.__init__()
    assert cfg.scientist_signal == "inner_cv_l1"
    monkeypatch.setenv("ROGUE_ONE_SCIENTIST_SIGNAL", "bogus")
    with pytest.raises(ValueError):
        cfg.__init__()
    monkeypatch.delenv("ROGUE_ONE_SCIENTIST_SIGNAL", raising=False)
    cfg.__init__()


def test_v2_defaults_sets_l1():
    text = (ROOT / "scripts" / "agent2027" / "v2_defaults.env").read_text(encoding="utf-8")
    assert "\nROGUE_ONE_SCIENTIST_SIGNAL=inner_cv_l1\n" in text


# --------------------------------------------------------------------------- splits
@pytest.mark.parametrize("task,n,classes", [
    ("classification", 120, 2), ("classification", 97, 3), ("classification", 9, 2),
    ("regression", 101, 0), ("regression", 8, 0),
])
def test_selection_splits_match_inner_cv_score_splitter(task, n, classes):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 3))
    y = rng.integers(0, max(classes, 2), size=n) if task == "classification" else rng.normal(size=n)
    if task == "classification" and n == 9:
        y = np.array([0] * 7 + [1] * 2)  # a class rarer than 4 -> KFold fallback
    got = ss.build_selection_splits(y, task, 4, 42)
    if task == "classification" and np.unique(y, return_counts=True)[1].min() >= 4:
        ref = StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(X, y)
    else:
        ref = KFold(n_splits=4, shuffle=True, random_state=42).split(X, y)
    for (a_tr, a_va), (b_tr, b_va) in zip(got, ref):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)


def test_selection_score_reproduced_from_build_selection_splits():
    """_inner_cv_score (untouched) == a recomputation on build_selection_splits."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.utils import class_weight

    from rogueone.llm.agents.core.tester.agent import _inner_cv_score

    X, y = _clf_data(100)
    got = _inner_cv_score(X, y, SMALL_PARAMS, "classification", seed=42, balance_weights=True)
    Xf = X.astype(np.float32)
    scores = []
    for tr, va in ss.build_selection_splits(y, "classification", 4, 42):
        m = xgb.XGBClassifier(**SMALL_PARAMS)
        m.fit(Xf[tr], y[tr], sample_weight=class_weight.compute_sample_weight("balanced", y=y[tr]))
        scores.append(accuracy_score(y[va], m.predict(Xf[va])))
    assert got == pytest.approx(float(np.mean(scores)), abs=1e-12)


@pytest.mark.parametrize("task,n,classes", [
    ("classification", 120, 2), ("classification", 97, 3), ("classification", 33, 2),
    ("regression", 101, 0), ("regression", 8, 0), ("classification", 8, 2),
])
def test_feedback_split_is_crossed_partition(task, n, classes):
    rng = np.random.default_rng(2)
    y = rng.integers(0, max(classes, 2), size=n) if task == "classification" else rng.normal(size=n)
    sel = ss.build_selection_splits(y, task, 4, 42)
    fb = ss.build_feedback_splits(y, sel, task, 4, 42)
    res = ss.check_l1_split(sel, fb, n, train_ids=list(range(n)), test_ids=[-1, -2])
    assert res["l1_split_disjoint"] == 1 and res["l1_ids_checked"] == 1
    assert res["l1_max_val_overlap"] <= 0.5
    # validation fold sizes balanced (within the number of selection folds)
    sizes = [len(va) for _, va in fb]
    assert max(sizes) - min(sizes) <= 4
    # deterministic, and differs from the selection split
    fb2 = ss.build_feedback_splits(y, sel, task, 4, 42)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(fb, fb2))
    sel_vals = [set(va) for _, va in sel]
    assert not any(set(va) in sel_vals for _, va in fb)
    if task == "classification" and n >= 97:
        # approximately stratified: each fold's class share within 0.1 of the overall
        overall = np.bincount(y) / n
        for _, va in fb:
            share = np.bincount(y[va], minlength=len(overall)) / len(va)
            assert np.abs(share - overall).max() < 0.1


def test_feedback_overlap_bound_is_tight():
    y = np.zeros(400, dtype=int)
    y[::3] = 1
    sel = ss.build_selection_splits(y, "classification", 4, 7)
    fb = ss.build_feedback_splits(y, sel, "classification", 4, 7)
    res = ss.check_l1_split(sel, fb, 400)
    assert res["l1_max_val_overlap"] == pytest.approx(0.25, abs=0.02)
    assert res["l1_ids_checked"] == 0


def test_check_rejects_same_split_and_leaks():
    n = 80
    y = np.arange(n) % 2
    sel = ss.build_selection_splits(y, "classification", 4, 42)
    with pytest.raises(ss.L1SplitLeakageError):
        ss.check_l1_split(sel, sel, n)  # "same inner split" = coupled
    shifted = ss.build_selection_splits(y, "classification", 4, 43)  # another seed only
    with pytest.raises(ss.L1SplitLeakageError):
        # a different seed alone does not satisfy the rule: some fold overlaps > ceil(|V|/K)
        ss.check_l1_split(sel, shifted, n)
    fb = ss.build_feedback_splits(y, sel, "classification", 4, 42)
    ids = [f"r{i}" for i in range(n)]
    with pytest.raises(ss.L1SplitLeakageError):
        ss.check_l1_split(sel, fb, n, train_ids=ids, test_ids=["r3", "t9"])  # test id in train
    with pytest.raises(ss.L1SplitLeakageError):
        ss.check_l1_split(sel, fb, n, train_ids=ids[:-1] + ["r0"], test_ids=[])  # dup id
    with pytest.raises(ss.L1SplitLeakageError):
        ss.check_l1_split(sel, fb, n, train_ids=ids[:-1], test_ids=[])  # length mismatch
    broken = [(tr, va[:-1]) for tr, va in fb]  # not a partition any more
    with pytest.raises(ss.L1SplitLeakageError):
        ss.check_l1_split(sel, broken, n)


# --------------------------------------------------------------------------- score
def test_l1_score_is_balanced_accuracy_and_rmse():
    import xgboost as xgb
    from sklearn.metrics import balanced_accuracy_score, mean_squared_error

    X, y = _clf_data(160, classes=3)
    sel = ss.build_selection_splits(y, "classification", 4, 42)
    fb = ss.build_feedback_splits(y, sel, "classification", 4, 42)
    got = ss.l1_inner_cv_score(X, y, SMALL_PARAMS, "classification", fb, balance_weights=True)
    # accuracy with "balanced" sample weights == sklearn's balanced accuracy
    from sklearn.utils import class_weight

    ref = []
    for tr, va in fb:
        m = xgb.XGBClassifier(**SMALL_PARAMS)
        m.fit(X[tr].astype(np.float32), y[tr],
              sample_weight=class_weight.compute_sample_weight("balanced", y=y[tr]))
        ref.append(balanced_accuracy_score(y[va], m.predict(X[va].astype(np.float32))))
    assert got == pytest.approx(float(np.mean(ref)), abs=1e-9)

    yr = X[:, 0] * 2 + 0.1 * np.arange(160) % 1
    sel = ss.build_selection_splits(yr, "regression", 4, 42)
    fb = ss.build_feedback_splits(yr, sel, "regression", 4, 42)
    got = ss.l1_inner_cv_score(X, yr, SMALL_PARAMS, "regression", fb)
    ref = []
    for tr, va in fb:
        m = xgb.XGBRegressor(**SMALL_PARAMS)
        m.fit(X[tr].astype(np.float32), yr[tr])
        ref.append(math.sqrt(mean_squared_error(yr[va], m.predict(X[va].astype(np.float32)))))
    assert got == pytest.approx(float(np.mean(ref)), rel=1e-6)


def test_compute_l1_for_tester_audit_and_no_instrumentation_leak():
    ss.reset_audit()
    instrumentation.pop_inner_cv_scores()
    instrumentation.note_inner_cv([0.5, 0.6])
    X, y = _clf_data(100)
    out = ss.compute_l1_for_tester(
        X_train=X, y_train=y, base_params=SMALL_PARAMS, task_type="classification",
        seed=42, balance_weights=True, train_ids=list(range(100)), test_ids=[1000, 1001],
    )
    assert out["l1_status"] == "ok" and 0.0 <= out["l1_cv_accuracy"] <= 1.0
    assert out["l1_split_disjoint"] == 1
    # the selection signal's per-fold scores are untouched by the L1 computation
    assert instrumentation.pop_inner_cv_scores() == [0.5, 0.6]
    summ = ss.audit_summary("inner_cv_l1")
    assert summ == {
        "scientist_signal": "inner_cv_l1", "l1_split_checks": 1, "l1_split_failures": 0,
        "l1_split_disjoint": 1, "l1_ids_checked": 1,
        "l1_max_val_overlap": pytest.approx(out["l1_max_val_overlap"]),
    }
    assert ss.audit_summary("train_fit") == {"scientist_signal": "train_fit"}
    # too few rows -> no score, no check
    out = ss.compute_l1_for_tester(
        X_train=X[:5], y_train=y[:5], base_params=SMALL_PARAMS, task_type="classification",
        seed=42, balance_weights=True, train_ids=None, test_ids=None,
    )
    assert out == {"l1_cv_accuracy": None, "l1_status": "not_viable"}
    with pytest.raises(ss.L1SplitLeakageError):
        ss.compute_l1_for_tester(
            X_train=X, y_train=y, base_params=SMALL_PARAMS, task_type="classification",
            seed=42, balance_weights=True, train_ids=list(range(100)), test_ids=[3],
        )
    assert ss.audit_summary("inner_cv_l1")["l1_split_disjoint"] == 0
    ss.reset_audit()


# --------------------------------------------------------------------------- Tester
def _run_tester(tmp_path, monkeypatch, task_type, n_per_fold=30):
    import rogueone.llm.agents.core.tester.agent as tst_mod

    cfg = SimpleNamespace(
        output_dir=Path(tmp_path), experiment_description="d", context_and_goal="g",
        experiment_name="e", task_type=task_type, modality="tabular",
    )
    agent = tst_mod.TesterAgent(cfg, include_report=False, enable_wandb=False)

    class FakeRunner:
        @staticmethod
        async def run(a, *args, **kwargs):
            return SimpleNamespace(final_output="no pruning")

    monkeypatch.setattr(tst_mod, "Runner", FakeRunner)
    feats = ["f0", "f1", "f2"]
    expl = pd.DataFrame({"Attribute": feats, "Status": ["Active"] * 3,
                         "Pruned Iteration": [pd.NA] * 3, "Prune Rationale": [pd.NA] * 3})
    rng = np.random.default_rng(0)
    folds, nid = [], 0
    for _ in range(rouge_one_cfg.k_folds):
        d = pd.DataFrame(rng.normal(size=(n_per_fold, 3)), columns=feats)
        d["target"] = (
            (d["f0"] > 0).astype(int) if task_type == "classification" else d["f0"] * 3
        )
        d["id"] = range(nid, nid + n_per_fold)
        nid += n_per_fold
        folds.append(d)
    instrumentation.pop_inner_cv_scores()
    res, _ = asyncio.run(agent.test_hypotheses(folds, expl, step=1, trial_index=0))
    return res, instrumentation.pop_inner_cv_scores()


_PRE_E14_KEYS = {
    "classification": {
        "fold_id", "attempt", "status", "inner_cv_accuracy", "train_accuracy",
        "train_precision", "train_recall", "train_f1", "train_AUROC", "accuracy",
        "precision", "recall", "f1", "AUROC",
    },
    "regression": {
        "fold_id", "attempt", "status", "inner_cv_RMSE", "train_MSE", "train_MAE",
        "train_R2", "MSE", "MAE", "R2",
    },
}


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tester_train_fit_unchanged_and_l1_additive(tmp_path, monkeypatch, signal, task):
    ss.reset_audit()
    signal("train_fit")
    base, base_inner = _run_tester(tmp_path / "a", monkeypatch, task)
    assert set(base.fold_metrics[0]) == _PRE_E14_KEYS[task]
    assert ss.audit_summary("inner_cv_l1")["l1_split_checks"] == 0  # nothing ran

    signal("inner_cv_l1")
    l1, l1_inner = _run_tester(tmp_path / "b", monkeypatch, task)
    key = "l1_cv_accuracy" if task == "classification" else "l1_cv_RMSE"
    row = l1.fold_metrics[0]
    assert row[key] is not None and row["l1_split_disjoint"] == 1 and row["l1_ids_checked"] == 1
    # everything else (held-out metrics, train fit, selection signal) is identical
    assert {k: v for k, v in row.items() if k in _PRE_E14_KEYS[task]} == base.fold_metrics[0]
    assert json.dumps(l1.as_dict, default=str) == json.dumps(base.as_dict, default=str)
    assert l1_inner == base_inner
    summ = ss.audit_summary("inner_cv_l1")
    assert summ["l1_split_disjoint"] == 1 and summ["l1_split_checks"] == 1
    ss.reset_audit()


# --------------------------------------------------------------------------- feedback pool
def _pre_e14_feedback_row(test_results_obj, report, trial_id):
    """Verbatim copy of RogueOneAgentNetwork._build_scientist_feedback_row @52f526d."""
    fold_metrics = getattr(test_results_obj, "fold_metrics", None) or []
    successful = [m for m in fold_metrics if m.get("status") == "success"]

    def _mean_metric(key):
        values = [float(m[key]) for m in successful if key in m and m[key] is not None]
        if len(values) == 0:
            return None
        return float(sum(values) / len(values))

    return {
        "id": trial_id,
        "status": getattr(test_results_obj, "status", "completed"),
        "dataset_id": getattr(test_results_obj, "dataset_id", ""),
        "fold_seed": getattr(test_results_obj, "fold_seed", None),
        "aborted_fold_id": getattr(test_results_obj, "aborted_fold_id", None),
        "skip_reason": getattr(test_results_obj, "skip_reason", None),
        "feature_assessment": report,
        "train_accuracy": _mean_metric("train_accuracy"),
        "train_precision": _mean_metric("train_precision"),
        "train_recall": _mean_metric("train_recall"),
        "train_f1": _mean_metric("train_f1"),
        "train_AUROC": _mean_metric("train_AUROC"),
        "train_MSE": _mean_metric("train_MSE"),
        "train_MAE": _mean_metric("train_MAE"),
        "train_R2": _mean_metric("train_R2"),
    }


def _fake_results():
    ok = {"status": "success", "train_accuracy": 0.91, "train_precision": 0.9,
          "train_recall": 0.92, "train_f1": 0.905, "train_AUROC": 0.97,
          "l1_cv_accuracy": 0.81}
    reg = {"status": "success", "train_MSE": 1.5, "train_MAE": 0.9, "train_R2": 0.8,
           "l1_cv_RMSE": 1.7}
    mk = lambda fm, st="completed": SimpleNamespace(
        fold_metrics=fm, status=st, dataset_id="car", fold_seed=42,
        aborted_fold_id=None, skip_reason=None)
    return [mk([ok]), mk([reg]), mk([{"status": "failed", "error": "x"}, ok]), mk([]),
            mk([{k: v for k, v in ok.items() if not k.startswith("l1_")}])]


def test_feedback_pool_byte_identical_under_train_fit(signal):
    from rogue_one_main import RogueOneAgentNetwork

    signal("train_fit")
    build = RogueOneAgentNetwork._build_scientist_feedback_row
    new = pd.DataFrame([build(r, f"rep {i}", f"Trial {i}") for i, r in enumerate(_fake_results())])
    old = pd.DataFrame(
        [_pre_e14_feedback_row(r, f"rep {i}", f"Trial {i}") for i, r in enumerate(_fake_results())]
    )
    assert new.to_csv(index=False).encode() == old.to_csv(index=False).encode()


def test_feedback_pool_under_l1(signal):
    from rogue_one_main import RogueOneAgentNetwork

    signal("inner_cv_l1")
    rows = [RogueOneAgentNetwork._build_scientist_feedback_row(r, "rep", "Trial 0")
            for r in _fake_results()]
    for row in rows:
        assert not any(k.startswith("train_") for k in row)
        assert list(row)[-2:] == ["cv_accuracy", "cv_RMSE"]
    assert rows[0]["cv_accuracy"] == 0.81 and rows[0]["cv_RMSE"] is None
    assert rows[1]["cv_RMSE"] == 1.7 and rows[1]["cv_accuracy"] is None
    assert rows[2]["cv_accuracy"] == 0.81  # failed attempt ignored
    assert rows[3]["cv_accuracy"] is None and rows[4]["cv_accuracy"] is None


def test_report_scientist_signal_summary(tmp_path, signal):
    from unittest.mock import MagicMock

    from rogue_one_main import RogueOneAgentNetwork

    stub = MagicMock(spec=RogueOneAgentNetwork)
    stub.cfg = SimpleNamespace(output_dir=tmp_path)
    stub.enable_wandb = False
    signal("train_fit")
    assert RogueOneAgentNetwork._report_scientist_signal(stub) == {"scientist_signal": "train_fit"}
    assert not (tmp_path / "scientist_signal_summary.json").exists()
    signal("inner_cv_l1")
    ss.reset_audit()
    ss.note_check({"l1_ids_checked": 1, "l1_max_val_overlap": 0.26})
    out = RogueOneAgentNetwork._report_scientist_signal(stub)
    assert out["l1_split_disjoint"] == 1 and out["scientist_signal"] == "inner_cv_l1"
    on_disk = json.loads((tmp_path / "scientist_signal_summary.json").read_text())
    assert on_disk == out
    ss.reset_audit()


# --------------------------------------------------------------------------- Scientist prompt
def _pandas_tool_from_source():
    """Build a FunctionTool from the real generic_pandas_tool docstring/signature."""

    from agents import function_tool

    src = (ROOT / "rogueone" / "llm" / "agents" / "core" / "scientist" / "agent.py").read_text(
        encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "generic_pandas_tool")

    async def generic_pandas_tool(command: str) -> Any:
        return None

    generic_pandas_tool.__doc__ = ast.get_docstring(node, clean=False)
    return function_tool(generic_pandas_tool)


def test_scientist_tool_description_train_fit_identical_l1_rewritten():
    from rogueone.llm.agents.core.scientist.agent import apply_scientist_signal_to_pandas_tool

    tool = _pandas_tool_from_source()
    desc, schema = tool.description, json.dumps(tool.params_json_schema, sort_keys=True)
    assert ss.TRAIN_FIT_COLUMNS_LINE in desc and ss.TRAIN_FIT_EXAMPLE in desc
    same = apply_scientist_signal_to_pandas_tool(tool, "train_fit")
    assert same is tool and same.description == desc

    l1 = apply_scientist_signal_to_pandas_tool(_pandas_tool_from_source(), "inner_cv_l1")
    assert ss.L1_COLUMNS_LINE in l1.description and ss.L1_EXAMPLE in l1.description
    assert "train_accuracy" not in l1.description and "`train_" not in l1.description
    assert json.dumps(l1.params_json_schema, sort_keys=True) == schema
    assert l1.description == desc.replace(ss.TRAIN_FIT_COLUMNS_LINE, ss.L1_COLUMNS_LINE).replace(
        ss.TRAIN_FIT_EXAMPLE, ss.L1_EXAMPLE)


# --------------------------------------------------------------------------- cost
def test_l1_cost_roughly_one_selection_inner_cv():
    """Rough cost: the L1 scalar adds 4 XGBoost fits, like the selection inner CV."""
    from rogueone.llm.agents.core.tester.agent import _inner_cv_score

    params = dict(SMALL_PARAMS, n_estimators=400, max_depth=6, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9, n_jobs=2)
    X, y = _clf_data(800)
    t0 = time.perf_counter()
    _inner_cv_score(X, y, params, "classification", seed=42, balance_weights=True)
    t_sel = time.perf_counter() - t0
    t0 = time.perf_counter()
    ss.compute_l1_for_tester(X_train=X, y_train=y, base_params=params,
                             task_type="classification", seed=42, balance_weights=True,
                             train_ids=list(range(800)), test_ids=[])
    t_l1 = time.perf_counter() - t0
    print(f"[E14 cost] n=800 p=4 400 trees: selection inner CV {t_sel:.2f}s, L1 {t_l1:.2f}s")
    assert t_l1 < 5 * t_sel + 2.0
    ss.reset_audit()
