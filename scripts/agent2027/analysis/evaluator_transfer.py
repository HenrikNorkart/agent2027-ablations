"""A22 (M3): evaluator transfer -- CPU re-scoring of the inner-CV-selected feature sets.

Spec: ``docs/ablations/A22_M3_evaluator_transfer.md`` (section 4 design, section 6 rules).
Status in ``docs/agent2027/preregistration.md`` Part V.22: A22 was moved to the short paper
by D1 (ED5, 2026-10-09); its section 4 / section 6 rules "remain frozen by reference for
that paper". This script implements them as written; it is not part of the AGENT 2027
analysis. CPU only; never touches a run folder (inputs are read-only).

Subcommands (A22 "new names")::

  folds    A22-T7: Rogue One's own outer partition -> folds_<ds>_f<seed>.json
  fetch    A22-T10: thin wrapper around fetch_wandb.py (W&B run-output of group A05)
  score    A22-T10/T11: parity replay + 6 feature sets x 3 evaluators -> et_long.csv,
           et_episodes.csv, et_parity.csv; ``--parity-only`` stops after parity.
           Prints ``=== A22_SCORE_DONE rc=<rc> ===`` at the end.
  readout  A22-T13/T14 + section 6: parity gate, unit means, bootstrap medians, rec_sel split,
           ceiling share, |gain|/sigma_d, sensitivities, the T/X/B/N label -> et_readout.json
  log      A22-T12 step 5: writes ANALYSIS__evaluator_transfer__<YYYYMMDD> to W&B.

Inputs of ``score`` (CAIR layout; all mounted read-only)::

  --runs <session roots>   <root>/<row_id>/tasks/<type>/<ds>/output/ + <root>/status.tsv
  --manifest A05_M0.csv    row metadata (dataset, task type, fold seed, replicate, env)
  --tasks <tasks root>     <tasks>/<type>/<ds>/{data.csv,labels.csv}
  --meta <synp meta>       phi metadata (raw_phi), off-mount, analysis containers only
  --ofe <dir> [<dir> ...]  OpenFE dumps (<dir>/f<seed>/features/<ds>_openfe_fold<k>[_test].csv.gz);
                           earlier dirs take precedence (T12: /ofe_core before /ofe_synp)
  --episodes <csv>         per-episode rec_sel (row_id, fold, rec_sel), the a05-handoff table

Per-episode reconstruction (section 2): the selected trial (``fold_aggregation_best_iterations.csv``)
-> its E3 snapshot ``intermediate_steps/df_attributes_explanations_trail_<index>.csv`` -> fold-safe
replay of every row of the snapshot (``build_fold_safe_entities_folds``, the run's own function)
-> Tester matrices in Tester row and column order, filtered to the actually scored set.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
CORE10 = {  # A05 section 2 / R10 order
    "balance-scale": "classification", "junglechess": "classification", "vehicle": "classification",
    "heart": "classification", "credit-g": "classification", "myocardial": "classification",
    "housing": "regression", "insurance": "regression", "crab": "regression", "wine": "regression",
}
HEADROOM = ["balance-scale", "junglechess", "vehicle", "housing", "crab", "wine"]   # A22 section 2: h* >= 1
NO_HARM = ["heart", "credit-g", "myocardial", "insurance"]                          # h* < 1
SYNP8 = ["synp_p1", "synp_p2", "synp_p3", "synp_p4", "synp_p5", "synp_p6a", "synp_p6b", "synp_p7"]
SYNP_CLASSIFICATION = {"synp_p3", "synp_p4", "synp_p5", "synp_p7"}
P0 = "synp_p0"
NON_DISCRIMINATING = ("synp_p1",)  # prereg Part III.2 (M5S): P1 only
EVALUATORS = ("xgb400", "linear", "rf")
FEATURE_SETS = ("raw", "ro", "raw_ro", "raw_ofe_k", "raw_ofe_all", "raw_phi")
SENSITIVITY_SETS = ("raw_ro_nofb",)          # fallback features dropped (section 2 "fallback-excluded")
SENSITIVITY_EVALUATORS = ("linear_nowins",)  # linear without winsorisation (section 3 / T11)
K_FOLDS = 5
PARITY_C_ABS = 0.002     # section 4 parity gate, classification |d bacc| and |d AUROC|
PARITY_R_REL = 0.005     # regression |d RMSE| / RMSE
PARITY_GATE = 0.90       # >= 90% of re-scorable fold-episodes
MIN_PASS_PER_RUN = 3     # < 3/5 passing episodes -> dataset excluded from that run's row
MIN_S_EPISODES = 3       # S = discriminating SYN-P tasks with >= 3 passing rec_sel=1 episodes
H_MIN_USABLE = 4         # fewer than 4 of 6 H usable -> "not evaluable"
BOOT_N, BOOT_SEED, BOOT_LEVEL = 2000, 2027, 0.90   # T14
LINEAR_CS = np.logspace(-3, 3, 7)
RIDGE_ALPHAS = np.logspace(-3, 3, 7)
WINSOR_Q = (0.5, 99.5)
RF_TREES, RF_JOBS = 500, 4
DEFAULT_XGB_N_JOBS = 8   # metrics.py: min(ROGUE_ONE_N_JOBS | OMP | OPENBLAS, 8), 8 when unset
SCORE_DONE = "=== A22_SCORE_DONE rc={rc} ==="

_RUNTIME_ENV = {"EMBEDD_API_KEY": "x", "EMBEDD_MODEL": "x", "EMBEDD_PORT": "1", "LLM_MODEL": "x",
                "LLM_API_KEY": "x", "LLM_PORT": "1"}


def _default_repo() -> str:
    parents = Path(__file__).resolve().parents
    return str(parents[3]) if len(parents) > 3 else str(parents[0])


def _ensure_rogueone_importable(repo: Optional[str] = None) -> None:
    """``rogueone.utils`` builds config objects at import time; give it dummy LLM vars."""
    for k, v in _RUNTIME_ENV.items():
        os.environ.setdefault(k, v)
    root = repo or _default_repo()
    if root not in sys.path:
        sys.path.insert(0, root)


def task_kind(dataset_id: str) -> str:
    """'core10' | 'synp' | 'p0' | 'other'."""
    if dataset_id in CORE10:
        return "core10"
    if dataset_id == P0:
        return "p0"
    if dataset_id.startswith("synp_"):
        return "synp"
    return "other"


# =========================================================================== evaluators (pure)
def xgb400_params(task_type: str, fold_seed: int, n_jobs: int = DEFAULT_XGB_N_JOBS) -> dict[str, Any]:
    """The Tester's ``base_params``, copied verbatim (tst/agent.py get_regression_metrics /
    get_classification_metrics_tabular), with ``random_state = fold_seed``."""
    if task_type == "regression":
        return {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
                "random_state": fold_seed, "device": "cpu", "n_estimators": 400, "learning_rate": 0.05,
                "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9, "n_jobs": n_jobs}
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "random_state": fold_seed, "device": "cpu", "n_estimators": 400, "learning_rate": 0.05,
            "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9, "n_jobs": n_jobs}


def tester_matrix(X) -> np.ndarray:
    """The Tester's ``np.nan_to_num(float32, nan=0, posinf=1e6, neginf=-1e6)``."""
    return np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def encode_labels(y_train, y_test) -> tuple[np.ndarray, np.ndarray]:
    """LabelEncoder fitted on train + test labels (tst/agent.py)."""
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415
    le = LabelEncoder().fit(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    return (np.asarray(le.transform(y_train)).ravel().astype(int),
            np.asarray(le.transform(y_test)).ravel().astype(int))


def balanced_accuracy_tester(y_true_enc, y_pred_enc) -> float:
    """``accuracy_score`` with balanced test sample weights (the Tester's logged ``accuracy``)."""
    from sklearn.metrics import accuracy_score  # noqa: PLC0415
    from sklearn.utils import class_weight  # noqa: PLC0415
    w = class_weight.compute_sample_weight("balanced", y=y_true_enc)
    return float(accuracy_score(y_true_enc, y_pred_enc, sample_weight=w))


def auroc_tester(y_train_enc, y_test_enc, proba, model_classes) -> float:
    """The Tester's AUROC: binary (positive = encoded 1), else one-vs-rest weighted; NaN on error."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from sklearn.preprocessing import label_binarize  # noqa: PLC0415
    proba = np.asarray(proba)
    classes_enc = np.unique(np.concatenate([y_train_enc, y_test_enc]))
    try:
        if proba.ndim == 2 and proba.shape[1] == 2:
            try:
                pos = list(model_classes).index(1)
            except Exception:  # noqa: BLE001
                pos = 1
            p = proba[:, pos]
        else:
            p = proba
        if len(classes_enc) == 2:
            return float(roc_auc_score(y_test_enc, p))
        return float(roc_auc_score(label_binarize(y_test_enc, classes=classes_enc), proba,
                                   multi_class="ovr", average="weighted"))
    except Exception:  # noqa: BLE001
        return float("nan")


def regression_scores(y_train, y_test, pred) -> dict[str, float]:
    """RMSE, nRMSE = RMSE / training-portion target mean (tst/agent.py), R2."""
    from sklearn.metrics import mean_squared_error, r2_score  # noqa: PLC0415
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mean_train = float(np.mean(np.asarray(y_train, dtype=float)))
    return {"rmse": rmse, "nrmse": rmse / mean_train if mean_train != 0 else float("nan"),
            "r2": float(r2_score(y_test, pred))}


def linear_preprocess(X_train, X_test, *, winsorise: bool = True) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Section 4 ``linear`` preprocessing, all statistics from the training fold:
    +-inf -> NaN; NaN -> training median; winsorise at the 0.5/99.5 training percentiles;
    drop columns constant on train; StandardScaler fit on train. Returns the kept column
    indices. An all-NaN training column has no median and is constant -> dropped."""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    tr = np.asarray(X_train, dtype=np.float64).copy()
    te = np.asarray(X_test, dtype=np.float64).copy()
    if tr.ndim == 1:
        tr, te = tr[:, None], te[:, None]
    tr[~np.isfinite(tr)] = np.nan
    te[~np.isfinite(te)] = np.nan
    with np.errstate(all="ignore"):
        import warnings  # noqa: PLC0415
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(tr, axis=0) if tr.shape[1] else np.zeros(0)
    med = np.where(np.isfinite(med), med, 0.0)
    for j in range(tr.shape[1]):
        tr[np.isnan(tr[:, j]), j] = med[j]
        te[np.isnan(te[:, j]), j] = med[j]
    if winsorise and tr.shape[1]:
        lo, hi = np.percentile(tr, WINSOR_Q[0], axis=0), np.percentile(tr, WINSOR_Q[1], axis=0)
        tr, te = np.clip(tr, lo, hi), np.clip(te, lo, hi)
    keep = [j for j in range(tr.shape[1]) if np.nanmax(tr[:, j]) > np.nanmin(tr[:, j])] if len(tr) else []
    tr, te = tr[:, keep], te[:, keep]
    if tr.shape[1]:
        sc = StandardScaler().fit(tr)
        tr, te = sc.transform(tr), sc.transform(te)
    return tr, te, keep


def fit_score(evaluator: str, X_train, y_train, X_test, y_test, task_type: str, fold_seed: int,
              xgb_n_jobs: int = DEFAULT_XGB_N_JOBS) -> dict[str, Any]:
    """Fit one pre-registered evaluator on one fold-episode and score the test rows.

    Classification -> ``bacc`` (balanced-weight accuracy), ``auroc``; regression -> ``rmse``,
    ``nrmse``, ``r2``. ``linear_nowins`` is the no-winsorisation sensitivity variant."""
    from sklearn.utils import class_weight  # noqa: PLC0415
    out: dict[str, Any] = {"evaluator": evaluator, "n_features": int(np.asarray(X_train).shape[1])}
    is_c = task_type == "classification"
    if is_c:
        ytr, yte = encode_labels(y_train, y_test)
    else:
        ytr, yte = np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)
    if evaluator == "xgb400":
        import xgboost as xgb  # noqa: PLC0415
        Xtr, Xte = tester_matrix(X_train), tester_matrix(X_test)
        p = xgb400_params(task_type, fold_seed, xgb_n_jobs)
        if is_c:
            m = xgb.XGBClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=class_weight.compute_sample_weight("balanced", y=ytr))
        else:
            m = xgb.XGBRegressor(**p)
            m.fit(Xtr, ytr)
    elif evaluator in ("linear", "linear_nowins"):
        from sklearn.linear_model import LogisticRegressionCV, RidgeCV  # noqa: PLC0415
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
        Xtr, Xte, keep = linear_preprocess(X_train, X_test, winsorise=(evaluator == "linear"))
        out["n_features_used"] = len(keep)
        if Xtr.shape[1] == 0:  # nothing left: an intercept-only model is the honest fallback
            Xtr, Xte = np.zeros((len(ytr), 1)), np.zeros((len(yte), 1))
        if is_c:
            m = LogisticRegressionCV(Cs=LINEAR_CS, cv=StratifiedKFold(4, shuffle=True, random_state=fold_seed),
                                     penalty="l2", class_weight="balanced", scoring="balanced_accuracy",
                                     max_iter=5000)
        else:
            m = RidgeCV(alphas=RIDGE_ALPHAS)
        m.fit(Xtr, ytr)
    elif evaluator == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: PLC0415
        Xtr, Xte = tester_matrix(X_train), tester_matrix(X_test)
        if is_c:
            m = RandomForestClassifier(n_estimators=RF_TREES, random_state=fold_seed, n_jobs=RF_JOBS,
                                       class_weight="balanced")
        else:
            m = RandomForestRegressor(n_estimators=RF_TREES, random_state=fold_seed, n_jobs=RF_JOBS)
        m.fit(Xtr, ytr)
    else:
        raise ValueError(f"unknown evaluator {evaluator!r}")
    if is_c:
        pred = np.asarray(m.predict(Xte)).astype(int)
        out["bacc"] = balanced_accuracy_tester(yte, pred)
        out["auroc"] = auroc_tester(ytr, yte, m.predict_proba(Xte), getattr(m, "classes_", []))
    else:
        out.update(regression_scores(ytr, yte, np.asarray(m.predict(Xte), dtype=float)))
    return out


# =========================================================================== feature blocks (pure)
def is_categorical(series: pd.Series) -> bool:
    """``prep``'s object test, made dtype-safe (pandas 3 reads strings as ``str``; the E6
    ``is_categorical_raw_column`` rule)."""
    return not pd.api.types.is_numeric_dtype(series)


def prep_raw(data: pd.DataFrame) -> pd.DataFrame:
    """``classicfe_arm.prep`` on an id-carrying frame: object columns ordinal-encoded
    alphabetically over the full column domain; returns ``id`` + numeric columns."""
    X = data.drop(columns=["id"]).copy()
    for col in X.columns:
        if is_categorical(X[col]):
            cats = sorted(X[col].dropna().astype(str).unique().tolist())
            X[col] = X[col].astype(str).map({v: i for i, v in enumerate(cats)})
    X = X.apply(pd.to_numeric, errors="coerce")
    X.insert(0, "id", data["id"].to_numpy())
    return X


def raw_block(data: pd.DataFrame, train_ids: Sequence, test_ids: Sequence, evaluator: str
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw block ``B`` for one fold-episode, rows in the given id orders.

    Trees (xgb400, rf): ``prep`` ordinal codes. Linear: ``prep``'s numeric columns plus a
    one-hot of the original categorical columns, levels from the training fold (unseen -> 0)."""
    pr = prep_raw(data).set_index("id")
    if not evaluator.startswith("linear"):
        return pr.loc[list(train_ids)].reset_index(drop=True), pr.loc[list(test_ids)].reset_index(drop=True)
    cat_cols = [c for c in data.columns if c != "id" and is_categorical(data[c])]
    num = pr.drop(columns=cat_cols)
    tr, te = num.loc[list(train_ids)].reset_index(drop=True), num.loc[list(test_ids)].reset_index(drop=True)
    raw = data.set_index("id")
    for c in cat_cols:
        s_tr = raw.loc[list(train_ids), c].astype(str).where(raw.loc[list(train_ids), c].notna())
        s_te = raw.loc[list(test_ids), c].astype(str).where(raw.loc[list(test_ids), c].notna())
        for lv in sorted(s_tr.dropna().unique().tolist()):
            tr[f"{c}=={lv}"] = (s_tr.to_numpy() == lv).astype(float)
            te[f"{c}=={lv}"] = (s_te.to_numpy() == lv).astype(float)
    return tr, te


def is_live(train_col) -> bool:
    """A feature is live iff its replayed training column is neither all-NaN nor constant."""
    a = pd.to_numeric(pd.Series(np.asarray(train_col).ravel()), errors="coerce").to_numpy(dtype=float)
    a = a[~np.isnan(a)]
    return bool(len(a) and len(np.unique(a)) > 1)


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "1.0", "yes"})


def scored_names(snapshot: pd.DataFrame, mask_col: str = "Scored_post_prune") -> list[str]:
    """Attribute names the Tester scored, in attribute-table (creation) order.

    v2 scores the post-pruning set (E2/P4 ``_rebuild_after_pruning``), recorded as
    ``Scored_post_prune``; if it is empty the Tester kept the pre-pruning matrices
    (``Scored``)."""
    names = snapshot.loc[_flag(snapshot, mask_col), "Attribute"].astype(str).tolist()
    if not names and mask_col != "Scored":
        names = snapshot.loc[_flag(snapshot, "Scored"), "Attribute"].astype(str).tolist()
    return names


def tester_matrices(folds: Sequence[pd.DataFrame], test_fold: int, names: Sequence[str]) -> dict[str, Any]:
    """Tester row and column order: training rows = the non-test folds stacked in fold order
    (each in ``data.csv`` row order); columns = attribute order restricted to ``names``."""
    feat = [c for c in folds[0].columns if c not in ("id", "target")]
    keep = set(names)
    cols = [c for c in feat if c in keep]
    tr = [f for i, f in enumerate(folds) if i != test_fold]
    te = folds[test_fold]
    return {"cols": cols,
            "X_train": pd.concat([f[cols] for f in tr], ignore_index=True),
            "y_train": np.concatenate([f["target"].to_numpy() for f in tr]),
            "ids_train": np.concatenate([f["id"].to_numpy() for f in tr]),
            "X_test": te[cols].reset_index(drop=True), "y_test": te["target"].to_numpy(),
            "ids_test": te["id"].to_numpy()}


def ro_extra_columns(snapshot: pd.DataFrame, X_train: pd.DataFrame, cols: Sequence[str], *,
                     drop_fallback: bool = False) -> dict[str, list[str]]:
    """``raw_ro``'s extra features: scored, non-seeded, live (fallback kept unless
    ``drop_fallback``). Returns ``live`` and ``dead`` name lists in matrix order."""
    names = snapshot["Attribute"].astype(str)
    seeded = set(names[_flag(snapshot, "Seeded")])
    fb = set(names[_flag(snapshot, "Fallback")])
    live, dead = [], []
    for c in cols:
        if c in seeded or (drop_fallback and c in fb):
            continue
        (live if is_live(X_train[c]) else dead).append(c)
    return {"live": live, "dead": dead}


# =========================================================================== gains and parity (pure)
def primary_score(rec: Mapping[str, Any], task_type: str, kind: str) -> float:
    """Per-episode score, positive = better: C bacc; Core-10 R -nRMSE (gains use the relative
    form); SYN-P / P0 R R2 (section 4: Delta R2 primary on SYN-P R)."""
    if task_type == "classification":
        return float(rec.get("bacc", float("nan")))
    if kind == "core10":
        return -float(rec.get("nrmse", float("nan")))
    return float(rec.get("r2", float("nan")))


def gain(set_rec: Mapping[str, Any], raw_rec: Mapping[str, Any], task_type: str, kind: str) -> float:
    """``et_gain`` of one feature set over ``raw``: bacc difference (C); relative nRMSE
    reduction 1 - nRMSE(set)/nRMSE(raw) (Core-10 R); R2 difference (SYN-P / P0 R)."""
    if task_type == "regression" and kind == "core10":
        a, b = float(set_rec.get("nrmse", np.nan)), float(raw_rec.get("nrmse", np.nan))
        return 1.0 - a / b if b and np.isfinite(a) and np.isfinite(b) else float("nan")
    return primary_score(set_rec, task_type, kind) - primary_score(raw_rec, task_type, kind)


def parity_check(task_type: str, replay: Mapping[str, Any], logged: Mapping[str, Any]) -> dict[str, Any]:
    """Section 4 parity: C |d bacc| <= 0.002 and |d AUROC| <= 0.002; R |d RMSE|/RMSE <= 0.5%.
    A NaN logged metric -> ``rescorable = 0`` (listed, not a parity failure)."""
    if task_type == "classification":
        la, lu = _num(logged.get("accuracy")), _num(logged.get("AUROC"))
        if not np.isfinite(la):
            return {"rescorable": 0, "et_parity_ok": None, "et_parity_delta": None}
        da = abs(_num(replay.get("bacc")) - la)
        du = abs(_num(replay.get("auroc")) - lu) if np.isfinite(lu) else 0.0
        ok = bool(np.isfinite(da) and da <= PARITY_C_ABS and np.isfinite(du) and du <= PARITY_C_ABS)
        return {"rescorable": 1, "et_parity_ok": int(ok), "et_parity_delta": float(da),
                "et_parity_delta_auroc": float(du)}
    lr = _num(logged.get("RMSE"))
    if not np.isfinite(lr) or lr == 0:
        return {"rescorable": 0, "et_parity_ok": None, "et_parity_delta": None}
    d = abs(_num(replay.get("rmse")) - lr) / lr
    return {"rescorable": 1, "et_parity_ok": int(bool(np.isfinite(d) and d <= PARITY_R_REL)),
            "et_parity_delta": float(d)}


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


# =========================================================================== folds (T7)
def fold_test_ids(data: pd.DataFrame, labels: pd.DataFrame, fold_seed: int, *, stratified: bool,
                  task_type: str, k: int = K_FOLDS) -> list[list]:
    """Outer test-id lists exactly as ``rogue_one_main.py`` builds them: entity ids in
    first-appearance order, ``compute_entity_duplicate_groups`` on the raw table and
    ``deterministic_kfold_indices``; stratified only for classification with the flag set."""
    from rogueone.utils.train_test_splitter import (  # noqa: PLC0415
        compute_entity_duplicate_groups, deterministic_kfold_indices)
    ids = pd.Series(data["id"].unique())
    strat = None
    if stratified and task_type == "classification":
        lab = labels.rename(columns={"patient_id": "id"}) if "id" not in labels.columns else labels
        ordered = pd.DataFrame({"id": ids}).merge(lab[["id", "target"]].drop_duplicates("id"), on="id", how="left")
        if not ordered["target"].isna().any():
            strat = ordered["target"].to_numpy()
    groups = compute_entity_duplicate_groups(data, ids.tolist(), exclude_cols=("id",))
    idx = deterministic_kfold_indices(n_rows=len(ids), n_splits=k, random_state=fold_seed,
                                      stratify_labels=strat, dedup_groups=groups)
    return [[ids.iloc[i] for i in te] for _, te in idx]


def plain_kfold_test_ids(data: pd.DataFrame, labels: pd.DataFrame, task_type: str, k: int = K_FOLDS,
                         seed: int = 42, stratified: bool = False) -> list[list]:
    """``classicfe_arm.build_folds``: plain (Stratified)KFold, random_state 42, no groups."""
    from sklearn.model_selection import KFold, StratifiedKFold  # noqa: PLC0415
    ids = data["id"].to_numpy()
    if stratified and task_type == "classification":
        y = data[["id"]].merge(labels[["id", "target"]], on="id", how="left")["target"].to_numpy()
        sp = StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, y)
    else:
        sp = KFold(k, shuffle=True, random_state=seed).split(ids)
    return [ids[te].tolist() for _, te in sp]


def cmd_folds(a: argparse.Namespace) -> int:
    _ensure_rogueone_importable(a.repo)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds_list = list(CORE10) if a.datasets == ["core10"] else SYNP8 if a.datasets == ["synp8"] else a.datasets
    diff = []
    for ds in ds_list:
        if "covtype" in ds:
            raise SystemExit("covtype is never run")
        ttype = CORE10.get(ds) or ("classification" if ds in SYNP_CLASSIFICATION else "regression")
        tdir = Path(a.tasks) / ttype / ds
        data, labels = pd.read_csv(tdir / "data.csv"), pd.read_csv(tdir / "labels.csv")
        assert data["id"].is_unique, f"{ds}: data.csv ids are not unique"
        strat = bool(int(a.stratified_classification)) and ds not in CORE10
        for s in a.seeds:
            folds = fold_test_ids(data, labels, int(s), stratified=strat, task_type=ttype)
            rec = {"dataset_id": ds, "task_type": ttype, "fold_seed": int(s),
                   "stratified": bool(strat and ttype == "classification"),
                   "test_ids": [[_jsonable(i) for i in f] for f in folds]}
            (out / f"folds_{ds}_f{s}.json").write_text(json.dumps(rec), encoding="utf-8")
            plain = plain_kfold_test_ids(data, labels, ttype, stratified=False)
            same = int(s) == 42 and [sorted(map(str, f)) for f in folds] == [sorted(map(str, f)) for f in plain]
            diff.append({"dataset_id": ds, "fold_seed": int(s), "equals_classicfe_kfold42": int(same)})
    pd.DataFrame(diff).to_csv(out / "folds_vs_classicfe.csv", index=False)
    print(pd.DataFrame(diff).to_string(index=False))
    return 0


def _jsonable(v):
    return v.item() if hasattr(v, "item") else v


# =========================================================================== run discovery
def read_manifest(paths: Sequence[str]) -> pd.DataFrame:
    man = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    man = man[man["notes"].fillna("") != "smoke"]
    return man[~man["dataset_id"].str.contains("covtype", na=False)]


def final_status(root: Path, row_id: str) -> str:
    st = root / "status.tsv"
    last = "MISSING"
    if st.exists():
        for r in csv.DictReader(st.open(encoding="utf-8"), delimiter="\t"):
            if r.get("row_id") == row_id:
                last = r.get("status", "?")
    return last


def locate_row(roots: Sequence[Path], row_id: str) -> tuple[Optional[Path], str]:
    """Output dir of the row's final attempt (prefer the session where it ended OK)."""
    have = [r for r in roots if (r / row_id).exists()]
    root = next((r for r in have if final_status(r, row_id) == "OK"), have[-1] if have else None)
    if root is None:
        return None, "MISSING"
    hits = sorted((root / row_id).glob("tasks/*/*/output"))
    return (hits[0] if hits else None), final_status(root, row_id)


def load_snapshot(out_dir: Path, trial_id: str) -> Optional[pd.DataFrame]:
    m = re.search(r"(\d+)$", str(trial_id))
    if not m:
        return None
    p = out_dir / "intermediate_steps" / f"df_attributes_explanations_trail_{int(m.group(1))}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_ofe(ofe_roots: Sequence[Path], ds: str, seed: int, fold: int) -> dict[str, Any]:
    """OpenFE dump of (dataset, seed, fold): train and test frames + rank order. Generated
    features only (the renamed raw columns of the JSON ``column_map`` are dropped)."""
    for root in ofe_roots:
        fdir = Path(root) / f"f{seed}" / "features"
        tr_p, te_p = fdir / f"{ds}_openfe_fold{fold}.csv.gz", fdir / f"{ds}_openfe_fold{fold}_test.csv.gz"
        if not tr_p.exists():
            continue
        if not te_p.exists():
            return {"status": "control unavailable (no test-row dump)", "root": str(root)}
        js = fdir / f"{ds}_openfe_fold{fold}.json"
        meta = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
        raw_names = set((meta.get("column_map") or {}).keys())
        tr, te = pd.read_csv(tr_p), pd.read_csv(te_p)
        rank = [f["name"] for f in meta.get("features", []) if f.get("name") in tr.columns]
        if not rank:
            rank = [c for c in tr.columns if c != "id" and c not in raw_names]
        return {"status": "ok", "train": tr, "test": te, "rank": rank, "root": str(root)}
    return {"status": "control unavailable (no dump)"}


# =========================================================================== score (T10/T11)
def build_episode(out_dir: Path, data: pd.DataFrame, labels: pd.DataFrame, fold_seed: int, stratified: bool,
                  task_type: str, best_row: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the selected trial of one fold-episode and rebuild the Tester matrices."""
    from rogueone.utils.fold_safe_features import build_fold_safe_entities_folds  # noqa: PLC0415
    fold = int(best_row["test_fold_id"])
    snap = load_snapshot(out_dir, best_row["id"])
    if snap is None:
        return {"status": "no snapshot"}
    test_ids = fold_test_ids(data, labels, fold_seed, stratified=stratified, task_type=task_type)
    lab = labels.rename(columns={"patient_id": "id"}) if "id" not in labels.columns else labels
    attrs = snap.drop(columns=[c for c in ("Trial", "fold_run_id", "iteration_in_fold") if c in snap.columns])
    folds = build_fold_safe_entities_folds(df_raw=data, attribute_explanations=attrs, labels_df=lab,
                                           fixed_test_ids=[set(t) for t in test_ids], current_test_fold_id=fold)
    names = scored_names(snap)
    mats = tester_matrices(folds, fold, names)
    return {"status": "ok", "fold": fold, "snapshot": snap, **mats}


def score_row(row: Mapping[str, Any], out_dir: Path, data: pd.DataFrame, labels: pd.DataFrame, *,
              ofe_roots: Sequence[Path], phi: Optional[np.ndarray], parity_only: bool, xgb_n_jobs: int,
              cache: dict, log=print) -> tuple[list[dict], list[dict]]:
    """All fold-episodes of one A05 run -> (long rows, episode rows)."""
    ds, ttype, seed = row["dataset_id"], row["task_type"], int(row["fold_seed"])
    kind = task_kind(ds)
    strat = json.loads(row.get("env_overrides") or "{}").get("ROGUE_ONE_STRATIFIED_FOLDS", "False") == "True"
    best = pd.read_csv(out_dir / "fold_aggregation_best_iterations.csv", float_precision="round_trip")
    long_rows, ep_rows = [], []
    base = {"row_id": row["row_id"], "dataset_id": ds, "task_type": ttype, "kind": kind, "fold_seed": seed,
            "replicate": int(row["replicate"])}
    for _, b in best.sort_values("test_fold_id").iterrows():
        fold = int(b["test_fold_id"])
        ep = {**base, "fold": fold, "selected_trial": b["id"], "selected_iter": b.get("iteration_in_fold")}
        if str(b.get("status")) != "completed":
            ep_rows.append({**ep, "rescorable": 0, "note": f"selected row status {b.get('status')}"})
            continue
        try:
            e = build_episode(out_dir, data, labels, seed, strat, ttype, b)
        except Exception as exc:  # noqa: BLE001
            ep_rows.append({**ep, "rescorable": 1, "et_parity_ok": 0, "note": f"replay error {type(exc).__name__}: {exc}"})
            continue
        if e["status"] != "ok":
            ep_rows.append({**ep, "rescorable": 1, "et_parity_ok": 0, "note": e["status"]})
            continue
        try:
            par = fit_score("xgb400", e["X_train"], e["y_train"], e["X_test"], e["y_test"], ttype, seed, xgb_n_jobs)
        except Exception as exc:  # noqa: BLE001
            par = {"evaluator": "xgb400", "error": f"{type(exc).__name__}: {exc}"}
        pc = parity_check(ttype, par, b.to_dict())
        if "error" in par and pc.get("rescorable"):
            pc["et_parity_ok"] = 0
        long_rows.append({**base, "fold": fold, "feature_set": "ro", **par})
        ep.update(pc)
        ep["n_scored"] = len(e["cols"])
        log(f"[score] {row['row_id']} {ds} fold {fold}: parity ok={pc.get('et_parity_ok')} "
            f"delta={pc.get('et_parity_delta')}")
        if parity_only or not pc.get("et_parity_ok"):
            ep_rows.append(ep)
            continue
        # ---- feature sets
        snap, Xtr_ro, Xte_ro = e["snapshot"], e["X_train"], e["X_test"]
        ext = ro_extra_columns(snap, Xtr_ro, e["cols"])
        ext_nofb = ro_extra_columns(snap, Xtr_ro, e["cols"], drop_fallback=True)
        k = len(ext["live"])
        ep.update({"et_k": k, "et_k_dead": len(ext["dead"]), "et_k_nofb": len(ext_nofb["live"]),
                   "n_fallback_live": k - len(ext_nofb["live"])})
        ofe = cache.setdefault(("ofe", ds, seed, fold), load_ofe(ofe_roots, ds, seed, fold))
        if ofe["status"] == "ok":
            tr_ids = set(map(str, ofe["train"]["id"])); te_ids = set(map(str, ofe["test"]["id"]))
            if tr_ids != set(map(str, e["ids_train"])) or te_ids != set(map(str, e["ids_test"])):
                ofe = {"status": "control unavailable (partition mismatch)"}
                cache[("ofe", ds, seed, fold)] = ofe
        ep["ofe_status"] = ofe["status"]
        ep["ofe_n_available"] = len(ofe.get("rank", [])) if ofe["status"] == "ok" else None
        if ofe["status"] == "ok" and len(ofe["rank"]) < k:
            ep["ofe_k_capped"] = len(ofe["rank"])
        for ev in EVALUATORS + SENSITIVITY_EVALUATORS:
            Btr, Bte = raw_block(data, e["ids_train"], e["ids_test"], ev)
            sets: dict[str, tuple] = {"raw": (Btr, Bte)}
            if k > 0:
                sets["raw_ro"] = (pd.concat([Btr, Xtr_ro[ext["live"]].reset_index(drop=True).add_prefix("ro::")], axis=1),
                                  pd.concat([Bte, Xte_ro[ext["live"]].reset_index(drop=True).add_prefix("ro::")], axis=1))
            if ev in EVALUATORS and ext_nofb["live"]:
                sets["raw_ro_nofb"] = (
                    pd.concat([Btr, Xtr_ro[ext_nofb["live"]].reset_index(drop=True).add_prefix("ro::")], axis=1),
                    pd.concat([Bte, Xte_ro[ext_nofb["live"]].reset_index(drop=True).add_prefix("ro::")], axis=1))
            if ofe["status"] == "ok" and ev in EVALUATORS:
                otr = ofe["train"].set_index(ofe["train"]["id"].astype(str))
                ote = ofe["test"].set_index(ofe["test"]["id"].astype(str))
                itr, ite = list(map(str, e["ids_train"])), list(map(str, e["ids_test"]))
                if k > 0:
                    cols = ofe["rank"][:k]
                    sets["raw_ofe_k"] = (pd.concat([Btr, otr.loc[itr, cols].reset_index(drop=True).add_prefix("ofe::")], axis=1),
                                         pd.concat([Bte, ote.loc[ite, cols].reset_index(drop=True).add_prefix("ofe::")], axis=1))
                cols = ofe["rank"]
                sets["raw_ofe_all"] = (pd.concat([Btr, otr.loc[itr, cols].reset_index(drop=True).add_prefix("ofe::")], axis=1),
                                       pd.concat([Bte, ote.loc[ite, cols].reset_index(drop=True).add_prefix("ofe::")], axis=1))
            if phi is not None and ev in EVALUATORS:
                ph = pd.Series(phi, index=data["id"].to_numpy())
                sets["raw_phi"] = (Btr.assign(**{"phi::": ph.loc[list(e["ids_train"])].to_numpy()}),
                                   Bte.assign(**{"phi::": ph.loc[list(e["ids_test"])].to_numpy()}))
            for fs, (Xa, Xb) in sets.items():
                shared = fs in ("raw", "raw_ofe_all", "raw_phi")
                key = ("fit", ds, seed, fold, ev, fs)
                if shared and key in cache:
                    rec = cache[key]
                else:
                    try:
                        rec = fit_score(ev, Xa, e["y_train"], Xb, e["y_test"], ttype, seed, xgb_n_jobs)
                    except Exception as exc:  # noqa: BLE001
                        rec = {"evaluator": ev, "error": f"{type(exc).__name__}: {exc}"}
                    if shared:
                        cache[key] = rec
                long_rows.append({**base, "fold": fold, "feature_set": fs, **rec, "evaluator": ev})
        ep_rows.append(ep)
    return long_rows, ep_rows


def episode_gains(long: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, fold, evaluator): ``et_gain_<e>`` (raw_ro vs raw), ``et_gain_vs_ofe``
    (raw_ro vs raw_ofe_k), ``gain_phi`` (raw_phi vs raw), ``gain_nofb``, AUROC / R2 side
    scores. Episodes with k = 0 have no raw_ro (no control) and are kept with NaN gains."""
    out = []
    if long.empty:
        return pd.DataFrame()
    for (rid, fold, ev), g in long.groupby(["row_id", "fold", "evaluator"]):
        if ev == "xgb400" and set(g["feature_set"]) == {"ro"}:
            continue
        recs = {r["feature_set"]: r for r in g.to_dict("records")}
        if "raw" not in recs:
            continue
        r0 = recs["raw"]
        tt, kind = r0["task_type"], r0["kind"]
        row = {k: r0[k] for k in ("row_id", "dataset_id", "task_type", "kind", "fold_seed", "replicate")}
        row.update({"fold": fold, "evaluator": ev})
        for fs, nm in (("raw_ro", "gain"), ("raw_phi", "gain_phi"), ("raw_ofe_k", "gain_ofe_k"),
                       ("raw_ofe_all", "gain_ofe_all"), ("raw_ro_nofb", "gain_nofb")):
            row[nm] = gain(recs[fs], r0, tt, kind) if fs in recs else float("nan")
        # absolute difference in the logged metric's units (nRMSE on Core-10 R), for |gain| / sigma_d
        row["gain_abs"] = (float(r0.get("nrmse", np.nan)) - float(recs["raw_ro"].get("nrmse", np.nan))
                           if tt == "regression" and kind == "core10" else row["gain"]) if "raw_ro" in recs else 0.0
        row["k0"] = int("raw_ro" not in recs)
        if "raw_ro" not in recs:  # k = 0: raw_ro is B itself, so its gain is 0 by construction
            row["gain"] = 0.0
        row["gain_vs_ofe"] = (row["gain"] - row["gain_ofe_k"]) if np.isfinite(row["gain_ofe_k"]) else float("nan")
        if tt == "classification":
            row["gain_auroc"] = (recs["raw_ro"]["auroc"] - r0["auroc"]) if "raw_ro" in recs else float("nan")
        else:
            row["gain_r2"] = (recs["raw_ro"]["r2"] - r0["r2"]) if "raw_ro" in recs else float("nan")
        out.append(row)
    g = pd.DataFrame(out)
    keep = [c for c in ("row_id", "fold", "et_parity_ok", "rescorable", "et_k", "rec_sel") if c in eps.columns]
    return g.merge(eps[keep], on=["row_id", "fold"], how="left") if len(g) else g


def cmd_score(a: argparse.Namespace) -> int:  # pragma: no cover - I/O driver
    rc = 0
    import warnings  # noqa: PLC0415
    warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
    try:
        os.environ["ROGUE_ONE_SANDBOX"] = a.sandbox  # the replay runs as the agent's did
        _ensure_rogueone_importable(a.repo)
        from scripts.agent2027.analysis import recovery_detector as rd  # noqa: PLC0415
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        man = read_manifest(a.manifest)
        man = man[man["row_id"].isin(a.rows)] if a.rows else man[man["row_id"].str.match(r"^A05-0\d\d$")]
        if a.datasets:
            man = man[man["dataset_id"].isin(a.datasets)]
        roots = [Path(r) for r in a.runs]
        ofe_roots = [Path(r) for r in (a.ofe or [])]
        rec_sel = None
        if a.episodes:
            e = pd.read_csv(a.episodes)
            col = "rec_sel_amended" if "rec_sel_amended" in e.columns else "rec_sel"
            rec_sel = e[["row_id", "fold", col]].rename(columns={col: "rec_sel"})
        long_all, ep_all, runs = [], [], []
        data_cache: dict = {}
        cache: dict = {}
        for _, r in man.sort_values("row_id").iterrows():
            out_dir, status = locate_row(roots, r["row_id"])
            has_best = bool(out_dir and (out_dir / "fold_aggregation_best_iterations.csv").exists())
            runs.append({"row_id": r["row_id"], "dataset_id": r["dataset_id"], "status": status,
                         "has_best_iterations": int(has_best), "rescorable_run": int(status == "OK" and has_best)})
            if not (status == "OK" and has_best):
                print(f"[score] {r['row_id']}: not re-scorable (status={status}, best={has_best})", flush=True)
                continue
            ds, ttype = r["dataset_id"], r["task_type"]
            if ds not in data_cache:
                tdir = Path(a.tasks) / ttype / ds
                data, labels = pd.read_csv(tdir / "data.csv"), pd.read_csv(tdir / "labels.csv")
                phi = None
                if task_kind(ds) in ("synp", "p0") and a.meta and not a.parity_only:
                    phi, _ = rd.load_phi(a.meta, ds, data)
                data_cache[ds] = (data, labels, phi)
            data, labels, phi = data_cache[ds]
            lr, er = score_row(r.to_dict(), out_dir, data, labels, ofe_roots=ofe_roots, phi=phi,
                               parity_only=a.parity_only, xgb_n_jobs=a.xgb_n_jobs, cache=cache)
            long_all += lr
            ep_all += er
            pd.DataFrame(ep_all).to_csv(out / "et_parity.csv", index=False)  # incremental
        long = pd.DataFrame(long_all)
        eps = pd.DataFrame(ep_all)
        if rec_sel is not None and len(eps):
            eps = eps.merge(rec_sel, on=["row_id", "fold"], how="left")
        long.to_csv(out / "et_long.csv", index=False)
        eps.to_csv(out / "et_parity.csv", index=False)
        pd.DataFrame(runs).to_csv(out / "et_runs.csv", index=False)
        if not a.parity_only:
            episode_gains(long, eps).to_csv(out / "et_episodes.csv", index=False)
        meta = {"xgb_n_jobs": a.xgb_n_jobs, "sandbox": a.sandbox, "versions": _versions(),
                "parity_only": bool(a.parity_only), "n_runs": len(runs), "n_episodes": len(eps),
                "written": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        (out / "et_score_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if len(eps):
            rs = eps[eps["rescorable"] == 1]
            print(f"[score] parity: {int(rs['et_parity_ok'].fillna(0).sum())}/{len(rs)} re-scorable episodes pass")
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        rc = 1
    print(SCORE_DONE.format(rc=rc), flush=True)
    return rc


def _versions() -> dict[str, str]:
    v = {}
    for m in ("xgboost", "pandas", "numpy", "sklearn"):
        try:
            v[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            v[m] = "n/a"
    return v


# =========================================================================== readout (T13/T14, section 6)
def parity_gate(eps: pd.DataFrame) -> dict[str, Any]:
    """>= 90% of re-scorable fold-episodes pass (denominator: runs with a best-iterations file,
    minus NaN-logged episodes); per-run datasets with < 3/5 passing are excluded and named."""
    rs = eps[eps["rescorable"] == 1]
    n, k = len(rs), int(rs["et_parity_ok"].fillna(0).astype(int).sum())
    per_run = rs.groupby("row_id")["et_parity_ok"].apply(lambda s: int(s.fillna(0).astype(int).sum()))
    excluded = sorted(per_run[per_run < MIN_PASS_PER_RUN].index.tolist())
    return {"n_rescorable": n, "n_pass": k, "pass_rate": (k / n) if n else float("nan"),
            "gate_pass": bool(n and k / n >= PARITY_GATE), "not_rescorable": int((eps["rescorable"] == 0).sum()),
            "excluded_run_rows": excluded}


def usable_episodes(g: pd.DataFrame, excluded_rows: Iterable[str]) -> pd.DataFrame:
    """Parity-passing episodes of runs whose dataset kept >= 3/5 passing episodes."""
    ex = set(excluded_rows)
    return g[(g["et_parity_ok"] == 1) & (~g["row_id"].isin(ex))]


def unit_means(g: pd.DataFrame, col: str, evaluator: str, units: Sequence[str], *,
               rec_sel_only: bool = False) -> dict[str, float]:
    """Section 6 unit mean: the mean of the per-episode value over the unit's usable episodes in
    all runs (S: only ``rec_sel = 1`` episodes). NaN where the unit has no finite value."""
    out = {}
    for u in units:
        s = g[(g["dataset_id"] == u) & (g["evaluator"] == evaluator)]
        if rec_sel_only:
            s = s[s["rec_sel"] == 1]
        v = s[col].dropna()
        out[u] = float(v.mean()) if len(v) else float("nan")
    return out


def s_set(g: pd.DataFrame, nondisc: Sequence[str] = NON_DISCRIMINATING) -> list[str]:
    """S = discriminating SYN-P tasks with >= 3 parity-passing rec_sel=1 episodes (P0 never)."""
    ep = g.drop_duplicates(subset=["row_id", "fold"])
    out = []
    for t in SYNP8:
        if t in nondisc:
            continue
        n = int(((ep["dataset_id"] == t) & (ep["rec_sel"] == 1)).sum())
        if n >= MIN_S_EPISODES:
            out.append(t)
    return out


def assign_label(gain_lin_S: Mapping[str, float], gain_rf_S: Mapping[str, float],
                 gain_lin_H: Mapping[str, float], gain_rf_H: Mapping[str, float],
                 gain_xgb_S: Mapping[str, float], gain_xgb_H: Mapping[str, float],
                 vs_ofe_H: Mapping[str, Mapping[str, float]], vs_ofe_S: Mapping[str, Mapping[str, float]],
                 h_usable: Sequence[str]) -> dict[str, Any]:
    """Section 6 labels from unit means (all dicts: unit -> unit mean; ``vs_ofe_*``:
    evaluator -> unit -> mean ``et_gain_vs_ofe``; NaN = control unavailable / no data).

    T: for both linear and rf, gain > 0 on every task in S and >= 0 on >= 4 of the 6 in H.
    X (only if not T): U = units of S u H with xgb400 gain > 0; |U| >= 3 and, for each of
       linear and rf, at least half of U have gain <= 0.
    B: for each evaluator, vs_ofe <= 0 on >= 4 of the 6 in H (control-unavailable units never
       count) and the median over S of vs_ofe <= 0.
    N: none of T, X, B. "not evaluable" if < 4 of 6 H usable; T needs a non-empty S."""
    res: dict[str, Any] = {"h_usable": list(h_usable), "S": sorted(gain_lin_S)}
    if len(h_usable) < H_MIN_USABLE:
        res.update({"label": "not evaluable", "T": None, "X": None, "B": None})
        return res

    def n_ge0(d):
        return sum(1 for v in d.values() if np.isfinite(v) and v >= 0)
    S = list(gain_lin_S)
    t_s = bool(S) and all(np.isfinite(gain_lin_S[t]) and gain_lin_S[t] > 0 and np.isfinite(gain_rf_S.get(t, np.nan))
                          and gain_rf_S[t] > 0 for t in S)
    t_h = n_ge0(gain_lin_H) >= H_MIN_USABLE and n_ge0(gain_rf_H) >= H_MIN_USABLE
    T = (t_s and t_h) if S else None
    X = None
    if not T:
        U = [u for u, v in {**gain_xgb_S, **gain_xgb_H}.items() if np.isfinite(v) and v > 0]
        allg = {"linear": {**gain_lin_S, **gain_lin_H}, "rf": {**gain_rf_S, **gain_rf_H}}
        X = bool(len(U) >= 3 and all(sum(1 for u in U if allg[e].get(u, np.nan) <= 0) >= len(U) / 2.0
                                     for e in ("linear", "rf")))
    b_parts = {}
    for ev in EVALUATORS:
        h = vs_ofe_H.get(ev, {})
        n_le0 = sum(1 for v in h.values() if np.isfinite(v) and v <= 0)
        sv = [v for v in vs_ofe_S.get(ev, {}).values() if np.isfinite(v)]
        b_parts[ev] = bool(n_le0 >= H_MIN_USABLE and sv and float(np.median(sv)) <= 0)
    B = all(b_parts.values())
    labels = [nm for nm, v in (("T", T), ("X", X), ("B", B)) if v]
    res.update({"T": T, "T_S": t_s, "T_H": t_h, "X": X, "B": B, "B_by_evaluator": b_parts,
                "label": "+".join(labels) if labels else ("N" if T is not None else "T not assignable (S empty)")})
    return res


def boot_median(values: Sequence[float], n: int = BOOT_N, seed: int = BOOT_SEED, level: float = BOOT_LEVEL
                ) -> dict[str, float]:
    """Pooled median of unit means with a percentile bootstrap over units (T14)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_units": 0,
                "min": float("nan"), "max": float("nan")}
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), size=(n, len(v)))], axis=1)
    a = (1 - level) / 2
    return {"median": float(np.median(v)), "lo": float(np.quantile(meds, a)), "hi": float(np.quantile(meds, 1 - a)),
            "n_units": int(len(v)), "min": float(v.min()), "max": float(v.max())}


def readout(eps: pd.DataFrame, gains: pd.DataFrame, sigma: Optional[Mapping[str, float]] = None,
            nondisc: Sequence[str] = NON_DISCRIMINATING) -> dict[str, Any]:
    gate = parity_gate(eps)
    g = usable_episodes(gains, gate["excluded_run_rows"]) if len(gains) else gains
    out: dict[str, Any] = {"parity_gate": gate}
    if g is None or not len(g):
        out["label"] = {"label": "not evaluable", "reason": "no usable episodes"}
        return out
    if "rec_sel" not in g.columns:
        g = g.assign(rec_sel=np.nan)
    S = s_set(g, nondisc)
    H = [d for d in HEADROOM if d in set(g["dataset_id"])]
    um = {ev: {"H": unit_means(g, "gain", ev, HEADROOM), "S": unit_means(g, "gain", ev, S, rec_sel_only=True)}
          for ev in EVALUATORS}
    vs = {ev: {"H": unit_means(g, "gain_vs_ofe", ev, HEADROOM), "S": unit_means(g, "gain_vs_ofe", ev, S, rec_sel_only=True)}
          for ev in EVALUATORS}
    h_par = [d for d in HEADROOM if np.isfinite(um["xgb400"]["H"].get(d, np.nan))]
    h_ctrl = [d for d in h_par if all(np.isfinite(vs[ev]["H"].get(d, np.nan)) for ev in EVALUATORS)]
    out["units"] = {"S": S, "H_parity_passing": h_par, "H_control_available": h_ctrl}
    out["label"] = assign_label(um["linear"]["S"], um["rf"]["S"], um["linear"]["H"], um["rf"]["H"],
                                um["xgb400"]["S"], um["xgb400"]["H"],
                                {ev: vs[ev]["H"] for ev in EVALUATORS}, {ev: vs[ev]["S"] for ev in EVALUATORS},
                                h_usable=h_ctrl)
    # descriptive conditions, read on parity-passing H alone (the label above is the literal rule)
    out["label_conditions_parity_H"] = assign_label(
        um["linear"]["S"], um["rf"]["S"], um["linear"]["H"], um["rf"]["H"], um["xgb400"]["S"], um["xgb400"]["H"],
        {ev: vs[ev]["H"] for ev in EVALUATORS}, {ev: vs[ev]["S"] for ev in EVALUATORS}, h_usable=h_par)
    out["unit_means"] = um
    out["unit_means_vs_ofe"] = vs
    # per-dataset means and min-max over runs
    per_run = g.groupby(["dataset_id", "evaluator", "row_id"])["gain"].mean().reset_index()
    tbl = per_run.groupby(["dataset_id", "evaluator"])["gain"].agg(["mean", "min", "max", "count"]).reset_index()
    out["per_dataset_runs"] = tbl.to_dict("records")
    out["pooled"] = {ev: {"H": boot_median(list(um[ev]["H"].values())), "S": boot_median(list(um[ev]["S"].values()))}
                     for ev in EVALUATORS}
    sy = g[g["kind"] == "synp"]
    out["synp_by_rec_sel"] = sy.groupby(["evaluator", "rec_sel"])["gain"].agg(["mean", "count"]).reset_index() \
        .to_dict("records") if len(sy) else []
    gap = {}
    for ev in EVALUATORS:
        gap[ev] = {}
        for t in [t for t in SYNP8 if t not in nondisc]:
            for lab, sub in (("rec_sel1", sy[(sy["rec_sel"] == 1)]), ("all", sy)):
                s = sub[(sub["dataset_id"] == t) & (sub["evaluator"] == ev)].dropna(subset=["gain", "gain_phi"])
                mp = float(s["gain_phi"].mean()) if len(s) else float("nan")
                gap[ev].setdefault(t, {})[lab] = (float(s["gain"].mean()) / mp) if (len(s) and mp > 0) else "n/a"
    out["et_gap_closed"] = gap
    if sigma:
        xs = g[(g["evaluator"] == "xgb400") & (g["kind"] == "core10")]
        out["xgb400_abs_gain_over_sigma_d"] = {
            d: float(xs.loc[xs["dataset_id"] == d, "gain_abs"].abs().mean() / sigma[d])
            for d in sorted(set(xs["dataset_id"])) if d in sigma and sigma[d] > 0}
    sens = {}
    for col, nm in (("gain_nofb", "fallback_excluded"),):
        sens[nm] = {ev: {"H": unit_means(g.assign(gain=g[col]), "gain", ev, HEADROOM),
                         "S": unit_means(g.assign(gain=g[col]), "gain", ev, S, rec_sel_only=True)}
                    for ev in EVALUATORS}
    lw = g[g["evaluator"] == "linear_nowins"]
    sens["no_winsorisation_linear"] = {"H": unit_means(lw, "gain", "linear_nowins", HEADROOM),
                                       "S": unit_means(lw, "gain", "linear_nowins", S, rec_sel_only=True)}
    out["sensitivity"] = sens
    out["p0_sanity"] = g[g["dataset_id"] == P0].groupby("evaluator")["gain"].mean().to_dict()
    return out


def cmd_readout(a: argparse.Namespace) -> int:  # pragma: no cover - I/O driver
    d = Path(a.in_dir)
    eps = pd.read_csv(d / "et_parity.csv")
    if a.episodes:  # (re)join rec_sel from a per-episode table (row_id, fold, rec_sel)
        e = pd.read_csv(a.episodes)
        col = "rec_sel_amended" if "rec_sel_amended" in e.columns else "rec_sel"
        e = e[["row_id", "fold", col]].rename(columns={col: "rec_sel"}).drop_duplicates(["row_id", "fold"])
        eps = eps.drop(columns=[c for c in ("rec_sel",) if c in eps.columns]).merge(e, on=["row_id", "fold"], how="left")
    long = pd.read_csv(d / "et_long.csv") if (d / "et_long.csv").exists() else pd.DataFrame()
    gains = episode_gains(long, eps)  # recomputed from the long table (et_episodes.csv is the same)
    sigma = None
    if a.sigma:
        s = pd.read_csv(a.sigma)
        sigma = dict(zip(s["dataset_id"], s["sigma_d"].astype(float)))
    res = readout(eps, gains, sigma)
    out = Path(a.out or d)
    out.mkdir(parents=True, exist_ok=True)
    (out / "et_readout.json").write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    print(json.dumps({"parity_gate": res["parity_gate"], "units": res.get("units"), "label": res["label"]},
                     indent=1, default=_json_default))
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# =========================================================================== fetch / log
def cmd_fetch(a: argparse.Namespace) -> int:  # pragma: no cover - network
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fetch_wandb  # noqa: PLC0415
    return fetch_wandb.main(["--project", a.project, "--group", a.group, "--out", a.out])


def cmd_log(a: argparse.Namespace) -> int:  # pragma: no cover - network
    import wandb  # noqa: PLC0415
    d = Path(a.in_dir)
    name = f"ANALYSIS__evaluator_transfer__{_dt.date.today():%Y%m%d}"
    run = wandb.init(entity=a.entity, project=a.project, group="ANALYSIS", name=name, job_type="analysis",
                     config={"git_sha": a.git_sha, "image_tag": a.image_tag, "ablation": "A22", "status": "short paper (ED5)"})
    for f in ("et_long.csv", "et_episodes.csv", "et_parity.csv"):
        if (d / f).exists():
            run.log({f.replace(".csv", ""): wandb.Table(dataframe=pd.read_csv(d / f))})
    if (d / "et_readout.json").exists():
        run.summary["et_readout"] = json.loads((d / "et_readout.json").read_text(encoding="utf-8"))
    art = wandb.Artifact("a22-m3-outputs", type="analysis")
    art.add_dir(str(d))
    run.log_artifact(art)
    run.finish()
    return 0


# =========================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=_default_repo())
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--tasks", required=True)
    f.add_argument("--datasets", nargs="+", required=True, help="core10 | synp8 | <ids>")
    f.add_argument("--seeds", nargs="+", type=int, required=True)
    f.add_argument("--stratified-classification", choices=("0", "1"), default="0")
    f.add_argument("--out", required=True)
    fe = sub.add_parser("fetch")
    fe.add_argument("--project", default="RogueOne-AGENT2027")
    fe.add_argument("--group", default="A05")
    fe.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--runs", nargs="+", required=True, help="session run roots")
    s.add_argument("--manifest", nargs="+", required=True)
    s.add_argument("--tasks", required=True)
    s.add_argument("--meta", default=None)
    s.add_argument("--ofe", nargs="*", default=[])
    s.add_argument("--episodes", default=None, help="per-episode rec_sel (row_id, fold, rec_sel)")
    s.add_argument("--rows", nargs="*", default=None)
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--xgb-n-jobs", type=int, default=DEFAULT_XGB_N_JOBS)
    s.add_argument("--sandbox", choices=("on", "off"), default="on")
    s.add_argument("--parity-only", action="store_true")
    s.add_argument("--out", required=True)
    r = sub.add_parser("readout")
    r.add_argument("--in", dest="in_dir", required=True)
    r.add_argument("--sigma", default=None, help="G2_sigma_d.csv (dataset_id, sigma_d)")
    r.add_argument("--episodes", default=None, help="per-episode rec_sel table overriding the score-time join")
    r.add_argument("--out", default=None)
    lg = sub.add_parser("log")
    lg.add_argument("--in", dest="in_dir", required=True)
    lg.add_argument("--git-sha", required=True)
    lg.add_argument("--image-tag", required=True)
    lg.add_argument("--entity", default="PlanBench")
    lg.add_argument("--project", default="RogueOne-AGENT2027")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return {"folds": cmd_folds, "fetch": cmd_fetch, "score": cmd_score, "readout": cmd_readout,
            "log": cmd_log}[a.cmd](a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
