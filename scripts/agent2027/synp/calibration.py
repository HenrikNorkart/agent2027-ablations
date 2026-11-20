"""Headroom calibration for SYN-P (E8a, CPU only).

Headroom = score(XGB on raw + phi) - score(XGB on raw), cross-validated:

* classification: balanced accuracy (stratified 5-fold, balanced sample weights
  as the Tester does); pass line **>= 0.06** absolute.
* regression: relative RMSE reduction  (RMSE_raw - RMSE_phi) / RMSE_raw, i.e. the
  relative improvement of nRMSE (the nRMSE denominator is constant within a
  task); pass line **>= 0.15**.

XGBoost uses the Tester's assessment-model hyper-parameters
(``rogueone/llm/agents/core/tester/agent.py``: 300 trees, lr 0.05, depth 6,
subsample / colsample 0.9, hist, CPU).  Categorical columns are ordinal-coded
in alphabetical order of their level strings (as the CPU brute-force screen
does).  The reported headroom is the mean over fold seeds {42, 4, 5}; ``min``
over seeds is reported too.

``calibrate_task`` scans the signal parameter (regression: noise SD; classification:
logit slope beta) and picks the *weakest* signal whose seed-42 headroom reaches the
aim (pass line + margin), so tasks are not made easier than needed.  For P5 (a
deterministic rule) the flip grid runs cleanest-first: the least noisy variant that
reaches the aim is kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import class_weight

from scripts.agent2027.synp import tasks as T

FOLD_SEEDS = (42, 4, 5)
N_JOBS = 4
GRIDS = {
    "noise": (2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1),  # weakest signal first
    "beta": (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 1000.0),
    "flip": (0.0, 0.05, 0.1, 0.2, 0.3),
}


def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in ("id",) if c in df.columns]).copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.Categorical(out[c], categories=sorted(out[c].unique())).codes
    return out.astype(float)


def _model(task_type: str, seed: int, n_classes: int):
    base = dict(tree_method="hist", random_state=seed, device="cpu", n_estimators=300,
                learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, n_jobs=N_JOBS)
    if task_type == "regression":
        return xgb.XGBRegressor(objective="reg:squarederror", **base)
    if n_classes > 2:
        return xgb.XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **base)
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **base)


def cv_score(X: pd.DataFrame, y: np.ndarray, task_type: str, seed: int, n_splits: int = 5) -> float:
    """Mean fold score: balanced accuracy (C) or RMSE (R)."""
    y = np.asarray(y)
    n_classes = int(len(np.unique(y)))
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    else:
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    scores = []
    Xv = X.to_numpy(dtype=np.float32)
    for tr, te in splits:
        m = _model(task_type, seed, n_classes)
        if task_type == "classification":
            w = class_weight.compute_sample_weight("balanced", y=y[tr])
            m.fit(Xv[tr], y[tr], sample_weight=w)
            scores.append(balanced_accuracy_score(y[te], m.predict(Xv[te])))
        else:
            m.fit(Xv[tr], y[tr])
            scores.append(float(np.sqrt(np.mean((y[te] - m.predict(Xv[te])) ** 2))))
    return float(np.mean(scores))


def headroom(
    data: pd.DataFrame, phi: np.ndarray, y: np.ndarray, task_type: str,
    seeds=FOLD_SEEDS, extra: np.ndarray | None = None,
) -> dict[str, Any]:
    """Headroom of ``phi`` (or of ``extra`` if given, e.g. phi_wrong) over raw."""
    X0 = encode_frame(data)
    X1 = X0.copy()
    X1["__phi__"] = np.asarray(phi if extra is None else extra, dtype=float)
    per_seed = []
    for s in seeds:
        a, b = cv_score(X0, y, task_type, s), cv_score(X1, y, task_type, s)
        h = (b - a) if task_type == "classification" else (a - b) / a
        per_seed.append({"seed": s, "raw": a, "with_phi": b, "headroom": h})
    hs = [r["headroom"] for r in per_seed]
    return {"task_type": task_type, "per_seed": per_seed, "mean": float(np.mean(hs)),
            "min": float(np.min(hs)),
            "raw_mean": float(np.mean([r["raw"] for r in per_seed])),
            "with_phi_mean": float(np.mean([r["with_phi"] for r in per_seed]))}


@dataclass
class CalibrationResult:
    key: str
    param: str
    value: float
    target: float
    aim: float
    achieved_mean: float
    achieved_min: float
    seeds: tuple[int, ...]
    passed: bool
    scan: list[dict[str, Any]]
    raw_score: float
    phi_score: float
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def calibrate_task(key: str, *, master_seed: int = T.MASTER_SEED, n: int = T.N_ROWS,
                   p7_secret: str = "", p7_constants=None, p6a_candidate: str = T.DEFAULT_P6A,
                   verbose: bool = False) -> CalibrationResult:
    t0 = time.time()
    task_type = T.TASK_TYPES[key]
    param = next(iter(T.SIGNAL_DEFAULTS[key]))
    target, aim = T.HEADROOM_TARGET[task_type], T.HEADROOM_AIM[task_type]
    kw = dict(master_seed=master_seed, n=n, p7_secret=p7_secret, p7_constants=p7_constants,
              p6a_candidate=p6a_candidate)
    scan, chosen = [], None
    for v in GRIDS[param]:
        g = T.generate_task(key, params={param: v}, **kw)
        h = headroom(g.data, g.phi, g.labels["target"].to_numpy(), task_type, seeds=(42,))
        scan.append({param: v, "headroom_seed42": h["mean"]})
        if verbose:
            print(f"  {key} {param}={v}: {h['mean']:.4f}")
        if chosen is None and h["mean"] >= aim:
            chosen = v
            break
    if chosen is None:  # strongest signal still short: use it and report the miss
        chosen = scan[-1][param]
    g = T.generate_task(key, params={param: chosen}, **kw)
    h = headroom(g.data, g.phi, g.labels["target"].to_numpy(), task_type)
    return CalibrationResult(
        key=key, param=param, value=float(chosen), target=target, aim=aim,
        achieved_mean=h["mean"], achieved_min=h["min"], seeds=tuple(FOLD_SEEDS),
        passed=bool(h["mean"] >= target), scan=scan, raw_score=h["raw_mean"],
        phi_score=h["with_phi_mean"], seconds=time.time() - t0,
    )
