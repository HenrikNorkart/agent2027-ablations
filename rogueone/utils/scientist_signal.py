"""Scientist feedback signal (AGENT 2027 E14, ``ROGUE_ONE_SCIENTIST_SIGNAL``).

Decision D2 (HQ #245, 2026-10-09): Full-v2 gives the Scientist an inner-CV scalar
("L1") instead of the in-sample training fit, computed on an inner split disjoint
from the one used for selection (01 §4.4 item 2).

Modes
-----
``train_fit`` (code default)
    Today's behaviour, unchanged byte for byte: the Scientist's feedback pool carries
    the ``train_*`` metrics of the reported XGBoost model scored on its own training
    rows (``RogueOneAgentNetwork._build_scientist_feedback_row``).
``inner_cv_l1`` (Full-v2, ``scripts/agent2027/v2_defaults.env``)
    The ``train_*`` columns are replaced by one scalar per trial, ``cv_accuracy``
    (classification) or ``cv_RMSE`` (regression), computed by :func:`l1_inner_cv_score`.
Any other value raises ``ValueError`` (at config load).

The L1 split, precisely
-----------------------
Let ``R`` be the training rows of the outer fold (the four outer folds that are not
the held-out test fold ``T``; ``R`` and ``T`` are disjoint by construction) and
``K = 4`` the inner fold count.

* **Selection split** ``S = {V_1..V_K}`` is exactly the split ``_inner_cv_score``
  (``ROGUE_ONE_SELECTION_SIGNAL=inner_cv``) has always used: ``StratifiedKFold`` (or
  ``KFold`` when a class has fewer than ``K`` rows, or for regression) with
  ``shuffle=True, random_state=fold_seed`` over ``R``
  (:func:`build_selection_splits` rebuilds it; ``_inner_cv_score`` itself is left
  untouched and a unit test pins the two to the same indices).
* **Feedback split** ``F = {W_1..W_K}`` is a *crossed* partition of ``R``
  (:func:`build_feedback_splits`): every selection validation fold ``V_k`` is shuffled
  with ``numpy.random.RandomState(fold_seed + L1_SEED_OFFSET)``, its rows are grouped by
  class (classification; shuffled within class), and dealt round-robin over the ``K``
  feedback folds, starting at fold ``k``. Each ``W_j`` therefore takes
  ``floor(|V_k|/K)`` or ``ceil(|V_k|/K)`` rows of every ``V_k`` (about ``1/K``) and is
  approximately stratified.

**"Disjoint" means split-disjoint** (the rule :func:`check_l1_split` asserts):

1. Both splits use only rows of ``R``; no row id of ``T`` occurs in either (and ``R``
   has no duplicate ids). The test fold is never touched.
2. Each split is a partition of ``R``: its validation folds are pairwise disjoint and
   cover ``R``, and each training set is the complement of its validation fold.
3. **No fold is shared**: for every ``j, k``, ``|W_j ∩ V_k| <= ceil(|V_k| / K)``, so
   no feedback validation fold equals, contains or is mostly made of any selection
   validation fold (``W_j != V_k`` and ``W_j``'s training set ``!= V_k``'s). The largest
   overlap fraction is logged as ``l1_max_val_overlap`` (≈ ``1/K`` = 0.25).

It is *not* row-disjoint, and cannot be while the selection signal is unchanged:
selection validates on every row of ``R``, so any feedback validation fold inside
``R`` shares rows with it. A row-disjoint alternative (holding a subset of ``R``
apart for the feedback) would shrink the selection split, change the selection
signal relative to every other arm ("selection by inner CV" is fixed in every arm,
AGENT2027 §3.2) and confound the LF2 L1-vs-training-fit contrast, whose only
difference must be the Scientist's scalar. What the crossed split removes is the
coupling through the *same fold assignment*: the scalar the Scientist steers by is
never the selection score itself, nor computed on any of its (train, validation)
pairs. Both estimates still see the labels of ``R`` and remain correlated; that
residual coupling must be stated (and is what the T2 "same vs disjoint inner split"
coupling check measures).

Metric family (same as now)
---------------------------
Same XGBoost params as the reported model (400 trees). Classification: accuracy on
each inner validation fold, with ``compute_sample_weight("balanced")`` on both the
inner fit and the scoring under ``ROGUE_ONE_DO_BALANCING`` (i.e. balanced accuracy,
as the pool's ``accuracy`` and ``train_accuracy`` are); mean over inner folds.
Regression: RMSE per inner fold, mean (as the selection signal). Cost: ``K = 4`` extra
XGBoost fits per iteration on 3/4 of ``R`` (the selection inner CV already does 4).
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Environment flag (01 §7 R4). Code default ``train_fit``.
SCIENTIST_SIGNAL_ENV = "ROGUE_ONE_SCIENTIST_SIGNAL"
TRAIN_FIT = "train_fit"
INNER_CV_L1 = "inner_cv_l1"
MODES: tuple[str, ...] = (TRAIN_FIT, INNER_CV_L1)

#: Inner fold count shared by the selection split and the feedback split.
INNER_N_SPLITS = 4
#: Offset added to ``fold_seed`` for the feedback split's shuffle.
L1_SEED_OFFSET = 1_000_003

#: Scientist-facing column names under ``inner_cv_l1``.
L1_COLUMN_CLASSIFICATION = "cv_accuracy"
L1_COLUMN_REGRESSION = "cv_RMSE"
#: Keys the Tester writes into a successful ``fold_metrics`` row under ``inner_cv_l1``.
L1_FOLD_KEY_CLASSIFICATION = "l1_cv_accuracy"
L1_FOLD_KEY_REGRESSION = "l1_cv_RMSE"


class L1SplitLeakageError(AssertionError):
    """The feedback split violates the §4.4 item 2 disjointness rule."""


# --------------------------------------------------------------------------- mode
def parse_scientist_signal(value: Optional[str]) -> str:
    """Validate a ``ROGUE_ONE_SCIENTIST_SIGNAL`` value (unset/empty = ``train_fit``)."""
    mode = (value if value is not None else "").strip().lower() or TRAIN_FIT
    if mode not in MODES:
        raise ValueError(
            f"{SCIENTIST_SIGNAL_ENV} must be one of {MODES}, got {value!r}"
        )
    return mode


def scientist_signal(environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    return parse_scientist_signal(env.get(SCIENTIST_SIGNAL_ENV))


# --------------------------------------------------------------------------- splits
def build_selection_splits(
    y: Any, task_type: str, n_splits: int = INNER_N_SPLITS, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The selection signal's inner split (verbatim the pre-E14 ``_inner_cv_score`` logic).

    Raises whatever the sklearn splitter raises; the caller handles it.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    y = np.asarray(y).ravel()
    X_dummy = np.zeros((y.shape[0], 1))
    if task_type == "classification":
        _, class_counts = np.unique(y, return_counts=True)
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if class_counts.min() >= n_splits
            else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X_dummy, y))


def build_feedback_splits(
    y: Any,
    selection_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    task_type: str,
    n_splits: int = INNER_N_SPLITS,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The L1 feedback split: a partition of the same rows crossed with the selection split."""
    y = np.asarray(y).ravel()
    n = y.shape[0]
    rng = np.random.RandomState((int(seed) + L1_SEED_OFFSET) % (2**32))
    fold_of = np.full(n, -1, dtype=int)
    for k, (_, val_idx) in enumerate(selection_splits):
        rows = np.asarray(val_idx, dtype=int)
        rows = rows[rng.permutation(rows.shape[0])]
        if task_type == "classification":
            rows = rows[np.argsort(y[rows], kind="stable")]
        fold_of[rows] = (np.arange(rows.shape[0]) + k) % n_splits
    if (fold_of < 0).any():
        raise L1SplitLeakageError("selection split does not cover every training row")
    all_rows = np.arange(n)
    return [
        (all_rows[fold_of != j], all_rows[fold_of == j]) for j in range(n_splits)
    ]


def _check_partition(
    name: str, splits: Sequence[tuple[np.ndarray, np.ndarray]], n: int
) -> list[set[int]]:
    vals: list[set[int]] = []
    seen: set[int] = set()
    for i, (tr, va) in enumerate(splits):
        tr_s, va_s = set(map(int, tr)), set(map(int, va))
        if not va_s:
            raise L1SplitLeakageError(f"{name} fold {i}: empty validation fold")
        if any(r < 0 or r >= n for r in tr_s | va_s):
            raise L1SplitLeakageError(f"{name} fold {i}: row outside the training rows")
        if tr_s & va_s or len(tr_s) + len(va_s) != n:
            raise L1SplitLeakageError(
                f"{name} fold {i}: training set is not the complement of validation"
            )
        if seen & va_s:
            raise L1SplitLeakageError(f"{name} fold {i}: validation folds overlap")
        seen |= va_s
        vals.append(va_s)
    if len(seen) != n:
        raise L1SplitLeakageError(f"{name}: validation folds do not cover the rows")
    return vals


def check_l1_split(
    selection_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    feedback_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    n_rows: int,
    train_ids: Optional[Sequence[Any]] = None,
    test_ids: Optional[Sequence[Any]] = None,
) -> dict[str, Any]:
    """Assert the §4.4 item 2 rule (module docstring, items 1-3); raise on violation.

    ``train_ids`` are the row ids of the outer-train rows in matrix order (position
    ``i`` = row ``i`` of ``X_train``); ``test_ids`` the outer test fold's ids.
    """
    ids_checked = 0
    if train_ids is not None:
        train_list = list(train_ids)
        if len(train_list) != n_rows:
            raise L1SplitLeakageError(
                f"{len(train_list)} train ids for {n_rows} training rows"
            )
        if len(set(train_list)) != n_rows:
            raise L1SplitLeakageError("duplicate row ids among the training rows")
        if test_ids is not None:
            leaked = set(train_list) & set(test_ids)
            if leaked:
                raise L1SplitLeakageError(
                    f"{len(leaked)} outer test-fold ids among the inner-CV rows"
                )
            ids_checked = 1
    sel = _check_partition("selection", selection_splits, n_rows)
    fb = _check_partition("feedback", feedback_splits, n_rows)
    k = len(fb)
    max_frac = 0.0
    for j, w in enumerate(fb):
        for i, v in enumerate(sel):
            inter = len(w & v)
            if w == v or inter > math.ceil(len(v) / k):
                raise L1SplitLeakageError(
                    f"feedback fold {j} shares {inter}/{len(v)} rows with selection fold {i}"
                )
            max_frac = max(max_frac, inter / len(v))
    return {
        "l1_split_disjoint": 1,
        "l1_ids_checked": ids_checked,
        "l1_max_val_overlap": float(max_frac),
    }


# --------------------------------------------------------------------------- score
def l1_inner_cv_score(
    X_train: Any,
    y_train: Any,
    base_params: dict[str, Any],
    task_type: str,
    feedback_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    balance_weights: bool = False,
) -> Optional[float]:
    """Mean inner-fold score on the feedback split (balanced accuracy or RMSE).

    Mirrors ``_inner_cv_score`` (same params, same per-fold class re-indexing, same
    NaN handling) except for the split and, for classification, the balanced scoring
    weights. Does not touch ``instrumentation`` (the selection scores stay unmixed).
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.utils import class_weight

    X = np.nan_to_num(
        np.asarray(X_train).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6
    )
    y = np.asarray(y_train).ravel()
    scores: list[float] = []
    for tr, va in feedback_splits:
        try:
            if task_type == "classification":
                present = np.unique(y[tr])
                local = np.searchsorted(present, y[tr])
                model = xgb.XGBClassifier(**base_params)
                model.fit(
                    X[tr],
                    local,
                    sample_weight=(
                        class_weight.compute_sample_weight("balanced", y=local)
                        if balance_weights
                        else None
                    ),
                )
                pred = present[np.asarray(model.predict(X[va])).astype(int)]
                scores.append(
                    float(
                        accuracy_score(
                            y[va],
                            pred,
                            sample_weight=(
                                class_weight.compute_sample_weight("balanced", y=y[va])
                                if balance_weights
                                else None
                            ),
                        )
                    )
                )
            else:
                model = xgb.XGBRegressor(**base_params)
                model.fit(X[tr], y[tr])
                scores.append(
                    float(np.sqrt(mean_squared_error(y[va], model.predict(X[va]))))
                )
        except Exception as exc:  # noqa: BLE001 - one bad inner fold is not fatal
            print(f"[l1_inner_cv] inner fold failed: {exc!r}")
    if not scores:
        return None
    value = float(np.mean(scores))
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- run audit
_AUDIT: dict[str, Any] = {}


def reset_audit() -> None:
    _AUDIT.clear()
    _AUDIT.update(
        l1_split_checks=0, l1_split_failures=0, l1_ids_checked=1, l1_max_val_overlap=0.0
    )


reset_audit()


def note_check(result: Optional[dict[str, Any]], failed: bool = False) -> None:
    _AUDIT["l1_split_checks"] += 1
    if failed or result is None:
        _AUDIT["l1_split_failures"] += 1
        return
    _AUDIT["l1_ids_checked"] = min(_AUDIT["l1_ids_checked"], int(result["l1_ids_checked"]))
    _AUDIT["l1_max_val_overlap"] = max(
        _AUDIT["l1_max_val_overlap"], float(result["l1_max_val_overlap"])
    )


def audit_summary(mode: str) -> dict[str, Any]:
    """Run-level W&B summary keys (``scientist_signal`` always; ``l1_*`` under L1)."""
    out: dict[str, Any] = {"scientist_signal": mode}
    if mode == INNER_CV_L1:
        checks = int(_AUDIT["l1_split_checks"])
        out.update(
            l1_split_checks=checks,
            l1_split_failures=int(_AUDIT["l1_split_failures"]),
            l1_split_disjoint=int(checks > 0 and _AUDIT["l1_split_failures"] == 0),
            l1_ids_checked=int(_AUDIT["l1_ids_checked"]) if checks else 0,
            l1_max_val_overlap=float(_AUDIT["l1_max_val_overlap"]),
        )
    return out


def compute_l1_for_tester(
    *,
    X_train: Any,
    y_train: Any,
    base_params: dict[str, Any],
    task_type: str,
    seed: int,
    balance_weights: bool,
    train_ids: Optional[Sequence[Any]],
    test_ids: Optional[Sequence[Any]],
) -> dict[str, Any]:
    """Build both splits, assert the rule, score L1. Returns ``fold_metrics`` keys.

    Raises :class:`L1SplitLeakageError` when the rule fails (after noting it in the
    run audit); a split that is not viable (too few rows) gives a ``None`` score.
    """
    key = (
        L1_FOLD_KEY_CLASSIFICATION
        if task_type == "classification"
        else L1_FOLD_KEY_REGRESSION
    )
    y = np.asarray(y_train).ravel()
    n = y.shape[0]
    if n < INNER_N_SPLITS * 2 or np.asarray(X_train).shape[0] != n:
        return {key: None, "l1_status": "not_viable"}
    try:
        selection = build_selection_splits(y, task_type, INNER_N_SPLITS, seed)
    except Exception as exc:  # noqa: BLE001 - same fallback as the selection signal
        print(f"[l1_inner_cv] split not viable, no feedback signal: {exc!r}")
        return {key: None, "l1_status": "not_viable"}
    try:
        feedback = build_feedback_splits(y, selection, task_type, INNER_N_SPLITS, seed)
        check = check_l1_split(selection, feedback, n, train_ids, test_ids)
    except L1SplitLeakageError:
        note_check(None, failed=True)
        raise
    note_check(check)
    score = l1_inner_cv_score(
        X_train, y, base_params, task_type, feedback, balance_weights=balance_weights
    )
    return {key: score, "l1_status": "ok", **check}


# --------------------------------------------------------------------------- prompt text
#: The ``generic_pandas_tool`` docstring lines that name the performance columns.
TRAIN_FIT_COLUMNS_LINE = (
    "- Performance columns are TRAIN-set metrics only: `train_accuracy`, "
    "`train_precision`, `train_recall`, `train_f1`, `train_AUROC`, `train_MSE`, "
    "`train_MAE`, `train_R2`. Held-out metrics are deliberately not visible to the "
    "Scientist (Constitution II — leakage prevention)."
)
L1_COLUMNS_LINE = (
    "- Performance columns: `cv_accuracy` (classification: class-balanced accuracy, "
    "higher is better) and `cv_RMSE` (regression: root-mean-squared error, lower is "
    "better); the column of the other task type is empty. Each value is the mean score "
    "of a 4-fold cross-validation inside this fold's training rows, refitted on that "
    "trial's active attributes. Held-out metrics are deliberately not visible to the "
    "Scientist (Constitution II — leakage prevention)."
)
TRAIN_FIT_EXAMPLE = "result = df_test_results['train_accuracy'].mean()"
L1_EXAMPLE = "result = df_test_results['cv_accuracy'].mean()"


def l1_pandas_tool_description(description: str) -> str:
    """The ``generic_pandas_tool`` description rewritten for ``inner_cv_l1``."""
    for old in (TRAIN_FIT_COLUMNS_LINE, TRAIN_FIT_EXAMPLE):
        if old not in description:
            raise RuntimeError(
                f"generic_pandas_tool description no longer contains {old[:60]!r}; "
                "update rogueone.utils.scientist_signal"
            )
    return description.replace(TRAIN_FIT_COLUMNS_LINE, L1_COLUMNS_LINE).replace(
        TRAIN_FIT_EXAMPLE, L1_EXAMPLE
    )
