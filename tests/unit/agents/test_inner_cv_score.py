"""Unit tests for the nested-CV inner selection signal (`_inner_cv_score`).

Regression coverage for the arrhythmia failure found while collecting T0.1 for the
camera-ready: on a multiclass dataset with a class rarer than the number of inner
folds, the inner split falls back to an unstratified `KFold`, which can leave a class
entirely out of an inner training half. XGBoost then raises "Invalid classes inferred
from unique values of `y`". The original implementation wrapped the whole fold loop in
a single `try`, so that one bad fold returned `None` and voided the selection signal
for every trial on the dataset — silently reverting nested CV to test-fold selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from rogueone.llm.agents.core.tester.agent import _inner_cv_score

BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "device": "cpu",
    "n_estimators": 8,
    "max_depth": 3,
    "n_jobs": 1,
}

# The regression path in the Tester builds its own params; mirror them here rather
# than reusing the logistic objective above, which XGBoost rejects for real targets.
REG_PARAMS = {**BASE_PARAMS, "objective": "reg:squarederror", "eval_metric": "rmse"}


def _multiclass_with_rare_class(
    n_common: int = 120, rare_count: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Data with three well-populated classes plus one class of `rare_count` rows."""
    rng = np.random.RandomState(seed)
    y = np.concatenate(
        [
            np.zeros(n_common, dtype=int),
            np.ones(n_common, dtype=int),
            np.full(n_common, 2, dtype=int),
            np.full(rare_count, 3, dtype=int),
        ]
    )
    # Give each class a slightly different mean so the score is not pure noise.
    X = rng.normal(loc=y[:, None], scale=1.0, size=(len(y), 5))
    return X, y


def test_rare_class_still_yields_a_selection_signal() -> None:
    """A class rarer than n_splits must not void the inner-CV score."""
    X, y = _multiclass_with_rare_class(rare_count=2)
    # Precondition: the rare class forces the unstratified fallback splitter.
    assert np.unique(y, return_counts=True)[1].min() < 4

    score = _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4, seed=42)

    assert score is not None, "rare class must not void the nested-CV selection signal"
    assert 0.0 <= score <= 1.0


def test_class_absent_from_inner_training_half_counts_as_error_not_crash() -> None:
    """A class present only in the validation half is scored, not raised on."""
    X, y = _multiclass_with_rare_class(rare_count=1, seed=3)
    score = _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4, seed=42)
    assert score is not None
    # The single unseen-class row can never be predicted correctly, so a perfect
    # score is impossible, but the fold must still contribute a real number.
    assert 0.0 <= score < 1.0


def test_well_populated_multiclass_is_unaffected() -> None:
    """The stratified path (every class >= n_splits) keeps working as before."""
    X, y = _multiclass_with_rare_class(rare_count=40)
    assert np.unique(y, return_counts=True)[1].min() >= 4
    score = _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4, seed=42)
    assert score is not None
    assert score > 0.5


def test_binary_classification_signal_is_deterministic() -> None:
    """Same data and seed must give the same score (selection must be reproducible)."""
    rng = np.random.RandomState(1)
    y = np.array([0] * 80 + [1] * 80)
    X = rng.normal(loc=y[:, None], scale=1.0, size=(len(y), 4))
    a = _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4, seed=42)
    b = _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4, seed=42)
    assert a is not None and a == b


def test_regression_returns_rmse() -> None:
    """Regression path returns a positive RMSE on the inner folds."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(120, 4))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.1, size=120)
    score = _inner_cv_score(X, y, REG_PARAMS, "regression", n_splits=4, seed=42)
    assert score is not None and score > 0.0


def test_too_few_rows_returns_none() -> None:
    """Below n_splits * 2 rows there is no viable inner split."""
    X = np.zeros((5, 3))
    y = np.array([0, 1, 0, 1, 0])
    assert _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4) is None


def test_mismatched_lengths_return_none() -> None:
    """Guard against a caller passing misaligned X and y."""
    X = np.zeros((20, 3))
    y = np.zeros(19, dtype=int)
    assert _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4) is None


def test_every_fold_failing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no inner fold can be scored the caller still gets None, not a crash."""
    import xgboost as xgb

    class Exploding(xgb.XGBClassifier):
        def fit(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "rogueone.llm.agents.core.tester.agent.xgb.XGBClassifier", Exploding
    )
    X, y = _multiclass_with_rare_class(rare_count=40)
    assert _inner_cv_score(X, y, BASE_PARAMS, "classification", n_splits=4) is None
