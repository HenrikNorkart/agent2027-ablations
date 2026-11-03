"""Feature quality metric computation utilities for TesterAgent.

This module computes pre-computed per-feature metrics used by the TesterAgent's
assessment and pruning stages.
"""

from dataclasses import dataclass, field
from typing import Literal
import os

import numpy as np
import shap
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance


def _resolve_n_jobs() -> int:
    """Resolve the worker count for parallel sklearn/XGBoost calls.

    Reads ``ROGUE_ONE_N_JOBS`` if set, otherwise falls back to
    ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` (capped at 8). Limiting this
    is important: the OpenBLAS build on this image supports at most 128 thread
    regions, so outer parallelism × inner BLAS threads must stay under that
    cap to avoid the "too many memory regions" abort.
    """
    raw = (
        os.environ.get("ROGUE_ONE_N_JOBS")
        or os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("OPENBLAS_NUM_THREADS")
        or "8"
    )
    try:
        value = int(raw)
    except ValueError:
        value = 8
    return max(1, min(value, 8))


_N_JOBS = _resolve_n_jobs()


@dataclass(frozen=True)
class FeatureMetricProfile:
    """Per-feature metric profile used for assessment and pruning."""

    feature_name: str
    mutual_information: float | None
    vif: float | None
    permutation_importance_mean: float | None
    permutation_importance_std: float | None
    noise_sensitivity: float | None
    mean_abs_shap: float | None
    computation_warnings: list[str] = field(default_factory=list)


def _compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    task_type: Literal["classification", "regression"],
) -> list[float | None]:
    if X.size == 0 or y.size == 0:
        return [None] * X.shape[1]

    try:
        if task_type == "classification":
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
        return [float(v) for v in scores]
    except Exception:
        return [None] * X.shape[1]


def _compute_vif(X: np.ndarray, feature_names: list[str]) -> list[float | None]:
    _ = feature_names
    n_features = X.shape[1]
    if n_features < 2 or X.shape[0] == 0:
        return [None] * n_features

    X_float = np.asarray(X, dtype=float)
    # Small epsilon added to the (1 - R^2) denominator to avoid divide-by-zero
    # warnings/Infs from statsmodels when a feature is perfectly collinear
    # with the others (R^2 == 1).
    eps = 1e-12
    vifs: list[float | None] = []
    for idx in range(n_features):
        try:
            if np.isclose(np.std(X_float[:, idx]), 0.0):
                vifs.append(None)
                continue
            # Compute R^2 of regressing feature `idx` on the remaining features
            # via a least-squares fit with an intercept column. This mirrors
            # statsmodels.variance_inflation_factor but lets us safely guard
            # the denominator.
            y_i = X_float[:, idx]
            others = np.delete(X_float, idx, axis=1)
            ones = np.ones((others.shape[0], 1), dtype=float)
            design = np.hstack([ones, others])
            coef, *_ = np.linalg.lstsq(design, y_i, rcond=None)
            y_pred = design @ coef
            ss_res = float(np.sum((y_i - y_pred) ** 2))
            ss_tot = float(np.sum((y_i - y_i.mean()) ** 2))
            if ss_tot <= 0.0:
                vifs.append(None)
                continue
            r_squared = 1.0 - ss_res / ss_tot
            denom = max(1.0 - r_squared, 0.0) + eps
            value = 1.0 / denom
            if np.isfinite(value):
                vifs.append(float(value))
            else:
                vifs.append(None)
        except Exception:
            vifs.append(None)
    return vifs


def _compute_permutation_importance(
    model: xgb.XGBClassifier | xgb.XGBRegressor,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> list[tuple[float, float] | None]:
    if X.size == 0 or y.size == 0:
        return [None] * X.shape[1]

    try:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=5,
            random_state=seed,
            n_jobs=_N_JOBS,
        )
        return [
            (float(result.importances_mean[i]), float(result.importances_std[i]))
            for i in range(X.shape[1])
        ]
    except Exception:
        return [None] * X.shape[1]


def _compute_noise_sensitivity(
    model: xgb.XGBClassifier | xgb.XGBRegressor,
    X: np.ndarray,
    seed: int,
) -> list[float | None]:
    if X.size == 0:
        return [None] * X.shape[1]

    rng = np.random.default_rng(seed)

    try:
        if hasattr(model, "predict_proba"):
            base_pred = np.asarray(model.predict_proba(X), dtype=float)
        else:
            base_pred = np.asarray(model.predict(X), dtype=float)

        if base_pred.ndim == 1:
            pred_range = float(np.max(base_pred) - np.min(base_pred))
        else:
            pred_range = float(
                np.max(base_pred, axis=0).max() - np.min(base_pred, axis=0).min()
            )
        pred_range = pred_range if pred_range > 0 else 1.0

        scores: list[float | None] = []
        for idx in range(X.shape[1]):
            std = float(np.std(X[:, idx]))
            if np.isclose(std, 0.0):
                scores.append(0.0)
                continue

            sigma = 0.1 * std
            X_noisy = np.array(X, copy=True)
            X_noisy[:, idx] = X_noisy[:, idx] + rng.normal(0.0, sigma, size=X.shape[0])

            if hasattr(model, "predict_proba"):
                noisy_pred = np.asarray(model.predict_proba(X_noisy), dtype=float)
            else:
                noisy_pred = np.asarray(model.predict(X_noisy), dtype=float)

            delta = np.mean(np.abs(noisy_pred - base_pred))
            scores.append(float(delta / pred_range))

        return scores
    except Exception:
        return [None] * X.shape[1]


def _compute_mean_abs_shap(
    model: xgb.XGBClassifier | xgb.XGBRegressor,
    X: np.ndarray,
) -> list[float | None]:
    if X.size == 0:
        return [None] * X.shape[1]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            stacked = np.stack([np.abs(np.asarray(v)) for v in shap_values], axis=0)
            mean_abs = np.mean(stacked, axis=(0, 1))
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                mean_abs = np.mean(np.abs(arr), axis=(0, 2))
            else:
                mean_abs = np.mean(np.abs(arr), axis=0)

        return [float(v) for v in mean_abs]
    except Exception:
        return [None] * X.shape[1]


def compute_feature_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model: xgb.XGBClassifier | xgb.XGBRegressor,
    task_type: Literal["classification", "regression"],
    fold_seed: int,
) -> list[FeatureMetricProfile]:
    """Compute all pre-defined feature quality metrics for each feature.

    This function is resilient to per-metric failures and records warnings in the
    returned feature profiles.
    """
    if X_train.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X_train column count")

    mi_scores = _compute_mutual_information(X_train, y_train, task_type)
    vif_scores = _compute_vif(X_train, feature_names)
    pfi_scores = _compute_permutation_importance(model, X_train, y_train, fold_seed)
    noise_scores = _compute_noise_sensitivity(model, X_train, fold_seed)
    shap_scores = _compute_mean_abs_shap(model, X_train)

    profiles: list[FeatureMetricProfile] = []
    for idx, feature_name in enumerate(feature_names):
        warnings: list[str] = []

        mi = mi_scores[idx]
        vif = vif_scores[idx]
        pfi = pfi_scores[idx]
        noise = noise_scores[idx]
        shap_value = shap_scores[idx]

        if mi is None:
            warnings.append("mutual_information_unavailable")
        if vif is None:
            warnings.append("vif_unavailable")
        if pfi is None:
            warnings.append("permutation_importance_unavailable")
        if noise is None:
            warnings.append("noise_sensitivity_unavailable")
        if shap_value is None:
            warnings.append("mean_abs_shap_unavailable")

        pfi_mean = pfi[0] if pfi is not None else None
        pfi_std = pfi[1] if pfi is not None else None

        profiles.append(
            FeatureMetricProfile(
                feature_name=feature_name,
                mutual_information=mi,
                vif=vif,
                permutation_importance_mean=pfi_mean,
                permutation_importance_std=pfi_std,
                noise_sensitivity=noise,
                mean_abs_shap=shap_value,
                computation_warnings=warnings,
            )
        )

    return profiles
